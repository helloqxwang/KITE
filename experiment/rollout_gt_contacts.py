"""Online MuJoCo rollout of low-level contact policies.

The script bypasses the high-level policy. It reads ground-truth contact clouds
from a preprocessed high-level payload, observes the live tabletop scene,
optionally builds non-contact avoidance points from external RGBD cameras, and
executes the low-level policy closed-loop in MuJoCo.
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import hydra
import mujoco
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.manipulation.paper_execution import paper_expected_frame_count, validate_paper_control_dt
from experiment.embodiments import validate_embodiment_supported  # noqa: E402
from experiment.high_level_policy.payload import (  # noqa: E402
    build_common_meta,
    build_training_payload,
    concat_single_manipulator_action,
    load_high_level_payload,
    make_episode_ends,
    save_high_level_payload,
    single_manipulator_action_space,
    tensor_to_numpy,
)
from experiment.contact_runtime_utils import (  # noqa: E402
    WUJI_JOINT_DIM,
    WUJI_LOCAL_DOF,
    EpisodeSpec,
    _apply_fixed_zero_constraints,
    _build_episode_specs,
    _dataset_xyz_translation,
    _episode_ranges,
    _episode_value,
    _euler_xyz_to_quat_wxyz,
    _initial_wuji_q_from_retargeted,
    _load_robot_model_start_q,
    _payload_arrays,
    _quat_wxyz_to_euler_xyz,
    _retarget_episode_map,
    _shift_low_level_q,
)
from experiment.tabletop_sim.translation import (  # noqa: E402
    _shift_low_level_q as _shift_composite_low_level_q,
    _translation_groups as _composite_translation_groups,
)
from experiment.robotiq_stick_start_pose import (  # noqa: E402
    DEFAULT_ROBOTIQ_STICK_START_POSE_RELATIVE,
    ROBOTIQ_STICK_LOCAL_DOF,
    ROBOTIQ_STICK_ROBOT_NAME,
    load_robotiq_stick_start_q,
    validate_robotiq_stick_q,
)
from experiment.tabletop_sim.rotation import (  # noqa: E402
    unwrap_robot_model_q_rotation, robot_model_q_to_native, robot_model_q_to_world,
)
from experiment.wuji_start_pose import (  # noqa: E402
    DEFAULT_WUJI_START_POSE_RELATIVE,
    load_wuji_start_q,
    validate_wuji_q,
)
from experiment.pipeline_high_low_policy import (  # noqa: E402
    _load_cotrain_low_level_policy,
    _peek_cotrain_meta,
    _resolve_low_level_backend,
    _rng_devices,
    load_root_hydra_config,
    predict_low_level_q_from_contact_latent,
)
from experiment.pump_bottle_metrics import pump_press_order_metrics  # noqa: E402
from experiment.rollout_utils import (  # noqa: E402
    _compact_payload,
    _extend_flat_trace,
    _print_progress,
    _print_summary,
    _save_vis_trace_npz,
    _should_report_progress,
    _validate_robot_model_q_source_matches_env,
    RolloutResult,
)
from experiment.replay_utils import (  # noqa: E402
    ViewerReplayControl,
    _load_training_pt,
    _set_cube_state,
    _source_scalar,
    _trajectory_array,
    _trajectory_count,
    _trajectory_value,
)
from experiment.tabletop_sim.env import TabletopManipulationEnv  # noqa: E402
from experiment.tabletop_sim.factory import build_env_from_config, optional_project_path  # noqa: E402
from experiment.tabletop_sim.parallel_rollout import (  # noqa: E402
    configure_torch_threads,
    launch_sharded_workers,
    validate_headless_parallel,
)
from experiment.tabletop_sim.perception import (  # noqa: E402
    deproject_pixels_cv,
    farthest_point_sample_indices,
)
from experiment.tabletop_sim.profiling import RolloutProfiler  # noqa: E402
from experiment.tabletop_sim.scene import (  # noqa: E402
    ONLINE_ROLLOUT_RENDER_CAMERA_NAME,
    look_at_quat_wxyz,
)
from experiment.tabletop_sim.task_success import accumulate_task_success  # noqa: E402
from experiment.tabletop_sim.tasks import _body_tree_geom_min_z  # noqa: E402
from experiment.tabletop_sim.trace import append_trace_frame, ctrl_names, new_trace_buffers, qpos_names  # noqa: E402
from experiment.task_start_pose import (  # noqa: E402
    StartPoseResolution,
    TASK_CUBE_PICK,
    describe_start_pose_resolution,
    resolve_start_pose_path,
)
from experiment.object_start_pose import (  # noqa: E402
    StartPose,
    add_episode_start_pose_metadata,
    load_object_start_poses,
)
from high_level_policy.train_utils import prepare_device  # noqa: E402
from robot_model.collision_primitives import (  # noqa: E402
    PrimitiveCollisionModel,
    collision_primitive_spec_path,
    load_collision_primitive_spec,
)
from robot_model.mano_collision_primitives import ManoPrimitiveCollisionModel, mano_collision_primitive_spec_path  # noqa: E402
from robot_model.robot_model import create_robot_model, resolve_robot_model_name, _is_mano_robot_name  # noqa: E402
from contact_policy_probe import load_contact_policy_probe_checkpoint  # noqa: E402
from experiment.tabletop_sim.trace_capture import configure_vis_trace_mode  # noqa: E402


from paper_config import runtime_config


@dataclass
class OnlineLowLevelRuntime:
    device: torch.device
    low_level_cfg: Any
    low_level_policy: torch.nn.Module
    low_level_config_path: str
    low_level_checkpoint_path: str
    low_level_meta: dict[str, Any]
    robot_model: Any
    local_dof: int
    action_valid_mask: torch.Tensor
    robot_index: int
    encoder_backend: str
    contact_feature_dim: int
    low_level_manipulator: str
    hand_fps_num_points: int
    non_contact_enabled: bool
    non_contact_input_points: int
    non_contact_sampling_radius: float
    non_contact_sdf_clearance: float
    collision_model: Any | None
    collision_primitive_spec_path: str | None


@dataclass(frozen=True)
class NonContactStats:
    raw_count: int
    after_contact_exclusion: int
    valid_count: int
    selected_count: int


@dataclass(frozen=True)
class PumpBottleRolloutResult:
    source_index: int
    source_seed: int | None
    action_frame_count: int
    completed_steps: int
    success: bool
    max_cube_z: float
    final_cube_pose: np.ndarray
    error: str
    lift_success: bool = False
    button_success: bool = False
    max_lift_delta: float = float("nan")
    final_lift_delta: float = float("nan")
    max_lift_clearance_delta: float = float("nan")
    final_lift_clearance_delta: float = float("nan")
    initial_min_geom_z: float = float("nan")
    final_min_geom_z: float = float("nan")
    min_button_slide: float = float("nan")
    final_button_slide: float = float("nan")
    first_button_success_frame: int = -1
    first_lift_success_frame: int = -1
    lift_clearance_delta_at_first_button_success: float = float("nan")
    button_after_lift_success: bool = False
    ordered_success: bool = False
    target_key_success: np.ndarray | None = None
    final_target_key_slide: np.ndarray | None = None
    min_target_key_slide: np.ndarray | None = None


@dataclass
class OnlineTrainingEpisode:
    spec: EpisodeSpec
    source_seed: int | None = None
    frames: dict[str, list[np.ndarray]] = field(default_factory=dict)


@dataclass(frozen=True)
class OnlineRolloutImageRenderConfig:
    enabled: bool
    output_dir: Path
    width: int
    height: int
    offset_xyz: np.ndarray
    fovy: float
    camera_name: str
    overwrite: bool


class OnlineRolloutImageRenderer:
    def __init__(self, config: OnlineRolloutImageRenderConfig, *, task_name: str, embodiment: str) -> None:
        self.config = config
        self.task_name = _safe_path_component(task_name)
        self.embodiment = _safe_path_component(embodiment)
        self._renderer: mujoco.Renderer | None = None
        self._renderer_model: mujoco.MjModel | None = None
        self._trajectory_dir: Path | None = None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._renderer_model = None

    def start_trajectory(self, env: Any, spec: EpisodeSpec) -> None:
        if not self.config.enabled:
            return
        target = _object_geometry_center(env)
        camera_pos = target + self.config.offset_xyz
        _set_named_camera_pose(
            env,
            camera_name=self.config.camera_name,
            camera_pos=camera_pos,
            target=target,
            fovy=self.config.fovy,
        )
        folder = f"{self.task_name}_{self.embodiment}_traj{int(spec.source_trajectory_index):04d}"
        trajectory_dir = self.config.output_dir / folder
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        self._trajectory_dir = trajectory_dir

    def render_step(self, env: Any, *, step_idx: int) -> Path | None:
        if not self.config.enabled:
            return None
        if self._trajectory_dir is None:
            raise RuntimeError("render_images trajectory directory is not initialized")
        path = self._trajectory_dir / f"step_{int(step_idx):06d}.png"
        if path.exists() and not self.config.overwrite:
            raise FileExistsError(f"render_images output already exists: {path}")
        _release_env_camera_renderer(env)
        try:
            self._ensure_renderer(env)
            assert self._renderer is not None
            self._renderer.update_scene(env.data, camera=self.config.camera_name)
            _disable_renderer_shadows_if_model_requests(self._renderer, env.model)
            rgb = np.asarray(self._renderer.render(), dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[2] < 3:
                raise RuntimeError(f"Expected RGB render with shape HxWx3, got {rgb.shape}")
            from PIL import Image

            Image.fromarray(rgb[:, :, :3], mode="RGB").save(path)
        finally:
            # Camera observations use their own MuJoCo renderer. Releasing this
            # high-res renderer after each capture avoids concurrent EGL contexts.
            self.close()
        return path

    def _ensure_renderer(self, env: Any) -> None:
        if self._renderer is not None and self._renderer_model is env.model:
            return
        self.close()
        env.model.vis.global_.offwidth = max(int(env.model.vis.global_.offwidth), int(self.config.width))
        env.model.vis.global_.offheight = max(int(env.model.vis.global_.offheight), int(self.config.height))
        self._renderer = mujoco.Renderer(env.model, height=int(self.config.height), width=int(self.config.width))
        self._renderer_model = env.model


def _release_env_camera_renderer(env: Any) -> None:
    camera_renderer = getattr(env, "_camera_renderer", None)
    if camera_renderer is None:
        return
    camera_renderer.close()
    setattr(env, "_camera_renderer", None)


def _disable_renderer_shadows_if_model_requests(renderer: mujoco.Renderer, model: mujoco.MjModel) -> None:
    if int(model.vis.quality.shadowsize) > 0:
        return
    try:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    except Exception:
        return


_PUMP_BOTTLE_LIFT_SUCCESS_METRIC_ALIASES = {
    "max": "max_lift_delta",
    "maximum": "max_lift_delta",
    "peak": "max_lift_delta",
    "any": "max_lift_delta",
    "any_timestep": "max_lift_delta",
    "max_lift_delta": "max_lift_delta",
    "final": "final_lift_delta",
    "last": "final_lift_delta",
    "last_timestep": "final_lift_delta",
    "final_lift_delta": "final_lift_delta",
}


def _pump_bottle_lift_success_metric(value: Any, *, default: str) -> str:
    raw = str(default if value is None else value).strip().lower()
    metric = _PUMP_BOTTLE_LIFT_SUCCESS_METRIC_ALIASES.get(raw)
    if metric is None:
        valid = "max_lift_delta or final_lift_delta"
        raise ValueError(f"Invalid pump-bottle lift success metric {value!r}; expected {valid}")
    return metric


def _pump_bottle_lift_success_metric_from_cfg(
    cfg: DictConfig,
    *,
    section: str,
    default: str,
) -> str:
    value = OmegaConf.select(cfg, f"{section}.pump_bottle_lift_success_metric", default=None)
    if value is None:
        value = OmegaConf.select(cfg, f"{section}.pump_bottle_lift_success_source", default=None)
    return _pump_bottle_lift_success_metric(value, default=default)


def _pump_bottle_lift_success_for_metric(
    metric: str,
    *,
    max_lift_delta: float,
    final_lift_delta: float,
    threshold: float,
) -> bool:
    value = final_lift_delta if metric == "final_lift_delta" else max_lift_delta
    return bool(np.isfinite(value) and value >= float(threshold))


def _pump_bottle_min_geom_z(env: Any) -> float:
    """Return the lowest point on the complete articulated bottle geometry."""

    return _body_tree_geom_min_z(
        env.model,
        env.data,
        env.scene.pump_bottle_body_name,
    )


def _maybe_print_button_slide_debug(
    *,
    enabled: bool,
    prefix: str,
    step_idx: int,
    repeat_idx: int | None,
    obs: dict[str, Any],
) -> None:
    if not enabled or "button_slide" not in obs:
        return
    button_slide = float(np.asarray(obs["button_slide"], dtype=np.float32))
    if "lift_delta" in obs:
        lift_delta_text = f" lift_delta={float(np.asarray(obs['lift_delta'], dtype=np.float32)):.6f}"
    else:
        lift_delta_text = ""
    repeat_text = "" if repeat_idx is None else f" repeat={int(repeat_idx)}"
    print(
        f"{prefix} step={int(step_idx)}{repeat_text} button_slide={button_slide:.6f}{lift_delta_text}",
        flush=True,
    )


@hydra.main(version_base="1.2", config_path="conf", config_name="rollout_gt_contacts/object_suite_wuji")
def main(cfg: DictConfig) -> None:
    cfg = runtime_config(cfg)
    OmegaConf.resolve(cfg)
    from experiment.manipulation.paper_execution import validate_paper_execution
    validate_paper_execution(cfg, 'online_rollout')
    configure_torch_threads(cfg, section="online_rollout")
    validate_embodiment_supported(
        str(cfg.env.embodiment),
        "low_level_online_rollout",
        label="env.embodiment",
    )
    vis_trace_capture = configure_vis_trace_mode(
        cfg,
        project_root=ROOT,
        rollout_config=(
            str(HydraConfig.get().job.config_name)
            if HydraConfig.initialized()
            else None
        ),
    )
    if vis_trace_capture is not None:
        print(
            "vis_trace mode: "
            f"sources={list(vis_trace_capture.source_indices)} "
            f"results={vis_trace_capture.results_path} "
            f"trace={vis_trace_capture.trace_path} "
            f"config={vis_trace_capture.config_snapshot_path} "
            "render=false render_images.enabled=false",
            flush=True,
        )
    embodiment = str(cfg.env.embodiment)
    if embodiment == "robotiq_gripper_stick_armless":
        _main_robotiq_stick_armless(cfg)
        return
    if embodiment in {"allegro_right", "barrett", "wujihand_right"}:
        _main_floating_robot_model_hand(cfg)
        return
    if embodiment != "marvin_wuji":
        raise ValueError("This online low-level rollout currently expects env.embodiment=marvin_wuji")

    contact_path = _required_path(
        OmegaConf.select(cfg, "online_rollout.contact_source_pt", default=None),
        "online_rollout.contact_source_pt",
    )
    retarget_path = _optional_path(OmegaConf.select(cfg, "online_rollout.retargeted_wuji_pt", default=None))
    object_reset_source_path = _optional_path(
        OmegaConf.select(cfg, "online_rollout.object_reset_source_pt", default=None)
    )
    if retarget_path is not None and object_reset_source_path is not None:
        raise ValueError(
            "Set only one of online_rollout.retargeted_wuji_pt and "
            "online_rollout.object_reset_source_pt"
        )
    if retarget_path is None and object_reset_source_path is None:
        raise ValueError(
            "Wuji online rollout requires online_rollout.retargeted_wuji_pt, or "
            "online_rollout.object_reset_source_pt with a saved initial pose"
        )
    reset_source_path = retarget_path or object_reset_source_path
    assert reset_source_path is not None
    high_level_config_path, high_level_cfg = _high_level_config_from_cfg(cfg, key="online_rollout.high_level_config_path")
    low_level_config_path = _low_level_config_from_cfg(
        cfg,
        high_level_cfg,
        high_level_config_path=high_level_config_path,
    )
    low_level_checkpoint_path = _low_level_checkpoint_from_cfg(
        cfg,
        high_level_cfg,
        high_level_config_path=high_level_config_path,
    )
    low_level_manipulator = _low_level_manipulator_from_high_cfg(
        high_level_cfg,
        env_embodiment=str(cfg.env.embodiment),
    )
    low_level_use_ema = bool(OmegaConf.select(high_level_cfg, "preprocess.latent_prefer_ema", default=True))
    initial_pose_mode, initial_pose_path, initial_wuji_q = _initial_pose_from_cfg(cfg)

    contact_payload = load_high_level_payload(contact_path)
    contact = _payload_arrays(contact_payload)
    if retarget_path is not None:
        reset_payload = load_high_level_payload(retarget_path)
        reset_source = _payload_arrays(reset_payload)
        reset_source_kind = "retargeted_wuji"
    else:
        if initial_pose_mode != "saved":
            raise ValueError(
                "online_rollout.object_reset_source_pt requires "
                "online_rollout.initial_pose.mode='saved' for Wuji"
            )
        assert object_reset_source_path is not None
        reset_payload = None
        reset_source = _load_training_pt(object_reset_source_path)
        _validate_object_reset_source(reset_source, source_path=object_reset_source_path)
        reset_source_kind = "object_only"
    _validate_contact_source(contact)
    contact_input_mode = _contact_input_mode_from_cfg(cfg)
    if contact_input_mode == "latent":
        _validate_latent_contact_source(contact_payload, contact_path)
    task_name = str(OmegaConf.select(cfg, "task.name", default="cube_pick")).strip()
    if task_name == "object_suite_pick_place":
        _validate_object_suite_contact_source(contact_payload, contact_path)
    elif task_name == "pumpbottle_use":
        _validate_pump_bottle_contact_source(contact_payload, contact_path)

    contact_ranges = _episode_ranges(np.asarray(tensor_to_numpy(contact_payload["episode_ends"]), dtype=np.int64))
    selected = _selected_indices_from_cfg(
        len(contact_ranges),
        indices=OmegaConf.select(cfg, "online_rollout.indices", default=None),
        start_index=int(OmegaConf.select(cfg, "online_rollout.start_index", default=0)),
        count=OmegaConf.select(cfg, "online_rollout.count", default=None),
    )
    if not selected:
        raise ValueError("online_rollout selected no episodes")

    dataset_shift = _dataset_xyz_translation(contact_payload)
    runtime = _load_online_low_level_runtime(
        low_level_config_path=low_level_config_path,
        low_level_checkpoint_path=low_level_checkpoint_path,
        low_level_manipulator=low_level_manipulator,
        low_level_use_ema=low_level_use_ema,
        device=str(OmegaConf.select(cfg, "online_rollout.device", default="auto")),
    )
    if runtime.local_dof != WUJI_LOCAL_DOF:
        raise ValueError(f"Expected Wuji local_dof={WUJI_LOCAL_DOF}, got {runtime.local_dof}")

    save_training_pt = optional_project_path(OmegaConf.select(cfg, "online_rollout.save_training_pt", default=None))
    _configure_camera_for_runtime(cfg, runtime, force_camera_obs=save_training_pt is not None)
    if reset_source_kind == "retargeted_wuji":
        assert reset_payload is not None
        reset_ranges = _episode_ranges(
            np.asarray(tensor_to_numpy(reset_payload["episode_ends"]), dtype=np.int64)
        )
        retarget_map = _retarget_episode_map(reset_payload, len(reset_ranges))
        specs = _build_episode_specs(
            selected,
            contact_payload=contact_payload,
            retarget_payload=reset_payload,
            contact_ranges=contact_ranges,
            retarget_ranges=reset_ranges,
            retarget_map=retarget_map,
            max_frames=OmegaConf.select(cfg, "online_rollout.max_frames", default=None),
        )
        reset_source_count = len(reset_ranges)
    else:
        specs = _build_contact_only_episode_specs(
            selected,
            contact_payload=contact_payload,
            contact_ranges=contact_ranges,
            max_frames=OmegaConf.select(cfg, "online_rollout.max_frames", default=None),
        )
        reset_source_count = _trajectory_count(reset_source)
        _validate_object_reset_source_specs(
            reset_source,
            specs,
            source_count=reset_source_count,
        )
    _validate_expected_frame_count(
        specs,
        paper_expected_frame_count(cfg, "online_rollout"),
    )
    object_ids: list[str] = []
    if task_name == "object_suite_pick_place" or (
        task_name == "pumpbottle_use"
        and _pump_bottle_uses_episode_objects(contact_payload)
    ):
        for spec in specs:
            contact_object_id = _object_suite_episode_id(
                contact_payload,
                spec.contact_episode_index,
                label="contact source",
            )
            if reset_source_kind == "retargeted_wuji":
                assert reset_payload is not None
                reset_object_id = _object_suite_episode_id(
                    reset_payload,
                    spec.retarget_episode_index,
                    label="retargeted source",
                )
            else:
                reset_object_id = _object_suite_source_id(
                    reset_source,
                    spec.source_trajectory_index,
                    label="object reset source",
                )
            if contact_object_id != reset_object_id:
                raise ValueError(
                    f"Object mismatch for source trajectory {spec.source_trajectory_index}: "
                    f"contact={contact_object_id!r} reset_source={reset_object_id!r}"
                )
            object_ids.append(contact_object_id)

    backend = str(OmegaConf.select(cfg, "online_rollout.backend", default=cfg.env.backend))
    viewer_enabled = bool(OmegaConf.select(cfg, "online_rollout.render", default=True))
    viewer_control = (
        ViewerReplayControl(
            start_paused=bool(OmegaConf.select(cfg, "online_rollout.viewer_start_paused", default=False))
        )
        if viewer_enabled
        else None
    )
    save_vis_trace_npz = optional_project_path(OmegaConf.select(cfg, "online_rollout.save_vis_trace_npz", default=None))
    save_results_npz = optional_project_path(OmegaConf.select(cfg, "online_rollout.save_results_npz", default=None))
    store_trace = save_vis_trace_npz is not None
    capture_camera_every_controller_step = bool(
        OmegaConf.select(cfg, "online_rollout.capture_camera_every_controller_step", default=False)
    )
    render_images_config = _render_images_config_from_cfg(cfg)
    continue_on_error = bool(OmegaConf.select(cfg, "online_rollout.continue_on_error", default=False))
    progress_interval = int(OmegaConf.select(cfg, "online_rollout.progress_interval", default=1))
    seed_offset = int(OmegaConf.select(cfg, "online_rollout.seed_offset", default=0))
    proprio_source = _proprio_source_from_cfg(cfg)
    pump_bottle_lift_success_metric = _pump_bottle_lift_success_metric_from_cfg(
        cfg,
        section="online_rollout",
        default="max_lift_delta",
    )
    print_button_slide_each_step = bool(
        OmegaConf.select(cfg, "online_rollout.print_button_slide_each_step", default=False)
    )
    controller_steps_per_prediction = _positive_int_from_cfg(
        cfg,
        "online_rollout.controller_steps_per_prediction",
        default=1,
    )
    process_count = validate_headless_parallel(
        cfg,
        section="online_rollout",
        render=viewer_enabled,
        save_vis_trace_npz=save_vis_trace_npz,
    )
    _validate_render_images_runtime(render_images_config, process_count, viewer_enabled=viewer_enabled)

    print(
        f"loaded contact={contact_path} reset_source={reset_source_path} "
        f"reset_source_kind={reset_source_kind} selected={len(specs)} "
        f"policy_frames={sum(spec.frame_count for spec in specs)} shift={dataset_shift.tolist()} "
        f"high_level_config={high_level_config_path} "
        f"low_level={runtime.low_level_checkpoint_path} backend={runtime.encoder_backend} "
        f"non_contact_enabled={runtime.non_contact_enabled} initial_pose_mode={initial_pose_mode}"
        f"{'' if initial_pose_path is None else f' initial_pose_pt={initial_pose_path}'} "
        f"proprio_source={proprio_source} contact_input={contact_input_mode} "
        f"controller_steps_per_prediction={controller_steps_per_prediction}"
        f"{'' if not render_images_config.enabled else f' render_images_dir={render_images_config.output_dir}'}"
        f"{'' if save_training_pt is None else f' save_training_pt={save_training_pt}'}",
        flush=True,
    )

    if process_count > 1:
        print(
            f"running online low-level rollout in parallel process_count={process_count} selected={len(specs)}",
            flush=True,
        )
        del runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with tempfile.TemporaryDirectory(prefix="tabletop_low_level_parallel_") as tmp:
            result_paths = launch_sharded_workers(
                script_path=Path(__file__).resolve(),
                cfg=cfg,
                section="online_rollout",
                selected=[int(spec.source_trajectory_index) for spec in specs],
                process_count=process_count,
                tmpdir=Path(tmp),
                result_prefix="online_rollout",
                seed_offset_key="online_rollout.seed_offset",
                extra_worker_outputs=(
                    {"online_rollout.save_training_pt": ("online_rollout_training", ".pt")}
                    if save_training_pt is not None
                    else None
                ),
            )
            merged_payload = _merge_result_payloads(result_paths)
            training_payload_paths = sorted(Path(tmp).glob("online_rollout_training_worker_*.pt"))
            if save_training_pt is not None:
                out = _merge_training_payloads(training_payload_paths, save_training_pt)
                print(f"saved online rollout training payload: {out}", flush=True)
        results = _results_from_payload(merged_payload)
        _print_summary(results)
        if save_results_npz is not None:
            out = _save_npz(save_results_npz, merged_payload)
            print(f"saved online rollout results: {out}", flush=True)
        return

    initial_env_cfg = _config_with_active_object(cfg, object_ids[0]) if object_ids else cfg
    env = build_env_from_config(
        initial_env_cfg,
        backend=backend,
        render=viewer_enabled,
        viewer_key_callback=viewer_control.key_callback if viewer_control is not None else None,
    )
    dt = float(env.model.opt.timestep) * int(env.backend.frame_skip)
    validate_paper_control_dt(cfg, dt)
    results: list[RolloutResult] = []
    flat_trace: dict[str, list[np.ndarray]] = {}
    trajectory_frame_start: list[int] = []
    trajectory_frame_count: list[int] = []
    training_records: list[tuple[OnlineTrainingEpisode, RolloutResult]] = []
    flat_frame_count = 0
    image_renderer = (
        OnlineRolloutImageRenderer(render_images_config, task_name=str(cfg.task.name), embodiment=str(cfg.env.embodiment))
        if render_images_config.enabled
        else None
    )

    try:
        for local_idx, spec in enumerate(specs):
            if object_ids and local_idx > 0:
                env.close()
                env = build_env_from_config(
                    _config_with_active_object(cfg, object_ids[local_idx]),
                    backend=backend,
                    render=viewer_enabled,
                    viewer_key_callback=viewer_control.key_callback if viewer_control is not None else None,
                )
                episode_dt = float(env.model.opt.timestep) * int(env.backend.frame_skip)
                if not np.isclose(episode_dt, dt):
                    raise ValueError(f"Object-suite env dt changed from {dt} to {episode_dt}")
            if object_ids:
                print(
                    f"trajectory={spec.source_trajectory_index} object_id={object_ids[local_idx]}",
                    flush=True,
                )
            trace = new_trace_buffers() if store_trace else None
            training_episode = OnlineTrainingEpisode(spec=spec) if save_training_pt is not None else None
            frame_start = flat_frame_count
            try:
                result = _rollout_episode(
                    env=env,
                    cfg=cfg,
                    spec=spec,
                    contact=contact,
                    reset_source=reset_source,
                    reset_source_kind=reset_source_kind,
                    runtime=runtime,
                    contact_input_mode=contact_input_mode,
                    dataset_shift=dataset_shift,
                    initial_wuji_q=initial_wuji_q,
                    proprio_source=proprio_source,
                    controller_steps_per_prediction=controller_steps_per_prediction,
                    store_trace=store_trace,
                    trace=trace,
                    viewer_control=viewer_control,
                    dt=dt,
                    seed=int(OmegaConf.select(cfg, "online_rollout.seed", default=0))
                    + (int(seed_offset) + int(local_idx)) * 100003,
                    capture_camera_every_controller_step=capture_camera_every_controller_step,
                    training_episode=training_episode,
                    image_renderer=image_renderer,
                    profiler=RolloutProfiler(
                        enabled=bool(OmegaConf.select(cfg, "online_rollout.profile.enabled", default=False)),
                        device=runtime.device,
                        label=f"trajectory {local_idx + 1}/{len(specs)} profile",
                    ),
                )
            except Exception as exc:
                if not continue_on_error:
                    raise RuntimeError(f"online rollout trajectory {spec.source_trajectory_index} failed") from exc
                result = RolloutResult(
                    source_index=int(spec.source_trajectory_index),
                    source_seed=None if spec.source_trajectory_seed < 0 else int(spec.source_trajectory_seed),
                    action_frame_count=int(spec.frame_count),
                    completed_steps=0,
                    success=False,
                    max_cube_z=float("nan"),
                    final_cube_pose=np.full((7,), np.nan, dtype=np.float32),
                    error=f"{type(exc).__name__}: {exc}",
                )
                trace = None

            _validate_completed_frame_count(
                result,
                paper_expected_frame_count(cfg, "online_rollout"),
                controller_steps_per_prediction=controller_steps_per_prediction,
            )
            results.append(result)
            if training_episode is not None and _training_episode_frame_count(training_episode) > 0:
                training_records.append((training_episode, result))
            trajectory_frame_start.append(frame_start)
            trajectory_frame_count.append(int(result.completed_steps))
            if trace is not None:
                _extend_flat_trace(flat_trace, trace)
                flat_frame_count += int(result.completed_steps)

            if _should_report_progress(local_idx, len(specs), progress_interval):
                _print_progress(results, len(specs))
            if viewer_control is not None and (viewer_control.stop_requested or not env.viewer_is_running()):
                break

        _print_summary(results)

        if save_results_npz is not None or save_vis_trace_npz is not None or save_training_pt is not None:
            compact_payload = _compact_online_payload(
                results,
                cfg=cfg,
                env=env,
                reset_source=reset_source,
                reset_source_path=reset_source_path,
                reset_source_kind=reset_source_kind,
                contact_path=contact_path,
                high_level_config_path=high_level_config_path,
                runtime=runtime,
                specs=specs,
                reset_source_count=reset_source_count,
                backend=backend,
                dt=dt,
                dataset_shift=dataset_shift,
                initial_pose_mode=initial_pose_mode,
                initial_pose_path=initial_pose_path,
                initial_wuji_q=initial_wuji_q,
                proprio_source=proprio_source,
                controller_steps_per_prediction=controller_steps_per_prediction,
            )
            if save_results_npz is not None:
                out = _save_npz(save_results_npz, compact_payload)
                print(f"saved online rollout results: {out}", flush=True)
            if save_vis_trace_npz is not None:
                compact_payload["trajectory_frame_start"] = np.asarray(trajectory_frame_start, dtype=np.int32)
                compact_payload["trajectory_frame_count"] = np.asarray(trajectory_frame_count, dtype=np.int32)
                for key, frames in flat_trace.items():
                    compact_payload[key] = np.asarray(frames)
                out = _save_vis_trace_npz(save_vis_trace_npz, compact_payload)
                print(f"saved online rollout trace: {out}", flush=True)
            if save_training_pt is not None:
                out = _save_online_training_payload(
                    save_training_pt,
                    training_records=training_records,
                    cfg=cfg,
                    env=env,
                    contact_payload=contact_payload,
                    contact_path=contact_path,
                    high_level_config_path=high_level_config_path,
                    runtime=runtime,
                    backend=backend,
                    dt=dt,
                    dataset_shift=dataset_shift,
                    initial_pose_mode=initial_pose_mode,
                    initial_pose_path=initial_pose_path,
                    initial_q=initial_wuji_q,
                    proprio_source=proprio_source,
                    controller_steps_per_prediction=controller_steps_per_prediction,
                    action_format="wuji_single_manipulator",
                    retarget_path=retarget_path,
                    object_reset_source_path=object_reset_source_path,
                    source_metadata=reset_source,
                )
                print(f"saved online rollout training payload: {out}", flush=True)

        if viewer_control is not None and not bool(OmegaConf.select(cfg, "online_rollout.viewer_no_hold_open", default=False)):
            viewer_control.hold_open(
                env,
                completed_steps=sum(result.completed_steps for result in results),
                total_steps=sum(result.action_frame_count for result in results),
            )
    finally:
        if image_renderer is not None:
            image_renderer.close()
        env.close()


def _main_robotiq_stick_armless(cfg: DictConfig) -> None:
    task_name = str(OmegaConf.select(cfg, "task.name", default="")).strip()
    if task_name not in {"pumpbottle_use", "keyboard_press"}:
        raise ValueError(
            "robotiq_gripper_stick_armless online rollout expects "
            "task.name=pumpbottle_use or keyboard_press"
        )

    contact_path = _required_path(
        OmegaConf.select(cfg, "online_rollout.contact_source_pt", default=None),
        "online_rollout.contact_source_pt",
    )
    high_level_config_path, high_level_cfg = _high_level_config_from_cfg(
        cfg,
        key="online_rollout.high_level_config_path",
    )
    low_level_config_path = _low_level_config_from_cfg(
        cfg,
        high_level_cfg,
        high_level_config_path=high_level_config_path,
    )
    low_level_checkpoint_path = _low_level_checkpoint_from_cfg(
        cfg,
        high_level_cfg,
        high_level_config_path=high_level_config_path,
    )
    low_level_manipulator = _low_level_manipulator_from_rollout_or_high_cfg(
        cfg,
        high_level_cfg,
        env_embodiment=str(cfg.env.embodiment),
    )
    low_level_use_ema = bool(OmegaConf.select(high_level_cfg, "preprocess.latent_prefer_ema", default=True))
    initial_pose_mode, initial_pose_path, initial_q = _initial_robotiq_stick_pose_from_cfg(cfg)
    object_initial_poses = load_object_start_poses(
        cfg, section="online_rollout", loader=_initial_robotiq_stick_pose_from_cfg
    )
    episode_initial_poses: list[StartPose] = []
    initial_poses_by_source_index: dict[int, StartPose] = {}

    contact_payload = load_high_level_payload(contact_path)
    contact = _payload_arrays(contact_payload)
    _validate_contact_source(contact)
    if task_name == "pumpbottle_use":
        _validate_pump_bottle_contact_source(contact_payload, contact_path)
    else:
        _validate_keyboard_contact_source(contact_payload, contact_path)

    contact_ranges = _episode_ranges(np.asarray(tensor_to_numpy(contact_payload["episode_ends"]), dtype=np.int64))
    selected = _selected_indices_from_cfg(
        len(contact_ranges),
        indices=OmegaConf.select(cfg, "online_rollout.indices", default=None),
        start_index=int(OmegaConf.select(cfg, "online_rollout.start_index", default=0)),
        count=OmegaConf.select(cfg, "online_rollout.count", default=None),
    )
    if not selected:
        raise ValueError("online_rollout selected no episodes")

    dataset_shift = _dataset_xyz_translation(contact_payload)
    runtime = _load_online_low_level_runtime(
        low_level_config_path=low_level_config_path,
        low_level_checkpoint_path=low_level_checkpoint_path,
        low_level_manipulator=low_level_manipulator,
        low_level_use_ema=low_level_use_ema,
        device=str(OmegaConf.select(cfg, "online_rollout.device", default="auto")),
    )
    if int(runtime.local_dof) != ROBOTIQ_STICK_LOCAL_DOF:
        raise ValueError(
            f"Expected {ROBOTIQ_STICK_LOCAL_DOF}D {ROBOTIQ_STICK_ROBOT_NAME} runtime, got {runtime.local_dof}"
        )
    translation_groups = _composite_translation_groups(runtime.robot_model, runtime.local_dof)

    save_training_pt = optional_project_path(OmegaConf.select(cfg, "online_rollout.save_training_pt", default=None))
    _configure_camera_for_runtime(cfg, runtime, force_camera_obs=save_training_pt is not None)
    specs = _build_contact_only_episode_specs(
        selected,
        contact_payload=contact_payload,
        contact_ranges=contact_ranges,
        max_frames=OmegaConf.select(cfg, "online_rollout.max_frames", default=None),
    )
    _validate_expected_frame_count(
        specs,
        paper_expected_frame_count(cfg, "online_rollout"),
    )
    object_ids = (
        [
            _object_suite_episode_id(contact_payload, spec.contact_episode_index, label="contact source")
            for spec in specs
        ]
        if task_name == "pumpbottle_use"
        and _pump_bottle_uses_episode_objects(contact_payload)
        else []
    )

    backend = str(OmegaConf.select(cfg, "online_rollout.backend", default=cfg.env.backend))
    viewer_enabled = bool(OmegaConf.select(cfg, "online_rollout.render", default=True))
    viewer_control = (
        ViewerReplayControl(
            start_paused=bool(OmegaConf.select(cfg, "online_rollout.viewer_start_paused", default=False))
        )
        if viewer_enabled
        else None
    )
    save_vis_trace_npz = optional_project_path(OmegaConf.select(cfg, "online_rollout.save_vis_trace_npz", default=None))
    save_results_npz = optional_project_path(OmegaConf.select(cfg, "online_rollout.save_results_npz", default=None))
    store_trace = save_vis_trace_npz is not None
    capture_camera_every_controller_step = bool(
        OmegaConf.select(cfg, "online_rollout.capture_camera_every_controller_step", default=False)
    )
    render_images_config = _render_images_config_from_cfg(cfg)
    continue_on_error = bool(OmegaConf.select(cfg, "online_rollout.continue_on_error", default=False))
    progress_interval = int(OmegaConf.select(cfg, "online_rollout.progress_interval", default=1))
    seed_offset = int(OmegaConf.select(cfg, "online_rollout.seed_offset", default=0))
    proprio_source = _proprio_source_from_cfg(cfg)
    pump_bottle_lift_success_metric = _pump_bottle_lift_success_metric_from_cfg(
        cfg,
        section="online_rollout",
        default="max_lift_delta",
    )
    print_button_slide_each_step = bool(
        OmegaConf.select(cfg, "online_rollout.print_button_slide_each_step", default=False)
    )
    controller_steps_per_prediction = _positive_int_from_cfg(
        cfg,
        "online_rollout.controller_steps_per_prediction",
        default=1,
    )
    functional_mapping_enabled, functional_mapping_gain = (
        _robotiq_functional_contact_mapping_from_cfg(cfg)
    )
    gripper_min_enabled, gripper_min_command, gripper_min_phases = (
        _robotiq_gripper_min_command_from_cfg(cfg)
    )
    stick_offset_enabled, stick_offset_xyz, stick_offset_phases = (
        _robotiq_stick_execution_offset_from_cfg(cfg)
    )
    process_count = validate_headless_parallel(
        cfg,
        section="online_rollout",
        render=viewer_enabled,
        save_vis_trace_npz=save_vis_trace_npz,
    )
    _validate_render_images_runtime(render_images_config, process_count, viewer_enabled=viewer_enabled)
    print(
        f"loaded contact={contact_path} selected={len(specs)} "
        f"policy_frames={sum(spec.frame_count for spec in specs)} shift={dataset_shift.tolist()} "
        f"translation_groups={translation_groups} high_level_config={high_level_config_path} "
        f"low_level={runtime.low_level_checkpoint_path} manipulator={runtime.low_level_manipulator} "
        f"backend={runtime.encoder_backend} non_contact_enabled={runtime.non_contact_enabled} "
        f"initial_pose_mode={initial_pose_mode} initial_pose_pt={initial_pose_path} "
        f"proprio_source={proprio_source} controller_steps_per_prediction={controller_steps_per_prediction} "
        f"functional_contact_mapping={functional_mapping_enabled} "
        f"functional_contact_mapping_gain={functional_mapping_gain} "
        f"gripper_min={gripper_min_enabled}:{gripper_min_command}:"
        f"{sorted(gripper_min_phases)} "
        f"stick_offset={stick_offset_enabled}:{stick_offset_xyz.tolist()}:"
        f"{sorted(stick_offset_phases)} "
        f"pump_bottle_lift_success_metric={pump_bottle_lift_success_metric} "
        f"print_button_slide_each_step={print_button_slide_each_step}"
        f"{'' if not render_images_config.enabled else f' render_images_dir={render_images_config.output_dir}'}"
        f"{'' if save_training_pt is None else f' save_training_pt={save_training_pt}'}",
        flush=True,
    )

    if process_count > 1:
        print(
            f"running armless {task_name} online rollout in parallel "
            f"process_count={process_count} selected={len(specs)}",
            flush=True,
        )
        del runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with tempfile.TemporaryDirectory(prefix=f"{task_name}_low_level_parallel_") as tmp:
            result_paths = launch_sharded_workers(
                script_path=Path(__file__).resolve(),
                cfg=cfg,
                section="online_rollout",
                selected=[int(spec.source_trajectory_index) for spec in specs],
                process_count=process_count,
                tmpdir=Path(tmp),
                result_prefix="online_rollout",
                seed_offset_key="online_rollout.seed_offset",
                extra_worker_outputs=(
                    {"online_rollout.save_training_pt": ("online_rollout_training", ".pt")}
                    if save_training_pt is not None
                    else None
                ),
            )
            merged_payload = _merge_result_payloads(result_paths)
            training_payload_paths = sorted(Path(tmp).glob("online_rollout_training_worker_*.pt"))
            if save_training_pt is not None:
                out = _merge_training_payloads(training_payload_paths, save_training_pt)
                print(f"saved online rollout training payload: {out}", flush=True)
        results = _results_from_payload(merged_payload)
        _print_summary(results)
        if task_name == "pumpbottle_use":
            _print_ordered_pump_payload_summary(merged_payload)
        if save_results_npz is not None:
            out = _save_npz(save_results_npz, merged_payload)
            print(f"saved online rollout results: {out}", flush=True)
        return

    env = build_env_from_config(
        _config_with_active_object(cfg, object_ids[0]) if object_ids else cfg,
        backend=backend,
        render=viewer_enabled,
        viewer_key_callback=viewer_control.key_callback if viewer_control is not None else None,
    )
    dt = float(env.model.opt.timestep) * int(env.backend.frame_skip)
    validate_paper_control_dt(cfg, dt)
    results: list[PumpBottleRolloutResult] = []
    flat_trace: dict[str, list[np.ndarray]] = {}
    trajectory_frame_start: list[int] = []
    trajectory_frame_count: list[int] = []
    training_records: list[tuple[OnlineTrainingEpisode, PumpBottleRolloutResult]] = []
    flat_frame_count = 0
    image_renderer = (
        OnlineRolloutImageRenderer(render_images_config, task_name=task_name, embodiment=str(cfg.env.embodiment))
        if render_images_config.enabled
        else None
    )

    try:
        for local_idx, spec in enumerate(specs):
            if object_ids and local_idx > 0:
                env.close()
                env = build_env_from_config(
                    _config_with_active_object(cfg, object_ids[local_idx]),
                    backend=backend,
                    render=viewer_enabled,
                    viewer_key_callback=viewer_control.key_callback if viewer_control is not None else None,
                )
                episode_dt = float(env.model.opt.timestep) * int(env.backend.frame_skip)
                if not np.isclose(episode_dt, dt):
                    raise ValueError(f"Pump-bottle suite env dt changed from {dt} to {episode_dt}")
            if object_ids:
                print(
                    f"trajectory={spec.source_trajectory_index} object_id={object_ids[local_idx]}",
                    flush=True,
                )
            trace = new_trace_buffers() if store_trace else None
            training_episode = OnlineTrainingEpisode(spec=spec) if save_training_pt is not None else None
            frame_start = flat_frame_count
            try:
                episode_initial_pose = object_initial_poses.get(
                    object_ids[local_idx] if object_ids else str(
                        OmegaConf.select(cfg, "pump_bottle.active_id", default="")
                    ),
                    (initial_pose_mode, initial_pose_path, initial_q),
                )
                if object_initial_poses:
                    episode_initial_poses.append(episode_initial_pose)
                    initial_poses_by_source_index[int(spec.source_trajectory_index)] = episode_initial_pose
                result = _rollout_episode_robotiq_stick(
                    env=env,
                    cfg=cfg,
                    spec=spec,
                    contact=contact,
                    runtime=runtime,
                    dataset_shift=dataset_shift,
                    translation_groups=translation_groups,
                    initial_q=episode_initial_pose[2],
                    proprio_source=proprio_source,
                    controller_steps_per_prediction=controller_steps_per_prediction,
                    store_trace=store_trace,
                    trace=trace,
                    viewer_control=viewer_control,
                    dt=dt,
                    seed=int(OmegaConf.select(cfg, "online_rollout.seed", default=0))
                    + (int(seed_offset) + int(local_idx)) * 100003,
                    capture_camera_every_controller_step=capture_camera_every_controller_step,
                    pump_bottle_lift_success_metric=pump_bottle_lift_success_metric,
                    print_button_slide_each_step=print_button_slide_each_step,
                    training_episode=training_episode,
                    image_renderer=image_renderer,
                    profiler=RolloutProfiler(
                        enabled=bool(OmegaConf.select(cfg, "online_rollout.profile.enabled", default=False)),
                        device=runtime.device,
                        label=f"trajectory {local_idx + 1}/{len(specs)} profile",
                    ),
                )
            except Exception as exc:
                if not continue_on_error:
                    raise RuntimeError(f"online rollout trajectory {spec.source_trajectory_index} failed") from exc
                result = PumpBottleRolloutResult(
                    source_index=int(spec.source_trajectory_index),
                    source_seed=None if spec.source_trajectory_seed < 0 else int(spec.source_trajectory_seed),
                    action_frame_count=int(spec.frame_count),
                    completed_steps=0,
                    success=False,
                    max_cube_z=float("nan"),
                    final_cube_pose=np.full((7,), np.nan, dtype=np.float32),
                    error=f"{type(exc).__name__}: {exc}",
                )
                trace = None

            _validate_completed_frame_count(
                result,
                paper_expected_frame_count(cfg, "online_rollout"),
                controller_steps_per_prediction=controller_steps_per_prediction,
            )
            results.append(result)
            if training_episode is not None and _training_episode_frame_count(training_episode) > 0:
                training_records.append((training_episode, result))
            trajectory_frame_start.append(frame_start)
            trajectory_frame_count.append(int(result.completed_steps))
            if trace is not None:
                _extend_flat_trace(flat_trace, trace)
                flat_frame_count += int(result.completed_steps)

            if _should_report_progress(local_idx, len(specs), progress_interval):
                _print_progress(results, len(specs))
            if viewer_control is not None and (viewer_control.stop_requested or not env.viewer_is_running()):
                break

        _print_summary(results)
        if task_name == "pumpbottle_use":
            _print_ordered_pump_summary(results)

        if save_results_npz is not None or save_vis_trace_npz is not None or save_training_pt is not None:
            compact_payload = _compact_online_payload_robotiq_stick(
                results,
                cfg=cfg,
                env=env,
                contact_payload=contact_payload,
                contact_path=contact_path,
                high_level_config_path=high_level_config_path,
                runtime=runtime,
                specs=specs,
                source_count=len(contact_ranges),
                backend=backend,
                dt=dt,
                dataset_shift=dataset_shift,
                translation_groups=translation_groups,
                initial_pose_mode=initial_pose_mode,
                initial_pose_path=initial_pose_path,
                initial_q=initial_q,
                proprio_source=proprio_source,
                controller_steps_per_prediction=controller_steps_per_prediction,
            )
            if object_initial_poses:
                add_episode_start_pose_metadata(compact_payload, episode_initial_poses)
            if save_results_npz is not None:
                out = _save_npz(save_results_npz, compact_payload)
                print(f"saved online rollout results: {out}", flush=True)
            if save_vis_trace_npz is not None:
                compact_payload["trajectory_frame_start"] = np.asarray(trajectory_frame_start, dtype=np.int32)
                compact_payload["trajectory_frame_count"] = np.asarray(trajectory_frame_count, dtype=np.int32)
                for key, frames in flat_trace.items():
                    compact_payload[key] = np.asarray(frames)
                out = _save_vis_trace_npz(save_vis_trace_npz, compact_payload)
                print(f"saved online rollout trace: {out}", flush=True)
            if save_training_pt is not None:
                out = _save_online_training_payload(
                    save_training_pt,
                    training_records=training_records,
                    cfg=cfg,
                    env=env,
                    contact_payload=contact_payload,
                    contact_path=contact_path,
                    high_level_config_path=high_level_config_path,
                    runtime=runtime,
                    backend=backend,
                    dt=dt,
                    dataset_shift=dataset_shift,
                    initial_pose_mode=initial_pose_mode,
                    initial_pose_path=initial_pose_path,
                    initial_q=initial_q,
                    proprio_source=proprio_source,
                    controller_steps_per_prediction=controller_steps_per_prediction,
                    action_format="robotiq_stick_robot_model_q",
                    translation_groups=translation_groups,
                    initial_poses_by_source_index=initial_poses_by_source_index or None,
                )
                print(f"saved online rollout training payload: {out}", flush=True)

        if viewer_control is not None and not bool(OmegaConf.select(cfg, "online_rollout.viewer_no_hold_open", default=False)):
            viewer_control.hold_open(
                env,
                completed_steps=sum(result.completed_steps for result in results),
                total_steps=sum(result.action_frame_count for result in results),
            )
    finally:
        if image_renderer is not None:
            image_renderer.close()
        env.close()


def _main_floating_robot_model_hand(cfg: DictConfig) -> None:
    task_name = str(OmegaConf.select(cfg, "task.name", default="cube_pick")).strip()
    if task_name not in {"cube_pick", "object_suite_pick_place", "keyboard_press", "pumpbottle_use"}:
        raise ValueError(
            "floating Allegro/Barrett/WujiHand online rollout expects "
            "task.name=cube_pick, object_suite_pick_place, keyboard_press, or pumpbottle_use"
        )
    embodiment = str(cfg.env.embodiment)
    if embodiment not in {"allegro_right", "barrett", "wujihand_right"}:
        raise ValueError(f"Unsupported floating hand embodiment={embodiment!r}")

    contact_path = _required_path(
        OmegaConf.select(cfg, "online_rollout.contact_source_pt", default=None),
        "online_rollout.contact_source_pt",
    )
    robot_model_source_path = _floating_robot_model_source_path_from_cfg(cfg, task_name=task_name)
    object_reset_source_path = _optional_path(
        OmegaConf.select(cfg, "online_rollout.object_reset_source_pt", default=None)
    )
    if robot_model_source_path is not None and object_reset_source_path is not None:
        raise ValueError(
            "Set only one of online_rollout.robot_model_q_source_pt and "
            "online_rollout.object_reset_source_pt"
        )
    reset_source_path = robot_model_source_path or object_reset_source_path
    high_level_config_path, high_level_cfg = _high_level_config_from_cfg(
        cfg,
        key="online_rollout.high_level_config_path",
    )
    low_level_config_path = _low_level_config_from_cfg(
        cfg,
        high_level_cfg,
        high_level_config_path=high_level_config_path,
    )
    low_level_checkpoint_path = _low_level_checkpoint_from_cfg(
        cfg,
        high_level_cfg,
        high_level_config_path=high_level_config_path,
    )
    low_level_manipulator = _low_level_manipulator_from_rollout_or_high_cfg(
        cfg,
        high_level_cfg,
        env_embodiment=embodiment,
    )
    low_level_use_ema = bool(OmegaConf.select(high_level_cfg, "preprocess.latent_prefer_ema", default=True))

    contact_payload = load_high_level_payload(contact_path)
    contact = _payload_arrays(contact_payload)
    _validate_contact_source(contact)
    contact_input_mode = _contact_input_mode_from_cfg(cfg)
    if contact_input_mode == "latent":
        _validate_latent_contact_source(contact_payload, contact_path)
    if task_name == "keyboard_press":
        _validate_keyboard_contact_source(contact_payload, contact_path)
    elif task_name == "pumpbottle_use":
        _validate_pump_bottle_contact_source(contact_payload, contact_path)
    elif task_name == "object_suite_pick_place":
        _validate_object_suite_contact_source(contact_payload, contact_path)
    source = None
    if reset_source_path is not None:
        source = _load_training_pt(reset_source_path)
        if robot_model_source_path is not None:
            _validate_robot_model_q_source(source, embodiment=embodiment, source_path=robot_model_source_path)
        else:
            _validate_object_reset_source(source, source_path=reset_source_path)

    contact_ranges = _episode_ranges(np.asarray(tensor_to_numpy(contact_payload["episode_ends"]), dtype=np.int64))
    selected = _selected_indices_from_cfg(
        len(contact_ranges),
        indices=OmegaConf.select(cfg, "online_rollout.indices", default=None),
        start_index=int(OmegaConf.select(cfg, "online_rollout.start_index", default=0)),
        count=OmegaConf.select(cfg, "online_rollout.count", default=None),
    )
    if not selected:
        raise ValueError("online_rollout selected no episodes")

    dataset_shift = _dataset_xyz_translation(contact_payload)
    runtime = _load_online_low_level_runtime(
        low_level_config_path=low_level_config_path,
        low_level_checkpoint_path=low_level_checkpoint_path,
        low_level_manipulator=low_level_manipulator,
        low_level_use_ema=low_level_use_ema,
        device=str(OmegaConf.select(cfg, "online_rollout.device", default="auto")),
    )
    if int(runtime.local_dof) != int(getattr(runtime.robot_model, "dof", runtime.local_dof)):
        raise ValueError(
            f"Runtime action_dim={runtime.local_dof} does not match robot_model.dof="
            f"{getattr(runtime.robot_model, 'dof', None)}"
        )
    translation_groups = _composite_translation_groups(runtime.robot_model, runtime.local_dof)
    initial_pose_mode, initial_pose_path, initial_robot_model_q = _robot_model_q_initial_pose_from_cfg(
        cfg,
        runtime=runtime,
        embodiment=embodiment,
        source_path=robot_model_source_path,
    )
    object_initial_poses = load_object_start_poses(
        cfg,
        section="online_rollout",
        loader=lambda scoped: _robot_model_q_initial_pose_from_cfg(
            scoped, runtime=runtime, embodiment=embodiment, source_path=robot_model_source_path
        ),
    )
    episode_initial_poses: list[StartPose] = []
    initial_poses_by_source_index: dict[int, StartPose] = {}

    save_training_pt = optional_project_path(OmegaConf.select(cfg, "online_rollout.save_training_pt", default=None))
    _configure_camera_for_runtime(cfg, runtime, force_camera_obs=save_training_pt is not None)
    specs = _build_contact_only_episode_specs(
        selected,
        contact_payload=contact_payload,
        contact_ranges=contact_ranges,
        max_frames=OmegaConf.select(cfg, "online_rollout.max_frames", default=None),
    )
    _validate_expected_frame_count(
        specs,
        paper_expected_frame_count(cfg, "online_rollout"),
    )
    source_count = _trajectory_count(source) if source is not None else len(contact_ranges)
    if source is not None:
        if robot_model_source_path is not None:
            _validate_robot_model_source_specs(
                source,
                specs,
                source_count=source_count,
                expected_dim=runtime.local_dof,
            )
        else:
            _validate_object_reset_source_specs(source, specs, source_count=source_count)
    elif initial_robot_model_q is None:
        raise ValueError(
            "online_rollout.robot_model_q_source_pt is required when "
            "online_rollout.initial_pose.mode is robot_model_q_source"
        )
    object_ids: list[str] = []
    episode_pump_bottle = task_name == "pumpbottle_use" and _pump_bottle_uses_episode_objects(contact_payload)
    if task_name == "object_suite_pick_place" or episode_pump_bottle:
        if task_name == "object_suite_pick_place" and source is None:
            raise ValueError(f"{task_name} floating-hand rollout requires a reset source")
        for spec in specs:
            contact_object_id = _object_suite_episode_id(
                contact_payload,
                spec.contact_episode_index,
                label="contact source",
            )
            if source is not None:
                reset_object_id = _object_suite_source_id(
                    source,
                    spec.source_trajectory_index,
                    label="robot_model_q source" if robot_model_source_path is not None else "object reset source",
                )
                if contact_object_id != reset_object_id:
                    raise ValueError(
                        f"Object mismatch for source trajectory {spec.source_trajectory_index}: "
                        f"contact={contact_object_id!r} reset_source={reset_object_id!r}"
                    )
            object_ids.append(contact_object_id)

    backend = str(OmegaConf.select(cfg, "online_rollout.backend", default=cfg.env.backend))
    viewer_enabled = bool(OmegaConf.select(cfg, "online_rollout.render", default=True))
    viewer_control = (
        ViewerReplayControl(
            start_paused=bool(OmegaConf.select(cfg, "online_rollout.viewer_start_paused", default=False))
        )
        if viewer_enabled
        else None
    )
    save_vis_trace_npz = optional_project_path(OmegaConf.select(cfg, "online_rollout.save_vis_trace_npz", default=None))
    save_results_npz = optional_project_path(OmegaConf.select(cfg, "online_rollout.save_results_npz", default=None))
    store_trace = save_vis_trace_npz is not None
    capture_camera_every_controller_step = bool(
        OmegaConf.select(cfg, "online_rollout.capture_camera_every_controller_step", default=False)
    )
    render_images_config = _render_images_config_from_cfg(cfg)
    continue_on_error = bool(OmegaConf.select(cfg, "online_rollout.continue_on_error", default=False))
    progress_interval = int(OmegaConf.select(cfg, "online_rollout.progress_interval", default=1))
    seed_offset = int(OmegaConf.select(cfg, "online_rollout.seed_offset", default=0))
    proprio_source = _proprio_source_from_cfg(cfg)
    pump_bottle_lift_success_metric = _pump_bottle_lift_success_metric_from_cfg(
        cfg,
        section="online_rollout",
        default="max_lift_delta",
    )
    print_button_slide_each_step = bool(
        OmegaConf.select(cfg, "online_rollout.print_button_slide_each_step", default=False)
    )
    controller_steps_per_prediction = _positive_int_from_cfg(
        cfg,
        "online_rollout.controller_steps_per_prediction",
        default=1,
    )
    process_count = validate_headless_parallel(
        cfg,
        section="online_rollout",
        render=viewer_enabled,
        save_vis_trace_npz=save_vis_trace_npz,
    )
    _validate_render_images_runtime(render_images_config, process_count, viewer_enabled=viewer_enabled)

    grasp_hold_enabled, grasp_hold_capture_phase_ids, grasp_hold_apply_phase_ids = (
        _grasp_hold_config_from_cfg(cfg)
    )
    (
        execution_root_offset_enabled,
        execution_root_offset_xyz,
        execution_root_offset_phase_ids,
    ) = _robot_model_execution_root_offset_from_cfg(cfg)
    (
        effector_execution_offset_enabled,
        effector_execution_offset_phase_ids,
        effector_execution_offset_indices,
        effector_execution_offset_values,
    ) = _effector_execution_offset_from_cfg(cfg)
    contact_dt = _payload_dt(contact_payload)
    configured_dt = float(OmegaConf.select(cfg, "env.timestep", default=0.002)) * int(cfg.env.frame_skip)
    if (
        bool(OmegaConf.select(cfg, "online_rollout.require_contact_dt_match", default=False))
        and contact_dt is not None
        and not np.isclose(configured_dt, contact_dt)
    ):
        raise ValueError(
            "Configured rollout action dt does not match the contact payload: "
            f"env.timestep * env.frame_skip={configured_dt:.6f}, contact dt={contact_dt:.6f}"
        )

    print(
        f"loaded contact={contact_path}"
        f"{'' if robot_model_source_path is None else f' robot_model_q_source={robot_model_source_path}'} "
        f"{'' if object_reset_source_path is None else f' object_reset_source={object_reset_source_path}'} "
        f"selected={len(specs)} policy_frames={sum(spec.frame_count for spec in specs)} "
        f"shift={dataset_shift.tolist()} translation_groups={translation_groups} "
        f"high_level_config={high_level_config_path} low_level={runtime.low_level_checkpoint_path} "
        f"manipulator={runtime.low_level_manipulator} backend={runtime.encoder_backend} "
        f"non_contact_enabled={runtime.non_contact_enabled} proprio_source={proprio_source} "
        f"contact_input={contact_input_mode} "
        f"initial_pose_mode={initial_pose_mode} initial_pose_pt={initial_pose_path} "
        f"controller_steps_per_prediction={controller_steps_per_prediction} "
        f"contact_dt={contact_dt} configured_dt={configured_dt} "
        f"grasp_hold={grasp_hold_enabled} "
        f"grasp_hold_capture={sorted(grasp_hold_capture_phase_ids)} "
        f"grasp_hold_apply={sorted(grasp_hold_apply_phase_ids)} "
        f"execution_root_offset={execution_root_offset_enabled} "
        f"execution_root_offset_xyz={execution_root_offset_xyz.tolist()} "
        f"execution_root_offset_phases={sorted(execution_root_offset_phase_ids)} "
        f"effector_execution_offset={effector_execution_offset_enabled} "
        f"effector_execution_offset_phases={sorted(effector_execution_offset_phase_ids)} "
        f"effector_execution_offset_indices={effector_execution_offset_indices.tolist()} "
        f"effector_execution_offset_values={effector_execution_offset_values.tolist()} "
        f"pump_bottle_lift_success_metric={pump_bottle_lift_success_metric} "
        f"print_button_slide_each_step={print_button_slide_each_step}"
        f"{'' if not render_images_config.enabled else f' render_images_dir={render_images_config.output_dir}'}"
        f"{'' if save_training_pt is None else f' save_training_pt={save_training_pt}'}",
        flush=True,
    )

    if process_count > 1:
        print(
            f"running floating {embodiment} online rollout in parallel "
            f"process_count={process_count} selected={len(specs)}",
            flush=True,
        )
        del runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with tempfile.TemporaryDirectory(prefix=f"{embodiment}_low_level_parallel_") as tmp:
            result_paths = launch_sharded_workers(
                script_path=Path(__file__).resolve(),
                cfg=cfg,
                section="online_rollout",
                selected=[int(spec.source_trajectory_index) for spec in specs],
                process_count=process_count,
                tmpdir=Path(tmp),
                result_prefix="online_rollout",
                seed_offset_key="online_rollout.seed_offset",
                extra_worker_outputs=(
                    {"online_rollout.save_training_pt": ("online_rollout_training", ".pt")}
                    if save_training_pt is not None
                    else None
                ),
            )
            merged_payload = _merge_result_payloads(result_paths)
            training_payload_paths = sorted(Path(tmp).glob("online_rollout_training_worker_*.pt"))
            if save_training_pt is not None:
                out = _merge_training_payloads(training_payload_paths, save_training_pt)
                print(f"saved online rollout training payload: {out}", flush=True)
        results = _results_from_payload(merged_payload)
        _print_summary(results)
        if task_name == "pumpbottle_use":
            _print_ordered_pump_payload_summary(merged_payload)
        if save_results_npz is not None:
            out = _save_npz(save_results_npz, merged_payload)
            print(f"saved online rollout results: {out}", flush=True)
        return

    initial_env_cfg = _config_with_active_object(cfg, object_ids[0]) if object_ids else cfg
    env = build_env_from_config(
        initial_env_cfg,
        backend=backend,
        render=viewer_enabled,
        viewer_key_callback=viewer_control.key_callback if viewer_control is not None else None,
    )
    if robot_model_source_path is not None and source is not None:
        _validate_robot_model_q_source_matches_env(source, env)
        _validate_robot_model_source_env_dim(source, specs, env)
    dt = float(env.model.opt.timestep) * int(env.backend.frame_skip)
    validate_paper_control_dt(cfg, dt)
    results: list[PumpBottleRolloutResult] = []
    flat_trace: dict[str, list[np.ndarray]] = {}
    trajectory_frame_start: list[int] = []
    trajectory_frame_count: list[int] = []
    training_records: list[tuple[OnlineTrainingEpisode, PumpBottleRolloutResult]] = []
    flat_frame_count = 0
    image_renderer = (
        OnlineRolloutImageRenderer(render_images_config, task_name=task_name, embodiment=embodiment)
        if render_images_config.enabled
        else None
    )

    try:
        for local_idx, spec in enumerate(specs):
            if object_ids and local_idx > 0:
                env.close()
                env = build_env_from_config(
                    _config_with_active_object(cfg, object_ids[local_idx]),
                    backend=backend,
                    render=viewer_enabled,
                    viewer_key_callback=viewer_control.key_callback if viewer_control is not None else None,
                )
                if robot_model_source_path is not None and source is not None:
                    _validate_robot_model_q_source_matches_env(source, env)
                episode_dt = float(env.model.opt.timestep) * int(env.backend.frame_skip)
                if not np.isclose(episode_dt, dt):
                    raise ValueError(f"Object-suite env dt changed from {dt} to {episode_dt}")
            if object_ids:
                print(
                    f"trajectory={spec.source_trajectory_index} object_id={object_ids[local_idx]}",
                    flush=True,
                )
            trace = new_trace_buffers() if store_trace else None
            training_episode = OnlineTrainingEpisode(spec=spec) if save_training_pt is not None else None
            frame_start = flat_frame_count
            episode_initial_pose = object_initial_poses.get(
                object_ids[local_idx] if object_ids else str(
                    OmegaConf.select(cfg, "pump_bottle.active_id", default="")
                ),
                (initial_pose_mode, initial_pose_path, initial_robot_model_q),
            )
            if object_initial_poses:
                episode_initial_poses.append(episode_initial_pose)
                initial_poses_by_source_index[int(spec.source_trajectory_index)] = episode_initial_pose
            try:
                result = _rollout_episode_robot_model_hand(
                    env=env,
                    cfg=cfg,
                    spec=spec,
                    contact=contact,
                    source=source,
                    runtime=runtime,
                    contact_input_mode=contact_input_mode,
                    dataset_shift=dataset_shift,
                    translation_groups=translation_groups,
                    proprio_source=proprio_source,
                    controller_steps_per_prediction=controller_steps_per_prediction,
                    store_trace=store_trace,
                    trace=trace,
                    viewer_control=viewer_control,
                    dt=dt,
                    seed=int(OmegaConf.select(cfg, "online_rollout.seed", default=0))
                    + (int(seed_offset) + int(local_idx)) * 100003,
                    capture_camera_every_controller_step=capture_camera_every_controller_step,
                    initial_q_override=episode_initial_pose[2],
                    pump_bottle_lift_success_metric=pump_bottle_lift_success_metric,
                    print_button_slide_each_step=print_button_slide_each_step,
                    training_episode=training_episode,
                    image_renderer=image_renderer,
                    profiler=RolloutProfiler(
                        enabled=bool(OmegaConf.select(cfg, "online_rollout.profile.enabled", default=False)),
                        device=runtime.device,
                        label=f"trajectory {local_idx + 1}/{len(specs)} profile",
                    ),
                )
            except Exception as exc:
                if not continue_on_error:
                    raise RuntimeError(f"online rollout trajectory {spec.source_trajectory_index} failed") from exc
                result = PumpBottleRolloutResult(
                    source_index=int(spec.source_trajectory_index),
                    source_seed=None if spec.source_trajectory_seed < 0 else int(spec.source_trajectory_seed),
                    action_frame_count=int(spec.frame_count),
                    completed_steps=0,
                    success=False,
                    max_cube_z=float("nan"),
                    final_cube_pose=np.full((7,), np.nan, dtype=np.float32),
                    error=f"{type(exc).__name__}: {exc}",
                )
                trace = None

            _validate_completed_frame_count(
                result,
                paper_expected_frame_count(cfg, "online_rollout"),
                controller_steps_per_prediction=controller_steps_per_prediction,
            )
            results.append(result)
            if training_episode is not None and _training_episode_frame_count(training_episode) > 0:
                training_records.append((training_episode, result))
            trajectory_frame_start.append(frame_start)
            trajectory_frame_count.append(int(result.completed_steps))
            if trace is not None:
                _extend_flat_trace(flat_trace, trace)
                flat_frame_count += int(result.completed_steps)

            if _should_report_progress(local_idx, len(specs), progress_interval):
                _print_progress(results, len(specs))
            if viewer_control is not None and (viewer_control.stop_requested or not env.viewer_is_running()):
                break

        _print_summary(results)
        if task_name == "pumpbottle_use":
            _print_ordered_pump_summary(results)

        if save_results_npz is not None or save_vis_trace_npz is not None or save_training_pt is not None:
            compact_payload = _compact_online_payload_robot_model_hand(
                results,
                cfg=cfg,
                env=env,
                contact_payload=contact_payload,
                source=source,
                source_path=reset_source_path,
                contact_path=contact_path,
                high_level_config_path=high_level_config_path,
                runtime=runtime,
                specs=specs,
                source_count=source_count,
                backend=backend,
                dt=dt,
                dataset_shift=dataset_shift,
                translation_groups=translation_groups,
                proprio_source=proprio_source,
                controller_steps_per_prediction=controller_steps_per_prediction,
                initial_pose_mode=initial_pose_mode,
                initial_pose_path=initial_pose_path,
                initial_q=initial_robot_model_q,
            )
            if object_initial_poses:
                add_episode_start_pose_metadata(compact_payload, episode_initial_poses)
            if save_results_npz is not None:
                out = _save_npz(save_results_npz, compact_payload)
                print(f"saved online rollout results: {out}", flush=True)
            if save_vis_trace_npz is not None:
                compact_payload["trajectory_frame_start"] = np.asarray(trajectory_frame_start, dtype=np.int32)
                compact_payload["trajectory_frame_count"] = np.asarray(trajectory_frame_count, dtype=np.int32)
                for key, frames in flat_trace.items():
                    compact_payload[key] = np.asarray(frames)
                out = _save_vis_trace_npz(save_vis_trace_npz, compact_payload)
                print(f"saved online rollout trace: {out}", flush=True)
            if save_training_pt is not None:
                out = _save_online_training_payload(
                    save_training_pt,
                    training_records=training_records,
                    cfg=cfg,
                    env=env,
                    contact_payload=contact_payload,
                    contact_path=contact_path,
                    high_level_config_path=high_level_config_path,
                    runtime=runtime,
                    backend=backend,
                    dt=dt,
                    dataset_shift=dataset_shift,
                    initial_pose_mode=initial_pose_mode,
                    initial_pose_path=initial_pose_path,
                    initial_q=initial_robot_model_q,
                    proprio_source=proprio_source,
                    controller_steps_per_prediction=controller_steps_per_prediction,
                    action_format="robot_model_q",
                    translation_groups=translation_groups,
                    robot_model_q_source_path=robot_model_source_path,
                    object_reset_source_path=object_reset_source_path,
                    source_metadata=source,
                    initial_poses_by_source_index=initial_poses_by_source_index or None,
                )
                print(f"saved online rollout training payload: {out}", flush=True)

        if viewer_control is not None and not bool(OmegaConf.select(cfg, "online_rollout.viewer_no_hold_open", default=False)):
            viewer_control.hold_open(
                env,
                completed_steps=sum(result.completed_steps for result in results),
                total_steps=sum(result.action_frame_count for result in results),
            )
    finally:
        if image_renderer is not None:
            image_renderer.close()
        env.close()


def _required_path(value: Any, label: str) -> Path:
    path = _optional_path(value)
    if path is None:
        raise ValueError(f"{label} must be set")
    return path


def _floating_robot_model_source_path_from_cfg(cfg: DictConfig, *, task_name: str) -> Path | None:
    path = _optional_path(OmegaConf.select(cfg, "online_rollout.robot_model_q_source_pt", default=None))
    object_reset_source_path = _optional_path(
        OmegaConf.select(cfg, "online_rollout.object_reset_source_pt", default=None)
    )
    initial_mode = str(
        OmegaConf.select(cfg, "online_rollout.initial_pose.mode", default="robot_model_q_source")
    ).strip().lower()
    if task_name == "object_suite_pick_place" and initial_mode == "saved" and object_reset_source_path is not None:
        return path
    if task_name in {"keyboard_press", "pumpbottle_use"} and initial_mode == "saved":
        return path
    if path is None:
        raise ValueError(
            "online_rollout.robot_model_q_source_pt must be set for floating Allegro/Barrett/WujiHand "
            f"task.name={task_name!r} with initial_pose.mode={initial_mode!r}"
        )
    return path


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def _contact_input_mode_from_cfg(cfg: DictConfig) -> str:
    mode = str(OmegaConf.select(cfg, "online_rollout.contact_input", default="contact_points")).strip().lower()
    aliases = {
        "contact": "contact_points",
        "contact_points": "contact_points",
        "points": "contact_points",
        "latent": "latent",
        "contact_latent": "latent",
    }
    if mode not in aliases:
        raise ValueError("online_rollout.contact_input must be contact_points or latent")
    return aliases[mode]


def _pump_bottle_uses_episode_objects(payload: dict[str, Any]) -> bool:
    """Honor explicit bottle IDs in single-object shards as well as mixed datasets."""
    if str(payload.get("meta", {}).get("pump_bottle_mode", "single_005")) == "mixed_three":
        return True
    return any(
        isinstance(item, dict) and bool(str(item.get("object_id", "") or "").strip())
        for item in payload.get("episode_meta", [])
    )


def _object_suite_episode_id(payload: dict[str, Any], episode_index: int, *, label: str) -> str:
    episode_meta = list(payload.get("episode_meta", []))
    if not 0 <= int(episode_index) < len(episode_meta):
        raise IndexError(f"{label} episode index {episode_index} is out of range")
    item = episode_meta[int(episode_index)]
    object_id = item.get("object_id") if isinstance(item, dict) else None
    if object_id is None or not str(object_id).strip():
        raise KeyError(f"{label} episode {episode_index} is missing episode_meta.object_id")
    return str(object_id).strip()


def _object_suite_source_id(source: dict[str, np.ndarray], trajectory_index: int, *, label: str) -> str:
    object_id = _trajectory_value(source, "object_id", int(trajectory_index))
    if object_id is None or not str(object_id).strip():
        raise KeyError(f"{label} trajectory {trajectory_index} is missing object_id")
    return str(object_id).strip()


def _config_with_active_object(cfg: DictConfig, object_id: str) -> DictConfig:
    episode_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.update(episode_cfg, "objects.active_id", str(object_id), merge=False, force_add=True)
    OmegaConf.update(episode_cfg, "pump_bottle.active_id", str(object_id), merge=False, force_add=True)
    OmegaConf.resolve(episode_cfg)
    return episode_cfg


def _high_level_config_from_cfg(cfg: DictConfig, *, key: str) -> tuple[Path, DictConfig]:
    path = _required_path(OmegaConf.select(cfg, key, default=None), key)
    return path, load_root_hydra_config(path)


def _low_level_config_from_cfg(
    cfg: DictConfig,
    high_cfg: DictConfig,
    *,
    high_level_config_path: Path,
) -> Path:
    value = OmegaConf.select(cfg, "online_rollout.low_level_config_path", default=None)
    if value is not None and str(value).strip().lower() not in {"", "none", "null", "auto"}:
        path = _required_path(value, "online_rollout.low_level_config_path")
        if not path.is_file():
            raise FileNotFoundError(f"online_rollout.low_level_config_path not found: {path}")
        return path
    return _low_level_config_from_high_cfg(high_cfg, high_level_config_path=high_level_config_path)


def _low_level_config_from_high_cfg(high_cfg: DictConfig, *, high_level_config_path: Path) -> Path:
    value = OmegaConf.select(high_cfg, "preprocess.action_policy_config", default=None)
    path = _required_path(
        value,
        f"{high_level_config_path}: preprocess.action_policy_config",
    )
    if not path.is_file():
        raise FileNotFoundError(f"preprocess.action_policy_config not found: {path}")
    return path


def _low_level_checkpoint_from_cfg(
    cfg: DictConfig,
    high_cfg: DictConfig,
    *,
    high_level_config_path: Path,
) -> Path | None:
    value = OmegaConf.select(cfg, "online_rollout.low_level_checkpoint_path", default=None)
    if value is not None and str(value).strip().lower() not in {"", "none", "null", "auto"}:
        path = _required_path(value, "online_rollout.low_level_checkpoint_path")
        if not path.is_file():
            raise FileNotFoundError(f"online_rollout.low_level_checkpoint_path not found: {path}")
        return path
    return _low_level_checkpoint_from_high_cfg(high_cfg, high_level_config_path=high_level_config_path)


def _low_level_checkpoint_from_high_cfg(high_cfg: DictConfig, *, high_level_config_path: Path) -> Path | None:
    value = OmegaConf.select(high_cfg, "preprocess.latent_contact_encoder_checkpoint", default=None)
    path = _optional_path(value)
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"preprocess.latent_contact_encoder_checkpoint not found: {path}")
    return path


def _low_level_manipulator_from_rollout_or_high_cfg(
    cfg: DictConfig,
    high_cfg: DictConfig,
    *,
    env_embodiment: str,
) -> str:
    override = OmegaConf.select(cfg, "online_rollout.low_level_manipulator", default=None)
    if override is not None and str(override).strip().lower() not in {"", "none", "null", "auto"}:
        return str(override).strip()
    return _low_level_manipulator_from_high_cfg(high_cfg, env_embodiment=env_embodiment)


def _low_level_manipulator_from_high_cfg(high_cfg: DictConfig, *, env_embodiment: str) -> str:
    requested = str(OmegaConf.select(high_cfg, "preprocess.manipulator_name", default="auto")).strip()
    if requested and requested.lower() != "auto":
        return requested
    embodiment = str(env_embodiment).strip()
    if embodiment in {"marvin_wuji", "wujihand_right"}:
        return "wujihand_right"
    if embodiment in {"allegro_right", "barrett"}:
        return embodiment
    raise ValueError(
        f"Cannot infer low-level manipulator from env.embodiment={embodiment!r}; "
        "set preprocess.manipulator_name in the high-level config."
    )


def _initial_pose_from_cfg(cfg: DictConfig) -> tuple[str, Path | None, np.ndarray | None]:
    mode = str(OmegaConf.select(cfg, "online_rollout.initial_pose.mode", default="saved")).strip().lower()
    if mode not in {"saved", "retargeted"}:
        raise ValueError(f"online_rollout.initial_pose.mode must be 'saved' or 'retargeted', got {mode!r}")
    if mode == "retargeted":
        return mode, None, None
    resolution = resolve_start_pose_path(
        "wuji",
        _start_pose_task_from_cfg(cfg, section="online_rollout", default=TASK_CUBE_PICK),
        configured_pose_pt=OmegaConf.select(
            cfg,
            "online_rollout.initial_pose.pose_pt",
            default=DEFAULT_WUJI_START_POSE_RELATIVE,
        ),
        explicit_pose_pt=False,
        prefer_task_specific=bool(
            OmegaConf.select(cfg, "online_rollout.initial_pose.prefer_task_start_pose", default=True)
        ),
        default_task=TASK_CUBE_PICK,
    )
    _enforce_required_task_start_pose(cfg, section="online_rollout", resolution=resolution)
    pose_path = resolution.path
    print(f"resolved Wuji initial pose: {describe_start_pose_resolution(resolution)}", flush=True)
    return mode, pose_path, validate_wuji_q(load_wuji_start_q(pose_path), label=str(pose_path))


def _initial_robotiq_stick_pose_from_cfg(cfg: DictConfig) -> tuple[str, Path, np.ndarray]:
    mode = str(OmegaConf.select(cfg, "online_rollout.initial_pose.mode", default="saved")).strip().lower()
    if mode != "saved":
        raise ValueError(
            "robotiq_gripper_stick_armless requires online_rollout.initial_pose.mode='saved'; "
            "there is no retargeted robotiq+stick source payload for this branch"
        )
    resolution = resolve_start_pose_path(
        ROBOTIQ_STICK_ROBOT_NAME,
        _start_pose_task_from_cfg(cfg, section="online_rollout", default="pumpbottle_use"),
        configured_pose_pt=OmegaConf.select(
            cfg,
            "online_rollout.initial_pose.pose_pt",
            default=DEFAULT_ROBOTIQ_STICK_START_POSE_RELATIVE,
        ),
        explicit_pose_pt=False,
        prefer_task_specific=bool(
            OmegaConf.select(cfg, "online_rollout.initial_pose.prefer_task_start_pose", default=True)
        ),
        default_task="pumpbottle_use",
    )
    _enforce_required_task_start_pose(cfg, section="online_rollout", resolution=resolution)
    pose_path = resolution.path
    print(f"resolved Robotiq+stick initial pose: {describe_start_pose_resolution(resolution)}", flush=True)
    return mode, pose_path, validate_robotiq_stick_q(load_robotiq_stick_start_q(pose_path), label=str(pose_path))


def _robot_model_q_initial_pose_from_cfg(
    cfg: DictConfig,
    *,
    runtime: OnlineLowLevelRuntime,
    embodiment: str,
    source_path: Path | None,
) -> tuple[str, Path, np.ndarray | None]:
    mode = str(
        OmegaConf.select(cfg, "online_rollout.initial_pose.mode", default="robot_model_q_source")
    ).strip().lower()
    if mode in {"retargeted", "source", "robot_model_q_source"}:
        if source_path is None:
            raise ValueError(
                "online_rollout.robot_model_q_source_pt must be set when "
                "online_rollout.initial_pose.mode is robot_model_q_source"
            )
        return "robot_model_q_source", source_path, None
    if mode != "saved":
        raise ValueError(
            "online_rollout.initial_pose.mode for floating robot_model_q hands must be "
            f"'saved' or 'robot_model_q_source', got {mode!r}"
        )
    resolution = resolve_start_pose_path(
        embodiment,
        _start_pose_task_from_cfg(cfg, section="online_rollout", default=TASK_CUBE_PICK),
        configured_pose_pt=OmegaConf.select(cfg, "online_rollout.initial_pose.pose_pt", default=None),
        explicit_pose_pt=False,
        prefer_task_specific=bool(
            OmegaConf.select(cfg, "online_rollout.initial_pose.prefer_task_start_pose", default=True)
        ),
        default_task=TASK_CUBE_PICK,
    )
    _enforce_required_task_start_pose(cfg, section="online_rollout", resolution=resolution)
    pose_path = resolution.path
    print(f"resolved {embodiment} initial pose: {describe_start_pose_resolution(resolution)}", flush=True)
    q = _load_robot_model_start_q(pose_path, runtime=runtime, expected_robot_name=str(embodiment))
    return "saved", pose_path, q


def _enforce_required_task_start_pose(
    cfg: DictConfig,
    *,
    section: str,
    resolution: StartPoseResolution,
) -> None:
    required = bool(OmegaConf.select(cfg, f"{section}.initial_pose.require_task_start_pose", default=False))
    if not required or resolution.used_task_specific:
        return
    if resolution.task_specific_path is None:
        raise ValueError(
            f"{section}.initial_pose.require_task_start_pose=true requires a task-specific start-pose path, "
            f"but task={resolution.task!r} did not resolve one"
        )
    raise FileNotFoundError(
        f"{section}.initial_pose.require_task_start_pose=true requires task-specific start pose "
        f"{resolution.task_specific_path}; resolved fallback {resolution.path} "
        f"because {resolution.fallback_reason}"
    )


def _start_pose_task_from_cfg(cfg: DictConfig, *, section: str, default: str) -> str:
    explicit = OmegaConf.select(cfg, f"{section}.initial_pose.task", default=None)
    if explicit is not None and str(explicit).strip().lower() not in {"", "none", "null", "auto"}:
        return str(explicit)
    task_name = OmegaConf.select(cfg, "task.name", default=None)
    if task_name is not None and str(task_name).strip().lower() not in {"", "none", "null", "auto"}:
        return str(task_name)
    return str(default)


def _clamp_robotiq_stick_gripper_cmd(q: np.ndarray) -> np.ndarray:
    q_np = np.asarray(q, dtype=np.float32)
    if q_np.shape[-1] != ROBOTIQ_STICK_LOCAL_DOF:
        raise ValueError(f"robotiq_stick q must have last dim {ROBOTIQ_STICK_LOCAL_DOF}, got {q_np.shape}")
    out = q_np.astype(np.float32, copy=True)
    out[..., 6] = np.clip(out[..., 6], 0.0, 255.0)
    return out


def _proprio_source_from_cfg(cfg: DictConfig) -> str:
    value = str(OmegaConf.select(cfg, "online_rollout.proprio_source", default="executed")).strip().lower()
    if value not in {"executed", "target"}:
        raise ValueError(
            f"online_rollout.proprio_source must be 'executed' or 'target', got {value!r}"
        )
    return value


def _positive_int_from_cfg(cfg: DictConfig, key: str, *, default: int) -> int:
    value = int(OmegaConf.select(cfg, key, default=default))
    if value < 1:
        raise ValueError(f"{key} must be >= 1, got {value}")
    return value


def _gripper_execution_action_offset_xyz_from_cfg(cfg: DictConfig) -> np.ndarray:
    key = "online_rollout.gripper_execution_action_offset_xyz"
    offset_xyz = np.asarray(
        OmegaConf.select(cfg, key, default=[0.0, 0.0, 0.0]),
        dtype=np.float32,
    ).reshape(-1)
    if offset_xyz.shape != (3,) or not np.all(np.isfinite(offset_xyz)):
        raise ValueError(f"{key} must contain three finite floats, got {offset_xyz}")
    return offset_xyz.astype(np.float32, copy=True)


def _robotiq_functional_contact_mapping_from_cfg(cfg: DictConfig) -> tuple[bool, float]:
    """Read the opt-in mapping from source contact roles to Robotiq effectors."""

    section = "online_rollout.robotiq_functional_contact_mapping"
    enabled = bool(OmegaConf.select(cfg, f"{section}.enabled", default=False))
    gain = float(OmegaConf.select(cfg, f"{section}.gain", default=1.0))
    if not np.isfinite(gain) or not 0.0 <= gain <= 1.0:
        raise ValueError(f"{section}.gain must be in [0, 1], got {gain}")
    return enabled, gain


def _robotiq_gripper_min_command_from_cfg(
    cfg: DictConfig,
) -> tuple[bool, float, frozenset[int]]:
    """Read an opt-in phase-specific minimum Robotiq closure command."""

    section = "online_rollout.robotiq_gripper_min_command"
    enabled = bool(OmegaConf.select(cfg, f"{section}.enabled", default=False))
    value = float(OmegaConf.select(cfg, f"{section}.value", default=0.0))
    phase_ids = frozenset(
        int(phase_id)
        for phase_id in OmegaConf.select(cfg, f"{section}.phase_ids", default=[2, 3, 4])
    )
    if not np.isfinite(value) or not 0.0 <= value <= 255.0:
        raise ValueError(f"{section}.value must be finite and in [0, 255], got {value}")
    if enabled and not phase_ids:
        raise ValueError(f"{section}.phase_ids must be non-empty when enabled")
    return enabled, value, phase_ids


def _apply_robotiq_gripper_min_command(
    q: np.ndarray,
    *,
    phase_id: int | None,
    enabled: bool,
    value: float,
    phase_ids: frozenset[int],
) -> np.ndarray:
    """Clamp only the executed gripper command without changing policy feedback."""

    q_np = validate_robotiq_stick_q(q, label="robotiq_stick_execution_q")
    if not enabled or phase_id not in phase_ids:
        return q_np
    q_np[6] = max(float(q_np[6]), float(value))
    return validate_robotiq_stick_q(
        q_np,
        label="clamped_robotiq_stick_execution_q",
    )


def _robotiq_stick_execution_offset_from_cfg(
    cfg: DictConfig,
) -> tuple[bool, np.ndarray, frozenset[int]]:
    """Read a world-XYZ execution offset for the independent pump stick."""

    section = "online_rollout.robotiq_stick_execution_offset"
    enabled = bool(OmegaConf.select(cfg, f"{section}.enabled", default=False))
    offset_xyz = np.asarray(
        OmegaConf.select(cfg, f"{section}.offset_xyz", default=[0.0, 0.0, 0.0]),
        dtype=np.float32,
    ).reshape(-1)
    if offset_xyz.shape != (3,) or not np.all(np.isfinite(offset_xyz)):
        raise ValueError(f"{section}.offset_xyz must contain three finite floats")
    phase_ids = frozenset(
        int(phase_id)
        for phase_id in OmegaConf.select(cfg, f"{section}.phase_ids", default=[4])
    )
    if enabled and not phase_ids:
        raise ValueError(f"{section}.phase_ids must be non-empty when enabled")
    return enabled, offset_xyz.astype(np.float32, copy=True), phase_ids


def _apply_robotiq_stick_execution_offset(
    q: np.ndarray,
    *,
    phase_id: int | None,
    enabled: bool,
    offset_xyz: np.ndarray,
    phase_ids: frozenset[int],
) -> np.ndarray:
    q_np = validate_robotiq_stick_q(q, label="robotiq_stick_execution_q")
    if not enabled or phase_id not in phase_ids:
        return q_np
    q_np[7:10] += np.asarray(offset_xyz, dtype=np.float32).reshape(3)
    return validate_robotiq_stick_q(
        q_np,
        label="offset_robotiq_stick_execution_q",
    )


def _robot_model_execution_root_offset_from_cfg(
    cfg: DictConfig,
) -> tuple[bool, np.ndarray, frozenset[int]]:
    """Read an opt-in world-XYZ offset applied only to executed root targets."""

    section = "online_rollout.robot_model_execution_root_offset"
    enabled = bool(OmegaConf.select(cfg, f"{section}.enabled", default=False))
    offset_xyz = np.asarray(
        OmegaConf.select(cfg, f"{section}.offset_xyz", default=[0.0, 0.0, 0.0]),
        dtype=np.float32,
    ).reshape(-1)
    if offset_xyz.shape != (3,) or not np.all(np.isfinite(offset_xyz)):
        raise ValueError(
            f"{section}.offset_xyz must contain three finite floats, got {offset_xyz}"
        )
    phase_ids = frozenset(
        int(value)
        for value in OmegaConf.select(cfg, f"{section}.phase_ids", default=[2, 3, 4])
    )
    if enabled and not phase_ids:
        raise ValueError(f"{section}.phase_ids must be non-empty when enabled")
    return enabled, offset_xyz.astype(np.float32, copy=True), phase_ids


def _effector_execution_offset_from_cfg(
    cfg: DictConfig,
    *,
    effector_dof: int | None = None,
) -> tuple[bool, frozenset[int], np.ndarray, np.ndarray]:
    """Read opt-in offsets indexed relative to the executed effector vector."""

    section = "online_rollout.effector_execution_offset"
    enabled = bool(OmegaConf.select(cfg, f"{section}.enabled", default=False))
    phase_values = list(OmegaConf.select(cfg, f"{section}.phase_ids", default=[3, 4]))
    if any(not np.isfinite(float(value)) or float(value) != int(value) for value in phase_values):
        raise ValueError(f"{section}.phase_ids must contain finite integers, got {phase_values}")
    phase_ids = frozenset(int(value) for value in phase_values)

    index_values = list(OmegaConf.select(cfg, f"{section}.indices", default=[]))
    if any(not np.isfinite(float(value)) or float(value) != int(value) for value in index_values):
        raise ValueError(f"{section}.indices must contain finite integers, got {index_values}")
    indices = np.asarray([int(value) for value in index_values], dtype=np.int32)
    if len(set(indices.tolist())) != int(indices.size):
        raise ValueError(f"{section}.indices must be unique, got {indices.tolist()}")
    if np.any(indices < 0):
        raise ValueError(f"{section}.indices must be non-negative, got {indices.tolist()}")

    offsets = np.asarray(
        OmegaConf.select(cfg, f"{section}.offsets", default=[]),
        dtype=np.float32,
    ).reshape(-1)
    if not np.all(np.isfinite(offsets)):
        raise ValueError(f"{section}.offsets must be finite, got {offsets}")
    if indices.shape != offsets.shape:
        raise ValueError(
            f"{section}.indices and offsets must have the same length, got "
            f"{indices.size} and {offsets.size}"
        )
    if enabled and not phase_ids:
        raise ValueError(f"{section}.phase_ids must be non-empty when enabled")
    if enabled and indices.size == 0:
        raise ValueError(f"{section}.indices must be non-empty when enabled")
    if effector_dof is not None:
        dof = int(effector_dof)
        if dof <= 0:
            raise ValueError(f"effector_dof must be positive, got {dof}")
        if indices.size and int(indices.max()) >= dof:
            raise ValueError(
                f"{section}.indices must be in [0, {dof - 1}], got {indices.tolist()}"
            )
    return (
        enabled,
        phase_ids,
        indices.astype(np.int32, copy=True),
        offsets.astype(np.float32, copy=True),
    )


def _apply_effector_execution_offset(
    q_execute_world: np.ndarray,
    *,
    phase_id: int | None,
    effector_dof: int,
    enabled: bool,
    phase_ids: frozenset[int],
    indices: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Offset selected executed effector joints without changing policy feedback."""

    q_execute = np.asarray(q_execute_world, dtype=np.float32).reshape(-1).copy()
    index_array = np.asarray(indices, dtype=np.int32).reshape(-1)
    offset_array = np.asarray(offsets, dtype=np.float32).reshape(-1)
    if index_array.shape != offset_array.shape:
        raise ValueError(
            "effector execution offset indices and offsets must have matching shapes, "
            f"got {index_array.shape} and {offset_array.shape}"
        )
    if not np.all(np.isfinite(offset_array)):
        raise ValueError(f"effector execution offsets must be finite, got {offset_array}")
    if len(set(index_array.tolist())) != int(index_array.size):
        raise ValueError(f"effector execution offset indices must be unique, got {index_array.tolist()}")
    dof = int(effector_dof)
    if dof <= 0 or dof >= int(q_execute.size):
        raise ValueError(
            f"effector_dof must be in [1, {int(q_execute.size) - 1}], got {dof}"
        )
    if np.any(index_array < 0) or (index_array.size and int(index_array.max()) >= dof):
        raise ValueError(
            f"effector execution offset indices must be in [0, {dof - 1}], "
            f"got {index_array.tolist()}"
        )
    if enabled and phase_id in phase_ids:
        q_execute[int(q_execute.size) - dof + index_array] += offset_array
    return q_execute


def _apply_robot_model_execution_root_offset(
    q_execute_world: np.ndarray,
    *,
    phase_id: int | None,
    enabled: bool,
    offset_xyz: np.ndarray,
    phase_ids: frozenset[int],
) -> np.ndarray:
    """Offset executed root XYZ without changing the decoded policy target."""

    q_execute = np.asarray(q_execute_world, dtype=np.float32).reshape(-1).copy()
    if q_execute.size < 3:
        raise ValueError(
            f"robot-model execution target must contain root XYZ, got shape {q_execute.shape}"
        )
    offset = np.asarray(offset_xyz, dtype=np.float32).reshape(-1)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError(f"robot-model execution root offset must contain three finite floats, got {offset}")
    if enabled and phase_id in phase_ids:
        q_execute[:3] += offset
    return q_execute


def _apply_robotiq_gripper_execution_action_offset(
    q_target_world: np.ndarray,
    offset_xyz: np.ndarray,
) -> np.ndarray:
    offset = np.asarray(offset_xyz, dtype=np.float32).reshape(-1)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError(f"gripper execution offset must contain three finite floats, got {offset}")
    q_execute_world = np.asarray(q_target_world, dtype=np.float32).reshape(-1).copy()
    if q_execute_world.shape != (ROBOTIQ_STICK_LOCAL_DOF,):
        raise ValueError(
            f"robotiq_stick execution action must have shape ({ROBOTIQ_STICK_LOCAL_DOF},), "
            f"got {q_execute_world.shape}"
        )
    q_execute_world[0:3] += offset
    return validate_robotiq_stick_q(q_execute_world, label="q_execute_world")


def _render_images_config_from_cfg(cfg: DictConfig) -> OnlineRolloutImageRenderConfig:
    section = "online_rollout.render_images"
    enabled = bool(OmegaConf.select(cfg, f"{section}.enabled", default=False))
    default_output_dir = "./outputs/tabletop_low_level_online_rollout/render_images"
    output_dir_raw = OmegaConf.select(cfg, f"{section}.output_dir", default=default_output_dir)
    output_dir = optional_project_path(default_output_dir if output_dir_raw is None else output_dir_raw)
    if output_dir is None:
        output_dir = optional_project_path(default_output_dir)
    width = int(OmegaConf.select(cfg, f"{section}.width", default=640))
    height = int(OmegaConf.select(cfg, f"{section}.height", default=360))
    if width < 1 or height < 1:
        raise ValueError(f"{section}.width/height must be positive, got {width}x{height}")
    offset_xyz = np.asarray(
        OmegaConf.select(cfg, f"{section}.offset_xyz", default=[0.55, -0.60, 0.40]),
        dtype=np.float64,
    ).reshape(-1)
    if offset_xyz.shape != (3,) or not np.all(np.isfinite(offset_xyz)):
        raise ValueError(f"{section}.offset_xyz must contain three finite floats, got {offset_xyz}")
    fovy = float(OmegaConf.select(cfg, f"{section}.fovy", default=45.0))
    if not np.isfinite(fovy) or not (0.0 < fovy < 180.0):
        raise ValueError(f"{section}.fovy must be in (0, 180), got {fovy}")
    camera_name = str(OmegaConf.select(cfg, f"{section}.camera_name", default=ONLINE_ROLLOUT_RENDER_CAMERA_NAME))
    if not camera_name.strip():
        raise ValueError(f"{section}.camera_name must be non-empty")
    return OnlineRolloutImageRenderConfig(
        enabled=enabled,
        output_dir=Path(str(output_dir)).expanduser().resolve(),
        width=width,
        height=height,
        offset_xyz=offset_xyz.astype(np.float64, copy=True),
        fovy=fovy,
        camera_name=camera_name,
        overwrite=bool(OmegaConf.select(cfg, f"{section}.overwrite", default=True)),
    )


def _validate_render_images_runtime(
    config: OnlineRolloutImageRenderConfig,
    process_count: int,
    *,
    viewer_enabled: bool,
) -> None:
    if not config.enabled:
        return
    if viewer_enabled:
        raise ValueError(
            "online_rollout.render_images.enabled=true cannot be used with "
            "online_rollout.render=true. Disable the passive viewer for PNG export "
            "with online_rollout.render=false, or disable PNG export with "
            "online_rollout.render_images.enabled=false."
        )
    if int(process_count) > 1:
        raise ValueError(
            "online_rollout.render_images.enabled=true is not supported with "
            "parallel.num_workers/envs_per_worker > 1"
        )


def _safe_path_component(value: str) -> str:
    text = str(value).strip()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return safe or "unknown"


def _set_named_camera_pose(
    env: Any,
    *,
    camera_name: str,
    camera_pos: np.ndarray,
    target: np.ndarray,
    fovy: float,
) -> None:
    cam_id = int(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, str(camera_name)))
    if cam_id < 0:
        raise KeyError(f"Render camera {camera_name!r} is missing from the MuJoCo model")
    pos = np.asarray(camera_pos, dtype=np.float64).reshape(3)
    lookat = np.asarray(target, dtype=np.float64).reshape(3)
    env.model.cam_pos[cam_id] = pos
    env.model.cam_quat[cam_id] = np.asarray(
        look_at_quat_wxyz(tuple(float(x) for x in pos), tuple(float(x) for x in lookat)),
        dtype=np.float64,
    )
    env.model.cam_fovy[cam_id] = float(fovy)
    mujoco.mj_forward(env.model, env.data)


def _object_geometry_center(env: Any) -> np.ndarray:
    root_name = _object_root_body_name(env)
    root_bid = int(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, root_name))
    if root_bid < 0:
        raise KeyError(f"Object root body {root_name!r} is missing from the MuJoCo model")
    mujoco.mj_forward(env.model, env.data)
    mins = np.full((3,), np.inf, dtype=np.float64)
    maxs = np.full((3,), -np.inf, dtype=np.float64)
    found = False
    for geom_id in range(int(env.model.ngeom)):
        body_id = int(env.model.geom_bodyid[geom_id])
        if not _body_is_descendant(env.model, body_id, root_bid):
            continue
        geom_min, geom_max = _geom_world_bounds(env.model, env.data, int(geom_id))
        mins = np.minimum(mins, geom_min)
        maxs = np.maximum(maxs, geom_max)
        found = True
    if not found:
        raise ValueError(f"Object root body {root_name!r} does not contain any geoms")
    return ((mins + maxs) * 0.5).astype(np.float64, copy=False)


def _object_root_body_name(env: Any) -> str:
    scene = env.scene
    object_kind = str(getattr(scene, "object_kind", "cube")).strip().lower()
    if object_kind == "cube":
        return str(scene.cube_body_name)
    if object_kind == "ycb_object":
        return str(scene.object_body_name)
    if object_kind == "pump_bottle":
        return str(scene.pump_bottle_body_name)
    if object_kind == "mini_keyboard":
        return str(scene.mini_keyboard_body_name)
    raise ValueError(f"Unsupported render_images object_kind={object_kind!r}")


def _body_is_descendant(model: mujoco.MjModel, body_id: int, root_body_id: int) -> bool:
    current = int(body_id)
    while current >= 0:
        if current == int(root_body_id):
            return True
        parent = int(model.body_parentid[current])
        if parent == current:
            break
        current = parent
    return False


def _geom_world_bounds(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    geom_type = int(model.geom_type[geom_id])
    xpos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64).reshape(3)
    xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id >= 0:
            vert_adr = int(model.mesh_vertadr[mesh_id])
            vert_num = int(model.mesh_vertnum[mesh_id])
            verts = np.asarray(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=np.float64)
            if verts.size:
                world = xpos[None, :] + verts @ xmat.T
                return np.min(world, axis=0), np.max(world, axis=0)
        radius = float(model.geom_rbound[geom_id])
        return xpos - radius, xpos + radius

    size = np.asarray(model.geom_size[geom_id], dtype=np.float64).reshape(-1)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        half = size[:3]
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        half = np.asarray([size[0], size[0], size[1]], dtype=np.float64)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        half = np.full((3,), float(size[0]), dtype=np.float64)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        half = np.asarray([size[0], size[0], size[0] + size[1]], dtype=np.float64)
    elif hasattr(mujoco.mjtGeom, "mjGEOM_ELLIPSOID") and geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        half = size[:3]
    else:
        radius = float(model.geom_rbound[geom_id])
        return xpos - radius, xpos + radius
    corners = _box_corners(half)
    world = xpos[None, :] + corners @ xmat.T
    return np.min(world, axis=0), np.max(world, axis=0)


def _box_corners(half: np.ndarray) -> np.ndarray:
    hx, hy, hz = np.asarray(half, dtype=np.float64).reshape(3)
    return np.asarray(
        [
            [-hx, -hy, -hz],
            [-hx, -hy, hz],
            [-hx, hy, -hz],
            [-hx, hy, hz],
            [hx, -hy, -hz],
            [hx, -hy, hz],
            [hx, hy, -hz],
            [hx, hy, hz],
        ],
        dtype=np.float64,
    )


def _validate_contact_source(contact: dict[str, np.ndarray]) -> None:
    for key in ("contact_action", "contact_action_valid_mask"):
        if key not in contact:
            raise KeyError(f"contact source payload must contain {key!r}")
    contact_cloud = np.asarray(contact["contact_action"])
    contact_mask = np.asarray(contact["contact_action_valid_mask"])
    if contact_cloud.ndim != 3 or int(contact_cloud.shape[-1]) < 6:
        raise ValueError(f"contact_action must have shape (T,N,>=6), got {contact_cloud.shape}")
    if tuple(contact_mask.shape) != tuple(contact_cloud.shape[:2]):
        raise ValueError(
            f"contact_action_valid_mask shape {contact_mask.shape} does not match contact_action {contact_cloud.shape[:2]}"
        )


def _validate_latent_contact_source(payload: dict[str, Any], path: Path) -> None:
    data = dict(payload.get("data", {}))
    if "latent_action" not in data:
        raise KeyError(f"{path} must contain data.latent_action when online_rollout.contact_input=latent")
    latent = np.asarray(tensor_to_numpy(data["latent_action"]))
    if latent.ndim != 2 or int(latent.shape[1]) <= 0 or not np.all(np.isfinite(latent)):
        raise ValueError(f"{path} must contain finite latent_action with shape (T,D), got {latent.shape}")
    action_space = dict(payload.get("meta", {}).get("action_space", {}))
    if bool(action_space.get("latent_use_xyz_translation", False)):
        raise ValueError(
            "online_rollout.contact_input=latent currently expects latent_use_xyz_translation=false"
        )


def _validate_object_suite_contact_source(payload: dict[str, Any], path: Path) -> None:
    meta = dict(payload.get("meta", {}))
    if str(meta.get("task", "")).strip() != "object_suite_pick_place":
        raise ValueError(
            f"{path} must be an object_suite_pick_place contact source, got meta.task={meta.get('task')!r}"
        )
    action_space = meta.get("action_space", {})
    if not isinstance(action_space, dict) or str(action_space.get("type", "")).strip() != "latent_contact":
        raise ValueError(f"{path} must have meta.action_space.type='latent_contact', got {action_space!r}")
    data = dict(payload.get("data", {}))
    if "object_pose" not in data:
        raise KeyError(f"{path} must contain data.object_pose for object-suite reset")
    episode_meta = list(payload.get("episode_meta", []))
    missing = [idx for idx, item in enumerate(episode_meta) if not isinstance(item, dict) or not item.get("object_id")]
    if missing:
        raise KeyError(
            f"{path} is missing episode_meta.object_id for episodes {missing}; "
            "regenerate it with experiment/encode_contacts.py"
        )


def _validate_pump_bottle_contact_source(payload: dict[str, Any], path: Path) -> None:
    meta = dict(payload.get("meta", {}))
    if str(meta.get("task", "")).strip() != "pumpbottle_use":
        raise ValueError(f"{path} must be a pumpbottle_use contact source, got meta.task={meta.get('task')!r}")
    action_space = meta.get("action_space", {})
    if not isinstance(action_space, dict) or str(action_space.get("type", "")).strip() != "latent_contact":
        raise ValueError(f"{path} must have meta.action_space.type='latent_contact', got {action_space!r}")
    data = dict(payload.get("data", {}))
    contact_action = np.asarray(tensor_to_numpy(data.get("contact_action")))
    _validate_paper_contact_slots(payload, path)
    if "object_pose" not in data and "initial_object_pose" not in data:
        raise KeyError(f"{path} must contain data.object_pose or data.initial_object_pose for pump-bottle reset")
    if _pump_bottle_uses_episode_objects(payload):
        episode_meta = list(payload.get("episode_meta", []))
        missing = [
            idx
            for idx, item in enumerate(episode_meta)
            if not isinstance(item, dict) or not str(item.get("object_id", "")).strip()
        ]
        if missing:
            raise KeyError(f"{path} is missing episode_meta.object_id for bottle episodes {missing}")


def _validate_keyboard_contact_source(payload: dict[str, Any], path: Path) -> None:
    meta = dict(payload.get("meta", {}))
    if str(meta.get("task", "")).strip() != "keyboard_press":
        raise ValueError(f"{path} must be a keyboard_press contact source, got meta.task={meta.get('task')!r}")
    object_kind = str(meta.get("object_kind", "mini_keyboard")).strip()
    if object_kind not in {"mini_keyboard", "keyboard", "keyboard_press"}:
        raise ValueError(f"{path} must describe a mini_keyboard object, got meta.object_kind={object_kind!r}")
    action_space = meta.get("action_space", {})
    if not isinstance(action_space, dict) or str(action_space.get("type", "")).strip() != "latent_contact":
        raise ValueError(f"{path} must have meta.action_space.type='latent_contact', got {action_space!r}")
    data = dict(payload.get("data", {}))
    contact_action = np.asarray(tensor_to_numpy(data.get("contact_action")))
    _validate_paper_contact_slots(payload, path)
    if "object_pose" not in data and "initial_object_pose" not in data:
        raise KeyError(f"{path} must contain data.object_pose or data.initial_object_pose for keyboard reset")


def _validate_paper_contact_slots(payload: dict[str, Any], path: Path) -> None:
    from experiment.manipulation.sparse_contact_extraction import CONTACT_EXTRACTION_VERSION

    if payload["meta"].get("contact_extraction_version") != CONTACT_EXTRACTION_VERSION:
        raise ValueError(f"{path} requires paper trajectory contact extraction")
    data = payload["data"]
    cloud = np.asarray(tensor_to_numpy(data["contact_action"]))
    mask = np.asarray(tensor_to_numpy(data["contact_action_valid_mask"]), dtype=bool)
    if cloud.ndim != 3 or cloud.shape[1:] != (6, 6) or mask.shape != cloud.shape[:2]:
        raise ValueError(f"{path} requires six padded contact slots and a matching valid mask")
    counts = mask.sum(axis=1)
    if np.any((counts < 2) | (counts > 6)) or not np.isfinite(cloud[mask]).all():
        raise ValueError(f"{path} requires two to six finite selected contacts per frame")


def _build_contact_only_episode_specs(
    selected: list[int],
    *,
    contact_payload: dict[str, Any],
    contact_ranges: list[tuple[int, int]],
    max_frames: Any,
) -> list[EpisodeSpec]:
    specs: list[EpisodeSpec] = []
    max_frames_i = None if max_frames is None else int(max_frames)
    for contact_ep in selected:
        c_start, c_end = contact_ranges[int(contact_ep)]
        frame_count = int(c_end - c_start)
        if max_frames_i is not None and max_frames_i > 0:
            frame_count = min(frame_count, max_frames_i)
        if frame_count <= 0:
            raise RuntimeError(f"Episode {contact_ep} has no frames")
        specs.append(
            EpisodeSpec(
                contact_episode_index=int(contact_ep),
                retarget_episode_index=int(contact_ep),
                source_trajectory_index=_episode_source_index(contact_payload, int(contact_ep)),
                source_trajectory_seed=_episode_meta_int(
                    contact_payload,
                    int(contact_ep),
                    ("source_trajectory_seed", "trajectory_seed"),
                    default=-1,
                ),
                source_attempt_index=_episode_meta_int(
                    contact_payload,
                    int(contact_ep),
                    ("source_attempt_index", "attempt_index"),
                    default=-1,
                ),
                frame_count=int(frame_count),
                contact_start=int(c_start),
                retarget_start=int(c_start),
            )
        )
    return specs


def _validate_expected_frame_count(
    specs: list[EpisodeSpec],
    expected_frame_count: Any,
) -> None:
    """Reject accidentally cropped trajectories when a study fixes its horizon."""

    if expected_frame_count is None:
        return
    expected = int(expected_frame_count)
    if expected <= 0:
        raise ValueError(
            "online_rollout.expected_frame_count must be positive when configured, "
            f"got {expected}"
        )
    mismatches = [
        (int(spec.source_trajectory_index), int(spec.frame_count))
        for spec in specs
        if int(spec.frame_count) != expected
    ]
    if mismatches:
        preview = ", ".join(
            f"source {source_index}: {frame_count}"
            for source_index, frame_count in mismatches[:8]
        )
        suffix = "" if len(mismatches) <= 8 else f", ... ({len(mismatches)} total)"
        raise ValueError(
            f"Expected every selected trajectory to have {expected} frames, got "
            f"{preview}{suffix}"
        )


def _validate_completed_frame_count(result: Any, expected_frame_count: Any, *, controller_steps_per_prediction: int = 1) -> None:
    """Prevent a stopped viewer from being saved as a complete evaluation."""

    if expected_frame_count is None or str(getattr(result, "error", "")):
        return
    expected = int(expected_frame_count) * int(controller_steps_per_prediction)
    completed = int(getattr(result, "completed_steps"))
    if completed != expected:
        raise RuntimeError(
            f"Trajectory {int(getattr(result, 'source_index'))} completed {completed}/{expected} "
            "controller steps. A fixed-horizon evaluation cannot save a partial viewer rollout."
        )


def _print_ordered_pump_summary(results: list[PumpBottleRolloutResult]) -> None:
    if not results:
        return
    ordered = np.asarray(
        [bool(result.ordered_success) for result in results], dtype=np.bool_
    )
    print(
        f"ordered_lift_then_press={int(np.sum(ordered))}/{int(ordered.size)} "
        f"({100.0 * float(np.mean(ordered)):.1f}%) "
        "definition=legacy_success_and_lift_at_first_threshold_press",
        flush=True,
    )


def _print_ordered_pump_payload_summary(payload: dict[str, object]) -> None:
    if "ordered_success" not in payload:
        return
    ordered = np.asarray(payload["ordered_success"], dtype=np.bool_).reshape(-1)
    print(
        f"ordered_lift_then_press={int(np.sum(ordered))}/{int(ordered.size)} "
        f"({100.0 * float(np.mean(ordered)):.1f}%) "
        "definition=legacy_success_and_lift_at_first_threshold_press",
        flush=True,
    )


def _validate_robot_model_q_source(source: dict[str, np.ndarray], *, embodiment: str, source_path: Path) -> None:
    action_space_type = _source_scalar(source, "action_space_type")
    if str(action_space_type).strip() != "robot_model_q":
        raise ValueError(f"{source_path} must have action_space_type='robot_model_q', got {action_space_type!r}")
    source_embodiment = _source_scalar(source, "embodiment")
    if source_embodiment is not None and str(source_embodiment).strip() != str(embodiment):
        raise ValueError(
            f"{source_path} embodiment={source_embodiment!r} does not match env.embodiment={embodiment!r}"
        )
    source_mode = _source_scalar(source, "mode")
    if source_mode is not None and str(source_mode).strip() not in {"floating_right", "single_right"}:
        raise ValueError(f"{source_path} mode={source_mode!r} is not a supported floating right-hand source")


def _validate_object_reset_source(source: dict[str, np.ndarray], *, source_path: Path) -> None:
    action_space_type = str(_source_scalar(source, "action_space_type") or "").strip()
    if action_space_type != "single_manipulator":
        raise ValueError(
            f"{source_path} must have action_space_type='single_manipulator' for object-only reset, "
            f"got {action_space_type!r}"
        )
    if "object_pose" not in source and "cube_pose" not in source:
        raise KeyError(f"{source_path} must contain object_pose or cube_pose for object-only reset")


def _validate_object_reset_source_specs(
    source: dict[str, np.ndarray],
    specs: list[EpisodeSpec],
    *,
    source_count: int,
) -> None:
    pose_key = "object_pose" if "object_pose" in source else "cube_pose"
    for spec in specs:
        source_idx = int(spec.source_trajectory_index)
        if source_idx < 0 or source_idx >= int(source_count):
            raise IndexError(
                f"source_trajectory_index={source_idx} is out of range for "
                f"object_reset_source trajectory_count={source_count}"
            )
        if _trajectory_array(source, pose_key, source_idx, required=False) is None:
            raise KeyError(f"object_reset_source trajectory {source_idx} is missing {pose_key}")


def _validate_robot_model_source_specs(
    source: dict[str, np.ndarray],
    specs: list[EpisodeSpec],
    *,
    source_count: int,
    expected_dim: int,
) -> None:
    for spec in specs:
        source_idx = int(spec.source_trajectory_index)
        if source_idx < 0 or source_idx >= int(source_count):
            raise IndexError(
                f"source_trajectory_index={source_idx} is out of range for "
                f"robot_model_q_source trajectory_count={source_count}"
            )
        frame_count = _source_trajectory_frame_count(source, source_idx)
        if int(spec.frame_count) > int(frame_count):
            raise ValueError(
                f"contact episode {spec.contact_episode_index} needs {spec.frame_count} frames, "
                f"but robot_model_q source trajectory {source_idx} has {frame_count}"
            )
        q0 = _initial_robot_model_q_from_source(source, spec, expected_dim=expected_dim)
        if q0.shape != (int(expected_dim),):
            raise ValueError(f"robot_model_q source trajectory {source_idx} q0 has shape {q0.shape}")


def _validate_robot_model_source_env_dim(
    source: dict[str, np.ndarray],
    specs: list[EpisodeSpec],
    env: TabletopManipulationEnv,
) -> None:
    if not env.robot.has_group("right"):
        raise ValueError("floating robot_model_q online rollout requires a right control group")
    right = env.robot.right
    expected_dim = int(right.arm_dim + right.effector_ctrl_dim)
    for spec in specs:
        q0 = _initial_robot_model_q_from_source(source, spec, expected_dim=expected_dim)
        if q0.shape != (expected_dim,):
            raise ValueError(
                f"robot_model_q source trajectory {spec.source_trajectory_index} has dim {q0.shape[0]}, "
                f"expected env dim {expected_dim}"
            )


def _source_trajectory_frame_count(source: dict[str, np.ndarray], source_index: int) -> int:
    for key in ("manipulator_action", "action_ee_pos", "action_arm_q"):
        arr = _trajectory_array(source, key, int(source_index), required=False)
        if arr is not None:
            return int(np.asarray(arr).shape[0])
    raise KeyError("robot_model_q source must contain manipulator_action, action_ee_pos, or action_arm_q")


def _episode_source_index(payload: dict[str, Any], episode_idx: int) -> int:
    item = dict(list(payload.get("episode_meta", []))[int(episode_idx)])
    for key in ("source_trajectory_index", "trajectory_index", "episode_index"):
        if item.get(key) is not None:
            return int(item[key])
    return int(episode_idx)


def _episode_meta_int(payload: dict[str, Any], episode_idx: int, keys: tuple[str, ...], default: int) -> int:
    item = dict(list(payload.get("episode_meta", []))[int(episode_idx)])
    for key in keys:
        if item.get(key) is not None:
            return int(item[key])
    return int(default)


def _selected_indices_from_cfg(
    episode_count: int,
    *,
    indices: Any,
    start_index: int,
    count: Any,
) -> list[int]:
    if indices is not None and not (
        isinstance(indices, str) and indices.strip().lower() in {"", "none", "null"}
    ):
        if isinstance(indices, ListConfig | list | tuple):
            selected = [int(idx) for idx in list(indices)]
        else:
            selected = []
            for part in str(indices).split(","):
                part = part.strip()
                if not part:
                    continue
                if ":" in part:
                    pieces = [int(x) if x else None for x in part.split(":")]
                    if len(pieces) > 3:
                        raise ValueError(f"Invalid slice {part!r}")
                    selected.extend(range(*slice(*pieces).indices(int(episode_count))))
                else:
                    selected.append(int(part))
    else:
        if int(start_index) < 0:
            raise ValueError("online_rollout.start_index must be non-negative")
        stop = int(episode_count) if count is None else min(int(episode_count), int(start_index) + int(count))
        selected = list(range(int(start_index), stop))

    invalid = [idx for idx in selected if idx < 0 or idx >= int(episode_count)]
    if invalid:
        raise IndexError(f"Episode indices out of range for {episode_count} episodes: {invalid}")
    return list(dict.fromkeys(int(idx) for idx in selected))


def _configure_camera_for_runtime(
    cfg: DictConfig,
    runtime: OnlineLowLevelRuntime,
    *,
    force_camera_obs: bool = False,
) -> None:
    if runtime.non_contact_enabled:
        cfg.perception.include_camera_obs = True
        cfg.perception.camera_names = ["ext_cam_0", "ext_cam_1"]
        cfg.perception.policy_camera_names = ["ext_cam_0", "ext_cam_1"]
        cfg.perception.capture_segmentation = True
        cfg.perception.capture_rgb = False
        return
    if bool(force_camera_obs):
        cfg.perception.include_camera_obs = True
        cfg.perception.capture_segmentation = True
        cfg.perception.capture_rgb = False
        return
    if bool(OmegaConf.select(cfg, "online_rollout.non_contact.force_camera_when_enabled", default=True)):
        cfg.perception.include_camera_obs = False
        cfg.perception.capture_segmentation = False
        cfg.perception.capture_rgb = False


def _load_online_low_level_runtime(
    *,
    low_level_config_path: Path,
    low_level_checkpoint_path: Path | None,
    low_level_manipulator: str,
    low_level_use_ema: bool,
    device: str,
) -> OnlineLowLevelRuntime:
    device_name = "cuda" if str(device).lower() == "auto" else str(device)
    device_t = prepare_device(device_name)
    low_cfg = load_root_hydra_config(low_level_config_path)
    resolved_ckpt = (
        (Path(low_cfg.training.save_dir) / "last.pt").expanduser().resolve()
        if low_level_checkpoint_path is None
        else Path(low_level_checkpoint_path).expanduser().resolve()
    )
    cotrain_meta = _peek_cotrain_meta(resolved_ckpt)
    if cotrain_meta is not None:
        low_policy, ckpt_resolved, low_meta, low_cfg_resolved = _load_cotrain_low_level_policy(
            low_cfg,
            checkpoint_path=str(resolved_ckpt),
            device=device_t,
            prefer_ema=bool(low_level_use_ema),
            selected_manipulator=str(low_level_manipulator),
            cotrain_meta=cotrain_meta,
        )
    else:
        low_policy, ckpt_resolved, low_meta, low_cfg_resolved = load_contact_policy_probe_checkpoint(
            low_cfg,
            device=device_t,
            ckpt_path=str(resolved_ckpt),
            prefer_ema=bool(low_level_use_ema),
            return_build_cfg=True,
        )

    encoder_backend = _resolve_low_level_backend(low_cfg_resolved, low_policy)
    contact_feature_dim = (
        int(low_cfg_resolved.model.contact_encoder.out_channels)
        if encoder_backend == "pointnet"
        else int(getattr(low_policy, "d_model"))
    )
    robot_kwargs = dict(low_meta.get("robot_model_kwargs", {}))
    robot_name = str(
        low_meta.get(
            "robot_name",
            low_level_manipulator if cotrain_meta is not None else low_cfg_resolved.probe.robot_name,
        )
    )
    robot_model = create_robot_model(
        robot_name=robot_name,
        device=device_t,
        num_points=int(low_meta.get("surface_num_points", 256)),
        **robot_kwargs,
    )
    local_dof = int(low_meta.get("action_dim", getattr(robot_model, "dof", 0)))
    action_valid_mask = torch.ones((local_dof,), dtype=torch.bool, device=device_t)
    if hasattr(robot_model, "fixed_zero_joint_indices") and len(robot_model.fixed_zero_joint_indices) > 0:
        action_valid_mask[robot_model.fixed_zero_joint_indices] = False

    hand_fps_num_points = 0
    if encoder_backend == "paired_transformer":
        hand_fps_num_points = int(low_meta.get("hand_points_fps", 0))
        if hand_fps_num_points <= 0:
            hand_fps_num_points = int(getattr(low_cfg_resolved.model, "hand_points", 32))
        robot_model.build_hand_fps_template(int(hand_fps_num_points))

    non_contact_input_points = 0
    non_contact_sampling_radius = 0.0
    non_contact_sdf_clearance = 0.0
    cfg_uses_non_contact = False
    if OmegaConf.select(low_cfg_resolved, "probe", default=None) is not None:
        remove_nc = bool(getattr(low_cfg_resolved.probe, "remove_non_contact_points", False))
        non_contact_input_points = int(getattr(low_cfg_resolved.probe, "non_contact_input_points", 0))
        non_contact_sampling_radius = float(getattr(low_cfg_resolved.probe, "non_contact_sampling_radius", 0.10))
        non_contact_sdf_clearance = float(getattr(low_cfg_resolved.probe, "non_contact_sdf_clearance", 0.002))
        cfg_uses_non_contact = (not remove_nc) and non_contact_input_points > 0
    model_uses_non_contact = _policy_uses_non_contact(low_policy)
    non_contact_enabled = bool(cfg_uses_non_contact and model_uses_non_contact)
    if cfg_uses_non_contact and not model_uses_non_contact:
        print(
            "WARNING: low-level config requests non-contact points, but the loaded policy does not expose "
            "a non-contact stream; running without non-contact inputs.",
            flush=True,
        )
    if model_uses_non_contact and not cfg_uses_non_contact:
        print(
            "WARNING: loaded policy exposes a non-contact stream, but low-level config disables it; "
            "running without non-contact inputs.",
            flush=True,
        )

    collision_model = None
    collision_spec_path = None
    if non_contact_enabled:
        spec_dir = _optional_path(getattr(low_cfg_resolved.probe, "collision_primitive_spec_dir", "./assets/collision_primitives"))
        if spec_dir is None:
            spec_dir = ROOT / "assets" / "collision_primitives"
        collision_spec_path = _resolve_collision_primitive_spec_path(
            spec_dir,
            robot_name=robot_name,
            low_level_manipulator=low_level_manipulator,
            low_meta=low_meta,
        )
        spec = load_collision_primitive_spec(collision_spec_path)
        if str(spec.get("model_family", "")) == "mano":
            collision_model = ManoPrimitiveCollisionModel(robot_model, spec)
        else:
            collision_model = PrimitiveCollisionModel(robot_model, spec)

    low_policy.eval()
    return OnlineLowLevelRuntime(
        device=device_t,
        low_level_cfg=low_cfg_resolved,
        low_level_policy=low_policy,
        low_level_config_path=str(Path(low_level_config_path).expanduser().resolve()),
        low_level_checkpoint_path=str(Path(ckpt_resolved).expanduser().resolve()),
        low_level_meta=dict(low_meta),
        robot_model=robot_model,
        local_dof=local_dof,
        action_valid_mask=action_valid_mask,
        robot_index=0,
        encoder_backend=encoder_backend,
        contact_feature_dim=contact_feature_dim,
        low_level_manipulator=robot_name,
        hand_fps_num_points=int(hand_fps_num_points),
        non_contact_enabled=bool(non_contact_enabled),
        non_contact_input_points=int(non_contact_input_points if non_contact_enabled else 0),
        non_contact_sampling_radius=float(non_contact_sampling_radius),
        non_contact_sdf_clearance=float(non_contact_sdf_clearance),
        collision_model=collision_model,
        collision_primitive_spec_path=None if collision_spec_path is None else str(collision_spec_path),
    )


def _policy_uses_non_contact(policy: torch.nn.Module) -> bool:
    if bool(getattr(policy, "use_non_contact_points", False)):
        return True
    if getattr(policy, "non_contact_context_encoder", None) is not None:
        return True
    heads = getattr(policy, "heads", None)
    if isinstance(heads, dict):
        return any(_policy_uses_non_contact(head) for head in heads.values())
    return False


def _resolve_collision_primitive_spec_path(
    spec_dir: Path,
    *,
    robot_name: str,
    low_level_manipulator: str,
    low_meta: dict[str, Any],
) -> Path:
    candidates: list[Path] = []
    for name in (
        low_meta.get("robot_model_name"),
        low_meta.get("robot_name"),
        robot_name,
        low_level_manipulator,
    ):
        if name is None:
            continue
        name_s = str(name)
        mano_name = _is_mano_robot_name(name_s)
        if mano_name is not None:
            side = "right" if str(mano_name).endswith("right") else "left"
            candidates.append(mano_collision_primitive_spec_path(spec_dir, side=side))
        candidates.append(collision_primitive_spec_path(spec_dir, name_s))
        try:
            candidates.append(collision_primitive_spec_path(spec_dir, resolve_robot_model_name(name_s)))
        except Exception:
            pass

    seen: set[str] = set()
    deduped = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    for path in deduped:
        if path.exists():
            return path
    searched = ", ".join(str(path) for path in deduped)
    raise FileNotFoundError(f"Could not find collision primitive sidecar for non-contact inference; searched: {searched}")


def _rollout_episode(
    *,
    env: TabletopManipulationEnv,
    cfg: DictConfig,
    spec: EpisodeSpec,
    contact: dict[str, np.ndarray],
    reset_source: dict[str, np.ndarray],
    reset_source_kind: str,
    runtime: OnlineLowLevelRuntime,
    contact_input_mode: str,
    dataset_shift: np.ndarray,
    initial_wuji_q: np.ndarray | None,
    proprio_source: str,
    controller_steps_per_prediction: int,
    store_trace: bool,
    trace: dict[str, list[np.ndarray]] | None,
    viewer_control: ViewerReplayControl | None,
    dt: float,
    seed: int,
    capture_camera_every_controller_step: bool,
    training_episode: OnlineTrainingEpisode | None = None,
    image_renderer: OnlineRolloutImageRenderer | None = None,
    profiler: RolloutProfiler | None = None,
) -> RolloutResult:
    profiler = profiler or RolloutProfiler(enabled=False)
    with profiler.measure("reset"):
        if reset_source_kind == "object_only":
            obs, source_seed = _reset_from_robot_model_source(env, reset_source, spec)
            if initial_wuji_q is None:
                raise ValueError("Wuji object-only reset requires a saved initial pose")
            initial_q = validate_wuji_q(initial_wuji_q, label="initial_wuji_q")
        elif reset_source_kind == "retargeted_wuji":
            obs, source_seed = _reset_from_retarget(env, reset_source, spec)
            initial_q = (
                validate_wuji_q(initial_wuji_q, label="initial_wuji_q")
                if initial_wuji_q is not None
                else _initial_wuji_q_from_retargeted(reset_source, spec)
            )
        else:
            raise ValueError(f"Unsupported Wuji reset_source_kind={reset_source_kind!r}")
        _set_env_wuji_q(env, initial_q)
        obs = env.observe()
        render_step_offset = 0
        if image_renderer is not None:
            image_renderer.start_trajectory(env, spec)
            with profiler.measure("render_image"):
                image_renderer.render_step(env, step_idx=0)
            render_step_offset = 1
        if training_episode is not None:
            training_episode.source_seed = source_seed

    current_q_world = _current_wuji_q_from_env(env)
    target_feedback_q_world = np.asarray(initial_q, dtype=np.float32).reshape(-1).copy()
    final_cube_pose = _object_pose_from_obs(obs)
    max_cube_z = float(final_cube_pose[2])
    success = bool(obs["success"])
    completed_steps = 0
    nc_counts: list[int] = []
    right_arm_actuator_ids = np.asarray(
        env.robot.side("right").arm_actuator_ids,
        dtype=np.int32,
    )
    default_arm_kp = np.asarray(
        env.model.actuator_gainprm[right_arm_actuator_ids, 0],
        dtype=np.float64,
    ).copy()
    default_arm_kv = np.asarray(
        -env.model.actuator_biasprm[right_arm_actuator_ids, 2],
        dtype=np.float64,
    ).copy()
    right_arm_joint_ids = np.unique(
        np.asarray(env.model.actuator_trnid[right_arm_actuator_ids, 0], dtype=np.int32)
    )
    right_arm_joint_ids = right_arm_joint_ids[right_arm_joint_ids >= 0]
    default_arm_joint_force_range = np.asarray(
        env.model.jnt_actfrcrange[right_arm_joint_ids],
        dtype=np.float64,
    ).copy()
    right_hand_actuator_ids = np.asarray(
        env.robot.side("right").hand_actuator_ids,
        dtype=np.int32,
    )
    default_hand_kp = np.asarray(
        env.model.actuator_gainprm[right_hand_actuator_ids, 0],
        dtype=np.float64,
    ).copy()
    default_hand_kv = np.asarray(
        -env.model.actuator_biasprm[right_hand_actuator_ids, 2],
        dtype=np.float64,
    ).copy()

    for step_idx in range(int(spec.frame_count)):
        if viewer_control is not None:
            if not viewer_control.wait_before_step(
                env,
                step_idx=step_idx,
                total_steps=int(spec.frame_count),
                trajectory_index=int(spec.source_trajectory_index),
            ):
                break

        contact_frame, contact_mask = _contact_frame(contact, spec, step_idx)
        q_feedback_world = (
            target_feedback_q_world if str(proprio_source) == "target" else current_q_world
        )
        q_prev_world = q_feedback_world.reshape(1, -1).astype(np.float32, copy=True)
        q_prev_shifted = _shift_low_level_q(q_prev_world, dataset_shift, sign=1.0)
        non_contact_cloud = None
        non_contact_valid = None
        if runtime.non_contact_enabled:
            with profiler.measure("non_contact_input"):
                non_contact_cloud, non_contact_valid, nc_stats = _build_non_contact_input(
                    cfg=cfg,
                    env=env,
                    runtime=runtime,
                    obs=obs,
                    q_prev_shifted=q_prev_shifted[0],
                    contact_cloud_shifted=contact_frame,
                    contact_valid_mask=contact_mask,
                    dataset_shift=dataset_shift,
                    seed=int(seed) + step_idx,
                )
            nc_counts.append(int(nc_stats.selected_count))

        with profiler.measure("low_level_policy"):
            if contact_input_mode == "latent":
                q_pred_shifted = _predict_low_level_q_from_saved_latent(
                    runtime=runtime,
                    latent_action=_latent_contact_frame(contact, spec, step_idx),
                    q_prev_shifted=q_prev_shifted,
                    seed=int(seed) + step_idx,
                )
            else:
                q_pred_shifted = _predict_low_level_q_batch(
                    runtime=runtime,
                    contact_cloud_shifted=contact_frame.reshape(1, *contact_frame.shape),
                    contact_valid_mask=contact_mask.reshape(1, *contact_mask.shape),
                    q_prev_shifted=q_prev_shifted,
                    seed=int(seed) + step_idx,
                    non_contact_cloud_shifted=None if non_contact_cloud is None else non_contact_cloud.reshape(1, *non_contact_cloud.shape),
                    non_contact_valid_mask=None if non_contact_valid is None else non_contact_valid.reshape(1, *non_contact_valid.shape),
                )
        q_pred_world = _shift_low_level_q(q_pred_shifted, dataset_shift, sign=-1.0)
        q_pred_world = _apply_fixed_zero_constraints(runtime, q_pred_world)
        q_target_world = q_pred_world[0].astype(np.float32, copy=True)
        phase_value = _contact_source_frame_or_none(contact, spec, step_idx, "phase")
        phase_id = None if phase_value is None else int(np.asarray(phase_value).reshape(()))
        _apply_phase_arm_gains(
            env,
            cfg=cfg,
            phase_id=phase_id,
            default_kp=default_arm_kp,
            default_kv=default_arm_kv,
            joint_ids=right_arm_joint_ids,
            default_joint_force_range=default_arm_joint_force_range,
        )
        _apply_phase_hand_gains(
            env,
            cfg=cfg,
            phase_id=phase_id,
            default_kp=default_hand_kp,
            default_kv=default_hand_kv,
        )
        target_feedback_q_world = q_target_world.copy()
        action = _env_action_from_wuji_q(q_target_world)
        training_obs = _training_observation(env, obs) if training_episode is not None else None
        last_obs: dict[str, Any] | None = None
        last_info: dict[str, Any] | None = None
        for repeat_idx in range(int(controller_steps_per_prediction)):
            capture_camera = bool(
                (
                    runtime.non_contact_enabled
                    and (
                        capture_camera_every_controller_step
                        or store_trace
                        or repeat_idx + 1 >= int(controller_steps_per_prediction)
                    )
                )
                or (
                    training_episode is not None
                    and repeat_idx + 1 >= int(controller_steps_per_prediction)
                )
            )
            with profiler.measure("env_step_with_camera" if capture_camera else "env_step_state_only"):
                obs, _, _, _, info = env.step(action, include_camera_obs=capture_camera)
            last_obs = obs
            last_info = info
            completed_steps += 1
            current_q_world = _current_wuji_q_from_env(env)
            final_cube_pose = _object_pose_from_obs(obs)
            max_cube_z = max(max_cube_z, float(final_cube_pose[2]))
            success = accumulate_task_success(
                success,
                env.task,
                obs,
                max_object_z=max_cube_z,
            )

            if store_trace and trace is not None:
                append_trace_frame(trace, env, info, obs=obs)
                _append_online_low_level_trace_fields(
                    trace,
                    policy_step=step_idx,
                    repeat_idx=repeat_idx,
                    q_input_world=q_prev_world[0],
                    q_target_world=q_target_world,
                    proprio_source=proprio_source,
                    controller_steps_per_prediction=controller_steps_per_prediction,
                )
            if viewer_control is not None:
                viewer_control.update_overlay(
                    env,
                    step_idx=step_idx + 1,
                    total_steps=int(spec.frame_count),
                    trajectory_index=int(spec.source_trajectory_index),
                )
                if viewer_control.stop_requested or not env.viewer_is_running():
                    break
                time.sleep(float(dt))
        if training_episode is not None and last_obs is not None and last_info is not None:
            _append_online_training_frame(
                training_episode,
                obs=training_obs if training_obs is not None else last_obs,
                info=last_info,
                contact=contact,
                spec=spec,
                step_idx=step_idx,
                dataset_shift=dataset_shift,
                q_input_world=q_prev_world[0],
                q_input_shifted=q_prev_shifted[0],
                q_action_shifted=q_pred_shifted[0],
                q_action_world=q_target_world,
                q_executed_world=current_q_world,
            )
        if image_renderer is not None and last_obs is not None:
            with profiler.measure("render_image"):
                image_renderer.render_step(env, step_idx=step_idx + render_step_offset)
        if viewer_control is not None and (viewer_control.stop_requested or not env.viewer_is_running()):
            break

    if nc_counts:
        counts = np.asarray(nc_counts, dtype=np.float32)
        print(
            f"trajectory={spec.source_trajectory_index} non_contact_selected="
            f"min={int(np.min(counts))} mean={float(np.mean(counts)):.1f} max={int(np.max(counts))}",
            flush=True,
        )

    profiler.print_summary()
    return RolloutResult(
        source_index=int(spec.source_trajectory_index),
        source_seed=source_seed,
        action_frame_count=int(spec.frame_count) * int(controller_steps_per_prediction),
        completed_steps=int(completed_steps),
        success=bool(success),
        max_cube_z=float(max_cube_z),
        final_cube_pose=final_cube_pose,
        error="",
    )


def _object_pose_from_obs(obs: dict[str, Any]) -> np.ndarray:
    for key in ("cube_pose", "object_pose"):
        if key in obs:
            return np.asarray(obs[key], dtype=np.float32).reshape(7).copy()
    return np.full((7,), np.nan, dtype=np.float32)


def _rollout_episode_robot_model_hand(
    *,
    env: TabletopManipulationEnv,
    cfg: DictConfig,
    spec: EpisodeSpec,
    contact: dict[str, np.ndarray],
    source: dict[str, np.ndarray] | None,
    runtime: OnlineLowLevelRuntime,
    contact_input_mode: str,
    dataset_shift: np.ndarray,
    translation_groups: tuple[tuple[int, int, int], ...],
    proprio_source: str,
    controller_steps_per_prediction: int,
    store_trace: bool,
    trace: dict[str, list[np.ndarray]] | None,
    viewer_control: ViewerReplayControl | None,
    dt: float,
    seed: int,
    capture_camera_every_controller_step: bool,
    pump_bottle_lift_success_metric: str = "max_lift_delta",
    print_button_slide_each_step: bool = False,
    initial_q_override: np.ndarray | None = None,
    training_episode: OnlineTrainingEpisode | None = None,
    image_renderer: OnlineRolloutImageRenderer | None = None,
    profiler: RolloutProfiler | None = None,
) -> PumpBottleRolloutResult:
    profiler = profiler or RolloutProfiler(enabled=False)
    task_name = str(OmegaConf.select(cfg, "task.name", default="cube_pick")).strip()
    with profiler.measure("reset"):
        if task_name in {"keyboard_press", "pumpbottle_use"}:
            obs, source_seed = _reset_from_contact_task(env, contact, spec, cfg=cfg)
        else:
            if source is None:
                raise ValueError("robot_model_q source is required for floating cube-pick reset")
            obs, source_seed = _reset_from_robot_model_source(env, source, spec)
        if initial_q_override is not None:
            initial_q = _validate_robot_model_q(
                initial_q_override,
                expected_dim=runtime.local_dof,
                label="saved initial robot_model_q",
            )
        else:
            if source is None:
                raise ValueError(
                    "robot_model_q source is required when no saved initial robot_model_q is configured"
                )
            initial_q = _initial_robot_model_q_from_source(source, spec, expected_dim=runtime.local_dof)
        _set_env_robot_model_q(env, initial_q)
        obs = env.observe()
        render_step_offset = 0
        if image_renderer is not None:
            image_renderer.start_trajectory(env, spec)
            with profiler.measure("render_image"):
                image_renderer.render_step(env, step_idx=0)
            render_step_offset = 1
        if training_episode is not None:
            training_episode.source_seed = source_seed

    current_q_world = _current_robot_model_q_from_env(env)
    target_feedback_q_world = np.asarray(initial_q, dtype=np.float32).reshape(-1).copy()
    final_object_pose = _object_pose_from_obs(obs)
    max_object_z = float(final_object_pose[2])
    max_lift_delta = (
        float(np.asarray(obs["lift_delta"], dtype=np.float32))
        if "lift_delta" in obs
        else float("nan")
    )
    final_lift_delta = max_lift_delta
    initial_min_geom_z = (
        _pump_bottle_min_geom_z(env)
        if task_name == "pumpbottle_use"
        else float("nan")
    )
    final_min_geom_z = initial_min_geom_z
    max_lift_clearance_delta = 0.0 if np.isfinite(initial_min_geom_z) else float("nan")
    final_lift_clearance_delta = max_lift_clearance_delta
    min_button_slide = (
        float(np.asarray(obs["button_slide"], dtype=np.float32))
        if "button_slide" in obs
        else float("nan")
    )
    final_button_slide = min_button_slide
    lift_success = bool(obs.get("lift_success", False))
    button_success = bool(obs.get("button_success", False))
    button_slide_history: list[float] = []
    lift_clearance_history: list[float] = []
    first_button_success_frame = -1
    first_lift_success_frame = -1
    lift_clearance_at_first_button_success = float("nan")
    button_after_lift_success = False
    ordered_success = False
    target_key_success = np.asarray(
        obs.get("target_key_success", np.zeros((0,), dtype=np.bool_)),
        dtype=np.bool_,
    ).reshape(-1).copy()
    final_target_key_slide = np.asarray(
        obs.get("target_key_slide", np.full((target_key_success.shape[0],), np.nan, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1).copy()
    min_target_key_slide = final_target_key_slide.copy()
    success = bool(obs["success"])
    completed_steps = 0
    nc_counts: list[int] = []
    grasp_hold_enabled, grasp_hold_capture_phase_ids, grasp_hold_apply_phase_ids = (
        _grasp_hold_config_from_cfg(cfg)
    )
    (
        execution_root_offset_enabled,
        execution_root_offset_xyz,
        execution_root_offset_phase_ids,
    ) = _robot_model_execution_root_offset_from_cfg(cfg)
    effector_dof = int(env.robot.side("right").effector_ctrl_dim)
    (
        effector_execution_offset_enabled,
        effector_execution_offset_phase_ids,
        effector_execution_offset_indices,
        effector_execution_offset_values,
    ) = _effector_execution_offset_from_cfg(cfg, effector_dof=effector_dof)
    grasp_hold_q: np.ndarray | None = None

    for step_idx in range(int(spec.frame_count)):
        if viewer_control is not None:
            if not viewer_control.wait_before_step(
                env,
                step_idx=step_idx,
                total_steps=int(spec.frame_count),
                trajectory_index=int(spec.source_trajectory_index),
            ):
                break

        contact_frame, contact_mask = _contact_frame(contact, spec, step_idx)
        q_feedback_world = (
            target_feedback_q_world if str(proprio_source) == "target" else current_q_world
        )
        q_prev_world = q_feedback_world.reshape(1, -1).astype(np.float32, copy=True)
        q_prev_shifted = _shift_composite_low_level_q(
            q_prev_world,
            dataset_shift,
            translation_groups=translation_groups,
            sign=1.0,
        )
        non_contact_cloud = None
        non_contact_valid = None
        if runtime.non_contact_enabled:
            with profiler.measure("non_contact_input"):
                non_contact_cloud, non_contact_valid, nc_stats = _build_non_contact_input(
                    cfg=cfg,
                    env=env,
                    runtime=runtime,
                    obs=obs,
                    q_prev_shifted=q_prev_shifted[0],
                    contact_cloud_shifted=contact_frame,
                    contact_valid_mask=contact_mask,
                    dataset_shift=dataset_shift,
                    seed=int(seed) + step_idx,
                )
            nc_counts.append(int(nc_stats.selected_count))

        with profiler.measure("low_level_policy"):
            if contact_input_mode == "latent":
                q_pred_shifted = _predict_low_level_q_from_saved_latent(
                    runtime=runtime,
                    latent_action=_latent_contact_frame(contact, spec, step_idx),
                    q_prev_shifted=q_prev_shifted,
                    seed=int(seed) + step_idx,
                )
            else:
                q_pred_shifted = _predict_low_level_q_batch(
                    runtime=runtime,
                    contact_cloud_shifted=contact_frame.reshape(1, *contact_frame.shape),
                    contact_valid_mask=contact_mask.reshape(1, *contact_mask.shape),
                    q_prev_shifted=q_prev_shifted,
                    seed=int(seed) + step_idx,
                    non_contact_cloud_shifted=None if non_contact_cloud is None else non_contact_cloud.reshape(1, *non_contact_cloud.shape),
                    non_contact_valid_mask=None if non_contact_valid is None else non_contact_valid.reshape(1, *non_contact_valid.shape),
                )
        q_pred_world = _shift_composite_low_level_q(
            q_pred_shifted,
            dataset_shift,
            translation_groups=translation_groups,
            sign=-1.0,
        )
        q_pred_world = _apply_fixed_zero_constraints(runtime, q_pred_world)
        q_target_world = _validate_robot_model_q(q_pred_world[0], expected_dim=runtime.local_dof, label="q_target_world")
        q_target_world = _unwrap_robot_model_q_rotation(q_target_world, reference=q_feedback_world)
        target_feedback_q_world = q_target_world.copy()
        phase_value = _contact_source_frame_or_none(contact, spec, step_idx, "phase")
        phase_id = None if phase_value is None else int(np.asarray(phase_value).reshape(()))
        _apply_phase_finger_gains(env, cfg=cfg, phase_id=phase_id)
        _apply_phase_finger_effort(env, cfg=cfg, phase_id=phase_id)
        phase_repeat_count = _phase_controller_steps_per_prediction(
            cfg,
            phase_id=phase_id,
            default=controller_steps_per_prediction,
        )
        q_execute_world, grasp_hold_q = _apply_grasp_hold_execution_target(
            q_target_world,
            phase_id=phase_id,
            effector_dof=effector_dof,
            enabled=grasp_hold_enabled,
            capture_phase_ids=grasp_hold_capture_phase_ids,
            apply_phase_ids=grasp_hold_apply_phase_ids,
            held_effector_q=grasp_hold_q,
        )
        q_execute_world = _apply_effector_execution_offset(
            q_execute_world,
            phase_id=phase_id,
            effector_dof=effector_dof,
            enabled=effector_execution_offset_enabled,
            phase_ids=effector_execution_offset_phase_ids,
            indices=effector_execution_offset_indices,
            offsets=effector_execution_offset_values,
        )
        q_execute_world = _apply_robot_model_execution_root_offset(
            q_execute_world,
            phase_id=phase_id,
            enabled=execution_root_offset_enabled,
            offset_xyz=execution_root_offset_xyz,
            phase_ids=execution_root_offset_phase_ids,
        )
        action = _env_action_from_robot_model_q(env, q_execute_world)

        training_obs = _training_observation(env, obs) if training_episode is not None else None
        last_obs: dict[str, Any] | None = None
        last_info: dict[str, Any] | None = None
        for repeat_idx in range(phase_repeat_count):
            capture_camera = bool(
                (
                    runtime.non_contact_enabled
                    and (
                        capture_camera_every_controller_step
                        or store_trace
                        or repeat_idx + 1 >= phase_repeat_count
                    )
                )
                or (
                    training_episode is not None
                    and repeat_idx + 1 >= phase_repeat_count
                )
            )
            with profiler.measure("env_step_with_camera" if capture_camera else "env_step_state_only"):
                obs, _, _, _, info = env.step(action, include_camera_obs=capture_camera)
            last_obs = obs
            last_info = info
            completed_steps += 1
            current_q_world = _current_robot_model_q_from_env(env)
            final_object_pose = _object_pose_from_obs(obs)
            max_object_z = max(max_object_z, float(final_object_pose[2]))
            if "lift_delta" in obs:
                lift_delta = float(np.asarray(obs["lift_delta"], dtype=np.float32))
                final_lift_delta = lift_delta
                max_lift_delta = lift_delta if not np.isfinite(max_lift_delta) else max(max_lift_delta, lift_delta)
            if task_name == "pumpbottle_use":
                final_min_geom_z = _pump_bottle_min_geom_z(env)
                final_lift_clearance_delta = final_min_geom_z - initial_min_geom_z
                max_lift_clearance_delta = max(
                    max_lift_clearance_delta,
                    final_lift_clearance_delta,
                )
            if "button_slide" in obs:
                button_slide = float(np.asarray(obs["button_slide"], dtype=np.float32))
                final_button_slide = button_slide
                min_button_slide = button_slide if not np.isfinite(min_button_slide) else min(min_button_slide, button_slide)
            if task_name == "pumpbottle_use":
                button_slide_history.append(float(final_button_slide))
                lift_clearance_history.append(float(final_lift_clearance_delta))
            lift_success = bool(lift_success or bool(obs.get("lift_success", False)))
            button_success = bool(button_success or bool(obs.get("button_success", False)))
            _maybe_print_button_slide_debug(
                enabled=bool(print_button_slide_each_step and task_name == "pumpbottle_use"),
                prefix=f"trajectory={int(spec.source_trajectory_index)}",
                step_idx=step_idx,
                repeat_idx=repeat_idx,
                obs=obs,
            )
            if "target_key_slide" in obs:
                current_slide = np.asarray(obs["target_key_slide"], dtype=np.float32).reshape(-1)
                if final_target_key_slide.shape != current_slide.shape:
                    final_target_key_slide = np.full(current_slide.shape, np.nan, dtype=np.float32)
                    min_target_key_slide = np.full(current_slide.shape, np.nan, dtype=np.float32)
                final_target_key_slide = current_slide.astype(np.float32, copy=True)
                min_target_key_slide = np.fmin(min_target_key_slide, current_slide).astype(np.float32, copy=False)
                current_success = np.asarray(
                    obs.get("target_key_success", np.zeros(current_slide.shape, dtype=np.bool_)),
                    dtype=np.bool_,
                ).reshape(current_slide.shape)
                if target_key_success.shape != current_success.shape:
                    target_key_success = np.zeros(current_success.shape, dtype=np.bool_)
                target_key_success = np.logical_or(target_key_success, current_success)
            success = accumulate_task_success(
                success,
                env.task,
                obs,
                max_object_z=max_object_z,
            )

            if store_trace and trace is not None:
                append_trace_frame(trace, env, info, obs=obs)
                _append_online_low_level_trace_fields(
                    trace,
                    policy_step=step_idx,
                    repeat_idx=repeat_idx,
                    q_input_world=q_prev_world[0],
                    q_target_world=q_target_world,
                    q_execute_world=q_execute_world,
                    proprio_source=proprio_source,
                    controller_steps_per_prediction=phase_repeat_count,
                )
            if viewer_control is not None:
                viewer_control.update_overlay(
                    env,
                    step_idx=step_idx + 1,
                    total_steps=int(spec.frame_count),
                    trajectory_index=int(spec.source_trajectory_index),
                )
                if viewer_control.stop_requested or not env.viewer_is_running():
                    break
                time.sleep(float(dt))
        if training_episode is not None and last_obs is not None and last_info is not None:
            _append_online_training_frame(
                training_episode,
                obs=training_obs if training_obs is not None else last_obs,
                info=last_info,
                contact=contact,
                spec=spec,
                step_idx=step_idx,
                dataset_shift=dataset_shift,
                q_input_world=q_prev_world[0],
                q_input_shifted=q_prev_shifted[0],
                q_action_shifted=q_pred_shifted[0],
                q_action_world=q_execute_world,
                q_executed_world=current_q_world,
            )
        if image_renderer is not None and last_obs is not None:
            with profiler.measure("render_image"):
                image_renderer.render_step(env, step_idx=step_idx + render_step_offset)
        if viewer_control is not None and (viewer_control.stop_requested or not env.viewer_is_running()):
            break

    if nc_counts:
        counts = np.asarray(nc_counts, dtype=np.float32)
        print(
            f"trajectory={spec.source_trajectory_index} non_contact_selected="
            f"min={int(np.min(counts))} mean={float(np.mean(counts)):.1f} max={int(np.max(counts))}",
            flush=True,
        )

    if task_name == "pumpbottle_use":
        pump_bottle_lift_success_metric = _pump_bottle_lift_success_metric(
            pump_bottle_lift_success_metric,
            default="max_lift_delta",
        )
        lift_threshold = float(getattr(env.task, "lift_success_delta", float("inf")))
        lift_success = _pump_bottle_lift_success_for_metric(
            pump_bottle_lift_success_metric,
            max_lift_delta=float(max_lift_clearance_delta),
            final_lift_delta=float(final_lift_clearance_delta),
            threshold=lift_threshold,
        )
        success = bool(lift_success and button_success)
        timing = pump_press_order_metrics(
            button_slide_history,
            lift_clearance_history,
            button_success_slide=float(getattr(env.task, "button_success_slide", float("-inf"))),
            lift_success_delta=lift_threshold,
        )
        first_button_success_frame = int(timing.first_button_press_frame)
        first_lift_success_frame = int(timing.first_lift_frame)
        lift_clearance_at_first_button_success = float(
            timing.lift_clearance_at_first_press
        )
        button_after_lift_success = bool(timing.press_after_lift)
        ordered_success = bool(success and button_after_lift_success)

    profiler.print_summary()
    return PumpBottleRolloutResult(
        source_index=int(spec.source_trajectory_index),
        source_seed=source_seed,
        action_frame_count=int(spec.frame_count) * int(controller_steps_per_prediction),
        completed_steps=int(completed_steps),
        success=bool(success),
        max_cube_z=float(max_object_z),
        final_cube_pose=final_object_pose,
        error="",
        lift_success=bool(lift_success),
        button_success=bool(button_success),
        max_lift_delta=float(max_lift_delta),
        final_lift_delta=float(final_lift_delta),
        max_lift_clearance_delta=float(max_lift_clearance_delta),
        final_lift_clearance_delta=float(final_lift_clearance_delta),
        initial_min_geom_z=float(initial_min_geom_z),
        final_min_geom_z=float(final_min_geom_z),
        min_button_slide=float(min_button_slide),
        final_button_slide=float(final_button_slide),
        first_button_success_frame=first_button_success_frame,
        first_lift_success_frame=first_lift_success_frame,
        lift_clearance_delta_at_first_button_success=lift_clearance_at_first_button_success,
        button_after_lift_success=button_after_lift_success,
        ordered_success=ordered_success,
        target_key_success=target_key_success.astype(np.bool_, copy=True),
        final_target_key_slide=final_target_key_slide.astype(np.float32, copy=True),
        min_target_key_slide=min_target_key_slide.astype(np.float32, copy=True),
    )


def _rollout_episode_robotiq_stick(
    *,
    env: Any,
    cfg: DictConfig,
    spec: EpisodeSpec,
    contact: dict[str, np.ndarray],
    runtime: OnlineLowLevelRuntime,
    dataset_shift: np.ndarray,
    translation_groups: tuple[tuple[int, int, int], ...],
    initial_q: np.ndarray,
    proprio_source: str,
    controller_steps_per_prediction: int,
    store_trace: bool,
    trace: dict[str, list[np.ndarray]] | None,
    viewer_control: ViewerReplayControl | None,
    dt: float,
    seed: int,
    capture_camera_every_controller_step: bool,
    pump_bottle_lift_success_metric: str = "max_lift_delta",
    print_button_slide_each_step: bool = False,
    training_episode: OnlineTrainingEpisode | None = None,
    image_renderer: OnlineRolloutImageRenderer | None = None,
    profiler: RolloutProfiler | None = None,
) -> PumpBottleRolloutResult:
    profiler = profiler or RolloutProfiler(enabled=False)
    task_name = str(OmegaConf.select(cfg, "task.name", default="pumpbottle_use")).strip()
    with profiler.measure("reset"):
        obs, source_seed = _reset_from_contact_task(env, contact, spec, cfg=cfg)
        initial_q_world = validate_robotiq_stick_q(initial_q, label="initial_robotiq_stick_q")
        env.set_q(initial_q_world)
        obs = env.observe()
        render_step_offset = 0
        if image_renderer is not None:
            image_renderer.start_trajectory(env, spec)
            with profiler.measure("render_image"):
                image_renderer.render_step(env, step_idx=0)
            render_step_offset = 1
        if training_episode is not None:
            training_episode.source_seed = source_seed

    current_q_world = env.current_q()
    target_feedback_q_world = initial_q_world.astype(np.float32, copy=True)
    final_object_pose = np.asarray(obs["object_pose"], dtype=np.float32).reshape(7).copy()
    max_object_z = float(final_object_pose[2])
    max_lift_delta = float(np.asarray(obs.get("lift_delta", 0.0), dtype=np.float32))
    final_lift_delta = max_lift_delta
    initial_min_geom_z = (
        _pump_bottle_min_geom_z(env)
        if task_name == "pumpbottle_use"
        else float("nan")
    )
    final_min_geom_z = initial_min_geom_z
    max_lift_clearance_delta = 0.0 if np.isfinite(initial_min_geom_z) else float("nan")
    final_lift_clearance_delta = max_lift_clearance_delta
    min_button_slide = float(np.asarray(obs.get("button_slide", 0.0), dtype=np.float32))
    final_button_slide = min_button_slide
    lift_success = bool(obs.get("lift_success", False))
    button_success = bool(obs.get("button_success", False))
    button_slide_history: list[float] = []
    lift_clearance_history: list[float] = []
    first_button_success_frame = -1
    first_lift_success_frame = -1
    lift_clearance_at_first_button_success = float("nan")
    button_after_lift_success = False
    ordered_success = False
    target_key_success = np.asarray(
        obs.get("target_key_success", np.zeros((0,), dtype=np.bool_)),
        dtype=np.bool_,
    ).reshape(-1).copy()
    final_target_key_slide = np.asarray(
        obs.get("target_key_slide", np.full((target_key_success.shape[0],), np.nan, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1).copy()
    min_target_key_slide = final_target_key_slide.copy()
    success = bool(obs["success"])
    completed_steps = 0
    nc_counts: list[int] = []
    gripper_execution_action_offset_xyz = _gripper_execution_action_offset_xyz_from_cfg(cfg)
    gripper_min_enabled, gripper_min_command, gripper_min_phases = (
        _robotiq_gripper_min_command_from_cfg(cfg)
    )
    stick_offset_enabled, stick_offset_xyz, stick_offset_phases = (
        _robotiq_stick_execution_offset_from_cfg(cfg)
    )

    for step_idx in range(int(spec.frame_count)):
        if viewer_control is not None:
            if not viewer_control.wait_before_step(
                env,
                step_idx=step_idx,
                total_steps=int(spec.frame_count),
                trajectory_index=int(spec.source_trajectory_index),
            ):
                break

        contact_frame, contact_mask = _contact_frame(contact, spec, step_idx)
        q_feedback_world = (
            target_feedback_q_world if str(proprio_source) == "target" else current_q_world
        )
        q_prev_world = q_feedback_world.reshape(1, -1).astype(np.float32, copy=True)
        q_prev_shifted = _shift_composite_low_level_q(
            q_prev_world,
            dataset_shift,
            translation_groups=translation_groups,
            sign=1.0,
        )
        non_contact_cloud = None
        non_contact_valid = None
        if runtime.non_contact_enabled:
            with profiler.measure("non_contact_input"):
                non_contact_cloud, non_contact_valid, nc_stats = _build_non_contact_input(
                    cfg=cfg,
                    env=env,
                    runtime=runtime,
                    obs=obs,
                    q_prev_shifted=q_prev_shifted[0],
                    contact_cloud_shifted=contact_frame,
                    contact_valid_mask=contact_mask,
                    dataset_shift=dataset_shift,
                    seed=int(seed) + step_idx,
                )
            nc_counts.append(int(nc_stats.selected_count))

        with profiler.measure("low_level_policy"):
            q_pred_shifted = _predict_low_level_q_batch(
                runtime=runtime,
                contact_cloud_shifted=contact_frame.reshape(1, *contact_frame.shape),
                contact_valid_mask=contact_mask.reshape(1, *contact_mask.shape),
                q_prev_shifted=q_prev_shifted,
                seed=int(seed) + step_idx,
                non_contact_cloud_shifted=None if non_contact_cloud is None else non_contact_cloud.reshape(1, *non_contact_cloud.shape),
                non_contact_valid_mask=None if non_contact_valid is None else non_contact_valid.reshape(1, *non_contact_valid.shape),
            )
        q_pred_world = _shift_composite_low_level_q(
            q_pred_shifted,
            dataset_shift,
            translation_groups=translation_groups,
            sign=-1.0,
        )
        q_pred_world = _apply_fixed_zero_constraints(runtime, q_pred_world)
        q_pred_world = _clamp_robotiq_stick_gripper_cmd(q_pred_world)
        q_target_world = validate_robotiq_stick_q(q_pred_world[0], label="q_target_world")
        target_feedback_q_world = q_target_world.copy()
        phase_value = _contact_source_frame_or_none(contact, spec, step_idx, "phase")
        phase_id = None if phase_value is None else int(np.asarray(phase_value).reshape(()))
        q_execute_world = _apply_robotiq_gripper_execution_action_offset(
            q_target_world,
            gripper_execution_action_offset_xyz,
        )
        q_execute_world = _apply_robotiq_gripper_min_command(
            q_execute_world,
            phase_id=phase_id,
            enabled=gripper_min_enabled,
            value=gripper_min_command,
            phase_ids=gripper_min_phases,
        )
        q_execute_world = _apply_robotiq_stick_execution_offset(
            q_execute_world,
            phase_id=phase_id,
            enabled=stick_offset_enabled,
            offset_xyz=stick_offset_xyz,
            phase_ids=stick_offset_phases,
        )

        training_obs = _training_observation(env, obs) if training_episode is not None else None
        last_obs: dict[str, Any] | None = None
        last_info: dict[str, Any] | None = None
        for repeat_idx in range(int(controller_steps_per_prediction)):
            capture_camera = bool(
                (
                    runtime.non_contact_enabled
                    and (
                        capture_camera_every_controller_step
                        or store_trace
                        or repeat_idx + 1 >= int(controller_steps_per_prediction)
                    )
                )
                or (
                    training_episode is not None
                    and repeat_idx + 1 >= int(controller_steps_per_prediction)
                )
            )
            with profiler.measure("env_step_with_camera" if capture_camera else "env_step_state_only"):
                obs, _, _, _, info = env.step(q_execute_world, include_camera_obs=capture_camera)
            last_obs = obs
            last_info = info
            completed_steps += 1
            current_q_world = env.current_q()
            final_object_pose = np.asarray(obs["object_pose"], dtype=np.float32).reshape(7).copy()
            max_object_z = max(max_object_z, float(final_object_pose[2]))
            final_lift_delta = float(np.asarray(obs.get("lift_delta", final_lift_delta), dtype=np.float32))
            max_lift_delta = max(max_lift_delta, final_lift_delta)
            if task_name == "pumpbottle_use":
                final_min_geom_z = _pump_bottle_min_geom_z(env)
                final_lift_clearance_delta = final_min_geom_z - initial_min_geom_z
                max_lift_clearance_delta = max(
                    max_lift_clearance_delta,
                    final_lift_clearance_delta,
                )
            final_button_slide = float(np.asarray(obs.get("button_slide", final_button_slide), dtype=np.float32))
            min_button_slide = min(min_button_slide, final_button_slide)
            if task_name == "pumpbottle_use":
                button_slide_history.append(float(final_button_slide))
                lift_clearance_history.append(float(final_lift_clearance_delta))
            lift_success = bool(lift_success or bool(obs.get("lift_success", False)))
            button_success = bool(button_success or bool(obs.get("button_success", False)))
            _maybe_print_button_slide_debug(
                enabled=bool(print_button_slide_each_step and task_name == "pumpbottle_use"),
                prefix=f"trajectory={int(spec.source_trajectory_index)}",
                step_idx=step_idx,
                repeat_idx=repeat_idx,
                obs=obs,
            )
            if "target_key_slide" in obs:
                current_slide = np.asarray(obs["target_key_slide"], dtype=np.float32).reshape(-1)
                if final_target_key_slide.shape != current_slide.shape:
                    final_target_key_slide = np.full(current_slide.shape, np.nan, dtype=np.float32)
                    min_target_key_slide = np.full(current_slide.shape, np.nan, dtype=np.float32)
                final_target_key_slide = current_slide.astype(np.float32, copy=True)
                min_target_key_slide = np.fmin(min_target_key_slide, current_slide).astype(np.float32, copy=False)
                current_success = np.asarray(
                    obs.get("target_key_success", np.zeros(current_slide.shape, dtype=np.bool_)),
                    dtype=np.bool_,
                ).reshape(current_slide.shape)
                if target_key_success.shape != current_success.shape:
                    target_key_success = np.zeros(current_success.shape, dtype=np.bool_)
                target_key_success = np.logical_or(target_key_success, current_success)
            success = bool(success or bool(obs["success"]))

            if store_trace and trace is not None:
                append_trace_frame(trace, env, info, obs=obs)
                _append_online_low_level_trace_fields(
                    trace,
                    policy_step=step_idx,
                    repeat_idx=repeat_idx,
                    q_input_world=q_prev_world[0],
                    q_target_world=q_target_world,
                    q_execute_world=q_execute_world,
                    proprio_source=proprio_source,
                    controller_steps_per_prediction=controller_steps_per_prediction,
                )
            if viewer_control is not None:
                viewer_control.update_overlay(
                    env,
                    step_idx=step_idx + 1,
                    total_steps=int(spec.frame_count),
                    trajectory_index=int(spec.source_trajectory_index),
                )
                if viewer_control.stop_requested or not env.viewer_is_running():
                    break
                time.sleep(float(dt))
        if training_episode is not None and last_obs is not None and last_info is not None:
            _append_online_training_frame(
                training_episode,
                obs=training_obs if training_obs is not None else last_obs,
                info=last_info,
                contact=contact,
                spec=spec,
                step_idx=step_idx,
                dataset_shift=dataset_shift,
                q_input_world=q_prev_world[0],
                q_input_shifted=q_prev_shifted[0],
                q_action_shifted=q_pred_shifted[0],
                q_action_world=q_execute_world,
                q_executed_world=current_q_world,
            )
        if image_renderer is not None and last_obs is not None:
            with profiler.measure("render_image"):
                image_renderer.render_step(env, step_idx=step_idx + render_step_offset)
        if viewer_control is not None and (viewer_control.stop_requested or not env.viewer_is_running()):
            break

    if nc_counts:
        counts = np.asarray(nc_counts, dtype=np.float32)
        print(
            f"trajectory={spec.source_trajectory_index} non_contact_selected="
            f"min={int(np.min(counts))} mean={float(np.mean(counts)):.1f} max={int(np.max(counts))}",
            flush=True,
        )

    if task_name == "pumpbottle_use":
        pump_bottle_lift_success_metric = _pump_bottle_lift_success_metric(
            pump_bottle_lift_success_metric,
            default="max_lift_delta",
        )
        lift_threshold = float(getattr(env.task, "lift_success_delta", float("inf")))
        lift_success = _pump_bottle_lift_success_for_metric(
            pump_bottle_lift_success_metric,
            max_lift_delta=float(max_lift_clearance_delta),
            final_lift_delta=float(final_lift_clearance_delta),
            threshold=lift_threshold,
        )
        success = bool(lift_success and button_success)
        timing = pump_press_order_metrics(
            button_slide_history,
            lift_clearance_history,
            button_success_slide=float(getattr(env.task, "button_success_slide", float("-inf"))),
            lift_success_delta=lift_threshold,
        )
        first_button_success_frame = int(timing.first_button_press_frame)
        first_lift_success_frame = int(timing.first_lift_frame)
        lift_clearance_at_first_button_success = float(
            timing.lift_clearance_at_first_press
        )
        button_after_lift_success = bool(timing.press_after_lift)
        ordered_success = bool(success and button_after_lift_success)

    profiler.print_summary()
    return PumpBottleRolloutResult(
        source_index=int(spec.source_trajectory_index),
        source_seed=source_seed,
        action_frame_count=int(spec.frame_count) * int(controller_steps_per_prediction),
        completed_steps=int(completed_steps),
        success=bool(success),
        max_cube_z=float(max_object_z),
        final_cube_pose=final_object_pose,
        error="",
        lift_success=bool(lift_success),
        button_success=bool(button_success),
        max_lift_delta=float(max_lift_delta),
        final_lift_delta=float(final_lift_delta),
        max_lift_clearance_delta=float(max_lift_clearance_delta),
        final_lift_clearance_delta=float(final_lift_clearance_delta),
        initial_min_geom_z=float(initial_min_geom_z),
        final_min_geom_z=float(final_min_geom_z),
        min_button_slide=float(min_button_slide),
        final_button_slide=float(final_button_slide),
        first_button_success_frame=first_button_success_frame,
        first_lift_success_frame=first_lift_success_frame,
        lift_clearance_delta_at_first_button_success=lift_clearance_at_first_button_success,
        button_after_lift_success=button_after_lift_success,
        ordered_success=ordered_success,
        target_key_success=target_key_success.astype(np.bool_, copy=True),
        final_target_key_slide=final_target_key_slide.astype(np.float32, copy=True),
        min_target_key_slide=min_target_key_slide.astype(np.float32, copy=True),
    )


def _append_online_low_level_trace_fields(
    trace: dict[str, list[np.ndarray]],
    *,
    policy_step: int,
    repeat_idx: int,
    q_input_world: np.ndarray,
    q_target_world: np.ndarray,
    q_execute_world: np.ndarray | None = None,
    proprio_source: str,
    controller_steps_per_prediction: int,
) -> None:
    trace.setdefault("low_level_policy_step", []).append(np.asarray(int(policy_step), dtype=np.int32))
    trace.setdefault("low_level_action_repeat_index", []).append(np.asarray(int(repeat_idx), dtype=np.int32))
    trace.setdefault("low_level_q_input_world", []).append(
        np.asarray(q_input_world, dtype=np.float32).reshape(-1).copy()
    )
    trace.setdefault("low_level_q_target_world", []).append(
        np.asarray(q_target_world, dtype=np.float32).reshape(-1).copy()
    )
    if q_execute_world is not None:
        trace.setdefault("low_level_q_execute_world", []).append(
            np.asarray(q_execute_world, dtype=np.float32).reshape(-1).copy()
        )
    trace.setdefault("low_level_proprio_source", []).append(np.asarray(str(proprio_source)))
    trace.setdefault("low_level_controller_steps_per_prediction", []).append(
        np.asarray(int(controller_steps_per_prediction), dtype=np.int32)
    )


def _training_episode_frame_count(episode: OnlineTrainingEpisode) -> int:
    frames = dict(episode.frames)
    if "low_level_q_action_world" not in frames:
        return 0
    return len(frames["low_level_q_action_world"])


def _append_frame_value(frames: dict[str, list[np.ndarray]], key: str, value: Any, *, dtype: Any | None = None) -> None:
    arr = np.asarray(value, dtype=dtype).copy() if dtype is not None else np.asarray(value).copy()
    frames.setdefault(str(key), []).append(arr)


def _contact_source_frame_or_none(
    contact: dict[str, np.ndarray],
    spec: EpisodeSpec,
    step_idx: int,
    key: str,
) -> np.ndarray | None:
    if key not in contact:
        return None
    arr = np.asarray(contact[key])
    if arr.ndim == 0:
        return arr.copy()
    frame_idx = int(spec.contact_start) + int(step_idx)
    if frame_idx < int(arr.shape[0]):
        return np.asarray(arr[frame_idx]).copy()
    return None


def _grasp_hold_config_from_cfg(cfg: DictConfig) -> tuple[bool, frozenset[int], frozenset[int]]:
    section = "online_rollout.grasp_hold"
    enabled = bool(OmegaConf.select(cfg, f"{section}.enabled", default=False))
    capture = frozenset(
        int(value)
        for value in OmegaConf.select(cfg, f"{section}.capture_phase_ids", default=[2])
    )
    apply = frozenset(
        int(value)
        for value in OmegaConf.select(cfg, f"{section}.apply_phase_ids", default=[3, 4])
    )
    if enabled and not capture:
        raise ValueError(f"{section}.capture_phase_ids must be non-empty when enabled")
    if enabled and not apply:
        raise ValueError(f"{section}.apply_phase_ids must be non-empty when enabled")
    if capture & apply:
        raise ValueError(f"{section} capture and apply phase IDs must not overlap")
    return enabled, capture, apply


def _set_grasp_hold_payload_metadata(payload: dict[str, object], cfg: DictConfig) -> None:
    enabled, capture_phase_ids, apply_phase_ids = _grasp_hold_config_from_cfg(cfg)
    payload["grasp_hold_enabled"] = np.asarray(enabled)
    payload["grasp_hold_capture_phase_ids"] = np.asarray(
        sorted(capture_phase_ids), dtype=np.int8
    )
    payload["grasp_hold_apply_phase_ids"] = np.asarray(
        sorted(apply_phase_ids), dtype=np.int8
    )


def _set_robot_model_execution_root_offset_payload_metadata(
    payload: dict[str, object],
    cfg: DictConfig,
) -> None:
    enabled, offset_xyz, phase_ids = _robot_model_execution_root_offset_from_cfg(cfg)
    payload["robot_model_execution_root_offset_enabled"] = np.asarray(enabled)
    payload["robot_model_execution_root_offset_xyz"] = np.asarray(offset_xyz, dtype=np.float32)
    payload["robot_model_execution_root_offset_phase_ids"] = np.asarray(
        sorted(phase_ids), dtype=np.int8
    )


def _set_effector_execution_offset_payload_metadata(
    payload: dict[str, object],
    cfg: DictConfig,
) -> None:
    enabled, phase_ids, indices, offsets = _effector_execution_offset_from_cfg(cfg)
    payload["effector_execution_offset_enabled"] = np.asarray(enabled)
    payload["effector_execution_offset_phase_ids"] = np.asarray(
        sorted(phase_ids), dtype=np.int32
    )
    payload["effector_execution_offset_indices"] = np.asarray(indices, dtype=np.int32)
    payload["effector_execution_offset_values"] = np.asarray(offsets, dtype=np.float32)


def _set_pump_bottle_physics_payload_metadata(
    payload: dict[str, object],
    cfg: DictConfig,
) -> None:
    """Persist the effective shared pump-bottle and target-contact settings."""

    scalar_fields = {
        "pump_bottle_mass": "pump_bottle.mass",
        "pump_bottle_button_mass": "pump_bottle.button_mass",
        "pump_bottle_button_joint_stiffness": "pump_bottle.button_joint_stiffness",
        "pump_bottle_button_joint_damping": "pump_bottle.button_joint_damping",
        "pump_bottle_button_actuator_kp": "pump_bottle.button_actuator_kp",
        "pump_bottle_contact_margin": "pump_bottle.contact_margin",
        "barrett_fingertip_capsule_radius": "env.barrett_fingertip_capsule_radius",
    }
    for output_key, config_key in scalar_fields.items():
        value = OmegaConf.select(cfg, config_key, default=None)
        if value is not None:
            payload[output_key] = np.asarray(float(value), dtype=np.float32)

    vector_fields = {
        "base_rotation_reference_euler_xyz": "controller.base_rpy.reference_euler_xyz",
        "pump_bottle_friction": "pump_bottle.friction",
        "pump_bottle_button_joint_range": "pump_bottle.button_joint_range",
        "barrett_fingertip_friction": "env.barrett_fingertip_friction",
        "robotiq_gripper_pad_friction": "env.gripper_pad_friction",
        "robotiq_gripper_pad_patch_grid": "env.gripper_pad_patch_grid",
    }
    for output_key, config_key in vector_fields.items():
        value = OmegaConf.select(cfg, config_key, default=None)
        if value is not None:
            payload[output_key] = np.asarray(value)

    text_fields = {
        "pump_bottle_lower_collision_mode": "pump_bottle.lower_collision_mode",
        "pump_bottle_neck_collision_mode": "pump_bottle.neck_collision_mode",
        "barrett_fingertip_collision_model": "env.barrett_fingertip_collision_model",
    }
    for output_key, config_key in text_fields.items():
        value = OmegaConf.select(cfg, config_key, default=None)
        if value is not None:
            payload[output_key] = np.asarray(str(value))


def _payload_dt(payload: dict[str, Any]) -> float | None:
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        return None
    time_meta = meta.get("time", {})
    value = time_meta.get("dt") if isinstance(time_meta, dict) else None
    if value is None:
        value = meta.get("dt")
    if value is None:
        fps = time_meta.get("fps") if isinstance(time_meta, dict) else meta.get("fps")
        if fps is not None:
            value = 1.0 / float(fps)
    if value is None:
        return None
    dt = float(value)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"contact payload dt must be positive and finite, got {value}")
    return dt


def _apply_grasp_hold_execution_target(
    q_target_world: np.ndarray,
    *,
    phase_id: int | None,
    effector_dof: int,
    enabled: bool,
    capture_phase_ids: frozenset[int],
    apply_phase_ids: frozenset[int],
    held_effector_q: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply the most recently captured finger command in configured phases."""

    q_execute = np.asarray(q_target_world, dtype=np.float32).reshape(-1).copy()
    if not enabled:
        return q_execute, held_effector_q
    if effector_dof <= 0 or effector_dof >= int(q_execute.size):
        raise ValueError(
            f"grasp hold effector_dof must be in [1, {int(q_execute.size) - 1}], got {effector_dof}"
        )
    if phase_id in capture_phase_ids:
        held_effector_q = q_execute[-effector_dof:].copy()
    if phase_id in apply_phase_ids:
        if held_effector_q is None:
            raise ValueError(
                f"grasp hold reached apply phase {phase_id} before a capture phase"
            )
        held = np.asarray(held_effector_q, dtype=np.float32).reshape(-1)
        if held.shape != (effector_dof,):
            raise ValueError(
                f"grasp hold command must have shape ({effector_dof},), got {held.shape}"
            )
        q_execute[-effector_dof:] = held
    return q_execute, held_effector_q


def _apply_phase_finger_effort(
    env: TabletopManipulationEnv,
    *,
    cfg: DictConfig,
    phase_id: int | None,
) -> float:
    """Apply an optional absolute finger effort limit for selected phases."""

    default_effort = float(
        OmegaConf.select(cfg, "controller.fingers.max_effort", default=0.5)
    )
    section = "online_rollout.phase_finger_effort"
    enabled = bool(OmegaConf.select(cfg, f"{section}.enabled", default=False))
    phase_ids = frozenset(
        int(value) for value in OmegaConf.select(cfg, f"{section}.phase_ids", default=[5])
    )
    override_effort = float(
        OmegaConf.select(cfg, f"{section}.max_effort", default=default_effort)
    )
    if default_effort <= 0.0 or not np.isfinite(default_effort):
        raise ValueError("controller.fingers.max_effort must be positive and finite")
    if enabled and (not phase_ids or override_effort <= 0.0 or not np.isfinite(override_effort)):
        raise ValueError(
            f"{section} requires non-empty phase_ids and a positive finite max_effort"
        )
    effort = override_effort if enabled and phase_id in phase_ids else default_effort
    actuator_ids = np.asarray(env.robot.side("right").hand_actuator_ids, dtype=np.int32)
    env.model.actuator_forcerange[actuator_ids, 0] = -effort
    env.model.actuator_forcerange[actuator_ids, 1] = effort
    return effort


def _apply_phase_arm_gains(
    env: TabletopManipulationEnv,
    *,
    cfg: DictConfig,
    phase_id: int | None,
    default_kp: np.ndarray,
    default_kv: np.ndarray,
    joint_ids: np.ndarray | None = None,
    default_joint_force_range: np.ndarray | None = None,
) -> tuple[float, float]:
    """Scale Wuji arm gains in selected phases and restore the configured baseline."""

    default_kp_scale = float(OmegaConf.select(cfg, "controller.arm_kp_scale", default=1.0))
    default_kv_scale = float(OmegaConf.select(cfg, "controller.arm_kv_scale", default=1.0))
    section = "online_rollout.phase_arm_gains"
    enabled = bool(OmegaConf.select(cfg, f"{section}.enabled", default=False))
    phase_ids = frozenset(
        int(value) for value in OmegaConf.select(cfg, f"{section}.phase_ids", default=[1, 2, 3])
    )
    object_ids = frozenset(
        str(value)
        for value in OmegaConf.select(cfg, f"{section}.object_ids", default=[])
    )
    override_kp_scale = float(
        OmegaConf.select(cfg, f"{section}.kp_scale", default=default_kp_scale)
    )
    override_kv_scale = float(
        OmegaConf.select(cfg, f"{section}.kv_scale", default=default_kv_scale)
    )
    override_joint_force_scale = float(
        OmegaConf.select(cfg, f"{section}.joint_force_scale", default=1.0)
    )
    values = (
        default_kp_scale,
        default_kv_scale,
        override_kp_scale,
        override_kv_scale,
        override_joint_force_scale,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"{section} and controller.arm gain scales must be positive and finite")
    if enabled and not phase_ids:
        raise ValueError(f"{section}.phase_ids must be non-empty when enabled")

    actuator_ids = np.asarray(env.robot.side("right").arm_actuator_ids, dtype=np.int32)
    baseline_kp = np.asarray(default_kp, dtype=np.float64).reshape(-1)
    baseline_kv = np.asarray(default_kv, dtype=np.float64).reshape(-1)
    expected_shape = (int(actuator_ids.size),)
    if baseline_kp.shape != expected_shape or baseline_kv.shape != expected_shape:
        raise ValueError(
            f"Wuji arm baseline gains must have shape {expected_shape}, got "
            f"kp={baseline_kp.shape} kv={baseline_kv.shape}"
        )

    task_object_spec = getattr(getattr(env, "task", None), "object_spec", None)
    task_object_id = getattr(task_object_spec, "object_id", None)
    object_matches = not object_ids or str(task_object_id) in object_ids
    kp_scale, kv_scale = (
        (override_kp_scale, override_kv_scale)
        if enabled and object_matches and phase_id in phase_ids
        else (default_kp_scale, default_kv_scale)
    )
    kp = baseline_kp * (kp_scale / default_kp_scale)
    kv = baseline_kv * (kv_scale / default_kv_scale)
    env.model.actuator_gainprm[actuator_ids, 0] = kp
    env.model.actuator_biasprm[actuator_ids, 1] = -kp
    env.model.actuator_biasprm[actuator_ids, 2] = -kv
    if (joint_ids is None) != (default_joint_force_range is None):
        raise ValueError("joint_ids and default_joint_force_range must be provided together")
    if joint_ids is not None and default_joint_force_range is not None:
        resolved_joint_ids = np.asarray(joint_ids, dtype=np.int32).reshape(-1)
        baseline_force_range = np.asarray(
            default_joint_force_range,
            dtype=np.float64,
        )
        expected_force_shape = (int(resolved_joint_ids.size), 2)
        if baseline_force_range.shape != expected_force_shape:
            raise ValueError(
                "Wuji arm baseline joint force ranges must have shape "
                f"{expected_force_shape}, got {baseline_force_range.shape}"
            )
        force_scale = (
            override_joint_force_scale
            if enabled and object_matches and phase_id in phase_ids
            else 1.0
        )
        env.model.jnt_actfrcrange[resolved_joint_ids] = baseline_force_range * force_scale
    return kp_scale, kv_scale


def _apply_phase_hand_gains(
    env: TabletopManipulationEnv,
    *,
    cfg: DictConfig,
    phase_id: int | None,
    default_kp: np.ndarray,
    default_kv: np.ndarray,
) -> tuple[float, float]:
    """Scale Wuji hand gains in selected phases and restore the configured baseline."""

    default_kp_scale = float(OmegaConf.select(cfg, "controller.hand_kp_scale", default=1.0))
    default_kv_scale = float(OmegaConf.select(cfg, "controller.hand_kv_scale", default=1.0))
    section = "online_rollout.phase_hand_gains"
    enabled = bool(OmegaConf.select(cfg, f"{section}.enabled", default=False))
    phase_ids = frozenset(
        int(value) for value in OmegaConf.select(cfg, f"{section}.phase_ids", default=[5])
    )
    override_kp_scale = float(
        OmegaConf.select(cfg, f"{section}.kp_scale", default=default_kp_scale)
    )
    override_kv_scale = float(
        OmegaConf.select(cfg, f"{section}.kv_scale", default=default_kv_scale)
    )
    values = (default_kp_scale, default_kv_scale, override_kp_scale, override_kv_scale)
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"{section} and controller.hand gain scales must be positive and finite")
    if enabled and not phase_ids:
        raise ValueError(f"{section}.phase_ids must be non-empty when enabled")

    actuator_ids = np.asarray(env.robot.side("right").hand_actuator_ids, dtype=np.int32)
    baseline_kp = np.asarray(default_kp, dtype=np.float64).reshape(-1)
    baseline_kv = np.asarray(default_kv, dtype=np.float64).reshape(-1)
    expected_shape = (int(actuator_ids.size),)
    if baseline_kp.shape != expected_shape or baseline_kv.shape != expected_shape:
        raise ValueError(
            f"Wuji hand baseline gains must have shape {expected_shape}, got "
            f"kp={baseline_kp.shape} kv={baseline_kv.shape}"
        )

    kp_scale, kv_scale = (
        (override_kp_scale, override_kv_scale)
        if enabled and phase_id in phase_ids
        else (default_kp_scale, default_kv_scale)
    )
    kp = baseline_kp * (kp_scale / default_kp_scale)
    kv = baseline_kv * (kv_scale / default_kv_scale)
    env.model.actuator_gainprm[actuator_ids, 0] = kp
    env.model.actuator_biasprm[actuator_ids, 1] = -kp
    env.model.actuator_biasprm[actuator_ids, 2] = -kv
    return kp_scale, kv_scale


def _apply_phase_finger_gains(
    env: TabletopManipulationEnv,
    *,
    cfg: DictConfig,
    phase_id: int | None,
) -> tuple[float, float]:
    """Apply optional finger position gains in selected source phases."""

    default_kp = float(OmegaConf.select(cfg, "controller.fingers.kp", default=16.0))
    default_kv_value = OmegaConf.select(cfg, "controller.fingers.kv", default=8.0)
    default_kv = (
        2.0 * float(np.sqrt(default_kp))
        if isinstance(default_kv_value, str) and default_kv_value.strip().lower() == "auto"
        else float(default_kv_value)
    )
    section = "online_rollout.phase_finger_gains"
    enabled = bool(OmegaConf.select(cfg, f"{section}.enabled", default=False))
    phase_ids = frozenset(
        int(value) for value in OmegaConf.select(cfg, f"{section}.phase_ids", default=[2, 3])
    )
    override_kp = float(OmegaConf.select(cfg, f"{section}.kp", default=default_kp))
    override_kv = float(OmegaConf.select(cfg, f"{section}.kv", default=default_kv))
    values = (default_kp, default_kv, override_kp, override_kv)
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"{section} and controller.fingers gains must be positive and finite")
    if enabled and not phase_ids:
        raise ValueError(f"{section}.phase_ids must be non-empty when enabled")

    kp, kv = (
        (override_kp, override_kv)
        if enabled and phase_id in phase_ids
        else (default_kp, default_kv)
    )
    actuator_ids = np.asarray(env.robot.side("right").hand_actuator_ids, dtype=np.int32)
    env.model.actuator_gainprm[actuator_ids, 0] = kp
    env.model.actuator_biasprm[actuator_ids, 1] = -kp
    env.model.actuator_biasprm[actuator_ids, 2] = -kv
    return kp, kv


def _phase_controller_steps_per_prediction(
    cfg: DictConfig,
    *,
    phase_id: int | None,
    default: int,
) -> int:
    """Return a phase-specific action repeat count when explicitly enabled."""

    if int(default) <= 0:
        raise ValueError(f"controller_steps_per_prediction must be positive, got {default}")
    section = "online_rollout.phase_controller_steps"
    if not bool(OmegaConf.select(cfg, f"{section}.enabled", default=False)):
        return int(default)
    phase_ids = frozenset(
        int(value) for value in OmegaConf.select(cfg, f"{section}.phase_ids", default=[2])
    )
    repeat_count = int(OmegaConf.select(cfg, f"{section}.steps_per_prediction", default=default))
    if not phase_ids or repeat_count <= 0:
        raise ValueError(
            f"{section} requires non-empty phase_ids and positive steps_per_prediction"
        )
    return repeat_count if phase_id in phase_ids else int(default)


def _contact_world_frame(
    contact: dict[str, np.ndarray],
    spec: EpisodeSpec,
    step_idx: int,
    key: str,
    shift: np.ndarray,
    fallback: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    world = _contact_source_frame_or_none(contact, spec, step_idx, f"{key}_world")
    if world is not None:
        return np.asarray(world, dtype=np.float32).copy()
    out = np.asarray(fallback, dtype=np.float32).copy()
    valid = np.asarray(mask, dtype=np.bool_)
    if out.ndim >= 2 and out.shape[-1] >= 3 and valid.shape == out.shape[:-1]:
        out[..., :3][valid] -= np.asarray(shift, dtype=np.float32).reshape(1, 3)
    return out


def _pointcloud_xyz_from_obs_or_contact(
    obs: dict[str, Any],
    contact: dict[str, np.ndarray],
    spec: EpisodeSpec,
    step_idx: int,
) -> np.ndarray:
    target_num = _training_pointcloud_target_count(obs, contact, spec, step_idx)
    object_xyz = _object_pointcloud_xyz_from_obs(obs, target_num=target_num)
    if object_xyz is not None:
        return object_xyz
    if "pointcloud" in obs:
        pointcloud = obs["pointcloud"]
        if isinstance(pointcloud, dict) and "xyzw" in pointcloud:
            xyzw = np.asarray(pointcloud["xyzw"], dtype=np.float32)
            if xyzw.ndim == 2 and xyzw.shape[1] >= 3:
                return xyzw[:, :3].astype(np.float32, copy=True)
    for key in ("obs_point_cloud", "manipulated_point_cloud"):
        frame = _contact_source_frame_or_none(contact, spec, step_idx, key)
        if frame is not None:
            arr = np.asarray(frame, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return arr[:, :3].astype(np.float32, copy=True)
    raise KeyError(
        "online_rollout.save_training_pt requires a point cloud in the live observation "
        "or in the contact source payload"
    )


def _training_pointcloud_target_count(
    obs: dict[str, Any],
    contact: dict[str, np.ndarray],
    spec: EpisodeSpec,
    step_idx: int,
) -> int | None:
    if "pointcloud" in obs:
        pointcloud = obs["pointcloud"]
        if isinstance(pointcloud, dict) and "xyzw" in pointcloud:
            xyzw = np.asarray(pointcloud["xyzw"])
            if xyzw.ndim == 2:
                return int(xyzw.shape[0])
    for key in ("obs_point_cloud", "manipulated_point_cloud"):
        frame = _contact_source_frame_or_none(contact, spec, step_idx, key)
        if frame is not None:
            arr = np.asarray(frame)
            if arr.ndim == 2:
                return int(arr.shape[0])
    return None


def _object_pointcloud_xyz_from_obs(obs: dict[str, Any], *, target_num: int | None) -> np.ndarray | None:
    sensor_data = obs.get("sensor_data")
    sensor_param = obs.get("sensor_param")
    if not isinstance(sensor_data, dict) or not isinstance(sensor_param, dict):
        return None
    camera_names = obs.get("policy_camera_names", None)
    if camera_names is None:
        camera_names = tuple(sensor_data.keys())

    points: list[np.ndarray] = []
    for camera_name_raw in camera_names:
        camera_name = str(camera_name_raw)
        if camera_name not in sensor_data or camera_name not in sensor_param:
            continue
        cam_data = sensor_data[camera_name]
        cam_param = sensor_param[camera_name]
        if not isinstance(cam_data, dict) or not isinstance(cam_param, dict):
            continue
        mask = None
        for mask_key in ("object_mask", "cube_mask"):
            if mask_key in cam_data:
                candidate = np.asarray(cam_data[mask_key], dtype=np.bool_)
                mask = candidate if mask is None else np.logical_or(mask, candidate)
        if mask is None or "depth" not in cam_data or "intrinsic_cv" not in cam_param or "extrinsic_cv" not in cam_param:
            continue
        depth = np.asarray(cam_data["depth"], dtype=np.float32)
        if mask.shape != depth.shape:
            continue
        valid = mask & np.isfinite(depth) & (depth > 1.0e-5)
        valid_flat = np.flatnonzero(valid.reshape(-1))
        if valid_flat.size == 0:
            continue
        ys, xs = np.divmod(valid_flat, depth.shape[1])
        z = depth.reshape(-1)[valid_flat]
        xyz = deproject_pixels_cv(
            xs.astype(np.float32),
            ys.astype(np.float32),
            z.astype(np.float32),
            cam_param["intrinsic_cv"],
            cam_param["extrinsic_cv"],
        )
        finite = np.isfinite(xyz).all(axis=1)
        if np.any(finite):
            points.append(xyz[finite].astype(np.float32, copy=False))
    if not points:
        return None
    xyz_all = np.concatenate(points, axis=0).astype(np.float32, copy=False)
    return _sample_or_pad_training_pointcloud(xyz_all, target_num=target_num)


def _sample_or_pad_training_pointcloud(points: np.ndarray, *, target_num: int | None) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if target_num is None or int(target_num) < 0:
        return xyz.copy()
    target = int(target_num)
    if target == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if xyz.shape[0] == 0:
        return np.zeros((target, 3), dtype=np.float32)
    if int(xyz.shape[0]) > target:
        return xyz[farthest_point_sample_indices(xyz, target)].astype(np.float32, copy=True)
    if int(xyz.shape[0]) < target:
        repeat = target - int(xyz.shape[0])
        repeat_indices = np.resize(np.arange(int(xyz.shape[0]), dtype=np.int64), repeat)
        xyz = np.concatenate([xyz, xyz[repeat_indices]], axis=0)
    return xyz.astype(np.float32, copy=True)


def _obs_has_training_pointcloud(obs: dict[str, Any]) -> bool:
    if "pointcloud" in obs:
        pointcloud = obs["pointcloud"]
        if isinstance(pointcloud, dict) and "xyzw" in pointcloud:
            xyzw = np.asarray(pointcloud["xyzw"])
            if xyzw.ndim == 2 and xyzw.shape[1] >= 3:
                return True
    sensor_data = obs.get("sensor_data")
    sensor_param = obs.get("sensor_param")
    if not isinstance(sensor_data, dict) or not isinstance(sensor_param, dict):
        return False
    for camera_name_raw in obs.get("policy_camera_names", tuple(sensor_data.keys())):
        camera_name = str(camera_name_raw)
        cam_data = sensor_data.get(camera_name)
        cam_param = sensor_param.get(camera_name)
        if not isinstance(cam_data, dict) or not isinstance(cam_param, dict):
            continue
        if "depth" not in cam_data or "intrinsic_cv" not in cam_param or "extrinsic_cv" not in cam_param:
            continue
        if "object_mask" in cam_data or "cube_mask" in cam_data:
            return True
    return False


def _training_observation(env: Any, obs: dict[str, Any]) -> dict[str, Any]:
    return obs if _obs_has_training_pointcloud(obs) else env.observe(include_camera_obs=True)


def _append_online_training_frame(
    episode: OnlineTrainingEpisode,
    *,
    obs: dict[str, Any],
    info: dict[str, Any],
    contact: dict[str, np.ndarray],
    spec: EpisodeSpec,
    step_idx: int,
    dataset_shift: np.ndarray,
    q_input_world: np.ndarray,
    q_input_shifted: np.ndarray,
    q_action_shifted: np.ndarray,
    q_action_world: np.ndarray,
    q_executed_world: np.ndarray,
) -> None:
    frames = episode.frames
    obs_xyz = _pointcloud_xyz_from_obs_or_contact(obs, contact, spec, step_idx)
    _append_frame_value(frames, "obs_point_cloud", obs_xyz, dtype=np.float32)
    _append_frame_value(frames, "manipulated_point_cloud", obs_xyz, dtype=np.float32)
    _append_frame_value(frames, "context_point_cloud", np.zeros((0, 3), dtype=np.float32))

    contact_action = np.asarray(_contact_source_frame_or_none(contact, spec, step_idx, "contact_action"), dtype=np.float32)
    contact_action_valid = np.asarray(
        _contact_source_frame_or_none(contact, spec, step_idx, "contact_action_valid_mask"),
        dtype=np.bool_,
    )
    contact_proprio = _contact_source_frame_or_none(contact, spec, step_idx, "contact_proprio")
    contact_proprio_valid = _contact_source_frame_or_none(contact, spec, step_idx, "contact_proprio_valid_mask")
    if contact_proprio is None:
        contact_proprio = contact_action
    if contact_proprio_valid is None:
        contact_proprio_valid = contact_action_valid
    contact_proprio = np.asarray(contact_proprio, dtype=np.float32)
    contact_proprio_valid = np.asarray(contact_proprio_valid, dtype=np.bool_)

    _append_frame_value(frames, "contact_action", contact_action, dtype=np.float32)
    _append_frame_value(frames, "contact_action_valid_mask", contact_action_valid, dtype=np.bool_)
    _append_frame_value(frames, "contact_proprio", contact_proprio, dtype=np.float32)
    _append_frame_value(frames, "contact_proprio_valid_mask", contact_proprio_valid, dtype=np.bool_)
    _append_frame_value(frames, "low_level_contact_input_shifted", contact_action, dtype=np.float32)
    _append_frame_value(frames, "low_level_contact_input_valid_mask", contact_action_valid, dtype=np.bool_)
    _append_frame_value(
        frames,
        "contact_action_world",
        _contact_world_frame(
            contact,
            spec,
            step_idx,
            "contact_action",
            dataset_shift,
            contact_action,
            contact_action_valid,
        ),
        dtype=np.float32,
    )
    _append_frame_value(
        frames,
        "low_level_contact_input_world",
        _contact_world_frame(
            contact,
            spec,
            step_idx,
            "contact_action",
            dataset_shift,
            contact_action,
            contact_action_valid,
        ),
        dtype=np.float32,
    )
    _append_frame_value(
        frames,
        "contact_proprio_world",
        _contact_world_frame(contact, spec, step_idx, "contact_proprio", dataset_shift, contact_proprio, contact_proprio_valid),
        dtype=np.float32,
    )
    for key in (
        "contact_sparse_mask",
        "contact_real_sparse_mask",
        "contact_source_frame_index",
        "contact_has_real",
        "contact_selected_template_indices",
        "contact_num_selected_per_frame",
        "contact_slot_real_mask",
        "contact_slot_supplement_mask",
        "contact_slot_source_frame",
    ):
        value = _contact_source_frame_or_none(contact, spec, step_idx, key)
        if value is not None:
            _append_frame_value(frames, key, value)

    q_input_world = np.asarray(q_input_world, dtype=np.float32).reshape(-1)
    q_input_shifted = np.asarray(q_input_shifted, dtype=np.float32).reshape(-1)
    q_action_shifted = np.asarray(q_action_shifted, dtype=np.float32).reshape(-1)
    q_action_world = np.asarray(q_action_world, dtype=np.float32).reshape(-1)
    q_executed_world = np.asarray(q_executed_world, dtype=np.float32).reshape(-1)
    _append_frame_value(frames, "low_level_q_input_world", q_input_world, dtype=np.float32)
    _append_frame_value(frames, "low_level_q_input_shifted", q_input_shifted, dtype=np.float32)
    _append_frame_value(frames, "low_level_q_action_shifted", q_action_shifted, dtype=np.float32)
    _append_frame_value(frames, "low_level_q_action_world", q_action_world, dtype=np.float32)
    _append_frame_value(frames, "low_level_q_target_shifted", q_action_shifted, dtype=np.float32)
    _append_frame_value(frames, "low_level_q_target_world", q_action_world, dtype=np.float32)
    _append_frame_value(frames, "low_level_q_executed_world", q_executed_world, dtype=np.float32)
    _append_frame_value(frames, "manipulator_proprio", q_input_world, dtype=np.float32)
    _append_frame_value(frames, "manipulator_action", q_action_world, dtype=np.float32)
    _append_frame_value(frames, "manipulator_executed", q_executed_world, dtype=np.float32)
    _append_frame_value(frames, "source_contact_frame_index", int(spec.contact_start) + int(step_idx), dtype=np.int64)
    _append_frame_value(frames, "low_level_policy_step", int(step_idx), dtype=np.int32)

    _append_action_component_fields(frames, q_action_world=q_action_world, q_input_world=q_input_world)
    for key in (
        "object_pose",
        "object_pos",
        "object_quat_wxyz",
        "object_yaw",
        "cube_pose",
        "cube_side_length",
        "cube_height",
        "button_slide",
        "button_site_pos",
        "object_reset_z",
        "lift_delta",
        "lift_success",
        "button_success",
        "key_slide",
        "key_slide_range",
        "target_key_slide",
        "target_key_threshold",
        "target_key_success",
    ):
        if key in obs:
            _append_frame_value(frames, key, obs[key])

    command = dict(info.get("command", {}))
    for key in (
        "pre_qpos",
        "pre_ctrl",
        "requested_qpos",
        "target_qpos",
        "target_ctrl",
        "executed_qpos",
        "executed_qvel",
        "executed_ctrl",
        "object_pose_before",
        "object_pose_after",
        "cube_pose_before",
        "cube_pose_after",
        "robotiq_stick_q_requested",
        "robotiq_stick_q_target",
        "robotiq_stick_q_executed",
    ):
        if key in command:
            _append_frame_value(frames, key, command[key])


def _append_action_component_fields(
    frames: dict[str, list[np.ndarray]],
    *,
    q_action_world: np.ndarray,
    q_input_world: np.ndarray,
) -> None:
    if q_action_world.shape[0] == ROBOTIQ_STICK_LOCAL_DOF:
        _append_frame_value(frames, "action_gripper_pos", q_action_world[0:3], dtype=np.float32)
        _append_frame_value(frames, "action_gripper_euler_xyz", q_action_world[3:6], dtype=np.float32)
        _append_frame_value(frames, "action_gripper_cmd", q_action_world[6:7], dtype=np.float32)
        _append_frame_value(frames, "action_stick_pos", q_action_world[7:10], dtype=np.float32)
        _append_frame_value(frames, "action_stick_euler_xyz", q_action_world[10:13], dtype=np.float32)
        _append_frame_value(frames, "proprio_gripper_pos", q_input_world[0:3], dtype=np.float32)
        _append_frame_value(frames, "proprio_gripper_euler_xyz", q_input_world[3:6], dtype=np.float32)
        _append_frame_value(frames, "proprio_gripper_cmd", q_input_world[6:7], dtype=np.float32)
        _append_frame_value(frames, "proprio_stick_pos", q_input_world[7:10], dtype=np.float32)
        _append_frame_value(frames, "proprio_stick_euler_xyz", q_input_world[10:13], dtype=np.float32)
        return
    if q_action_world.shape[0] >= WUJI_LOCAL_DOF:
        _append_frame_value(frames, "action_ee_pos", q_action_world[0:3], dtype=np.float32)
        _append_frame_value(frames, "action_ee_quat_wxyz", _euler_xyz_to_quat_wxyz(q_action_world[3:6]), dtype=np.float32)
        _append_frame_value(frames, "action_hand_q", q_action_world[6 : 6 + WUJI_JOINT_DIM], dtype=np.float32)
        _append_frame_value(frames, "proprio_ee_pos", q_input_world[0:3], dtype=np.float32)
        _append_frame_value(frames, "proprio_ee_quat_wxyz", _euler_xyz_to_quat_wxyz(q_input_world[3:6]), dtype=np.float32)
        _append_frame_value(frames, "proprio_hand_q", q_input_world[6 : 6 + WUJI_JOINT_DIM], dtype=np.float32)


def _reset_from_retarget(
    env: TabletopManipulationEnv,
    retarget: dict[str, np.ndarray],
    spec: EpisodeSpec,
) -> tuple[dict[str, Any], int | None]:
    seed = None if int(spec.source_trajectory_seed) < 0 else int(spec.source_trajectory_seed)
    env.reset(seed=seed)
    object_suite = str(env.task.metadata.get("task", "")) == "object_suite_pick_place"
    pose_key = "object_pose" if object_suite and "object_pose" in retarget else "cube_pose"
    cube_pose = np.asarray(retarget[pose_key][int(spec.retarget_start)], dtype=np.float64).reshape(7)
    if object_suite:
        _set_cube_state(env, cube_pose, side_length=None, height=None)
        return env.observe(), seed
    side_length = _episode_value(
        retarget,
        "cube_side_length",
        int(spec.retarget_episode_index),
        default=np.asarray(env.task.observe(env.model, env.data, env.scene)["cube_side_length"]).item(),
    )
    height = _episode_value(retarget, "cube_height", int(spec.retarget_episode_index), default=side_length)
    _set_cube_state(env, cube_pose, side_length=side_length, height=height)
    return env.observe(), seed


def _reset_from_robot_model_source(
    env: TabletopManipulationEnv,
    source: dict[str, np.ndarray],
    spec: EpisodeSpec,
) -> tuple[dict[str, Any], int | None]:
    source_idx = int(spec.source_trajectory_index)
    source_seed = _trajectory_value(source, "source_trajectory_seed", source_idx)
    if source_seed is None:
        source_seed = _trajectory_value(source, "trajectory_seed", source_idx)
    seed = None if source_seed is None or int(source_seed) < 0 else int(source_seed)
    env.reset(seed=seed)

    object_suite = str(env.task.metadata.get("task", "")) == "object_suite_pick_place"
    pose_key = "object_pose" if object_suite and "object_pose" in source else "cube_pose"
    cube_pose = _trajectory_array(source, pose_key, source_idx, required=False)
    if cube_pose is None:
        raise ValueError(f"Floating-hand reset source must contain {pose_key}")
    cube_pose_arr = np.asarray(cube_pose, dtype=np.float64)
    cube_pose0 = cube_pose_arr[0] if cube_pose_arr.ndim == 2 else cube_pose_arr
    if object_suite:
        _set_cube_state(env, np.asarray(cube_pose0, dtype=np.float64).reshape(7), side_length=None, height=None)
        return env.observe(), seed
    side_length = _trajectory_value(source, "initial_cube_side_length", source_idx)
    if side_length is None:
        side_length = _trajectory_value(source, "cube_side_length", source_idx)
    height = _trajectory_value(source, "initial_cube_height", source_idx)
    if height is None:
        height = _trajectory_value(source, "cube_height", source_idx)
    _set_cube_state(env, np.asarray(cube_pose0, dtype=np.float64).reshape(7), side_length=side_length, height=height)
    return env.observe(), seed


def _reset_from_contact_pump_bottle(
    env: Any,
    contact: dict[str, np.ndarray],
    spec: EpisodeSpec,
) -> tuple[dict[str, Any], int | None]:
    seed = None if int(spec.source_trajectory_seed) < 0 else int(spec.source_trajectory_seed)
    env.reset(seed=seed)
    start = int(spec.contact_start)
    if "object_pose" in contact:
        object_pose = np.asarray(contact["object_pose"][start], dtype=np.float64).reshape(7)
    elif "initial_object_pose" in contact:
        initial = np.asarray(contact["initial_object_pose"], dtype=np.float64)
        if initial.ndim == 2:
            object_pose = initial[int(spec.contact_episode_index)].reshape(7)
        else:
            object_pose = initial.reshape(7)
    else:
        return env.observe(), seed

    qadr, dadr = env.task._object_qpos_qvel_adr(env.model, env.scene)  # noqa: SLF001
    env.data.qpos[qadr : qadr + 7] = object_pose
    env.data.qvel[dadr : dadr + 6] = 0.0

    if "button_slide" in contact:
        button_qadr, button_dadr = env.task._button_qpos_qvel_adr(env.model, env.scene)  # noqa: SLF001
        lo, hi = env.task._button_range(env.model, env.scene)  # noqa: SLF001
        button_slide = float(np.clip(float(np.asarray(contact["button_slide"][start])), lo, hi))
        env.data.qpos[button_qadr] = button_slide
        env.data.qvel[button_dadr] = 0.0
        actuator_id = env.task._button_actuator_id(env.model)  # noqa: SLF001
        if actuator_id >= 0:
            env.data.ctrl[actuator_id] = button_slide

    env.backend.forward()
    if hasattr(env.task, "ensure_above_table"):
        env.task.ensure_above_table(env.model, env.data, env.scene)
    # The contact payload pose can differ from the preceding env.reset pose,
    # and ensure_above_table may translate it again. Refresh both lift
    # baselines from the actual settled source state used for this rollout.
    if hasattr(env.task, "_reset_object_z"):
        env.task._reset_object_z = float(env.data.qpos[qadr + 2])  # noqa: SLF001
    if hasattr(env.task, "_reset_min_geom_z"):
        env.task._reset_min_geom_z = _pump_bottle_min_geom_z(env)  # noqa: SLF001
    return env.observe(), seed


def _reset_from_contact_keyboard(
    env: Any,
    contact: dict[str, np.ndarray],
    spec: EpisodeSpec,
    *,
    cfg: DictConfig | None = None,
) -> tuple[dict[str, Any], int | None]:
    seed = None if int(spec.source_trajectory_seed) < 0 else int(spec.source_trajectory_seed)
    env.reset(seed=seed)
    start = int(spec.contact_start)
    if "object_pose" in contact:
        object_pose = np.asarray(contact["object_pose"][start], dtype=np.float64).reshape(7)
    elif "initial_object_pose" in contact:
        initial = np.asarray(contact["initial_object_pose"], dtype=np.float64)
        if initial.ndim == 2:
            object_pose = initial[int(spec.contact_episode_index)].reshape(7)
        else:
            object_pose = initial.reshape(7)
    else:
        return env.observe(), seed

    bid = env.task._keyboard_body_id(env.model, env.scene)  # noqa: SLF001
    env.model.body_pos[bid] = object_pose[:3]
    env.model.body_quat[bid] = object_pose[3:7]

    if "key_slide" in contact:
        key_slide = np.asarray(contact["key_slide"][start], dtype=np.float64).reshape(-1)
        key_ids = env.task._key_joint_ids(env.model, env.scene)  # noqa: SLF001
        if key_slide.shape[0] != int(key_ids.shape[0]):
            raise ValueError(
                f"Keyboard key_slide has {key_slide.shape[0]} values, expected {int(key_ids.shape[0])}"
            )
        for value, jid in zip(key_slide, key_ids):
            qadr = int(env.model.jnt_qposadr[int(jid)])
            dadr = int(env.model.jnt_dofadr[int(jid)])
            lo, hi = env.model.jnt_range[int(jid)]
            env.data.qpos[qadr] = float(np.clip(float(value), float(lo), float(hi)))
            env.data.qvel[dadr] = 0.0
    elif "target_key_slide" in contact and "target_key_joint_indices" in contact:
        target_slide = np.asarray(contact["target_key_slide"][start], dtype=np.float64).reshape(-1)
        target_indices = np.asarray(contact["target_key_joint_indices"], dtype=np.int64).reshape(-1)
        key_ids = env.task._key_joint_ids(env.model, env.scene)  # noqa: SLF001
        if target_slide.shape[0] != target_indices.shape[0]:
            raise ValueError(
                f"target_key_slide count {target_slide.shape[0]} does not match "
                f"target_key_joint_indices count {target_indices.shape[0]}"
            )
        for value, key_idx in zip(target_slide, target_indices):
            if int(key_idx) < 0 or int(key_idx) >= int(key_ids.shape[0]):
                raise IndexError(f"target_key_joint_indices contains out-of-range index {int(key_idx)}")
            jid = int(key_ids[int(key_idx)])
            qadr = int(env.model.jnt_qposadr[jid])
            dadr = int(env.model.jnt_dofadr[jid])
            lo, hi = env.model.jnt_range[jid]
            env.data.qpos[qadr] = float(np.clip(float(value), float(lo), float(hi)))
            env.data.qvel[dadr] = 0.0

    _apply_target_key_initial_push_down_offset(env, cfg)
    env.backend.forward()
    return env.observe(), seed


def _apply_target_key_initial_push_down_offset(env: Any, cfg: DictConfig | None) -> None:
    target_ids = env.task._target_key_joint_ids(env.model)  # noqa: SLF001
    spring_defaults = _target_key_spring_defaults(env, target_ids)
    value = (
        0.0
        if cfg is None
        else OmegaConf.select(
            cfg,
            "keyboard.target_key_initial_push_down_offset_fraction",
            default=OmegaConf.select(cfg, "keyboard.target_key_initial_push_down_fraction", default=0.0),
        )
    )
    fraction = float(value)
    if not np.isfinite(fraction) or fraction < 0.0:
        raise ValueError(
            "keyboard.target_key_initial_push_down_offset_fraction must be a non-negative finite value, "
            f"got {value!r}"
        )
    for local_idx, jid_raw in enumerate(target_ids):
        jid = int(jid_raw)
        qadr = int(env.model.jnt_qposadr[jid])
        dadr = int(env.model.jnt_dofadr[jid])
        lo, hi = (float(x) for x in env.model.jnt_range[jid])
        travel = float(hi - lo)
        if travel <= 0.0:
            raise ValueError(f"Target key joint {jid} has invalid range [{lo}, {hi}]")
        spring_ref = float(np.clip(float(spring_defaults[local_idx]) - fraction * travel, lo, hi))
        env.model.qpos_spring[qadr] = spring_ref
        env.data.qpos[qadr] = float(np.clip(float(env.data.qpos[qadr]) + spring_ref, lo, hi))
        env.data.qvel[dadr] = 0.0


def _target_key_spring_defaults(env: Any, target_ids: np.ndarray) -> np.ndarray:
    attr_name = "_online_rollout_target_key_spring_defaults"
    if not hasattr(env, attr_name):
        defaults = []
        for jid_raw in target_ids:
            qadr = int(env.model.jnt_qposadr[int(jid_raw)])
            defaults.append(float(env.model.qpos_spring[qadr]))
        setattr(env, attr_name, np.asarray(defaults, dtype=np.float64))
    defaults = np.asarray(getattr(env, attr_name), dtype=np.float64).reshape(-1)
    if defaults.shape[0] != int(target_ids.shape[0]):
        raise ValueError(
            f"Cached target-key spring defaults have {defaults.shape[0]} values, expected {int(target_ids.shape[0])}"
        )
    return defaults


def _reset_from_contact_task(
    env: Any,
    contact: dict[str, np.ndarray],
    spec: EpisodeSpec,
    *,
    cfg: DictConfig | None = None,
) -> tuple[dict[str, Any], int | None]:
    object_kind = str(getattr(env.scene, "object_kind", ""))
    if object_kind == "pump_bottle":
        return _reset_from_contact_pump_bottle(env, contact, spec)
    if object_kind == "mini_keyboard":
        return _reset_from_contact_keyboard(env, contact, spec, cfg=cfg)
    raise ValueError(f"Unsupported armless online rollout object_kind={object_kind!r}")


def _initial_robot_model_q_from_source(
    source: dict[str, np.ndarray],
    spec: EpisodeSpec,
    *,
    expected_dim: int,
) -> np.ndarray:
    source_idx = int(spec.source_trajectory_index)
    q_traj = _trajectory_array(source, "manipulator_action", source_idx)
    q_arr = np.asarray(q_traj, dtype=np.float32)
    q0 = q_arr[0] if q_arr.ndim == 2 else q_arr
    return _validate_robot_model_q(q0, expected_dim=expected_dim, label=f"robot_model_q_source[{source_idx}]")


def _validate_robot_model_q(q: np.ndarray, *, expected_dim: int, label: str) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float32).reshape(-1)
    if arr.shape != (int(expected_dim),):
        raise ValueError(f"{label} must have shape ({int(expected_dim)},), got {arr.shape}")
    if not bool(np.isfinite(arr).all()):
        raise ValueError(f"{label} contains non-finite values")
    return arr.astype(np.float32, copy=True)


def _unwrap_robot_model_q_rotation(q_world: np.ndarray, *, reference: np.ndarray) -> np.ndarray:
    return unwrap_robot_model_q_rotation(q_world, reference=reference)


def _set_env_robot_model_q(env: TabletopManipulationEnv, q_world: np.ndarray) -> None:
    right = env.robot.side("right")
    q = _validate_robot_model_q(
        q_world,
        expected_dim=int(right.arm_dim + right.effector_ctrl_dim),
        label="robot_model_q",
    )
    q = robot_model_q_to_native(env, q, reset=True)
    arm_q = np.clip(q[: right.arm_dim], right.arm_q_range[:, 0], right.arm_q_range[:, 1])
    hand_q = np.clip(
        q[right.arm_dim : right.arm_dim + right.effector_ctrl_dim],
        right.hand_q_range[:, 0],
        right.hand_q_range[:, 1],
    )
    env.data.qpos[right.arm_qpos_indices] = arm_q
    env.data.qvel[right.arm_qvel_indices] = 0.0
    if right.effector_qpos_dim:
        env.data.qpos[right.hand_qpos_indices] = hand_q
        env.data.qvel[right.hand_qvel_indices] = 0.0
    env.robot.set_side_ctrl(env.data, "right", arm_q=arm_q, hand_q=hand_q)
    env.backend.forward()


def _current_robot_model_q_from_env(env: TabletopManipulationEnv) -> np.ndarray:
    right = env.robot.side("right")
    arm_q = np.asarray(env.data.qpos[right.arm_qpos_indices], dtype=np.float32).reshape(-1)
    hand_q = np.asarray(env.data.qpos[right.hand_qpos_indices], dtype=np.float32).reshape(-1)
    return robot_model_q_to_world(env, np.concatenate([arm_q, hand_q], axis=0).astype(np.float32, copy=False))


def _env_action_from_robot_model_q(env: TabletopManipulationEnv, q_world: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    right = env.robot.side("right")
    q = _validate_robot_model_q(
        q_world,
        expected_dim=int(right.arm_dim + right.effector_ctrl_dim),
        label="robot_model_q_action",
    )
    q = robot_model_q_to_native(env, q)
    return {
        "right": {
            "arm_q": q[: right.arm_dim].astype(np.float32),
            "hand_q": q[right.arm_dim : right.arm_dim + right.effector_ctrl_dim].astype(np.float32),
        }
    }


def _set_env_wuji_q(env: TabletopManipulationEnv, q_world: np.ndarray) -> None:
    q = np.asarray(q_world, dtype=np.float32).reshape(-1)
    if q.shape[0] < WUJI_LOCAL_DOF:
        raise ValueError(f"Expected Wuji q dim >= {WUJI_LOCAL_DOF}, got {q.shape}")
    right = env.robot.side("right")
    if env.robot.has_group("left"):
        env.robot.apply_home(env.data, sides=("left",), set_qpos=True)
    hand_q = np.clip(q[6 : 6 + WUJI_JOINT_DIM], right.hand_ctrl_range[:, 0], right.hand_ctrl_range[:, 1])
    command = env._apply_side_action(  # noqa: SLF001
        "right",
        {
            "ee_pos": q[:3],
            "ee_quat_wxyz": _euler_xyz_to_quat_wxyz(q[3:6]),
            "hand_q": hand_q,
        },
    )
    env.data.qpos[right.arm_qpos_indices] = command["arm_q_target"]
    env.data.qpos[right.hand_qpos_indices] = hand_q
    env.data.qvel[right.qvel_indices] = 0.0
    env.robot.set_side_ctrl(env.data, "right", arm_q=command["arm_q_target"], hand_q=hand_q)
    env.backend.forward()


def _current_wuji_q_from_env(env: TabletopManipulationEnv) -> np.ndarray:
    right = env.robot.side("right")
    pos, quat = env.robot.wrist_pose(env.data, "right")
    hand_q = np.asarray(env.data.qpos[right.hand_qpos_indices], dtype=np.float32).reshape(-1)[:WUJI_JOINT_DIM]
    return np.concatenate(
        [
            np.asarray(pos, dtype=np.float32).reshape(3),
            _quat_wxyz_to_euler_xyz(np.asarray(quat, dtype=np.float32).reshape(4)),
            hand_q,
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _env_action_from_wuji_q(q_world: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    q = np.asarray(q_world, dtype=np.float32).reshape(-1)
    return {
        "right": {
            "ee_pos": q[:3].astype(np.float32),
            "ee_quat_wxyz": _euler_xyz_to_quat_wxyz(q[3:6]).astype(np.float32),
            "hand_q": q[6 : 6 + WUJI_JOINT_DIM].astype(np.float32),
        }
    }


def _contact_frame(
    contact: dict[str, np.ndarray],
    spec: EpisodeSpec,
    step_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = int(spec.contact_start) + int(step_idx)
    cloud = np.asarray(contact["contact_action"][idx], dtype=np.float32)
    mask = np.asarray(contact["contact_action_valid_mask"][idx], dtype=np.bool_)
    return cloud, mask


def _latent_contact_frame(
    contact: dict[str, np.ndarray],
    spec: EpisodeSpec,
    step_idx: int,
) -> np.ndarray:
    idx = int(spec.contact_start) + int(step_idx)
    return np.asarray(contact["latent_action"][idx], dtype=np.float32).reshape(-1)


def _predict_low_level_q_from_saved_latent(
    *,
    runtime: OnlineLowLevelRuntime,
    latent_action: np.ndarray,
    q_prev_shifted: np.ndarray,
    seed: int,
) -> np.ndarray:
    if runtime.non_contact_enabled:
        raise ValueError("Saved-latent decoding does not support the optional non-contact stream")
    latent = np.asarray(latent_action, dtype=np.float32).reshape(-1)
    if int(latent.shape[0]) != int(runtime.contact_feature_dim):
        raise ValueError(
            f"Saved contact latent has dim={latent.shape[0]}, but the low-level decoder expects "
            f"{runtime.contact_feature_dim}"
        )
    q_prev = np.asarray(q_prev_shifted, dtype=np.float32)
    if q_prev.ndim != 2 or int(q_prev.shape[0]) != 1:
        raise ValueError(f"Saved-latent decoding expects q_prev shape (1,D), got {q_prev.shape}")
    q_pred = predict_low_level_q_from_contact_latent(
        runtime=runtime,
        contact_feature=latent,
        q_prev=q_prev[0],
        seed=int(seed),
    )
    return np.asarray(q_pred, dtype=np.float32).reshape(1, -1)


def _build_non_contact_input(
    *,
    cfg: DictConfig,
    env: TabletopManipulationEnv,
    runtime: OnlineLowLevelRuntime,
    obs: dict[str, Any],
    q_prev_shifted: np.ndarray,
    contact_cloud_shifted: np.ndarray,
    contact_valid_mask: np.ndarray,
    dataset_shift: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, NonContactStats]:
    target_count = int(runtime.non_contact_input_points)
    if target_count <= 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.bool_),
            NonContactStats(raw_count=0, after_contact_exclusion=0, valid_count=0, selected_count=0),
        )

    raw = _reconstruct_non_contact_candidates_world(env, obs, cfg)
    raw_count = int(raw.shape[0])
    contact_exclusion_radius = OmegaConf.select(
        cfg,
        "online_rollout.non_contact.contact_exclusion_radius",
        default=None,
    )
    exclusion_radius = (
        float(runtime.non_contact_sdf_clearance)
        if contact_exclusion_radius is None or str(contact_exclusion_radius).lower() in {"none", "null"}
        else float(contact_exclusion_radius)
    )
    raw = _remove_points_near_contacts(
        raw,
        contact_cloud_shifted=contact_cloud_shifted,
        contact_valid_mask=contact_valid_mask,
        dataset_shift=dataset_shift,
        radius=float(exclusion_radius),
    )
    after_contact = int(raw.shape[0])
    max_raw = int(OmegaConf.select(cfg, "online_rollout.non_contact.max_raw_candidates", default=8192))
    if max_raw > 0 and int(raw.shape[0]) > max_raw:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(int(raw.shape[0]), size=max_raw, replace=False)
        raw = raw[np.sort(idx)]

    if raw.size == 0:
        cloud, valid = _empty_non_contact(target_count)
        return cloud, valid, NonContactStats(raw_count=raw_count, after_contact_exclusion=after_contact, valid_count=0, selected_count=0)

    shifted = raw.astype(np.float32, copy=True)
    shifted += np.asarray(dataset_shift, dtype=np.float32).reshape(1, 3)
    valid = _filter_non_contact_by_sdf(
        runtime=runtime,
        candidates_shifted=shifted,
        q_prev_shifted=np.asarray(q_prev_shifted, dtype=np.float32),
        sdf_batch_size=int(OmegaConf.select(cfg, "online_rollout.non_contact.sdf_batch_size", default=8192)),
    )
    candidates = shifted[valid]
    valid_count = int(candidates.shape[0])
    if valid_count == 0:
        cloud, mask = _empty_non_contact(target_count)
        return cloud, mask, NonContactStats(raw_count=raw_count, after_contact_exclusion=after_contact, valid_count=0, selected_count=0)
    if valid_count > target_count:
        indices = farthest_point_sample_indices(candidates, target_count)
        candidates = candidates[indices]
    cloud, mask = _pad_non_contact(candidates, target_count)
    return (
        cloud,
        mask,
        NonContactStats(
            raw_count=raw_count,
            after_contact_exclusion=after_contact,
            valid_count=valid_count,
            selected_count=int(np.count_nonzero(mask)),
        ),
    )


def _reconstruct_non_contact_candidates_world(
    env: TabletopManipulationEnv,
    obs: dict[str, Any],
    cfg: DictConfig,
) -> np.ndarray:
    if "sensor_data" not in obs or "sensor_param" not in obs:
        raise KeyError("non-contact inference requires camera observations; obs is missing sensor_data/sensor_param")
    sensor_data = obs["sensor_data"]
    sensor_param = obs["sensor_param"]
    camera_names = tuple(str(x) for x in getattr(env, "policy_camera_names", ("ext_cam_0", "ext_cam_1")))
    bounds = np.asarray(env.pointcloud_workspace_bounds, dtype=np.float32).reshape(3, 2)
    exclude_table = bool(OmegaConf.select(cfg, "perception.exclude_table_pointcloud", default=False))
    exclude_cube = bool(OmegaConf.select(cfg, "online_rollout.non_contact.exclude_cube", default=False))
    points: list[np.ndarray] = []
    for camera_name in camera_names:
        if camera_name not in sensor_data or camera_name not in sensor_param:
            continue
        cam_data = sensor_data[camera_name]
        depth = np.asarray(cam_data["depth"], dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 1.0e-5)
        if "robot_mask" in cam_data:
            valid &= ~np.asarray(cam_data["robot_mask"], dtype=np.bool_)
        if exclude_table and "table_mask" in cam_data:
            valid &= ~np.asarray(cam_data["table_mask"], dtype=np.bool_)
        if exclude_cube and "cube_mask" in cam_data:
            valid &= ~np.asarray(cam_data["cube_mask"], dtype=np.bool_)
        valid_flat = np.flatnonzero(valid.reshape(-1))
        if valid_flat.size == 0:
            continue
        ys, xs = np.divmod(valid_flat, depth.shape[1])
        z = depth.reshape(-1)[valid_flat]
        xyz = deproject_pixels_cv(
            xs.astype(np.float32),
            ys.astype(np.float32),
            z.astype(np.float32),
            sensor_param[camera_name]["intrinsic_cv"],
            sensor_param[camera_name]["extrinsic_cv"],
        )
        in_bounds = np.all((xyz >= bounds[:, 0]) & (xyz <= bounds[:, 1]), axis=1)
        if np.any(in_bounds):
            points.append(xyz[in_bounds].astype(np.float32, copy=False))
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(points, axis=0).astype(np.float32, copy=False)


def _remove_points_near_contacts(
    points_world: np.ndarray,
    *,
    contact_cloud_shifted: np.ndarray,
    contact_valid_mask: np.ndarray,
    dataset_shift: np.ndarray,
    radius: float,
) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    if points.size == 0 or float(radius) <= 0.0:
        return points
    contacts = np.asarray(contact_cloud_shifted, dtype=np.float32)[np.asarray(contact_valid_mask, dtype=np.bool_), :3]
    if contacts.size == 0:
        return points
    contacts = contacts - np.asarray(dataset_shift, dtype=np.float32).reshape(1, 3)
    keep = np.ones((int(points.shape[0]),), dtype=np.bool_)
    radius2 = float(radius) * float(radius)
    for start in range(0, int(points.shape[0]), 65536):
        end = min(start + 65536, int(points.shape[0]))
        diff = points[start:end, None, :] - contacts[None, :, :]
        near = np.any(np.sum(diff * diff, axis=-1) <= radius2, axis=1)
        keep[start:end] = ~near
    return points[keep]


@torch.inference_mode()
def _filter_non_contact_by_sdf(
    *,
    runtime: OnlineLowLevelRuntime,
    candidates_shifted: np.ndarray,
    q_prev_shifted: np.ndarray,
    sdf_batch_size: int,
) -> np.ndarray:
    if runtime.collision_model is None:
        raise RuntimeError("non-contact inference requires a collision primitive model")
    candidates = np.asarray(candidates_shifted, dtype=np.float32).reshape(-1, 3)
    if candidates.size == 0:
        return np.zeros((0,), dtype=np.bool_)
    q = np.asarray(q_prev_shifted, dtype=np.float32).reshape(1, -1)[:, : runtime.local_dof]
    batch_size = max(1, int(sdf_batch_size))
    sdf_chunks: list[np.ndarray] = []
    for start in range(0, int(candidates.shape[0]), batch_size):
        end = min(start + batch_size, int(candidates.shape[0]))
        sdf = runtime.collision_model.query_sdf(
            torch.from_numpy(q).to(device=runtime.device, dtype=torch.float32),
            torch.from_numpy(candidates[start:end]).to(device=runtime.device, dtype=torch.float32),
        )
        sdf_chunks.append(sdf.reshape(-1).detach().cpu().numpy().astype(np.float32, copy=False))
    sdf_np = np.concatenate(sdf_chunks, axis=0)
    center_dist = np.linalg.norm(candidates - q.reshape(-1)[:3][None, :], axis=1)
    clearance = float(runtime.non_contact_sdf_clearance)
    radius = float(runtime.non_contact_sampling_radius)
    near_embodiment = (sdf_np <= radius) | (center_dist <= radius)
    return np.isfinite(sdf_np) & (sdf_np >= clearance) & near_embodiment


def _empty_non_contact(target_count: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((int(target_count), 3), dtype=np.float32),
        np.zeros((int(target_count),), dtype=np.bool_),
    )


def _pad_non_contact(points: np.ndarray, target_count: int) -> tuple[np.ndarray, np.ndarray]:
    target = int(target_count)
    cloud = np.zeros((target, 3), dtype=np.float32)
    mask = np.zeros((target,), dtype=np.bool_)
    count = min(target, int(points.shape[0]))
    if count > 0:
        cloud[:count] = np.asarray(points[:count], dtype=np.float32)
        mask[:count] = True
    return cloud, mask


@torch.inference_mode()
def _predict_low_level_q_batch(
    *,
    runtime: OnlineLowLevelRuntime,
    contact_cloud_shifted: np.ndarray,
    contact_valid_mask: np.ndarray,
    q_prev_shifted: np.ndarray,
    seed: int,
    non_contact_cloud_shifted: np.ndarray | None = None,
    non_contact_valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    device = runtime.device
    contact_np = np.asarray(contact_cloud_shifted, dtype=np.float32)
    mask_np = np.asarray(contact_valid_mask, dtype=np.bool_)
    q_prev_np = np.asarray(q_prev_shifted, dtype=np.float32)
    if contact_np.ndim != 3 or int(contact_np.shape[-1]) < 6:
        raise ValueError(f"Expected contact cloud shape (B,N,>=6), got {contact_np.shape}")
    if mask_np.shape != contact_np.shape[:2]:
        raise ValueError(f"contact_valid_mask shape {mask_np.shape} does not match {contact_np.shape[:2]}")
    if q_prev_np.ndim != 2 or int(q_prev_np.shape[1]) < runtime.local_dof:
        raise ValueError(f"Expected q_prev shape (B,{runtime.local_dof}), got {q_prev_np.shape}")

    q_prev_t = torch.from_numpy(q_prev_np[:, : runtime.local_dof]).to(device=device, dtype=torch.float32)
    batch: dict[str, torch.Tensor] = {
        "contact_cloud": torch.from_numpy(contact_np[..., :6]).to(device=device, dtype=torch.float32),
        "contact_valid_mask": torch.from_numpy(mask_np).to(device=device, dtype=torch.bool),
        "q1_padded": q_prev_t,
        "action_valid_mask": runtime.action_valid_mask.view(1, -1).expand(q_prev_t.shape[0], -1),
        "robot_index": torch.full((q_prev_t.shape[0],), int(runtime.robot_index), dtype=torch.long, device=device),
        "local_dof": torch.full((q_prev_t.shape[0],), int(runtime.local_dof), dtype=torch.long, device=device),
    }
    if runtime.encoder_backend == "paired_transformer":
        hand_points, hand_normals = runtime.robot_model.transform_hand_fps_template_batch(q_prev_t)
        batch["hand_points_world"] = hand_points.to(device=device, dtype=torch.float32)
        batch["hand_normals_world"] = hand_normals.to(device=device, dtype=torch.float32)

    if runtime.non_contact_enabled:
        if non_contact_cloud_shifted is None or non_contact_valid_mask is None:
            raise ValueError("runtime requires non-contact inputs, but none were provided")
        nc_np = np.asarray(non_contact_cloud_shifted, dtype=np.float32)
        nc_mask_np = np.asarray(non_contact_valid_mask, dtype=np.bool_)
        if nc_np.ndim != 3 or int(nc_np.shape[-1]) < 3:
            raise ValueError(f"Expected non_contact_cloud shape (B,N,>=3), got {nc_np.shape}")
        if nc_mask_np.shape != nc_np.shape[:2]:
            raise ValueError(f"non_contact_valid_mask shape {nc_mask_np.shape} does not match {nc_np.shape[:2]}")
        batch["non_contact_cloud"] = torch.from_numpy(nc_np[..., :3]).to(device=device, dtype=torch.float32)
        batch["non_contact_valid_mask"] = torch.from_numpy(nc_mask_np).to(device=device, dtype=torch.bool)
        batch["non_contact_blocker_mask"] = torch.zeros_like(batch["non_contact_valid_mask"], dtype=torch.bool)
    else:
        batch["non_contact_cloud"] = torch.zeros((q_prev_t.shape[0], 0, 3), dtype=torch.float32, device=device)
        batch["non_contact_valid_mask"] = torch.zeros((q_prev_t.shape[0], 0), dtype=torch.bool, device=device)
        batch["non_contact_blocker_mask"] = torch.zeros((q_prev_t.shape[0], 0), dtype=torch.bool, device=device)

    with torch.random.fork_rng(devices=_rng_devices(device)):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        out = runtime.low_level_policy.predict_action(batch)
    action = out["action_pred" if "action_pred" in out else "action"]
    q_pred = action[:, : runtime.local_dof].detach().cpu().numpy().astype(np.float32, copy=False)
    return _apply_fixed_zero_constraints(runtime, q_pred)


def _save_online_training_payload(
    output_path: Path,
    *,
    training_records: list[tuple[OnlineTrainingEpisode, Any]],
    cfg: DictConfig,
    env: Any,
    contact_payload: dict[str, Any],
    contact_path: Path,
    high_level_config_path: Path,
    runtime: OnlineLowLevelRuntime,
    backend: str,
    dt: float,
    dataset_shift: np.ndarray,
    initial_pose_mode: str,
    initial_pose_path: Path | None,
    initial_q: np.ndarray | None,
    proprio_source: str,
    controller_steps_per_prediction: int,
    action_format: str,
    translation_groups: tuple[tuple[int, int, int], ...] | None = None,
    retarget_path: Path | None = None,
    robot_model_q_source_path: Path | None = None,
    object_reset_source_path: Path | None = None,
    source_metadata: dict[str, Any] | None = None,
    initial_poses_by_source_index: dict[int, StartPose] | None = None,
) -> Path:
    if not training_records:
        raise RuntimeError("No online rollout frames were recorded for online_rollout.save_training_pt")
    lengths = [_training_episode_frame_count(record) for record, _ in training_records]
    if any(length <= 0 for length in lengths):
        raise RuntimeError(f"Invalid empty online training episode lengths: {lengths}")
    episode_ends = make_episode_ends(lengths)
    obs_point_cloud = _concat_training_key(training_records, "obs_point_cloud", required=True).astype(
        np.float32,
        copy=False,
    )
    q_action_world = _concat_training_key(training_records, "low_level_q_action_world", required=True).astype(
        np.float32,
        copy=False,
    )
    if str(action_format) == "wuji_single_manipulator":
        action_ee_pos = q_action_world[:, 0:3]
        action_ee_quat = np.stack([_euler_xyz_to_quat_wxyz(q[3:6]) for q in q_action_world], axis=0)
        action_hand_q = q_action_world[:, 6 : 6 + WUJI_JOINT_DIM]
        action = concat_single_manipulator_action(action_ee_pos, action_ee_quat, action_hand_q)
        action_space = single_manipulator_action_space(
            joint_key="action_hand_q",
            joint_dim=int(action_hand_q.shape[1]),
        )
    elif str(action_format) == "robotiq_stick_robot_model_q":
        action = q_action_world
        action_space = _robotiq_stick_action_space(runtime)
    elif str(action_format) == "robot_model_q":
        action = q_action_world
        action_space = _robot_model_q_action_space(runtime, env)
    else:
        raise ValueError(f"Unsupported online training action_format={action_format!r}")

    contact_meta = dict(contact_payload.get("meta", {}))
    source_indices = [int(record.spec.source_trajectory_index) for record, _ in training_records]
    source = {
        "kind": "online_low_level_rollout",
        "script": "experiment/rollout_gt_contacts.py",
        "contact_source_pt": str(contact_path),
        "output_pt": str(output_path),
        "source_indices": source_indices,
    }
    if retarget_path is not None:
        source["retargeted_wuji_pt"] = str(retarget_path)
    if robot_model_q_source_path is not None:
        source["robot_model_q_source_pt"] = str(robot_model_q_source_path)
    if object_reset_source_path is not None:
        source["object_reset_source_pt"] = str(object_reset_source_path)

    object_kind = str(getattr(env.scene, "object_kind", contact_meta.get("object_kind", "")) or "")
    meta_extra = _online_training_meta_extra(
        cfg=cfg,
        env=env,
        contact_meta=contact_meta,
        high_level_config_path=high_level_config_path,
        runtime=runtime,
        backend=backend,
        dt=dt,
        dataset_shift=dataset_shift,
        initial_pose_mode=initial_pose_mode,
        initial_pose_path=initial_pose_path,
        initial_q=initial_q,
        proprio_source=proprio_source,
        controller_steps_per_prediction=controller_steps_per_prediction,
        action_format=action_format,
        object_kind=object_kind,
        translation_groups=translation_groups,
        source_metadata=source_metadata,
    )
    if initial_poses_by_source_index:
        add_episode_start_pose_metadata(
            meta_extra,
            [initial_poses_by_source_index[index] for index in source_indices],
        )
        meta_extra = {key: _metadata_value(value) for key, value in meta_extra.items()}
    meta = build_common_meta(
        source=source,
        task=str(OmegaConf.select(cfg, "task.name", default=contact_meta.get("task", ""))),
        embodiment=str(cfg.env.embodiment),
        dt=float(dt),
        fps=1.0 / float(dt) if float(dt) > 0.0 else None,
        observation_point_count=int(obs_point_cloud.shape[1]),
        observation_frame=str(contact_meta.get("object_pointcloud_frame", "world")),
        observation_dtype=str(obs_point_cloud.dtype),
        action_space=action_space,
        extra=meta_extra,
    )

    source_episode_meta = list(contact_payload.get("episode_meta", []))
    episode_meta: list[dict[str, Any]] = []
    for episode_idx, (record, result) in enumerate(training_records):
        spec = record.spec
        original = (
            dict(source_episode_meta[int(spec.contact_episode_index)])
            if int(spec.contact_episode_index) < len(source_episode_meta)
            else {}
        )
        episode_meta.append(
            {
                **original,
                "episode_index": int(episode_idx),
                "contact_episode_index": int(spec.contact_episode_index),
                "source_trajectory_index": int(spec.source_trajectory_index),
                "source_trajectory_seed": (
                    int(record.source_seed)
                    if record.source_seed is not None
                    else int(spec.source_trajectory_seed)
                ),
                "source_attempt_index": int(spec.source_attempt_index),
                "num_frames": int(lengths[episode_idx]),
                "completed_controller_steps": int(getattr(result, "completed_steps", 0)),
                "success": bool(getattr(result, "success", False)),
                "low_level_manipulator": str(runtime.low_level_manipulator),
                "initial_pose_mode": str(initial_pose_mode),
                "initial_pose_path": "" if initial_pose_path is None else str(initial_pose_path),
                "proprio_source": str(proprio_source),
                "controller_steps_per_prediction": int(controller_steps_per_prediction),
            }
        )
        if initial_poses_by_source_index:
            mode, pose_path, q = initial_poses_by_source_index[int(spec.source_trajectory_index)]
            episode_meta[-1].update(
                initial_pose_mode=mode,
                initial_pose_path=str(pose_path),
                initial_q_world=q.tolist(),
            )

    extra_data = _online_training_extra_data(training_records, action_format=action_format, initial_q=initial_q)
    if initial_poses_by_source_index:
        extra_data[_initial_q_extra_key(action_format)] = np.stack(
            [initial_poses_by_source_index[index][2] for index in source_indices]
        ).astype(np.float32)
    payload = build_training_payload(
        obs_point_cloud=obs_point_cloud,
        action=action,
        episode_ends=episode_ends,
        episode_meta=episode_meta,
        meta=meta,
        extra_data=extra_data,
    )
    return save_high_level_payload(payload, output_path, validate=True)


def _online_training_meta_extra(
    *,
    cfg: DictConfig,
    env: Any,
    contact_meta: dict[str, Any],
    high_level_config_path: Path,
    runtime: OnlineLowLevelRuntime,
    backend: str,
    dt: float,
    dataset_shift: np.ndarray,
    initial_pose_mode: str,
    initial_pose_path: Path | None,
    initial_q: np.ndarray | None,
    proprio_source: str,
    controller_steps_per_prediction: int,
    action_format: str,
    object_kind: str,
    translation_groups: tuple[tuple[int, int, int], ...] | None,
    source_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "version": 1,
        "protocol": "high_level_training_v2",
        "dataset_name": Path(str(OmegaConf.select(cfg, "online_rollout.save_training_pt", default="online_rollout"))).stem,
        "object_kind": str(object_kind),
        "backend": str(backend),
        "mode": str(cfg.env.mode),
        "frame_skip": int(env.backend.frame_skip),
        "fps": 1.0 / float(dt) if float(dt) > 0.0 else None,
        "dt": float(dt),
        "low_level_config_path": str(runtime.low_level_config_path),
        "low_level_checkpoint_path": str(runtime.low_level_checkpoint_path),
        "low_level_manipulator": str(runtime.low_level_manipulator),
        "low_level_backend": str(runtime.encoder_backend),
        "low_level_action_format": str(action_format),
        "low_level_contact_input_frame": "shifted_low_level_training_frame",
        "dataset_xyz_translation": np.asarray(dataset_shift, dtype=np.float32).tolist(),
        "dataset_xyz_translation_applied": True,
        "dataset_xyz_translation_applied_to": "low_level_contact_and_q_proprio",
        "high_level_config_path": str(high_level_config_path),
        "initial_pose_mode": str(initial_pose_mode),
        "initial_pose_path": "" if initial_pose_path is None else str(initial_pose_path),
        "proprio_source": str(proprio_source),
        "controller_steps_per_prediction": int(controller_steps_per_prediction),
        "non_contact_enabled": bool(runtime.non_contact_enabled),
        "non_contact_input_points": int(runtime.non_contact_input_points),
        "target_pointcloud_num": int(env.target_pointcloud_num),
        "exclude_robot_pointcloud": bool(env.exclude_robot_pointcloud),
        "exclude_table_pointcloud": bool(env.exclude_table_pointcloud),
        "pointcloud_workspace_bounds": np.asarray(env.pointcloud_workspace_bounds, dtype=np.float32).tolist(),
        "camera_names": [str(x) for x in getattr(env, "camera_names", ())],
        "policy_camera_names": [str(x) for x in getattr(env, "policy_camera_names", ())],
        "qpos_names": qpos_names(env.model),
        "ctrl_names": ctrl_names(env.model),
        "contact_source_action_space": contact_meta.get("action_space", {}),
        "contact_source_task": contact_meta.get("task"),
        "contact_mode": contact_meta.get("contact_mode"),
        "contact_fill_policy": contact_meta.get("contact_fill_policy"),
    }
    if initial_q is not None:
        extra["initial_low_level_q_world"] = np.asarray(initial_q, dtype=np.float32).tolist()
    if translation_groups is not None:
        extra["low_level_q_translation_groups"] = [list(group) for group in translation_groups]
    for controller_key in (
        "arm_kp_scale",
        "arm_kv_scale",
        "arm_force_scale",
        "hand_kp_scale",
        "hand_kv_scale",
        "hand_force_scale",
        "armless_root_max_translation_step",
        "armless_root_max_rotation_step",
    ):
        value = OmegaConf.select(cfg, f"controller.{controller_key}", default=None)
        if value is not None:
            extra[f"controller_{controller_key}"] = float(value)
    for group_key in ("base_xyz", "base_rpy", "fingers"):
        if OmegaConf.select(cfg, f"controller.{group_key}", default=None) is None:
            continue
        for field_key in ("kp", "kv", "max_effort"):
            value = OmegaConf.select(cfg, f"controller.{group_key}.{field_key}", default=None)
            if value is not None:
                extra[f"controller_{group_key}_{field_key}"] = _controller_meta_value(value)
    _copy_cube_geometry_meta(extra, cfg=cfg, source_metadata=source_metadata, contact_meta=contact_meta)
    for key in _CONTACT_META_COPY_KEYS:
        if key in contact_meta and key not in extra:
            extra[key] = contact_meta[key]
    if str(OmegaConf.select(cfg, "task.name", default="")) == "object_suite_pick_place":
        extra["object_manifest"] = str(optional_project_path(cfg.objects.manifest))
        extra["bin_center_xy"] = [float(value) for value in cfg.bin.center_xy]
        extra["bin_inner_size"] = [float(value) for value in cfg.bin.inner_size]
        extra["bin_wall_height"] = float(cfg.bin.wall_height)
    if str(object_kind) == "mini_keyboard":
        extra["keyboard_support_box_height"] = float(
            getattr(env.task, "support_box_height", contact_meta.get("keyboard_support_box_height", 0.0))
        )
        extra["target_key_joint_names"] = [str(x) for x in getattr(env.task, "target_key_joint_names", ())]
    if str(object_kind) == "pump_bottle":
        for attr, key in (
            ("button_success_slide", "pump_bottle_button_success_slide"),
            ("lift_success_delta", "pump_bottle_lift_success_delta"),
        ):
            if hasattr(env.task, attr):
                extra[key] = float(getattr(env.task, attr))
        extra["pump_bottle_lift_success_metric"] = _pump_bottle_lift_success_metric_from_cfg(
            cfg,
            section="online_rollout",
            default="max_lift_delta",
        )
    return extra


def _controller_meta_value(value: Any) -> float | str:
    if isinstance(value, str):
        return value
    return float(value)


def _copy_cube_geometry_meta(
    extra: dict[str, Any],
    *,
    cfg: DictConfig,
    source_metadata: dict[str, Any] | None,
    contact_meta: dict[str, Any],
) -> None:
    sources = [source_metadata or {}, contact_meta]
    for key in (
        "cube_x_range",
        "cube_y_range",
        "cube_side_length_range",
        "cube_height_ratio",
        "cube_height_range",
        "cube_yaw_range_deg",
    ):
        for source in sources:
            if key in source:
                extra[key] = _metadata_value(source[key])
                break
    if "cube_side_length_range" not in extra and OmegaConf.select(cfg, "cube.side_length_range", default=None) is not None:
        extra["cube_side_length_range"] = _metadata_value(OmegaConf.select(cfg, "cube.side_length_range"))
    if "cube_height_ratio" not in extra and OmegaConf.select(cfg, "cube.height_ratio", default=None) is not None:
        extra["cube_height_ratio"] = float(OmegaConf.select(cfg, "cube.height_ratio"))
    if (
        "cube_height_range" not in extra
        and "cube_side_length_range" in extra
        and "cube_height_ratio" in extra
    ):
        side = np.asarray(extra["cube_side_length_range"], dtype=np.float32)
        extra["cube_height_range"] = _metadata_value(side * np.float32(float(extra["cube_height_ratio"])))


def _metadata_value(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, (list, tuple, dict, str, int, float, bool)):
        return value
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    return arr.tolist()


_CONTACT_META_COPY_KEYS = (
    "dataset_name",
    "backend",
    "mode",
    "frame_skip",
    "object_asset_metadata",
    "object_asset_name",
    "object_asset_path",
    "object_manifest",
    "object_kind",
    "object_pointcloud_frame",
    "object_pose_quat_order",
    "pump_bottle_xml_path",
    "pump_bottle_mesh_dir",
    "pump_bottle_mesh_scale",
    "pump_bottle_x_range",
    "pump_bottle_y_range",
    "pump_bottle_yaw_range_deg",
    "pump_bottle_button_success_slide",
    "pump_bottle_lift_success_delta",
    "pump_bottle_lift_success_metric",
    "mini_keyboard_xml_path",
    "keyboard_support_box_height",
    "keyboard_default_pose",
    "keyboard_x_range",
    "keyboard_y_range",
    "keyboard_yaw_range_deg",
    "keyboard_key_success_fraction",
    "target_key_joint_names",
    "target_key_joint_indices",
    "pose_target_key_joint_names",
    "pose_target_key_joint_indices",
    "contact_target_key_joint_names",
    "contact_target_key_joint_indices",
    "contact_keyboard_joint_names",
    "pointcloud_workspace_bounds",
    "contact_sparse_capacity",
    "contact_slot_labels",
    "contact_slot_order",
    "contact_template_indices",
    "preset_contact_indices",
    "preset_contact_path",
    "surface_template_hash",
    "surface_template_size",
    "surface_template_path",
    "surface_template_cache_path",
)


def _online_training_extra_data(
    training_records: list[tuple[OnlineTrainingEpisode, Any]],
    *,
    action_format: str,
    initial_q: np.ndarray | None,
) -> dict[str, Any]:
    reserved = {"obs_point_cloud"}
    extra: dict[str, Any] = {}
    keys = sorted({key for record, _ in training_records for key in record.frames.keys()})
    for key in keys:
        if key in reserved:
            continue
        value = _concat_training_key(training_records, key, required=False)
        if value is not None:
            extra[key] = value
    specs = [record.spec for record, _ in training_records]
    results = [result for _, result in training_records]
    extra["source_trajectory_index"] = np.asarray([spec.source_trajectory_index for spec in specs], dtype=np.int32)
    extra["source_trajectory_seed"] = np.asarray([spec.source_trajectory_seed for spec in specs], dtype=np.int32)
    extra["source_attempt_index"] = np.asarray([spec.source_attempt_index for spec in specs], dtype=np.int32)
    extra["contact_episode_index"] = np.asarray([spec.contact_episode_index for spec in specs], dtype=np.int32)
    extra["use_success"] = np.asarray([bool(getattr(result, "success", False)) for result in results], dtype=np.bool_)
    if initial_q is not None:
        key = _initial_q_extra_key(action_format)
        extra[key] = np.repeat(np.asarray(initial_q, dtype=np.float32).reshape(1, -1), repeats=len(specs), axis=0)
    if "object_pose" in extra:
        object_pose = np.asarray(extra["object_pose"], dtype=np.float32)
        extra["max_object_z"] = _episode_reduce(training_records, object_pose[:, 2], np.max, dtype=np.float32)
        extra["final_object_z"] = _episode_final(training_records, object_pose[:, 2], dtype=np.float32)
    if "cube_pose" in extra:
        cube_pose = np.asarray(extra["cube_pose"], dtype=np.float32)
        extra["max_cube_z"] = _episode_reduce(training_records, cube_pose[:, 2], np.max, dtype=np.float32)
        extra["final_cube_pose"] = _episode_final(training_records, cube_pose, dtype=np.float32)
        extra["lift_success"] = np.asarray([bool(getattr(result, "success", False)) for result in results], dtype=np.bool_)
    if "lift_delta" in extra:
        lift_delta = np.asarray(extra["lift_delta"], dtype=np.float32).reshape(-1)
        extra["max_lift_delta"] = _episode_reduce(training_records, lift_delta, np.max, dtype=np.float32)
        extra["final_lift_delta"] = _episode_final(training_records, lift_delta, dtype=np.float32)
    if "button_slide" in extra:
        button_slide = np.asarray(extra["button_slide"], dtype=np.float32).reshape(-1)
        extra["min_button_slide"] = _episode_reduce(training_records, button_slide, np.min, dtype=np.float32)
    total_frames = sum(_training_episode_frame_count(record) for record, _ in training_records)
    for key in ("lift_success", "button_success"):
        if key in extra and np.asarray(extra[key]).shape[:1] == (int(total_frames),):
            arr = np.asarray(extra[key], dtype=np.bool_).reshape(-1)
            extra[f"final_{key}"] = _episode_final(training_records, arr, dtype=np.bool_)
    if "target_key_success" in extra:
        extra["final_target_key_success"] = _episode_final(
            training_records,
            np.asarray(extra["target_key_success"], dtype=np.bool_),
            dtype=np.bool_,
        )
    if "target_key_slide" in extra:
        target_slide = np.asarray(extra["target_key_slide"], dtype=np.float32)
        extra["final_target_key_slide"] = _episode_final(training_records, target_slide, dtype=np.float32)
        extra["min_target_key_slide"] = _episode_reduce(training_records, target_slide, np.min, dtype=np.float32)
    return extra


def _initial_q_extra_key(action_format: str) -> str:
    if str(action_format) == "robotiq_stick_robot_model_q":
        return "initial_robotiq_stick_q_world"
    if str(action_format) == "robot_model_q":
        return "initial_robot_model_q_world"
    return "initial_wuji_q_world"


def _concat_training_key(
    training_records: list[tuple[OnlineTrainingEpisode, Any]],
    key: str,
    *,
    required: bool,
) -> np.ndarray | None:
    arrays: list[np.ndarray] = []
    for record, _ in training_records:
        values = record.frames.get(key)
        if not values:
            if required:
                raise KeyError(f"Recorded online training episode is missing frame key {key!r}")
            return None
        arrays.append(np.stack([np.asarray(value) for value in values], axis=0))
    return np.concatenate(arrays, axis=0)


def _episode_slices_from_records(training_records: list[tuple[OnlineTrainingEpisode, Any]]) -> list[slice]:
    slices: list[slice] = []
    start = 0
    for record, _ in training_records:
        end = start + _training_episode_frame_count(record)
        slices.append(slice(start, end))
        start = end
    return slices


def _episode_final(
    training_records: list[tuple[OnlineTrainingEpisode, Any]],
    values: np.ndarray,
    *,
    dtype: Any,
) -> np.ndarray:
    arr = np.asarray(values)
    out = [arr[slc.stop - 1] for slc in _episode_slices_from_records(training_records)]
    return np.asarray(out, dtype=dtype)


def _episode_reduce(
    training_records: list[tuple[OnlineTrainingEpisode, Any]],
    values: np.ndarray,
    reducer: Any,
    *,
    dtype: Any,
) -> np.ndarray:
    arr = np.asarray(values)
    out = [reducer(arr[slc], axis=0) for slc in _episode_slices_from_records(training_records)]
    return np.asarray(out, dtype=dtype)


def _robotiq_stick_action_space(runtime: OnlineLowLevelRuntime) -> dict[str, Any]:
    dim = int(runtime.local_dof)
    return {
        "type": "robot_model_q",
        "dim": dim,
        "robot_name": ROBOTIQ_STICK_ROBOT_NAME,
        "component_slices": {
            "gripper_xyz": [0, 3],
            "gripper_euler_xyz": [3, 6],
            "gripper_cmd": [6, 7],
            "stick_xyz": [7, 10],
            "stick_euler_xyz": [10, 13],
        },
        "representation": "concat(gripper_xyz(3), gripper_euler_xyz(3), gripper_cmd(1), stick_xyz(3), stick_euler_xyz(3))",
        "low_level_manipulator": str(runtime.low_level_manipulator),
    }


def _robot_model_q_action_space(runtime: OnlineLowLevelRuntime, env: Any) -> dict[str, Any]:
    right = env.robot.side("right")
    dim = int(runtime.local_dof)
    hand_start = int(right.arm_dim)
    joint_orders = tuple(str(x) for x in tuple(right.arm_joint_names + right.hand_joint_names))
    return {
        "type": "robot_model_q",
        "dim": dim,
        "robot_name": str(runtime.low_level_manipulator),
        "component_slices": {
            "base_xyz": [0, 3],
            "base_euler_xyz": [3, 6],
            "hand_q": [hand_start, dim],
        },
        "joint_orders": list(joint_orders),
        "representation": f"concat(base_xyz(3), base_euler_xyz(3), hand_q({dim - hand_start}))",
        "low_level_manipulator": str(runtime.low_level_manipulator),
        "manipulator_name": str(runtime.low_level_manipulator),
    }


def _merge_training_payloads(paths: list[Path], output_path: Path) -> Path:
    if not paths:
        raise FileNotFoundError("No worker training payloads were produced for online_rollout.save_training_pt")
    sorted_paths = sorted(paths, key=_worker_output_sort_key)
    payloads = [load_high_level_payload(path, validate=True) for path in sorted_paths]
    if len(payloads) == 1:
        payload = payloads[0]
        payload["meta"]["source"]["worker_payloads"] = [str(path) for path in sorted_paths]
        payload["meta"]["source"]["output_pt"] = str(output_path)
        payload["meta"]["dataset_name"] = Path(output_path).stem
        return save_high_level_payload(payload, output_path, validate=True)

    frame_counts = [int(torch.as_tensor(payload["episode_ends"]).reshape(-1)[-1].item()) for payload in payloads]
    episode_counts = [int(torch.as_tensor(payload["episode_ends"]).numel()) for payload in payloads]
    data: dict[str, torch.Tensor] = {}
    for key in sorted({key for payload in payloads for key in payload["data"].keys()}):
        values = [payload["data"][key] for payload in payloads if key in payload["data"]]
        if len(values) != len(payloads):
            continue
        tensors = [torch.as_tensor(value).detach().cpu() for value in values]
        if all(tensor.ndim > 0 and int(tensor.shape[0]) == frame_counts[idx] for idx, tensor in enumerate(tensors)):
            data[key] = torch.cat(tensors, dim=0)
        elif all(tensor.ndim > 0 and int(tensor.shape[0]) == episode_counts[idx] for idx, tensor in enumerate(tensors)):
            data[key] = torch.cat(tensors, dim=0)
        elif all(torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
            data[key] = tensors[0]

    lengths: list[int] = []
    for payload in payloads:
        ends = torch.as_tensor(payload["episode_ends"], dtype=torch.long).reshape(-1)
        starts = torch.cat([torch.zeros((1,), dtype=torch.long), ends[:-1]])
        lengths.extend((ends - starts).detach().cpu().numpy().astype(np.int64).tolist())
    episode_ends = make_episode_ends(lengths)
    episode_meta = [item for payload in payloads for item in payload["episode_meta"]]
    meta = dict(payloads[0]["meta"])
    for key in ("episode_initial_pose_paths", "episode_initial_q_world"):
        values = [payload["meta"].get(key) for payload in payloads]
        if any(value is not None for value in values):
            if any(
                value is None or len(value) != count
                for value, count in zip(values, episode_counts)
            ):
                raise ValueError(f"Worker training metadata {key!r} has inconsistent episode rows")
            meta[key] = [row for value in values for row in value]
    meta["dataset_name"] = Path(output_path).stem
    meta["num_episodes"] = int(len(episode_meta))
    meta["num_frames"] = int(sum(lengths))
    source = dict(meta.get("source", {}))
    source["worker_payloads"] = [str(path) for path in sorted_paths]
    source["output_pt"] = str(output_path)
    source["source_indices"] = [
        int(item.get("source_trajectory_index", idx))
        for idx, item in enumerate(episode_meta)
    ]
    meta["source"] = source
    payload = {
        "meta": meta,
        "data": data,
        "episode_ends": episode_ends,
        "episode_meta": episode_meta,
    }
    return save_high_level_payload(payload, output_path, validate=True)


def _worker_output_sort_key(path: Path) -> tuple[str, int]:
    stem = Path(path).stem
    try:
        return stem.rsplit("_", 1)[0], int(stem.rsplit("_", 1)[1])
    except Exception:
        return stem, 0


def _contact_object_ids_for_specs(
    contact_payload: dict[str, Any],
    specs: list[EpisodeSpec],
    *,
    count: int | None = None,
) -> np.ndarray:
    selected_specs = specs if count is None else specs[: int(count)]
    return np.asarray(
        [
            _object_suite_episode_id(
                contact_payload,
                int(spec.contact_episode_index),
                label="contact source",
            )
            for spec in selected_specs
        ]
    )


def _compact_online_payload(
    results: list[RolloutResult],
    *,
    cfg: DictConfig,
    env: TabletopManipulationEnv,
    reset_source: dict[str, np.ndarray],
    reset_source_path: Path,
    reset_source_kind: str,
    contact_path: Path,
    high_level_config_path: Path,
    runtime: OnlineLowLevelRuntime,
    specs: list[EpisodeSpec],
    reset_source_count: int,
    backend: str,
    dt: float,
    dataset_shift: np.ndarray,
    initial_pose_mode: str,
    initial_pose_path: Path | None,
    initial_wuji_q: np.ndarray | None,
    proprio_source: str,
    controller_steps_per_prediction: int,
) -> dict[str, object]:
    payload = _compact_payload(
        results,
        cfg=cfg,
        env=env,
        source=_compact_source_metadata(reset_source),
        source_path=reset_source_path,
        selected=[int(spec.source_trajectory_index) for spec in specs],
        source_count=int(reset_source_count),
        backend=backend,
        dt=dt,
    )
    payload["contact_source_pt_path"] = np.asarray(str(contact_path))
    payload["reset_source_kind"] = np.asarray(str(reset_source_kind))
    payload["high_level_config_path"] = np.asarray(str(high_level_config_path))
    payload["low_level_config_path"] = np.asarray(str(runtime.low_level_config_path))
    payload["low_level_checkpoint_path"] = np.asarray(str(runtime.low_level_checkpoint_path))
    payload["non_contact_enabled"] = np.asarray(bool(runtime.non_contact_enabled))
    payload["dataset_xyz_translation"] = np.asarray(dataset_shift, dtype=np.float32)
    payload["initial_pose_mode"] = np.asarray(str(initial_pose_mode))
    payload["initial_pose_path"] = np.asarray("" if initial_pose_path is None else str(initial_pose_path))
    payload["proprio_source"] = np.asarray(str(proprio_source))
    payload["contact_input"] = np.asarray(_contact_input_mode_from_cfg(cfg))
    payload["controller_steps_per_prediction"] = np.asarray(
        int(controller_steps_per_prediction), dtype=np.int32
    )
    for controller_key in (
        "arm_kp_scale",
        "arm_kv_scale",
        "arm_force_scale",
        "hand_kp_scale",
        "hand_kv_scale",
        "hand_force_scale",
    ):
        payload[f"controller_{controller_key}"] = np.asarray(
            float(OmegaConf.select(cfg, f"controller.{controller_key}", default=1.0)),
            dtype=np.float32,
        )
    if initial_wuji_q is not None:
        payload["initial_wuji_q_world"] = np.asarray(initial_wuji_q, dtype=np.float32)
    return payload


def _pump_bottle_order_payload(
    results: list[PumpBottleRolloutResult],
) -> dict[str, object]:
    ordered_success = np.asarray(
        [result.ordered_success for result in results], dtype=np.bool_
    )
    return {
        "rollout_first_button_success_frame": np.asarray(
            [result.first_button_success_frame for result in results], dtype=np.int32
        ),
        "rollout_first_lift_success_frame": np.asarray(
            [result.first_lift_success_frame for result in results], dtype=np.int32
        ),
        "rollout_lift_clearance_delta_at_first_button_success": np.asarray(
            [result.lift_clearance_delta_at_first_button_success for result in results],
            dtype=np.float32,
        ),
        "rollout_button_after_lift_success": np.asarray(
            [result.button_after_lift_success for result in results], dtype=np.bool_
        ),
        "ordered_success": ordered_success,
        "ordered_success_count": np.asarray(
            int(np.sum(ordered_success)), dtype=np.int32
        ),
        "ordered_success_rate": np.asarray(
            float(np.mean(ordered_success)) if ordered_success.size else 0.0,
            dtype=np.float32,
        ),
        "ordered_success_definition": np.asarray(
            "legacy_success_and_complete_body_lift_at_first_button_success_frame"
        ),
    }


def _compact_online_payload_robot_model_hand(
    results: list[PumpBottleRolloutResult],
    *,
    cfg: DictConfig,
    env: TabletopManipulationEnv,
    contact_payload: dict[str, Any],
    source: dict[str, np.ndarray] | None,
    source_path: Path | None,
    contact_path: Path,
    high_level_config_path: Path,
    runtime: OnlineLowLevelRuntime,
    specs: list[EpisodeSpec],
    source_count: int,
    backend: str,
    dt: float,
    dataset_shift: np.ndarray,
    translation_groups: tuple[tuple[int, int, int], ...],
    proprio_source: str,
    controller_steps_per_prediction: int,
    initial_pose_mode: str,
    initial_pose_path: Path,
    initial_q: np.ndarray | None,
) -> dict[str, object]:
    task_name = str(OmegaConf.select(cfg, "task.name", default="cube_pick")).strip()
    contact_dt = _payload_dt(contact_payload)
    if task_name in {"keyboard_press", "pumpbottle_use"}:
        source_seed = np.asarray(
            [-1 if result.source_seed is None else int(result.source_seed) for result in results],
            dtype=np.int32,
        )
        has_source_seed = np.asarray([result.source_seed is not None for result in results], dtype=np.bool_)
        success = np.asarray([result.success for result in results], dtype=np.bool_)
        failed_indices = np.asarray([result.source_index for result in results if not result.success], dtype=np.int32)
        error_indices = np.asarray([result.source_index for result in results if result.error], dtype=np.int32)
        final_object_pose = np.asarray([result.final_cube_pose for result in results], dtype=np.float32)
        max_object_z = np.asarray([result.max_cube_z for result in results], dtype=np.float32)
        payload: dict[str, object] = {
            "schema_version": np.asarray(1, dtype=np.int32),
            "source_pt_path": np.asarray(str(contact_path.expanduser().resolve())),
            "contact_source_pt_path": np.asarray(str(contact_path.expanduser().resolve())),
            "source_trajectory_count": np.asarray(int(source_count), dtype=np.int32),
            "source_indices": np.asarray([int(spec.source_trajectory_index) for spec in specs], dtype=np.int32),
            "contact_episode_indices": np.asarray([int(spec.contact_episode_index) for spec in specs], dtype=np.int32),
            "embodiment": np.asarray(str(cfg.env.embodiment)),
            "mode": np.asarray(str(cfg.env.mode)),
            "backend": np.asarray(str(backend)),
            "dt": np.asarray(float(dt), dtype=np.float32),
            "frame_skip": np.asarray(int(env.backend.frame_skip), dtype=np.int32),
            "source_dt": np.asarray(
                np.nan if contact_dt is None else float(contact_dt), dtype=np.float32
            ),
            "contact_dt": np.asarray(
                np.nan if contact_dt is None else float(contact_dt), dtype=np.float32
            ),
            "require_contact_dt_match": np.asarray(
                bool(OmegaConf.select(cfg, "online_rollout.require_contact_dt_match", default=False))
            ),
            "gravity_compensation": np.asarray(env.gravity_compensation),
            "enable_fingertip_collision_pads": np.asarray(
                bool(OmegaConf.select(cfg, "env.enable_fingertip_collision_pads", default=True))
            ),
            "allegro_hand_collision_model": np.asarray(
                str(OmegaConf.select(cfg, "env.allegro_hand_collision_model", default="full"))
            ),
            "object_kind": np.asarray(str(getattr(env.scene, "object_kind", "mini_keyboard"))),
            "success": success,
            "success_count": np.asarray(int(np.sum(success)), dtype=np.int32),
            "success_rate": np.asarray(float(np.mean(success)) if success.size else 0.0, dtype=np.float32),
            "failed_indices": failed_indices,
            "error_indices": error_indices,
            "error_messages": np.asarray([result.error for result in results]),
            "completed_steps": np.asarray([result.completed_steps for result in results], dtype=np.int32),
            "action_frame_count": np.asarray([result.action_frame_count for result in results], dtype=np.int32),
            "max_cube_z": max_object_z,
            "final_cube_pose": final_object_pose,
            "max_object_z": max_object_z,
            "final_object_pose": final_object_pose,
            "rollout_lift_success": np.asarray([result.lift_success for result in results], dtype=np.bool_),
            "rollout_button_success": np.asarray([result.button_success for result in results], dtype=np.bool_),
            "rollout_max_lift_delta": np.asarray([result.max_lift_delta for result in results], dtype=np.float32),
            "rollout_final_lift_delta": np.asarray([result.final_lift_delta for result in results], dtype=np.float32),
            "rollout_max_lift_clearance_delta": np.asarray(
                [result.max_lift_clearance_delta for result in results], dtype=np.float32
            ),
            "rollout_final_lift_clearance_delta": np.asarray(
                [result.final_lift_clearance_delta for result in results], dtype=np.float32
            ),
            "rollout_initial_min_geom_z": np.asarray(
                [result.initial_min_geom_z for result in results], dtype=np.float32
            ),
            "rollout_final_min_geom_z": np.asarray(
                [result.final_min_geom_z for result in results], dtype=np.float32
            ),
            "rollout_min_button_slide": np.asarray([result.min_button_slide for result in results], dtype=np.float32),
            "rollout_final_button_slide": np.asarray([result.final_button_slide for result in results], dtype=np.float32),
            "rollout_target_key_success": _result_matrix(
                results,
                "target_key_success",
                dtype=np.bool_,
                fill=False,
            ),
            "rollout_final_target_key_slide": _result_matrix(
                results,
                "final_target_key_slide",
                dtype=np.float32,
                fill=float("nan"),
            ),
            "rollout_min_target_key_slide": _result_matrix(
                results,
                "min_target_key_slide",
                dtype=np.float32,
                fill=float("nan"),
            ),
            "source_trajectory_seed": source_seed,
            "has_source_trajectory_seed": has_source_seed,
            "robot_xml_path": np.asarray(str(env.scene.robot_xml_path)),
            "group_names": np.asarray(env.robot.group_names),
            "high_level_config_path": np.asarray(str(high_level_config_path)),
            "low_level_config_path": np.asarray(str(runtime.low_level_config_path)),
            "low_level_checkpoint_path": np.asarray(str(runtime.low_level_checkpoint_path)),
            "low_level_manipulator": np.asarray(str(runtime.low_level_manipulator)),
            "non_contact_enabled": np.asarray(bool(runtime.non_contact_enabled)),
            "dataset_xyz_translation": np.asarray(dataset_shift, dtype=np.float32),
            "low_level_q_translation_groups": np.asarray(translation_groups, dtype=np.int32),
            "initial_pose_mode": np.asarray(str(initial_pose_mode)),
            "initial_pose_path": np.asarray(str(initial_pose_path.expanduser().resolve())),
            "proprio_source": np.asarray(str(proprio_source)),
            "controller_steps_per_prediction": np.asarray(int(controller_steps_per_prediction), dtype=np.int32),
        }
        if task_name == "pumpbottle_use":
            payload["pump_bottle_object_ids"] = _contact_object_ids_for_specs(
                contact_payload,
                specs,
                count=len(results),
            )
            payload.update(_pump_bottle_order_payload(results))
        object_kind = str(getattr(env.scene, "object_kind", ""))
        if object_kind == "mini_keyboard":
            payload["target_key_joint_names"] = np.asarray(
                tuple(str(x) for x in getattr(env.task, "target_key_joint_names", ())),
                dtype=object,
            )
            payload["keyboard_support_box_height"] = np.asarray(
                float(getattr(env.task, "support_box_height", float("nan"))),
                dtype=np.float32,
            )
            payload["keyboard_key_success_fraction"] = np.asarray(
                float(getattr(env.task, "key_success_fraction", float("nan"))),
                dtype=np.float32,
            )
        elif object_kind == "pump_bottle":
            payload["pump_bottle_button_success_slide"] = np.asarray(
                float(getattr(env.task, "button_success_slide", float("nan"))),
                dtype=np.float32,
            )
            payload["pump_bottle_lift_success_delta"] = np.asarray(
                float(getattr(env.task, "lift_success_delta", float("nan"))),
                dtype=np.float32,
            )
            payload["pump_bottle_lift_success_metric"] = np.asarray(
                _pump_bottle_lift_success_metric_from_cfg(
                    cfg,
                    section="online_rollout",
                    default="max_lift_delta",
                )
            )
            payload["pump_bottle_lift_measure"] = np.asarray("min_body_geom_z_delta")
            _set_pump_bottle_physics_payload_metadata(payload, cfg)
        if source_path is not None:
            source_key = (
                "robot_model_q_source_pt_path"
                if source is not None and str(_source_scalar(source, "action_space_type") or "").strip() == "robot_model_q"
                else "object_reset_source_pt_path"
            )
            payload[source_key] = np.asarray(str(source_path.expanduser().resolve()))
        if initial_q is not None:
            payload["initial_robot_model_q_world"] = np.asarray(initial_q, dtype=np.float32)
        _set_grasp_hold_payload_metadata(payload, cfg)
        _set_robot_model_execution_root_offset_payload_metadata(payload, cfg)
        _set_effector_execution_offset_payload_metadata(payload, cfg)
        return payload

    if source is None or source_path is None:
        raise ValueError("robot_model_q source is required when compacting floating cube-pick payloads")
    payload = _compact_payload(
        results,
        cfg=cfg,
        env=env,
        source=source,
        source_path=source_path,
        selected=[int(spec.source_trajectory_index) for spec in specs],
        source_count=int(source_count),
        backend=backend,
        dt=dt,
    )
    payload["contact_source_pt_path"] = np.asarray(str(contact_path.expanduser().resolve()))
    source_key = (
        "robot_model_q_source_pt_path"
        if str(_source_scalar(source, "action_space_type") or "").strip() == "robot_model_q"
        else "object_reset_source_pt_path"
    )
    payload[source_key] = np.asarray(str(source_path.expanduser().resolve()))
    payload["contact_episode_indices"] = np.asarray([int(spec.contact_episode_index) for spec in specs], dtype=np.int32)
    payload["high_level_config_path"] = np.asarray(str(high_level_config_path))
    payload["low_level_config_path"] = np.asarray(str(runtime.low_level_config_path))
    payload["low_level_checkpoint_path"] = np.asarray(str(runtime.low_level_checkpoint_path))
    payload["low_level_manipulator"] = np.asarray(str(runtime.low_level_manipulator))
    payload["non_contact_enabled"] = np.asarray(bool(runtime.non_contact_enabled))
    payload["dataset_xyz_translation"] = np.asarray(dataset_shift, dtype=np.float32)
    payload["low_level_q_translation_groups"] = np.asarray(translation_groups, dtype=np.int32)
    payload["initial_pose_mode"] = np.asarray(str(initial_pose_mode))
    payload["initial_pose_path"] = np.asarray(str(initial_pose_path.expanduser().resolve()))
    if initial_q is not None:
        payload["initial_robot_model_q_world"] = np.asarray(initial_q, dtype=np.float32)
    payload["proprio_source"] = np.asarray(str(proprio_source))
    payload["contact_input"] = np.asarray(_contact_input_mode_from_cfg(cfg))
    payload["controller_steps_per_prediction"] = np.asarray(int(controller_steps_per_prediction), dtype=np.int32)
    _set_grasp_hold_payload_metadata(payload, cfg)
    _set_robot_model_execution_root_offset_payload_metadata(payload, cfg)
    _set_effector_execution_offset_payload_metadata(payload, cfg)
    payload["require_contact_dt_match"] = np.asarray(
        bool(OmegaConf.select(cfg, "online_rollout.require_contact_dt_match", default=False))
    )
    phase_effort_section = "online_rollout.phase_finger_effort"
    payload["phase_finger_effort_enabled"] = np.asarray(
        bool(OmegaConf.select(cfg, f"{phase_effort_section}.enabled", default=False))
    )
    payload["phase_finger_effort_phase_ids"] = np.asarray(
        sorted(
            int(value)
            for value in OmegaConf.select(
                cfg, f"{phase_effort_section}.phase_ids", default=[5]
            )
        ),
        dtype=np.int8,
    )
    payload["phase_finger_effort_max_effort"] = np.asarray(
        float(
            OmegaConf.select(
                cfg,
                f"{phase_effort_section}.max_effort",
                default=OmegaConf.select(cfg, "controller.fingers.max_effort", default=0.5),
            )
        ),
        dtype=np.float32,
    )
    phase_gains_section = "online_rollout.phase_finger_gains"
    payload["phase_finger_gains_enabled"] = np.asarray(
        bool(OmegaConf.select(cfg, f"{phase_gains_section}.enabled", default=False))
    )
    payload["phase_finger_gains_phase_ids"] = np.asarray(
        sorted(
            int(value)
            for value in OmegaConf.select(
                cfg, f"{phase_gains_section}.phase_ids", default=[2, 3]
            )
        ),
        dtype=np.int8,
    )
    payload["phase_finger_gains_kp"] = np.asarray(
        float(
            OmegaConf.select(
                cfg,
                f"{phase_gains_section}.kp",
                default=OmegaConf.select(cfg, "controller.fingers.kp", default=16.0),
            )
        ),
        dtype=np.float32,
    )
    payload["phase_finger_gains_kv"] = np.asarray(
        float(
            OmegaConf.select(
                cfg,
                f"{phase_gains_section}.kv",
                default=OmegaConf.select(cfg, "controller.fingers.kv", default=8.0),
            )
        ),
        dtype=np.float32,
    )
    phase_steps_section = "online_rollout.phase_controller_steps"
    payload["phase_controller_steps_enabled"] = np.asarray(
        bool(OmegaConf.select(cfg, f"{phase_steps_section}.enabled", default=False))
    )
    payload["phase_controller_steps_phase_ids"] = np.asarray(
        sorted(
            int(value)
            for value in OmegaConf.select(cfg, f"{phase_steps_section}.phase_ids", default=[2])
        ),
        dtype=np.int8,
    )
    payload["phase_controller_steps_per_prediction"] = np.asarray(
        int(
            OmegaConf.select(
                cfg,
                f"{phase_steps_section}.steps_per_prediction",
                default=controller_steps_per_prediction,
            )
        ),
        dtype=np.int32,
    )
    payload["allegro_hand_friction"] = np.asarray(
        OmegaConf.select(cfg, "env.allegro_hand_friction", default=[1.0, 0.005, 0.0001]),
        dtype=np.float32,
    )
    payload["allegro_hand_collision_model"] = np.asarray(
        str(OmegaConf.select(cfg, "env.allegro_hand_collision_model", default="full"))
    )
    payload["allegro_hand_condim"] = np.asarray(
        int(OmegaConf.select(cfg, "env.allegro_hand_condim", default=6)), dtype=np.int32
    )
    payload["allegro_hand_priority"] = np.asarray(
        int(OmegaConf.select(cfg, "env.allegro_hand_priority", default=0)), dtype=np.int32
    )
    payload["allegro_hand_solref"] = np.asarray(
        OmegaConf.select(cfg, "env.allegro_hand_solref", default=[0.02, 1.0]),
        dtype=np.float32,
    )
    payload["allegro_hand_solimp"] = np.asarray(
        OmegaConf.select(cfg, "env.allegro_hand_solimp", default=[0.9, 0.95, 0.001, 0.5, 2.0]),
        dtype=np.float32,
    )
    payload["barrett_fingertip_collision_model"] = np.asarray(
        str(OmegaConf.select(cfg, "env.barrett_fingertip_collision_model", default="capsule"))
    )
    payload["barrett_fingertip_capsule_radius"] = np.asarray(
        float(OmegaConf.select(cfg, "env.barrett_fingertip_capsule_radius", default=0.014)),
        dtype=np.float32,
    )
    payload["barrett_fingertip_friction"] = np.asarray(
        OmegaConf.select(cfg, "env.barrett_fingertip_friction", default=[8.0, 0.25, 0.03]),
        dtype=np.float32,
    )
    payload["wujihand_contact_friction"] = np.asarray(
        OmegaConf.select(cfg, "env.wujihand_contact_friction", default=[1.0, 0.005, 0.0001]),
        dtype=np.float32,
    )
    if task_name == "object_suite_pick_place":
        payload["object_id"] = np.asarray(
            [
                _object_suite_source_id(
                    source, spec.source_trajectory_index, label="object reset source"
                )
                for spec in specs
            ]
        )
        payload["object_friction"] = np.asarray(
            OmegaConf.select(cfg, "objects.friction", default=[0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        payload["object_collision_mesh_mode"] = np.asarray(
            str(OmegaConf.select(cfg, "objects.collision_mesh_mode", default="single_mesh"))
        )
        object_spec = getattr(env.task, "object_spec", None)
        collision_paths = getattr(object_spec, "effective_collision_mesh_paths", ())
        collision_part_count = len(collision_paths) if collision_paths else 1
        payload["object_collision_part_count"] = np.full(
            len(results), collision_part_count, dtype=np.int32
        )
    return payload


def _result_matrix(
    results: list[PumpBottleRolloutResult],
    attr: str,
    *,
    dtype: Any,
    fill: float | bool,
) -> np.ndarray:
    width = 0
    for result in results:
        value = getattr(result, attr)
        if value is None:
            continue
        arr = np.asarray(value).reshape(-1)
        if arr.size:
            width = int(arr.shape[0])
            break
    if width == 0:
        return np.zeros((len(results), 0), dtype=dtype)
    out = np.full((len(results), width), fill, dtype=dtype)
    for idx, result in enumerate(results):
        value = getattr(result, attr)
        if value is None:
            continue
        arr = np.asarray(value, dtype=dtype).reshape(-1)
        if int(arr.shape[0]) != width:
            raise ValueError(f"{attr} result width mismatch: expected {width}, got {arr.shape[0]}")
        out[idx] = arr
    return out


def _compact_online_payload_robotiq_stick(
    results: list[PumpBottleRolloutResult],
    *,
    cfg: DictConfig,
    env: Any,
    contact_payload: dict[str, Any],
    contact_path: Path,
    high_level_config_path: Path,
    runtime: OnlineLowLevelRuntime,
    specs: list[EpisodeSpec],
    source_count: int,
    backend: str,
    dt: float,
    dataset_shift: np.ndarray,
    translation_groups: tuple[tuple[int, int, int], ...],
    initial_pose_mode: str,
    initial_pose_path: Path,
    initial_q: np.ndarray,
    proprio_source: str,
    controller_steps_per_prediction: int,
) -> dict[str, object]:
    task_name = str(OmegaConf.select(cfg, "task.name", default="cube_pick")).strip()
    source_seed = np.asarray(
        [-1 if result.source_seed is None else int(result.source_seed) for result in results],
        dtype=np.int32,
    )
    has_source_seed = np.asarray([result.source_seed is not None for result in results], dtype=np.bool_)
    success = np.asarray([result.success for result in results], dtype=np.bool_)
    final_object_pose = np.asarray([result.final_cube_pose for result in results], dtype=np.float32)
    max_object_z = np.asarray([result.max_cube_z for result in results], dtype=np.float32)
    failed_indices = np.asarray([result.source_index for result in results if not result.success], dtype=np.int32)
    error_indices = np.asarray([result.source_index for result in results if result.error], dtype=np.int32)
    contact_meta = dict(contact_payload.get("meta", {}))
    contact_dt = _payload_dt(contact_payload)
    object_kind = str(getattr(env.scene, "object_kind", "pump_bottle"))
    functional_mapping_enabled, functional_mapping_gain = (
        _robotiq_functional_contact_mapping_from_cfg(cfg)
    )
    gripper_min_enabled, gripper_min_command, gripper_min_phases = (
        _robotiq_gripper_min_command_from_cfg(cfg)
    )
    stick_offset_enabled, stick_offset_xyz, stick_offset_phases = (
        _robotiq_stick_execution_offset_from_cfg(cfg)
    )

    payload: dict[str, object] = {
        "schema_version": np.asarray(1, dtype=np.int32),
        "source_pt_path": np.asarray(str(contact_path.expanduser().resolve())),
        "contact_source_pt_path": np.asarray(str(contact_path.expanduser().resolve())),
        "source_trajectory_count": np.asarray(int(source_count), dtype=np.int32),
        "source_indices": np.asarray([int(spec.source_trajectory_index) for spec in specs], dtype=np.int32),
        "contact_episode_indices": np.asarray([int(spec.contact_episode_index) for spec in specs], dtype=np.int32),
        "embodiment": np.asarray(str(cfg.env.embodiment)),
        "mode": np.asarray(str(cfg.env.mode)),
        "backend": np.asarray(str(backend)),
        "dt": np.asarray(float(dt), dtype=np.float32),
        "frame_skip": np.asarray(int(env.backend.frame_skip), dtype=np.int32),
        "source_dt": np.asarray(
            np.nan if contact_dt is None else float(contact_dt),
            dtype=np.float32,
        ),
        "contact_dt": np.asarray(
            np.nan if contact_dt is None else float(contact_dt),
            dtype=np.float32,
        ),
        "require_contact_dt_match": np.asarray(
            bool(OmegaConf.select(cfg, "online_rollout.require_contact_dt_match", default=False))
        ),
        "gravity_compensation": np.asarray(False),
        "object_kind": np.asarray(object_kind),
        "success": success,
        "success_count": np.asarray(int(np.sum(success)), dtype=np.int32),
        "success_rate": np.asarray(float(np.mean(success)) if success.size else 0.0, dtype=np.float32),
        "failed_indices": failed_indices,
        "error_indices": error_indices,
        "error_messages": np.asarray([result.error for result in results]),
        "completed_steps": np.asarray([result.completed_steps for result in results], dtype=np.int32),
        "action_frame_count": np.asarray([result.action_frame_count for result in results], dtype=np.int32),
        "max_cube_z": max_object_z,
        "final_cube_pose": final_object_pose,
        "max_object_z": max_object_z,
        "final_object_pose": final_object_pose,
        "rollout_lift_success": np.asarray([result.lift_success for result in results], dtype=np.bool_),
        "rollout_button_success": np.asarray([result.button_success for result in results], dtype=np.bool_),
        "rollout_max_lift_delta": np.asarray([result.max_lift_delta for result in results], dtype=np.float32),
        "rollout_final_lift_delta": np.asarray([result.final_lift_delta for result in results], dtype=np.float32),
        "rollout_max_lift_clearance_delta": np.asarray(
            [result.max_lift_clearance_delta for result in results], dtype=np.float32
        ),
        "rollout_final_lift_clearance_delta": np.asarray(
            [result.final_lift_clearance_delta for result in results], dtype=np.float32
        ),
        "rollout_initial_min_geom_z": np.asarray(
            [result.initial_min_geom_z for result in results], dtype=np.float32
        ),
        "rollout_final_min_geom_z": np.asarray(
            [result.final_min_geom_z for result in results], dtype=np.float32
        ),
        "rollout_min_button_slide": np.asarray([result.min_button_slide for result in results], dtype=np.float32),
        "rollout_final_button_slide": np.asarray([result.final_button_slide for result in results], dtype=np.float32),
        "rollout_target_key_success": _result_matrix(
            results,
            "target_key_success",
            dtype=np.bool_,
            fill=False,
        ),
        "rollout_final_target_key_slide": _result_matrix(
            results,
            "final_target_key_slide",
            dtype=np.float32,
            fill=float("nan"),
        ),
        "rollout_min_target_key_slide": _result_matrix(
            results,
            "min_target_key_slide",
            dtype=np.float32,
            fill=float("nan"),
        ),
        "source_trajectory_seed": source_seed,
        "has_source_trajectory_seed": has_source_seed,
        "robot_xml_path": np.asarray(str(env.scene.robot_xml_path)),
        "group_names": np.asarray(("gripper", "stick")),
        "high_level_config_path": np.asarray(str(high_level_config_path)),
        "low_level_config_path": np.asarray(str(runtime.low_level_config_path)),
        "low_level_checkpoint_path": np.asarray(str(runtime.low_level_checkpoint_path)),
        "low_level_manipulator": np.asarray(str(runtime.low_level_manipulator)),
        "non_contact_enabled": np.asarray(bool(runtime.non_contact_enabled)),
        "dataset_xyz_translation": np.asarray(dataset_shift, dtype=np.float32),
        "low_level_q_translation_groups": np.asarray(translation_groups, dtype=np.int32),
        "initial_pose_mode": np.asarray(str(initial_pose_mode)),
        "initial_pose_path": np.asarray(str(initial_pose_path)),
        "initial_robotiq_stick_q_world": np.asarray(initial_q, dtype=np.float32),
        "proprio_source": np.asarray(str(proprio_source)),
        "controller_steps_per_prediction": np.asarray(int(controller_steps_per_prediction), dtype=np.int32),
        "robotiq_functional_contact_mapping_enabled": np.asarray(functional_mapping_enabled),
        "robotiq_functional_contact_mapping_gain": np.asarray(
            functional_mapping_gain,
            dtype=np.float32,
        ),
        "robotiq_gripper_min_command_enabled": np.asarray(gripper_min_enabled),
        "robotiq_gripper_min_command": np.asarray(
            gripper_min_command, dtype=np.float32
        ),
        "robotiq_gripper_min_command_phase_ids": np.asarray(
            sorted(gripper_min_phases), dtype=np.int8
        ),
        "robotiq_stick_execution_offset_enabled": np.asarray(stick_offset_enabled),
        "robotiq_stick_execution_offset_xyz": np.asarray(
            stick_offset_xyz, dtype=np.float32
        ),
        "robotiq_stick_execution_offset_phase_ids": np.asarray(
            sorted(stick_offset_phases), dtype=np.int8
        ),
        "camera_names": np.asarray(env.camera_names),
        "policy_camera_names": np.asarray(env.policy_camera_names),
        "camera_width": np.asarray(env.camera_width, dtype=np.int32),
        "camera_height": np.asarray(env.camera_height, dtype=np.int32),
        "include_camera_obs": np.asarray(env.include_camera_obs),
        "target_pointcloud_num": np.asarray(env.target_pointcloud_num, dtype=np.int32),
        "exclude_robot_pointcloud": np.asarray(env.exclude_robot_pointcloud),
        "exclude_table_pointcloud": np.asarray(env.exclude_table_pointcloud),
        "capture_camera_rgb": np.asarray(env.capture_camera_rgb),
        "capture_camera_segmentation": np.asarray(env.capture_camera_segmentation),
        "pointcloud_workspace_bounds": np.asarray(env.pointcloud_workspace_bounds, dtype=np.float32),
    }
    if task_name == "pumpbottle_use":
        payload["pump_bottle_object_ids"] = _contact_object_ids_for_specs(
            contact_payload,
            specs,
            count=len(results),
        )
        payload.update(_pump_bottle_order_payload(results))
    if object_kind == "mini_keyboard":
        payload["target_key_joint_names"] = np.asarray(
            tuple(str(x) for x in getattr(env.task, "target_key_joint_names", ())),
            dtype=object,
        )
        payload["keyboard_support_box_height"] = np.asarray(
            float(getattr(env.task, "support_box_height", float("nan"))),
            dtype=np.float32,
        )
        payload["keyboard_key_success_fraction"] = np.asarray(
            float(env.task.key_success_fraction), dtype=np.float32,
        )
    elif object_kind == "pump_bottle":
        payload["pump_bottle_button_success_slide"] = np.asarray(
            float(getattr(env.task, "button_success_slide", float("nan"))),
            dtype=np.float32,
        )
        payload["pump_bottle_lift_success_delta"] = np.asarray(
            float(getattr(env.task, "lift_success_delta", float("nan"))),
            dtype=np.float32,
        )
        payload["pump_bottle_lift_success_metric"] = np.asarray(
            _pump_bottle_lift_success_metric_from_cfg(
                cfg,
                section="online_rollout",
                default="max_lift_delta",
            )
        )
        payload["pump_bottle_lift_measure"] = np.asarray("min_body_geom_z_delta")
        _set_pump_bottle_physics_payload_metadata(payload, cfg)
    for key in (
        "object_asset_name",
        "pump_bottle_xml_path",
        "pump_bottle_mesh_dir",
        "pump_bottle_mesh_scale",
        "pump_bottle_button_success_slide",
        "pump_bottle_lift_success_delta",
        "mini_keyboard_xml_path",
        "keyboard_support_box_height",
        "keyboard_key_success_fraction",
        "target_key_joint_names",
        "target_key_joint_indices",
        "pose_target_key_joint_indices",
        "contact_target_key_joint_names",
        "contact_target_key_joint_indices",
        "contact_mode",
        "contact_sparse_capacity",
        "contact_template_indices",
        "preset_contact_indices",
        "surface_template_hash",
        "surface_template_size",
        "surface_template_path",
    ):
        if key in contact_meta:
            payload[f"source_{key}"] = np.asarray(contact_meta[key])
    return payload


def _merge_result_payloads(paths: list[Path]) -> dict[str, object]:
    if not paths:
        raise ValueError("No worker payloads to merge")
    payloads = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            payloads.append({key: data[key].copy() for key in data.files})
    merged: dict[str, object] = dict(payloads[0])
    concat_keys = (
        "source_indices",
        "success",
        "error_messages",
        "completed_steps",
        "action_frame_count",
        "max_cube_z",
        "final_cube_pose",
        "max_object_z",
        "final_object_pose",
        "object_id",
        "object_collision_part_count",
        "rollout_lift_success",
        "rollout_max_lift_delta",
        "rollout_final_lift_delta",
        "rollout_first_lift_success_frame",
        "contact_episode_indices",
        "source_trajectory_seed",
        "has_source_trajectory_seed",
        "pump_bottle_object_ids",
        "rollout_button_success",
        "rollout_max_lift_clearance_delta",
        "rollout_final_lift_clearance_delta",
        "rollout_initial_min_geom_z",
        "rollout_final_min_geom_z",
        "rollout_min_button_slide",
        "rollout_final_button_slide",
        "rollout_first_button_success_frame",
        "rollout_lift_clearance_delta_at_first_button_success",
        "rollout_button_after_lift_success",
        "ordered_success",
        "rollout_target_key_success",
        "rollout_final_target_key_slide",
        "rollout_min_target_key_slide",
        "episode_initial_pose_paths",
        "episode_initial_q_world",
    )
    for key in concat_keys:
        arrays = [payload[key] for payload in payloads if key in payload]
        if arrays:
            if len(arrays) != len(payloads) or any(
                np.asarray(array).ndim == 0
                or len(array) != len(payload["source_indices"])
                for array, payload in zip(arrays, payloads)
            ):
                raise ValueError(
                    f"Worker result field {key!r} has inconsistent episode rows"
                )
            merged[key] = np.concatenate(arrays, axis=0)
    source_indices = np.asarray(merged["source_indices"], dtype=np.int32)
    if len(np.unique(source_indices)) != len(source_indices):
        raise ValueError("Worker results contain duplicate source episode indices")
    order = np.argsort(source_indices, kind="stable")
    for key in concat_keys:
        if key in merged:
            merged[key] = np.asarray(merged[key])[order]
    source_indices = np.asarray(merged["source_indices"], dtype=np.int32)
    success = np.asarray(merged["success"], dtype=np.bool_)
    errors = np.asarray(merged["error_messages"])
    error_mask = np.asarray([str(x) != "" for x in errors.reshape(-1)], dtype=np.bool_)
    merged["success_count"] = np.asarray(int(np.sum(success)), dtype=np.int32)
    merged["success_rate"] = np.asarray(
        float(np.mean(success)) if success.size else 0.0, dtype=np.float32
    )
    merged["failed_indices"] = source_indices[~success].astype(np.int32, copy=False)
    merged["error_indices"] = source_indices[error_mask].astype(np.int32, copy=False)
    if "ordered_success" in merged:
        ordered = np.asarray(merged["ordered_success"], dtype=np.bool_)
        merged["ordered_success_count"] = np.asarray(int(ordered.sum()), dtype=np.int32)
        merged["ordered_success_rate"] = np.asarray(float(ordered.mean()) if ordered.size else 0, dtype=np.float32)
    return merged


def _results_from_payload(payload: dict[str, object]) -> list[RolloutResult]:
    source_indices = np.asarray(payload["source_indices"], dtype=np.int32).reshape(-1)
    seeds = np.asarray(payload["source_trajectory_seed"], dtype=np.int32).reshape(-1)
    has_seed = np.asarray(payload["has_source_trajectory_seed"], dtype=np.bool_).reshape(-1)
    completed = np.asarray(payload["completed_steps"], dtype=np.int32).reshape(-1)
    action_frames = np.asarray(payload["action_frame_count"], dtype=np.int32).reshape(-1)
    success = np.asarray(payload["success"], dtype=np.bool_).reshape(-1)
    max_z = np.asarray(payload["max_cube_z"], dtype=np.float32).reshape(-1)
    final_pose = np.asarray(payload["final_cube_pose"], dtype=np.float32).reshape(len(source_indices), 7)
    errors = np.asarray(payload["error_messages"]).reshape(-1)
    return [
        RolloutResult(
            source_index=int(source_indices[idx]),
            source_seed=None if not bool(has_seed[idx]) else int(seeds[idx]),
            action_frame_count=int(action_frames[idx]),
            completed_steps=int(completed[idx]),
            success=bool(success[idx]),
            max_cube_z=float(max_z[idx]),
            final_cube_pose=final_pose[idx].astype(np.float32, copy=True),
            error=str(errors[idx]),
        )
        for idx in range(int(source_indices.shape[0]))
    ]


def _save_npz(out: str | Path, payload: dict[str, object]) -> Path:
    out_path = Path(out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    return out_path


def _compact_source_metadata(retarget: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in (
        "embodiment",
        "mode",
        "frame_skip",
        "cube_x_range",
        "cube_y_range",
        "cube_side_length_range",
        "cube_yaw_range_deg",
    ):
        if key in retarget:
            out[key] = np.asarray(retarget[key])
    return out


if __name__ == "__main__":
    from paper_config import canonicalize_cli
    canonicalize_cli()
    main()
