"""Closed-loop tabletop evaluation for trained high-level iDP3 policies."""

from __future__ import annotations

import sys
import time
import os
import tempfile
import warnings
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib")
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.embodiments import validate_embodiment_supported  # noqa: E402
from experiment.high_level_policy.dataset import HighLevelSequenceDataset  # noqa: E402
from experiment.high_level_policy.payload import (  # noqa: E402
    load_high_level_payload,
    save_high_level_payload,
    single_manipulator_payload_to_replay_source,
    with_object_only_world_observation,
)
from experiment.high_level_policy.train_utils import (
    prepare_device,
    set_seed,
)  # noqa: E402
from experiment.contact_runtime_utils import (  # noqa: E402
    WUJI_JOINT_DIM,
    WUJI_LOCAL_DOF,
    _apply_fixed_zero_constraints,
    _euler_xyz_to_quat_wxyz,
    _load_robot_model_start_q,
    _quat_wxyz_to_euler_xyz,
    _retarget_episode_map,
    _validate_robot_model_q,
)
from experiment.pipeline_high_low_policy import (  # noqa: E402
    HighLowPipelineRuntime,
    load_high_low_pipeline_runtime,
    predict_low_level_q_from_contact_action,
    predict_low_level_q_from_latent_action,
)
from experiment.rollout_utils import (  # noqa: E402
    _ensure_source_cube_size,
    _extend_flat_trace,
    _initial_cube_pose,
    _parse_indices,
    _validate_robot_model_q_source_matches_env,
)
from experiment.replay_utils import (  # noqa: E402
    ViewerReplayControl,
    _set_cube_state,
    _source_scalar,
    _trajectory_array,
    _sync_cube_height_ratio_from_source,
    _trajectory_count,
    _trajectory_value,
    _validate_object_suite_source_object,
    _validate_source_compatible,
)
from experiment.tabletop_sim.env import TabletopManipulationEnv  # noqa: E402
from experiment.tabletop_sim.factory import (  # noqa: E402
    build_env_from_config,
    optional_project_path,
)
from experiment.tabletop_sim.perception import (  # noqa: E402
    deproject_pixels_cv,
    farthest_point_sample_indices,
)
from experiment.tabletop_sim.parallel_rollout import (  # noqa: E402
    configure_torch_threads,
    launch_sharded_workers,
    validate_headless_parallel,
)
from experiment.tabletop_sim.profiling import RolloutProfiler  # noqa: E402
from experiment.tabletop_sim.rotation import (  # noqa: E402
    unwrap_robot_model_q_rotation, robot_model_q_to_native, robot_model_q_to_world,
)
from experiment.tabletop_sim.task_success import (  # noqa: E402
    accumulate_task_success,
    task_success_from_observation,
)
from experiment.tabletop_sim.trace import (  # noqa: E402
    append_trace_frame,
    ctrl_names,
    new_trace_buffers,
    qpos_names,
    save_trace_npz,
)
from experiment.task_start_pose import (  # noqa: E402
    TASK_CUBE_PICK,
    describe_start_pose_resolution,
    resolve_start_pose_path,
)
from experiment.wuji_start_pose import (  # noqa: E402
    DEFAULT_WUJI_START_POSE_RELATIVE,
    load_wuji_start_q,
    validate_wuji_q,
)


HIGH_LOW_ACTION_SPACE_TYPES = {"latent_contact", "contact_points"}


from paper_config import runtime_config


@dataclass(frozen=True)
class PolicyRuntime:
    high_level_config_path: Path
    high_level_cfg: DictConfig
    checkpoint_path: Path
    payload_path: Path
    reset_source_path: Path
    payload: dict[str, Any]
    source: dict[str, np.ndarray]
    dataset: HighLevelSequenceDataset
    policy: torch.nn.Module
    device: torch.device
    action_space: dict[str, Any]
    state_key: str
    action_space_type: str
    source_episode_indices: np.ndarray
    episode_count: int
    reset_source_kind: str
    low_level_config_path: Path | None = None
    high_low_runtime: HighLowPipelineRuntime | None = None

    @property
    def observation_point_count(self) -> int:
        if self.high_low_runtime is not None:
            from experiment.pipeline_high_low_policy import _point_cloud_count_for_payload
            return _point_cloud_count_for_payload(self.high_low_runtime.high_level_cfg, self.payload)
        return int(self.dataset.num_points)


@dataclass(frozen=True)
class EvalResult:
    source_index: int
    source_seed: int | None
    completed_steps: int
    policy_call_count: int
    success: bool
    max_cube_z: float
    final_cube_pose: np.ndarray
    error: str


@dataclass
class EvalTrainingEpisode:
    source_index: int
    reset_source_index: int | None = None
    source_seed: int | None = None
    frames: dict[str, list[np.ndarray]] = field(default_factory=dict)


@hydra.main(
    version_base="1.2",
    config_path="conf",
    config_name="evaluate_policy/object_suite_wuji",
)
def main(cfg: DictConfig) -> None:
    cfg = runtime_config(cfg)
    OmegaConf.resolve(cfg)
    from experiment.manipulation.paper_execution import validate_paper_execution, validate_paper_control_dt
    validate_paper_execution(cfg, 'policy_eval')
    configure_torch_threads(cfg, section="policy_eval")
    validate_embodiment_supported(
        str(cfg.env.embodiment),
        "high_level_policy_eval",
        label="env.embodiment",
    )

    high_cfg_path = _resolve_project_path(
        OmegaConf.select(
            cfg,
            "policy_eval.high_level_config_path",
            default="./experiment/conf/train_policy/object_suite_paper2400.yaml",
        ),
        label="policy_eval.high_level_config_path",
    )
    high_cfg = _load_hydra_config(high_cfg_path)
    seed_value = OmegaConf.select(cfg, "policy_eval.seed", default=None)
    if seed_value is None or str(seed_value).strip().lower() in {"", "null", "none"}:
        seed_value = OmegaConf.select(high_cfg, "seed", default=42)
    seed = int(seed_value)
    set_seed(seed)

    runtime = _load_policy_runtime(cfg, high_cfg_path=high_cfg_path, high_cfg=high_cfg)
    _validate_runtime_matches_env(cfg, runtime)
    _enable_policy_camera_observations(cfg, point_count=runtime.observation_point_count)
    _validate_reset_source_compatible(
        runtime.source, cfg, action_space_type=runtime.action_space_type
    )

    source_count = int(runtime.episode_count)
    selected = _selected_eval_indices(
        split_indices=runtime.dataset.selected_episode_indices.tolist(),
        source_count=source_count,
        split=str(runtime.dataset.split),
        indices=OmegaConf.select(cfg, "policy_eval.indices", default=None),
        start_index=int(OmegaConf.select(cfg, "policy_eval.start_index", default=0)),
        count=OmegaConf.select(cfg, "policy_eval.count", default=None),
    )
    if not selected:
        raise ValueError("policy_eval selected no episodes")
    for source_index in selected:
        reset_source_index = int(runtime.source_episode_indices[int(source_index)])
        _validate_object_suite_source_object(runtime.source, cfg, reset_source_index)

    backend = str(
        OmegaConf.select(cfg, "policy_eval.backend", default=str(cfg.env.backend))
    )
    viewer_enabled = bool(OmegaConf.select(cfg, "policy_eval.render", default=False))
    viewer_control = (
        ViewerReplayControl(
            start_paused=bool(
                OmegaConf.select(cfg, "policy_eval.viewer_start_paused", default=False)
            )
        )
        if viewer_enabled
        else None
    )
    save_vis_trace_npz = optional_project_path(
        OmegaConf.select(cfg, "policy_eval.save_vis_trace_npz", default=None)
    )
    save_results_npz = optional_project_path(
        OmegaConf.select(cfg, "policy_eval.save_results_npz", default=None)
    )
    save_training_pt = optional_project_path(
        OmegaConf.select(cfg, "policy_eval.save_training_pt", default=None)
    )
    store_trace = save_vis_trace_npz is not None
    capture_camera_every_controller_step = bool(
        OmegaConf.select(
            cfg, "policy_eval.capture_camera_every_controller_step", default=False
        )
    )
    continue_on_error = bool(
        OmegaConf.select(cfg, "policy_eval.continue_on_error", default=False)
    )
    progress_interval = int(
        OmegaConf.select(cfg, "policy_eval.progress_interval", default=1)
    )
    step_progress_interval = int(
        OmegaConf.select(cfg, "policy_eval.step_progress_interval", default=30)
    )
    max_policy_frames = _max_policy_frames_from_cfg(cfg)
    controller_steps_per_prediction = _positive_int_from_cfg(
        cfg,
        "policy_eval.controller_steps_per_prediction",
        default=1,
    )
    proprio_source = _proprio_source_from_cfg(cfg)
    initial_pose_mode = "source"
    initial_pose_path: Path | None = None
    initial_low_level_q: np.ndarray | None = None
    if runtime.action_space_type in HIGH_LOW_ACTION_SPACE_TYPES:
        initial_pose_mode, initial_pose_path, initial_low_level_q = (
            _initial_low_level_pose_from_cfg(cfg, runtime)
        )

    process_count = validate_headless_parallel(
        cfg,
        section="policy_eval",
        render=viewer_enabled,
        save_vis_trace_npz=save_vis_trace_npz,
    )

    print(
        f"loaded high_level_config={runtime.high_level_config_path} checkpoint={runtime.checkpoint_path} "
        f"payload={runtime.payload_path} split={runtime.dataset.split} selected={len(selected)} "
        f"backend={backend} embodiment={cfg.env.embodiment} action_space_type={runtime.action_space_type} "
        f"state_key={runtime.state_key}"
        f"{'' if save_training_pt is None else f' save_training_pt={save_training_pt}'}",
        flush=True,
    )
    if runtime.high_low_runtime is not None:
        print(
            f"low_level_config={runtime.low_level_config_path} "
            f"low_level_checkpoint={runtime.high_low_runtime.low_level_checkpoint_path} "
            f"low_level_manipulator={runtime.high_low_runtime.low_level_manipulator} "
            f"low_level_backend={runtime.high_low_runtime.encoder_backend} "
            f"reset_source={runtime.reset_source_path} initial_pose_mode={initial_pose_mode}"
            f"{'' if initial_pose_path is None else f' initial_pose_pt={initial_pose_path}'} "
            f"proprio_source={proprio_source} "
            f"controller_steps_per_prediction={controller_steps_per_prediction}",
            flush=True,
        )
        _warn_if_mutable_low_level_checkpoint_may_not_match_payload(
            payload_path=runtime.payload_path,
            checkpoint_path=Path(runtime.high_low_runtime.low_level_checkpoint_path),
        )

    if process_count > 1:
        print(
            f"running policy eval in parallel process_count={process_count} selected={len(selected)}",
            flush=True,
        )
        del runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with tempfile.TemporaryDirectory(
            prefix="tabletop_policy_eval_parallel_"
        ) as tmp:
            result_paths = launch_sharded_workers(
                script_path=Path(__file__).resolve(),
                cfg=cfg,
                section="policy_eval",
                selected=selected,
                process_count=process_count,
                tmpdir=Path(tmp),
                result_prefix="policy_eval",
                extra_worker_outputs=(
                    {"policy_eval.save_training_pt": ("policy_eval_training", ".pt")}
                    if save_training_pt is not None
                    else None
                ),
            )
            merged_payload = _merge_result_payloads(result_paths)
            training_payload_paths = sorted(
                Path(tmp).glob("policy_eval_training_worker_*.pt")
            )
            if save_training_pt is not None:
                out = _merge_training_payloads(training_payload_paths, save_training_pt)
                print(f"saved policy-eval training payload: {out}", flush=True)
        results = _results_from_payload(merged_payload)
        _print_summary(results)
        if save_results_npz is not None:
            out = _save_npz(save_results_npz, merged_payload)
            print(f"saved policy-eval results: {out}", flush=True)
        return

    print("building tabletop env", flush=True)
    env = build_env_from_config(
        cfg,
        backend=backend,
        render=viewer_enabled,
        viewer_key_callback=(
            viewer_control.key_callback if viewer_control is not None else None
        ),
    )
    if (
        str(OmegaConf.select(cfg, "task.name", default="")).strip()
        == "object_suite_pick_place"
    ):
        if (
            not bool(env.include_camera_obs)
            or not bool(env.object_only_pointcloud)
            or not bool(env.capture_camera_segmentation)
            or int(env.object_mask_erosion_pixels) != 1
            or abs(float(env.object_depth_neighbor_threshold_m) - 0.03) > 1.0e-12
            or env.object_bounds_filter_tolerance_m is None
            or abs(float(env.object_bounds_filter_tolerance_m) - 0.005) > 1.0e-12
            or str(OmegaConf.select(cfg, "perception.pointcloud_frame", default=""))
            .strip()
            .lower()
            != "world"
            or str(OmegaConf.select(cfg, "perception.pointcloud_centering", default=""))
            .strip()
            .lower()
            != "none"
        ):
            raise RuntimeError(
                "Object-suite evaluation requires the geometry-filtered, eroded object-mask-only camera point cloud "
                "in the fixed world frame without recentering"
            )
    if runtime.reset_source_kind == "robot_model_q":
        _validate_robot_model_q_source_matches_env(runtime.source, env)
        _validate_robot_model_source_env_dim(
            runtime.source,
            [int(runtime.source_episode_indices[int(idx)]) for idx in selected],
            env,
        )
    print(
        f"env ready selected={len(selected)} max_policy_frames={max_policy_frames} action_schedule="
        f"{OmegaConf.select(cfg, 'policy_eval.action_schedule', default='chunked')}",
        flush=True,
    )
    dt = float(env.model.opt.timestep) * int(env.backend.frame_skip)
    validate_paper_control_dt(cfg, dt)
    results: list[EvalResult] = []
    flat_trace: dict[str, list[np.ndarray]] = {}
    trajectory_frame_start: list[int] = []
    trajectory_frame_count: list[int] = []
    flat_frame_count = 0
    training_records: list[tuple[EvalTrainingEpisode, EvalResult]] = []

    try:
        for local_idx, source_index in enumerate(selected):
            trace = new_trace_buffers() if store_trace else None
            frame_start = flat_frame_count
            episode_start_time = time.perf_counter()
            print(
                f"episode {local_idx + 1}/{len(selected)} source_index={int(source_index)} start",
                flush=True,
            )
            try:
                training_episode = (
                    EvalTrainingEpisode(source_index=int(source_index))
                    if save_training_pt is not None
                    else None
                )
                result = _rollout_episode(
                    env,
                    runtime,
                    source_index,
                    max_policy_frames=max_policy_frames,
                    action_schedule=str(
                        OmegaConf.select(
                            cfg, "policy_eval.action_schedule", default="chunked"
                        )
                    ),
                    controller_steps_per_prediction=controller_steps_per_prediction,
                    proprio_source=proprio_source,
                    initial_low_level_q=initial_low_level_q,
                    store_trace=store_trace,
                    trace=trace,
                    viewer_control=viewer_control,
                    capture_camera_every_controller_step=capture_camera_every_controller_step,
                    training_episode=training_episode,
                    dt=dt,
                    step_progress_interval=step_progress_interval,
                    progress_label=f"episode {local_idx + 1}/{len(selected)} source_index={int(source_index)}",
                    profiler=RolloutProfiler(
                        enabled=bool(
                            OmegaConf.select(
                                cfg, "policy_eval.profile.enabled", default=False
                            )
                        ),
                        device=runtime.device,
                        label=f"episode {local_idx + 1}/{len(selected)} profile",
                    ),
                )
            except Exception as exc:
                if not continue_on_error:
                    raise RuntimeError(f"episode {source_index} failed") from exc
                result = EvalResult(
                    source_index=int(source_index),
                    source_seed=None,
                    completed_steps=0,
                    policy_call_count=0,
                    success=False,
                    max_cube_z=float("nan"),
                    final_cube_pose=np.full((7,), np.nan, dtype=np.float32),
                    error=f"{type(exc).__name__}: {exc}",
                )
                trace = None
                training_episode = None

            episode_elapsed = time.perf_counter() - episode_start_time
            _print_episode_result(result, local_idx + 1, len(selected), episode_elapsed)
            results.append(result)
            if (
                training_episode is not None
                and _training_episode_frame_count(training_episode) > 0
            ):
                training_records.append((training_episode, result))
            trajectory_frame_start.append(frame_start)
            trajectory_frame_count.append(int(result.completed_steps))
            if trace is not None:
                _extend_flat_trace(flat_trace, trace)
                flat_frame_count += int(result.completed_steps)

            if _should_report_progress(local_idx, len(selected), progress_interval):
                _print_progress(results, len(selected))
            if viewer_control is not None and (
                viewer_control.stop_requested or not env.viewer_is_running()
            ):
                break

        _print_summary(results)
        compact_payload = _compact_payload(
            results,
            cfg=cfg,
            runtime=runtime,
            env=env,
            selected=selected,
            source_count=source_count,
            backend=backend,
            dt=dt,
            initial_pose_mode=initial_pose_mode,
            initial_pose_path=initial_pose_path,
            initial_low_level_q=initial_low_level_q,
        )
        if save_results_npz is not None:
            out = _save_npz(save_results_npz, compact_payload)
            print(f"saved policy-eval results: {out}", flush=True)
        if save_training_pt is not None:
            out = _save_eval_training_payload(
                save_training_pt,
                training_records=training_records,
                cfg=cfg,
                env=env,
                runtime=runtime,
                backend=backend,
                dt=dt,
                initial_pose_mode=initial_pose_mode,
                initial_pose_path=initial_pose_path,
                initial_low_level_q=initial_low_level_q,
                proprio_source=proprio_source,
                controller_steps_per_prediction=controller_steps_per_prediction,
            )
            print(f"saved policy-eval training payload: {out}", flush=True)
        if save_vis_trace_npz is not None:
            trace_extra = dict(compact_payload)
            trace_extra.pop("schema_version", None)
            trace_extra["trajectory_frame_start"] = np.asarray(
                trajectory_frame_start, dtype=np.int32
            )
            trace_extra["trajectory_frame_count"] = np.asarray(
                trajectory_frame_count, dtype=np.int32
            )
            out = save_trace_npz(
                save_vis_trace_npz,
                env=env,
                trace=flat_trace,
                backend=backend,
                mode=str(cfg.env.mode),
                dt=dt,
                extra=trace_extra,
                include_camera_payload=True,
            )
            print(f"saved policy-eval trace: {out}", flush=True)
        if viewer_control is not None and not bool(
            OmegaConf.select(cfg, "policy_eval.viewer_no_hold_open", default=False)
        ):
            viewer_control.hold_open(
                env,
                completed_steps=sum(result.completed_steps for result in results),
                total_steps=len(selected)
                * max_policy_frames
                * controller_steps_per_prediction,
            )
    finally:
        env.close()


def _load_policy_runtime(
    cfg: DictConfig, *, high_cfg_path: Path, high_cfg: DictConfig
) -> PolicyRuntime:
    from experiment.high_level_policy.idp3_adapter import (
        build_idp3_policy,
        idp3_model_cfg,
    )

    payload_override = OmegaConf.select(cfg, "policy_eval.payload_path", default=None)
    payload_path = _resolve_experiment_path(
        (
            payload_override
            if payload_override is not None
            and str(payload_override).strip().lower() not in {"", "null", "none"}
            else OmegaConf.select(high_cfg, "dataset.payload_path", default=None)
        ),
        label="policy_eval.payload_path or high-level dataset.payload_path",
    )
    payload = load_high_level_payload(payload_path)
    action_space_name = _cfg_action_space(high_cfg)
    model_cfg = idp3_model_cfg(high_cfg)
    dataset = HighLevelSequenceDataset(
        payload_path=payload_path,
        action_space=action_space_name,
        horizon=int(model_cfg.horizon),
        pad_before=int(model_cfg.n_obs_steps) - 1,
        pad_after=int(model_cfg.n_action_steps) - 1,
        split=str(OmegaConf.select(cfg, "policy_eval.split", default="test")),
        test_episode_count=int(
            OmegaConf.select(high_cfg, "dataset.test_episode_count", default=20)
        ),
        split_seed=int(OmegaConf.select(high_cfg, "dataset.split_seed", default=42)),
        split_group_key=OmegaConf.select(
            high_cfg, "dataset.split_group_key", default=None
        ),
        allow_empty=True,
    )
    action_space = dict(payload["meta"].get("action_space", {}))
    action_space_type = str(action_space.get("type", "")).strip()
    source, reset_source_path, source_episode_indices, reset_source_kind = (
        _reset_source_for_payload(
            cfg,
            payload,
            payload_path=payload_path,
            action_space_type=action_space_type,
        )
    )
    checkpoint_path = _resolve_checkpoint_path(cfg, high_cfg)
    device_value = OmegaConf.select(cfg, "policy_eval.device", default=None)
    if device_value is None or str(device_value).strip().lower() in {
        "",
        "null",
        "none",
    }:
        device_value = OmegaConf.select(high_cfg, "training.device", default="cuda:0")
    device = prepare_device(str(device_value))

    if action_space_type in HIGH_LOW_ACTION_SPACE_TYPES:
        low_level_config_path = _resolve_low_level_config_path(
            high_cfg, high_level_config_path=high_cfg_path
        )
        low_level_checkpoint_path = _resolve_low_level_checkpoint_path(
            high_cfg,
            high_level_config_path=high_cfg_path,
        )
        low_level_manipulator = _resolve_low_level_manipulator(
            high_cfg,
            env_embodiment=str(cfg.env.embodiment),
        )
        high_low_runtime = load_high_low_pipeline_runtime(
            high_level_payload=payload,
            high_level_checkpoint_path=checkpoint_path,
            low_level_config_path=low_level_config_path,
            low_level_checkpoint_path=low_level_checkpoint_path,
            low_level_manipulator=low_level_manipulator,
            device=device,
            high_level_use_ema=bool(
                OmegaConf.select(cfg, "policy_eval.use_ema", default=True)
            ),
            low_level_use_ema=bool(
                OmegaConf.select(high_cfg, "preprocess.latent_prefer_ema", default=True)
            ),
        )
        state_key = _checkpoint_state_key(
            checkpoint_path,
            use_ema=bool(OmegaConf.select(cfg, "policy_eval.use_ema", default=True)),
        )
        return PolicyRuntime(
            high_level_config_path=high_cfg_path,
            high_level_cfg=high_cfg,
            checkpoint_path=checkpoint_path,
            payload_path=payload_path,
            reset_source_path=reset_source_path,
            payload=payload,
            source=source,
            dataset=dataset,
            policy=high_low_runtime.high_level_policy,
            device=device,
            action_space=action_space,
            state_key=state_key,
            action_space_type=action_space_type,
            source_episode_indices=source_episode_indices,
            episode_count=len(payload["episode_meta"]),
            reset_source_kind=reset_source_kind,
            low_level_config_path=low_level_config_path,
            high_low_runtime=high_low_runtime,
        )

    if action_space_type != "single_manipulator":
        raise ValueError(
            "Closed-loop tabletop high-level evaluation supports meta.action_space.type="
            f"'single_manipulator', 'latent_contact', or 'contact_points', got {action_space_type!r}"
        )

    checkpoint = torch.load(
        str(checkpoint_path), map_location=device, weights_only=False
    )

    build_cfg = high_cfg
    ckpt_cfg = checkpoint.get("config")
    if isinstance(ckpt_cfg, dict):
        build_cfg = OmegaConf.create(ckpt_cfg)
        OmegaConf.resolve(build_cfg)

    policy = build_idp3_policy(
        build_cfg,
        action_dim=int(dataset.action_dim),
        state_dim=int(dataset.state_dim),
        point_count=int(dataset.num_points),
    )
    policy = policy.to(device=device, dtype=torch.float32)
    state_key = (
        "ema_model"
        if bool(OmegaConf.select(cfg, "policy_eval.use_ema", default=True))
        and "ema_model" in checkpoint
        else "model"
    )
    if state_key not in checkpoint:
        raise KeyError(
            f"Checkpoint {checkpoint_path} is missing required state key {state_key!r}"
        )
    policy.load_state_dict(checkpoint[state_key], strict=True)
    if "normalizer" not in checkpoint:
        raise KeyError(
            f"Checkpoint {checkpoint_path} is missing saved high-level normalizer"
        )
    policy.normalizer.load_state_dict(checkpoint["normalizer"])
    policy = policy.to(device=device, dtype=torch.float32)
    policy.eval()

    return PolicyRuntime(
        high_level_config_path=high_cfg_path,
        high_level_cfg=high_cfg,
        checkpoint_path=checkpoint_path,
        payload_path=payload_path,
        reset_source_path=reset_source_path,
        payload=payload,
        source=source,
        dataset=dataset,
        policy=policy,
        device=device,
        action_space=action_space,
        state_key=state_key,
        action_space_type=action_space_type,
        source_episode_indices=source_episode_indices,
        episode_count=len(payload["episode_meta"]),
        reset_source_kind=reset_source_kind,
        low_level_config_path=None,
        high_low_runtime=None,
    )


def _checkpoint_state_key(checkpoint_path: Path, *, use_ema: bool) -> str:
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    return "ema_model" if bool(use_ema) and "ema_model" in checkpoint else "model"


def _warn_if_mutable_low_level_checkpoint_may_not_match_payload(
    *,
    payload_path: Path,
    checkpoint_path: Path,
) -> None:
    checkpoint_name = str(checkpoint_path.name).strip().lower()
    if checkpoint_name not in {"last.pt", "best.pt"}:
        return
    try:
        payload_mtime = float(Path(payload_path).stat().st_mtime)
        checkpoint_mtime = float(Path(checkpoint_path).stat().st_mtime)
    except OSError:
        return
    if checkpoint_mtime <= payload_mtime + 1.0:
        return
    warnings.warn(
        "Latent-contact payload points at a mutable low-level checkpoint that is newer than the payload: "
        f"payload={payload_path} checkpoint={checkpoint_path}. If this checkpoint was overwritten after "
        "preprocessing/training, the high-level policy can have low training loss while eval decodes its "
        "latents with a different low-level decoder. Use the exact low-level checkpoint snapshot used to "
        "generate the payload, or regenerate the latent payload and retrain.",
        RuntimeWarning,
        stacklevel=2,
    )


def _payload_source_episode_indices(
    payload: dict[str, Any], *, source_count: int
) -> np.ndarray:
    episode_count = len(list(payload["episode_meta"]))
    meta = dict(payload.get("meta", {}))
    source_meta = (
        meta.get("source", {}) if isinstance(meta.get("source", {}), dict) else {}
    )
    candidates: list[Any] = [
        meta.get("source_indices"),
        source_meta.get("source_indices"),
        [
            dict(item).get("source_trajectory_index")
            for item in payload.get("episode_meta", [])
        ],
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, list) and any(item is None for item in candidate):
            continue
        arr = np.asarray(candidate, dtype=np.int64).reshape(-1)
        if int(arr.shape[0]) != int(episode_count):
            continue
        if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= int(source_count)):
            raise IndexError(
                f"Payload source episode mapping has values outside reset source range 0..{source_count - 1}: "
                f"min={int(arr.min())} max={int(arr.max())}"
            )
        return arr.astype(np.int64, copy=False)
    if int(episode_count) > int(source_count):
        raise ValueError(
            f"Payload has {episode_count} episodes but reset source has only {source_count}, and no valid "
            "source episode mapping was found in payload metadata."
        )
    return np.arange(int(episode_count), dtype=np.int64)


def _latent_source_payload_path(payload: dict[str, Any], *, payload_path: Path) -> Path:
    meta = dict(payload.get("meta", {}))
    source_meta = (
        meta.get("source", {}) if isinstance(meta.get("source", {}), dict) else {}
    )
    for value in (
        meta.get("source_pt_path"),
        source_meta.get("path"),
        _first_episode_meta_value(payload, "source_pt_path"),
    ):
        if value is None or str(value).strip().lower() in {"", "null", "none"}:
            continue
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (payload_path.parent / path).resolve()
    raise ValueError(
        "Latent-contact high-level eval needs the source single-manipulator .pt path in "
        "meta.source_pt_path, meta.source.path, or episode_meta[*].source_pt_path so cube resets can match "
        "the training trajectories."
    )


def _first_episode_meta_value(payload: dict[str, Any], key: str) -> Any | None:
    for item in payload.get("episode_meta", []):
        if isinstance(item, dict) and item.get(key) is not None:
            return item.get(key)
    return None


def _reset_source_for_payload(
    cfg: DictConfig,
    payload: dict[str, Any],
    *,
    payload_path: Path,
    action_space_type: str,
) -> tuple[dict[str, np.ndarray], Path, np.ndarray, str]:
    if action_space_type == "single_manipulator":
        source = single_manipulator_payload_to_replay_source(
            payload, source_path=payload_path
        )
        indices = _payload_source_episode_indices(
            payload, source_count=_trajectory_count(source)
        )
        return source, payload_path, indices, "single_manipulator"
    if action_space_type in HIGH_LOW_ACTION_SPACE_TYPES:
        env_embodiment = str(
            OmegaConf.select(cfg, "env.embodiment", default="")
        ).strip()
        default_object_reset_source = (
            "robot_model_q_source"
            if env_embodiment in {"allegro_right", "barrett"}
            else "retargeted_wuji"
        )
        object_reset_source = (
            str(
                OmegaConf.select(
                    cfg,
                    "policy_eval.object_reset_source",
                    default=default_object_reset_source,
                )
            )
            .strip()
            .lower()
        )
        if object_reset_source == "latent_source":
            initial_pose_mode = (
                str(
                    OmegaConf.select(
                        cfg,
                        "policy_eval.initial_pose.mode",
                        default="robot_model_q_source",
                    )
                )
                .strip()
                .lower()
            )
            if initial_pose_mode != "saved":
                raise ValueError(
                    "policy_eval.object_reset_source='latent_source' requires "
                    "policy_eval.initial_pose.mode='saved'"
                )
            source_path = _latent_source_payload_path(
                payload, payload_path=payload_path
            )
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Latent source payload not found: {source_path}"
                )
            source_payload = load_high_level_payload(source_path)
            source = single_manipulator_payload_to_replay_source(
                source_payload,
                source_path=source_path,
            )
            source_action_space_type = str(
                _source_scalar(source, "action_space_type") or ""
            ).strip()
            if source_action_space_type != "single_manipulator":
                raise ValueError(
                    "policy_eval.object_reset_source='latent_source' requires the latent payload's "
                    f"source to use action_space.type='single_manipulator', got {source_action_space_type!r}"
                )
            if "object_pose" not in source and "cube_pose" not in source:
                raise KeyError(
                    f"Latent source payload {source_path} must contain object_pose or cube_pose for reset"
                )
            indices = _payload_source_episode_indices(
                payload, source_count=_trajectory_count(source)
            )
            return source, source_path, indices, "latent_source_object_pose"

        if env_embodiment in {"allegro_right", "barrett"}:
            if object_reset_source != "robot_model_q_source":
                raise ValueError(
                    "policy_eval.object_reset_source for floating Allegro/Barrett evaluation must be "
                    f"'robot_model_q_source' or 'latent_source', got {object_reset_source!r}"
                )
            robot_model_source_path = _resolve_robot_model_q_source_payload_path(
                cfg,
                payload_path=payload_path,
                env_embodiment=env_embodiment,
            )
            robot_model_payload = load_high_level_payload(robot_model_source_path)
            source = single_manipulator_payload_to_replay_source(
                robot_model_payload,
                source_path=robot_model_source_path,
            )
            _validate_robot_model_q_source_basic(
                source,
                embodiment=env_embodiment,
                source_path=robot_model_source_path,
            )
            indices = _payload_source_episode_indices(
                payload, source_count=_trajectory_count(source)
            )
            return source, robot_model_source_path, indices, "robot_model_q"

        if object_reset_source not in {"retargeted", "retargeted_wuji"}:
            raise ValueError(
                "policy_eval.object_reset_source for Wuji evaluation must be "
                f"'retargeted_wuji' or 'latent_source', got {object_reset_source!r}"
            )

        retarget_path = _resolve_retargeted_wuji_payload_path(
            cfg, payload_path=payload_path
        )
        retarget_payload = load_high_level_payload(retarget_path)
        source = single_manipulator_payload_to_replay_source(
            retarget_payload, source_path=retarget_path
        )
        high_level_source_count = _latent_source_episode_count(
            payload, payload_path=payload_path
        )
        source_indices = _payload_source_episode_indices(
            payload, source_count=high_level_source_count
        )
        retarget_map = _retarget_episode_map(
            retarget_payload, _trajectory_count(source)
        )
        missing = [int(idx) for idx in source_indices if int(idx) not in retarget_map]
        if missing:
            raise KeyError(
                "Retargeted Wuji payload is missing episodes for high-level source_trajectory_index values: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )
        indices = np.asarray(
            [retarget_map[int(idx)] for idx in source_indices], dtype=np.int64
        )
        return source, retarget_path, indices, "retargeted_wuji"
    raise ValueError(
        f"Unsupported high-level payload action_space.type={action_space_type!r}"
    )


def _latent_source_episode_count(payload: dict[str, Any], *, payload_path: Path) -> int:
    try:
        source_path = _latent_source_payload_path(payload, payload_path=payload_path)
        source_payload = load_high_level_payload(source_path)
        source = single_manipulator_payload_to_replay_source(
            source_payload, source_path=source_path
        )
        return int(_trajectory_count(source))
    except Exception:
        return len(list(payload["episode_meta"]))


def _resolve_retargeted_wuji_payload_path(
    cfg: DictConfig, *, payload_path: Path
) -> Path:
    explicit = _optional_project_path_value(
        OmegaConf.select(cfg, "policy_eval.retargeted_wuji_pt", default=None)
    )
    if explicit is None:
        raise ValueError(
            "Latent-contact high-level eval needs policy_eval.retargeted_wuji_pt so simulator resets and "
            "initial Wuji poses match the low-level online rollout."
        )
    if not explicit.is_file():
        raise FileNotFoundError(f"policy_eval.retargeted_wuji_pt not found: {explicit}")
    return explicit


def _resolve_robot_model_q_source_payload_path(
    cfg: DictConfig,
    *,
    payload_path: Path,
    env_embodiment: str,
) -> Path:
    explicit = _optional_project_path_value(
        OmegaConf.select(cfg, "policy_eval.robot_model_q_source_pt", default=None)
    )
    if explicit is None:
        raise ValueError(
            f"Latent-contact high-level eval for env.embodiment={env_embodiment!r} needs "
            "policy_eval.robot_model_q_source_pt so simulator resets and initial floating-hand poses match "
            f"the high-level payload {payload_path}."
        )
    if not explicit.is_file():
        raise FileNotFoundError(
            f"policy_eval.robot_model_q_source_pt not found: {explicit}"
        )
    return explicit


def _optional_project_path_value(value: Any) -> Path | None:
    if value is None or str(value).strip().lower() in {"", "null", "none"}:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def _resolve_low_level_config_path(
    high_cfg: DictConfig, *, high_level_config_path: Path
) -> Path:
    resolved = _optional_project_path_value(
        OmegaConf.select(high_cfg, "preprocess.action_policy_config", default=None)
    )
    if resolved is not None:
        if not resolved.is_file():
            raise FileNotFoundError(
                f"preprocess.action_policy_config not found: {resolved}"
            )
        return resolved
    raise ValueError(
        f"Could not resolve a low-level config from {high_level_config_path}; "
        "set preprocess.action_policy_config in the high-level config."
    )


def _resolve_low_level_checkpoint_path(
    high_cfg: DictConfig,
    *,
    high_level_config_path: Path,
) -> Path | None:
    checkpoint = _optional_project_path_value(
        OmegaConf.select(
            high_cfg, "preprocess.latent_contact_encoder_checkpoint", default=None
        )
    )
    if checkpoint is None:
        return None
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"preprocess.latent_contact_encoder_checkpoint from {high_level_config_path} not found: {checkpoint}"
        )
    return checkpoint


def _resolve_low_level_manipulator(high_cfg: DictConfig, *, env_embodiment: str) -> str:
    requested = str(
        OmegaConf.select(high_cfg, "preprocess.manipulator_name", default="auto")
    ).strip()
    if requested and requested.lower() != "auto":
        return requested
    embodiment = str(env_embodiment).strip()
    if embodiment == "marvin_wuji":
        return "wujihand_right"
    if embodiment == "franka_robotiq":
        raise ValueError(
            "preprocess.manipulator_name=auto cannot run latent_contact eval with env.embodiment='franka_robotiq': "
            "the kinematic decoder bank has no Franka head. "
            "Use env.embodiment=marvin_wuji or set preprocess.manipulator_name in the high-level config."
        )
    if embodiment in {"allegro_right", "barrett", "wujihand_right"}:
        return embodiment
    raise ValueError(
        f"preprocess.manipulator_name=auto does not know how to map env.embodiment={embodiment!r}"
    )


def _max_policy_frames_from_cfg(cfg: DictConfig) -> int:
    value = OmegaConf.select(cfg, "policy_eval.max_frames", default=None)
    if value is None or str(value).strip().lower() in {"", "null", "none"}:
        value = OmegaConf.select(cfg, "policy_eval.max_steps", default=120)
    frames = int(value)
    if frames < 1:
        raise ValueError(f"policy_eval.max_frames/max_steps must be >= 1, got {frames}")
    return frames


def _positive_int_from_cfg(cfg: DictConfig, key: str, *, default: int) -> int:
    value = int(OmegaConf.select(cfg, key, default=default))
    if value < 1:
        raise ValueError(f"{key} must be >= 1, got {value}")
    return value


def _proprio_source_from_cfg(cfg: DictConfig) -> str:
    value = (
        str(OmegaConf.select(cfg, "policy_eval.proprio_source", default="executed"))
        .strip()
        .lower()
    )
    if value not in {"executed", "target"}:
        raise ValueError(
            f"policy_eval.proprio_source must be 'executed' or 'target', got {value!r}"
        )
    return value


def _initial_low_level_pose_from_cfg(
    cfg: DictConfig,
    runtime: PolicyRuntime,
) -> tuple[str, Path | None, np.ndarray | None]:
    latent_runtime = runtime.high_low_runtime
    if latent_runtime is None:
        raise RuntimeError("Internal error: latent-contact runtime is missing")
    manipulator = str(latent_runtime.low_level_manipulator).strip().lower()
    env_embodiment = str(OmegaConf.select(cfg, "env.embodiment", default="")).strip()
    if manipulator == "wujihand_right":
        if env_embodiment not in {"marvin_wuji", "wujihand_right"}:
            raise ValueError(
                "low_level_manipulator='wujihand_right' requires env.embodiment "
                f"'marvin_wuji' or 'wujihand_right', got {env_embodiment!r}"
            )
        return _initial_wuji_pose_from_cfg(cfg)
    if manipulator in {"allegro_right", "barrett"}:
        if env_embodiment != manipulator:
            raise ValueError(
                f"low_level_manipulator={manipulator!r} requires matching env.embodiment, got {env_embodiment!r}"
            )
        return _initial_robot_model_q_pose_from_cfg(
            cfg,
            runtime=latent_runtime,
            embodiment=manipulator,
            source_path=runtime.reset_source_path,
        )
    raise ValueError(
        f"Latent tabletop eval does not know how to materialize initial pose for "
        f"low_level_manipulator={latent_runtime.low_level_manipulator!r}"
    )


def _initial_wuji_pose_from_cfg(
    cfg: DictConfig,
) -> tuple[str, Path | None, np.ndarray | None]:
    mode = (
        str(
            OmegaConf.select(cfg, "policy_eval.initial_pose.mode", default="retargeted")
        )
        .strip()
        .lower()
    )
    if mode not in {"saved", "retargeted"}:
        raise ValueError(
            f"policy_eval.initial_pose.mode must be 'saved' or 'retargeted', got {mode!r}"
        )
    if mode == "retargeted":
        return mode, None, None
    resolution = resolve_start_pose_path(
        "wuji",
        _start_pose_task_from_cfg(cfg, section="policy_eval", default=TASK_CUBE_PICK),
        configured_pose_pt=OmegaConf.select(
            cfg,
            "policy_eval.initial_pose.pose_pt",
            default=DEFAULT_WUJI_START_POSE_RELATIVE,
        ),
        explicit_pose_pt=False,
        prefer_task_specific=bool(
            OmegaConf.select(
                cfg, "policy_eval.initial_pose.prefer_task_start_pose", default=True
            )
        ),
        default_task=TASK_CUBE_PICK,
    )
    pose_path = resolution.path
    if not pose_path.is_file():
        raise FileNotFoundError(
            f"policy_eval.initial_pose.pose_pt not found: {pose_path}"
        )
    print(
        f"resolved Wuji initial pose: {describe_start_pose_resolution(resolution)}",
        flush=True,
    )
    return (
        mode,
        pose_path,
        validate_wuji_q(load_wuji_start_q(pose_path), label=str(pose_path)),
    )


def _initial_robot_model_q_pose_from_cfg(
    cfg: DictConfig,
    *,
    runtime: HighLowPipelineRuntime,
    embodiment: str,
    source_path: Path,
) -> tuple[str, Path | None, np.ndarray | None]:
    mode = (
        str(
            OmegaConf.select(
                cfg, "policy_eval.initial_pose.mode", default="robot_model_q_source"
            )
        )
        .strip()
        .lower()
    )
    if mode in {"retargeted", "source", "robot_model_q_source"}:
        return "robot_model_q_source", source_path, None
    if mode != "saved":
        raise ValueError(
            "policy_eval.initial_pose.mode for floating robot_model_q hands must be "
            f"'saved' or 'robot_model_q_source', got {mode!r}"
        )
    resolution = resolve_start_pose_path(
        embodiment,
        _start_pose_task_from_cfg(cfg, section="policy_eval", default=TASK_CUBE_PICK),
        configured_pose_pt=OmegaConf.select(
            cfg, "policy_eval.initial_pose.pose_pt", default=None
        ),
        explicit_pose_pt=False,
        prefer_task_specific=bool(
            OmegaConf.select(
                cfg, "policy_eval.initial_pose.prefer_task_start_pose", default=True
            )
        ),
        default_task=TASK_CUBE_PICK,
    )
    pose_path = resolution.path
    if not pose_path.is_file():
        raise FileNotFoundError(
            f"policy_eval.initial_pose.pose_pt not found: {pose_path}"
        )
    print(
        f"resolved {embodiment} initial pose: {describe_start_pose_resolution(resolution)}",
        flush=True,
    )
    return (
        "saved",
        pose_path,
        _load_robot_model_start_q(
            pose_path,
            runtime=runtime,
            expected_robot_name=str(embodiment),
        ),
    )


def _start_pose_task_from_cfg(cfg: DictConfig, *, section: str, default: str) -> str:
    explicit = OmegaConf.select(cfg, f"{section}.initial_pose.task", default=None)
    if explicit is not None and str(explicit).strip().lower() not in {
        "",
        "none",
        "null",
        "auto",
    }:
        return str(explicit)
    task_name = OmegaConf.select(cfg, "task.name", default=None)
    if task_name is not None and str(task_name).strip().lower() not in {
        "",
        "none",
        "null",
        "auto",
    }:
        return str(task_name)
    return str(default)


def _rollout_episode(
    env: TabletopManipulationEnv,
    runtime: PolicyRuntime,
    source_index: int,
    *,
    max_policy_frames: int,
    action_schedule: str,
    controller_steps_per_prediction: int,
    proprio_source: str,
    initial_low_level_q: np.ndarray | None,
    store_trace: bool,
    trace: dict[str, list[np.ndarray]] | None,
    viewer_control: ViewerReplayControl | None,
    capture_camera_every_controller_step: bool,
    training_episode: EvalTrainingEpisode | None,
    dt: float,
    step_progress_interval: int,
    progress_label: str,
    profiler: RolloutProfiler | None = None,
) -> EvalResult:
    profiler = profiler or RolloutProfiler(enabled=False)
    schedule = str(action_schedule).strip().lower()
    if schedule not in {"chunked", "replan_every_step"}:
        raise ValueError(
            "policy_eval.action_schedule must be 'chunked' or 'replan_every_step'"
        )
    reset_source_index = int(runtime.source_episode_indices[int(source_index)])
    latent_runtime = runtime.high_low_runtime
    if training_episode is not None:
        training_episode.reset_source_index = int(reset_source_index)
    with profiler.measure("reset"):
        if runtime.action_space_type in HIGH_LOW_ACTION_SPACE_TYPES:
            if latent_runtime is None:
                raise RuntimeError(
                    "Internal error: high-low runtime is missing the low-level policy runtime"
                )
            if runtime.reset_source_kind == "robot_model_q":
                obs, source_seed = _reset_from_robot_model_source(
                    env, runtime.source, reset_source_index
                )
                initial_q = (
                    _validate_robot_model_q(
                        initial_low_level_q,
                        expected_dim=int(latent_runtime.local_dof),
                        label="initial_robot_model_q",
                    )
                    if initial_low_level_q is not None
                    else _initial_robot_model_q_from_source(
                        runtime.source,
                        reset_source_index,
                        expected_dim=int(latent_runtime.local_dof),
                    )
                )
                _set_env_robot_model_q(env, initial_q)
            elif runtime.reset_source_kind == "latent_source_object_pose":
                obs, source_seed = _reset_from_source(
                    env, runtime.source, reset_source_index
                )
                if initial_low_level_q is None:
                    raise ValueError(
                        "A saved low-level pose is required when reset_source_kind="
                        "'latent_source_object_pose'"
                    )
                if (
                    str(latent_runtime.low_level_manipulator).strip().lower()
                    == "wujihand_right"
                    and str(env.embodiment) == "marvin_wuji"
                ):
                    initial_q = validate_wuji_q(
                        initial_low_level_q, label="initial_wuji_q"
                    )
                    _set_env_wuji_q(env, initial_q, latent_runtime)
                else:
                    initial_q = _validate_robot_model_q(
                        initial_low_level_q,
                        expected_dim=int(latent_runtime.local_dof),
                        label="initial_robot_model_q",
                    )
                    _set_env_robot_model_q(env, initial_q)
            elif runtime.reset_source_kind == "retargeted_wuji":
                obs, source_seed = _reset_from_retarget_source(
                    env, runtime.source, reset_source_index
                )
                initial_q = (
                    validate_wuji_q(initial_low_level_q, label="initial_wuji_q")
                    if initial_low_level_q is not None
                    else _initial_wuji_q_from_source(runtime.source, reset_source_index)
                )
                _set_env_wuji_q(env, initial_q, latent_runtime)
            else:
                raise ValueError(
                    f"Unsupported high-low reset_source_kind={runtime.reset_source_kind!r}"
                )
            obs = env.observe()
            current_q_world = _live_state_to_low_level_q(env, latent_runtime)
            target_feedback_q_world = (
                np.asarray(initial_q, dtype=np.float32).reshape(-1).copy()
            )
        else:
            obs, source_seed = _reset_from_source(
                env, runtime.source, reset_source_index
            )
            current_q_world = None
            target_feedback_q_world = None
    if training_episode is not None:
        training_episode.source_seed = source_seed

    final_cube_pose = _object_pose_from_obs(obs)
    max_cube_z = float(final_cube_pose[2])
    success = task_success_from_observation(env.task, obs, max_object_z=max_cube_z)
    completed_steps = 0
    policy_frame_count = 0
    policy_call_count = 0
    n_obs_steps = int(runtime.policy.n_obs_steps)
    repeat_count = int(controller_steps_per_prediction)
    max_env_steps = int(max_policy_frames) * repeat_count

    point_history: deque[np.ndarray] = deque(maxlen=n_obs_steps)
    agent_history: deque[np.ndarray] = deque(maxlen=n_obs_steps)
    initial_cloud = _point_cloud_from_obs(
        obs,
        expected_points=runtime.observation_point_count,
        prefer_env_pointcloud=bool(env.object_only_pointcloud),
    )
    if runtime.action_space_type in HIGH_LOW_ACTION_SPACE_TYPES:
        initial_agent = _initial_latent_proprio(runtime)
    else:
        initial_agent = _flat_action_from_live_state(env, runtime.action_space)
    for _ in range(n_obs_steps):
        point_history.append(initial_cloud.copy())
        agent_history.append(initial_agent.copy())

    with torch.inference_mode():
        while policy_frame_count < int(max_policy_frames) and not success:
            with profiler.measure("high_level_input"):
                policy_input = _history_to_policy_input(
                    point_history,
                    agent_history,
                    device=runtime.device,
                )
            with profiler.measure("high_level_policy"):
                predicted = runtime.policy.predict_action(policy_input)
            action_chunk = (
                predicted["action"][0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            policy_call_count += 1
            if action_chunk.ndim != 2 or action_chunk.shape[1] != int(
                runtime.dataset.action_dim
            ):
                raise ValueError(
                    f"Policy returned action chunk shape {action_chunk.shape}, expected (*, {runtime.dataset.action_dim})"
                )

            for chunk_idx, flat_action in enumerate(action_chunk):
                if policy_frame_count >= int(max_policy_frames) or success:
                    break
                if viewer_control is not None:
                    if not viewer_control.wait_before_step(
                        env,
                        step_idx=policy_frame_count,
                        total_steps=int(max_policy_frames),
                        trajectory_index=int(source_index),
                    ):
                        break
                q_input_world = None
                q_target_world = None
                if runtime.action_space_type in HIGH_LOW_ACTION_SPACE_TYPES:
                    if (
                        latent_runtime is None
                        or current_q_world is None
                        or target_feedback_q_world is None
                    ):
                        raise RuntimeError(
                            "Internal error: high-low rollout has no low-level q state"
                        )
                    q_input_world = (
                        target_feedback_q_world
                        if str(proprio_source) == "target"
                        else current_q_world
                    )
                    with profiler.measure("low_level_decode"):
                        if runtime.action_space_type == "latent_contact":
                            q_pred = predict_low_level_q_from_latent_action(
                                runtime=latent_runtime,
                                latent_action=flat_action,
                                q_prev=q_input_world,
                                seed=int(source_index) * 1000003
                                + int(policy_frame_count),
                            )
                        else:
                            q_pred = predict_low_level_q_from_contact_action(
                                runtime=latent_runtime,
                                contact_action=flat_action,
                                q_prev=q_input_world,
                                seed=int(source_index) * 1000003
                                + int(policy_frame_count),
                            )
                    q_target_world = _postprocess_low_level_q_target(
                        q_pred,
                        env=env,
                        runtime=latent_runtime,
                        reference=q_input_world,
                    )
                    target_feedback_q_world = q_target_world.copy()
                    action = _low_level_q_to_env_action(
                        q_target_world, env, latent_runtime
                    )
                else:
                    action = _flat_action_to_env_action(
                        flat_action, env, runtime.action_space
                    )

                last_info: dict[str, Any] | None = None
                step_executed = False
                for repeat_idx in range(repeat_count):
                    capture_camera = bool(
                        capture_camera_every_controller_step
                        or store_trace
                        or training_episode is not None
                        or repeat_idx + 1 >= repeat_count
                    )
                    with profiler.measure(
                        "env_step_with_camera"
                        if capture_camera
                        else "env_step_state_only"
                    ):
                        obs, _, _, _, info = env.step(
                            action, include_camera_obs=capture_camera
                        )
                    last_info = info
                    step_executed = True
                    completed_steps += 1
                    final_cube_pose = _object_pose_from_obs(obs)
                    max_cube_z = max(max_cube_z, float(final_cube_pose[2]))
                    success = accumulate_task_success(
                        success,
                        env.task,
                        obs,
                        max_object_z=max_cube_z,
                    )
                    if runtime.action_space_type in HIGH_LOW_ACTION_SPACE_TYPES:
                        if latent_runtime is None:
                            raise RuntimeError(
                                "Internal error: high-low runtime has disappeared"
                            )
                        current_q_world = _live_state_to_low_level_q(
                            env, latent_runtime
                        )
                    q_executed_world = current_q_world
                    if runtime.action_space_type not in HIGH_LOW_ACTION_SPACE_TYPES:
                        q_executed_world = _flat_action_from_live_state(
                            env, runtime.action_space
                        )

                    if training_episode is not None:
                        _append_eval_training_frame(
                            training_episode,
                            env=env,
                            runtime=runtime,
                            obs=obs,
                            info=info,
                            policy_frame=policy_frame_count,
                            chunk_idx=chunk_idx,
                            repeat_idx=repeat_idx,
                            flat_action=flat_action,
                            q_input_world=q_input_world,
                            q_target_world=q_target_world,
                            q_executed_world=q_executed_world,
                            proprio_source=proprio_source,
                            controller_steps_per_prediction=repeat_count,
                            success=success,
                            max_cube_z=max_cube_z,
                        )

                    if (
                        int(step_progress_interval) > 0
                        and completed_steps % int(step_progress_interval) == 0
                    ):
                        print(
                            f"{progress_label} step={completed_steps}/{max_env_steps} "
                            f"policy_frame={policy_frame_count + 1}/{int(max_policy_frames)} "
                            f"success={bool(success)} max_cube_z={max_cube_z:.4f}",
                            flush=True,
                        )

                    if store_trace and trace is not None:
                        append_trace_frame(trace, env, info, obs=obs)
                        if runtime.action_space_type in HIGH_LOW_ACTION_SPACE_TYPES:
                            _append_latent_trace_fields(
                                trace,
                                policy_frame=policy_frame_count,
                                chunk_idx=chunk_idx,
                                repeat_idx=repeat_idx,
                                latent_action=flat_action,
                                q_input_world=q_input_world,
                                q_target_world=q_target_world,
                                proprio_source=proprio_source,
                                controller_steps_per_prediction=repeat_count,
                            )
                    if viewer_control is not None:
                        viewer_control.update_overlay(
                            env,
                            step_idx=policy_frame_count + 1,
                            total_steps=int(max_policy_frames),
                            trajectory_index=int(source_index),
                        )
                        if viewer_control.stop_requested or not env.viewer_is_running():
                            break
                        time.sleep(float(dt))
                    if success:
                        break
                if not step_executed:
                    break

                policy_frame_count += 1
                if success:
                    break

                point_history.append(
                    _point_cloud_from_obs(
                        obs,
                        expected_points=runtime.observation_point_count,
                        prefer_env_pointcloud=bool(env.object_only_pointcloud),
                    )
                )
                if runtime.action_space_type in HIGH_LOW_ACTION_SPACE_TYPES:
                    agent_history.append(
                        np.asarray(flat_action, dtype=np.float32).reshape(-1).copy()
                    )
                else:
                    if last_info is None:
                        raise RuntimeError(
                            "Internal error: missing env step info after action execution"
                        )
                    agent_history.append(
                        _flat_action_from_info(last_info, env, runtime.action_space)
                    )

                if schedule == "replan_every_step" or chunk_idx + 1 >= int(
                    action_chunk.shape[0]
                ):
                    break
                if viewer_control is not None and (
                    viewer_control.stop_requested or not env.viewer_is_running()
                ):
                    break
            if viewer_control is not None and (
                viewer_control.stop_requested or not env.viewer_is_running()
            ):
                break

    profiler.print_summary()
    return EvalResult(
        source_index=int(source_index),
        source_seed=source_seed,
        completed_steps=int(completed_steps),
        policy_call_count=int(policy_call_count),
        success=bool(success),
        max_cube_z=float(max_cube_z),
        final_cube_pose=final_cube_pose,
        error="",
    )


def _reset_from_source(
    env: TabletopManipulationEnv,
    source: dict[str, np.ndarray],
    trajectory_index: int,
) -> tuple[dict[str, np.ndarray], int | None]:
    source_seed = _trajectory_value(source, "trajectory_seed", trajectory_index)
    seed = _nonnegative_seed_or_none(source_seed)
    if seed is not None:
        env.reset(seed=seed)
        _ensure_source_cube_size(env, source, trajectory_index)
        return env.observe(), seed

    initial_pose = _initial_cube_pose(source, trajectory_index)
    side_length = _trajectory_value(
        source, "initial_cube_side_length", trajectory_index
    )
    if side_length is None:
        side_length = _trajectory_value(source, "cube_side_length", trajectory_index)
    height = _trajectory_value(source, "initial_cube_height", trajectory_index)
    if height is None:
        height = _trajectory_value(source, "cube_height", trajectory_index)

    env.reset(seed=None)
    _set_cube_state(
        env,
        np.asarray(initial_pose, dtype=np.float64).reshape(7),
        side_length=side_length,
        height=height,
    )
    return env.observe(), None


def _object_pose_from_obs(obs: dict[str, Any]) -> np.ndarray:
    key = "object_pose" if "object_pose" in obs else "cube_pose"
    return np.asarray(obs[key], dtype=np.float32).reshape(7).copy()


def _nonnegative_seed_or_none(value: Any | None) -> int | None:
    if value is None:
        return None
    seed = int(value)
    return None if seed < 0 else seed


def _reset_from_retarget_source(
    env: TabletopManipulationEnv,
    source: dict[str, np.ndarray],
    trajectory_index: int,
) -> tuple[dict[str, np.ndarray], int | None]:
    source_seed = _trajectory_value(source, "source_trajectory_seed", trajectory_index)
    if source_seed is None:
        source_seed = _trajectory_value(source, "trajectory_seed", trajectory_index)
    seed = _nonnegative_seed_or_none(source_seed)
    env.reset(seed=seed)
    initial_pose = _initial_cube_pose(source, trajectory_index)
    side_length = _trajectory_value(
        source, "initial_cube_side_length", trajectory_index
    )
    if side_length is None:
        side_length = _trajectory_value(source, "cube_side_length", trajectory_index)
    height = _trajectory_value(source, "initial_cube_height", trajectory_index)
    if height is None:
        height = _trajectory_value(source, "cube_height", trajectory_index)
    _set_cube_state(
        env,
        np.asarray(initial_pose, dtype=np.float64).reshape(7),
        side_length=side_length,
        height=height,
    )
    return env.observe(), seed


def _reset_from_robot_model_source(
    env: TabletopManipulationEnv,
    source: dict[str, np.ndarray],
    trajectory_index: int,
) -> tuple[dict[str, Any], int | None]:
    source_seed = _trajectory_value(source, "source_trajectory_seed", trajectory_index)
    if source_seed is None:
        source_seed = _trajectory_value(source, "trajectory_seed", trajectory_index)
    seed = _nonnegative_seed_or_none(source_seed)
    env.reset(seed=seed)

    object_pose = _trajectory_array(
        source, "object_pose", trajectory_index, required=False
    )
    if object_pose is None:
        object_pose = _trajectory_array(
            source, "cube_pose", trajectory_index, required=False
        )
    if object_pose is None:
        raise ValueError(
            "policy_eval.robot_model_q_source_pt must contain object_pose or cube_pose "
            "for floating-hand reset"
        )
    object_pose_arr = np.asarray(object_pose, dtype=np.float64)
    object_pose0 = object_pose_arr[0] if object_pose_arr.ndim == 2 else object_pose_arr
    side_length = _trajectory_value(
        source, "initial_cube_side_length", trajectory_index
    )
    if side_length is None:
        side_length = _trajectory_value(source, "cube_side_length", trajectory_index)
    height = _trajectory_value(source, "initial_cube_height", trajectory_index)
    if height is None:
        height = _trajectory_value(source, "cube_height", trajectory_index)
    _set_cube_state(
        env,
        np.asarray(object_pose0, dtype=np.float64).reshape(7),
        side_length=side_length,
        height=height,
    )
    return env.observe(), seed


def _initial_wuji_q_from_source(
    source: dict[str, np.ndarray], trajectory_index: int
) -> np.ndarray:
    pos = _first_source_frame(source, "action_ee_pos", trajectory_index, dim=3)
    quat = _first_source_frame(source, "action_ee_quat_wxyz", trajectory_index, dim=4)
    hand_key = "action_hand_q" if "action_hand_q" in source else "action_effector_ctrl"
    hand_q = _first_source_frame(source, hand_key, trajectory_index, dim=WUJI_JOINT_DIM)
    return np.concatenate([pos, _quat_wxyz_to_euler_xyz(quat), hand_q], axis=0).astype(
        np.float32, copy=False
    )


def _initial_robot_model_q_from_source(
    source: dict[str, np.ndarray],
    trajectory_index: int,
    *,
    expected_dim: int,
) -> np.ndarray:
    q_traj = _trajectory_array(source, "manipulator_action", trajectory_index)
    q_arr = np.asarray(q_traj, dtype=np.float32)
    q0 = q_arr[0] if q_arr.ndim == 2 else q_arr
    return _validate_robot_model_q(
        q0,
        expected_dim=int(expected_dim),
        label=f"robot_model_q_source[{int(trajectory_index)}]",
    )


def _first_source_frame(
    source: dict[str, np.ndarray], key: str, trajectory_index: int, *, dim: int
) -> np.ndarray:
    value = _trajectory_value(source, key, trajectory_index)
    if value is None:
        raise KeyError(f"Retargeted reset source is missing {key!r}")
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        out = arr.reshape(-1)
    else:
        out = arr.reshape(arr.shape[0], -1)[0]
    out = out[: int(dim)]
    if out.shape != (int(dim),):
        raise ValueError(
            f"{key} first frame has shape {out.shape}, expected {(int(dim),)}"
        )
    return out.astype(np.float32, copy=False)


def _initial_latent_proprio(runtime: PolicyRuntime) -> np.ndarray:
    return np.zeros(int(runtime.dataset.state_dim), dtype=np.float32)


def _set_env_wuji_q(
    env: TabletopManipulationEnv, q_world: np.ndarray, runtime: HighLowPipelineRuntime
) -> None:
    manipulator = str(runtime.low_level_manipulator).strip().lower()
    if manipulator != "wujihand_right" or str(env.embodiment) != "marvin_wuji":
        raise ValueError(
            "Latent tabletop initial-q materialization currently supports wujihand_right in "
            f"env.embodiment='marvin_wuji', got manipulator={runtime.low_level_manipulator!r} "
            f"env.embodiment={env.embodiment!r}"
        )
    q = np.asarray(q_world, dtype=np.float32).reshape(-1)
    if int(q.shape[0]) < WUJI_LOCAL_DOF or int(runtime.local_dof) != WUJI_LOCAL_DOF:
        raise ValueError(
            f"Expected Wuji q dim/runtime.local_dof={WUJI_LOCAL_DOF}, got q={q.shape} local_dof={runtime.local_dof}"
        )
    side_name = _single_controlled_side(env)
    if side_name != "right":
        raise ValueError(
            f"wujihand_right latent eval expects the controlled tabletop side to be 'right', got {side_name!r}"
        )
    right = env.robot.side("right")
    if env.robot.has_group("left"):
        env.robot.apply_home(env.data, sides=("left",), set_qpos=True)
    hand_q = np.clip(
        q[6 : 6 + WUJI_JOINT_DIM],
        right.hand_ctrl_range[:, 0],
        right.hand_ctrl_range[:, 1],
    )
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
    env.robot.set_side_ctrl(
        env.data, "right", arm_q=command["arm_q_target"], hand_q=hand_q
    )
    env.backend.forward()


def _set_env_robot_model_q(env: TabletopManipulationEnv, q_world: np.ndarray) -> None:
    right = env.robot.side("right")
    q = _validate_robot_model_q(
        q_world,
        expected_dim=int(right.arm_dim + right.effector_ctrl_dim),
        label="robot_model_q",
    )
    q = robot_model_q_to_native(env, q, reset=True)
    arm_q = np.clip(
        q[: right.arm_dim], right.arm_q_range[:, 0], right.arm_q_range[:, 1]
    )
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


def _live_state_to_low_level_q(
    env: TabletopManipulationEnv, runtime: HighLowPipelineRuntime
) -> np.ndarray:
    manipulator = str(runtime.low_level_manipulator).strip().lower()
    if (
        manipulator in {"allegro_right", "barrett", "wujihand_right"}
        and str(env.embodiment) == manipulator
    ):
        q = _current_robot_model_q_from_env(env)
        if int(q.shape[0]) != int(runtime.local_dof):
            raise ValueError(
                f"{manipulator} low-level local_dof={runtime.local_dof} does not match tabletop q dim {q.shape[0]}"
            )
        return q
    if manipulator in {"allegro_right", "barrett"}:
        if str(env.embodiment) != manipulator:
            raise ValueError(
                f"low_level_manipulator={runtime.low_level_manipulator!r} requires matching "
                f"env.embodiment, got {env.embodiment!r}"
            )
    if manipulator != "wujihand_right":
        raise ValueError(
            f"Tabletop latent eval can convert live simulator state only for low_level_manipulator='wujihand_right', "
            f"got {runtime.low_level_manipulator!r}"
        )
    side_name = _single_controlled_side(env)
    side = env.robot.side(side_name)
    if side.effector_action != "joint_position":
        raise ValueError("Wuji latent eval expects a joint_position hand effector")
    pos, quat_wxyz = env.robot.wrist_pose(env.data, side_name)
    euler = _quat_wxyz_to_euler_xyz(np.asarray(quat_wxyz, dtype=np.float32).reshape(4))
    hand_q = env.data.qpos[side.hand_qpos_indices].astype(np.float32).reshape(-1)
    expected = 6 + int(hand_q.shape[0])
    if int(runtime.local_dof) != expected:
        raise ValueError(
            f"wujihand_right low-level local_dof={runtime.local_dof} does not match tabletop q layout "
            f"3 pos + 3 euler + {hand_q.shape[0]} hand joints = {expected}"
        )
    return np.concatenate(
        [
            np.asarray(pos, dtype=np.float32).reshape(3),
            euler.reshape(3),
            hand_q,
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _current_robot_model_q_from_env(env: TabletopManipulationEnv) -> np.ndarray:
    right = env.robot.side("right")
    arm_q = np.asarray(env.data.qpos[right.arm_qpos_indices], dtype=np.float32).reshape(
        -1
    )
    hand_q = np.asarray(
        env.data.qpos[right.hand_qpos_indices], dtype=np.float32
    ).reshape(-1)
    return robot_model_q_to_world(env, np.concatenate([arm_q, hand_q], axis=0).astype(np.float32, copy=False))


def _postprocess_low_level_q_target(
    q: np.ndarray,
    *,
    env: TabletopManipulationEnv,
    runtime: HighLowPipelineRuntime,
    reference: np.ndarray | None,
) -> np.ndarray:
    manipulator = str(runtime.low_level_manipulator).strip().lower()
    q_np = np.asarray(q, dtype=np.float32).reshape(-1).copy()
    if (
        manipulator in {"allegro_right", "barrett", "wujihand_right"}
        and str(env.embodiment) == manipulator
    ):
        q_np = _apply_fixed_zero_constraints(runtime, q_np.reshape(1, -1))[0]
        if reference is not None:
            q_np = unwrap_robot_model_q_rotation(
                q_np, reference=np.asarray(reference, dtype=np.float32).reshape(-1)
            )
        right = env.robot.side("right")
        return _validate_robot_model_q(
            q_np,
            expected_dim=int(right.arm_dim + right.effector_ctrl_dim),
            label="decoded robot_model_q target",
        )
    return q_np.astype(np.float32, copy=False)


def _low_level_q_to_env_action(
    q: np.ndarray,
    env: TabletopManipulationEnv,
    runtime: HighLowPipelineRuntime,
) -> dict[str, dict[str, np.ndarray]]:
    manipulator = str(runtime.low_level_manipulator).strip().lower()
    if (
        manipulator in {"allegro_right", "barrett", "wujihand_right"}
        and str(env.embodiment) == manipulator
    ):
        return _env_action_from_robot_model_q(env, q)
    if manipulator in {"allegro_right", "barrett"}:
        if str(env.embodiment) != manipulator:
            raise ValueError(
                f"Decoded {runtime.low_level_manipulator!r} latent actions require matching "
                f"env.embodiment, got {env.embodiment!r}"
            )
    if manipulator != "wujihand_right" or str(env.embodiment) != "marvin_wuji":
        raise ValueError(
            "Latent tabletop env action conversion currently supports wujihand_right -> "
            f"env.embodiment='marvin_wuji', got manipulator={runtime.low_level_manipulator!r} "
            f"env.embodiment={env.embodiment!r}"
        )
    side_name = _single_controlled_side(env)
    side = env.robot.side(side_name)
    q_np = np.asarray(q, dtype=np.float32).reshape(-1)
    if int(q_np.shape[0]) < int(runtime.local_dof):
        raise ValueError(
            f"Decoded q has dim {q_np.shape[0]}, expected at least {runtime.local_dof}"
        )
    q_local = q_np[: int(runtime.local_dof)]
    quat_wxyz = _euler_xyz_to_quat_wxyz(q_local[3:6])
    hand_q = q_local[6 : 6 + int(side.effector_ctrl_dim)].astype(np.float32, copy=True)
    if hand_q.shape != (int(side.effector_ctrl_dim),):
        raise ValueError(
            f"Decoded hand_q has shape {hand_q.shape}, expected {(int(side.effector_ctrl_dim),)}"
        )
    hand_q = np.clip(
        hand_q, side.hand_ctrl_range[:, 0], side.hand_ctrl_range[:, 1]
    ).astype(np.float32, copy=False)
    return {
        side_name: {
            "ee_pos": q_local[:3].astype(np.float32, copy=False),
            "ee_quat_wxyz": quat_wxyz.astype(np.float32, copy=False),
            "hand_q": hand_q,
        }
    }


def _env_action_from_robot_model_q(
    env: TabletopManipulationEnv, q_world: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
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
            "hand_q": q[right.arm_dim : right.arm_dim + right.effector_ctrl_dim].astype(
                np.float32
            ),
        }
    }


def _append_latent_trace_fields(
    trace: dict[str, list[np.ndarray]],
    *,
    policy_frame: int,
    chunk_idx: int,
    repeat_idx: int,
    latent_action: np.ndarray,
    q_input_world: np.ndarray | None,
    q_target_world: np.ndarray | None,
    proprio_source: str,
    controller_steps_per_prediction: int,
) -> None:
    if q_input_world is None or q_target_world is None:
        raise RuntimeError(
            "Latent trace fields require q_input_world and q_target_world"
        )
    trace.setdefault("high_level_policy_step", []).append(
        np.asarray(int(policy_frame), dtype=np.int32)
    )
    trace.setdefault("high_level_chunk_index", []).append(
        np.asarray(int(chunk_idx), dtype=np.int32)
    )
    trace.setdefault("low_level_action_repeat_index", []).append(
        np.asarray(int(repeat_idx), dtype=np.int32)
    )
    trace.setdefault("high_level_latent_action", []).append(
        np.asarray(latent_action, dtype=np.float32).reshape(-1).copy()
    )
    trace.setdefault("low_level_q_input_world", []).append(
        np.asarray(q_input_world, dtype=np.float32).reshape(-1).copy()
    )
    trace.setdefault("low_level_q_target_world", []).append(
        np.asarray(q_target_world, dtype=np.float32).reshape(-1).copy()
    )
    trace.setdefault("low_level_proprio_source", []).append(
        np.asarray(str(proprio_source))
    )
    trace.setdefault("low_level_controller_steps_per_prediction", []).append(
        np.asarray(int(controller_steps_per_prediction), dtype=np.int32)
    )


def _append_eval_training_frame(
    episode: EvalTrainingEpisode,
    *,
    env: TabletopManipulationEnv,
    runtime: PolicyRuntime,
    obs: dict[str, Any],
    info: dict[str, Any],
    policy_frame: int,
    chunk_idx: int,
    repeat_idx: int,
    flat_action: np.ndarray,
    q_input_world: np.ndarray | None,
    q_target_world: np.ndarray | None,
    q_executed_world: np.ndarray | None,
    proprio_source: str,
    controller_steps_per_prediction: int,
    success: bool,
    max_cube_z: float,
) -> None:
    frames = episode.frames
    obs_xyz = _point_cloud_from_obs(
        obs,
        expected_points=runtime.observation_point_count,
        prefer_env_pointcloud=bool(env.object_only_pointcloud),
    )
    _append_training_frame_value(frames, "obs_point_cloud", obs_xyz, dtype=np.float32)
    _append_training_frame_value(
        frames, "manipulated_point_cloud", obs_xyz, dtype=np.float32
    )

    if q_target_world is None:
        q_target_world = np.asarray(flat_action, dtype=np.float32).reshape(-1)
    if q_executed_world is None:
        q_executed_world = q_target_world
    q_target = np.asarray(q_target_world, dtype=np.float32).reshape(-1)
    q_executed = np.asarray(q_executed_world, dtype=np.float32).reshape(-1)
    _append_training_frame_value(
        frames, "manipulator_action", q_target, dtype=np.float32
    )
    _append_training_frame_value(
        frames, "manipulator_target", q_target, dtype=np.float32
    )
    _append_training_frame_value(
        frames, "manipulator_executed", q_executed, dtype=np.float32
    )
    _append_training_frame_value(
        frames, "manipulator_current", q_executed, dtype=np.float32
    )
    _append_training_frame_value(
        frames, "high_level_action", flat_action, dtype=np.float32
    )
    if q_input_world is not None:
        _append_training_frame_value(
            frames, "low_level_q_input_world", q_input_world, dtype=np.float32
        )
    if runtime.action_space_type in HIGH_LOW_ACTION_SPACE_TYPES:
        if runtime.action_space_type == "latent_contact":
            _append_training_frame_value(
                frames, "high_level_latent_action", flat_action, dtype=np.float32
            )
        else:
            _append_training_frame_value(
                frames, "high_level_contact_action", flat_action, dtype=np.float32
            )
        _append_training_frame_value(
            frames, "low_level_q_target_world", q_target, dtype=np.float32
        )
        _append_training_frame_value(
            frames, "low_level_q_executed_world", q_executed, dtype=np.float32
        )
        _append_training_frame_value(
            frames,
            "low_level_controller_steps_per_prediction",
            int(controller_steps_per_prediction),
            dtype=np.int32,
        )
        _append_training_frame_value(
            frames, "low_level_proprio_source", np.asarray(str(proprio_source))
        )

    for key in (
        "cube_pose",
        "cube_pos",
        "cube_quat_wxyz",
        "cube_side_length",
        "cube_height",
        "cube_yaw",
        "object_pose",
        "object_pos",
        "object_quat_wxyz",
        "object_yaw",
    ):
        if key in obs:
            _append_training_frame_value(frames, key, obs[key])
    if "object_pose" not in obs and "cube_pose" in obs:
        _append_training_frame_value(
            frames, "object_pose", obs["cube_pose"], dtype=np.float32
        )
    if "object_pos" not in obs and "cube_pos" in obs:
        _append_training_frame_value(
            frames, "object_pos", obs["cube_pos"], dtype=np.float32
        )
    if "object_quat_wxyz" not in obs and "cube_quat_wxyz" in obs:
        _append_training_frame_value(
            frames, "object_quat_wxyz", obs["cube_quat_wxyz"], dtype=np.float32
        )

    command = dict(info.get("command", {}))
    for key in (
        "pre_qpos",
        "pre_ctrl",
        "target_qpos",
        "target_ctrl",
        "executed_qpos",
        "executed_qvel",
        "executed_ctrl",
        "object_pose_before",
        "object_pose_after",
        "cube_pose_before",
        "cube_pose_after",
    ):
        if key in command:
            _append_training_frame_value(frames, key, command[key])
    _append_training_frame_value(
        frames, "high_level_policy_step", int(policy_frame), dtype=np.int32
    )
    _append_training_frame_value(
        frames, "high_level_chunk_index", int(chunk_idx), dtype=np.int32
    )
    _append_training_frame_value(
        frames, "low_level_action_repeat_index", int(repeat_idx), dtype=np.int32
    )
    _append_training_frame_value(
        frames, "source_trajectory_index", int(episode.source_index), dtype=np.int32
    )
    if episode.reset_source_index is not None:
        _append_training_frame_value(
            frames,
            "reset_source_index",
            int(episode.reset_source_index),
            dtype=np.int32,
        )
    _append_training_frame_value(frames, "success", bool(success), dtype=np.bool_)
    _append_training_frame_value(
        frames, "max_cube_z", float(max_cube_z), dtype=np.float32
    )


def _append_training_frame_value(
    frames: dict[str, list[np.ndarray]],
    key: str,
    value: Any,
    *,
    dtype: Any | None = None,
) -> None:
    arr = (
        np.asarray(value, dtype=dtype).copy()
        if dtype is not None
        else np.asarray(value).copy()
    )
    frames.setdefault(str(key), []).append(arr)


def _training_episode_frame_count(episode: EvalTrainingEpisode) -> int:
    if "obs_point_cloud" not in episode.frames:
        return 0
    return int(len(episode.frames["obs_point_cloud"]))


def _save_eval_training_payload(
    output_path: Path,
    *,
    training_records: list[tuple[EvalTrainingEpisode, EvalResult]],
    cfg: DictConfig,
    env: TabletopManipulationEnv,
    runtime: PolicyRuntime,
    backend: str,
    dt: float,
    initial_pose_mode: str,
    initial_pose_path: Path | None,
    initial_low_level_q: np.ndarray | None,
    proprio_source: str,
    controller_steps_per_prediction: int,
) -> Path:
    records = [
        (episode, result)
        for episode, result in training_records
        if _training_episode_frame_count(episode) > 0
    ]
    if not records:
        raise RuntimeError(
            "No policy-eval frames were recorded for policy_eval.save_training_pt"
        )
    lengths = [_training_episode_frame_count(episode) for episode, _ in records]
    episode_ends = _make_episode_ends_np(lengths)
    obs_point_cloud = _concat_training_key(
        records, "obs_point_cloud", required=True
    ).astype(np.float32, copy=False)
    action = _concat_training_key(records, "manipulator_action", required=True).astype(
        np.float32, copy=False
    )
    proprio = _concat_training_key(
        records, "manipulator_executed", required=True
    ).astype(np.float32, copy=False)
    action_space = _eval_training_action_space(
        runtime, env, action_dim=int(action.reshape(action.shape[0], -1).shape[1])
    )

    data: dict[str, Any] = {
        "obs_point_cloud": torch.as_tensor(obs_point_cloud, dtype=torch.float32),
        "action": torch.as_tensor(action, dtype=torch.float32),
        "proprio": torch.as_tensor(proprio, dtype=torch.float32),
    }
    reserved = {"obs_point_cloud", "action", "proprio"}
    for key in sorted({key for episode, _ in records for key in episode.frames.keys()}):
        if key in reserved:
            continue
        values = [_concat_training_key(records, key, required=False)]
        if values[0] is None:
            continue
        value = values[0]
        value_arr = np.asarray(value)
        if value_arr.dtype.kind in {"U", "S", "O"}:
            data[str(key)] = value_arr.copy()
        else:
            data[str(key)] = torch.as_tensor(value_arr).detach().cpu()

    source_indices = [int(episode.source_index) for episode, _ in records]
    reset_source_indices = [
        -1 if episode.reset_source_index is None else int(episode.reset_source_index)
        for episode, _ in records
    ]
    object_kind = str(getattr(env.scene, "object_kind", "cube") or "cube")
    task_name = str(OmegaConf.select(cfg, "task.name", default="cube_pick"))
    meta: dict[str, Any] = {
        "schema": "high_level_eval_rollout_v1",
        "protocol": "high_level_eval_lightweight_pt",
        "source": {
            "kind": "closed_loop_high_level_policy_eval",
            "script": "experiment/evaluate_policy.py",
            "high_level_payload_path": str(runtime.payload_path),
            "reset_source_pt_path": str(runtime.reset_source_path),
            "reset_source_kind": str(runtime.reset_source_kind),
            "output_pt": str(output_path),
            "source_indices": source_indices,
            "reset_source_indices": reset_source_indices,
        },
        "task": task_name,
        "embodiment": str(cfg.env.embodiment),
        "mode": str(cfg.env.mode),
        "object_kind": object_kind,
        "backend": str(backend),
        "frame_skip": int(env.backend.frame_skip),
        "dt": float(dt),
        "fps": 1.0 / float(dt) if float(dt) > 0.0 else None,
        "time": {"dt": float(dt), "fps": 1.0 / float(dt) if float(dt) > 0.0 else None},
        "observation": {
            "point_count": int(obs_point_cloud.shape[1]),
            "frame": "world",
            "dtype": str(obs_point_cloud.dtype),
        },
        "action_space": action_space,
        "dataset_name": Path(output_path).stem,
        "num_episodes": int(len(records)),
        "num_frames": int(sum(lengths)),
        "high_level_config_path": str(runtime.high_level_config_path),
        "high_level_checkpoint_path": str(runtime.checkpoint_path),
        "high_level_checkpoint_state_key": str(runtime.state_key),
        "high_level_action_space_type": str(runtime.action_space_type),
        "proprio_source": str(proprio_source),
        "controller_steps_per_prediction": int(controller_steps_per_prediction),
        "initial_pose_mode": str(initial_pose_mode),
        "initial_pose_path": (
            "" if initial_pose_path is None else str(initial_pose_path)
        ),
        "robot_xml_path": str(env.scene.robot_xml_path),
        "group_names": list(env.robot.group_names),
        "qpos_names": qpos_names(env.model),
        "ctrl_names": ctrl_names(env.model),
        "target_pointcloud_num": int(env.target_pointcloud_num),
        "camera_names": list(env.camera_names),
        "policy_camera_names": list(env.policy_camera_names),
    }
    meta = with_object_only_world_observation(meta)
    if runtime.high_low_runtime is not None:
        meta.update(
            {
                "low_level_config_path": (
                    ""
                    if runtime.low_level_config_path is None
                    else str(runtime.low_level_config_path)
                ),
                "low_level_checkpoint_path": str(
                    runtime.high_low_runtime.low_level_checkpoint_path
                ),
                "low_level_manipulator": str(
                    runtime.high_low_runtime.low_level_manipulator
                ),
                "low_level_backend": str(runtime.high_low_runtime.encoder_backend),
                "low_level_local_dof": int(runtime.high_low_runtime.local_dof),
            }
        )
    if initial_low_level_q is not None:
        meta["initial_low_level_q_world"] = (
            np.asarray(initial_low_level_q, dtype=np.float32).reshape(-1).tolist()
        )

    episode_meta: list[dict[str, Any]] = []
    for episode_idx, (episode, result) in enumerate(records):
        episode_meta.append(
            {
                "episode_index": int(episode_idx),
                "source_trajectory_index": int(episode.source_index),
                "reset_source_index": (
                    -1
                    if episode.reset_source_index is None
                    else int(episode.reset_source_index)
                ),
                "source_trajectory_seed": (
                    -1 if episode.source_seed is None else int(episode.source_seed)
                ),
                "num_frames": int(lengths[episode_idx]),
                "completed_steps": int(result.completed_steps),
                "policy_call_count": int(result.policy_call_count),
                "success": bool(result.success),
                "max_cube_z": float(result.max_cube_z),
                "initial_pose_mode": str(initial_pose_mode),
                "initial_pose_path": (
                    "" if initial_pose_path is None else str(initial_pose_path)
                ),
                "proprio_source": str(proprio_source),
                "controller_steps_per_prediction": int(controller_steps_per_prediction),
            }
        )

    payload = {
        "meta": meta,
        "data": data,
        "episode_ends": torch.as_tensor(episode_ends, dtype=torch.long),
        "episode_meta": episode_meta,
    }
    return save_high_level_payload(payload, output_path, validate=False)


def _make_episode_ends_np(lengths: Iterable[int]) -> np.ndarray:
    total = 0
    ends: list[int] = []
    for length in lengths:
        value = int(length)
        if value <= 0:
            raise ValueError(f"Episode frame counts must be positive, got {value}")
        total += value
        ends.append(total)
    return np.asarray(ends, dtype=np.int64)


def _concat_training_key(
    records: list[tuple[EvalTrainingEpisode, EvalResult]],
    key: str,
    *,
    required: bool,
) -> np.ndarray | None:
    per_episode: list[np.ndarray] = []
    for episode, _ in records:
        frames = episode.frames
        if key not in frames:
            if required:
                raise KeyError(
                    f"Recorded policy-eval episode {episode.source_index} is missing {key!r}"
                )
            return None
        values = frames[key]
        if not values:
            if required:
                raise RuntimeError(
                    f"Recorded policy-eval episode {episode.source_index} has no {key!r} frames"
                )
            return None
        try:
            per_episode.append(np.stack(values, axis=0))
        except ValueError:
            if required:
                raise
            return None
    return np.concatenate(per_episode, axis=0)


def _eval_training_action_space(
    runtime: PolicyRuntime,
    env: TabletopManipulationEnv,
    *,
    action_dim: int,
) -> dict[str, Any]:
    if runtime.high_low_runtime is not None:
        manipulator = str(runtime.high_low_runtime.low_level_manipulator).strip()
        if manipulator.lower() in {"allegro_right", "barrett"}:
            return _robot_model_q_action_space(runtime.high_low_runtime, env)
    return {
        "type": "eval_manipulator_state",
        "dim": int(action_dim),
        "component_slices": {"state": [0, int(action_dim)]},
        "representation": "eval target/current manipulator state",
    }


def _robot_model_q_action_space(
    runtime: HighLowPipelineRuntime, env: TabletopManipulationEnv
) -> dict[str, Any]:
    right = env.robot.side("right")
    dim = int(runtime.local_dof)
    hand_start = int(right.arm_dim)
    joint_orders = tuple(
        str(x) for x in tuple(right.arm_joint_names + right.hand_joint_names)
    )
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
        raise FileNotFoundError(
            "No worker training payloads were produced for policy_eval.save_training_pt"
        )
    sorted_paths = sorted(paths, key=_worker_output_sort_key)
    payloads = [load_high_level_payload(path, validate=True) for path in sorted_paths]
    if len(payloads) == 1:
        payload = payloads[0]
        payload["meta"]["source"]["worker_payloads"] = [
            str(path) for path in sorted_paths
        ]
        payload["meta"]["source"]["output_pt"] = str(output_path)
        payload["meta"]["dataset_name"] = Path(output_path).stem
        return save_high_level_payload(payload, output_path, validate=False)

    frame_counts = [
        int(torch.as_tensor(payload["episode_ends"]).reshape(-1)[-1].item())
        for payload in payloads
    ]
    episode_counts = [
        int(torch.as_tensor(payload["episode_ends"]).numel()) for payload in payloads
    ]
    data: dict[str, Any] = {}
    for key in sorted({key for payload in payloads for key in payload["data"].keys()}):
        values = [
            payload["data"][key] for payload in payloads if key in payload["data"]
        ]
        if len(values) != len(payloads):
            continue
        arrays = [np.asarray(value) for value in values]
        if any(array.dtype.kind in {"U", "S", "O"} for array in arrays):
            if all(
                array.ndim > 0 and int(array.shape[0]) == frame_counts[idx]
                for idx, array in enumerate(arrays)
            ):
                data[key] = np.concatenate(arrays, axis=0)
            elif all(
                array.ndim > 0 and int(array.shape[0]) == episode_counts[idx]
                for idx, array in enumerate(arrays)
            ):
                data[key] = np.concatenate(arrays, axis=0)
            elif all(np.array_equal(arrays[0], array) for array in arrays[1:]):
                data[key] = arrays[0]
            continue
        tensors = [torch.as_tensor(array).detach().cpu() for array in arrays]
        if all(
            tensor.ndim > 0 and int(tensor.shape[0]) == frame_counts[idx]
            for idx, tensor in enumerate(tensors)
        ):
            data[key] = torch.cat(tensors, dim=0)
        elif all(
            tensor.ndim > 0 and int(tensor.shape[0]) == episode_counts[idx]
            for idx, tensor in enumerate(tensors)
        ):
            data[key] = torch.cat(tensors, dim=0)
        elif all(torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
            data[key] = tensors[0]

    lengths: list[int] = []
    for payload in payloads:
        ends = torch.as_tensor(payload["episode_ends"], dtype=torch.long).reshape(-1)
        starts = torch.cat([torch.zeros((1,), dtype=torch.long), ends[:-1]])
        lengths.extend((ends - starts).detach().cpu().numpy().astype(np.int64).tolist())
    episode_ends = torch.as_tensor(_make_episode_ends_np(lengths), dtype=torch.long)
    episode_meta = [item for payload in payloads for item in payload["episode_meta"]]
    meta = dict(payloads[0]["meta"])
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
    return save_high_level_payload(payload, output_path, validate=False)


def _worker_output_sort_key(path: Path) -> tuple[str, int]:
    stem = Path(path).stem
    try:
        return stem.rsplit("_", 1)[0], int(stem.rsplit("_", 1)[1])
    except Exception:
        return stem, 0


def _flat_action_to_env_action(
    flat_action: np.ndarray,
    env: TabletopManipulationEnv,
    action_space: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    side_name = _single_controlled_side(env)
    side = env.robot.side(side_name)
    base_pos = _action_component(flat_action, action_space, "base_pos", dim=3)
    base_quat = _action_component(flat_action, action_space, "base_quat_wxyz", dim=4)
    joint_target = _action_component(
        flat_action, action_space, "joint_target", dim=side.effector_ctrl_dim
    )
    quat_norm = float(np.linalg.norm(base_quat))
    if quat_norm <= 1e-8:
        raise ValueError("Predicted base_quat_wxyz is near zero")
    entry: dict[str, np.ndarray] = {
        "ee_pos": base_pos.astype(np.float32, copy=False),
        "ee_quat_wxyz": (base_quat / quat_norm).astype(np.float32, copy=False),
    }
    if side.effector_action == "gripper":
        entry["gripper"] = np.clip(
            joint_target,
            side.hand_ctrl_range[:, 0],
            side.hand_ctrl_range[:, 1],
        ).astype(np.float32, copy=False)
    else:
        if joint_target.shape == (side.effector_ctrl_dim,):
            joint_target = np.clip(
                joint_target, side.hand_ctrl_range[:, 0], side.hand_ctrl_range[:, 1]
            )
        entry["hand_q"] = joint_target.astype(np.float32, copy=False)
    return {side_name: entry}


def _flat_action_from_live_state(
    env: TabletopManipulationEnv, action_space: dict[str, Any]
) -> np.ndarray:
    side_name = _single_controlled_side(env)
    side = env.robot.side(side_name)
    pos, quat = env.robot.wrist_pose(env.data, side_name)
    if side.effector_action == "gripper":
        joint_target = env.data.ctrl[side.hand_actuator_ids].astype(np.float32).copy()
    else:
        joint_target = env.data.qpos[side.hand_qpos_indices].astype(np.float32).copy()
    return _pack_flat_action(pos, quat, joint_target, action_space)


def _flat_action_from_info(
    info: dict[str, Any], env: TabletopManipulationEnv, action_space: dict[str, Any]
) -> np.ndarray:
    side_name = _single_controlled_side(env)
    side = env.robot.side(side_name)
    side_cmd = info["command"]["sides"][side_name]
    if side.effector_action == "gripper":
        joint_target = side_cmd["gripper_target"]
    else:
        joint_target = side_cmd.get(
            "hand_q_target", side_cmd.get("effector_ctrl_target")
        )
    return _pack_flat_action(
        side_cmd["ee_pos_target"],
        side_cmd["ee_quat_wxyz_target"],
        joint_target,
        action_space,
    )


def _pack_flat_action(
    base_pos: Any,
    base_quat_wxyz: Any,
    joint_target: Any,
    action_space: dict[str, Any],
) -> np.ndarray:
    dim = int(action_space["dim"])
    out = np.zeros((dim,), dtype=np.float32)
    for name, value in (
        ("base_pos", np.asarray(base_pos, dtype=np.float32).reshape(-1)),
        ("base_quat_wxyz", np.asarray(base_quat_wxyz, dtype=np.float32).reshape(-1)),
        ("joint_target", np.asarray(joint_target, dtype=np.float32).reshape(-1)),
    ):
        start, end = _component_slice(action_space, name)
        if int(end - start) != int(value.shape[0]):
            raise ValueError(
                f"Cannot pack {name}: action_space slice has dim {end - start}, value has dim {value.shape[0]}"
            )
        out[start:end] = value
    return out


def _action_component(
    flat_action: np.ndarray, action_space: dict[str, Any], name: str, *, dim: int | None
) -> np.ndarray:
    start, end = _component_slice(action_space, name)
    value = np.asarray(flat_action, dtype=np.float32).reshape(-1)[start:end].copy()
    if dim is not None and value.shape != (int(dim),):
        raise ValueError(
            f"Action component {name!r} has shape {value.shape}, expected {(int(dim),)}"
        )
    return value


def _component_slice(action_space: dict[str, Any], name: str) -> tuple[int, int]:
    slices = action_space.get("component_slices")
    if not isinstance(slices, dict) or name not in slices:
        raise KeyError(f"Action space is missing component slice {name!r}")
    start, end = [int(x) for x in slices[name]]
    if start < 0 or end <= start or end > int(action_space["dim"]):
        raise ValueError(
            f"Invalid action_space component slice {name!r}: {slices[name]!r}"
        )
    return start, end


def _point_cloud_from_obs(
    obs: dict[str, Any],
    *,
    expected_points: int,
    prefer_env_pointcloud: bool = False,
) -> np.ndarray:
    if not bool(prefer_env_pointcloud):
        object_cloud = _object_point_cloud_from_obs(
            obs, expected_points=int(expected_points)
        )
        if object_cloud is not None:
            return _validate_policy_point_cloud_health(
                object_cloud, expected_points=int(expected_points)
            )
    if "pointcloud" not in obs:
        raise KeyError(
            "Environment observation is missing live policy pointcloud; set perception.include_camera_obs=true"
        )
    xyzw = np.asarray(obs["pointcloud"]["xyzw"], dtype=np.float32)
    if xyzw.ndim != 2 or xyzw.shape[1] < 3:
        raise ValueError(
            f"Expected obs['pointcloud']['xyzw'] with shape (N, >=3), got {xyzw.shape}"
        )
    if int(xyzw.shape[0]) != int(expected_points):
        raise ValueError(
            f"Live pointcloud has {xyzw.shape[0]} points, but the high-level policy expects {expected_points}"
        )
    return _validate_policy_point_cloud_health(
        xyzw[:, :3], expected_points=int(expected_points)
    )


def _validate_policy_point_cloud_health(
    points: np.ndarray, *, expected_points: int
) -> np.ndarray:
    """Reject invalid live observations before they reach the high-level policy."""

    xyz = np.asarray(points, dtype=np.float32)
    expected_shape = (int(expected_points), 3)
    if xyz.shape != expected_shape:
        raise ValueError(
            f"Live policy point cloud has shape {xyz.shape}, expected {expected_shape}"
        )
    if not np.isfinite(xyz).all():
        invalid_count = int(np.size(xyz) - np.count_nonzero(np.isfinite(xyz)))
        raise RuntimeError(
            "Unhealthy live policy point cloud: "
            f"found {invalid_count} non-finite XYZ values. Check policy-camera depth and calibration."
        )

    max_extent_m = float(np.max(np.ptp(xyz, axis=0))) if xyz.shape[0] else 0.0
    if not np.any(xyz) or max_extent_m < 1.0e-4:
        raise RuntimeError(
            "Unhealthy live policy point cloud: the observation is zero-filled or geometrically degenerate "
            f"(maximum XYZ extent={max_extent_m:.3e} m). Check object segmentation and keep collision "
            "visualization disabled during policy evaluation."
        )
    return xyz.astype(np.float32, copy=True)


def _object_point_cloud_from_obs(
    obs: dict[str, Any], *, expected_points: int
) -> np.ndarray | None:
    sensor_data = obs.get("sensor_data")
    sensor_param = obs.get("sensor_param")
    if not isinstance(sensor_data, dict) or not isinstance(sensor_param, dict):
        return None

    camera_names = obs.get("policy_camera_names", tuple(sensor_data.keys()))
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
            if mask_key not in cam_data:
                continue
            candidate = np.asarray(cam_data[mask_key], dtype=np.bool_)
            mask = candidate if mask is None else np.logical_or(mask, candidate)
        if mask is None or "depth" not in cam_data:
            continue
        if "intrinsic_cv" not in cam_param or "extrinsic_cv" not in cam_param:
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
    return _sample_or_pad_policy_point_cloud(
        xyz_all, expected_points=int(expected_points)
    )


def _sample_or_pad_policy_point_cloud(
    points: np.ndarray, *, expected_points: int
) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    target = int(expected_points)
    if target < 0:
        raise ValueError(f"expected_points must be nonnegative, got {expected_points}")
    if target == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if xyz.shape[0] == 0:
        return np.zeros((target, 3), dtype=np.float32)
    if int(xyz.shape[0]) > target:
        return xyz[farthest_point_sample_indices(xyz, target)].astype(
            np.float32, copy=True
        )
    if int(xyz.shape[0]) < target:
        repeat = target - int(xyz.shape[0])
        repeat_indices = np.resize(np.arange(int(xyz.shape[0]), dtype=np.int64), repeat)
        xyz = np.concatenate([xyz, xyz[repeat_indices]], axis=0)
    return xyz.astype(np.float32, copy=True)


def _history_to_policy_input(
    point_history: Iterable[np.ndarray],
    agent_history: Iterable[np.ndarray],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    point_cloud = (
        torch.from_numpy(np.stack(list(point_history), axis=0))
        .unsqueeze(0)
        .to(device, dtype=torch.float32)
    )
    agent_pos = (
        torch.from_numpy(np.stack(list(agent_history), axis=0))
        .unsqueeze(0)
        .to(device, dtype=torch.float32)
    )
    return {"point_cloud": point_cloud, "agent_pos": agent_pos}


def _single_controlled_side(env: TabletopManipulationEnv) -> str:
    if env.mode != "single_right":
        raise ValueError(
            "Closed-loop high-level eval currently supports env.mode=single_right only"
        )
    if not env.robot.has_group("right"):
        raise ValueError(f"Embodiment {env.embodiment!r} has no right control group")
    return "right"


def _selected_eval_indices(
    *,
    split_indices: list[int],
    source_count: int,
    split: str,
    indices: object | None,
    start_index: int,
    count: object | None,
) -> list[int]:
    split_selected = [int(idx) for idx in split_indices]
    if indices is not None and str(indices).strip().lower() not in {"", "null", "none"}:
        selected = _parse_indices(indices, source_count)
        invalid = [idx for idx in selected if idx < 0 or idx >= int(source_count)]
        if invalid:
            raise IndexError(
                f"Episode indices out of range for {source_count} episodes: {invalid}"
            )
        if str(split).strip().lower() != "all":
            allowed = set(split_selected)
            outside = [idx for idx in selected if idx not in allowed]
            if outside:
                raise ValueError(
                    f"policy_eval.indices contains episodes outside split={split!r}: {outside}"
                )
        return list(dict.fromkeys(int(idx) for idx in selected))

    if int(start_index) < 0:
        raise ValueError("policy_eval.start_index must be non-negative")
    if int(start_index) > len(split_selected):
        raise IndexError(
            f"policy_eval.start_index={start_index} is out of range for split={split!r} with {len(split_selected)} episodes"
        )
    if count is None or str(count).strip().lower() in {"", "null", "none"}:
        stop = len(split_selected)
    else:
        count_int = int(count)
        if count_int < 0:
            raise ValueError("policy_eval.count must be non-negative when provided")
        stop = min(len(split_selected), int(start_index) + count_int)
    return split_selected[int(start_index) : stop]


def _load_hydra_config(config_path: Path) -> DictConfig:
    resolved = Path(config_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Hydra config not found: {resolved}")
    config_dir = resolved.parent
    config_name = resolved.stem
    for parent in resolved.parents:
        if parent.name == "conf":
            config_dir = parent
            config_name = resolved.relative_to(parent).with_suffix("").as_posix()
            break
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.2", config_dir=str(config_dir)):
        cfg = runtime_config(compose(config_name=config_name))
    GlobalHydra.instance().clear()
    OmegaConf.resolve(cfg)
    return cfg


def _resolve_project_path(value: Any, *, label: str) -> Path:
    if value is None:
        raise ValueError(f"{label} must be set")
    text = str(value).strip()
    if text == "" or text.lower() == "null":
        raise ValueError(f"{label} must be set")
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def _resolve_experiment_path(value: Any, *, label: str) -> Path:
    if value is None:
        raise ValueError(f"{label} must be set")
    text = str(value).strip()
    if text == "" or text.lower() == "null":
        raise ValueError(f"{label} must be set")
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == "experiment":
        return (ROOT / path).resolve()
    return (EXPERIMENT_DIR / path).resolve()


def _resolve_checkpoint_path(cfg: DictConfig, high_cfg: DictConfig) -> Path:
    explicit = OmegaConf.select(cfg, "policy_eval.checkpoint_path", default=None)
    if explicit is not None and str(explicit).strip().lower() not in {
        "",
        "null",
        "none",
    }:
        path = _resolve_experiment_path(explicit, label="policy_eval.checkpoint_path")
    else:
        save_dir = _resolve_experiment_path(
            OmegaConf.select(high_cfg, "training.save_dir", default=None),
            label="high-level training.save_dir",
        )
        checkpoint_name = str(
            OmegaConf.select(cfg, "policy_eval.checkpoint_name", default="last.pt")
        )
        path = (save_dir / checkpoint_name).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"High-level checkpoint not found: {path}. "
            "Generate it with: python experiment/train_policy.py "
            "(see the Objects Placing pipeline in README.md)."
        )
    return path


def _cfg_action_space(cfg: DictConfig) -> str:
    action_space = OmegaConf.select(cfg, "policy.action_space", default=None)
    if action_space is None or str(action_space).strip() == "":
        raise KeyError("Expected high-level config key policy.action_space.")
    return str(action_space).strip().lower()


def _validate_runtime_matches_env(cfg: DictConfig, runtime: PolicyRuntime) -> None:
    action_space_name = _cfg_action_space(runtime.high_level_cfg)
    env_embodiment = str(cfg.env.embodiment).strip()
    payload_embodiment = str(runtime.payload["meta"].get("embodiment", "")).strip()
    if runtime.action_space_type in HIGH_LOW_ACTION_SPACE_TYPES:
        expected_action_space = (
            "latent"
            if runtime.action_space_type == "latent_contact"
            else "contact_points"
        )
        if action_space_name != expected_action_space:
            raise ValueError(
                f"{runtime.action_space_type} payload eval requires high-level "
                f"policy.action_space={expected_action_space!r}, got {action_space_name!r}"
            )
        if runtime.high_low_runtime is not None:
            low_manipulator = (
                str(runtime.high_low_runtime.low_level_manipulator).strip().lower()
            )
            if low_manipulator == "wujihand_right" and env_embodiment not in {
                "marvin_wuji",
                "wujihand_right",
            }:
                raise ValueError(
                    "Decoded wujihand_right high-low actions require env.embodiment "
                    "'marvin_wuji' or 'wujihand_right', "
                    f"got {env_embodiment!r}"
                )
            if (
                low_manipulator in {"allegro_right", "barrett"}
                and env_embodiment != low_manipulator
            ):
                raise ValueError(
                    f"Decoded {low_manipulator!r} high-low actions require matching env.embodiment, "
                    f"got {env_embodiment!r}"
                )
            if low_manipulator not in {"wujihand_right", "allegro_right", "barrett"}:
                raise ValueError(
                    f"Unsupported high-low tabletop low_level_manipulator={low_manipulator!r}; "
                    "supported values are wujihand_right, allegro_right, and barrett."
                )
        if runtime.reset_source_kind == "robot_model_q" and env_embodiment not in {
            "allegro_right",
            "barrett",
            "wujihand_right",
        }:
            raise ValueError(
                f"robot_model_q reset source requires a floating hand env, got env.embodiment={env_embodiment!r}"
            )
        if (
            runtime.reset_source_kind == "latent_source_object_pose"
            and env_embodiment
            not in {
                "allegro_right",
                "barrett",
                "marvin_wuji",
                "wujihand_right",
            }
        ):
            raise ValueError(
                "latent_source_object_pose reset source requires an Allegro, Barrett, or Wuji env, "
                f"got env.embodiment={env_embodiment!r}"
            )
        if (
            runtime.reset_source_kind == "retargeted_wuji"
            and env_embodiment != "marvin_wuji"
        ):
            raise ValueError(
                f"retargeted_wuji reset source requires env.embodiment='marvin_wuji', got {env_embodiment!r}"
            )
        if len(runtime.source_episode_indices) != len(runtime.payload["episode_meta"]):
            raise ValueError(
                "High-low source episode mapping length does not match payload episode_meta length"
            )
        return

    if action_space_name != env_embodiment:
        raise ValueError(
            f"High-level policy action_space={action_space_name!r} does not match env.embodiment={env_embodiment!r}"
        )
    if payload_embodiment != env_embodiment:
        raise ValueError(
            f"Payload embodiment={payload_embodiment!r} does not match env.embodiment={env_embodiment!r}"
        )
    if int(_trajectory_count(runtime.source)) != int(
        len(runtime.payload["episode_meta"])
    ):
        raise ValueError(
            "Replay source episode count does not match high-level payload episode_meta length"
        )


def _validate_reset_source_compatible(
    source: dict[str, np.ndarray],
    cfg: DictConfig,
    *,
    action_space_type: str,
) -> None:
    object_suite = (
        str(OmegaConf.select(cfg, "task.name", default="cube_pick"))
        == "object_suite_pick_place"
    )
    if not object_suite:
        _sync_cube_height_ratio_from_source(source, cfg)
    if action_space_type not in HIGH_LOW_ACTION_SPACE_TYPES:
        _validate_source_compatible(source, cfg)
        return

    source_action_space_type = str(
        _source_scalar(source, "action_space_type") or ""
    ).strip()
    frame_skip = _source_scalar(source, "frame_skip")
    if (
        source_action_space_type != "robot_model_q"
        and frame_skip is not None
        and int(frame_skip) != int(cfg.env.frame_skip)
    ):
        raise ValueError(
            f"Latent reset source frame_skip={int(frame_skip)} does not match config env.frame_skip={int(cfg.env.frame_skip)}"
        )

    if object_suite:
        if source_action_space_type == "robot_model_q":
            _validate_robot_model_q_source_basic(
                source,
                embodiment=str(cfg.env.embodiment),
                source_path=Path(
                    str(
                        source.get(
                            "source_training_pt_path",
                            "policy_eval.robot_model_q_source_pt",
                        )
                    )
                ),
            )
        return

    expected_arrays = {
        "cube_x_range": np.asarray(cfg.cube.x_range, dtype=np.float32),
        "cube_y_range": np.asarray(cfg.cube.y_range, dtype=np.float32),
        "cube_side_length_range": np.asarray(
            cfg.cube.side_length_range, dtype=np.float32
        ),
        "cube_height_range": np.asarray(cfg.cube.side_length_range, dtype=np.float32)
        * np.float32(OmegaConf.select(cfg, "cube.height_ratio", default=1.0)),
        "cube_yaw_range_deg": np.asarray(cfg.cube.yaw_range_deg, dtype=np.float32),
    }
    for key, expected in expected_arrays.items():
        if key not in source:
            continue
        actual = np.asarray(source[key], dtype=np.float32)
        if actual.shape == expected.shape and not np.allclose(
            actual, expected, atol=1e-6
        ):
            raise ValueError(
                f"Latent reset source {key}={actual.tolist()} does not match config {key}={expected.tolist()}"
            )

    if source_action_space_type == "robot_model_q":
        _validate_robot_model_q_source_basic(
            source,
            embodiment=str(cfg.env.embodiment),
            source_path=Path(
                str(
                    source.get(
                        "source_training_pt_path", "policy_eval.robot_model_q_source_pt"
                    )
                )
            ),
        )


def _validate_robot_model_q_source_basic(
    source: dict[str, np.ndarray],
    *,
    embodiment: str,
    source_path: Path,
) -> None:
    action_space_type = _source_scalar(source, "action_space_type")
    if str(action_space_type).strip() != "robot_model_q":
        raise ValueError(
            f"{source_path} must have action_space_type='robot_model_q', got {action_space_type!r}"
        )
    source_embodiment = _source_scalar(source, "embodiment")
    if source_embodiment is not None and str(source_embodiment).strip() != str(
        embodiment
    ):
        raise ValueError(
            f"{source_path} embodiment={source_embodiment!r} does not match env.embodiment={embodiment!r}"
        )
    source_mode = _source_scalar(source, "mode")
    if source_mode is not None and str(source_mode).strip() not in {
        "floating_right",
        "single_right",
    }:
        raise ValueError(
            f"{source_path} mode={source_mode!r} is not a supported floating right-hand source"
        )
    if "manipulator_action" not in source:
        raise KeyError(
            f"{source_path} must contain manipulator_action for robot_model_q reset/control"
        )
    if "object_pose" not in source and "cube_pose" not in source:
        raise KeyError(
            f"{source_path} must contain object_pose or cube_pose for robot_model_q reset"
        )


def _validate_robot_model_source_env_dim(
    source: dict[str, np.ndarray],
    reset_source_indices: Iterable[int],
    env: TabletopManipulationEnv,
) -> None:
    if not env.robot.has_group("right"):
        raise ValueError(
            "floating robot_model_q high-level eval requires a right control group"
        )
    right = env.robot.side("right")
    expected_dim = int(right.arm_dim + right.effector_ctrl_dim)
    for source_index in reset_source_indices:
        q0 = _initial_robot_model_q_from_source(
            source, int(source_index), expected_dim=expected_dim
        )
        if q0.shape != (expected_dim,):
            raise ValueError(
                f"robot_model_q source trajectory {int(source_index)} has dim {q0.shape[0]}, "
                f"expected env dim {expected_dim}"
            )


def _enable_policy_camera_observations(cfg: DictConfig, *, point_count: int) -> None:
    cfg.perception.include_camera_obs = True
    cfg.perception.capture_segmentation = True
    if bool(
        OmegaConf.select(cfg, "policy_eval.match_payload_point_count", default=True)
    ):
        cfg.perception.target_pointcloud_num = int(point_count)


def _should_report_progress(local_idx: int, total: int, progress_interval: int) -> bool:
    done = int(local_idx) + 1
    return done == int(total) or (
        int(progress_interval) > 0 and done % int(progress_interval) == 0
    )


def _print_progress(results: list[EvalResult], total: int) -> None:
    done = len(results)
    successes = sum(1 for result in results if result.success)
    rate = successes / done if done else 0.0
    print(
        f"progress {done}/{total}: successes={successes} success_rate={rate:.3%}",
        flush=True,
    )


def _print_episode_result(
    result: EvalResult, done: int, total: int, elapsed_s: float
) -> None:
    status = "success" if result.success else "fail"
    if result.error:
        status = "error"
    print(
        f"episode {done}/{total} source_index={result.source_index} {status} "
        f"steps={result.completed_steps} policy_calls={result.policy_call_count} "
        f"max_cube_z={result.max_cube_z:.4f} elapsed_s={elapsed_s:.1f}",
        flush=True,
    )


def _print_summary(results: list[EvalResult]) -> None:
    successes = sum(1 for result in results if result.success)
    rate = successes / len(results) if results else 0.0
    failed_indices = [result.source_index for result in results if not result.success]
    error_indices = [result.source_index for result in results if result.error]
    heights = np.asarray([result.max_cube_z for result in results], dtype=np.float32)
    finite = heights[np.isfinite(heights)]

    print(
        f"total_episodes={len(results)} successes={successes} success_rate={rate:.3%}",
        flush=True,
    )
    print(f"failed_indices={failed_indices}", flush=True)
    if finite.size:
        print(
            "max_cube_z_summary="
            f"min={float(np.min(finite)):.4f} "
            f"mean={float(np.mean(finite)):.4f} "
            f"max={float(np.max(finite)):.4f}",
            flush=True,
        )
    else:
        print("max_cube_z_summary=min=nan mean=nan max=nan", flush=True)
    if error_indices:
        print(f"error_indices={error_indices}", flush=True)
        for result in results:
            if result.error:
                print(
                    f"episode {result.source_index} error: {result.error}", flush=True
                )


def _compact_payload(
    results: list[EvalResult],
    *,
    cfg: DictConfig,
    runtime: PolicyRuntime,
    env: TabletopManipulationEnv,
    selected: list[int],
    source_count: int,
    backend: str,
    dt: float,
    initial_pose_mode: str,
    initial_pose_path: Path | None,
    initial_low_level_q: np.ndarray | None,
) -> dict[str, object]:
    source_seed = np.asarray(
        [
            -1 if result.source_seed is None else int(result.source_seed)
            for result in results
        ],
        dtype=np.int32,
    )
    has_source_seed = np.asarray(
        [result.source_seed is not None for result in results], dtype=np.bool_
    )
    success = np.asarray([result.success for result in results], dtype=np.bool_)
    max_cube_z = np.asarray([result.max_cube_z for result in results], dtype=np.float32)
    evaluated_indices = [int(result.source_index) for result in results]
    payload: dict[str, object] = {
        "schema_version": np.asarray(1, dtype=np.int32),
        "high_level_config_path": np.asarray(str(runtime.high_level_config_path)),
        "high_level_payload_path": np.asarray(str(runtime.payload_path)),
        "reset_source_pt_path": np.asarray(str(runtime.reset_source_path)),
        "reset_source_kind": np.asarray(str(runtime.reset_source_kind)),
        "high_level_checkpoint_path": np.asarray(str(runtime.checkpoint_path)),
        "high_level_checkpoint_state_key": np.asarray(str(runtime.state_key)),
        "high_level_action_space_type": np.asarray(str(runtime.action_space_type)),
        "split": np.asarray(str(runtime.dataset.split)),
        "source_episode_count": np.asarray(int(source_count), dtype=np.int32),
        "source_indices": np.asarray(evaluated_indices, dtype=np.int32),
        "embodiment": np.asarray(str(cfg.env.embodiment)),
        "mode": np.asarray(str(cfg.env.mode)),
        "task": np.asarray(
            str(OmegaConf.select(cfg, "task.name", default="cube_pick"))
        ),
        "object_kind": np.asarray(
            str(getattr(env.scene, "object_kind", "cube") or "cube")
        ),
        "backend": np.asarray(str(backend)),
        "dt": np.asarray(float(dt), dtype=np.float32),
        "frame_skip": np.asarray(int(env.backend.frame_skip), dtype=np.int32),
        "gravity_compensation": np.asarray(env.gravity_compensation),
        "action_schedule": np.asarray(
            str(OmegaConf.select(cfg, "policy_eval.action_schedule", default="chunked"))
        ),
        "max_policy_frames": np.asarray(
            int(_max_policy_frames_from_cfg(cfg)), dtype=np.int32
        ),
        "max_steps": np.asarray(int(_max_policy_frames_from_cfg(cfg)), dtype=np.int32),
        "controller_steps_per_prediction": np.asarray(
            int(
                _positive_int_from_cfg(
                    cfg, "policy_eval.controller_steps_per_prediction", default=1
                )
            ),
            dtype=np.int32,
        ),
        "controller_arm_kp_scale": np.asarray(
            float(OmegaConf.select(cfg, "controller.arm_kp_scale", default=1.0)),
            dtype=np.float32,
        ),
        "controller_arm_kv_scale": np.asarray(
            float(OmegaConf.select(cfg, "controller.arm_kv_scale", default=1.0)),
            dtype=np.float32,
        ),
        "controller_arm_force_scale": np.asarray(
            float(OmegaConf.select(cfg, "controller.arm_force_scale", default=1.0)),
            dtype=np.float32,
        ),
        "controller_hand_kp_scale": np.asarray(
            float(OmegaConf.select(cfg, "controller.hand_kp_scale", default=1.0)),
            dtype=np.float32,
        ),
        "controller_hand_kv_scale": np.asarray(
            float(OmegaConf.select(cfg, "controller.hand_kv_scale", default=1.0)),
            dtype=np.float32,
        ),
        "controller_hand_force_scale": np.asarray(
            float(OmegaConf.select(cfg, "controller.hand_force_scale", default=1.0)),
            dtype=np.float32,
        ),
        "proprio_source": np.asarray(str(_proprio_source_from_cfg(cfg))),
        "initial_pose_mode": np.asarray(str(initial_pose_mode)),
        "initial_pose_path": np.asarray(
            "" if initial_pose_path is None else str(initial_pose_path)
        ),
        "success": success,
        "success_count": np.asarray(int(np.sum(success)), dtype=np.int32),
        "success_rate": np.asarray(
            float(np.mean(success)) if success.size else 0.0, dtype=np.float32
        ),
        "failed_indices": np.asarray(
            [result.source_index for result in results if not result.success],
            dtype=np.int32,
        ),
        "error_indices": np.asarray(
            [result.source_index for result in results if result.error], dtype=np.int32
        ),
        "error_messages": np.asarray([result.error for result in results]),
        "completed_steps": np.asarray(
            [result.completed_steps for result in results], dtype=np.int32
        ),
        "policy_call_count": np.asarray(
            [result.policy_call_count for result in results], dtype=np.int32
        ),
        "max_cube_z": max_cube_z,
        "max_object_z": max_cube_z,
        "final_cube_pose": np.asarray(
            [result.final_cube_pose for result in results], dtype=np.float32
        ),
        "final_object_pose": np.asarray(
            [result.final_cube_pose for result in results], dtype=np.float32
        ),
        "source_trajectory_seed": source_seed,
        "has_source_trajectory_seed": has_source_seed,
        "success_criterion": np.asarray(
            "task_observation_or_lift_threshold"
            if hasattr(env.task, "lift_success_z")
            else "task_observation"
        ),
        "robot_xml_path": np.asarray(str(env.scene.robot_xml_path)),
        "group_names": np.asarray(env.robot.group_names),
    }
    if hasattr(env.task, "lift_success_z"):
        payload["lift_success_z"] = np.asarray(
            float(env.task.lift_success_z), dtype=np.float32
        )
    if (
        str(OmegaConf.select(cfg, "task.name", default="cube_pick"))
        == "object_suite_pick_place"
    ):
        object_ids: list[str] = []
        for source_index in evaluated_indices:
            reset_source_index = int(runtime.source_episode_indices[source_index])
            object_id = _trajectory_value(
                runtime.source, "object_id", reset_source_index
            )
            object_ids.append("" if object_id is None else str(object_id))
        payload["object_id"] = np.asarray(object_ids)
        payload["object_collision_mesh_mode"] = np.asarray(
            str(
                OmegaConf.select(
                    cfg, "objects.collision_mesh_mode", default="single_mesh"
                )
            )
        )
        object_spec = getattr(env.task, "object_spec", None)
        collision_paths = getattr(object_spec, "effective_collision_mesh_paths", ())
        collision_part_count = len(collision_paths) if collision_paths else 1
        payload["object_collision_part_count"] = np.full(
            len(results),
            collision_part_count,
            dtype=np.int32,
        )
    for group_key in ("base_xyz", "base_rpy", "fingers"):
        if OmegaConf.select(cfg, f"controller.{group_key}", default=None) is None:
            continue
        for field_key in ("kp", "kv", "max_effort"):
            value = OmegaConf.select(
                cfg, f"controller.{group_key}.{field_key}", default=None
            )
            if value is not None and str(value).strip().lower() not in {
                "",
                "null",
                "none",
                "auto",
            }:
                payload[f"controller_{group_key}_{field_key}"] = np.asarray(
                    float(value), dtype=np.float32
                )
    if runtime.high_low_runtime is not None:
        payload.update(
            {
                "low_level_config_path": np.asarray(
                    ""
                    if runtime.low_level_config_path is None
                    else str(runtime.low_level_config_path)
                ),
                "low_level_checkpoint_path": np.asarray(
                    str(runtime.high_low_runtime.low_level_checkpoint_path)
                ),
                "low_level_manipulator": np.asarray(
                    str(runtime.high_low_runtime.low_level_manipulator)
                ),
                "low_level_backend": np.asarray(
                    str(runtime.high_low_runtime.encoder_backend)
                ),
                "low_level_local_dof": np.asarray(
                    int(runtime.high_low_runtime.local_dof), dtype=np.int32
                ),
                "reset_source_indices": np.asarray(
                    runtime.source_episode_indices[evaluated_indices], dtype=np.int32
                ),
            }
        )
        if runtime.reset_source_kind == "robot_model_q":
            payload["robot_model_q_source_pt_path"] = np.asarray(
                str(runtime.reset_source_path)
            )
        if initial_low_level_q is not None:
            low_level_manipulator = (
                str(runtime.high_low_runtime.low_level_manipulator).strip().lower()
            )
            key = (
                "initial_robot_model_q_world"
                if low_level_manipulator in {"allegro_right", "barrett"}
                else "initial_wuji_q_world"
            )
            payload[key] = np.asarray(initial_low_level_q, dtype=np.float32)
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
        "policy_call_count",
        "max_cube_z",
        "max_object_z",
        "final_cube_pose",
        "final_object_pose",
        "object_id",
        "object_collision_part_count",
        "source_trajectory_seed",
        "has_source_trajectory_seed",
        "reset_source_indices",
    )
    for key in concat_keys:
        arrays = [payload[key] for payload in payloads if key in payload]
        if arrays:
            merged[key] = np.concatenate(arrays, axis=0)

    source_indices = np.asarray(merged["source_indices"], dtype=np.int32)
    if source_indices.size:
        order = np.argsort(source_indices, kind="stable")
        source_indices = source_indices[order]
        for key in concat_keys:
            if key not in merged:
                continue
            value = np.asarray(merged[key])
            if value.ndim > 0 and int(value.shape[0]) == int(order.shape[0]):
                merged[key] = value[order]
        merged["source_indices"] = source_indices
    success = np.asarray(merged["success"], dtype=np.bool_)
    errors = np.asarray(merged["error_messages"])
    error_mask = np.asarray([str(x) != "" for x in errors.reshape(-1)], dtype=np.bool_)
    merged["success_count"] = np.asarray(int(np.sum(success)), dtype=np.int32)
    merged["success_rate"] = np.asarray(
        float(np.mean(success)) if success.size else 0.0, dtype=np.float32
    )
    merged["failed_indices"] = source_indices[~success].astype(np.int32, copy=False)
    merged["error_indices"] = source_indices[error_mask].astype(np.int32, copy=False)
    return merged


def _results_from_payload(payload: dict[str, object]) -> list[EvalResult]:
    source_indices = np.asarray(payload["source_indices"], dtype=np.int32).reshape(-1)
    seeds = np.asarray(payload["source_trajectory_seed"], dtype=np.int32).reshape(-1)
    has_seed = np.asarray(
        payload["has_source_trajectory_seed"], dtype=np.bool_
    ).reshape(-1)
    completed = np.asarray(payload["completed_steps"], dtype=np.int32).reshape(-1)
    calls = np.asarray(payload["policy_call_count"], dtype=np.int32).reshape(-1)
    success = np.asarray(payload["success"], dtype=np.bool_).reshape(-1)
    max_z = np.asarray(payload["max_cube_z"], dtype=np.float32).reshape(-1)
    final_pose = np.asarray(payload["final_cube_pose"], dtype=np.float32).reshape(
        len(source_indices), 7
    )
    errors = np.asarray(payload["error_messages"]).reshape(-1)
    return [
        EvalResult(
            source_index=int(source_indices[idx]),
            source_seed=None if not bool(has_seed[idx]) else int(seeds[idx]),
            completed_steps=int(completed[idx]),
            policy_call_count=int(calls[idx]),
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


if __name__ == "__main__":
    from paper_config import canonicalize_cli
    canonicalize_cli()
    main()
