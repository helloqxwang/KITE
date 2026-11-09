"""Collect balanced YCB demonstrations with generated or hand-selected grasps."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.embodiments import validate_embodiment_supported
from experiment.high_level_policy.payload import (
    build_common_meta,
    build_training_payload,
    concat_single_manipulator_action,
    make_episode_ends,
    previous_action_proprio,
    save_high_level_payload,
    single_manipulator_action_space,
    with_object_only_world_observation,
)
from experiment.high_level_policy.pointcloud_contract import validate_object_only_world_points
from experiment.tabletop_sim.factory import optional_project_path
from experiment.tabletop_sim.object_suite import (
    FIXED_GRASP_SCHEMA,
    FixedGraspSet,
    GraspBank,
    ObjectSpec,
    load_object_suite,
)
from experiment.tabletop_sim.scripted_policy import (
    FrankaObjectPickPlaceConfig,
    build_franka_object_pick_only_replay,
    build_franka_object_pick_replay,
    build_franka_object_transfer_release_replay,
    choose_reachable_object_grasp,
    franka_object_contact_flags,
    minimize_parallel_jaw_grasp_rotation,
    validate_fixed_object_grasp_reachable,
)


PHASE_IDS = {
    "pregrasp": 0,
    "approach": 1,
    "close": 2,
    "lift": 3,
    "transfer": 4,
    "release": 5,
    "approach_high": 6,
}


from paper_config import runtime_config


@dataclass(frozen=True)
class AttemptGrasp:
    pose_object: np.ndarray
    candidate_rank: int | None
    score: float | None
    pad_swap_applied: bool


@hydra.main(version_base="1.2", config_path="conf", config_name="collect_object_suite/default")
def main(cfg: DictConfig) -> None:
    cfg = runtime_config(cfg)
    OmegaConf.resolve(cfg)
    _validate_pointcloud_config(cfg)
    validate_embodiment_supported(
        str(cfg.env.embodiment),
        "object_suite_pick_place_synthesis",
        label="env.embodiment",
    )

    objects = load_object_suite(
        optional_project_path(cfg.objects.manifest),
        mass_kg_override=OmegaConf.select(cfg, "objects.mass_kg_override", default=None),
    )
    grasp_pose_mode = _grasp_pose_mode_from_config(cfg)
    collection_objects = _collection_objects_from_config(cfg, objects)
    fixed_grasps, fixed_grasp_path = _fixed_grasps_from_config(
        cfg,
        objects=tuple(spec for _, spec in collection_objects),
        mode=grasp_pose_mode,
    )
    objective = _task_objective_from_config(cfg)
    target_counts = _collection_target_counts(cfg, collection_objects)
    bank_dir = None
    if grasp_pose_mode == "generated":
        resolved_bank_dir = optional_project_path(cfg.objects.grasp_bank_dir)
        if resolved_bank_dir is None:
            raise ValueError("objects.grasp_bank_dir must be set in generated grasp mode")
        bank_dir = Path(resolved_bank_dir)
    policy_config = _policy_from_config(cfg)
    contact_validation = _contact_validation_from_config(cfg)
    require_both_pads_after_lift = contact_validation["require_both_pads_after_lift"]
    require_both_pads_during_attempt = contact_validation["require_both_pads_during_attempt"]
    require_no_pad_contact_at_end = contact_validation["require_no_pad_contact_at_end"]
    from experiment.tabletop_sim.factory import build_env_from_config
    saved: list[dict[str, object]] = []
    dt: float | None = None

    for object_index, object_spec in collection_objects:
        target_count = target_counts[object_spec.object_id]
        max_attempts = target_count * int(cfg.dataset.max_attempt_multiplier)
        object_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        object_cfg.objects.active_id = object_spec.object_id
        asset_hash = object_spec.asset_hash
        bank = None
        fixed_pose = None
        fixed_source_candidate_rank = None
        fixed_source_candidate_score = None
        if grasp_pose_mode == "generated":
            if bank_dir is None:
                raise RuntimeError("Generated grasp mode requires objects.grasp_bank_dir")
            bank = GraspBank.load(
                bank_dir / f"{object_spec.object_id}.npz",
                expected_object_id=object_spec.object_id,
                expected_asset_hash=asset_hash,
            )
        else:
            if fixed_grasps is None:
                raise RuntimeError("Fixed grasp mode did not load a fixed-grasp artifact")
            fixed_pose = fixed_grasps.pose_for(object_spec.object_id)
            fixed_source_candidate_rank, fixed_source_candidate_score = (
                fixed_grasps.source_candidate_for(object_spec.object_id)
            )
        env = build_env_from_config(object_cfg, render=bool(cfg.runtime.viewer))
        _validate_pointcloud_runtime(env)
        object_dt = float(env.model.opt.timestep) * int(env.backend.frame_skip)
        dt = object_dt if dt is None else dt
        if abs(float(dt) - object_dt) > 1.0e-9:
            raise ValueError("All object environments must use the same dt")

        successes = 0
        attempts = 0
        try:
            while successes < target_count and attempts < max_attempts:
                seed = int(cfg.dataset.seed) + object_index * 100_000 + attempts
                obs, _ = env.reset(seed=seed)
                initial_z = float(obs["object_pose"][2])
                rng = np.random.default_rng(seed)
                try:
                    selected_grasp = _grasp_for_attempt(
                        env,
                        mode=grasp_pose_mode,
                        bank=bank,
                        fixed_pose_object=fixed_pose,
                        config=policy_config,
                        max_candidates=int(cfg.policy.max_grasp_candidates),
                        rng=rng,
                    )
                    if objective == "pick_only":
                        pick_actions = build_franka_object_pick_only_replay(
                            env,
                            grasp_pose_object=selected_grasp.pose_object,
                            config=policy_config,
                        )
                    else:
                        pick_actions = build_franka_object_pick_replay(
                            env,
                            grasp_pose_object=selected_grasp.pose_object,
                            config=policy_config,
                        )
                    pick_attempt = _run_attempt(env, pick_actions)
                    pick_lift = float(np.max(pick_attempt["object_pose"][:, 2]) - initial_z)
                    held_after_lift = bool(pick_attempt["contact_flags"][-1, 2])
                    if pick_lift < float(env.task.lift_delta) or (
                        require_both_pads_after_lift and not held_after_lift
                    ):
                        raise RuntimeError("Object was not securely held at the end of the lift")
                    if objective == "pick_only":
                        attempt = pick_attempt
                    else:
                        release_actions = build_franka_object_transfer_release_replay(
                            env,
                            config=policy_config,
                        )
                        attempt = _concatenate_attempts(
                            pick_attempt,
                            _run_attempt(env, release_actions),
                        )
                except (RuntimeError, ValueError) as exc:
                    print(f"object={object_spec.object_id} attempt={attempts} seed={seed} failed={exc}")
                    attempts += 1
                    continue

                max_lift = float(np.max(attempt["object_pose"][:, 2]) - initial_z)
                final_contact = bool(np.any(attempt["contact_flags"][-1, :2]))
                held_during_attempt = bool(np.any(attempt["contact_flags"][:, 2]))
                success = bool(
                    max_lift >= float(env.task.lift_delta)
                    and (held_during_attempt or not require_both_pads_during_attempt)
                    and bool(attempt["task_success"][-1])
                    and (not final_contact or not require_no_pad_contact_at_end)
                )
                print(
                    f"objective={objective} object={object_spec.object_id} attempt={attempts} "
                    f"saved={successes}/{target_count} success={success} max_lift={max_lift:.3f} "
                    f"placement_margin={float(attempt['placement_margin'][-1]):.3f} "
                    f"pad_swap={selected_grasp.pad_swap_applied}"
                )
                if success:
                    attempt.update(
                        object_index=object_index,
                        object_id=object_spec.object_id,
                        object_name=object_spec.name,
                        object_asset_hash=asset_hash,
                        object_bounds_m=np.asarray(object_spec.mesh_bounds_m, dtype=np.float32),
                        object_mass_kg=float(object_spec.mass_kg),
                        object_upright_quat_wxyz=np.asarray(
                            object_spec.upright_quat_wxyz,
                            dtype=np.float32,
                        ),
                        trajectory_seed=seed,
                        grasp_pose_mode=grasp_pose_mode,
                        grasp_pose_object=np.asarray(selected_grasp.pose_object, dtype=np.float32),
                        parallel_jaw_pad_swap_applied=bool(selected_grasp.pad_swap_applied),
                        max_lift=max_lift,
                    )
                    if selected_grasp.candidate_rank is not None and selected_grasp.score is not None:
                        attempt.update(
                            candidate_rank=int(selected_grasp.candidate_rank),
                            grasp_score=float(selected_grasp.score),
                        )
                    else:
                        attempt.update(
                            fixed_grasp_pose_path=str(fixed_grasp_path),
                            fixed_grasp_artifact_sha256=str(fixed_grasps.file_sha256),
                            fixed_grasp_source_candidate_rank=int(fixed_source_candidate_rank),
                            fixed_grasp_source_candidate_score=float(fixed_source_candidate_score),
                        )
                    saved.append(attempt)
                    successes += 1
                attempts += 1
        finally:
            env.close()
        if successes != target_count:
            raise RuntimeError(
                f"Collected {successes}/{target_count} successes for {object_spec.object_id} "
                f"after {attempts} attempts"
            )

    output = optional_project_path(cfg.dataset.save_training_pt)
    if output is None or dt is None:
        raise ValueError("dataset.save_training_pt must be set")
    path = _save_payload(Path(output), cfg=cfg, episodes=saved, dt=float(dt))
    print(
        f"saved {len(saved)} object-suite demonstrations "
        f"for objective={objective}: {path}"
    )


def _collection_objects_from_config(
    cfg: DictConfig,
    objects: tuple[ObjectSpec, ...],
) -> tuple[tuple[int, ObjectSpec], ...]:
    configured = OmegaConf.select(cfg, "objects.collection_ids", default=None)
    if configured is None or str(configured).strip().lower() in {"", "null", "none"}:
        return tuple(enumerate(objects))

    requested = [str(configured)] if isinstance(configured, str) else [str(value) for value in configured]
    if not requested or any(not object_id.strip() for object_id in requested):
        raise ValueError("objects.collection_ids must contain at least one nonempty object ID")
    if len(set(requested)) != len(requested):
        raise ValueError(f"objects.collection_ids contains duplicates: {requested}")

    indexed = {spec.object_id: (index, spec) for index, spec in enumerate(objects)}
    unknown = [object_id for object_id in requested if object_id not in indexed]
    if unknown:
        raise ValueError(
            f"objects.collection_ids contains unknown IDs {unknown}; "
            f"available={list(indexed)}"
        )
    return tuple(indexed[object_id] for object_id in requested)


def _collection_target_counts(
    cfg: DictConfig,
    collection_objects: tuple[tuple[int, ObjectSpec], ...],
) -> dict[str, int]:
    """Resolve a deterministic, near-uniform episode count for each object."""

    if not collection_objects:
        raise ValueError("Object-suite collection selected no objects")
    total = OmegaConf.select(cfg, "dataset.num_trajectories_total", default=None)
    if total is None or str(total).strip().lower() in {"", "null", "none"}:
        per_object = int(cfg.dataset.num_trajectories_per_object)
        if per_object <= 0:
            raise ValueError("dataset.num_trajectories_per_object must be positive")
        return {spec.object_id: per_object for _, spec in collection_objects}

    total_count = int(total)
    if total_count <= 0:
        raise ValueError("dataset.num_trajectories_total must be positive when set")
    if total_count < len(collection_objects):
        raise ValueError(
            "dataset.num_trajectories_total must be at least the number of selected objects "
            f"({len(collection_objects)})"
        )
    base, remainder = divmod(total_count, len(collection_objects))
    return {
        spec.object_id: base + int(order < remainder)
        for order, (_, spec) in enumerate(collection_objects)
    }


def _task_objective_from_config(cfg: DictConfig) -> str:
    objective = str(
        OmegaConf.select(cfg, "task.objective", default="pick_place")
    ).strip().lower()
    if objective not in {"pick_place", "pick_only"}:
        raise ValueError(
            "task.objective must be 'pick_place' or 'pick_only', "
            f"got {objective!r}"
        )
    return objective


def _validate_pointcloud_config(cfg: DictConfig) -> None:
    required_true = (
        "perception.include_camera_obs",
        "perception.capture_segmentation",
        "perception.object_only_pointcloud",
    )
    disabled = [path for path in required_true if not bool(OmegaConf.select(cfg, path, default=False))]
    if disabled:
        raise ValueError(
            "Object-suite collection requires object-mask-only camera observations; "
            f"set these options to true: {disabled}"
        )
    frame = str(OmegaConf.select(cfg, "perception.pointcloud_frame", default="")).strip().lower()
    centering = str(OmegaConf.select(cfg, "perception.pointcloud_centering", default="")).strip().lower()
    if frame != "world" or centering != "none":
        raise ValueError(
            "Object-suite point clouds must use the fixed MuJoCo world frame without recentering; "
            f"got pointcloud_frame={frame!r}, pointcloud_centering={centering!r}"
        )
    erosion_pixels = int(OmegaConf.select(cfg, "perception.object_mask_erosion_pixels", default=-1))
    if erosion_pixels != 1:
        raise ValueError(
            "The object-only world-frame contract requires perception.object_mask_erosion_pixels=1; "
            f"got {erosion_pixels}"
        )
    depth_threshold = float(
        OmegaConf.select(cfg, "perception.object_depth_neighbor_threshold_m", default=-1.0)
    )
    if abs(depth_threshold - 0.03) > 1.0e-12:
        raise ValueError(
            "The object-only world-frame contract requires "
            "perception.object_depth_neighbor_threshold_m=0.03; "
            f"got {depth_threshold}"
        )
    bounds_tolerance = float(
        OmegaConf.select(cfg, "perception.object_bounds_filter_tolerance_m", default=-1.0)
    )
    if abs(bounds_tolerance - 0.005) > 1.0e-12:
        raise ValueError(
            "The object-only world-frame contract requires "
            "perception.object_bounds_filter_tolerance_m=0.005; "
            f"got {bounds_tolerance}"
        )
    if int(OmegaConf.select(cfg, "perception.target_pointcloud_num", default=0)) <= 0:
        raise ValueError("perception.target_pointcloud_num must be positive")


def _validate_pointcloud_runtime(env) -> None:
    if not bool(env.include_camera_obs):
        raise RuntimeError("Built environment has include_camera_obs=false")
    if not bool(env.capture_camera_segmentation):
        raise RuntimeError("Built environment has capture_camera_segmentation=false")
    if not bool(env.object_only_pointcloud):
        raise RuntimeError("Built environment has object_only_pointcloud=false")
    if int(env.object_mask_erosion_pixels) != 1:
        raise RuntimeError("Built environment must use one-pixel object-mask erosion")
    if abs(float(env.object_depth_neighbor_threshold_m) - 0.03) > 1.0e-12:
        raise RuntimeError("Built environment must use the 0.03 m object-depth neighbor filter")
    if (
        env.object_bounds_filter_tolerance_m is None
        or abs(float(env.object_bounds_filter_tolerance_m) - 0.005) > 1.0e-12
    ):
        raise RuntimeError("Built environment must use the 0.005 m object-bounds point filter")


def _grasp_pose_mode_from_config(cfg: DictConfig) -> str:
    mode = str(
        OmegaConf.select(cfg, "policy.grasp_pose_mode", default="generated")
    ).strip().lower()
    if mode not in {"fixed", "generated"}:
        raise ValueError(
            "policy.grasp_pose_mode must be 'fixed' or 'generated', "
            f"got {mode!r}"
        )
    return mode


def _fixed_grasps_from_config(
    cfg: DictConfig,
    *,
    objects,
    mode: str,
) -> tuple[FixedGraspSet | None, Path | None]:
    if mode != "fixed":
        return None, None
    configured = OmegaConf.select(cfg, "policy.fixed_grasp_pose_path", default=None)
    resolved = optional_project_path(configured)
    if resolved is None:
        raise ValueError(
            "policy.fixed_grasp_pose_path must be set when policy.grasp_pose_mode=fixed"
        )
    path = Path(resolved)
    try:
        # A selected-object run must not require meshes for the rest of the suite.
        # Still verify every selected object's exact saved asset hash.
        fixed_grasps = FixedGraspSet.load(path).reordered_for(objects, allow_extra=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Fixed grasp mode requires {path}. Install it with `python download_assets.py`."
        ) from exc
    return fixed_grasps, path


def _grasp_for_attempt(
    env,
    *,
    mode: str,
    bank: GraspBank | None,
    fixed_pose_object: np.ndarray | None,
    config: FrankaObjectPickPlaceConfig,
    max_candidates: int,
    rng: np.random.Generator,
) -> AttemptGrasp:
    candidate_rank: int | None
    score: float | None
    if mode == "generated":
        if bank is None:
            raise ValueError("Generated grasp mode requires a GraspGenX bank")
        candidate_rank, raw_pose = choose_reachable_object_grasp(
            env,
            bank.poses_object,
            config=config,
            max_candidates=max_candidates,
            rng=rng,
        )
        candidate_rank = int(candidate_rank)
        score = float(bank.scores[candidate_rank])
    elif mode == "fixed":
        if fixed_pose_object is None:
            raise ValueError("Fixed grasp mode requires one fixed object_from_gripper pose")
        raw_pose = np.asarray(fixed_pose_object, dtype=np.float64).reshape(4, 4).copy()
        candidate_rank = None
        score = None
    else:
        raise ValueError(f"Unsupported grasp pose mode: {mode!r}")

    pose, pad_swap_applied = minimize_parallel_jaw_grasp_rotation(
        env,
        raw_pose,
        enabled=bool(config.use_parallel_jaw_symmetry),
    )
    swapped_pose_validated = False
    if pad_swap_applied:
        try:
            validate_fixed_object_grasp_reachable(env, pose, config=config)
            swapped_pose_validated = True
        except RuntimeError:
            # The original generated pose was already checked by its selector;
            # fixed mode validates it below. Symmetry must not remove a usable grasp.
            pose = np.asarray(raw_pose, dtype=np.float64).copy()
            pad_swap_applied = False
    if mode == "fixed" and not swapped_pose_validated:
        validate_fixed_object_grasp_reachable(env, pose, config=config)

    return AttemptGrasp(
        pose_object=np.asarray(pose, dtype=np.float64).copy(),
        candidate_rank=candidate_rank,
        score=score,
        pad_swap_applied=bool(pad_swap_applied),
    )


def _policy_from_config(cfg: DictConfig) -> FrankaObjectPickPlaceConfig:
    return FrankaObjectPickPlaceConfig(
        rollout_len=int(cfg.policy.rollout_len),
        approach_distance=float(
            OmegaConf.select(cfg, "policy.approach_distance", default=0.20)
        ),
        pregrasp_distance=float(cfg.policy.pregrasp_distance),
        lift_distance=float(cfg.policy.lift_distance),
        min_downward_angle_deg=float(cfg.policy.min_downward_angle_deg),
        gripper_base_to_tcp=float(cfg.policy.gripper_base_to_tcp),
        use_parallel_jaw_symmetry=bool(
            OmegaConf.select(cfg, "policy.use_parallel_jaw_symmetry", default=True)
        ),
        open_gripper=float(cfg.policy.open_gripper),
        closed_gripper=float(cfg.policy.closed_gripper),
        timing_mode=str(
            OmegaConf.select(cfg, "policy.timing_mode", default="even_tcp_speed")
        ),
    )


def _contact_validation_from_config(cfg: DictConfig) -> dict[str, bool]:
    return {
        "require_both_pads_after_lift": bool(
            OmegaConf.select(cfg, "contact_validation.require_both_pads_after_lift", default=True)
        ),
        "require_both_pads_during_attempt": bool(
            OmegaConf.select(cfg, "contact_validation.require_both_pads_during_attempt", default=True)
        ),
        "require_no_pad_contact_at_end": bool(
            OmegaConf.select(cfg, "contact_validation.require_no_pad_contact_at_end", default=True)
        ),
    }


def _run_attempt(env, actions: list[dict[str, object]]) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = {
        "object_pose": [],
        "bin_pose": [],
        "placement_margin": [],
        "task_success": [],
        "action_ee_pos": [],
        "action_ee_quat_wxyz": [],
        "action_gripper": [],
        "executed_qpos": [],
        "executed_qvel": [],
        "contact_flags": [],
        "pointcloud": [],
        "phase": [],
        "pointcloud_occluded": [],
    }
    last_pointcloud: np.ndarray | None = None
    for action in actions:
        obs, _, _, _, info = env.step(action)
        command = info["command"]
        right = command["sides"]["right"]
        flags = franka_object_contact_flags(env)
        values["object_pose"].append(np.asarray(obs["object_pose"], dtype=np.float32))
        values["bin_pose"].append(np.asarray(obs["bin_pose"], dtype=np.float32))
        values["placement_margin"].append(np.asarray(obs["placement_margin"], dtype=np.float32))
        values["task_success"].append(np.asarray(obs["success"], dtype=np.bool_))
        values["action_ee_pos"].append(np.asarray(right["ee_pos_target"], dtype=np.float32))
        values["action_ee_quat_wxyz"].append(np.asarray(right["ee_quat_wxyz_target"], dtype=np.float32))
        values["action_gripper"].append(np.asarray(right["gripper_target"], dtype=np.float32))
        values["executed_qpos"].append(np.asarray(command["executed_qpos"], dtype=np.float32))
        values["executed_qvel"].append(np.asarray(command["executed_qvel"], dtype=np.float32))
        values["contact_flags"].append(
            np.asarray([flags["left_pad_contact"], flags["right_pad_contact"], flags["both_pad_contact"]])
        )
        pointcloud = np.asarray(obs["pointcloud"]["xyzw"], dtype=np.float32)[..., :3]
        occluded = not bool(np.any(np.linalg.norm(pointcloud, axis=-1) > 1.0e-6))
        if occluded:
            if last_pointcloud is None:
                raise RuntimeError("Object point cloud is empty before any visible frame was captured")
            pointcloud = last_pointcloud.copy()
        else:
            validate_object_only_world_points(
                pointcloud,
                obs["object_pose"],
                env.task.object_spec.mesh_bounds_m,
                label="collected object point cloud",
            )
            last_pointcloud = pointcloud.copy()
        values["pointcloud"].append(pointcloud)
        values["pointcloud_occluded"].append(np.asarray(occluded, dtype=np.bool_))
        values["phase"].append(np.asarray(PHASE_IDS[str(action["phase"])], dtype=np.int8))
    return {key: np.asarray(items) for key, items in values.items()}


def _concatenate_attempts(*parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([np.asarray(part[key]) for part in parts], axis=0)
        for key in parts[0]
    }


def _save_payload(path: Path, *, cfg: DictConfig, episodes: list[dict[str, object]], dt: float) -> Path:
    lengths = [int(np.asarray(episode["object_pose"]).shape[0]) for episode in episodes]
    episode_ends = make_episode_ends(lengths)
    pointcloud = np.concatenate([np.asarray(episode["pointcloud"]) for episode in episodes], axis=0)
    ee_pos = np.concatenate([np.asarray(episode["action_ee_pos"]) for episode in episodes], axis=0)
    ee_quat = np.concatenate([np.asarray(episode["action_ee_quat_wxyz"]) for episode in episodes], axis=0)
    gripper = np.concatenate([np.asarray(episode["action_gripper"]) for episode in episodes], axis=0)
    action = concat_single_manipulator_action(ee_pos, ee_quat, gripper)
    proprio = previous_action_proprio(action, episode_ends)
    action_space = single_manipulator_action_space(joint_key="action_gripper", joint_dim=gripper.shape[-1])
    episode_meta: list[dict[str, object]] = []
    for index, episode in enumerate(episodes):
        item: dict[str, object] = {
            "episode_index": index,
            "object_index": int(episode["object_index"]),
            "object_id": str(episode["object_id"]),
            "object_name": str(episode["object_name"]),
            "object_asset_hash": str(episode["object_asset_hash"]),
            "object_bounds_m": np.asarray(episode["object_bounds_m"]).tolist(),
            "object_mass_kg": float(episode["object_mass_kg"]),
            "object_upright_quat_wxyz": np.asarray(
                episode["object_upright_quat_wxyz"]
            ).tolist(),
            "trajectory_seed": int(episode["trajectory_seed"]),
            "grasp_pose_mode": str(episode["grasp_pose_mode"]),
            "grasp_pose_object": np.asarray(episode["grasp_pose_object"]).tolist(),
            "parallel_jaw_pad_swap_applied": bool(
                episode["parallel_jaw_pad_swap_applied"]
            ),
            "max_lift": float(episode["max_lift"]),
            "success": True,
            "num_frames": lengths[index],
        }
        if episode.get("candidate_rank") is not None:
            item["candidate_rank"] = int(episode["candidate_rank"])
        if episode.get("grasp_score") is not None:
            item["grasp_score"] = float(episode["grasp_score"])
        for key, converter in (
            ("fixed_grasp_pose_path", str),
            ("fixed_grasp_artifact_sha256", str),
            ("fixed_grasp_source_candidate_rank", int),
            ("fixed_grasp_source_candidate_score", float),
        ):
            if episode.get(key) is not None:
                item[key] = converter(episode[key])
        episode_meta.append(item)

    grasp_pose_mode = str(episodes[0]["grasp_pose_mode"])
    grasp_meta: dict[str, object] = {"grasp_pose_mode": grasp_pose_mode}
    if grasp_pose_mode == "fixed":
        grasp_meta.update(
            fixed_grasp_schema=FIXED_GRASP_SCHEMA,
            fixed_grasp_pose_path=str(episodes[0]["fixed_grasp_pose_path"]),
            fixed_grasp_artifact_sha256=str(episodes[0]["fixed_grasp_artifact_sha256"]),
        )
    objective = _task_objective_from_config(cfg)
    protocol = "object_suite_pick_place_v3" if objective == "pick_place" else "object_suite_pick_only_v1"
    episode_counts = dict(Counter(str(episode["object_id"]) for episode in episodes))
    uniform_episode_count = next(iter(episode_counts.values())) if len(set(episode_counts.values())) == 1 else None
    meta = with_object_only_world_observation(
        build_common_meta(
            source={
                "kind": "scripted_tabletop_synthesis",
                "script": "experiment/collect_object_suite.py",
                "factory": str(OmegaConf.select(cfg, "runtime.factory", default="experiment")),
                "path": str(path.resolve()),
                "source_indices": list(range(len(episodes))),
            },
            task="object_suite_pick_place",
            embodiment=str(cfg.env.embodiment),
            dt=dt,
            fps=1.0 / dt,
            observation_point_count=int(pointcloud.shape[1]),
            observation_frame="world",
            observation_dtype="float32",
            action_space=action_space,
            extra={
                "version": 3,
                "protocol": protocol,
                "task_objective": objective,
                "num_episodes": len(episodes),
                "num_frames": int(sum(lengths)),
                "objects_per_suite": len({str(episode["object_id"]) for episode in episodes}),
                "collection_object_ids": list(
                    dict.fromkeys(str(episode["object_id"]) for episode in episodes)
                ),
                # Keep the legacy scalar for uniform datasets and record the
                # exact per-object distribution for finite balanced test sets.
                "episodes_per_object": uniform_episode_count,
                "episodes_per_object_counts": episode_counts,
                "phase_ids": PHASE_IDS,
                "bin_center_xy": list(cfg.bin.center_xy),
                "bin_inner_size": list(cfg.bin.inner_size),
                "bin_wall_height": float(cfg.bin.wall_height),
                "placement_reference": str(
                    OmegaConf.select(cfg, "bin.placement_reference", default="bounds")
                ),
                "placement_margin": float(cfg.bin.placement_margin),
                "object_manifest": str(optional_project_path(cfg.objects.manifest)),
                "object_mass_kg_override": OmegaConf.select(
                    cfg,
                    "objects.mass_kg_override",
                    default=None,
                ),
                "contact_validation": _contact_validation_from_config(cfg),
                "use_parallel_jaw_symmetry": bool(
                    OmegaConf.select(cfg, "policy.use_parallel_jaw_symmetry", default=True)
                ),
                "pick_timing_mode": str(
                    OmegaConf.select(cfg, "policy.timing_mode", default="even_tcp_speed")
                ),
                "approach_distance": float(
                    OmegaConf.select(cfg, "policy.approach_distance", default=0.20)
                ),
                **grasp_meta,
            },
        )
    )
    extra_data = {
        key: np.concatenate([np.asarray(episode[key]) for episode in episodes], axis=0)
        for key in (
            "object_pose",
            "bin_pose",
            "placement_margin",
            "task_success",
            "contact_flags",
            "phase",
            "pointcloud_occluded",
            "executed_qpos",
            "executed_qvel",
            "action_ee_pos",
            "action_ee_quat_wxyz",
            "action_gripper",
        )
    }
    extra_data["object_index"] = np.concatenate(
        [np.full(length, int(episode["object_index"]), dtype=np.int16) for episode, length in zip(episodes, lengths)]
    )
    payload = build_training_payload(
        obs_point_cloud=pointcloud,
        action=action,
        proprio=proprio,
        episode_ends=episode_ends,
        episode_meta=episode_meta,
        meta=meta,
        extra_data=extra_data,
    )
    return save_high_level_payload(payload, path)


if __name__ == "__main__":
    from paper_config import canonicalize_cli
    canonicalize_cli()
    main()
