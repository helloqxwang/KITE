"""Scripted tabletop policies for data collection demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from .assets import TABLE_TOP_Z
from .env import TabletopManipulationEnv
from .ik import solve_arm_ik


@dataclass(frozen=True)
class FrankaObjectPickPlaceConfig:
    rollout_len: int = 120
    approach_distance: float = 0.20
    pregrasp_distance: float = 0.10
    lift_distance: float = 0.10
    min_downward_angle_deg: float = 10.0
    gripper_base_to_tcp: float = 0.152
    use_parallel_jaw_symmetry: bool = True
    open_gripper: float = 0.0
    closed_gripper: float = 255.0
    timing_mode: str = "even_tcp_speed"


def choose_reachable_object_grasp(
    env: TabletopManipulationEnv,
    poses_object: np.ndarray,
    *,
    config: FrankaObjectPickPlaceConfig,
    max_candidates: int = 128,
    rng: np.random.Generator | None = None,
) -> tuple[int, np.ndarray]:
    """Choose among the five best reachable, downward-facing GraspGenX poses."""

    if env.robot.spec.key != "franka_robotiq":
        raise ValueError(f"Object-suite source policy requires franka_robotiq, got {env.robot.spec.key!r}")
    min_downward_angle_deg = float(config.min_downward_angle_deg)
    if not np.isfinite(min_downward_angle_deg) or not 0.0 <= min_downward_angle_deg <= 90.0:
        raise ValueError(
            "min_downward_angle_deg must be finite and between 0 and 90 degrees, "
            f"got {min_downward_angle_deg}"
        )
    object_pose = np.asarray(env.observe(include_camera_obs=False)["object_pose"], dtype=np.float64)
    table_top_z = float(getattr(env.task, "table_top_z", TABLE_TOP_Z))
    world_from_object = _pose7_to_matrix(object_pose)
    feasible: list[tuple[int, np.ndarray]] = []
    candidate_limit = min(len(poses_object), max(0, int(max_candidates)))
    max_approach_z = -float(np.sin(np.deg2rad(min_downward_angle_deg)))
    downward_count = 0
    for index, object_from_gripper in enumerate(np.asarray(poses_object, dtype=np.float64)[:candidate_limit]):
        world_from_gripper = world_from_object @ object_from_gripper
        grasp_pos, grasp_quat = _gripper_base_to_tcp_pose(world_from_gripper, config.gripper_base_to_tcp)
        approach = world_from_gripper[:3, 2]
        if float(approach[2]) > max_approach_z + 1.0e-9:
            continue
        downward_count += 1
        pregrasp_pos = grasp_pos - float(config.pregrasp_distance) * approach
        if min(float(grasp_pos[2]), float(pregrasp_pos[2])) <= table_top_z + 0.015:
            continue
        if _ik_reaches(env, pregrasp_pos, grasp_quat) and _ik_reaches(env, grasp_pos, grasp_quat):
            feasible.append((index, object_from_gripper.copy()))
            if len(feasible) == 5:
                break
    if feasible:
        generator = np.random.default_rng() if rng is None else rng
        return feasible[int(generator.integers(0, len(feasible)))]
    if downward_count == 0:
        raise RuntimeError(
            f"No grasp among the first {candidate_limit} candidates faces at least "
            f"{min_downward_angle_deg:g} degrees below horizontal"
        )
    raise RuntimeError(
        f"No reachable grasp among the {downward_count} downward-facing candidates "
        f"in the first {candidate_limit} grasps"
    )


def object_grasp_world_transform(
    object_pose: np.ndarray,
    grasp_pose_object: np.ndarray,
) -> np.ndarray:
    """Transform one fixed ``object_from_gripper`` grasp into the world frame."""

    grasp = np.asarray(grasp_pose_object, dtype=np.float64)
    if grasp.shape == (16,):
        grasp = grasp.reshape(4, 4)
    if grasp.shape != (4, 4) or not np.all(np.isfinite(grasp)):
        raise ValueError(
            "grasp_pose_object must be a finite 4x4 object_from_gripper transform, "
            f"got {grasp.shape}"
        )
    return _pose7_to_matrix(np.asarray(object_pose, dtype=np.float64)) @ grasp


def minimize_parallel_jaw_grasp_rotation(
    env: TabletopManipulationEnv,
    grasp_pose_object: np.ndarray,
    *,
    enabled: bool = True,
) -> tuple[np.ndarray, bool]:
    """Use Robotiq's 180-degree pad symmetry when it shortens the reset-to-grasp rotation.

    Rotating ``object_from_gripper`` by 180 degrees around the gripper's local
    +Z approach axis preserves the grasp line while exchanging the left and
    right pads. The symmetric pose is selected only when the original wrist
    rotation is over 90 degrees and the exchanged-pad pose is strictly shorter.
    """

    grasp = np.asarray(grasp_pose_object, dtype=np.float64)
    if grasp.shape == (16,):
        grasp = grasp.reshape(4, 4)
    if grasp.shape != (4, 4) or not np.all(np.isfinite(grasp)):
        raise ValueError(
            "grasp_pose_object must be a finite 4x4 object_from_gripper transform, "
            f"got {grasp.shape}"
        )
    grasp = grasp.copy()
    if not bool(enabled):
        return grasp, False
    if env.robot.spec.key != "franka_robotiq":
        raise ValueError(
            f"Parallel-jaw symmetry requires franka_robotiq, got {env.robot.spec.key!r}"
        )

    object_pose = np.asarray(env.observe(include_camera_obs=False)["object_pose"], dtype=np.float64)
    world_from_gripper = object_grasp_world_transform(object_pose, grasp)
    _, current_quat_wxyz = env.robot.wrist_pose(env.data, "right")
    current_quat_wxyz = np.asarray(current_quat_wxyz, dtype=np.float64).reshape(4)
    current_rotation = R.from_quat(
        [
            current_quat_wxyz[1],
            current_quat_wxyz[2],
            current_quat_wxyz[3],
            current_quat_wxyz[0],
        ]
    ).as_matrix()

    original_rotation = world_from_gripper[:3, :3]
    original_angle = float(R.from_matrix(current_rotation.T @ original_rotation).magnitude())
    if original_angle <= 0.5 * np.pi + 1.0e-9:
        return grasp, False

    exchange_pads = np.eye(4, dtype=np.float64)
    exchange_pads[:3, :3] = R.from_euler("Z", np.pi).as_matrix()
    symmetric_grasp = grasp @ exchange_pads
    symmetric_rotation = original_rotation @ exchange_pads[:3, :3]
    symmetric_angle = float(R.from_matrix(current_rotation.T @ symmetric_rotation).magnitude())
    if symmetric_angle + 1.0e-9 >= original_angle:
        return grasp, False
    return symmetric_grasp, True


def validate_fixed_object_grasp_reachable(
    env: TabletopManipulationEnv,
    grasp_pose_object: np.ndarray,
    *,
    config: FrankaObjectPickPlaceConfig,
) -> None:
    """Validate an exact fixed grasp without replacing it with another candidate."""

    if env.robot.spec.key != "franka_robotiq":
        raise ValueError(f"Object-suite source policy requires franka_robotiq, got {env.robot.spec.key!r}")
    object_pose = np.asarray(env.observe(include_camera_obs=False)["object_pose"], dtype=np.float64)
    world_from_gripper = object_grasp_world_transform(object_pose, grasp_pose_object)
    grasp_pos, grasp_quat = _gripper_base_to_tcp_pose(
        world_from_gripper,
        config.gripper_base_to_tcp,
    )
    approach = world_from_gripper[:3, 2]
    pregrasp_pos = grasp_pos - float(config.pregrasp_distance) * approach
    table_top_z = float(getattr(env.task, "table_top_z", TABLE_TOP_Z))
    if min(float(grasp_pos[2]), float(pregrasp_pos[2])) <= table_top_z + 0.015:
        raise RuntimeError("The fixed grasp or pregrasp is too close to or below the table")
    if not _ik_reaches(env, pregrasp_pos, grasp_quat):
        raise RuntimeError("The fixed pregrasp is not reachable by the configured Franka IK")
    if not _ik_reaches(env, grasp_pos, grasp_quat):
        raise RuntimeError("The fixed grasp is not reachable by the configured Franka IK")


def build_franka_object_pick_replay(
    env: TabletopManipulationEnv,
    *,
    grasp_pose_object: np.ndarray,
    config: FrankaObjectPickPlaceConfig | None = None,
) -> list[dict[str, Any]]:
    """Approach, close on, and lift an object using a GraspGenX pose."""

    cfg = config or FrankaObjectPickPlaceConfig()
    if env.robot.spec.key != "franka_robotiq":
        raise ValueError(f"Object-suite source policy requires franka_robotiq, got {env.robot.spec.key!r}")
    object_pose = np.asarray(env.observe(include_camera_obs=False)["object_pose"], dtype=np.float64)
    world_from_gripper = object_grasp_world_transform(object_pose, grasp_pose_object)
    grasp_pos, grasp_quat = _gripper_base_to_tcp_pose(world_from_gripper, cfg.gripper_base_to_tcp)
    pregrasp_pos = grasp_pos - float(cfg.pregrasp_distance) * world_from_gripper[:3, 2]
    lift_pos = grasp_pos.copy()
    lift_pos[2] += float(cfg.lift_distance)
    current_pos, current_quat = env.robot.wrist_pose(env.data, "right")
    lengths = _object_pick_place_phase_lengths(cfg)[:4]

    positions = np.concatenate(
        [
            _linear_segment(current_pos, pregrasp_pos, lengths[0]),
            _linear_segment(pregrasp_pos, grasp_pos, lengths[1], include_start=False),
            np.repeat(grasp_pos.reshape(1, 3), lengths[2], axis=0),
            _linear_segment(grasp_pos, lift_pos, lengths[3], include_start=False),
        ],
        axis=0,
    )
    quats = np.concatenate(
        [
            _slerp_quat_wxyz(current_quat, grasp_quat, lengths[0]),
            np.repeat(grasp_quat.reshape(1, 4), sum(lengths[1:]), axis=0),
        ],
        axis=0,
    )
    gripper = np.concatenate(
        [
            np.full(sum(lengths[:2]), float(cfg.open_gripper)),
            np.linspace(float(cfg.open_gripper), float(cfg.closed_gripper), lengths[2]),
            np.full(lengths[3], float(cfg.closed_gripper)),
        ]
    )
    phases = np.concatenate(
        [np.repeat(name, length) for name, length in zip(("pregrasp", "approach", "close", "lift"), lengths)]
    )
    return _object_actions(positions, quats, gripper, phases)


def build_franka_object_pick_only_replay(
    env: TabletopManipulationEnv,
    *,
    grasp_pose_object: np.ndarray,
    config: FrankaObjectPickPlaceConfig | None = None,
) -> list[dict[str, Any]]:
    """Pick an object using the current cube collector's five-stage timing."""

    cfg = config or FrankaObjectPickPlaceConfig()
    if env.robot.spec.key != "franka_robotiq":
        raise ValueError(
            f"Object-suite source policy requires franka_robotiq, got {env.robot.spec.key!r}"
        )
    if int(cfg.rollout_len) < 8:
        raise ValueError("rollout_len must be at least 8")
    if float(cfg.approach_distance) <= float(cfg.pregrasp_distance):
        raise ValueError("approach_distance must be greater than pregrasp_distance")

    object_pose = np.asarray(env.observe(include_camera_obs=False)["object_pose"], dtype=np.float64)
    world_from_gripper = object_grasp_world_transform(object_pose, grasp_pose_object)
    grasp_pos, grasp_quat = _gripper_base_to_tcp_pose(
        world_from_gripper,
        cfg.gripper_base_to_tcp,
    )
    approach_axis = world_from_gripper[:3, 2]
    high_approach_pos = grasp_pos - float(cfg.approach_distance) * approach_axis
    pregrasp_pos = grasp_pos - float(cfg.pregrasp_distance) * approach_axis
    lift_pos = grasp_pos.copy()
    lift_pos[2] += float(cfg.lift_distance)
    current_pos, current_quat = env.robot.wrist_pose(env.data, "right")
    waypoints = (
        current_pos,
        high_approach_pos,
        pregrasp_pos,
        grasp_pos,
        lift_pos,
    )

    timing_mode = str(cfg.timing_mode).strip().lower()
    if timing_mode == "legacy_phase_weights":
        lengths = _phase_lengths(
            int(cfg.rollout_len),
            weights=(0.25, 0.20, 0.12, 0.18, 0.25),
        )
        omit_boundary_repeats = False
    elif timing_mode == "even_tcp_speed":
        lengths = _even_tcp_speed_phase_lengths(
            int(cfg.rollout_len),
            waypoints=waypoints,
            close_weight=0.18,
        )
        omit_boundary_repeats = True
    else:
        raise ValueError(
            "FrankaObjectPickPlaceConfig.timing_mode must be "
            "'legacy_phase_weights' or 'even_tcp_speed', "
            f"got {cfg.timing_mode!r}"
        )

    positions = np.concatenate(
        [
            _linear_segment(waypoints[0], waypoints[1], lengths[0]),
            _linear_segment(
                waypoints[1],
                waypoints[2],
                lengths[1],
                include_start=not omit_boundary_repeats,
            ),
            _linear_segment(
                waypoints[2],
                waypoints[3],
                lengths[2],
                include_start=not omit_boundary_repeats,
            ),
            np.repeat(waypoints[3].reshape(1, 3), lengths[3], axis=0),
            _linear_segment(
                waypoints[3],
                waypoints[4],
                lengths[4],
                include_start=not omit_boundary_repeats,
            ),
        ],
        axis=0,
    )
    quats = np.concatenate(
        [
            _slerp_quat_wxyz(current_quat, grasp_quat, lengths[0]),
            np.repeat(grasp_quat.reshape(1, 4), sum(lengths[1:]), axis=0),
        ],
        axis=0,
    )
    gripper = np.concatenate(
        [
            np.full(sum(lengths[:3]), float(cfg.open_gripper)),
            np.linspace(float(cfg.open_gripper), float(cfg.closed_gripper), lengths[3]),
            np.full(lengths[4], float(cfg.closed_gripper)),
        ]
    )
    phases = np.concatenate(
        [
            np.repeat(name, length)
            for name, length in zip(
                ("approach_high", "pregrasp", "approach", "close", "lift"),
                lengths,
            )
        ]
    )
    return _object_actions(positions, quats, gripper, phases)


def build_franka_object_transfer_release_replay(
    env: TabletopManipulationEnv,
    *,
    config: FrankaObjectPickPlaceConfig | None = None,
) -> list[dict[str, Any]]:
    """Transfer over the bin at the post-lift height, then release and settle."""

    cfg = config or FrankaObjectPickPlaceConfig()
    obs = env.observe(include_camera_obs=False)
    current_object = np.asarray(obs["object_pose"], dtype=np.float64)
    current_tcp_pos, current_tcp_quat = env.robot.wrist_pose(env.data, "right")
    world_from_object = _pose7_to_matrix(current_object)
    object_from_tcp = np.linalg.inv(world_from_object) @ _matrix_from_pose(current_tcp_pos, current_tcp_quat)

    current_bounds = env.task.world_bounds(current_object)
    desired_object = current_object.copy()
    desired_object[:2] += np.asarray(env.task.bin_center_xy, dtype=np.float64) - 0.5 * (
        current_bounds[0, :2] + current_bounds[1, :2]
    )
    transfer_tcp = _pose7_to_matrix(desired_object) @ object_from_tcp
    transfer_pos = transfer_tcp[:3, 3]
    transfer_pos[2] = current_tcp_pos[2]
    transfer_quat = current_tcp_quat
    if not _ik_reaches(env, transfer_pos, transfer_quat):
        raise RuntimeError("The measured post-lift grasp cannot reach above the configured bin")

    lengths = _object_pick_place_phase_lengths(cfg)[4:]
    positions = np.concatenate(
        [
            _linear_segment(current_tcp_pos, transfer_pos, lengths[0]),
            np.repeat(transfer_pos.reshape(1, 3), lengths[1], axis=0),
        ],
        axis=0,
    )
    quats = np.repeat(np.asarray(transfer_quat).reshape(1, 4), sum(lengths), axis=0)
    release_steps = lengths[1]
    open_steps = max(1, release_steps // 4)
    gripper = np.concatenate(
        [
            np.full(lengths[0], float(cfg.closed_gripper)),
            np.linspace(float(cfg.closed_gripper), float(cfg.open_gripper), open_steps),
            np.full(release_steps - open_steps, float(cfg.open_gripper)),
        ]
    )
    phases = np.concatenate(
        [np.repeat(name, length) for name, length in zip(("transfer", "release"), lengths)]
    )
    return _object_actions(positions, quats, gripper, phases)


def _object_pick_place_phase_lengths(config: FrankaObjectPickPlaceConfig) -> list[int]:
    return _phase_lengths(
        int(config.rollout_len),
        weights=(0.20, 0.10, 0.10, 0.15, 0.25, 0.20),
    )


def _object_actions(
    positions: np.ndarray,
    quats: np.ndarray,
    gripper: np.ndarray,
    phases: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "phase": str(phase),
            "right": {
                "ee_pos": np.asarray(pos, dtype=np.float32),
                "ee_quat_wxyz": np.asarray(quat, dtype=np.float32),
                "gripper": np.asarray([grip], dtype=np.float32),
            },
        }
        for pos, quat, grip, phase in zip(positions, quats, gripper, phases)
    ]


def franka_object_contact_flags(env: TabletopManipulationEnv) -> dict[str, bool]:
    """Return Robotiq pad contacts with the active task object."""

    object_body_id = int(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, env.scene.object_body_name))
    if object_body_id < 0:
        raise KeyError(f"Object body is missing: {env.scene.object_body_name}")
    object_ids = {
        geom_id
        for geom_id in range(int(env.model.ngeom))
        if int(env.model.geom_bodyid[geom_id]) == object_body_id
    }
    left_ids = _geom_ids(env.model, ("g_left_pad1", "g_left_pad2"))
    right_ids = _geom_ids(env.model, ("g_right_pad1", "g_right_pad2"))
    left = False
    right = False
    for idx in range(int(env.data.ncon)):
        contact = env.data.contact[idx]
        pair = {int(contact.geom1), int(contact.geom2)}
        if not pair.intersection(object_ids):
            continue
        left |= bool(pair.intersection(left_ids))
        right |= bool(pair.intersection(right_ids))
    return {"left_pad_contact": left, "right_pad_contact": right, "both_pad_contact": left and right}


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_wxyz(quat_wxyz)
    return R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def _pose7_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_wxyz_to_matrix(pose[3:7])
    transform[:3, 3] = pose[:3]
    return transform


def _matrix_from_pose(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    return _pose7_to_matrix(np.concatenate([np.asarray(pos).reshape(3), np.asarray(quat_wxyz).reshape(4)]))


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_matrix(np.asarray(matrix, dtype=np.float64).reshape(3, 3)).as_quat()
    return np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def _gripper_base_to_tcp_pose(world_from_gripper: np.ndarray, tcp_offset: float) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(world_from_gripper, dtype=np.float64).reshape(4, 4)
    pos = transform[:3, 3] + transform[:3, 2] * float(tcp_offset)
    return pos, _matrix_to_quat_wxyz(transform[:3, :3])


def _ik_reaches(env: TabletopManipulationEnv, pos: np.ndarray, quat: np.ndarray) -> bool:
    side = env.robot.side("right")
    arm_q = solve_arm_ik(env.model, env.data, side, pos, quat, config=env.ik_config)
    work = mujoco.MjData(env.model)
    work.qpos[:] = env.data.qpos
    work.qvel[:] = env.data.qvel
    work.ctrl[:] = env.data.ctrl
    work.qpos[side.arm_qpos_indices] = arm_q
    mujoco.mj_forward(env.model, work)
    reached_pos = np.asarray(work.site_xpos[side.wrist_site_id], dtype=np.float64)
    reached_rot = R.from_matrix(np.asarray(work.site_xmat[side.wrist_site_id]).reshape(3, 3))
    target_rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    return bool(
        np.linalg.norm(reached_pos - np.asarray(pos, dtype=np.float64)) <= 2.0 * env.ik_config.pos_tolerance
        and np.linalg.norm((target_rot * reached_rot.inv()).as_rotvec()) <= 2.0 * env.ik_config.ori_tolerance
    )


def _slerp_quat_wxyz(start_wxyz: np.ndarray, end_wxyz: np.ndarray, n_steps: int) -> np.ndarray:
    start = _normalize_quat_wxyz(start_wxyz)
    end = _normalize_quat_wxyz(end_wxyz)
    if int(n_steps) <= 1:
        return start.reshape(1, 4)
    if float(np.dot(start, end)) < 0.0:
        end = -end
    rotations = R.from_quat(
        [
            [start[1], start[2], start[3], start[0]],
            [end[1], end[2], end[3], end[0]],
        ]
    )
    interp = Slerp([0.0, 1.0], rotations)(np.linspace(0.0, 1.0, int(n_steps), dtype=np.float64))
    quat_xyzw = interp.as_quat()
    return np.column_stack([quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]]).astype(np.float64)


def _normalize_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        raise ValueError("quat_wxyz must be nonzero")
    return quat / norm


def _unit_axis(axis: np.ndarray | tuple[float, float, float]) -> np.ndarray:
    out = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(out))
    if norm <= 1e-8:
        raise ValueError("axis must be nonzero")
    return out / norm


def _orthogonal_unit_axis(
    axis: np.ndarray | tuple[float, float, float],
    *,
    reference: np.ndarray,
) -> np.ndarray:
    out = _unit_axis(axis)
    if abs(float(np.dot(out, _unit_axis(reference)))) > 1e-6:
        raise ValueError("axes must be orthogonal")
    return out


def _wrap_to_pi(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)


def _phase_lengths(total: int, *, weights: tuple[float, ...]) -> list[int]:
    raw = np.asarray(weights, dtype=np.float64)
    raw = raw / raw.sum()
    lengths = np.maximum(1, np.floor(raw * int(total)).astype(np.int32))
    while int(lengths.sum()) < int(total):
        lengths[int(np.argmax(raw * int(total) - lengths))] += 1
    while int(lengths.sum()) > int(total):
        idx = int(np.argmax(lengths))
        if lengths[idx] <= 1:
            break
        lengths[idx] -= 1
    return [int(x) for x in lengths]


def _even_tcp_speed_phase_lengths(
    total: int,
    *,
    waypoints: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    close_weight: float,
) -> list[int]:
    legacy_lengths = _phase_lengths(int(total), weights=(0.25, 0.20, 0.12, float(close_weight), 0.25))
    close_len = int(legacy_lengths[3])
    moving_total = max(4, int(total) - close_len)
    points = [np.asarray(value, dtype=np.float64).reshape(3) for value in waypoints]
    distances = tuple(float(np.linalg.norm(points[idx + 1] - points[idx])) for idx in range(len(points) - 1))
    weights = distances if float(np.sum(distances)) > 1.0e-12 else (1.0, 1.0, 1.0, 1.0)
    motion_lengths = _phase_lengths(moving_total, weights=weights)
    lengths = [motion_lengths[0], motion_lengths[1], motion_lengths[2], close_len, motion_lengths[3]]
    delta = int(total) - int(sum(lengths))
    lengths[-1] += delta
    return [max(1, int(value)) for value in lengths]


def _linear_segment(
    start: np.ndarray,
    end: np.ndarray,
    n_steps: int,
    *,
    include_start: bool = True,
) -> np.ndarray:
    if int(n_steps) <= 1:
        return np.asarray(end, dtype=np.float64).reshape(1, 3)
    start_alpha = 0.0 if bool(include_start) else 1.0 / float(n_steps)
    alpha = np.linspace(start_alpha, 1.0, int(n_steps), dtype=np.float64)
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    return (1.0 - alpha[:, None]) * start[None, :] + alpha[:, None] * end[None, :]


def _geom_ids(model: mujoco.MjModel, names: tuple[str, ...]) -> set[int]:
    ids: set[int] = set()
    for name in names:
        geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
        if geom_id < 0:
            raise KeyError(f"Required contact geom is missing: {name}")
        ids.add(geom_id)
    return ids
