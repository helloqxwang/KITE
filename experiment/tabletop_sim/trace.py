"""Trace helpers shared by tabletop rollout scripts."""

from __future__ import annotations
from pathlib import Path
import mujoco
import numpy as np
from .env import TabletopManipulationEnv
from .perception import append_camera_observation


def new_trace_buffers() -> dict[str, list[np.ndarray]]:
    return {
        "cube_pose_before": [],
        "cube_pose_after": [],
        "object_pose_before": [],
        "object_pose_after": [],
        "pre_qpos": [],
        "target_qpos": [],
        "executed_qpos": [],
        "executed_qvel": [],
        "pre_ctrl": [],
        "target_ctrl": [],
        "executed_ctrl": [],
        "active_group_mask": [],
        "action_ee_pos": [],
        "action_ee_quat_wxyz": [],
        "action_effector_ctrl": [],
        "lift_delta": [],
        "lift_success": [],
    }


def append_trace_frame(
    trace: dict[str, list[np.ndarray]],
    env: TabletopManipulationEnv,
    info: dict[str, object],
    obs: dict[str, object] | None = None,
) -> None:
    command = info["command"]
    groups = command["sides"]
    if "cube_pose_before" in command and "cube_pose_after" in command:
        trace["cube_pose_before"].append(
            np.asarray(command["cube_pose_before"], dtype=np.float32).copy()
        )
        trace["cube_pose_after"].append(
            np.asarray(command["cube_pose_after"], dtype=np.float32).copy()
        )
    if "object_pose_before" in command and "object_pose_after" in command:
        trace["object_pose_before"].append(
            np.asarray(command["object_pose_before"], dtype=np.float32).copy()
        )
        trace["object_pose_after"].append(
            np.asarray(command["object_pose_after"], dtype=np.float32).copy()
        )
    trace["pre_qpos"].append(np.asarray(command["pre_qpos"], dtype=np.float32).copy())
    trace["target_qpos"].append(
        np.asarray(command["target_qpos"], dtype=np.float32).copy()
    )
    trace["executed_qpos"].append(
        np.asarray(command["executed_qpos"], dtype=np.float32).copy()
    )
    trace["executed_qvel"].append(
        np.asarray(command["executed_qvel"], dtype=np.float32).copy()
    )
    trace["pre_ctrl"].append(np.asarray(command["pre_ctrl"], dtype=np.float32).copy())
    trace["target_ctrl"].append(
        np.asarray(command["target_ctrl"], dtype=np.float32).copy()
    )
    trace["executed_ctrl"].append(
        np.asarray(command["executed_ctrl"], dtype=np.float32).copy()
    )
    group_names = env.robot.group_names
    max_effector_ctrl_dim = max(
        (env.robot.side(name).effector_ctrl_dim for name in group_names), default=0
    )
    active_mask = np.zeros((len(group_names),), dtype=np.bool_)
    action_ee_pos = np.zeros((len(group_names), 3), dtype=np.float32)
    action_ee_quat = np.zeros((len(group_names), 4), dtype=np.float32)
    action_effector_ctrl = np.zeros(
        (len(group_names), max_effector_ctrl_dim), dtype=np.float32
    )
    for group_idx, group_name in enumerate(group_names):
        if group_name not in groups:
            continue
        active_mask[group_idx] = True
        group_cmd = groups[group_name]
        action_ee_pos[group_idx] = np.asarray(
            group_cmd["ee_pos_target"], dtype=np.float32
        )
        action_ee_quat[group_idx] = np.asarray(
            group_cmd["ee_quat_wxyz_target"], dtype=np.float32
        )
        effector_ctrl = np.asarray(
            group_cmd.get("effector_ctrl_target", []), dtype=np.float32
        ).reshape(-1)
        action_effector_ctrl[group_idx, : effector_ctrl.shape[0]] = effector_ctrl
    trace["active_group_mask"].append(active_mask)
    trace["action_ee_pos"].append(action_ee_pos)
    trace["action_ee_quat_wxyz"].append(action_ee_quat)
    trace["action_effector_ctrl"].append(action_effector_ctrl)
    if obs is not None:
        for key in ("lift_delta", "lift_success"):
            if key in obs:
                trace[key].append(np.asarray(obs[key]).copy())
        append_camera_observation(trace, obs)


def save_trace_npz(
    out: str | Path,
    *,
    env: TabletopManipulationEnv,
    trace: dict[str, list[np.ndarray]],
    backend: str,
    mode: str,
    dt: float,
    extra: dict[str, object] | None = None,
    include_camera_payload: bool = True,
) -> Path:
    out_path = Path(out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": np.asarray(3, dtype=np.int32),
        "embodiment": np.asarray(env.robot.spec.key),
        "mode": np.asarray(mode),
        "backend": np.asarray(backend),
        "dt": np.asarray(dt, dtype=np.float32),
        "frame_skip": np.asarray(env.backend.frame_skip, dtype=np.int32),
        "gravity_compensation": np.asarray(env.gravity_compensation),
        "robot_xml_path": np.asarray(str(env.scene.robot_xml_path)),
        "object_kind": np.asarray(str(env.scene.object_kind)),
        "group_names": np.asarray(env.robot.group_names),
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
        "pointcloud_workspace_bounds": np.asarray(
            env.pointcloud_workspace_bounds, dtype=np.float32
        ),
        "qpos_names": np.asarray(qpos_names(env.model)),
        "ctrl_names": np.asarray(ctrl_names(env.model)),
    }
    for key, frames in trace.items():
        if key.startswith("_"):
            continue
        if not include_camera_payload and key.startswith("camera_"):
            continue
        payload[key] = np.asarray(frames)
    if extra:
        for key, value in extra.items():
            payload[key] = np.asarray(value)
    np.savez(out_path, **payload)
    return out_path


def qpos_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    suffix_by_dim = {
        1: ("",),
        4: ("qw", "qx", "qy", "qz"),
        7: ("x", "y", "z", "qw", "qx", "qy", "qz"),
    }
    for joint_id in range(int(model.njnt)):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            or f"joint_{joint_id}"
        )
        qadr = int(model.jnt_qposadr[joint_id])
        if joint_id + 1 < int(model.njnt):
            dim = int(model.jnt_qposadr[joint_id + 1]) - qadr
        else:
            dim = int(model.nq) - qadr
        suffixes = suffix_by_dim.get(dim, tuple((str(i) for i in range(dim))))
        names.extend(
            [name if suffix == "" else f"{name}/{suffix}" for suffix in suffixes]
        )
    return names


def ctrl_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for actuator_id in range(int(model.nu)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        names.append(name or f"actuator_{actuator_id}")
    return names
