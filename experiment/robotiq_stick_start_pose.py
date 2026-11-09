"""Shared helpers for editable robotiq_gripper_stick start poses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


EXPERIMENT_DIR = Path(__file__).resolve().parent
CROSS_DIR = EXPERIMENT_DIR.parent

ROBOTIQ_STICK_LOCAL_DOF = 13
ROBOTIQ_STICK_ROBOT_NAME = "robotiq_gripper_stick"
DEFAULT_ROBOTIQ_STICK_START_POSE_RELATIVE = (
    "./experiment/data/high_level_policy/robotiq_gripper_stick_start_pose.pt"
)


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (CROSS_DIR / path).resolve()


def robotiq_stick_start_pose_joint_names() -> tuple[str, ...]:
    return (
        "gripper_x",
        "gripper_y",
        "gripper_z",
        "gripper_roll",
        "gripper_pitch",
        "gripper_yaw",
        "gripper_cmd",
        "stick_x",
        "stick_y",
        "stick_z",
        "stick_roll",
        "stick_pitch",
        "stick_yaw",
    )


def validate_robotiq_stick_q(q: Any, *, label: str = "q") -> np.ndarray:
    arr = _to_numpy(q).astype(np.float32, copy=False).reshape(-1)
    if arr.shape != (ROBOTIQ_STICK_LOCAL_DOF,):
        raise ValueError(f"{label} must have shape ({ROBOTIQ_STICK_LOCAL_DOF},), got {arr.shape}")
    if not bool(np.isfinite(arr).all()):
        raise ValueError(f"{label} contains non-finite values")
    if float(arr[6]) < 0.0 or float(arr[6]) > 255.0:
        raise ValueError(f"{label}[6] gripper_cmd must be in [0, 255], got {float(arr[6])}")
    return arr.astype(np.float32, copy=True)


def split_robotiq_stick_q(q: Any) -> dict[str, np.ndarray]:
    q_np = validate_robotiq_stick_q(q)
    gripper_euler = q_np[3:6].astype(np.float32, copy=True)
    stick_euler = q_np[10:13].astype(np.float32, copy=True)
    return {
        "q": q_np,
        "gripper_pos": q_np[0:3].astype(np.float32, copy=True),
        "gripper_euler_xyz": gripper_euler,
        "gripper_quat_wxyz": euler_xyz_to_quat_wxyz(gripper_euler),
        "gripper_cmd": q_np[6:7].astype(np.float32, copy=True),
        "stick_pos": q_np[7:10].astype(np.float32, copy=True),
        "stick_euler_xyz": stick_euler,
        "stick_quat_wxyz": euler_xyz_to_quat_wxyz(stick_euler),
    }


def load_robotiq_stick_start_pose(path_value: str | Path) -> dict[str, Any]:
    path = resolve_project_path(path_value)
    if not path.exists():
        raise FileNotFoundError(
            f"Robotiq+stick start pose file does not exist: {path}. "
            "Install the released start poses with python download_assets.py."
        )
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "q" in raw:
        q = validate_robotiq_stick_q(raw["q"], label=f"{path}:q")
        meta = dict(raw.get("meta", {})) if isinstance(raw.get("meta", {}), dict) else {}
        schema = str(raw.get("schema", "raw_q"))
    else:
        q = validate_robotiq_stick_q(raw, label=str(path))
        meta = {}
        schema = "raw_q"
    parts = split_robotiq_stick_q(q)
    return {
        "schema": schema,
        "path": str(path),
        "q": parts["q"],
        "gripper_pos": parts["gripper_pos"],
        "gripper_euler_xyz": parts["gripper_euler_xyz"],
        "gripper_quat_wxyz": parts["gripper_quat_wxyz"],
        "gripper_cmd": parts["gripper_cmd"],
        "stick_pos": parts["stick_pos"],
        "stick_euler_xyz": parts["stick_euler_xyz"],
        "stick_quat_wxyz": parts["stick_quat_wxyz"],
        "joint_names": tuple(robotiq_stick_start_pose_joint_names()),
        "meta": meta,
    }


def load_robotiq_stick_start_q(path_value: str | Path) -> np.ndarray:
    return np.asarray(load_robotiq_stick_start_pose(path_value)["q"], dtype=np.float32).copy()


def euler_xyz_to_quat_wxyz(euler_xyz: Any) -> np.ndarray:
    quat_xyzw = R.from_euler("XYZ", _to_numpy(euler_xyz).astype(np.float64, copy=False).reshape(3)).as_quat()
    quat_wxyz = np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    return _normalize_quat_wxyz(quat_wxyz).astype(np.float32, copy=False)


def _normalize_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1.0e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
