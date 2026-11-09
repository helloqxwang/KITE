"""Shared helpers for editable Wuji start poses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


EXPERIMENT_DIR = Path(__file__).resolve().parent
CROSS_DIR = EXPERIMENT_DIR.parent

WUJI_LOCAL_DOF = 26
WUJI_JOINT_DIM = 20
DEFAULT_WUJI_START_POSE_RELATIVE = "./experiment/data/high_level_policy/wuji_start_pose_12.pt"


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (CROSS_DIR / path).resolve()


def wuji_hand_joint_names(side: str = "right") -> tuple[str, ...]:
    side_s = str(side)
    return tuple(f"{side_s}_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5))


def wuji_start_pose_joint_names(side: str = "right") -> tuple[str, ...]:
    side_s = str(side)
    return (
        f"{side_s}_pos_x",
        f"{side_s}_pos_y",
        f"{side_s}_pos_z",
        f"{side_s}_rot_x",
        f"{side_s}_rot_y",
        f"{side_s}_rot_z",
        *wuji_hand_joint_names(side_s),
    )


def validate_wuji_q(q: Any, *, label: str = "q") -> np.ndarray:
    arr = _to_numpy(q).astype(np.float32, copy=False).reshape(-1)
    if arr.shape != (WUJI_LOCAL_DOF,):
        raise ValueError(f"{label} must have shape ({WUJI_LOCAL_DOF},), got {arr.shape}")
    if not bool(np.isfinite(arr).all()):
        raise ValueError(f"{label} contains non-finite values")
    return arr.astype(np.float32, copy=True)


def split_wuji_q(q: Any) -> dict[str, np.ndarray]:
    q_np = validate_wuji_q(q)
    base_pos = q_np[:3].astype(np.float32, copy=True)
    base_euler = q_np[3:6].astype(np.float32, copy=True)
    hand_q = q_np[6 : 6 + WUJI_JOINT_DIM].astype(np.float32, copy=True)
    return {
        "q": q_np,
        "base_pos": base_pos,
        "base_euler_xyz": base_euler,
        "base_quat_wxyz": euler_xyz_to_quat_wxyz(base_euler),
        "hand_q": hand_q,
    }


def load_wuji_start_pose(path_value: str | Path) -> dict[str, Any]:
    path = resolve_project_path(path_value)
    if not path.exists():
        raise FileNotFoundError(
            f"Wuji start pose file does not exist: {path}. "
            "Install the released start poses with python download_assets.py, or select initial_pose_mode=retargeted."
        )
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and "q" in raw:
        q = validate_wuji_q(raw["q"], label=f"{path}:q")
        meta = dict(raw.get("meta", {})) if isinstance(raw.get("meta", {}), dict) else {}
        schema = str(raw.get("schema", "raw_q"))
    else:
        q = validate_wuji_q(raw, label=str(path))
        meta = {}
        schema = "raw_q"
    parts = split_wuji_q(q)
    return {
        "schema": schema,
        "path": str(path),
        "q": parts["q"],
        "base_pos": parts["base_pos"],
        "base_euler_xyz": parts["base_euler_xyz"],
        "base_quat_wxyz": parts["base_quat_wxyz"],
        "hand_q": parts["hand_q"],
        "joint_names": tuple(wuji_start_pose_joint_names("right")),
        "meta": meta,
    }


def load_wuji_start_q(path_value: str | Path) -> np.ndarray:
    return np.asarray(load_wuji_start_pose(path_value)["q"], dtype=np.float32).copy()


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
