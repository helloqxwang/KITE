"""Episode, start-configuration and coordinate helpers shared by Oracle Intent
rollouts and interaction policy evaluation."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import hydra
import mujoco
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from scipy.spatial.transform import Rotation as R

EXPERIMENT_DIR = Path(__file__).resolve().parent
CROSS_DIR = EXPERIMENT_DIR.parent
if str(CROSS_DIR) not in sys.path:
    sys.path.insert(0, str(CROSS_DIR))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from experiment.high_level_policy.payload import (  # noqa: E402
    build_common_meta,
    build_training_payload,
    concat_single_manipulator_action,
    load_high_level_payload,
    make_episode_ends,
    previous_action_proprio,
    save_high_level_payload,
    single_manipulator_action_space,
    tensor_to_numpy,
)
from experiment.pipeline_high_low_policy import (  # noqa: E402
    _load_cotrain_low_level_policy,
    _peek_cotrain_meta,
    _resolve_low_level_backend,
    _rng_devices,
    load_root_hydra_config,
)
from experiment.tabletop_sim.env import TabletopManipulationEnv  # noqa: E402
from experiment.tabletop_sim.factory import build_env_from_config  # noqa: E402
from experiment.tabletop_sim.rotation import unwrap_robot_model_q_rotation  # noqa: E402
from experiment.tabletop_sim.trace import ctrl_names, qpos_names  # noqa: E402
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
from high_level_policy.train_utils import prepare_device  # noqa: E402
from robot_model.robot_model import create_robot_model  # noqa: E402
from contact_policy_probe import load_contact_policy_probe_checkpoint  # noqa: E402


WUJI_LOCAL_DOF = 26
WUJI_JOINT_DIM = 20
SIDE_NAMES = ("left", "right")
ACTION_FORMAT_WUJI = "wuji_single_manipulator"
ACTION_FORMAT_ROBOT_MODEL_Q = "robot_model_q"


@dataclass(frozen=True)
class EpisodeSpec:
    contact_episode_index: int
    retarget_episode_index: int
    source_trajectory_index: int
    source_trajectory_seed: int
    source_attempt_index: int
    frame_count: int
    contact_start: int
    retarget_start: int


@dataclass
class LowLevelRuntime:
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
    low_level_manipulator: str
    hand_fps_num_points: int


def _payload_arrays(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    return {str(key): tensor_to_numpy(value) for key, value in dict(payload["data"]).items()}


def _episode_ranges(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for end_raw in np.asarray(episode_ends, dtype=np.int64).reshape(-1):
        end = int(end_raw)
        if end <= start:
            raise ValueError("episode_ends must be strictly increasing")
        ranges.append((start, end))
        start = end
    return ranges


def _dataset_xyz_translation(payload: dict[str, Any]) -> np.ndarray:
    meta = dict(payload.get("meta", {}))
    shift = np.asarray(meta.get("dataset_xyz_translation", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
    if shift.shape != (3,):
        raise ValueError(f"meta.dataset_xyz_translation must have shape (3,), got {shift.shape}")
    return shift.astype(np.float32, copy=False)


def _episode_source_index(payload: dict[str, Any], episode_idx: int) -> int:
    item = dict(list(payload["episode_meta"])[int(episode_idx)])
    for key in ("source_trajectory_index", "trajectory_index", "episode_index"):
        if item.get(key) is not None:
            return int(item[key])
    return int(episode_idx)


def _episode_meta_int(payload: dict[str, Any], episode_idx: int, keys: tuple[str, ...], default: int) -> int:
    item = dict(list(payload["episode_meta"])[int(episode_idx)])
    for key in keys:
        if item.get(key) is not None:
            return int(item[key])
    return int(default)


def _retarget_episode_map(payload: dict[str, Any], episode_count: int) -> dict[int, int]:
    out: dict[int, int] = {}
    for episode_idx in range(int(episode_count)):
        source_idx = _episode_source_index(payload, episode_idx)
        out.setdefault(int(source_idx), int(episode_idx))
    return out


def _build_episode_specs(
    selected: list[int],
    *,
    contact_payload: dict[str, Any],
    retarget_payload: dict[str, Any],
    contact_ranges: list[tuple[int, int]],
    retarget_ranges: list[tuple[int, int]],
    retarget_map: dict[int, int],
    max_frames: int | None,
) -> list[EpisodeSpec]:
    specs: list[EpisodeSpec] = []
    for contact_ep in selected:
        source_idx = _episode_source_index(contact_payload, contact_ep)
        if source_idx not in retarget_map:
            raise KeyError(f"No retargeted Wuji episode found for source_trajectory_index={source_idx}")
        retarget_ep = int(retarget_map[source_idx])
        c_start, c_end = contact_ranges[int(contact_ep)]
        r_start, r_end = retarget_ranges[int(retarget_ep)]
        frame_count = min(int(c_end - c_start), int(r_end - r_start))
        if max_frames is not None and int(max_frames) > 0:
            frame_count = min(frame_count, int(max_frames))
        if frame_count <= 0:
            raise RuntimeError(f"Episode {contact_ep} has no frames")
        specs.append(
            EpisodeSpec(
                contact_episode_index=int(contact_ep),
                retarget_episode_index=int(retarget_ep),
                source_trajectory_index=int(source_idx),
                source_trajectory_seed=_episode_meta_int(
                    retarget_payload,
                    retarget_ep,
                    ("source_trajectory_seed", "trajectory_seed"),
                    default=-1,
                ),
                source_attempt_index=_episode_meta_int(
                    retarget_payload,
                    retarget_ep,
                    ("source_attempt_index", "attempt_index"),
                    default=-1,
                ),
                frame_count=int(frame_count),
                contact_start=int(c_start),
                retarget_start=int(r_start),
            )
        )
    return specs


def _load_robot_model_start_q(
    path: Path,
    *,
    runtime: LowLevelRuntime,
    expected_robot_name: str,
) -> np.ndarray:
    payload = torch.load(str(path.expanduser().resolve()), map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        if "q" in payload:
            q_raw = payload["q"]
        elif "robot_model_q" in payload:
            q_raw = payload["robot_model_q"]
        else:
            raise KeyError(f"robot_model start pose {path} must contain key 'q' or 'robot_model_q'")
        robot_name = payload.get("robot_name")
        if robot_name is not None and str(robot_name).strip() != str(expected_robot_name):
            print(
                f"WARNING: saved start pose {path} robot_name={robot_name!r} does not match "
                f"expected {expected_robot_name!r}; using q because dimensions will be validated.",
                flush=True,
            )
        saved_joint_names = payload.get("joint_names")
        if saved_joint_names is not None and hasattr(runtime.robot_model, "get_joint_orders"):
            saved = [str(name) for name in list(saved_joint_names)]
            current = [str(name) for name in runtime.robot_model.get_joint_orders()]
            if saved and current and saved != current:
                print(
                    f"WARNING: saved start pose {path} joint_names do not match current "
                    f"{runtime.low_level_manipulator} joint order; using q as-is. "
                    f"saved starts {saved[:8]}, current starts {current[:8]}",
                    flush=True,
                )
    else:
        q_raw = payload
    q = tensor_to_numpy(q_raw).astype(np.float32, copy=False)
    return _validate_robot_model_q(
        q,
        expected_dim=int(runtime.local_dof),
        label=f"saved robot_model start pose {path}",
    )


def _initial_wuji_q_from_retargeted(retarget: dict[str, np.ndarray], spec: EpisodeSpec) -> np.ndarray:
    idx = int(spec.retarget_start)
    pos = np.asarray(retarget["action_ee_pos"][idx], dtype=np.float32).reshape(3)
    quat = np.asarray(retarget["action_ee_quat_wxyz"][idx], dtype=np.float32).reshape(4)
    euler = _quat_wxyz_to_euler_xyz(quat)
    hand_q = np.asarray(retarget["action_hand_q"][idx], dtype=np.float32).reshape(-1)[:WUJI_JOINT_DIM]
    if hand_q.shape != (WUJI_JOINT_DIM,):
        raise ValueError(f"Expected retargeted action_hand_q dim {WUJI_JOINT_DIM}, got {hand_q.shape}")
    return np.concatenate([pos, euler, hand_q], axis=0).astype(np.float32, copy=False)


def _validate_robot_model_q(q: np.ndarray, *, expected_dim: int, label: str) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float32).reshape(-1)
    if arr.shape != (int(expected_dim),):
        raise ValueError(f"{label} must have shape ({int(expected_dim)},), got {arr.shape}")
    if not bool(np.isfinite(arr).all()):
        raise ValueError(f"{label} contains non-finite values")
    return arr.astype(np.float32, copy=True)


def _quat_wxyz_to_euler_xyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    quat = quat / np.clip(np.linalg.norm(quat), 1.0e-8, None)
    # Wuji low-level RobotModel q uses SciPy's "XYZ" Euler convention.
    return R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("XYZ").astype(np.float32)


def _euler_xyz_to_quat_wxyz(euler_xyz: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_euler("XYZ", np.asarray(euler_xyz, dtype=np.float64).reshape(3)).as_quat()
    quat_wxyz = np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    return (quat_wxyz / np.clip(np.linalg.norm(quat_wxyz), 1.0e-8, None)).astype(np.float32)


def _shift_low_level_q(q: np.ndarray, shift: np.ndarray, *, sign: float) -> np.ndarray:
    out = np.asarray(q, dtype=np.float32).copy()
    out[:, :3] += float(sign) * np.asarray(shift, dtype=np.float32).reshape(1, 3)
    return out


def _apply_fixed_zero_constraints(runtime: LowLevelRuntime, q: np.ndarray) -> np.ndarray:
    q_t = torch.from_numpy(np.asarray(q, dtype=np.float32)).to(runtime.robot_model.device, dtype=torch.float32)
    with torch.no_grad():
        q_out = runtime.robot_model._apply_fixed_zero_joint_constraints(q_t)  # noqa: SLF001
    return q_out.detach().cpu().numpy().astype(np.float32, copy=False)


def _episode_value(data: dict[str, np.ndarray], key: str, episode_idx: int, *, default: Any) -> Any:
    if key not in data:
        return default
    arr = np.asarray(data[key])
    if arr.ndim >= 1 and int(arr.shape[0]) > int(episode_idx):
        value = arr[int(episode_idx)]
        if np.asarray(value).size == 1:
            return np.asarray(value).reshape(-1)[0].item()
        return value
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return default


