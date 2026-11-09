"""Fixed saved reset poses selected by scene object ID."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

StartPose = tuple[str, Path, np.ndarray]


def load_object_start_poses(
    cfg: DictConfig,
    *,
    section: str,
    loader: Callable[[DictConfig], tuple[str, Path | None, np.ndarray | None]],
) -> dict[str, StartPose]:
    """Load explicit object overrides through the entrypoint's normal pose loader."""
    mapping = OmegaConf.select(cfg, f"{section}.initial_pose.pose_by_object", default=None)
    if mapping is None:
        return {}
    if not OmegaConf.is_dict(mapping):
        raise ValueError(f"{section}.initial_pose.pose_by_object must map object IDs to saved pose paths")
    if not mapping:
        return {}
    if str(OmegaConf.select(cfg, f"{section}.initial_pose.mode", default="saved")) != "saved":
        raise ValueError(f"{section}.initial_pose.pose_by_object requires mode=saved")
    result: dict[str, StartPose] = {}
    for object_id, path in mapping.items():
        if not isinstance(object_id, str) or not object_id.strip():
            raise ValueError("Initial-pose object IDs must be nonempty strings")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"Initial pose for object {object_id!r} must be a saved pose path")
        scoped = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        for key, value in (
            ("pose_pt", path),
            ("prefer_task_start_pose", False),
            ("require_task_start_pose", False),
        ):
            OmegaConf.update(scoped, f"{section}.initial_pose.{key}", value, force_add=True)
        mode, resolved_path, q = loader(scoped)
        if mode != "saved" or resolved_path is None or q is None:
            raise ValueError(f"Object {object_id!r} did not resolve to a saved initial configuration")
        result[object_id] = (mode, resolved_path, np.asarray(q, dtype=np.float32).copy())
    return result


def add_episode_start_pose_metadata(
    payload: dict[str, Any], poses: Sequence[StartPose]
) -> None:
    """Record actual resets per episode and label the former scalar fields as defaults."""
    if not poses:
        raise ValueError("Per-object reset metadata requires at least one episode")
    payload["initial_pose_mode"] = np.asarray("saved_per_object")
    payload["default_initial_pose_path"] = payload.get("initial_pose_path", np.asarray(""))
    payload["initial_pose_path"] = np.asarray("")
    for key in (
        "initial_robot_model_q_world",
        "initial_robotiq_stick_q_world",
        "initial_wuji_q_world",
    ):
        if key in payload:
            payload[f"default_{key}"] = payload.pop(key)
    payload["episode_initial_pose_paths"] = np.asarray([str(path) for _, path, _ in poses])
    payload["episode_initial_q_world"] = np.stack([q for _, _, q in poses]).astype(np.float32)
