"""Shared helpers for editable robot-model start poses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
CROSS_DIR = EXPERIMENT_DIR.parent

DEFAULT_BARRETT_START_POSE_RELATIVE = "./experiment/data/high_level_policy/barrett_start_pose.pt"
DEFAULT_ALLEGRO_RIGHT_START_POSE_RELATIVE = "./experiment/data/high_level_policy/allegro_right_start_pose.pt"
LEGACY_ALLEGRO_ORIGINAL_RIGHT_START_POSE_RELATIVE = (
    "./experiment/data/high_level_policy/allegro_original_right_start_pose.pt"
)


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (CROSS_DIR / path).resolve()


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
