from __future__ import annotations
from typing import Any
import numpy as np

def _translation_groups(robot_model: Any, local_dof: int) -> tuple[tuple[int, int, int], ...]:
    groups: list[tuple[int, int, int]] = []
    for group in getattr(robot_model, "pose_groups", []) or []:
        if not isinstance(group, dict):
            continue
        raw = group.get("translation_indices")
        if raw is None:
            continue
        indices = tuple(int(x) for x in raw)
        if len(indices) >= 3 and all(0 <= idx < int(local_dof) for idx in indices[:3]):
            groups.append(tuple(indices[:3]))
    if not groups:
        raw = list(getattr(robot_model, "base_translation_indices", []))
        if len(raw) >= 3:
            groups.append(tuple(int(x) for x in raw[:3]))
    if not groups and int(local_dof) >= 3:
        groups.append((0, 1, 2))
    if not groups:
        raise ValueError(f"Cannot infer translation indices for robot_model={robot_model!r}")
    return tuple(dict.fromkeys(groups))


def _shift_low_level_q(
    q: np.ndarray,
    shift: np.ndarray,
    *,
    translation_groups: tuple[tuple[int, int, int], ...],
    sign: float,
) -> np.ndarray:
    out = np.asarray(q, dtype=np.float32).copy()
    shift_np = np.asarray(shift, dtype=np.float32).reshape(3)
    if out.ndim == 1:
        out = out.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    for indices in translation_groups:
        out[:, list(indices)] += float(sign) * shift_np.reshape(1, 3)
    return out[0] if squeeze else out


