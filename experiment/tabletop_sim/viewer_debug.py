"""MuJoCo viewer debug helpers."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


COLLISION_DEBUG_RGBA = np.asarray([0.05, 0.85, 1.0, 0.45], dtype=np.float32)
COLLISION_GEOM_MODES = {"none", "overlay", "collision_only"}


def configure_collision_geom_debug_view(
    model: mujoco.MjModel,
    viewer: Any | None = None,
    *,
    mode: str = "overlay",
) -> None:
    """Configure MuJoCo viewer collision visualization.

    Modes:
    - ``none``: leave model visibility unchanged.
    - ``overlay``: show hidden collision geoms over normal visual geoms.
    - ``collision_only``: hide non-collision geoms and show active collision geoms.
    """

    mode = str(mode).strip().lower().replace("-", "_")
    if mode == "only":
        mode = "collision_only"
    if mode not in COLLISION_GEOM_MODES:
        raise ValueError(f"Unknown collision geom viewer mode {mode!r}; expected one of {sorted(COLLISION_GEOM_MODES)}")
    if mode == "none":
        return

    visible_groups: set[int] = set()
    for geom_id in range(int(model.ngeom)):
        contype = int(model.geom_contype[geom_id])
        conaffinity = int(model.geom_conaffinity[geom_id])
        active_collision = contype != 0 or conaffinity != 0
        if not active_collision:
            if mode == "collision_only":
                model.geom_rgba[geom_id, 3] = 0.0
            continue
        group = int(model.geom_group[geom_id])
        alpha = float(model.geom_rgba[geom_id, 3])
        if mode == "collision_only" or alpha <= 1.0e-4 or group >= 2:
            model.geom_rgba[geom_id] = COLLISION_DEBUG_RGBA
            visible_groups.add(group)

    if viewer is None or not hasattr(viewer, "opt"):
        return
    geomgroup = getattr(viewer.opt, "geomgroup", None)
    if geomgroup is None:
        return
    for group in visible_groups:
        if 0 <= group < len(geomgroup):
            geomgroup[group] = 1
