"""Geometry checks for object-only point clouds stored in a fixed world frame."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


DEFAULT_OBJECT_BOUNDS_TOLERANCE_M = 0.005


def validate_object_only_world_points(
    points_world: Any,
    object_pose_wxyz: Any,
    object_bounds_m: Any,
    *,
    occluded_frames: Any | None = None,
    tolerance_m: float = DEFAULT_OBJECT_BOUNDS_TOLERANCE_M,
    label: str = "object point cloud",
) -> dict[str, float | int]:
    """Require every visible point to lie inside the posed object's local bounds.

    The check inverts the saved object pose; it never modifies or recenters the
    point cloud returned to the caller.  A small bound expansion accommodates
    rasterization and float16 storage error.
    """

    points = np.asarray(points_world, dtype=np.float32)
    poses = np.asarray(object_pose_wxyz, dtype=np.float32)
    if points.ndim == 2:
        points = points[None, ...]
    if poses.ndim == 1:
        poses = poses[None, ...]
    if points.ndim != 3 or int(points.shape[-1]) != 3:
        raise ValueError(f"{label} must have shape (T, N, 3), got {points.shape}")
    if poses.shape != (int(points.shape[0]), 7):
        raise ValueError(
            f"{label} object poses must have shape {(int(points.shape[0]), 7)}, got {poses.shape}"
        )
    bounds = np.asarray(object_bounds_m, dtype=np.float32)
    if bounds.shape != (2, 3) or bool(np.any(bounds[1] < bounds[0])):
        raise ValueError(f"{label} object_bounds_m must have ordered shape (2, 3), got {bounds}")
    tolerance = float(tolerance_m)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"{label} tolerance_m must be finite and nonnegative, got {tolerance_m}")

    if occluded_frames is None:
        visible = np.ones((int(points.shape[0]),), dtype=np.bool_)
    else:
        occluded = np.asarray(occluded_frames, dtype=np.bool_).reshape(-1)
        if occluded.shape != (int(points.shape[0]),):
            raise ValueError(
                f"{label} occluded_frames must have shape {(int(points.shape[0]),)}, got {occluded.shape}"
            )
        visible = ~occluded
    if not bool(np.any(visible)):
        return {"visible_frames": 0, "point_count": 0, "outside_count": 0, "outside_fraction": 0.0}

    visible_points = points[visible]
    visible_poses = poses[visible]
    if not bool(np.isfinite(visible_points).all()) or not bool(np.isfinite(visible_poses).all()):
        raise ValueError(f"{label} contains non-finite visible points or object poses")
    quat_wxyz = visible_poses[:, 3:7]
    quat_xyzw = np.concatenate([quat_wxyz[:, 1:4], quat_wxyz[:, 0:1]], axis=1)
    rotation_world_from_local = R.from_quat(quat_xyzw).as_matrix().astype(np.float32)
    translated = visible_points - visible_poses[:, None, :3]
    points_local = np.einsum(
        "tni,tij->tnj",
        translated,
        rotation_world_from_local,
        optimize=True,
    )
    lower = bounds[0] - tolerance
    upper = bounds[1] + tolerance
    outside = np.any((points_local < lower) | (points_local > upper), axis=-1)
    outside_count = int(np.count_nonzero(outside))
    point_count = int(outside.size)
    outside_fraction = float(outside_count / point_count) if point_count else 0.0
    if outside_count:
        raise ValueError(
            f"{label} violates object-only world-frame geometry: {outside_count}/{point_count} "
            f"points ({outside_fraction:.3%}) fall outside object_bounds_m expanded by {tolerance:.4f} m"
        )
    return {
        "visible_frames": int(np.count_nonzero(visible)),
        "point_count": point_count,
        "outside_count": outside_count,
        "outside_fraction": outside_fraction,
    }
