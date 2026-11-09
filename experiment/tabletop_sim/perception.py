"""RGBD camera capture and point-cloud helpers for tabletop MuJoCo scenes."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from .assets import CAMERA_HEIGHT, CAMERA_WIDTH, TABLE_CENTER, TABLE_HALF_EXTENTS, TABLE_TOP_Z


EXTERNAL_CAMERA_NAMES = ("ext_cam_0", "ext_cam_1")
HAND_CAMERA_NAME = "hand_cam"
DEFAULT_CAMERA_NAMES = (*EXTERNAL_CAMERA_NAMES, HAND_CAMERA_NAME)
DEFAULT_POLICY_CAMERA_NAMES = EXTERNAL_CAMERA_NAMES
BASE_CAMERA_WIDTH = 1280
BASE_CAMERA_HEIGHT = 720
BASE_EXTERNAL_FOCAL = 635.0

# OpenCV camera coordinates use +X right, +Y down, +Z forward. MuJoCo cameras use
# +X right, +Y up, -Z forward.
_CV_TO_MUJOCO_CAMERA = np.diag([1.0, -1.0, -1.0]).astype(np.float64)

DEFAULT_POINTCLOUD_WORKSPACE_BOUNDS = np.asarray(
    [
        [TABLE_CENTER[0] - TABLE_HALF_EXTENTS[0] - 0.05, TABLE_CENTER[0] + TABLE_HALF_EXTENTS[0] + 0.05],
        [TABLE_CENTER[1] - TABLE_HALF_EXTENTS[1] - 0.05, TABLE_CENTER[1] + TABLE_HALF_EXTENTS[1] + 0.05],
        [TABLE_TOP_Z - 0.03, TABLE_TOP_Z + 0.50],
    ],
    dtype=np.float32,
)

CAMERA_FRAME_KEYS = (
    "rgb",
    "depth",
    "segmentation",
    "robot_mask",
    "object_mask",
    "cube_mask",
    "table_mask",
    "intrinsic_cv",
    "extrinsic_cv",
    "camera_pose_world",
    "camera_pose_world_wxyz",
)


@dataclass(frozen=True)
class CameraSpec:
    """Static camera calibration and scene placement metadata."""

    name: str
    width: int = CAMERA_WIDTH
    height: int = CAMERA_HEIGHT
    fovy: float = 60.0
    intrinsic_cv: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None = None
    pos: tuple[float, float, float] | None = None
    lookat: tuple[float, float, float] | None = None
    quat_wxyz: tuple[float, float, float, float] | None = None
    parent_body_candidates: tuple[str, ...] = ()

    def intrinsic_matrix(self) -> np.ndarray:
        if self.intrinsic_cv is not None:
            return np.asarray(self.intrinsic_cv, dtype=np.float64).reshape(3, 3)
        return intrinsic_from_fovy(self.width, self.height, self.fovy)


def default_camera_specs(
    *,
    table_top_z: float = TABLE_TOP_Z,
    width: int = CAMERA_WIDTH,
    height: int = CAMERA_HEIGHT,
) -> dict[str, CameraSpec]:
    """Return the default tabletop camera suite."""

    table_top_z = float(table_top_z)
    width = int(width)
    height = int(height)
    lookat = (float(TABLE_CENTER[0]) + 0.02, -0.12, table_top_z + 0.08)

    # External camera intrinsics, scaled from the 1280x720 calibration.
    ext_k = (
        (BASE_EXTERNAL_FOCAL * width / BASE_CAMERA_WIDTH, 0.0, (width - 1.0) * 0.5),
        (0.0, BASE_EXTERNAL_FOCAL * height / BASE_CAMERA_HEIGHT, (height - 1.0) * 0.5),
        (0.0, 0.0, 1.0),
    )
    hand_k = tuple(tuple(float(x) for x in row) for row in intrinsic_from_fovy(width, height, 52.0))
    ext_fovy = fovy_from_intrinsic(np.asarray(ext_k, dtype=np.float64), height)

    return {
        "ext_cam_0": CameraSpec(
            name="ext_cam_0",
            width=width,
            height=height,
            fovy=ext_fovy,
            intrinsic_cv=ext_k,
            pos=(0.78, -0.62, table_top_z + 0.52),
            lookat=lookat,
        ),
        "ext_cam_1": CameraSpec(
            name="ext_cam_1",
            width=width,
            height=height,
            fovy=ext_fovy,
            intrinsic_cv=ext_k,
            pos=(0.78, 0.52, table_top_z + 0.52),
            lookat=lookat,
        ),
        "hand_cam": CameraSpec(
            name="hand_cam",
            width=width,
            height=height,
            fovy=52.0,
            intrinsic_cv=hand_k,
            pos=(0.031, 0.074, -0.011),
            quat_wxyz=(0.0025, 0.0085, 0.9873, 0.1585),
            parent_body_candidates=(
                "g_base",
                "fr3_flange",
                "right_wuji_palm",
                "right_palm",
                "Link7_R",
                "base_link",
                "bh_base_link",
            ),
        ),
    }


def intrinsic_from_fovy(width: int, height: int, fovy_deg: float) -> np.ndarray:
    """Compute an OpenCV intrinsic matrix from MuJoCo vertical fovy."""

    fy = float(height) / (2.0 * math.tan(math.radians(float(fovy_deg)) * 0.5))
    fx = fy
    return np.asarray(
        [[fx, 0.0, (float(width) - 1.0) * 0.5], [0.0, fy, (float(height) - 1.0) * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def fovy_from_intrinsic(intrinsic_cv: np.ndarray, height: int) -> float:
    """Compute MuJoCo vertical fovy from an OpenCV intrinsic matrix."""

    intrinsic_cv = np.asarray(intrinsic_cv, dtype=np.float64).reshape(3, 3)
    fy = float(intrinsic_cv[1, 1])
    return float(math.degrees(2.0 * math.atan(float(height) / (2.0 * fy))))


class CameraRenderer:
    """Capture RGB, depth, segmentation, masks, and calibration from MuJoCo."""

    def __init__(self, model: mujoco.MjModel, *, camera_specs: dict[str, CameraSpec] | None = None) -> None:
        self.model = model
        self.camera_specs = dict(default_camera_specs() if camera_specs is None else camera_specs)
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def capture(
        self,
        data: mujoco.MjData,
        scene: Any,
        camera_names: tuple[str, ...] = DEFAULT_CAMERA_NAMES,
        *,
        capture_rgb: bool = True,
        capture_segmentation: bool = True,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
        sensor_data: dict[str, dict[str, np.ndarray]] = {}
        sensor_param: dict[str, dict[str, np.ndarray]] = {}
        for name in tuple(str(x) for x in camera_names):
            spec = self.camera_specs.get(name)
            width = int(spec.width if spec is not None else CAMERA_WIDTH)
            height = int(spec.height if spec is not None else CAMERA_HEIGHT)
            renderer = self._renderer(width=width, height=height)

            cam_data: dict[str, np.ndarray] = {}
            if capture_rgb:
                renderer.disable_depth_rendering()
                renderer.disable_segmentation_rendering()
                renderer.update_scene(data, camera=name)
                cam_data["rgb"] = np.asarray(renderer.render(), dtype=np.uint8).copy()

            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera=name)
            depth = np.asarray(renderer.render(), dtype=np.float32).copy()
            renderer.disable_depth_rendering()

            cam_data["depth"] = depth
            if capture_segmentation:
                renderer.enable_segmentation_rendering()
                renderer.update_scene(data, camera=name)
                segmentation = np.asarray(renderer.render(), dtype=np.int32).copy()
                renderer.disable_segmentation_rendering()

                masks = segmentation_masks(self.model, segmentation, scene)
                cam_data.update(
                    {
                        "segmentation": segmentation,
                        "robot_mask": masks["robot"],
                        "object_mask": masks["object"],
                        "cube_mask": masks["cube"],
                        "table_mask": masks["table"],
                    }
                )
            sensor_data[name] = cam_data
            sensor_param[name] = camera_parameters(self.model, data, name, width=width, height=height, spec=spec)
        return sensor_data, sensor_param

    def _renderer(self, *, width: int, height: int) -> mujoco.Renderer:
        key = (int(width), int(height))
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height=int(height), width=int(width))
            self._renderers[key] = renderer
        return renderer


def camera_parameters(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    *,
    width: int,
    height: int,
    spec: CameraSpec | None = None,
) -> dict[str, np.ndarray]:
    """Return OpenCV intrinsics and camera-to-world extrinsics for a camera."""

    cam_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, str(camera_name)))
    if cam_id < 0:
        raise KeyError(f"Camera is missing from model: {camera_name}")

    if spec is not None:
        intrinsic = spec.intrinsic_matrix()
    else:
        intrinsic = intrinsic_from_fovy(width, height, float(model.cam_fovy[cam_id]))

    pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64).reshape(3)
    rot_mj = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    pose_mj = np.eye(4, dtype=np.float64)
    pose_mj[:3, :3] = rot_mj
    pose_mj[:3, 3] = pos

    extrinsic_cv = np.eye(4, dtype=np.float64)
    extrinsic_cv[:3, :3] = rot_mj @ _CV_TO_MUJOCO_CAMERA
    extrinsic_cv[:3, 3] = pos

    quat_xyzw = R.from_matrix(rot_mj).as_quat()
    quat_wxyz = np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
    pose7 = np.concatenate([pos, quat_wxyz])

    return {
        "intrinsic_cv": intrinsic.astype(np.float32),
        "extrinsic_cv": extrinsic_cv.astype(np.float32),
        "camera_pose_world": pose_mj.astype(np.float32),
        "camera_pose_world_wxyz": pose7.astype(np.float32),
    }


def segmentation_masks(model: mujoco.MjModel, segmentation: np.ndarray, scene: Any) -> dict[str, np.ndarray]:
    """Build robot/cube/table masks from MuJoCo segmentation output."""

    seg = np.asarray(segmentation, dtype=np.int32)
    if seg.ndim != 3 or seg.shape[-1] < 2:
        raise ValueError(f"Expected segmentation shape (H, W, 2), got {seg.shape}")

    obj_ids = seg[..., 0]
    obj_types = seg[..., 1]
    geom_pixels = obj_types == int(mujoco.mjtObj.mjOBJ_GEOM)

    object_kind = str(getattr(scene, "object_kind", "cube")).strip().lower()
    cube_gid = _optional_id(model, mujoco.mjtObj.mjOBJ_GEOM, getattr(scene, "cube_geom_name", "cube_geom"))
    table_gid = _optional_id(model, mujoco.mjtObj.mjOBJ_GEOM, getattr(scene, "table_geom_name", "table_top"))
    robot_gids = robot_geom_ids(model)

    cube_gids = _body_descendant_geom_ids(model, getattr(scene, "cube_body_name", "cube"))
    if cube_gids.size:
        cube = geom_pixels & np.isin(obj_ids, cube_gids)
    else:
        cube = geom_pixels & (obj_ids == cube_gid) if cube_gid >= 0 else np.zeros(obj_ids.shape, dtype=np.bool_)
    if object_kind == "ycb_object":
        object_gids = _body_descendant_geom_ids(
            model,
            getattr(scene, "object_body_name", "task_object"),
        )
        obj = geom_pixels & np.isin(obj_ids, object_gids) if object_gids.size else np.zeros(obj_ids.shape, dtype=np.bool_)
    elif object_kind == "pump_bottle":
        object_gids = _body_descendant_geom_ids(
            model,
            getattr(scene, "pump_bottle_body_name", "pump_bottle_bottle_base"),
        )
        obj = geom_pixels & np.isin(obj_ids, object_gids) if object_gids.size else np.zeros(obj_ids.shape, dtype=np.bool_)
    elif object_kind == "mini_keyboard":
        object_gids = _body_descendant_geom_ids(
            model,
            getattr(scene, "mini_keyboard_body_name", "mini_keyboard"),
        )
        obj = geom_pixels & np.isin(obj_ids, object_gids) if object_gids.size else np.zeros(obj_ids.shape, dtype=np.bool_)
    else:
        obj = cube
    table = geom_pixels & (obj_ids == table_gid) if table_gid >= 0 else np.zeros(obj_ids.shape, dtype=np.bool_)
    robot = geom_pixels & np.isin(obj_ids, robot_gids) if robot_gids.size else np.zeros(obj_ids.shape, dtype=np.bool_)
    return {
        "robot": robot.astype(np.bool_),
        "object": obj.astype(np.bool_),
        "cube": cube.astype(np.bool_),
        "table": table.astype(np.bool_),
    }


def robot_geom_ids(
    model: mujoco.MjModel,
    *,
    root_body_name: str = "base",
    root_body_names: tuple[str, ...] | None = None,
) -> np.ndarray:
    roots = tuple(root_body_names or (root_body_name, "g_base", "robotiq_stick", "base_link", "bh_base_link"))
    ids = [
        geom_id
        for geom_id in range(int(model.ngeom))
        if any(_has_body_ancestor(model, int(model.geom_bodyid[geom_id]), root) for root in roots)
    ]
    return np.asarray(ids, dtype=np.int32)


def _body_descendant_geom_ids(model: mujoco.MjModel, root_body_name: str) -> np.ndarray:
    ids = [
        geom_id
        for geom_id in range(int(model.ngeom))
        if _has_body_ancestor(model, int(model.geom_bodyid[geom_id]), str(root_body_name))
    ]
    return np.asarray(ids, dtype=np.int32)


def build_pointcloud_observation(
    sensor_data: dict[str, dict[str, np.ndarray]],
    sensor_param: dict[str, dict[str, np.ndarray]],
    *,
    policy_camera_names: tuple[str, ...] = DEFAULT_POLICY_CAMERA_NAMES,
    target_num: int = 512,
    exclude_robot: bool = True,
    exclude_table: bool = True,
    object_only: bool = False,
    object_mask_erosion_pixels: int = 0,
    object_depth_neighbor_threshold_m: float = 0.0,
    object_pose_wxyz: np.ndarray | None = None,
    object_bounds_m: np.ndarray | None = None,
    object_bounds_filter_tolerance_m: float | None = None,
    workspace_bounds: np.ndarray = DEFAULT_POINTCLOUD_WORKSPACE_BOUNDS,
    rng: np.random.Generator | None = None,
    warn_on_padding: bool = True,
) -> dict[str, np.ndarray]:
    """Reconstruct and sample a policy point cloud from RGBD observations."""

    rng = np.random.default_rng() if rng is None else rng
    target_num = int(target_num)
    if target_num < 0:
        raise ValueError(f"target_num must be nonnegative, got {target_num}")
    bounds = np.asarray(workspace_bounds, dtype=np.float32).reshape(3, 2)
    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []

    for camera_name in tuple(str(x) for x in policy_camera_names):
        if camera_name not in sensor_data or camera_name not in sensor_param:
            continue
        cam_data = sensor_data[camera_name]
        depth = np.asarray(cam_data["depth"], dtype=np.float32)
        rgb = np.asarray(cam_data["rgb"], dtype=np.uint8) if "rgb" in cam_data else None
        valid = np.isfinite(depth) & (depth > 1e-5)
        if object_only:
            if "object_mask" not in cam_data:
                raise KeyError("object_only point clouds require captured segmentation/object_mask")
            valid &= erode_binary_mask(
                np.asarray(cam_data["object_mask"], dtype=bool),
                pixels=int(object_mask_erosion_pixels),
            )
        if exclude_robot and "robot_mask" in cam_data:
            valid &= ~np.asarray(cam_data["robot_mask"], dtype=bool)
        has_table_mask = "table_mask" in cam_data
        if exclude_table and has_table_mask:
            valid &= ~np.asarray(cam_data["table_mask"], dtype=bool)
        if object_only and float(object_depth_neighbor_threshold_m) > 0.0:
            valid = depth_neighbor_consistent_mask(
                depth,
                valid,
                max_delta_m=float(object_depth_neighbor_threshold_m),
            )

        valid_flat = np.flatnonzero(valid.reshape(-1))
        if valid_flat.size == 0:
            continue
        ys, xs = np.divmod(valid_flat, depth.shape[1])
        z = depth.reshape(-1)[valid_flat]
        xyz = deproject_pixels_cv(
            xs.astype(np.float32),
            ys.astype(np.float32),
            z.astype(np.float32),
            sensor_param[camera_name]["intrinsic_cv"],
            sensor_param[camera_name]["extrinsic_cv"],
        )
        in_bounds = np.all((xyz >= bounds[:, 0]) & (xyz <= bounds[:, 1]), axis=1)
        if object_only and object_bounds_filter_tolerance_m is not None:
            if object_pose_wxyz is None or object_bounds_m is None:
                raise ValueError(
                    "object_pose_wxyz and object_bounds_m are required when "
                    "object_bounds_filter_tolerance_m is set"
                )
            in_bounds &= object_local_bounds_mask(
                xyz,
                object_pose_wxyz,
                object_bounds_m,
                tolerance_m=float(object_bounds_filter_tolerance_m),
            )
        if exclude_table and not has_table_mask:
            in_bounds &= xyz[:, 2] > float(TABLE_TOP_Z + 0.002)
        if not np.any(in_bounds):
            continue
        points.append(xyz[in_bounds])
        if rgb is None:
            colors.append(np.full((int(np.count_nonzero(in_bounds)), 3), 180, dtype=np.uint8))
        else:
            colors.append(rgb.reshape(-1, 3)[valid_flat][in_bounds])

    if not points:
        if target_num > 0:
            if warn_on_padding:
                warnings.warn(
                    "Filtered policy point cloud has fewer points than requested: "
                    f"selected=0, target={target_num}. "
                    "Returning zero-filled placeholder points because no filtered points are available.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return {
                "xyzw": np.concatenate(
                    [np.zeros((target_num, 3), dtype=np.float32), np.ones((target_num, 1), dtype=np.float32)],
                    axis=1,
                ),
                "rgb": np.zeros((target_num, 3), dtype=np.uint8),
            }
        return {
            "xyzw": np.zeros((0, 4), dtype=np.float32),
            "rgb": np.zeros((0, 3), dtype=np.uint8),
        }

    xyz_all = np.concatenate(points, axis=0).astype(np.float32)
    rgb_all = np.concatenate(colors, axis=0).astype(np.uint8)
    if target_num == 0:
        return {
            "xyzw": np.zeros((0, 4), dtype=np.float32),
            "rgb": np.zeros((0, 3), dtype=np.uint8),
        }
    if target_num > 0:
        selected_count = int(xyz_all.shape[0])
        if selected_count < target_num:
            if warn_on_padding:
                warnings.warn(
                    "Filtered policy point cloud has fewer points than requested: "
                    f"selected={selected_count}, target={target_num}. "
                    "Repeating filtered points with replacement to reach the requested count.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if object_only:
                # Deterministic cyclic padding matches object-only evaluation.
                repeat_indices = np.resize(
                    np.arange(selected_count, dtype=np.int64),
                    target_num - selected_count,
                )
            else:
                repeat_indices = rng.choice(
                    selected_count,
                    size=target_num - selected_count,
                    replace=True,
                )
            xyz_all = np.concatenate([xyz_all, xyz_all[repeat_indices]], axis=0)
            rgb_all = np.concatenate([rgb_all, rgb_all[repeat_indices]], axis=0)
        elif selected_count > target_num:
            indices = farthest_point_sample_indices(xyz_all, target_num)
            xyz_all = xyz_all[indices]
            rgb_all = rgb_all[indices]
    xyzw = np.concatenate([xyz_all, np.ones((xyz_all.shape[0], 1), dtype=np.float32)], axis=1)
    return {"xyzw": xyzw.astype(np.float32), "rgb": rgb_all}


def erode_binary_mask(mask: np.ndarray, *, pixels: int = 1) -> np.ndarray:
    """Erode a 2D mask with a 3x3 square kernel using only NumPy."""

    result = np.asarray(mask, dtype=np.bool_)
    if result.ndim != 2:
        raise ValueError(f"mask must be 2D, got {result.shape}")
    count = int(pixels)
    if count < 0:
        raise ValueError(f"pixels must be nonnegative, got {pixels}")
    result = result.copy()
    for _ in range(count):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.logical_and.reduce(
            [padded[dy : dy + result.shape[0], dx : dx + result.shape[1]] for dy in range(3) for dx in range(3)]
        )
    return result


def depth_neighbor_consistent_mask(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    *,
    max_delta_m: float,
) -> np.ndarray:
    """Reject isolated depth pixels that disagree with every valid 8-neighbor."""

    depth_array = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if depth_array.ndim != 2 or valid.shape != depth_array.shape:
        raise ValueError(f"depth and valid_mask must have the same 2D shape, got {depth_array.shape}/{valid.shape}")
    threshold = float(max_delta_m)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"max_delta_m must be finite and positive, got {max_delta_m}")
    valid = valid & np.isfinite(depth_array) & (depth_array > 1.0e-5)
    padded_valid = np.pad(valid, 1, mode="constant", constant_values=False)
    padded_depth = np.pad(depth_array, 1, mode="constant", constant_values=np.nan)
    supported = np.zeros_like(valid)
    height, width = valid.shape
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            neighbor_valid = padded_valid[dy : dy + height, dx : dx + width]
            neighbor_depth = padded_depth[dy : dy + height, dx : dx + width]
            supported |= neighbor_valid & (np.abs(depth_array - neighbor_depth) <= threshold)
    return valid & supported


def object_local_bounds_mask(
    points_world: np.ndarray,
    object_pose_wxyz: np.ndarray,
    object_bounds_m: np.ndarray,
    *,
    tolerance_m: float,
) -> np.ndarray:
    """Select world-frame points inside the posed object's expanded local AABB."""

    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    pose = np.asarray(object_pose_wxyz, dtype=np.float32).reshape(7)
    bounds = np.asarray(object_bounds_m, dtype=np.float32)
    if bounds.shape != (2, 3) or bool(np.any(bounds[1] < bounds[0])):
        raise ValueError(f"object_bounds_m must have ordered shape (2, 3), got {bounds}")
    tolerance = float(tolerance_m)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"tolerance_m must be finite and nonnegative, got {tolerance_m}")
    quat_wxyz = pose[3:7]
    rotation_world_from_local = R.from_quat(
        np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
    ).as_matrix().astype(np.float32)
    points_local = (points - pose[None, :3]) @ rotation_world_from_local
    return np.all(
        (points_local >= bounds[0][None, :] - tolerance)
        & (points_local <= bounds[1][None, :] + tolerance),
        axis=1,
    )


def farthest_point_sample_indices(points: np.ndarray, target_num: int) -> np.ndarray:
    """Return deterministic farthest-point-sampling indices for XYZ points."""

    xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    target_num = int(target_num)
    point_count = int(xyz.shape[0])
    if target_num < 0:
        raise ValueError(f"target_num must be nonnegative, got {target_num}")
    if target_num > point_count:
        raise ValueError(f"Cannot sample {target_num} points from only {point_count} candidates")
    if target_num == 0:
        return np.zeros((0,), dtype=np.int64)
    if target_num == point_count:
        return np.arange(point_count, dtype=np.int64)

    selected = np.empty((target_num,), dtype=np.int64)
    centroid = np.mean(xyz, axis=0)
    first = int(np.argmax(np.sum((xyz - centroid[None, :]) ** 2, axis=1)))
    selected[0] = first
    min_dist_sq = np.sum((xyz - xyz[first][None, :]) ** 2, axis=1)
    min_dist_sq[first] = -1.0
    # Contiguous coordinate arrays avoid an Nx3 allocation and strided row
    # reduction at every iteration, especially when storing 4,096 points.
    x, y, z = xyz.T.copy()

    for sample_idx in range(1, target_num):
        next_idx = int(np.argmax(min_dist_sq))
        selected[sample_idx] = next_idx
        dist_sq = (x - x[next_idx]) ** 2 + (y - y[next_idx]) ** 2 + (z - z[next_idx]) ** 2
        np.minimum(min_dist_sq, dist_sq, out=min_dist_sq)
        # Earlier selected entries remain -1 under the minimum operation.
        min_dist_sq[next_idx] = -1.0
    return selected


def deproject_pixels_cv(
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    depth: np.ndarray,
    intrinsic_cv: np.ndarray,
    extrinsic_cv: np.ndarray,
) -> np.ndarray:
    """Deproject OpenCV pixels and depth to world XYZ."""

    x = np.asarray(pixel_x, dtype=np.float32).reshape(-1)
    y = np.asarray(pixel_y, dtype=np.float32).reshape(-1)
    z = np.asarray(depth, dtype=np.float32).reshape(-1)
    k = np.asarray(intrinsic_cv, dtype=np.float64).reshape(3, 3)
    t_world_cam = np.asarray(extrinsic_cv, dtype=np.float64).reshape(4, 4)

    x_cam = (x.astype(np.float64) - float(k[0, 2])) * z.astype(np.float64) / float(k[0, 0])
    y_cam = (y.astype(np.float64) - float(k[1, 2])) * z.astype(np.float64) / float(k[1, 1])
    pts_cam = np.stack([x_cam, y_cam, z.astype(np.float64)], axis=1)
    pts_world = pts_cam @ t_world_cam[:3, :3].T + t_world_cam[:3, 3][None, :]
    return pts_world.astype(np.float32)


def camera_frame_from_observation(obs: dict[str, Any]) -> dict[str, Any] | None:
    if "sensor_data" not in obs or "sensor_param" not in obs:
        return None
    sensor_data = obs["sensor_data"]
    sensor_param = obs["sensor_param"]
    camera_names = tuple(str(x) for x in obs.get("camera_names", tuple(sensor_data.keys())))
    policy_camera_names = tuple(str(x) for x in obs.get("policy_camera_names", DEFAULT_POLICY_CAMERA_NAMES))

    frame: dict[str, Any] = {
        "camera_names": camera_names,
        "policy_camera_names": policy_camera_names,
    }
    if all("rgb" in sensor_data[name] for name in camera_names):
        frame["rgb"] = np.stack([np.asarray(sensor_data[name]["rgb"], dtype=np.uint8) for name in camera_names], axis=0)
    frame["depth"] = np.stack([np.asarray(sensor_data[name]["depth"], dtype=np.float32) for name in camera_names], axis=0)
    for mask_key in ("robot_mask", "object_mask", "cube_mask", "table_mask"):
        if all(mask_key in sensor_data[name] for name in camera_names):
            frame[mask_key] = np.stack(
                [np.asarray(sensor_data[name][mask_key], dtype=np.bool_) for name in camera_names],
                axis=0,
            )
    if all("segmentation" in sensor_data[name] for name in camera_names):
        frame["segmentation"] = np.stack(
            [np.asarray(sensor_data[name]["segmentation"], dtype=np.int32) for name in camera_names],
            axis=0,
        )
    for param_key in ("intrinsic_cv", "extrinsic_cv", "camera_pose_world", "camera_pose_world_wxyz"):
        frame[param_key] = np.stack(
            [np.asarray(sensor_param[name][param_key], dtype=np.float32) for name in camera_names],
            axis=0,
        )
    return frame


def append_camera_observation(buffer: dict[str, Any], obs: dict[str, Any], *, prefix: str = "camera_") -> bool:
    frame = camera_frame_from_observation(obs)
    if frame is None:
        return False
    names = tuple(frame["camera_names"])
    policy_names = tuple(frame["policy_camera_names"])
    if buffer.get("_camera_names") is None:
        buffer["_camera_names"] = names
        buffer["_policy_camera_names"] = policy_names
    elif tuple(buffer["_camera_names"]) != names:
        raise ValueError(f"Camera names changed within buffer: {buffer['_camera_names']} -> {names}")

    for key in CAMERA_FRAME_KEYS:
        if key in frame:
            buffer.setdefault(_prefixed_camera_key(key, prefix), []).append(np.asarray(frame[key]).copy())
    return True


def _optional_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    if not name:
        return -1
    return int(mujoco.mj_name2id(model, obj_type, str(name)))


def _prefixed_camera_key(key: str, prefix: str) -> str:
    if prefix and str(key).startswith(str(prefix)):
        return str(key)
    return f"{prefix}{key}"


def _has_body_ancestor(model: mujoco.MjModel, body_id: int, name: str) -> bool:
    cur = int(body_id)
    while cur >= 0:
        cur_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cur)
        if cur_name == name:
            return True
        parent = int(model.body_parentid[cur])
        if parent == cur:
            return False
        cur = parent
    return False
