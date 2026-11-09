"""Task helpers for the tabletop simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .assets import (
    CUBE_HALF_EXTENT,
    CUBE_START_POS,
    CUBE_START_QUAT_WXYZ,
    MINI_KEYBOARD_ROOT_Z_OFFSET,
    MINI_KEYBOARD_SUPPORT_BOX_HEIGHT,
    TABLE_TOP_Z,
)
from .scene import (
    OBJECT_SUITE_BIN_FRICTION,
    SceneSpec,
    cube_visual_box_vertices,
    cube_visual_face_vertices,
)
from .object_suite import ObjectSpec

PUMP_BOTTLE_DEFAULT_QUAT_WXYZ = (
    0.00017366445,
    0.0099211456,
    0.0040169274,
    -0.99994272,
)
DEFAULT_KEYBOARD_TARGET_KEYS = (
    "white_key_00_C2_slide",
    "white_key_02_E2_slide",
    "white_key_04_G2_slide",
)
DEFAULT_KEYBOARD_QUAT_WXYZ = (
    0.70710678,
    0.0,
    0.0,
    -0.70710678,
)


@dataclass(frozen=True)
class ObjectSuitePickPlaceTask:
    """One upright YCB object placed into a fixed bin."""

    object_spec: ObjectSpec
    objective: str = "pick_place"
    xy_range: tuple[tuple[float, float], tuple[float, float]] = ((0.06, 0.18), (-0.18, 0.08))
    yaw_range_rad: tuple[float, float] = (-math.pi, math.pi)
    table_clearance: float = 0.003
    table_top_z: float = TABLE_TOP_Z
    bin_center_xy: tuple[float, float] = (0.24, 0.30)
    bin_inner_size: tuple[float, float] = (0.32, 0.32)
    bin_floor_thickness: float = 0.006
    bin_wall_height: float = 0.04
    bin_wall_thickness: float = 0.008
    bin_friction: tuple[float, float, float] = OBJECT_SUITE_BIN_FRICTION
    placement_margin: float = 0.005
    placement_reference: str = "bounds"
    bottom_tolerance: float = 0.03
    lift_delta: float = 0.05
    scene_object_kind: str = field(default="ycb_object", init=False)

    def __post_init__(self) -> None:
        if self.objective not in {"pick_place", "pick_only"}:
            raise ValueError(
                "objective must be 'pick_place' or 'pick_only', "
                f"got {self.objective!r}"
            )
        if self.placement_reference not in {"bounds", "center"}:
            raise ValueError(
                "placement_reference must be 'bounds' or 'center', "
                f"got {self.placement_reference!r}"
            )

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "task": "object_suite_pick_place",
            "task_objective": self.objective,
            "object_id": self.object_spec.object_id,
            "object_name": self.object_spec.name,
            "object_asset_hash": self.object_spec.asset_hash,
            "object_mass_kg": self.object_spec.mass_kg,
            "object_upright_quat_wxyz": self.object_spec.upright_quat_wxyz,
            "object_xy_range": self.xy_range,
            "object_yaw_range_rad": self.yaw_range_rad,
            "bin_center_xy": self.bin_center_xy,
            "bin_inner_size": self.bin_inner_size,
            "bin_wall_height": self.bin_wall_height,
            "bin_friction": self.bin_friction,
            "placement_margin": self.placement_margin,
            "placement_reference": self.placement_reference,
            "lift_delta": self.lift_delta,
            "pick_success_z": self.pick_success_z,
        }

    @property
    def pick_success_z(self) -> float:
        """Absolute object-origin height equivalent to a lift from reset."""

        upright_pose = np.asarray(
            [0.0, 0.0, 0.0, *self.object_spec.upright_quat_wxyz],
            dtype=np.float64,
        )
        bounds_at_origin = self.world_bounds(upright_pose)
        spawn_z = float(self.table_top_z - bounds_at_origin[0, 2] + self.table_clearance)
        return spawn_z + float(self.lift_delta)

    def reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        scene: SceneSpec,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        qadr, dadr = self._object_qpos_qvel_adr(model, scene)
        x = float(rng.uniform(*self.xy_range[0]))
        y = float(rng.uniform(*self.xy_range[1]))
        yaw = float(rng.uniform(*self.yaw_range_rad))
        quat = _quat_mul_wxyz(
            self._yaw_quat_wxyz(yaw),
            np.asarray(self.object_spec.upright_quat_wxyz, dtype=np.float64),
        )
        bounds_at_origin = self.world_bounds(np.asarray([0.0, 0.0, 0.0, *quat]))
        z = float(self.table_top_z - bounds_at_origin[0, 2] + self.table_clearance)
        data.qpos[qadr : qadr + 7] = np.asarray([x, y, z, *quat], dtype=np.float64)
        data.qvel[dadr : dadr + 6] = 0.0
        mujoco.mj_forward(model, data)
        return self.observe(model, data, scene)

    def observe(self, model: mujoco.MjModel, data: mujoco.MjData, scene: SceneSpec) -> dict[str, np.ndarray]:
        qadr, dadr = self._object_qpos_qvel_adr(model, scene)
        pose = np.asarray(data.qpos[qadr : qadr + 7], dtype=np.float64).copy()
        velocity = np.asarray(data.qvel[dadr : dadr + 6], dtype=np.float64).copy()
        bounds = self.world_bounds(pose)
        bin_pose = np.asarray([self.bin_center_xy[0], self.bin_center_xy[1], self.table_top_z, 1, 0, 0, 0])
        return {
            "object_pose": pose.astype(np.float32),
            "object_pos": pose[:3].astype(np.float32),
            "object_quat_wxyz": pose[3:7].astype(np.float32),
            "object_velocity": velocity.astype(np.float32),
            "object_bounds_world": bounds.astype(np.float32),
            "bin_pose": bin_pose.astype(np.float32),
            "placement_margin": np.asarray(self.placement_xy_margin(bounds), dtype=np.float32),
            "success": np.asarray(self.is_success(model, data, scene), dtype=np.bool_),
        }

    def is_success(self, model: mujoco.MjModel, data: mujoco.MjData, scene: SceneSpec) -> bool:
        qadr, _ = self._object_qpos_qvel_adr(model, scene)
        pose = np.asarray(data.qpos[qadr : qadr + 7], dtype=np.float64)
        if self.objective == "pick_only":
            return bool(float(pose[2]) >= self.pick_success_z)
        bounds = self.world_bounds(pose)
        inside = self.placement_xy_margin(bounds) >= float(self.placement_margin)
        return bool(inside)

    def world_bounds(self, pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64).reshape(7)
        lo, hi = self.object_spec.mesh_bounds_m
        corners = np.asarray(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
            dtype=np.float64,
        )
        rot_flat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot_flat, pose[3:7])
        world = corners @ rot_flat.reshape(3, 3).T + pose[:3]
        return np.stack([world.min(axis=0), world.max(axis=0)], axis=0)

    def placement_xy_margin(self, bounds: np.ndarray) -> float:
        bounds = np.asarray(bounds, dtype=np.float64).reshape(2, 3)
        half = 0.5 * np.asarray(self.bin_inner_size, dtype=np.float64)
        center = np.asarray(self.bin_center_xy, dtype=np.float64)
        lower = center - half
        upper = center + half
        if self.placement_reference == "center":
            object_center = 0.5 * (bounds[0, :2] + bounds[1, :2])
            distances = np.concatenate([object_center - lower, upper - object_center])
        else:
            distances = np.asarray(
                [
                    bounds[0, 0] - lower[0],
                    upper[0] - bounds[1, 0],
                    bounds[0, 1] - lower[1],
                    upper[1] - bounds[1, 1],
                ],
                dtype=np.float64,
            )
        return float(np.min(distances))

    def _object_qpos_qvel_adr(self, model: mujoco.MjModel, scene: SceneSpec) -> tuple[int, int]:
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, scene.object_joint_name))
        if jid < 0:
            raise KeyError(f"Object joint is missing from model: {scene.object_joint_name}")
        return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])

    @staticmethod
    def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
        half = 0.5 * float(yaw)
        return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


@dataclass(frozen=True)
class CubePickTask:
    """Small procedural cube-pick task."""

    cube_pos: tuple[float, float, float] = CUBE_START_POS
    cube_quat_wxyz: tuple[float, float, float, float] = CUBE_START_QUAT_WXYZ
    xy_randomization: float = 0.015
    cube_xy_range: tuple[tuple[float, float], tuple[float, float]] | None = None
    cube_yaw_range_rad: tuple[float, float] = (0.0, 0.0)
    cube_side_length_range: tuple[float, float] = (2.0 * CUBE_HALF_EXTENT, 2.0 * CUBE_HALF_EXTENT)
    cube_height_ratio: float = 1.0
    table_clearance: float = 0.003
    reference_cube_side_length: float = 2.0 * CUBE_HALF_EXTENT
    reference_cube_mass: float = 0.08
    lift_success_z: float = TABLE_TOP_Z + 0.12

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "task": "cube_pick",
            "cube_size_range": self.cube_side_length_range,
            "cube_height_ratio": self.cube_height_ratio,
            "cube_height_range": tuple(self._cube_height_from_side(value) for value in self.cube_side_length_range),
            "cube_xy_range": self.cube_xy_range,
            "cube_yaw_range_rad": self.cube_yaw_range_rad,
            "table_top_z": TABLE_TOP_Z,
            "lift_success_z": self.lift_success_z,
        }

    def reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        scene: SceneSpec,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        qadr, dadr = self._cube_qpos_qvel_adr(model, scene)
        side_length = self._sample_range(self.cube_side_length_range, rng)
        height = self._cube_height_from_side(side_length)
        half_height = 0.5 * height
        self._set_cube_size(model, scene, side_length=side_length, height=height)

        pos = np.asarray(self.cube_pos, dtype=np.float64).copy()
        if self.cube_xy_range is not None:
            pos[0] = rng.uniform(float(self.cube_xy_range[0][0]), float(self.cube_xy_range[0][1]))
            pos[1] = rng.uniform(float(self.cube_xy_range[1][0]), float(self.cube_xy_range[1][1]))
        elif self.xy_randomization > 0.0:
            pos[:2] += rng.uniform(-self.xy_randomization, self.xy_randomization, size=2)
        pos[2] = TABLE_TOP_Z + half_height + float(self.table_clearance)

        yaw = self._sample_range(self.cube_yaw_range_rad, rng)
        quat = self._yaw_quat_wxyz(yaw)
        data.qpos[qadr : qadr + 7] = np.concatenate([pos, quat])
        data.qvel[dadr : dadr + 6] = 0.0
        mujoco.mj_forward(model, data)
        return self.observe(model, data, scene)

    def observe(self, model: mujoco.MjModel, data: mujoco.MjData, scene: SceneSpec) -> dict[str, np.ndarray]:
        qadr, _ = self._cube_qpos_qvel_adr(model, scene)
        cube_pose = np.asarray(data.qpos[qadr : qadr + 7], dtype=np.float64).copy()
        cube_side_length = self._cube_side_length(model, scene)
        cube_height = self._cube_height(model, scene)
        cube_yaw = self._yaw_from_quat_wxyz(cube_pose[3:7])
        return {
            "cube_pose": cube_pose.astype(np.float32),
            "cube_pos": cube_pose[:3].astype(np.float32),
            "cube_quat_wxyz": cube_pose[3:7].astype(np.float32),
            "cube_side_length": np.asarray(cube_side_length, dtype=np.float32),
            "cube_height": np.asarray(cube_height, dtype=np.float32),
            "cube_yaw": np.asarray(cube_yaw, dtype=np.float32),
            "success": np.asarray(self.is_success(model, data, scene), dtype=np.bool_),
        }

    def is_success(self, model: mujoco.MjModel, data: mujoco.MjData, scene: SceneSpec) -> bool:
        qadr, _ = self._cube_qpos_qvel_adr(model, scene)
        return bool(float(data.qpos[qadr + 2]) >= float(self.lift_success_z))

    def _cube_qpos_qvel_adr(self, model: mujoco.MjModel, scene: SceneSpec) -> tuple[int, int]:
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, scene.cube_joint_name))
        if jid < 0:
            raise KeyError(f"Cube joint is missing from model: {scene.cube_joint_name}")
        return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])

    def _cube_geom_id(self, model: mujoco.MjModel, scene: SceneSpec) -> int:
        gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, scene.cube_geom_name))
        if gid < 0:
            raise KeyError(f"Cube geom is missing from model: {scene.cube_geom_name}")
        return gid

    def _cube_body_id(self, model: mujoco.MjModel, scene: SceneSpec) -> int:
        bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, scene.cube_body_name))
        if bid < 0:
            raise KeyError(f"Cube body is missing from model: {scene.cube_body_name}")
        return bid

    def _set_cube_size(
        self,
        model: mujoco.MjModel,
        scene: SceneSpec,
        *,
        side_length: float,
        height: float | None = None,
    ) -> None:
        side_length = float(side_length)
        height = side_length if height is None else float(height)
        if side_length <= 0.0:
            raise ValueError(f"cube side_length must be positive, got {side_length}")
        if height <= 0.0:
            raise ValueError(f"cube height must be positive, got {height}")
        half_side = 0.5 * side_length
        half_height = 0.5 * height
        gid = self._cube_geom_id(model, scene)
        bid = self._cube_body_id(model, scene)
        model.geom_size[gid] = np.asarray([half_side, half_side, half_height], dtype=np.float64)
        if hasattr(model, "geom_rbound"):
            model.geom_rbound[gid] = float(np.linalg.norm(model.geom_size[gid]))
        self._set_cube_visual_face_sizes(model, scene, half_side=half_side, half_height=half_height)

        reference_volume = float(self.reference_cube_side_length) ** 3
        mass = float(self.reference_cube_mass) * (side_length * side_length * height / reference_volume)
        inertia_x = mass * (side_length * side_length + height * height) / 12.0
        inertia_y = inertia_x
        inertia_z = mass * (side_length * side_length + side_length * side_length) / 12.0
        model.body_mass[bid] = mass
        model.body_inertia[bid] = np.asarray([inertia_x, inertia_y, inertia_z], dtype=np.float64)

    def _set_cube_visual_face_sizes(
        self,
        model: mujoco.MjModel,
        scene: SceneSpec,
        *,
        half_side: float,
        half_height: float,
    ) -> None:
        box_mesh_name = getattr(scene, "cube_visual_box_mesh_name", None)
        if box_mesh_name:
            mid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, str(box_mesh_name)))
            if mid >= 0:
                verts = np.asarray(
                    cube_visual_box_vertices(
                        half_side=float(half_side),
                        half_height=float(half_height),
                        visual_face_offset=float(getattr(scene, "cube_visual_face_offset", 8.0e-5)),
                    ),
                    dtype=np.float64,
                ).reshape(24, 3)
                vert_adr = int(model.mesh_vertadr[mid])
                vert_num = int(model.mesh_vertnum[mid])
                if vert_num == 24:
                    model.mesh_vert[vert_adr : vert_adr + vert_num] = verts
                    if hasattr(model, "mesh_rbound"):
                        model.mesh_rbound[mid] = float(np.linalg.norm(verts, axis=1).max())
        face_mesh_names = tuple(getattr(scene, "cube_visual_face_mesh_names", ()))
        if not face_mesh_names:
            return
        offset = float(getattr(scene, "cube_visual_face_offset", 8.0e-5))
        for mesh_name in face_mesh_names:
            mid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, str(mesh_name)))
            if mid < 0:
                continue
            face = str(mesh_name).removeprefix("cube_visual_").removesuffix("_mesh")
            verts = np.asarray(
                cube_visual_face_vertices(
                    face,
                    half_side=float(half_side),
                    half_height=float(half_height),
                    visual_face_offset=offset,
                ),
                dtype=np.float64,
            ).reshape(4, 3)
            vert_adr = int(model.mesh_vertadr[mid])
            vert_num = int(model.mesh_vertnum[mid])
            if vert_num != 4:
                continue
            model.mesh_vert[vert_adr : vert_adr + vert_num] = verts
            if hasattr(model, "mesh_rbound"):
                model.mesh_rbound[mid] = float(np.linalg.norm(verts, axis=1).max())

    def _cube_side_length(self, model: mujoco.MjModel, scene: SceneSpec) -> float:
        gid = self._cube_geom_id(model, scene)
        return float(2.0 * model.geom_size[gid, 0])

    def _cube_height(self, model: mujoco.MjModel, scene: SceneSpec) -> float:
        gid = self._cube_geom_id(model, scene)
        return float(2.0 * model.geom_size[gid, 2])

    def _cube_height_from_side(self, side_length: float) -> float:
        ratio = float(self.cube_height_ratio)
        if ratio <= 0.0:
            raise ValueError(f"cube_height_ratio must be positive, got {ratio}")
        return float(side_length) * ratio

    @staticmethod
    def _sample_range(bounds: tuple[float, float], rng: np.random.Generator) -> float:
        lo, hi = float(bounds[0]), float(bounds[1])
        if hi == lo:
            return lo
        return float(rng.uniform(lo, hi))

    @staticmethod
    def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
        half = 0.5 * float(yaw)
        return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)

    @staticmethod
    def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
        return _yaw_from_quat_wxyz(quat)


@dataclass
class PumpBottleUseTask:
    """Procedural articulated pump-bottle use task."""

    object_pos: tuple[float, float, float] = (0.15800001, 0.003, 0.6397543)
    object_quat_wxyz: tuple[float, float, float, float] = PUMP_BOTTLE_DEFAULT_QUAT_WXYZ
    xy_randomization: float = 0.015
    xy_range: tuple[tuple[float, float], tuple[float, float]] | None = None
    yaw_range_rad: tuple[float, float] = (0.0, 0.0)
    table_clearance: float = 0.0
    button_reset_slide: float = -0.00700733
    button_success_slide: float = -0.012
    lift_success_delta: float = 0.08
    safe_radius: float = 0.05
    object_id: str = "pump_bottle005"
    _reset_object_z: float = field(default=0.6397543, init=False, repr=False)
    _reset_min_geom_z: float = field(default=float("nan"), init=False, repr=False)

    scene_object_kind: str = "pump_bottle"

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "task": "pumpbottle_use",
            "object": self.object_id,
            "object_xy_range": self.xy_range,
            "object_quat_wxyz": self.object_quat_wxyz,
            "object_yaw_range_rad": self.yaw_range_rad,
            "table_top_z": TABLE_TOP_Z,
            "button_reset_slide": self.button_reset_slide,
            "button_success_slide": self.button_success_slide,
            "lift_success_delta": self.lift_success_delta,
            "lift_measure": "min_body_geom_z_delta",
            "safe_radius": self.safe_radius,
        }

    def reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        scene: SceneSpec,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        qadr, dadr = self._object_qpos_qvel_adr(model, scene)
        pos = np.asarray(self.object_pos, dtype=np.float64).copy()
        if self.xy_range is not None:
            pos[0] = rng.uniform(float(self.xy_range[0][0]), float(self.xy_range[0][1]))
            pos[1] = rng.uniform(float(self.xy_range[1][0]), float(self.xy_range[1][1]))
        elif self.xy_randomization > 0.0:
            pos[:2] += rng.uniform(-self.xy_randomization, self.xy_randomization, size=2)
        pos[2] = float(pos[2]) + float(self.table_clearance)

        yaw = _sample_range(self.yaw_range_rad, rng)
        quat = _quat_mul_wxyz(_yaw_quat_wxyz(yaw), _normalize_quat_wxyz(self.object_quat_wxyz))
        data.qpos[qadr : qadr + 7] = np.concatenate([pos, quat])
        data.qvel[dadr : dadr + 6] = 0.0
        self._reset_object_z = float(pos[2])

        button_qadr, button_dadr = self._button_qpos_qvel_adr(model, scene)
        lo, hi = self._button_range(model, scene)
        button_slide = float(np.clip(float(self.button_reset_slide), lo, hi))
        data.qpos[button_qadr] = button_slide
        data.qvel[button_dadr] = 0.0
        actuator_id = self._button_actuator_id(model)
        if actuator_id >= 0:
            data.ctrl[actuator_id] = button_slide

        mujoco.mj_forward(model, data)
        self.ensure_above_table(model, data, scene)
        self._reset_object_z = float(data.qpos[qadr + 2])
        self._reset_min_geom_z = _body_tree_geom_min_z(
            model,
            data,
            scene.pump_bottle_body_name,
        )
        return self.observe(model, data, scene)

    def observe(self, model: mujoco.MjModel, data: mujoco.MjData, scene: SceneSpec) -> dict[str, np.ndarray]:
        qadr, _ = self._object_qpos_qvel_adr(model, scene)
        object_pose = np.asarray(data.qpos[qadr : qadr + 7], dtype=np.float64).copy()
        button_qadr, _ = self._button_qpos_qvel_adr(model, scene)
        button_slide = float(data.qpos[button_qadr])
        lift_delta = float(object_pose[2] - self._reset_object_z)
        object_min_geom_z = _body_tree_geom_min_z(model, data, scene.pump_bottle_body_name)
        if not np.isfinite(self._reset_min_geom_z):
            self._reset_min_geom_z = float(object_min_geom_z)
        lift_clearance_delta = float(object_min_geom_z - self._reset_min_geom_z)
        lift_success = lift_clearance_delta >= float(self.lift_success_delta)
        button_success = button_slide <= float(self.button_success_slide)
        object_yaw = _yaw_from_quat_wxyz(object_pose[3:7])
        site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, scene.pump_bottle_button_site_name))
        button_site_pos = (
            np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
            if site_id >= 0
            else np.full((3,), np.nan, dtype=np.float64)
        )
        return {
            "object_pose": object_pose.astype(np.float32),
            "object_pos": object_pose[:3].astype(np.float32),
            "object_quat_wxyz": object_pose[3:7].astype(np.float32),
            "object_yaw": np.asarray(object_yaw, dtype=np.float32),
            "button_slide": np.asarray(button_slide, dtype=np.float32),
            "button_site_pos": button_site_pos.astype(np.float32),
            "object_reset_z": np.asarray(self._reset_object_z, dtype=np.float32),
            "lift_delta": np.asarray(lift_delta, dtype=np.float32),
            "object_min_geom_z": np.asarray(object_min_geom_z, dtype=np.float32),
            "object_reset_min_geom_z": np.asarray(self._reset_min_geom_z, dtype=np.float32),
            "lift_clearance_delta": np.asarray(lift_clearance_delta, dtype=np.float32),
            "lift_success": np.asarray(lift_success, dtype=np.bool_),
            "button_success": np.asarray(button_success, dtype=np.bool_),
            "success": np.asarray(bool(lift_success and button_success), dtype=np.bool_),
        }

    def is_success(self, model: mujoco.MjModel, data: mujoco.MjData, scene: SceneSpec) -> bool:
        obs = self.observe(model, data, scene)
        return bool(obs["success"])

    def _object_qpos_qvel_adr(self, model: mujoco.MjModel, scene: SceneSpec) -> tuple[int, int]:
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, scene.pump_bottle_free_joint_name))
        if jid < 0:
            raise KeyError(f"Pump bottle free joint is missing from model: {scene.pump_bottle_free_joint_name}")
        return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])

    def _button_qpos_qvel_adr(self, model: mujoco.MjModel, scene: SceneSpec) -> tuple[int, int]:
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, scene.pump_bottle_button_joint_name))
        if jid < 0:
            raise KeyError(f"Pump bottle button joint is missing from model: {scene.pump_bottle_button_joint_name}")
        return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])

    def _button_range(self, model: mujoco.MjModel, scene: SceneSpec) -> tuple[float, float]:
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, scene.pump_bottle_button_joint_name))
        if jid < 0:
            raise KeyError(f"Pump bottle button joint is missing from model: {scene.pump_bottle_button_joint_name}")
        return float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])

    @staticmethod
    def _button_actuator_id(model: mujoco.MjModel) -> int:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pump_bottle_button_position"))

    def ensure_above_table(self, model: mujoco.MjModel, data: mujoco.MjData, scene: SceneSpec) -> float:
        """Lift the free bottle just enough that its compiled geoms clear the tabletop."""

        qadr, _ = self._object_qpos_qvel_adr(model, scene)
        mujoco.mj_forward(model, data)
        min_z = _body_tree_geom_min_z(model, data, scene.pump_bottle_body_name)
        target_z = float(TABLE_TOP_Z) + max(0.0, float(self.table_clearance))
        delta = max(0.0, target_z - float(min_z))
        if delta > 1.0e-9:
            data.qpos[qadr + 2] += delta
            mujoco.mj_forward(model, data)
        self._reset_object_z = float(data.qpos[qadr + 2])
        return float(delta)


@dataclass
class KeyboardPressTask:
    """Procedural fixed-base mini-keyboard key-press task."""

    object_pos: tuple[float, float, float] = (
        0.15,
        0.0,
        TABLE_TOP_Z + MINI_KEYBOARD_SUPPORT_BOX_HEIGHT + MINI_KEYBOARD_ROOT_Z_OFFSET,
    )
    object_quat_wxyz: tuple[float, float, float, float] = DEFAULT_KEYBOARD_QUAT_WXYZ
    support_box_height: float = MINI_KEYBOARD_SUPPORT_BOX_HEIGHT
    xy_randomization: float = 0.015
    xy_range: tuple[tuple[float, float], tuple[float, float]] | None = None
    yaw_range_rad: tuple[float, float] = (0.0, 0.0)
    table_clearance: float = 0.0
    target_key_joint_names: tuple[str, ...] = DEFAULT_KEYBOARD_TARGET_KEYS
    key_success_fraction: float = 0.8
    safe_radius: float = 0.16

    scene_object_kind: str = "mini_keyboard"

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "task": "keyboard_press",
            "object": "akai_mpk_mini_mk3",
            "object_xy_range": self.xy_range,
            "object_quat_wxyz": self.object_quat_wxyz,
            "object_yaw_range_rad": self.yaw_range_rad,
            "table_top_z": TABLE_TOP_Z,
            "support_box_height": self.support_box_height,
            "target_key_joint_names": self.target_key_joint_names,
            "key_success_fraction": self.key_success_fraction,
            "safe_radius": self.safe_radius,
        }

    def reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        scene: SceneSpec,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        bid = self._keyboard_body_id(model, scene)
        pos = np.asarray(self.object_pos, dtype=np.float64).copy()
        if self.xy_range is not None:
            pos[0] = rng.uniform(float(self.xy_range[0][0]), float(self.xy_range[0][1]))
            pos[1] = rng.uniform(float(self.xy_range[1][0]), float(self.xy_range[1][1]))
        elif self.xy_randomization > 0.0:
            pos[:2] += rng.uniform(-self.xy_randomization, self.xy_randomization, size=2)
        pos[2] = float(pos[2]) + float(self.table_clearance)

        yaw = _sample_range(self.yaw_range_rad, rng)
        quat = _quat_mul_wxyz(_yaw_quat_wxyz(yaw), _normalize_quat_wxyz(self.object_quat_wxyz))
        model.body_pos[bid] = pos
        model.body_quat[bid] = quat

        for qadr, dadr in self._key_qpos_qvel_adrs(model, scene):
            data.qpos[qadr] = 0.0
            data.qvel[dadr] = 0.0
        mujoco.mj_forward(model, data)
        self.ensure_above_table(model, data, scene)
        return self.observe(model, data, scene)

    def observe(self, model: mujoco.MjModel, data: mujoco.MjData, scene: SceneSpec) -> dict[str, np.ndarray]:
        bid = self._keyboard_body_id(model, scene)
        object_pos = np.asarray(data.xpos[bid], dtype=np.float64).copy()
        object_quat = _normalize_quat_wxyz(np.asarray(data.xquat[bid], dtype=np.float64).copy())
        object_pose = np.concatenate([object_pos, object_quat])
        key_joint_names = tuple(str(name) for name in getattr(scene, "mini_keyboard_key_joint_names", ()))
        key_ids = self._key_joint_ids(model, scene)
        key_slide = np.asarray([float(data.qpos[int(model.jnt_qposadr[jid])]) for jid in key_ids], dtype=np.float32)
        key_ranges = np.asarray([model.jnt_range[jid] for jid in key_ids], dtype=np.float32)
        target_ids = self._target_key_joint_ids(model)
        target_names = tuple(str(name) for name in self.target_key_joint_names)
        target_slide = np.asarray([float(data.qpos[int(model.jnt_qposadr[jid])]) for jid in target_ids], dtype=np.float32)
        target_ranges = np.asarray([model.jnt_range[jid] for jid in target_ids], dtype=np.float32)
        thresholds = target_ranges[:, 0] * float(self.key_success_fraction)
        target_success = target_slide <= thresholds
        return {
            "object_pose": object_pose.astype(np.float32),
            "object_pos": object_pose[:3].astype(np.float32),
            "object_quat_wxyz": object_pose[3:7].astype(np.float32),
            "object_yaw": np.asarray(_yaw_from_quat_wxyz(object_pose[3:7]), dtype=np.float32),
            "keyboard_key_joint_names": np.asarray(key_joint_names, dtype=object),
            "key_slide": key_slide.astype(np.float32),
            "key_slide_range": key_ranges.astype(np.float32),
            "target_key_joint_names": np.asarray(target_names, dtype=object),
            "target_key_slide": target_slide.astype(np.float32),
            "target_key_threshold": thresholds.astype(np.float32),
            "target_key_success": target_success.astype(np.bool_),
            "success": np.asarray(bool(np.all(target_success)), dtype=np.bool_),
        }

    def is_success(self, model: mujoco.MjModel, data: mujoco.MjData, scene: SceneSpec) -> bool:
        obs = self.observe(model, data, scene)
        return bool(obs["success"])

    def ensure_above_table(self, model: mujoco.MjModel, data: mujoco.MjData, scene: SceneSpec) -> float:
        bid = self._keyboard_body_id(model, scene)
        mujoco.mj_forward(model, data)
        min_z = _body_tree_geom_min_z(model, data, scene.mini_keyboard_body_name)
        target_z = float(TABLE_TOP_Z) + max(0.0, float(self.table_clearance))
        delta = max(0.0, target_z - float(min_z))
        if delta > 1.0e-9:
            model.body_pos[bid, 2] += delta
            mujoco.mj_forward(model, data)
        return float(delta)

    def _keyboard_body_id(self, model: mujoco.MjModel, scene: SceneSpec) -> int:
        bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, scene.mini_keyboard_body_name))
        if bid < 0:
            raise KeyError(f"Mini-keyboard body is missing from model: {scene.mini_keyboard_body_name}")
        return bid

    def _key_joint_ids(self, model: mujoco.MjModel, scene: SceneSpec) -> np.ndarray:
        names = tuple(str(name) for name in getattr(scene, "mini_keyboard_key_joint_names", ()))
        if not names:
            names = tuple(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
                for joint_id in range(int(model.njnt))
                if "key_" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "")
            )
        ids = []
        for name in names:
            jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
            if jid < 0:
                raise KeyError(f"Mini-keyboard key joint is missing from model: {name}")
            ids.append(jid)
        return np.asarray(ids, dtype=np.int32)

    def _target_key_joint_ids(self, model: mujoco.MjModel) -> np.ndarray:
        names = tuple(str(name) for name in self.target_key_joint_names)
        if len(names) != 3:
            raise ValueError(f"target_key_joint_names must contain exactly 3 names, got {names}")
        ids = []
        for name in names:
            jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
            if jid < 0:
                raise KeyError(f"Target mini-keyboard key joint is missing from model: {name}")
            ids.append(jid)
        if len(set(ids)) != len(ids):
            raise ValueError(f"target_key_joint_names must be distinct, got {names}")
        return np.asarray(ids, dtype=np.int32)

    def _key_qpos_qvel_adrs(self, model: mujoco.MjModel, scene: SceneSpec) -> list[tuple[int, int]]:
        out = []
        for jid in self._key_joint_ids(model, scene):
            out.append((int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])))
        return out


def _sample_range(bounds: tuple[float, float], rng: np.random.Generator) -> float:
    lo, hi = float(bounds[0]), float(bounds[1])
    if hi == lo:
        return lo
    return float(rng.uniform(lo, hi))


def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.asarray([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, dtype=np.float64).reshape(4)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64).reshape(4)
    return _normalize_quat_wxyz(
        np.asarray(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dtype=np.float64,
        )
    )


def _normalize_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-8:
        raise ValueError("quat_wxyz must be nonzero")
    return quat / norm


def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = np.asarray(quat, dtype=np.float64).reshape(4)
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _body_tree_geom_min_z(model: mujoco.MjModel, data: mujoco.MjData, root_body_name: str) -> float:
    root_bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name))
    if root_bid < 0:
        raise KeyError(f"Body is missing from model: {root_body_name}")
    min_z = float("inf")
    for gid in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[gid])
        if not _is_body_or_descendant(model, body_id, root_bid):
            continue
        min_z = min(min_z, _geom_world_min_z(model, data, gid))
    if not np.isfinite(min_z):
        raise ValueError(f"Body {root_body_name!r} does not contain any geoms")
    return float(min_z)


def _is_body_or_descendant(model: mujoco.MjModel, body_id: int, root_body_id: int) -> bool:
    current = int(body_id)
    while current >= 0:
        if current == int(root_body_id):
            return True
        parent = int(model.body_parentid[current])
        if parent == current:
            break
        current = parent
    return False


def _geom_world_min_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    geom_type = int(model.geom_type[geom_id])
    xpos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        vert_adr = int(model.mesh_vertadr[mesh_id])
        vert_num = int(model.mesh_vertnum[mesh_id])
        verts = np.asarray(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=np.float64)
        world = xpos[None, :] + verts @ xmat.T
        return float(np.min(world[:, 2]))

    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        half = size[:3]
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        half = np.asarray([size[0], size[0], size[1]], dtype=np.float64)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        half = np.full(3, size[0], dtype=np.float64)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        half = np.asarray([size[0], size[0], size[0] + size[1]], dtype=np.float64)
    else:
        return float(xpos[2] - float(model.geom_rbound[geom_id]))
    corners = np.asarray(
        [
            [-half[0], -half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], half[1], half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], -half[2]],
            [half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )
    world = xpos[None, :] + corners @ xmat.T
    return float(np.min(world[:, 2]))
