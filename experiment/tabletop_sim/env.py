"""Gymnasium-style tabletop manipulation environment."""

from __future__ import annotations

from dataclasses import dataclass
import os
from collections.abc import Callable
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .backend import RawMujocoBackend, make_backend
from .ik import IKSolverConfig, solve_arm_ik
from .perception import (
    DEFAULT_CAMERA_NAMES,
    DEFAULT_POINTCLOUD_WORKSPACE_BOUNDS,
    DEFAULT_POLICY_CAMERA_NAMES,
    CameraRenderer,
    build_pointcloud_observation,
    default_camera_specs,
)
from .robot import ResolvedRobot, resolve_robot
from .rotation import FixedBaseRotationChart
from .scene import SceneSpec, compile_tabletop_scene
from .tasks import CubePickTask, KeyboardPressTask, ObjectSuitePickPlaceTask, PumpBottleUseTask
from .viewer_debug import configure_collision_geom_debug_view


@dataclass(frozen=True)
class ActuatorGainConfig:
    kp: float | None = None
    kv: float | str | None = None
    max_effort: float | None = None


@dataclass(frozen=True)
class ActuatorControlConfig:
    arm_kp_scale: float = 1.0
    arm_kv_scale: float = 1.0
    arm_force_scale: float = 1.0
    hand_kp_scale: float = 1.0
    hand_kv_scale: float = 1.0
    hand_force_scale: float = 1.0
    base_xyz: ActuatorGainConfig | None = None
    base_rpy: ActuatorGainConfig | None = None
    fingers: ActuatorGainConfig | None = None
    armless_root_max_translation_step: float | None = 0.006
    armless_root_max_rotation_step: float | None = 0.04


class TabletopManipulationEnv(gym.Env):
    """Gymnasium-style tabletop cube manipulation environment.

    Actions are dictionaries keyed by embodiment control group. Each group action
    contains ``ee_pos`` (3), ``ee_quat_wxyz`` (4), and an embodiment-specific
    effector command such as ``hand_q`` or ``gripper``.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        *,
        backend: str = "raw",
        mode: str = "single_right",
        embodiment: str = "marvin_wuji",
        floating_rotation_reference_xyz: tuple[float, float, float] | None = None,
        frame_skip: int = 50,
        include_proprio: bool = False,
        render_mode: str | None = None,
        viewer_key_callback: Callable[[int], None] | None = None,
        gravity_compensation: bool = True,
        ik_config: IKSolverConfig | None = None,
        task: CubePickTask | ObjectSuitePickPlaceTask | PumpBottleUseTask | KeyboardPressTask | None = None,
        object_kind: str | None = None,
        timestep: float | None = None,
        cube_half_extent: float | None = None,
        cube_mass: float | None = None,
        cube_friction: tuple[float, float, float] | None = None,
        cube_texture_path: str | None = None,
        cube_face_texture_paths: dict[str, str] | None = None,
        cube_face_texture_atlas_path: str | None = None,
        cube_visual_face_offset: float | None = None,
        cube_texrepeat: tuple[float, float] | None = None,
        cube_rgba: tuple[float, float, float, float] | None = None,
        cube_specular: float | None = None,
        cube_shininess: float | None = None,
        cube_reflectance: float | None = None,
        pump_bottle_mass: float | None = None,
        pump_button_mass: float | None = None,
        pump_button_joint_range: tuple[float, float] | None = None,
        pump_button_joint_stiffness: float | None = None,
        pump_button_joint_damping: float | None = None,
        pump_button_actuator_kp: float | None = None,
        pump_button_force_site_visible: bool | None = None,
        pump_bottle_mesh_scale: float | None = None,
        pump_bottle_friction: tuple[float, float, float] | None = None,
        pump_bottle_body_collision_mode: str | None = None,
        pump_bottle_neck_collision_mode: str | None = None,
        pump_bottle_visual_mesh_source: str | None = None,
        pump_bottle_contact_margin: float | None = None,
        pump_bottle_asset_id: str | None = None,
        pump_bottle_manifest_path: str | None = None,
        mini_keyboard_friction: tuple[float, float, float] | None = None,
        mini_keyboard_support_box_height: float | None = None,
        mini_keyboard_white_key_push_distance: float | None = None,
        mini_keyboard_black_key_push_distance: float | None = None,
        include_camera_obs: bool = False,
        camera_names: tuple[str, ...] = DEFAULT_CAMERA_NAMES,
        policy_camera_names: tuple[str, ...] = DEFAULT_POLICY_CAMERA_NAMES,
        camera_width: int = 640,
        camera_height: int = 360,
        target_pointcloud_num: int = 512,
        exclude_robot_pointcloud: bool = True,
        exclude_table_pointcloud: bool = True,
        object_only_pointcloud: bool = False,
        object_mask_erosion_pixels: int = 0,
        object_depth_neighbor_threshold_m: float = 0.0,
        object_bounds_filter_tolerance_m: float | None = None,
        capture_camera_rgb: bool = True,
        capture_camera_segmentation: bool = True,
        pointcloud_workspace_bounds: np.ndarray | None = None,
        actuator_control: ActuatorControlConfig | None = None,
        enable_fingertip_collision_pads: bool = True,
        robotiq_pad1_friction: tuple[float, float, float] | None = None,
        robotiq_pad2_friction: tuple[float, float, float] | None = None,
        robotiq_pad_condim: int | None = None,
        robotiq_pad_priority: int | None = None,
        robotiq_pad_solref: tuple[float, float] | None = None,
        robotiq_pad_solimp: tuple[float, float, float, float, float] | None = None,
        allegro_hand_friction: tuple[float, float, float] | None = None,
        allegro_hand_condim: int | None = None,
        allegro_hand_priority: int | None = None,
        allegro_hand_solref: tuple[float, float] | None = None,
        allegro_hand_solimp: tuple[float, float, float, float, float] | None = None,
        allegro_hand_collision_model: str | None = None,
        allegro_fingertip_friction: tuple[float, float, float] | None = None,
        allegro_fingertip_radius: float | None = None,
        allegro_fingertip_condim: int | None = None,
        allegro_fingertip_priority: int | None = None,
        allegro_fingertip_solref: tuple[float, float] | None = None,
        allegro_fingertip_solimp: tuple[float, float, float, float, float] | None = None,
        barrett_fingertip_friction: tuple[float, float, float] | None = None,
        barrett_fingertip_collision_model: str | None = None,
        barrett_fingertip_capsule_radius: float | None = None,
        wujihand_contact_friction: tuple[float, float, float] | None = None,
        render_shadows_enabled: bool | None = None,
        world_visual_style: str | None = None,
        world_floor_rgba: tuple[float, float, float, float] | None = None,
        world_wall_rgba: tuple[float, float, float, float] | None = None,
        world_skybox_rgba: tuple[float, float, float, float] | None = None,
        table_half_extents: tuple[float, float, float] | None = None,
        table_visual_style: str | None = None,
        table_texture_path: str | None = None,
        table_texture_fallback: str | None = None,
        table_texrepeat: tuple[float, float] | None = None,
        table_rgba: tuple[float, float, float, float] | None = None,
        table_specular: float | None = None,
        table_shininess: float | None = None,
        table_reflectance: float | None = None,
        wuji_right_hand_rgba: tuple[float, float, float, float] | None = None,
        wuji_right_hand_specular: float | None = None,
        wuji_right_hand_shininess: float | None = None,
        wuji_right_hand_reflectance: float | None = None,
        viewer_show_collision_geoms: bool = False,
        viewer_collision_geom_mode: str = "none",
    ) -> None:
        super().__init__()
        if mode not in {"single_right", "bimanual"}:
            raise ValueError("mode must be one of: single_right, bimanual")
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError("render_mode must be one of: None, human, rgb_array")

        self.mode = mode
        self.embodiment = str(embodiment)
        self.floating_rotation_chart = (
            None if floating_rotation_reference_xyz is None
            else FixedBaseRotationChart(self.embodiment, floating_rotation_reference_xyz)
        )
        self.include_proprio = bool(include_proprio)
        self.render_mode = render_mode
        self.viewer_key_callback = viewer_key_callback
        self.gravity_compensation = bool(gravity_compensation)
        self.ik_config = ik_config or IKSolverConfig()
        self.task = task or CubePickTask()
        self.object_kind = str(object_kind or getattr(self.task, "scene_object_kind", "cube")).strip().lower()
        self.include_camera_obs = bool(include_camera_obs)
        self.camera_names = tuple(str(name) for name in camera_names)
        self.policy_camera_names = tuple(str(name) for name in policy_camera_names)
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.target_pointcloud_num = int(target_pointcloud_num)
        self.exclude_robot_pointcloud = bool(exclude_robot_pointcloud)
        self.exclude_table_pointcloud = bool(exclude_table_pointcloud)
        self.object_only_pointcloud = bool(object_only_pointcloud)
        self.object_mask_erosion_pixels = int(object_mask_erosion_pixels)
        if self.object_mask_erosion_pixels < 0:
            raise ValueError("object_mask_erosion_pixels must be nonnegative")
        self.object_depth_neighbor_threshold_m = float(object_depth_neighbor_threshold_m)
        if not np.isfinite(self.object_depth_neighbor_threshold_m) or self.object_depth_neighbor_threshold_m < 0.0:
            raise ValueError("object_depth_neighbor_threshold_m must be finite and nonnegative")
        self.object_bounds_filter_tolerance_m = (
            None if object_bounds_filter_tolerance_m is None else float(object_bounds_filter_tolerance_m)
        )
        if self.object_bounds_filter_tolerance_m is not None and (
            not np.isfinite(self.object_bounds_filter_tolerance_m)
            or self.object_bounds_filter_tolerance_m < 0.0
        ):
            raise ValueError("object_bounds_filter_tolerance_m must be finite and nonnegative")
        self.capture_camera_rgb = bool(capture_camera_rgb)
        self.capture_camera_segmentation = bool(capture_camera_segmentation)
        self.actuator_control = actuator_control or ActuatorControlConfig()
        collision_mode = str(viewer_collision_geom_mode).strip().lower().replace("-", "_")
        if bool(viewer_show_collision_geoms) and collision_mode == "none":
            collision_mode = "overlay"
        self.viewer_show_collision_geoms = bool(viewer_show_collision_geoms) or collision_mode != "none"
        self.viewer_collision_geom_mode = collision_mode
        self.pointcloud_workspace_bounds = np.asarray(
            DEFAULT_POINTCLOUD_WORKSPACE_BOUNDS if pointcloud_workspace_bounds is None else pointcloud_workspace_bounds,
            dtype=np.float32,
        ).reshape(3, 2)
        self.camera_specs = default_camera_specs(width=self.camera_width, height=self.camera_height)

        scene_kwargs = {
            "add_cameras": True,
            "embodiment": self.embodiment,
            "floating_rotation_reference_xyz": floating_rotation_reference_xyz,
            "object_kind": self.object_kind,
            "camera_width": self.camera_width,
            "camera_height": self.camera_height,
            "enable_fingertip_collision_pads": bool(enable_fingertip_collision_pads),
        }
        if isinstance(self.task, ObjectSuitePickPlaceTask):
            object_spec = self.task.object_spec
            scene_kwargs.update(
                object_visual_mesh_path=str(object_spec.visual_mesh_path),
                object_collision_mesh_path=str(object_spec.collision_mesh_path),
                object_collision_mesh_paths=tuple(
                    str(path) for path in object_spec.effective_collision_mesh_paths
                ),
                object_texture_path=str(object_spec.texture_path),
                object_mass=float(object_spec.mass_kg),
                object_friction=tuple(float(x) for x in object_spec.friction),
                object_mesh_min_z=float(object_spec.min_bound[2]),
                bin_friction=tuple(float(x) for x in self.task.bin_friction),
                bin_center_xy=tuple(float(x) for x in self.task.bin_center_xy),
                bin_inner_size=tuple(float(x) for x in self.task.bin_inner_size),
                bin_wall_height=float(self.task.bin_wall_height),
                bin_wall_thickness=float(self.task.bin_wall_thickness),
            )
        if timestep is not None:
            scene_kwargs["timestep"] = float(timestep)
        if robotiq_pad1_friction is not None:
            scene_kwargs["robotiq_pad1_friction"] = tuple(float(x) for x in robotiq_pad1_friction)
        if robotiq_pad2_friction is not None:
            scene_kwargs["robotiq_pad2_friction"] = tuple(float(x) for x in robotiq_pad2_friction)
        if robotiq_pad_condim is not None:
            scene_kwargs["robotiq_pad_condim"] = int(robotiq_pad_condim)
        if robotiq_pad_priority is not None:
            scene_kwargs["robotiq_pad_priority"] = int(robotiq_pad_priority)
        if robotiq_pad_solref is not None:
            scene_kwargs["robotiq_pad_solref"] = tuple(float(x) for x in robotiq_pad_solref)
        if robotiq_pad_solimp is not None:
            scene_kwargs["robotiq_pad_solimp"] = tuple(float(x) for x in robotiq_pad_solimp)
        if allegro_hand_friction is not None:
            scene_kwargs["allegro_hand_friction"] = tuple(float(x) for x in allegro_hand_friction)
        if allegro_hand_condim is not None:
            scene_kwargs["allegro_hand_condim"] = int(allegro_hand_condim)
        if allegro_hand_priority is not None:
            scene_kwargs["allegro_hand_priority"] = int(allegro_hand_priority)
        if allegro_hand_solref is not None:
            scene_kwargs["allegro_hand_solref"] = tuple(float(x) for x in allegro_hand_solref)
        if allegro_hand_solimp is not None:
            scene_kwargs["allegro_hand_solimp"] = tuple(float(x) for x in allegro_hand_solimp)
        if allegro_hand_collision_model is not None:
            scene_kwargs["allegro_hand_collision_model"] = str(allegro_hand_collision_model)
        if allegro_fingertip_friction is not None:
            scene_kwargs["allegro_fingertip_friction"] = tuple(
                float(x) for x in allegro_fingertip_friction
            )
        if allegro_fingertip_radius is not None:
            scene_kwargs["allegro_fingertip_radius"] = float(allegro_fingertip_radius)
        if allegro_fingertip_condim is not None:
            scene_kwargs["allegro_fingertip_condim"] = int(allegro_fingertip_condim)
        if allegro_fingertip_priority is not None:
            scene_kwargs["allegro_fingertip_priority"] = int(allegro_fingertip_priority)
        if allegro_fingertip_solref is not None:
            scene_kwargs["allegro_fingertip_solref"] = tuple(float(x) for x in allegro_fingertip_solref)
        if allegro_fingertip_solimp is not None:
            scene_kwargs["allegro_fingertip_solimp"] = tuple(float(x) for x in allegro_fingertip_solimp)
        if barrett_fingertip_friction is not None:
            scene_kwargs["barrett_fingertip_friction"] = tuple(
                float(x) for x in barrett_fingertip_friction
            )
        if barrett_fingertip_collision_model is not None:
            scene_kwargs["barrett_fingertip_collision_model"] = str(barrett_fingertip_collision_model)
        if barrett_fingertip_capsule_radius is not None:
            scene_kwargs["barrett_fingertip_capsule_radius"] = float(barrett_fingertip_capsule_radius)
        if wujihand_contact_friction is not None:
            scene_kwargs["wujihand_contact_friction"] = tuple(
                float(x) for x in wujihand_contact_friction
            )
        if render_shadows_enabled is not None:
            scene_kwargs["render_shadows_enabled"] = bool(render_shadows_enabled)
        if world_visual_style is not None:
            scene_kwargs["world_visual_style"] = str(world_visual_style)
        if world_floor_rgba is not None:
            scene_kwargs["world_floor_rgba"] = tuple(float(x) for x in world_floor_rgba)
        if world_wall_rgba is not None:
            scene_kwargs["world_wall_rgba"] = tuple(float(x) for x in world_wall_rgba)
        if world_skybox_rgba is not None:
            scene_kwargs["world_skybox_rgba"] = tuple(float(x) for x in world_skybox_rgba)
        if table_half_extents is not None:
            scene_kwargs["table_half_extents"] = tuple(float(x) for x in table_half_extents)
        if table_visual_style is not None:
            scene_kwargs["table_visual_style"] = str(table_visual_style)
        if table_texture_path is not None:
            scene_kwargs["table_texture_path"] = str(table_texture_path)
        if table_texture_fallback is not None:
            scene_kwargs["table_texture_fallback"] = str(table_texture_fallback)
        if table_texrepeat is not None:
            scene_kwargs["table_texrepeat"] = tuple(float(x) for x in table_texrepeat)
        if table_rgba is not None:
            scene_kwargs["table_rgba"] = tuple(float(x) for x in table_rgba)
        if table_specular is not None:
            scene_kwargs["table_specular"] = float(table_specular)
        if table_shininess is not None:
            scene_kwargs["table_shininess"] = float(table_shininess)
        if table_reflectance is not None:
            scene_kwargs["table_reflectance"] = float(table_reflectance)
        if wuji_right_hand_rgba is not None:
            scene_kwargs["wuji_right_hand_rgba"] = tuple(float(x) for x in wuji_right_hand_rgba)
        if wuji_right_hand_specular is not None:
            scene_kwargs["wuji_right_hand_specular"] = float(wuji_right_hand_specular)
        if wuji_right_hand_shininess is not None:
            scene_kwargs["wuji_right_hand_shininess"] = float(wuji_right_hand_shininess)
        if wuji_right_hand_reflectance is not None:
            scene_kwargs["wuji_right_hand_reflectance"] = float(wuji_right_hand_reflectance)
        if cube_half_extent is not None:
            scene_kwargs["cube_half_extent"] = float(cube_half_extent)
        if cube_mass is not None:
            scene_kwargs["cube_mass"] = float(cube_mass)
        if cube_friction is not None:
            scene_kwargs["cube_friction"] = tuple(float(x) for x in cube_friction)
        if cube_texture_path is not None:
            scene_kwargs["cube_texture_path"] = str(cube_texture_path)
        if cube_face_texture_paths is not None:
            scene_kwargs["cube_face_texture_paths"] = {
                str(face): str(path) for face, path in cube_face_texture_paths.items()
            }
        if cube_face_texture_atlas_path is not None:
            scene_kwargs["cube_face_texture_atlas_path"] = str(cube_face_texture_atlas_path)
        if cube_visual_face_offset is not None:
            scene_kwargs["cube_visual_face_offset"] = float(cube_visual_face_offset)
        if cube_texrepeat is not None:
            scene_kwargs["cube_texrepeat"] = tuple(float(x) for x in cube_texrepeat)
        if cube_rgba is not None:
            scene_kwargs["cube_rgba"] = tuple(float(x) for x in cube_rgba)
        if cube_specular is not None:
            scene_kwargs["cube_specular"] = float(cube_specular)
        if cube_shininess is not None:
            scene_kwargs["cube_shininess"] = float(cube_shininess)
        if cube_reflectance is not None:
            scene_kwargs["cube_reflectance"] = float(cube_reflectance)
        if pump_bottle_mass is not None:
            scene_kwargs["pump_bottle_mass"] = float(pump_bottle_mass)
        if pump_button_mass is not None:
            scene_kwargs["pump_button_mass"] = float(pump_button_mass)
        if pump_button_joint_range is not None:
            scene_kwargs["pump_button_joint_range"] = tuple(float(x) for x in pump_button_joint_range)
        if pump_button_joint_stiffness is not None:
            scene_kwargs["pump_button_joint_stiffness"] = float(pump_button_joint_stiffness)
        if pump_button_joint_damping is not None:
            scene_kwargs["pump_button_joint_damping"] = float(pump_button_joint_damping)
        if pump_button_actuator_kp is not None:
            scene_kwargs["pump_button_actuator_kp"] = float(pump_button_actuator_kp)
        if pump_button_force_site_visible is not None:
            scene_kwargs["pump_button_force_site_visible"] = bool(pump_button_force_site_visible)
        if pump_bottle_mesh_scale is not None:
            scene_kwargs["pump_bottle_mesh_scale"] = float(pump_bottle_mesh_scale)
        if pump_bottle_friction is not None:
            scene_kwargs["pump_bottle_friction"] = tuple(float(x) for x in pump_bottle_friction)
        if pump_bottle_body_collision_mode is not None:
            scene_kwargs["pump_bottle_body_collision_mode"] = str(pump_bottle_body_collision_mode)
        if pump_bottle_neck_collision_mode is not None:
            scene_kwargs["pump_bottle_neck_collision_mode"] = str(pump_bottle_neck_collision_mode)
        if pump_bottle_visual_mesh_source is not None:
            scene_kwargs["pump_bottle_visual_mesh_source"] = str(pump_bottle_visual_mesh_source)
        if pump_bottle_contact_margin is not None:
            scene_kwargs["pump_bottle_contact_margin"] = float(pump_bottle_contact_margin)
        if pump_bottle_asset_id is not None:
            scene_kwargs["pump_bottle_asset_id"] = str(pump_bottle_asset_id)
        if pump_bottle_manifest_path is not None:
            scene_kwargs["pump_bottle_manifest_path"] = str(pump_bottle_manifest_path)
        if mini_keyboard_friction is not None:
            scene_kwargs["mini_keyboard_friction"] = tuple(float(x) for x in mini_keyboard_friction)
        if mini_keyboard_support_box_height is not None:
            scene_kwargs["mini_keyboard_support_box_height"] = float(mini_keyboard_support_box_height)
        if mini_keyboard_white_key_push_distance is not None:
            scene_kwargs["mini_keyboard_white_key_push_distance"] = float(mini_keyboard_white_key_push_distance)
        if mini_keyboard_black_key_push_distance is not None:
            scene_kwargs["mini_keyboard_black_key_push_distance"] = float(mini_keyboard_black_key_push_distance)
        self.spec, model, data, self.scene = compile_tabletop_scene(**scene_kwargs)
        self.backend: RawMujocoBackend = make_backend(backend, model=model, data=data, frame_skip=frame_skip)
        self.model = self.backend.model
        self.data = self.backend.data
        self.robot: ResolvedRobot = resolve_robot(self.model, self.embodiment)
        self._apply_actuator_control_config()
        if self.gravity_compensation:
            self.backend.set_gravity_compensation_dofs(
                np.concatenate([self.robot.side(name).qvel_indices for name in self.robot.group_names])
            )
        self.robot.apply_home(self.data)
        self.backend.forward()

        self._viewer = None
        self._renderer = None
        self._camera_renderer: CameraRenderer | None = None
        self._rng = np.random.default_rng()

        self.action_space = spaces.Dict(
            {name: self._side_action_space(self.robot.side(name)) for name in self.robot.group_names}
        )
        self.observation_space = self._observation_space()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.robot.apply_home(self.data)
        obs = self.task.reset(self.model, self.data, self.scene, self._rng)
        self.backend.forward()
        obs = self.observe()
        info = {
            "task": self.task.metadata,
            "backend": self.backend.backend_name,
            "mode": self.mode,
            "embodiment": self.embodiment,
            "gravity_compensation": self.gravity_compensation,
            "include_camera_obs": self.include_camera_obs,
            "camera_names": self.camera_names,
            "policy_camera_names": self.policy_camera_names,
            "camera_width": self.camera_width,
            "camera_height": self.camera_height,
            "capture_camera_rgb": self.capture_camera_rgb,
            "capture_camera_segmentation": self.capture_camera_segmentation,
            "actuator_control": self.actuator_control,
        }
        if self.render_mode == "human":
            self.render()
        return obs, info

    def observe(self, *, include_camera_obs: bool | None = None) -> dict[str, Any]:
        obs = self._state_observation()
        capture_camera = self.include_camera_obs if include_camera_obs is None else bool(include_camera_obs)
        if capture_camera:
            sensor_data, sensor_param = self._camera_capture()
            obs["sensor_data"] = sensor_data
            obs["sensor_param"] = sensor_param
            obs["camera_names"] = self.camera_names
            obs["policy_camera_names"] = self.policy_camera_names
            object_pose_for_filter = None
            object_bounds_for_filter = None
            if self.object_only_pointcloud and self.object_bounds_filter_tolerance_m is not None:
                object_spec = getattr(self.task, "object_spec", None)
                if object_spec is None or "object_pose" not in obs:
                    raise RuntimeError(
                        "Object-bounds point-cloud filtering requires task.object_spec and obs.object_pose"
                    )
                object_pose_for_filter = np.asarray(obs["object_pose"], dtype=np.float32)
                object_bounds_for_filter = np.asarray(object_spec.mesh_bounds_m, dtype=np.float32)
            obs["pointcloud"] = build_pointcloud_observation(
                sensor_data,
                sensor_param,
                policy_camera_names=self.policy_camera_names,
                target_num=self.target_pointcloud_num,
                exclude_robot=self.exclude_robot_pointcloud,
                exclude_table=self.exclude_table_pointcloud,
                object_only=self.object_only_pointcloud,
                object_mask_erosion_pixels=self.object_mask_erosion_pixels,
                object_depth_neighbor_threshold_m=self.object_depth_neighbor_threshold_m,
                object_pose_wxyz=object_pose_for_filter,
                object_bounds_m=object_bounds_for_filter,
                object_bounds_filter_tolerance_m=self.object_bounds_filter_tolerance_m,
                workspace_bounds=self.pointcloud_workspace_bounds,
                rng=self._rng,
            )
        return obs

    def _state_observation(self) -> dict[str, np.ndarray]:
        obs = self.task.observe(self.model, self.data, self.scene)
        if self.include_proprio:
            obs.update(self.robot.proprio(self.data))
        return obs

    def step(self, action: dict[str, Any], *, include_camera_obs: bool | None = None):
        controlled_sides = self._controlled_sides()
        pre_obs = self._state_observation()
        pre_qpos = self.data.qpos.copy()
        pre_ctrl = self.data.ctrl.copy()
        if self.mode == "single_right" and self.robot.has_group("left"):
            self.robot.apply_home(self.data, sides=("left",), set_qpos=True)

        commands: dict[str, dict[str, np.ndarray]] = {}
        for side in controlled_sides:
            if side not in action:
                raise KeyError(f"Action for side {side!r} is required in mode {self.mode!r}")
            commands[side] = self._apply_side_action(side, action[side])

        target_qpos = self.data.qpos.copy()
        for side, command in commands.items():
            side_spec = self.robot.side(side)
            target_qpos[side_spec.qpos_indices] = np.concatenate(
                [command["arm_q_target"], command["effector_q_target"]]
            )
        target_ctrl = self.data.ctrl.copy()

        self.backend.step()
        if self.mode == "single_right" and self.robot.has_group("left"):
            self.robot.apply_home(self.data, sides=("left",), set_qpos=True)
            self.backend.forward()
        self._clamp_task_joint_limits()

        obs = self.observe(include_camera_obs=include_camera_obs)
        success = bool(obs["success"])
        reward = 1.0 if success else 0.0
        terminated = success
        truncated = False
        info = {
            "task": self.task.metadata,
            "backend": self.backend.backend_name,
            "mode": self.mode,
            "embodiment": self.embodiment,
            "gravity_compensation": self.gravity_compensation,
            "command": {
                "controlled_sides": tuple(controlled_sides),
                "sides": commands,
                "pre_qpos": pre_qpos.astype(np.float32),
                "pre_ctrl": pre_ctrl.astype(np.float32),
                "target_qpos": target_qpos.astype(np.float32),
                "target_ctrl": target_ctrl.astype(np.float32),
                "executed_qpos": self.data.qpos.astype(np.float32).copy(),
                "executed_qvel": self.data.qvel.astype(np.float32).copy(),
                "executed_ctrl": self.data.ctrl.astype(np.float32).copy(),
            },
        }
        before_pose = _primary_object_pose(pre_obs)
        after_pose = _primary_object_pose(obs)
        if before_pose is not None and after_pose is not None:
            info["command"]["object_pose_before"] = before_pose.astype(np.float32).copy()
            info["command"]["object_pose_after"] = after_pose.astype(np.float32).copy()
        if "cube_pose" in pre_obs and "cube_pose" in obs:
            info["command"]["cube_pose_before"] = np.asarray(pre_obs["cube_pose"], dtype=np.float32).copy()
            info["command"]["cube_pose_after"] = np.asarray(obs["cube_pose"], dtype=np.float32).copy()
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _clamp_task_joint_limits(self) -> bool:
        if not isinstance(self.task, PumpBottleUseTask):
            return False

        button_qadr, button_dadr = self.task._button_qpos_qvel_adr(self.model, self.scene)  # noqa: SLF001
        lo, hi = self.task._button_range(self.model, self.scene)  # noqa: SLF001
        button_slide = float(self.data.qpos[button_qadr])
        clipped_slide = float(np.clip(button_slide, lo, hi))
        if clipped_slide == button_slide:
            return False

        self.data.qpos[button_qadr] = clipped_slide
        self.data.qvel[button_dadr] = 0.0
        actuator_id = self.task._button_actuator_id(self.model)  # noqa: SLF001
        if actuator_id >= 0 and bool(self.model.actuator_ctrllimited[actuator_id]):
            ctrl_lo, ctrl_hi = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = float(np.clip(self.data.ctrl[actuator_id], ctrl_lo, ctrl_hi))
        self.backend.forward()
        return True

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer

                self._viewer = mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                    key_callback=self.viewer_key_callback,
                )
                configure_collision_geom_debug_view(self.model, self._viewer, mode=self.viewer_collision_geom_mode)
            self._viewer.sync()
            return None

        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera=self.camera_names[0] if self.camera_names else -1)
            return self._renderer.render()
        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._camera_renderer is not None:
            self._camera_renderer.close()
            self._camera_renderer = None
        self.backend.close()

    def _apply_actuator_control_config(self) -> None:
        cfg = self.actuator_control
        for label, value in (
            ("controller.arm_kp_scale", cfg.arm_kp_scale),
            ("controller.arm_kv_scale", cfg.arm_kv_scale),
            ("controller.arm_force_scale", cfg.arm_force_scale),
            ("controller.hand_kp_scale", cfg.hand_kp_scale),
            ("controller.hand_kv_scale", cfg.hand_kv_scale),
            ("controller.hand_force_scale", cfg.hand_force_scale),
        ):
            if float(value) <= 0.0:
                raise ValueError(f"{label} must be positive, got {value}")

        for side_name in self.robot.group_names:
            side = self.robot.side(side_name)
            if cfg.base_xyz is not None or cfg.base_rpy is not None:
                if int(side.arm_actuator_ids.size) < 6:
                    raise ValueError(
                        "controller.base_xyz/base_rpy require at least six arm/root actuators; "
                        f"{self.embodiment}:{side_name} has {int(side.arm_actuator_ids.size)}"
                    )
                base_xyz_ids = side.arm_actuator_ids[:3]
                base_rpy_ids = side.arm_actuator_ids[3:6]
                extra_arm_ids = side.arm_actuator_ids[6:]
                if cfg.base_xyz is None:
                    _scale_position_actuators(
                        self.model,
                        base_xyz_ids,
                        kp_scale=float(cfg.arm_kp_scale),
                        kv_scale=float(cfg.arm_kv_scale),
                        force_scale=float(cfg.arm_force_scale),
                    )
                else:
                    _set_position_actuators(
                        self.model,
                        base_xyz_ids,
                        cfg.base_xyz,
                        label="controller.base_xyz",
                    )
                if cfg.base_rpy is None:
                    _scale_position_actuators(
                        self.model,
                        base_rpy_ids,
                        kp_scale=float(cfg.arm_kp_scale),
                        kv_scale=float(cfg.arm_kv_scale),
                        force_scale=float(cfg.arm_force_scale),
                    )
                else:
                    _set_position_actuators(
                        self.model,
                        base_rpy_ids,
                        cfg.base_rpy,
                        label="controller.base_rpy",
                    )
                _scale_position_actuators(
                    self.model,
                    extra_arm_ids,
                    kp_scale=float(cfg.arm_kp_scale),
                    kv_scale=float(cfg.arm_kv_scale),
                    force_scale=float(cfg.arm_force_scale),
                )
            else:
                _scale_position_actuators(
                    self.model,
                    side.arm_actuator_ids,
                    kp_scale=float(cfg.arm_kp_scale),
                    kv_scale=float(cfg.arm_kv_scale),
                    force_scale=float(cfg.arm_force_scale),
                )
            if cfg.fingers is None:
                _scale_position_actuators(
                    self.model,
                    side.hand_actuator_ids,
                    kp_scale=float(cfg.hand_kp_scale),
                    kv_scale=float(cfg.hand_kv_scale),
                    force_scale=float(cfg.hand_force_scale),
                )
            else:
                _set_position_actuators(
                    self.model,
                    side.hand_actuator_ids,
                    cfg.fingers,
                    label="controller.fingers",
                )

    def viewer_is_running(self) -> bool:
        return self._viewer is not None and bool(self._viewer.is_running())

    def set_viewer_text(self, title: str, detail: str = "") -> None:
        if self._viewer is None or not self._viewer.is_running():
            return
        self._viewer.set_texts((None, None, title, detail))

    def _controlled_sides(self) -> tuple[str, ...]:
        if self.mode == "single_right":
            if not self.robot.has_group("right"):
                raise ValueError(f"Embodiment {self.embodiment!r} has no 'right' control group")
            return ("right",)
        return self.robot.group_names

    def _side_action_space(self, side_spec) -> spaces.Dict:
        entries: dict[str, spaces.Space] = {
            "ee_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "ee_quat_wxyz": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        }
        if side_spec.effector_action == "gripper":
            entries["gripper"] = spaces.Box(
                low=side_spec.hand_ctrl_range[:, 0].astype(np.float32),
                high=side_spec.hand_ctrl_range[:, 1].astype(np.float32),
                dtype=np.float32,
            )
        elif side_spec.effector_ctrl_dim:
            entries["hand_q"] = spaces.Box(
                low=side_spec.hand_ctrl_range[:, 0].astype(np.float32),
                high=side_spec.hand_ctrl_range[:, 1].astype(np.float32),
                dtype=np.float32,
            )
        return spaces.Dict(entries)

    def _observation_space(self) -> spaces.Dict:
        if self.object_kind == "pump_bottle":
            return spaces.Dict(
                {
                    "object_pose": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
                    "object_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                    "object_quat_wxyz": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                    "button_slide": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                    "lift_success": spaces.Box(low=0, high=1, shape=(), dtype=np.bool_),
                    "button_success": spaces.Box(low=0, high=1, shape=(), dtype=np.bool_),
                    "success": spaces.Box(low=0, high=1, shape=(), dtype=np.bool_),
                }
            )
        if self.object_kind == "mini_keyboard":
            return spaces.Dict(
                {
                    "object_pose": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
                    "object_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                    "object_quat_wxyz": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                    "key_slide": spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32),
                    "key_slide_range": spaces.Box(low=-np.inf, high=np.inf, shape=(25, 2), dtype=np.float32),
                    "target_key_slide": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                    "target_key_threshold": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                    "target_key_success": spaces.Box(low=0, high=1, shape=(3,), dtype=np.bool_),
                    "success": spaces.Box(low=0, high=1, shape=(), dtype=np.bool_),
                }
            )
        if self.object_kind == "ycb_object":
            return spaces.Dict(
                {
                    "object_pose": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
                    "object_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                    "object_quat_wxyz": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                    "object_velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
                    "object_bounds_world": spaces.Box(low=-np.inf, high=np.inf, shape=(2, 3), dtype=np.float32),
                    "bin_pose": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
                    "placement_margin": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                    "success": spaces.Box(low=0, high=1, shape=(), dtype=np.bool_),
                }
            )
        return spaces.Dict(
            {
                "cube_pose": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
                "cube_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "cube_quat_wxyz": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                "cube_side_length": spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),
                "cube_height": spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),
                "success": spaces.Box(low=0, high=1, shape=(), dtype=np.bool_),
            }
        )

    def _camera_capture(self) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
        if self._camera_renderer is None:
            self._camera_renderer = CameraRenderer(self.model, camera_specs=self.camera_specs)
        return self._camera_renderer.capture(
            self.data,
            self.scene,
            self.camera_names,
            capture_rgb=self.capture_camera_rgb,
            capture_segmentation=self.capture_camera_segmentation,
        )

    def _apply_side_action(self, side: str, side_action: dict[str, Any]) -> dict[str, np.ndarray]:
        side_spec = self.robot.side(side)
        cur_pos, cur_quat = self.robot.wrist_pose(self.data, side)
        ee_pos = np.asarray(side_action.get("ee_pos", cur_pos), dtype=np.float64).reshape(3)
        ee_quat = np.asarray(side_action.get("ee_quat_wxyz", cur_quat), dtype=np.float64).reshape(4)
        quat_norm = float(np.linalg.norm(ee_quat))
        if quat_norm <= 1e-8:
            raise ValueError(f"{side} ee_quat_wxyz must be nonzero")
        ee_quat = ee_quat / quat_norm
        if "arm_q" in side_action:
            arm_q = np.asarray(side_action["arm_q"], dtype=np.float64).reshape(-1)
            if arm_q.shape != (side_spec.arm_dim,):
                raise ValueError(f"{side} arm_q must have shape {(side_spec.arm_dim,)}, got {arm_q.shape}")
            arm_q = np.clip(arm_q, side_spec.arm_ctrl_range[:, 0], side_spec.arm_ctrl_range[:, 1])
        else:
            arm_q = solve_arm_ik(
                self.model,
                self.data,
                side_spec,
                ee_pos,
                ee_quat,
                config=self.ik_config,
            )

        effector_q_target = np.asarray(self.data.qpos[side_spec.hand_qpos_indices], dtype=np.float64).copy()
        command: dict[str, np.ndarray] = {
            "ee_pos_target": ee_pos.astype(np.float32),
            "ee_quat_wxyz_target": ee_quat.astype(np.float32),
            "arm_q_target": arm_q.astype(np.float32),
            "effector_q_target": effector_q_target.astype(np.float32),
        }
        if side_spec.effector_action == "gripper":
            gripper = np.asarray(
                side_action.get("gripper", side_action.get("gripper_ctrl", side_spec.hand_home_ctrl)),
                dtype=np.float64,
            ).reshape(-1)
            if gripper.shape == ():
                gripper = gripper.reshape(1)
            if gripper.shape != (side_spec.effector_ctrl_dim,):
                raise ValueError(
                    f"{side} gripper must have shape {(side_spec.effector_ctrl_dim,)}, got {gripper.shape}"
                )
            gripper = np.clip(gripper, side_spec.hand_ctrl_range[:, 0], side_spec.hand_ctrl_range[:, 1])
            self.robot.set_side_ctrl(self.data, side, arm_q=arm_q, gripper=gripper)
            command["gripper_target"] = gripper.astype(np.float32)
            command["effector_ctrl_target"] = gripper.astype(np.float32)
            return command

        hand_q = np.asarray(side_action.get("hand_q", side_spec.hand_home_ctrl), dtype=np.float64).reshape(-1)
        if hand_q.shape == (26,) and side_spec.effector_ctrl_dim == 20:
            hand_q = hand_q[6:]
        if hand_q.shape != (side_spec.effector_ctrl_dim,):
            raise ValueError(
                f"{side} hand_q must have shape {(side_spec.effector_ctrl_dim,)}, got {hand_q.shape}"
            )
        hand_q = np.clip(hand_q, side_spec.hand_ctrl_range[:, 0], side_spec.hand_ctrl_range[:, 1])
        self.robot.set_side_ctrl(self.data, side, arm_q=arm_q, hand_q=hand_q)
        if side_spec.effector_qpos_dim == side_spec.effector_ctrl_dim:
            command["effector_q_target"] = hand_q.astype(np.float32)
        command["hand_q_target"] = hand_q.astype(np.float32)
        command["effector_ctrl_target"] = hand_q.astype(np.float32)
        return command


def _scale_position_actuators(
    model: mujoco.MjModel,
    actuator_ids: np.ndarray,
    *,
    kp_scale: float,
    kv_scale: float,
    force_scale: float,
) -> None:
    if int(actuator_ids.size) == 0:
        return
    ids = np.asarray(actuator_ids, dtype=np.int32).reshape(-1)
    model.actuator_gainprm[ids, 0] *= float(kp_scale)
    model.actuator_biasprm[ids, 1] *= float(kp_scale)
    model.actuator_biasprm[ids, 2] *= float(kv_scale)
    model.actuator_forcerange[ids, :] *= float(force_scale)


def _set_position_actuators(
    model: mujoco.MjModel,
    actuator_ids: np.ndarray,
    config: ActuatorGainConfig,
    *,
    label: str,
) -> None:
    ids = np.asarray(actuator_ids, dtype=np.int32).reshape(-1)
    if int(ids.size) == 0:
        return
    kp = _optional_positive_float(config.kp, f"{label}.kp")
    kv = _resolve_kv(config.kv, kp=kp, current_kp=model.actuator_gainprm[ids, 0], label=f"{label}.kv")
    max_effort = _optional_positive_float(config.max_effort, f"{label}.max_effort")
    if kp is not None:
        model.actuator_gainprm[ids, 0] = float(kp)
        model.actuator_biasprm[ids, 1] = -float(kp)
    if kv is not None:
        model.actuator_biasprm[ids, 2] = -kv
    if max_effort is not None:
        effort = float(max_effort)
        model.actuator_forcerange[ids, 0] = -effort
        model.actuator_forcerange[ids, 1] = effort
        model.actuator_forcelimited[ids] = 1


def _resolve_kv(
    value: float | str | None,
    *,
    kp: float | None,
    current_kp: np.ndarray,
    label: str,
) -> float | np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "auto":
        kp_source = np.asarray(float(kp) if kp is not None else current_kp, dtype=np.float64)
        if np.any(kp_source <= 0.0):
            raise ValueError(f"{label}=auto requires positive kp, got {kp_source}")
        return 2.0 * np.sqrt(kp_source)
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"{label} must be positive or 'auto', got {value}")
    return parsed


def _optional_positive_float(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"{label} must be positive, got {value}")
    return parsed


def _primary_object_pose(obs: dict[str, Any]) -> np.ndarray | None:
    if "object_pose" in obs:
        return np.asarray(obs["object_pose"], dtype=np.float32)
    if "cube_pose" in obs:
        return np.asarray(obs["cube_pose"], dtype=np.float32)
    return None
