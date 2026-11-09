"""Floating Robotiq+stick tabletop environment.

This adapter is intentionally smaller than :mod:`experiment.tabletop_sim.env`.
It keeps the task object in MuJoCo physics, while the 13D
``robotiq_gripper_stick`` q target drives two independent 6D actuated roots and
the Robotiq finger joints through MuJoCo position actuators.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from experiment.robotiq_stick_start_pose import (
    ROBOTIQ_STICK_LOCAL_DOF,
    validate_robotiq_stick_q,
)

from .backend import RawMujocoBackend, make_backend
from .perception import (
    DEFAULT_CAMERA_NAMES,
    DEFAULT_POINTCLOUD_WORKSPACE_BOUNDS,
    DEFAULT_POLICY_CAMERA_NAMES,
    CameraRenderer,
    build_pointcloud_observation,
    default_camera_specs,
)
from .scene import (
    ARMLESS_GRIPPER_ROOT_JOINT_NAMES,
    ARMLESS_ROBOTIQ_INTERNAL_JOINT_NAMES,
    ARMLESS_ROBOTIQ_PAD_FRICTION,
    ARMLESS_ROBOTIQ_PAD_PATCH_GRID,
    ARMLESS_STICK_ROOT_JOINT_NAMES,
    compile_tabletop_scene,
)
from .tasks import KeyboardPressTask, PumpBottleUseTask
from .viewer_debug import configure_collision_geom_debug_view


ROBOTIQ_INTERNAL_JOINT_NAMES = ARMLESS_ROBOTIQ_INTERNAL_JOINT_NAMES
ROBOTIQ_CLOSED_JOINT_VALUE = 0.824


@dataclass(frozen=True)
class _ArmlessSpec:
    key: str = "robotiq_gripper_stick_armless"


@dataclass(frozen=True)
class _ArmlessControllerConfig:
    arm_kp_scale: float = 1.0
    arm_kv_scale: float = 1.0
    arm_force_scale: float = 1.0
    hand_kp_scale: float = 1.0
    hand_kv_scale: float = 1.0
    hand_force_scale: float = 1.0
    base_xyz: Any | None = None
    base_rpy: Any | None = None
    fingers: Any | None = None
    armless_root_max_translation_step: float | None = 0.006
    armless_root_max_rotation_step: float | None = 0.04


class _ArmlessRobot:
    spec = _ArmlessSpec()
    group_names: tuple[str, ...] = ()

    def side(self, side: str) -> Any:
        raise ValueError(f"Armless robotiq+stick has no arm control side {side!r}")

    def has_group(self, side: str) -> bool:
        return str(side) in self.group_names


class ArmlessRobotiqStickEnv:
    """MuJoCo tabletop scene controlled by a 13D floating gripper+stick q."""

    def __init__(
        self,
        *,
        backend: str = "raw",
        mode: str = "floating",
        embodiment: str = "robotiq_gripper_stick_armless",
        frame_skip: int = 25,
        render_mode: str | None = None,
        viewer_key_callback: Callable[[int], None] | None = None,
        task: PumpBottleUseTask | KeyboardPressTask | None = None,
        gravity_compensation: bool = True,
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
        gripper_pad_friction: tuple[float, float, float] | None = None,
        gripper_pad_patch_grid: tuple[int, int] | None = None,
        enable_gripper_stick_collision: bool = False,
        include_camera_obs: bool = False,
        camera_names: tuple[str, ...] = DEFAULT_CAMERA_NAMES,
        policy_camera_names: tuple[str, ...] = DEFAULT_POLICY_CAMERA_NAMES,
        camera_width: int = 640,
        camera_height: int = 360,
        table_half_extents: tuple[float, float, float] | None = None,
        render_shadows_enabled: bool | None = None,
        world_visual_style: str | None = None,
        world_floor_rgba: tuple[float, float, float, float] | None = None,
        world_wall_rgba: tuple[float, float, float, float] | None = None,
        world_skybox_rgba: tuple[float, float, float, float] | None = None,
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
        target_pointcloud_num: int = 512,
        exclude_robot_pointcloud: bool = True,
        exclude_table_pointcloud: bool = True,
        capture_camera_rgb: bool = False,
        capture_camera_segmentation: bool = True,
        pointcloud_workspace_bounds: np.ndarray | None = None,
        actuator_control: Any | None = None,
        viewer_key_callback_label: str | None = None,
        viewer_show_collision_geoms: bool = False,
        viewer_collision_geom_mode: str = "none",
    ) -> None:
        if str(mode) != "floating":
            raise ValueError("robotiq_gripper_stick_armless expects env.mode='floating'")
        if str(embodiment) != "robotiq_gripper_stick_armless":
            raise ValueError(
                "ArmlessRobotiqStickEnv expects embodiment='robotiq_gripper_stick_armless'"
            )
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError("render_mode must be one of: None, human, rgb_array")

        self.mode = str(mode)
        self.embodiment = str(embodiment)
        self.include_proprio = True
        self.render_mode = render_mode
        self.viewer_key_callback = viewer_key_callback
        self.viewer_key_callback_label = viewer_key_callback_label
        collision_mode = str(viewer_collision_geom_mode).strip().lower().replace("-", "_")
        if bool(viewer_show_collision_geoms) and collision_mode == "none":
            collision_mode = "overlay"
        self.viewer_show_collision_geoms = bool(viewer_show_collision_geoms) or collision_mode != "none"
        self.viewer_collision_geom_mode = collision_mode
        self.gravity_compensation = bool(gravity_compensation)
        self.enable_gripper_stick_collision = bool(enable_gripper_stick_collision)
        pad_friction = (
            ARMLESS_ROBOTIQ_PAD_FRICTION
            if gripper_pad_friction is None
            else gripper_pad_friction
        )
        self.gripper_pad_friction = tuple(float(x) for x in pad_friction)
        pad_patch_grid = (
            ARMLESS_ROBOTIQ_PAD_PATCH_GRID
            if gripper_pad_patch_grid is None
            else gripper_pad_patch_grid
        )
        self.gripper_pad_patch_grid = tuple(int(x) for x in pad_patch_grid)
        self.actuator_control = _coerce_controller_config(actuator_control)
        self.task = task or PumpBottleUseTask()
        self.object_kind = str(getattr(self.task, "scene_object_kind", "pump_bottle"))
        if self.object_kind not in {"pump_bottle", "mini_keyboard"}:
            raise ValueError(
                "ArmlessRobotiqStickEnv supports task objects 'pump_bottle' and 'mini_keyboard', "
                f"got {self.object_kind!r}"
            )
        self.include_camera_obs = bool(include_camera_obs)
        self.camera_names = tuple(str(name) for name in camera_names)
        self.policy_camera_names = tuple(str(name) for name in policy_camera_names)
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.target_pointcloud_num = int(target_pointcloud_num)
        self.exclude_robot_pointcloud = bool(exclude_robot_pointcloud)
        self.exclude_table_pointcloud = bool(exclude_table_pointcloud)
        self.capture_camera_rgb = bool(capture_camera_rgb)
        self.capture_camera_segmentation = bool(capture_camera_segmentation)
        self.pointcloud_workspace_bounds = np.asarray(
            DEFAULT_POINTCLOUD_WORKSPACE_BOUNDS if pointcloud_workspace_bounds is None else pointcloud_workspace_bounds,
            dtype=np.float32,
        ).reshape(3, 2)
        self.camera_specs = default_camera_specs(width=self.camera_width, height=self.camera_height)

        scene_kwargs: dict[str, Any] = {
            "add_cameras": True,
            "embodiment": self.embodiment,
            "object_kind": self.object_kind,
            "camera_width": self.camera_width,
            "camera_height": self.camera_height,
        }
        if table_half_extents is not None:
            scene_kwargs["table_half_extents"] = tuple(float(x) for x in table_half_extents)
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
        if self.object_kind == "pump_bottle":
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
        elif self.object_kind == "mini_keyboard":
            if mini_keyboard_friction is not None:
                scene_kwargs["mini_keyboard_friction"] = tuple(float(x) for x in mini_keyboard_friction)
            if mini_keyboard_support_box_height is not None:
                scene_kwargs["mini_keyboard_support_box_height"] = float(mini_keyboard_support_box_height)
            if mini_keyboard_white_key_push_distance is not None:
                scene_kwargs["mini_keyboard_white_key_push_distance"] = float(mini_keyboard_white_key_push_distance)
            if mini_keyboard_black_key_push_distance is not None:
                scene_kwargs["mini_keyboard_black_key_push_distance"] = float(mini_keyboard_black_key_push_distance)
        scene_kwargs["gripper_pad_friction"] = self.gripper_pad_friction
        scene_kwargs["gripper_pad_patch_grid"] = self.gripper_pad_patch_grid
        scene_kwargs["enable_gripper_stick_collision"] = self.enable_gripper_stick_collision
        self.spec, model, data, self.scene = compile_tabletop_scene(**scene_kwargs)
        self.backend: RawMujocoBackend = make_backend(
            backend,
            model=model,
            data=data,
            frame_skip=int(frame_skip),
        )
        self.model = self.backend.model
        self.data = self.backend.data
        self.robot = _ArmlessRobot()
        self._viewer = None
        self._renderer = None
        self._camera_renderer: CameraRenderer | None = None
        self._rng = np.random.default_rng()
        self._last_q = np.zeros((ROBOTIQ_STICK_LOCAL_DOF,), dtype=np.float32)
        self._resolve_q_indices()
        self._apply_actuator_control_config()
        if self.gravity_compensation:
            self.backend.set_gravity_compensation_dofs(self._robot_qvel_indices)
        self.backend.forward()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.task.reset(self.model, self.data, self.scene, self._rng)
        self.set_q(self._last_q)
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
            "gripper_pad_friction": self.gripper_pad_friction,
            "gripper_pad_patch_grid": self.gripper_pad_patch_grid,
            "enable_gripper_stick_collision": self.enable_gripper_stick_collision,
        }
        if self.render_mode == "human":
            self.render()
        return obs, info

    def observe(self, *, include_camera_obs: bool | None = None) -> dict[str, Any]:
        obs = self.task.observe(self.model, self.data, self.scene)
        obs["robotiq_stick_q"] = self.current_q()
        capture_camera = self.include_camera_obs if include_camera_obs is None else bool(include_camera_obs)
        if capture_camera:
            sensor_data, sensor_param = self._camera_capture()
            obs["sensor_data"] = sensor_data
            obs["sensor_param"] = sensor_param
            obs["camera_names"] = self.camera_names
            obs["policy_camera_names"] = self.policy_camera_names
            obs["pointcloud"] = build_pointcloud_observation(
                sensor_data,
                sensor_param,
                policy_camera_names=self.policy_camera_names,
                target_num=self.target_pointcloud_num,
                exclude_robot=self.exclude_robot_pointcloud,
                exclude_table=self.exclude_table_pointcloud,
                workspace_bounds=self.pointcloud_workspace_bounds,
                rng=self._rng,
            )
        return obs

    def step(self, action: np.ndarray | dict[str, Any], *, include_camera_obs: bool | None = None):
        q_requested = self._q_from_action(action)
        q_target = self._rate_limited_q_target(q_requested)
        pre_obs = self.task.observe(self.model, self.data, self.scene)
        pre_qpos = self.data.qpos.copy()
        pre_ctrl = self.data.ctrl.copy()

        requested_qpos = self._target_qpos_from_q(q_requested)
        target_qpos = self._target_qpos_from_q(q_target)
        target_ctrl = self._target_ctrl_from_q(q_target)
        self.data.ctrl[:] = target_ctrl
        self.backend.step()
        self._last_q = q_target.astype(np.float32, copy=True)

        obs = self.observe(include_camera_obs=include_camera_obs)
        success = bool(obs["success"])
        reward = 1.0 if success else 0.0
        info = {
            "task": self.task.metadata,
            "backend": self.backend.backend_name,
            "mode": self.mode,
            "embodiment": self.embodiment,
            "gravity_compensation": self.gravity_compensation,
            "command": {
                "controlled_sides": tuple(),
                "sides": {},
                "pre_qpos": pre_qpos.astype(np.float32),
                "pre_ctrl": pre_ctrl.astype(np.float32),
                "requested_qpos": requested_qpos.astype(np.float32),
                "target_qpos": target_qpos.astype(np.float32),
                "target_ctrl": target_ctrl.astype(np.float32),
                "executed_qpos": self.data.qpos.astype(np.float32).copy(),
                "executed_qvel": self.data.qvel.astype(np.float32).copy(),
                "executed_ctrl": self.data.ctrl.astype(np.float32).copy(),
                "object_pose_before": np.asarray(pre_obs["object_pose"], dtype=np.float32).copy(),
                "object_pose_after": np.asarray(obs["object_pose"], dtype=np.float32).copy(),
                "robotiq_stick_q_requested": q_requested.astype(np.float32).copy(),
                "robotiq_stick_q_target": q_target.astype(np.float32).copy(),
                "robotiq_stick_q_executed": self.current_q().astype(np.float32).copy(),
            },
        }
        if self.render_mode == "human":
            self.render()
        return obs, reward, success, False, info

    def set_q(self, q: np.ndarray) -> None:
        self._apply_q(q)
        mujoco.mj_forward(self.model, self.data)

    def current_q(self) -> np.ndarray:
        gripper_root = self.data.qpos[self._gripper_root_qpos_adrs]
        stick_root = self.data.qpos[self._stick_root_qpos_adrs]
        joint_value = float(np.mean([self.data.qpos[adr] for adr in self._robotiq_joint_qpos_adrs]))
        cmd = np.clip(joint_value / ROBOTIQ_CLOSED_JOINT_VALUE * 255.0, 0.0, 255.0)
        return np.concatenate(
            [
                np.asarray(gripper_root[:6], dtype=np.float32),
                np.asarray([cmd], dtype=np.float32),
                np.asarray(stick_root[:6], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32, copy=False)

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

    def viewer_is_running(self) -> bool:
        return self._viewer is not None and bool(self._viewer.is_running())

    def set_viewer_text(self, title: str, detail: str = "") -> None:
        if self._viewer is None or not self._viewer.is_running():
            return
        self._viewer.set_texts((None, None, title, detail))

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

    def _resolve_q_indices(self) -> None:
        self._gripper_root_qpos_adrs = []
        self._gripper_root_qvel_adrs = []
        for name in ARMLESS_GRIPPER_ROOT_JOINT_NAMES:
            qadr, dadr = self._joint_addresses(name)
            self._gripper_root_qpos_adrs.append(qadr)
            self._gripper_root_qvel_adrs.append(dadr)
        self._stick_root_qpos_adrs = []
        self._stick_root_qvel_adrs = []
        for name in ARMLESS_STICK_ROOT_JOINT_NAMES:
            qadr, dadr = self._joint_addresses(name)
            self._stick_root_qpos_adrs.append(qadr)
            self._stick_root_qvel_adrs.append(dadr)
        self._robotiq_joint_qpos_adrs = []
        self._robotiq_joint_qvel_adrs = []
        for name in ROBOTIQ_INTERNAL_JOINT_NAMES:
            qadr, dadr = self._joint_addresses(name)
            self._robotiq_joint_qpos_adrs.append(qadr)
            self._robotiq_joint_qvel_adrs.append(dadr)
        self._gripper_root_qpos_adrs = np.asarray(self._gripper_root_qpos_adrs, dtype=np.int32)
        self._gripper_root_qvel_adrs = np.asarray(self._gripper_root_qvel_adrs, dtype=np.int32)
        self._stick_root_qpos_adrs = np.asarray(self._stick_root_qpos_adrs, dtype=np.int32)
        self._stick_root_qvel_adrs = np.asarray(self._stick_root_qvel_adrs, dtype=np.int32)
        self._robotiq_joint_qpos_adrs = np.asarray(self._robotiq_joint_qpos_adrs, dtype=np.int32)
        self._robotiq_joint_qvel_adrs = np.asarray(self._robotiq_joint_qvel_adrs, dtype=np.int32)
        self._gripper_root_actuator_ids = np.asarray(
            [self._actuator_id(f"{name}_actuator") for name in ARMLESS_GRIPPER_ROOT_JOINT_NAMES],
            dtype=np.int32,
        )
        self._stick_root_actuator_ids = np.asarray(
            [self._actuator_id(f"{name}_actuator") for name in ARMLESS_STICK_ROOT_JOINT_NAMES],
            dtype=np.int32,
        )
        self._robotiq_joint_actuator_ids = np.asarray(
            [self._actuator_id(f"{name}_actuator") for name in ROBOTIQ_INTERNAL_JOINT_NAMES],
            dtype=np.int32,
        )
        robot_qvel = list(self._gripper_root_qvel_adrs)
        robot_qvel.extend(self._robotiq_joint_qvel_adrs)
        robot_qvel.extend(self._stick_root_qvel_adrs)
        self._robot_qvel_indices = np.asarray(robot_qvel, dtype=np.int32)

    def _joint_addresses(self, name: str) -> tuple[int, int]:
        jid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, str(name)))
        if jid < 0:
            raise KeyError(f"Missing joint {name!r} in armless robotiq+stick scene")
        return int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid])

    def _actuator_id(self, name: str) -> int:
        aid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, str(name)))
        if aid < 0:
            raise KeyError(f"Missing actuator {name!r} in armless robotiq+stick scene")
        return aid

    def _q_from_action(self, action: np.ndarray | dict[str, Any]) -> np.ndarray:
        if isinstance(action, dict):
            if "q" not in action:
                raise KeyError("Armless robotiq+stick action dict must contain 'q'")
            action = action["q"]
        return validate_robotiq_stick_q(action, label="robotiq_stick_action")

    def _target_qpos_from_q(self, q: np.ndarray) -> np.ndarray:
        qpos = self.data.qpos.copy()
        gripper_root, joint_value, stick_root = self._target_parts_from_q(q)
        qpos[self._gripper_root_qpos_adrs] = gripper_root
        qpos[self._stick_root_qpos_adrs] = stick_root
        qpos[self._robotiq_joint_qpos_adrs] = joint_value
        return qpos

    def _target_parts_from_q(
        self,
        q: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        q_np = validate_robotiq_stick_q(q, label="robotiq_stick_q")
        joint_value = float(np.clip(q_np[6], 0.0, 255.0) * ROBOTIQ_CLOSED_JOINT_VALUE / 255.0)
        return (
            q_np[0:6].astype(np.float64),
            joint_value,
            q_np[7:13].astype(np.float64),
        )

    def _target_ctrl_from_q(self, q: np.ndarray) -> np.ndarray:
        gripper_root, joint_value, stick_root = self._target_parts_from_q(q)
        ctrl = self.data.ctrl.copy()
        self._set_ctrl_values(ctrl, self._gripper_root_actuator_ids, gripper_root)
        self._set_ctrl_values(ctrl, self._stick_root_actuator_ids, stick_root)
        self._set_ctrl_values(
            ctrl,
            self._robotiq_joint_actuator_ids,
            np.full((len(self._robotiq_joint_actuator_ids),), joint_value, dtype=np.float64),
        )
        return ctrl

    def _rate_limited_q_target(self, q_requested: np.ndarray) -> np.ndarray:
        requested = validate_robotiq_stick_q(q_requested, label="robotiq_stick_q_requested").astype(
            np.float64,
            copy=True,
        )
        current = self.current_q().astype(np.float64, copy=False)
        limited = requested.copy()
        max_translation = self.actuator_control.armless_root_max_translation_step
        max_rotation = self.actuator_control.armless_root_max_rotation_step
        for pos_start, rot_start in ((0, 3), (7, 10)):
            if max_translation is not None:
                pos_delta = requested[pos_start : pos_start + 3] - current[pos_start : pos_start + 3]
                limited[pos_start : pos_start + 3] = current[pos_start : pos_start + 3] + np.clip(
                    pos_delta,
                    -float(max_translation),
                    float(max_translation),
                )
            if max_rotation is not None:
                rot_delta = _wrap_to_pi(requested[rot_start : rot_start + 3] - current[rot_start : rot_start + 3])
                limited[rot_start : rot_start + 3] = current[rot_start : rot_start + 3] + np.clip(
                    rot_delta,
                    -float(max_rotation),
                    float(max_rotation),
                )
        return validate_robotiq_stick_q(limited.astype(np.float32), label="robotiq_stick_q_target")

    def _set_ctrl_values(self, ctrl: np.ndarray, actuator_ids: np.ndarray, values: np.ndarray) -> None:
        ids = np.asarray(actuator_ids, dtype=np.int32).reshape(-1)
        vals = np.asarray(values, dtype=np.float64).reshape(ids.shape)
        ranges = np.asarray(self.model.actuator_ctrlrange[ids], dtype=np.float64)
        limited = np.asarray(self.model.actuator_ctrllimited[ids], dtype=bool)
        if np.any(limited):
            vals = vals.copy()
            vals[limited] = np.clip(vals[limited], ranges[limited, 0], ranges[limited, 1])
        ctrl[ids] = vals

    def _apply_actuator_control_config(self) -> None:
        cfg = self.actuator_control
        for root_actuator_ids in (self._gripper_root_actuator_ids, self._stick_root_actuator_ids):
            base_xyz_ids = root_actuator_ids[:3]
            base_rpy_ids = root_actuator_ids[3:6]
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
        if cfg.fingers is None:
            _scale_position_actuators(
                self.model,
                self._robotiq_joint_actuator_ids,
                kp_scale=float(cfg.hand_kp_scale),
                kv_scale=float(cfg.hand_kv_scale),
                force_scale=float(cfg.hand_force_scale),
            )
        else:
            _set_position_actuators(
                self.model,
                self._robotiq_joint_actuator_ids,
                cfg.fingers,
                label="controller.fingers",
            )

    def _apply_q(self, q: np.ndarray) -> None:
        q_np = validate_robotiq_stick_q(q, label="robotiq_stick_q")
        gripper_root, joint_value, stick_root = self._target_parts_from_q(q_np)
        self.data.qpos[self._gripper_root_qpos_adrs] = gripper_root
        self.data.qpos[self._stick_root_qpos_adrs] = stick_root
        self.data.qpos[self._robotiq_joint_qpos_adrs] = joint_value
        self.data.qvel[self._robot_qvel_indices] = 0.0
        self.data.ctrl[:] = self._target_ctrl_from_q(q_np)
        self._last_q = q_np.astype(np.float32, copy=True)


def _coerce_controller_config(value: Any | None) -> _ArmlessControllerConfig:
    if value is None:
        config = _ArmlessControllerConfig()
    else:
        config = _ArmlessControllerConfig(
            arm_kp_scale=float(getattr(value, "arm_kp_scale", 1.0)),
            arm_kv_scale=float(getattr(value, "arm_kv_scale", 1.0)),
            arm_force_scale=float(getattr(value, "arm_force_scale", 1.0)),
            hand_kp_scale=float(getattr(value, "hand_kp_scale", 1.0)),
            hand_kv_scale=float(getattr(value, "hand_kv_scale", 1.0)),
            hand_force_scale=float(getattr(value, "hand_force_scale", 1.0)),
            base_xyz=getattr(value, "base_xyz", None),
            base_rpy=getattr(value, "base_rpy", None),
            fingers=getattr(value, "fingers", None),
            armless_root_max_translation_step=_optional_positive_float(
                getattr(value, "armless_root_max_translation_step", 0.006)
            ),
            armless_root_max_rotation_step=_optional_positive_float(
                getattr(value, "armless_root_max_rotation_step", 0.04)
            ),
        )
    for label, scale in (
        ("controller.arm_kp_scale", config.arm_kp_scale),
        ("controller.arm_kv_scale", config.arm_kv_scale),
        ("controller.arm_force_scale", config.arm_force_scale),
        ("controller.hand_kp_scale", config.hand_kp_scale),
        ("controller.hand_kv_scale", config.hand_kv_scale),
        ("controller.hand_force_scale", config.hand_force_scale),
    ):
        if float(scale) <= 0.0:
            raise ValueError(f"{label} must be positive, got {scale}")
    return config


def _optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"Expected a positive float or None, got {value}")
    return parsed


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


def _scale_position_actuators(
    model: mujoco.MjModel,
    actuator_ids: np.ndarray,
    *,
    kp_scale: float,
    kv_scale: float,
    force_scale: float,
) -> None:
    ids = np.asarray(actuator_ids, dtype=np.int32).reshape(-1)
    if int(ids.size) == 0:
        return
    model.actuator_gainprm[ids, 0] *= float(kp_scale)
    model.actuator_biasprm[ids, 1] *= float(kp_scale)
    model.actuator_biasprm[ids, 2] *= float(kv_scale)
    model.actuator_forcerange[ids, :] *= float(force_scale)


def _set_position_actuators(
    model: mujoco.MjModel,
    actuator_ids: np.ndarray,
    config: Any,
    *,
    label: str,
) -> None:
    ids = np.asarray(actuator_ids, dtype=np.int32).reshape(-1)
    if int(ids.size) == 0:
        return
    kp = _optional_positive_float_with_label(getattr(config, "kp", None), f"{label}.kp")
    kv = _resolve_kv(
        getattr(config, "kv", None),
        kp=kp,
        current_kp=model.actuator_gainprm[ids, 0],
        label=f"{label}.kv",
    )
    max_effort = _optional_positive_float_with_label(
        getattr(config, "max_effort", None),
        f"{label}.max_effort",
    )
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


def _optional_positive_float_with_label(value: Any, label: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"{label} must be positive or null, got {value}")
    return parsed


def _resolve_kv(
    value: Any,
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
