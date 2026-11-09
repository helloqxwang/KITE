"""Config-driven factory helpers for tabletop experiment scripts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .assets import (
    CROSS_EMBODIMENT_ROOT,
    MINI_KEYBOARD_ROOT_Z_OFFSET,
    MINI_KEYBOARD_SUPPORT_BOX_HEIGHT,
    TABLE_CENTER,
    TABLE_HALF_EXTENTS,
    TABLE_TOP_Z,
)
from .armless_robotiq_stick_env import ArmlessRobotiqStickEnv
from .env import ActuatorControlConfig, ActuatorGainConfig, TabletopManipulationEnv
from .ik import IKSolverConfig
from .perception import DEFAULT_CAMERA_NAMES, DEFAULT_POLICY_CAMERA_NAMES
from .object_suite import load_object_suite, object_spec_by_id
from .scene import (
    ALLEGRO_FINGERTIP_PAD_RADIUS,
    CUBE_TEXTURE_FACE_NAMES,
    FRANKA_ROBOTIQ_PAD1_FRICTION,
    FRANKA_ROBOTIQ_PAD2_FRICTION,
    FRANKA_ROBOTIQ_PAD_CONDIM,
    FRANKA_ROBOTIQ_PAD_PRIORITY,
    FRANKA_ROBOTIQ_PAD_SOLIMP,
    FRANKA_ROBOTIQ_PAD_SOLREF,
    MINI_KEYBOARD_BLACK_KEY_PUSH_DISTANCE,
    MINI_KEYBOARD_WHITE_KEY_PUSH_DISTANCE,
    OBJECT_SUITE_BIN_FRICTION,
)
from .tasks import CubePickTask, DEFAULT_KEYBOARD_QUAT_WXYZ, KeyboardPressTask, ObjectSuitePickPlaceTask, PumpBottleUseTask, PUMP_BOTTLE_DEFAULT_QUAT_WXYZ


DEFAULT_TABLE_TEXTURE_TILE_SIZE = 0.375


def optional_project_path(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "null":
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    return str((CROSS_EMBODIMENT_ROOT / path).resolve())


def required_project_path(value: Any, *, label: str) -> str:
    path = optional_project_path(value)
    if path is None:
        raise ValueError(f"{label} must be set")
    return path


def task_name_from_config(cfg: DictConfig) -> str:
    return str(OmegaConf.select(cfg, "task.name", default="cube_pick"))


def build_task_from_config(cfg: DictConfig) -> CubePickTask | ObjectSuitePickPlaceTask | PumpBottleUseTask | KeyboardPressTask:
    task_name = task_name_from_config(cfg)
    if task_name == "object_suite_pick_place":
        manifest_path = required_project_path(
            OmegaConf.select(cfg, "objects.manifest", default="./experiment/assets/ycb_object_suite.json"),
            label="objects.manifest",
        )
        objects = load_object_suite(
            manifest_path,
            mass_kg_override=OmegaConf.select(cfg, "objects.mass_kg_override", default=None),
            collision_mesh_mode=str(
                OmegaConf.select(cfg, "objects.collision_mesh_mode", default="single_mesh")
            ),
            convex_decomposition_root=optional_project_path(
                OmegaConf.select(
                    cfg,
                    "objects.convex_decomposition_root",
                    default="./experiment/assets/object_suite_convex_decomposition",
                )
            ),
            convex_decomposition_subdir=str(
                OmegaConf.select(
                    cfg,
                    "objects.convex_decomposition_subdir",
                    default="convex_decomposition",
                )
            ),
        )
        active_id = str(OmegaConf.select(cfg, "objects.active_id", default=objects[0].object_id))
        object_spec = object_spec_by_id(objects, active_id)
        object_spec = replace(
            object_spec,
            friction=_friction_from_config(cfg, "objects.friction", object_spec.friction),
        )
        object_spec.require_assets()
        return ObjectSuitePickPlaceTask(
            object_spec=object_spec,
            objective=str(
                OmegaConf.select(cfg, "task.objective", default="pick_place")
            ).strip().lower(),
            xy_range=(
                tuple(float(x) for x in cfg.objects.x_range),
                tuple(float(x) for x in cfg.objects.y_range),
            ),
            yaw_range_rad=(
                np.deg2rad(float(cfg.objects.yaw_range_deg[0])),
                np.deg2rad(float(cfg.objects.yaw_range_deg[1])),
            ),
            table_clearance=float(OmegaConf.select(cfg, "objects.table_clearance", default=0.003)),
            bin_center_xy=tuple(float(x) for x in cfg.bin.center_xy),
            bin_inner_size=tuple(float(x) for x in cfg.bin.inner_size),
            bin_wall_height=float(OmegaConf.select(cfg, "bin.wall_height", default=0.04)),
            bin_wall_thickness=float(OmegaConf.select(cfg, "bin.wall_thickness", default=0.008)),
            bin_friction=_friction_from_config(
                cfg,
                "bin.friction",
                OBJECT_SUITE_BIN_FRICTION,
            ),
            placement_margin=float(OmegaConf.select(cfg, "bin.placement_margin", default=0.005)),
            placement_reference=str(
                OmegaConf.select(cfg, "bin.placement_reference", default="bounds")
            ).strip().lower(),
            bottom_tolerance=float(OmegaConf.select(cfg, "bin.bottom_tolerance", default=0.03)),
            lift_delta=float(OmegaConf.select(cfg, "task.lift_delta", default=0.05)),
        )
    if task_name == "cube_pick":
        validate_task_config(cfg)
        return CubePickTask(
            cube_xy_range=(
                (float(cfg.cube.x_range[0]), float(cfg.cube.x_range[1])),
                (float(cfg.cube.y_range[0]), float(cfg.cube.y_range[1])),
            ),
            cube_yaw_range_rad=(np.deg2rad(float(cfg.cube.yaw_range_deg[0])), np.deg2rad(float(cfg.cube.yaw_range_deg[1]))),
            cube_side_length_range=(float(cfg.cube.side_length_range[0]), float(cfg.cube.side_length_range[1])),
            cube_height_ratio=float(OmegaConf.select(cfg, "cube.height_ratio", default=1.0)),
            table_clearance=float(cfg.cube.table_clearance),
            reference_cube_mass=float(OmegaConf.select(cfg, "cube.mass", default=0.08)),
            lift_success_z=_cube_lift_success_z_from_config(cfg),
        )
    if task_name == "pumpbottle_use":
        validate_pump_bottle_task_config(cfg)
        pose = _float_sequence(OmegaConf.select(cfg, "pump_bottle.default_pose", default=(0.15800001, 0.003, 0.6397543)), 3)
        quat = _float_sequence(
            OmegaConf.select(cfg, "pump_bottle.default_quat_wxyz", default=PUMP_BOTTLE_DEFAULT_QUAT_WXYZ),
            4,
        )
        return PumpBottleUseTask(
            object_pos=(float(pose[0]), float(pose[1]), float(pose[2])),
            object_quat_wxyz=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            xy_range=(
                (float(cfg.pump_bottle.x_range[0]), float(cfg.pump_bottle.x_range[1])),
                (float(cfg.pump_bottle.y_range[0]), float(cfg.pump_bottle.y_range[1])),
            ),
            yaw_range_rad=(
                np.deg2rad(float(cfg.pump_bottle.yaw_range_deg[0])),
                np.deg2rad(float(cfg.pump_bottle.yaw_range_deg[1])),
            ),
            table_clearance=float(OmegaConf.select(cfg, "pump_bottle.table_clearance", default=0.0)),
            button_reset_slide=float(OmegaConf.select(cfg, "pump_bottle.button_reset_slide", default=-0.00700733)),
            button_success_slide=float(OmegaConf.select(cfg, "pump_bottle.button_success_slide", default=-0.012)),
            lift_success_delta=float(OmegaConf.select(cfg, "pump_bottle.lift_success_delta", default=0.08)),
            safe_radius=float(OmegaConf.select(cfg, "pump_bottle.safe_radius", default=0.05)),
            object_id=_pump_bottle_active_id_from_config(cfg),
        )
    if task_name == "keyboard_press":
        validate_keyboard_task_config(cfg)
        support_box_height = _keyboard_support_box_height_from_config(cfg)
        pose = _float_sequence(
            OmegaConf.select(
                cfg,
                "keyboard.default_pose",
                default=(0.15, 0.0, _keyboard_root_z(support_box_height)),
            ),
            3,
        )
        pose[2] = _keyboard_root_z(support_box_height)
        quat = _float_sequence(
            OmegaConf.select(cfg, "keyboard.default_quat_wxyz", default=DEFAULT_KEYBOARD_QUAT_WXYZ),
            4,
        )
        return KeyboardPressTask(
            object_pos=(float(pose[0]), float(pose[1]), float(pose[2])),
            object_quat_wxyz=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            support_box_height=float(support_box_height),
            xy_range=(
                (float(cfg.keyboard.x_range[0]), float(cfg.keyboard.x_range[1])),
                (float(cfg.keyboard.y_range[0]), float(cfg.keyboard.y_range[1])),
            ),
            yaw_range_rad=(
                np.deg2rad(float(cfg.keyboard.yaw_range_deg[0])),
                np.deg2rad(float(cfg.keyboard.yaw_range_deg[1])),
            ),
            table_clearance=float(OmegaConf.select(cfg, "keyboard.table_clearance", default=0.0)),
            target_key_joint_names=tuple(
                str(x)
                for x in OmegaConf.select(
                    cfg,
                    "keyboard.target_key_joint_names",
                    default=("white_key_00_C2_slide", "white_key_01_D2_slide", "white_key_02_E2_slide"),
                )
            ),
            key_success_fraction=float(OmegaConf.select(cfg, "keyboard.key_success_fraction", default=0.8)),
            safe_radius=float(OmegaConf.select(cfg, "keyboard.safe_radius", default=0.16)),
        )
    raise ValueError(f"Unsupported tabletop task {task_name!r}; available: cube_pick, pumpbottle_use, keyboard_press")


def build_ik_from_config(cfg: DictConfig) -> IKSolverConfig:
    return IKSolverConfig(
        max_iters=int(OmegaConf.select(cfg, "ik.max_iters", default=120)),
        damping=float(OmegaConf.select(cfg, "ik.damping", default=0.05)),
        step_scale=float(OmegaConf.select(cfg, "ik.step_scale", default=1.0)),
        max_step_norm=float(OmegaConf.select(cfg, "ik.max_step_norm", default=0.25)),
        pos_tolerance=float(OmegaConf.select(cfg, "ik.pos_tolerance", default=0.002)),
        ori_tolerance=float(OmegaConf.select(cfg, "ik.ori_tolerance", default=0.03)),
        ori_weight=float(OmegaConf.select(cfg, "ik.ori_weight", default=1.0)),
    )


def build_actuator_control_from_config(cfg: DictConfig) -> ActuatorControlConfig:
    config = ActuatorControlConfig(
        arm_kp_scale=float(OmegaConf.select(cfg, "controller.arm_kp_scale", default=1.0)),
        arm_kv_scale=float(OmegaConf.select(cfg, "controller.arm_kv_scale", default=1.0)),
        arm_force_scale=float(OmegaConf.select(cfg, "controller.arm_force_scale", default=1.0)),
        hand_kp_scale=float(OmegaConf.select(cfg, "controller.hand_kp_scale", default=1.0)),
        hand_kv_scale=float(OmegaConf.select(cfg, "controller.hand_kv_scale", default=1.0)),
        hand_force_scale=float(OmegaConf.select(cfg, "controller.hand_force_scale", default=1.0)),
        base_xyz=_actuator_gain_config_from_config(cfg, "controller.base_xyz"),
        base_rpy=_actuator_gain_config_from_config(cfg, "controller.base_rpy"),
        fingers=_actuator_gain_config_from_config(cfg, "controller.fingers"),
        armless_root_max_translation_step=_optional_positive_float_from_config(
            cfg,
            "controller.armless_root_max_translation_step",
            default=0.006,
        ),
        armless_root_max_rotation_step=_optional_positive_float_from_config(
            cfg,
            "controller.armless_root_max_rotation_step",
            default=0.04,
        ),
    )
    for label, value in (
        ("controller.arm_kp_scale", config.arm_kp_scale),
        ("controller.arm_kv_scale", config.arm_kv_scale),
        ("controller.arm_force_scale", config.arm_force_scale),
        ("controller.hand_kp_scale", config.hand_kp_scale),
        ("controller.hand_kv_scale", config.hand_kv_scale),
        ("controller.hand_force_scale", config.hand_force_scale),
    ):
        if float(value) <= 0.0:
            raise ValueError(f"{label} must be positive, got {value}")
    return config


def _actuator_gain_config_from_config(cfg: DictConfig, path: str) -> ActuatorGainConfig | None:
    node = OmegaConf.select(cfg, path, default=None)
    if node is None:
        return None
    kp = _optional_positive_float_from_config(cfg, f"{path}.kp", default=None)
    kv = _optional_positive_float_or_auto_from_config(cfg, f"{path}.kv", default=None)
    max_effort = _optional_positive_float_from_config(cfg, f"{path}.max_effort", default=None)
    if kp is None and kv is None and max_effort is None:
        return None
    return ActuatorGainConfig(kp=kp, kv=kv, max_effort=max_effort)


def _optional_positive_float_or_auto_from_config(cfg: DictConfig, path: str, *, default: Any) -> float | str | None:
    value = OmegaConf.select(cfg, path, default=default)
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"{path} must be positive, 'auto', or null, got {value}")
    return parsed


def _optional_positive_float_from_config(
    cfg: DictConfig,
    path: str,
    *,
    default: float,
) -> float | None:
    value = OmegaConf.select(cfg, path, default=default)
    if value is None:
        return None
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"{path} must be positive or null, got {value}")
    return parsed


def build_env_from_config(
    cfg: DictConfig,
    *,
    backend: str | None = None,
    render: bool = False,
    viewer_key_callback=None,
) -> TabletopManipulationEnv | ArmlessRobotiqStickEnv:
    # Scene construction and the environment apply contact parameters and gains
    # to this model instance. Writing them into the shared asset is redundant
    # and makes concurrent task configurations race with each other.
    from paper_config import runtime_config
    cfg = runtime_config(cfg)
    task = build_task_from_config(cfg)
    camera_names = tuple(
        str(x) for x in OmegaConf.select(cfg, "perception.camera_names", default=DEFAULT_CAMERA_NAMES)
    )
    policy_camera_names = tuple(
        str(x) for x in OmegaConf.select(cfg, "perception.policy_camera_names", default=DEFAULT_POLICY_CAMERA_NAMES)
    )
    online_viewer_show = OmegaConf.select(cfg, "online_rollout.viewer_show_collision_geoms", default=None)
    viewer_show_collision_geoms = bool(
        online_viewer_show
        if online_viewer_show is not None
        else OmegaConf.select(cfg, "viewer.show_collision_geoms", default=False)
    )
    online_viewer_mode = OmegaConf.select(cfg, "online_rollout.viewer_collision_geom_mode", default=None)
    viewer_collision_geom_mode = str(
        online_viewer_mode
        if online_viewer_mode is not None
        else OmegaConf.select(
            cfg,
            "viewer.collision_geom_mode",
            default="overlay" if viewer_show_collision_geoms else "none",
        )
    )
    task_name = task_name_from_config(cfg)
    rotation_reference = OmegaConf.select(cfg, "controller.base_rpy.reference_euler_xyz", default=None)
    if rotation_reference is not None and str(cfg.env.embodiment) not in {"barrett", "allegro_right"}:
        raise ValueError("controller.base_rpy.reference_euler_xyz requires Barrett or Allegro")
    env_kwargs: dict[str, Any] = dict(
        backend=str(cfg.env.backend if backend is None else backend),
        mode=str(cfg.env.mode),
        embodiment=str(cfg.env.embodiment),
        frame_skip=int(cfg.env.frame_skip),
        timestep=float(OmegaConf.select(cfg, "env.timestep", default=0.002)),
        include_proprio=True,
        render_mode="human" if bool(render) else None,
        viewer_key_callback=viewer_key_callback,
        gravity_compensation=bool(cfg.env.gravity_compensation),
        ik_config=build_ik_from_config(cfg),
        task=task,
        include_camera_obs=bool(OmegaConf.select(cfg, "perception.include_camera_obs", default=True)),
        camera_names=camera_names,
        policy_camera_names=policy_camera_names,
        camera_width=int(OmegaConf.select(cfg, "perception.camera_width", default=640)),
        camera_height=int(OmegaConf.select(cfg, "perception.camera_height", default=360)),
        table_half_extents=_table_half_extents_from_config(cfg),
        target_pointcloud_num=int(OmegaConf.select(cfg, "perception.target_pointcloud_num", default=512)),
        exclude_robot_pointcloud=bool(OmegaConf.select(cfg, "perception.exclude_robot_pointcloud", default=True)),
        exclude_table_pointcloud=bool(OmegaConf.select(cfg, "perception.exclude_table_pointcloud", default=True)),
        object_only_pointcloud=bool(OmegaConf.select(cfg, "perception.object_only_pointcloud", default=False)),
        object_mask_erosion_pixels=int(
            OmegaConf.select(cfg, "perception.object_mask_erosion_pixels", default=0)
        ),
        object_depth_neighbor_threshold_m=float(
            OmegaConf.select(cfg, "perception.object_depth_neighbor_threshold_m", default=0.0)
        ),
        object_bounds_filter_tolerance_m=OmegaConf.select(
            cfg, "perception.object_bounds_filter_tolerance_m", default=None
        ),
        capture_camera_rgb=bool(OmegaConf.select(cfg, "perception.capture_rgb", default=True)),
        capture_camera_segmentation=bool(OmegaConf.select(cfg, "perception.capture_segmentation", default=True)),
        actuator_control=build_actuator_control_from_config(cfg),
        enable_fingertip_collision_pads=bool(
            OmegaConf.select(cfg, "env.enable_fingertip_collision_pads", default=True)
        ),
        robotiq_pad1_friction=_robotiq_pad1_friction_from_config(cfg),
        robotiq_pad2_friction=_robotiq_pad2_friction_from_config(cfg),
        robotiq_pad_condim=_robotiq_pad_condim_from_config(cfg),
        robotiq_pad_priority=_robotiq_pad_priority_from_config(cfg),
        robotiq_pad_solref=_robotiq_pad_solref_from_config(cfg),
        robotiq_pad_solimp=_robotiq_pad_solimp_from_config(cfg),
        allegro_hand_friction=_allegro_hand_friction_from_config(cfg),
        allegro_hand_condim=_allegro_hand_condim_from_config(cfg),
        allegro_hand_priority=_allegro_hand_priority_from_config(cfg),
        allegro_hand_solref=_allegro_hand_solref_from_config(cfg),
        allegro_hand_solimp=_allegro_hand_solimp_from_config(cfg),
        allegro_hand_collision_model=str(
            OmegaConf.select(cfg, "env.allegro_hand_collision_model", default="full")
        ),
        allegro_fingertip_friction=_allegro_fingertip_friction_from_config(cfg),
        allegro_fingertip_radius=_allegro_fingertip_radius_from_config(cfg),
        allegro_fingertip_condim=_allegro_fingertip_condim_from_config(cfg),
        allegro_fingertip_priority=_allegro_fingertip_priority_from_config(cfg),
        allegro_fingertip_solref=_allegro_fingertip_solref_from_config(cfg),
        allegro_fingertip_solimp=_allegro_fingertip_solimp_from_config(cfg),
        barrett_fingertip_friction=_barrett_fingertip_friction_from_config(cfg),
        barrett_fingertip_collision_model=str(
            OmegaConf.select(cfg, "env.barrett_fingertip_collision_model", default="capsule")
        ),
        barrett_fingertip_capsule_radius=float(
            OmegaConf.select(cfg, "env.barrett_fingertip_capsule_radius", default=0.014)
        ),
        wujihand_contact_friction=_wujihand_contact_friction_from_config(cfg),
        viewer_show_collision_geoms=viewer_show_collision_geoms,
        viewer_collision_geom_mode=viewer_collision_geom_mode,
    )
    env_kwargs.update(_render_appearance_kwargs_from_config(cfg))
    if rotation_reference is not None:
        env_kwargs["floating_rotation_reference_xyz"] = tuple(_float_sequence(rotation_reference, 3))
    if task_name == "pumpbottle_use":
        env_kwargs.update(
            object_kind="pump_bottle",
            pump_bottle_mass=float(OmegaConf.select(cfg, "pump_bottle.mass", default=0.18)),
            pump_button_mass=float(OmegaConf.select(cfg, "pump_bottle.button_mass", default=0.05)),
            pump_button_joint_range=_pump_button_joint_range_from_config(cfg),
            pump_button_joint_stiffness=float(
                OmegaConf.select(cfg, "pump_bottle.button_joint_stiffness", default=35.0)
            ),
            pump_button_joint_damping=float(
                OmegaConf.select(cfg, "pump_bottle.button_joint_damping", default=5.0)
            ),
            pump_button_actuator_kp=float(
                OmegaConf.select(cfg, "pump_bottle.button_actuator_kp", default=35.0)
            ),
            pump_button_force_site_visible=bool(
                OmegaConf.select(cfg, "pump_bottle.show_button_force_site", default=True)
            ),
            pump_bottle_mesh_scale=float(OmegaConf.select(cfg, "pump_bottle.mesh_scale", default=1.3)),
            pump_bottle_friction=_pump_bottle_friction_from_config(cfg),
            pump_bottle_body_collision_mode=_pump_bottle_collision_mode_from_config(
                cfg,
                "pump_bottle.lower_collision_mode",
                fallback_path="pump_bottle.body_collision_mode",
            ),
            pump_bottle_neck_collision_mode=_pump_bottle_collision_mode_from_config(
                cfg,
                "pump_bottle.neck_collision_mode",
            ),
            pump_bottle_visual_mesh_source=str(
                OmegaConf.select(cfg, "pump_bottle.visual_mesh_source", default="visual")
            ),
            pump_bottle_contact_margin=_pump_bottle_contact_margin_from_config(cfg),
            pump_bottle_asset_id=_pump_bottle_active_id_from_config(cfg),
            pump_bottle_manifest_path=OmegaConf.select(cfg, "pump_bottle.manifest_path", default=None),
        )
    elif task_name == "keyboard_press":
        support_box_height = _keyboard_support_box_height_from_config(cfg)
        white_key_push_distance, black_key_push_distance = _keyboard_key_push_distances_from_config(cfg)
        env_kwargs.update(
            object_kind="mini_keyboard",
            mini_keyboard_friction=_keyboard_friction_from_config(cfg),
            mini_keyboard_support_box_height=float(support_box_height),
            mini_keyboard_white_key_push_distance=float(white_key_push_distance),
            mini_keyboard_black_key_push_distance=float(black_key_push_distance),
        )
    elif task_name == "object_suite_pick_place":
        env_kwargs.update(object_kind="ycb_object")
    else:
        cube_height_ratio = float(OmegaConf.select(cfg, "cube.height_ratio", default=1.0))
        env_kwargs.update(
            object_kind="cube",
            cube_half_extent=0.5 * float(cfg.cube.side_length_range[1]) * max(1.0, cube_height_ratio),
            cube_mass=float(OmegaConf.select(cfg, "cube.mass", default=0.08)),
            cube_friction=_cube_friction_from_config(cfg),
        )
    if str(cfg.env.embodiment) == "robotiq_gripper_stick_armless":
        if task_name not in {"pumpbottle_use", "keyboard_press"}:
            raise ValueError(
                "robotiq_gripper_stick_armless currently supports task.name=pumpbottle_use or keyboard_press only"
            )
        armless_kwargs = dict(env_kwargs)
        armless_kwargs.pop("include_proprio", None)
        armless_kwargs.pop("ik_config", None)
        armless_kwargs.pop("object_kind", None)
        armless_kwargs.pop("timestep", None)
        armless_kwargs.pop("enable_fingertip_collision_pads", None)
        armless_kwargs.pop("robotiq_pad1_friction", None)
        armless_kwargs.pop("robotiq_pad2_friction", None)
        armless_kwargs.pop("robotiq_pad_condim", None)
        armless_kwargs.pop("robotiq_pad_priority", None)
        armless_kwargs.pop("robotiq_pad_solref", None)
        armless_kwargs.pop("robotiq_pad_solimp", None)
        armless_kwargs.pop("allegro_hand_friction", None)
        armless_kwargs.pop("allegro_hand_condim", None)
        armless_kwargs.pop("allegro_hand_priority", None)
        armless_kwargs.pop("allegro_hand_solref", None)
        armless_kwargs.pop("allegro_hand_solimp", None)
        armless_kwargs.pop("allegro_hand_collision_model", None)
        armless_kwargs.pop("allegro_fingertip_friction", None)
        armless_kwargs.pop("allegro_fingertip_radius", None)
        armless_kwargs.pop("allegro_fingertip_condim", None)
        armless_kwargs.pop("allegro_fingertip_priority", None)
        armless_kwargs.pop("allegro_fingertip_solref", None)
        armless_kwargs.pop("allegro_fingertip_solimp", None)
        armless_kwargs.pop("barrett_fingertip_friction", None)
        armless_kwargs.pop("barrett_fingertip_collision_model", None)
        armless_kwargs.pop("barrett_fingertip_capsule_radius", None)
        armless_kwargs.pop("wujihand_contact_friction", None)
        armless_kwargs.pop("object_only_pointcloud", None)
        armless_kwargs.pop("object_mask_erosion_pixels", None)
        armless_kwargs.pop("object_depth_neighbor_threshold_m", None)
        armless_kwargs.pop("object_bounds_filter_tolerance_m", None)
        armless_kwargs["enable_gripper_stick_collision"] = bool(
            OmegaConf.select(cfg, "env.enable_gripper_stick_collision", default=False)
        )
        armless_kwargs["gripper_pad_friction"] = _gripper_pad_friction_from_config(cfg)
        armless_kwargs["gripper_pad_patch_grid"] = _gripper_pad_patch_grid_from_config(cfg)
        return ArmlessRobotiqStickEnv(**armless_kwargs)
    return TabletopManipulationEnv(**env_kwargs)




def _friction_from_config(
    cfg: DictConfig,
    path: str,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    value = OmegaConf.select(cfg, path, default=default)
    friction = np.asarray(list(value), dtype=np.float64).reshape(-1)
    if friction.shape != (3,):
        raise ValueError(f"{path} must contain 3 values, got shape {friction.shape}")
    if np.any(friction < 0.0):
        raise ValueError(f"{path} must be non-negative, got {friction.tolist()}")
    return tuple(float(x) for x in friction)


def _float_vector_from_config(
    cfg: DictConfig,
    path: str,
    default: tuple[float, ...],
) -> tuple[float, ...]:
    value = OmegaConf.select(cfg, path, default=default)
    vector = np.asarray(list(value), dtype=np.float64).reshape(-1)
    if vector.shape != (len(default),):
        raise ValueError(f"{path} must contain {len(default)} values, got shape {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{path} must contain finite values, got {vector.tolist()}")
    return tuple(float(x) for x in vector)


def _pair_from_config(cfg: DictConfig, path: str, default: tuple[float, float]) -> tuple[float, float]:
    values = _float_vector_from_config(cfg, path, default)
    return float(values[0]), float(values[1])


def _table_half_extents_from_config(cfg: DictConfig) -> tuple[float, float, float]:
    values = _float_vector_from_config(cfg, "scene.table_half_extents", TABLE_HALF_EXTENTS)
    if any(value <= 0.0 for value in values):
        raise ValueError(f"scene.table_half_extents must be positive, got {values}")
    return float(values[0]), float(values[1]), float(values[2])


def _table_texrepeat_from_config(cfg: DictConfig) -> tuple[float, float]:
    value = OmegaConf.select(cfg, "render_appearance.table.texrepeat", default=(1.0, 1.0))
    if isinstance(value, str) and value.strip().lower().replace("-", "_") in {"auto", "adaptive"}:
        tile_size = _table_texture_tile_size_from_config(cfg)
        half_extents = _table_half_extents_from_config(cfg)
        return (2.0 * float(half_extents[0]) / tile_size[0], 2.0 * float(half_extents[1]) / tile_size[1])
    return _pair_from_config(cfg, "render_appearance.table.texrepeat", (1.0, 1.0))


def _table_texture_tile_size_from_config(cfg: DictConfig) -> tuple[float, float]:
    value = OmegaConf.select(
        cfg,
        "render_appearance.table.texture_tile_size",
        default=DEFAULT_TABLE_TEXTURE_TILE_SIZE,
    )
    if OmegaConf.is_list(value) or isinstance(value, (list, tuple)):
        arr = np.asarray(list(value), dtype=np.float64).reshape(-1)
        if arr.shape != (2,):
            raise ValueError(
                "render_appearance.table.texture_tile_size must be a positive scalar "
                f"or two positive values, got shape {arr.shape}"
            )
        tile_size = (float(arr[0]), float(arr[1]))
    else:
        scalar = float(value)
        tile_size = (scalar, scalar)
    if any(value <= 0.0 or not np.isfinite(value) for value in tile_size):
        raise ValueError(f"render_appearance.table.texture_tile_size must be positive, got {tile_size}")
    return tile_size


def _rgba_from_config(
    cfg: DictConfig,
    path: str,
    default: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    rgba = _float_vector_from_config(cfg, path, default)
    if any(value < 0.0 or value > 1.0 for value in rgba):
        raise ValueError(f"{path} values must be in [0, 1], got {rgba}")
    return float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])


def _five_tuple_from_config(
    cfg: DictConfig,
    path: str,
    default: tuple[float, float, float, float, float],
) -> tuple[float, float, float, float, float]:
    values = _float_vector_from_config(cfg, path, default)
    return float(values[0]), float(values[1]), float(values[2]), float(values[3]), float(values[4])


def _condim_from_config(cfg: DictConfig, path: str, default: int) -> int:
    value = int(OmegaConf.select(cfg, path, default=default))
    if value not in {1, 3, 4, 6}:
        raise ValueError(f"{path} must be one of 1, 3, 4, 6, got {value}")
    return value


def _priority_from_config(cfg: DictConfig, path: str, default: int) -> int:
    value = int(OmegaConf.select(cfg, path, default=default))
    if value < 0:
        raise ValueError(f"{path} must be non-negative, got {value}")
    return value


def _cube_friction_from_config(cfg: DictConfig) -> tuple[float, float, float]:
    return _friction_from_config(cfg, "cube.friction", (1.2, 0.05, 0.01))


def _pump_bottle_friction_from_config(cfg: DictConfig) -> tuple[float, float, float]:
    return _friction_from_config(cfg, "pump_bottle.friction", (2.0, 0.01, 0.0002))


def _pump_bottle_collision_mode_from_config(
    cfg: DictConfig,
    path: str,
    *,
    fallback_path: str | None = None,
) -> str:
    value = OmegaConf.select(cfg, path, default=None)
    if value is None and fallback_path is not None:
        value = OmegaConf.select(cfg, fallback_path, default=None)
    text = "cylinder" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "cylinder": "cylinder",
        "proxy": "cylinder",
        "cylinder_proxy": "cylinder",
        "mesh": "mesh",
        "original": "mesh",
        "original_mesh": "mesh",
        "obj": "mesh",
        "single_mesh": "processed_single_hull",
        "processed_single_hull": "processed_single_hull",
        "convex_decomposition": "nvidia_convex",
        "nvidia_convex": "nvidia_convex",
    }
    if text not in aliases:
        raise ValueError(
            f"{path} must be 'cylinder', 'mesh', 'processed_single_hull', or "
            f"'nvidia_convex', got {value!r}"
        )
    return aliases[text]


def _keyboard_friction_from_config(cfg: DictConfig) -> tuple[float, float, float]:
    return _friction_from_config(cfg, "keyboard.friction", (5.0, 0.1, 0.01))


def _keyboard_support_box_height_from_config(cfg: DictConfig) -> float:
    height = float(OmegaConf.select(cfg, "keyboard.support_box_height", default=MINI_KEYBOARD_SUPPORT_BOX_HEIGHT))
    if height < 0.0:
        raise ValueError(f"keyboard.support_box_height must be non-negative, got {height}")
    return height


def _keyboard_key_push_distances_from_config(cfg: DictConfig) -> tuple[float, float]:
    white_distance = float(
        OmegaConf.select(
            cfg,
            "keyboard.white_key_push_distance",
            default=MINI_KEYBOARD_WHITE_KEY_PUSH_DISTANCE,
        )
    )
    black_distance = float(
        OmegaConf.select(
            cfg,
            "keyboard.black_key_push_distance",
            default=MINI_KEYBOARD_BLACK_KEY_PUSH_DISTANCE,
        )
    )
    if white_distance <= 0.0:
        raise ValueError(f"keyboard.white_key_push_distance must be positive, got {white_distance}")
    if black_distance <= 0.0:
        raise ValueError(f"keyboard.black_key_push_distance must be positive, got {black_distance}")
    return white_distance, black_distance


def _keyboard_root_z(support_box_height: float) -> float:
    return float(TABLE_TOP_Z) + float(support_box_height) + float(MINI_KEYBOARD_ROOT_Z_OFFSET)


def _gripper_pad_friction_from_config(cfg: DictConfig) -> tuple[float, float, float]:
    return _friction_from_config(cfg, "env.gripper_pad_friction", (8.0, 0.25, 0.03))


def _barrett_fingertip_friction_from_config(cfg: DictConfig) -> tuple[float, float, float]:
    return _friction_from_config(cfg, "env.barrett_fingertip_friction", (8.0, 0.25, 0.03))


def _wujihand_contact_friction_from_config(cfg: DictConfig) -> tuple[float, float, float]:
    return _friction_from_config(cfg, "env.wujihand_contact_friction", (1.0, 0.005, 0.0001))


def _cube_face_texture_paths_from_config(cfg: DictConfig) -> dict[str, str] | None:
    node = OmegaConf.select(cfg, "render_appearance.cube.face_textures", default=None)
    if node is None:
        return None
    paths: dict[str, str] = {}
    for face in CUBE_TEXTURE_FACE_NAMES:
        path = optional_project_path(
            OmegaConf.select(cfg, f"render_appearance.cube.face_textures.{face}", default=None)
        )
        if path is None:
            raise ValueError(f"render_appearance.cube.face_textures.{face} must be set")
        if not Path(path).exists():
            raise FileNotFoundError(
                f"render_appearance.cube.face_textures.{face} does not exist: {path}"
            )
        paths[face] = path
    unknown = sorted(set(OmegaConf.to_container(node, resolve=True)).difference(CUBE_TEXTURE_FACE_NAMES))
    if unknown:
        raise ValueError(f"Unknown render_appearance.cube.face_textures keys: {unknown}")
    return paths


def _render_appearance_kwargs_from_config(cfg: DictConfig) -> dict[str, Any]:
    if not bool(OmegaConf.select(cfg, "render_appearance.enabled", default=False)):
        return {}
    table_texture_path = optional_project_path(
        OmegaConf.select(cfg, "render_appearance.table.texture_path", default=None)
    )
    if table_texture_path is not None and not Path(table_texture_path).exists():
        raise FileNotFoundError(f"render_appearance.table.texture_path does not exist: {table_texture_path}")
    kwargs = {
        "render_shadows_enabled": bool(
            OmegaConf.select(cfg, "render_appearance.shadows_enabled", default=True)
        ),
        "world_visual_style": str(
            OmegaConf.select(cfg, "render_appearance.world.style", default="mujoco_default")
        ),
        "world_floor_rgba": _rgba_from_config(
            cfg,
            "render_appearance.world.floor_rgba",
            (0.78, 0.80, 0.82, 1.0),
        ),
        "world_wall_rgba": _rgba_from_config(
            cfg,
            "render_appearance.world.wall_rgba",
            (0.70, 0.70, 0.68, 1.0),
        ),
        "world_skybox_rgba": _rgba_from_config(
            cfg,
            "render_appearance.world.skybox_rgba",
            (0.74, 0.77, 0.79, 1.0),
        ),
        "table_visual_style": str(
            OmegaConf.select(cfg, "render_appearance.table.style", default="default")
        ),
        "table_texture_path": table_texture_path,
        "table_texture_fallback": str(
            OmegaConf.select(cfg, "render_appearance.table.texture_fallback", default="none")
        ),
        "table_texrepeat": _table_texrepeat_from_config(cfg),
        "table_rgba": _rgba_from_config(
            cfg,
            "render_appearance.table.rgba",
            (0.55, 0.40, 0.25, 1.0),
        ),
        "table_specular": float(
            OmegaConf.select(cfg, "render_appearance.table.material.specular", default=0.06)
        ),
        "table_shininess": float(
            OmegaConf.select(cfg, "render_appearance.table.material.shininess", default=0.08)
        ),
        "table_reflectance": float(
            OmegaConf.select(cfg, "render_appearance.table.material.reflectance", default=0.0)
        ),
        "wuji_right_hand_rgba": _rgba_from_config(
            cfg,
            "render_appearance.wujihand.right_hand_rgba",
            (0.015, 0.015, 0.014, 1.0),
        ),
        "wuji_right_hand_specular": float(
            OmegaConf.select(cfg, "render_appearance.wujihand.material.specular", default=0.18)
        ),
        "wuji_right_hand_shininess": float(
            OmegaConf.select(cfg, "render_appearance.wujihand.material.shininess", default=0.20)
        ),
        "wuji_right_hand_reflectance": float(
            OmegaConf.select(cfg, "render_appearance.wujihand.material.reflectance", default=0.03)
        ),
    }
    if bool(OmegaConf.select(cfg, "render_appearance.cube.enabled", default=False)):
        cube_texture_path = optional_project_path(
            OmegaConf.select(cfg, "render_appearance.cube.texture_path", default=None)
        )
        if cube_texture_path is not None and not Path(cube_texture_path).exists():
            raise FileNotFoundError(f"render_appearance.cube.texture_path does not exist: {cube_texture_path}")
        cube_face_texture_paths = _cube_face_texture_paths_from_config(cfg)
        cube_face_texture_atlas_path = optional_project_path(
            OmegaConf.select(cfg, "render_appearance.cube.face_texture_atlas", default=None)
        )
        if cube_face_texture_atlas_path is not None and not Path(cube_face_texture_atlas_path).exists():
            raise FileNotFoundError(
                f"render_appearance.cube.face_texture_atlas does not exist: {cube_face_texture_atlas_path}"
            )
        kwargs.update(
            {
                "cube_texture_path": cube_texture_path,
                "cube_face_texture_paths": cube_face_texture_paths,
                "cube_face_texture_atlas_path": cube_face_texture_atlas_path,
                "cube_visual_face_offset": float(
                    OmegaConf.select(cfg, "render_appearance.cube.visual_face_offset", default=8.0e-5)
                ),
                "cube_texrepeat": _pair_from_config(
                    cfg,
                    "render_appearance.cube.texrepeat",
                    (1.0, 1.0),
                ),
                "cube_rgba": _rgba_from_config(
                    cfg,
                    "render_appearance.cube.rgba",
                    (0.92, 0.22, 0.12, 1.0),
                ),
                "cube_specular": float(
                    OmegaConf.select(cfg, "render_appearance.cube.material.specular", default=0.16)
                ),
                "cube_shininess": float(
                    OmegaConf.select(cfg, "render_appearance.cube.material.shininess", default=0.18)
                ),
                "cube_reflectance": float(
                    OmegaConf.select(cfg, "render_appearance.cube.material.reflectance", default=0.02)
                ),
            }
        )
    return kwargs


def _allegro_hand_friction_from_config(cfg: DictConfig) -> tuple[float, float, float]:
    return _friction_from_config(cfg, "env.allegro_hand_friction", (1.0, 0.005, 0.0001))


def _robotiq_pad1_friction_from_config(cfg: DictConfig) -> tuple[float, float, float]:
    return _friction_from_config(cfg, "env.robotiq_pad1_friction", FRANKA_ROBOTIQ_PAD1_FRICTION)


def _robotiq_pad2_friction_from_config(cfg: DictConfig) -> tuple[float, float, float]:
    return _friction_from_config(cfg, "env.robotiq_pad2_friction", FRANKA_ROBOTIQ_PAD2_FRICTION)


def _robotiq_pad_condim_from_config(cfg: DictConfig) -> int:
    return _condim_from_config(cfg, "env.robotiq_pad_condim", FRANKA_ROBOTIQ_PAD_CONDIM)


def _robotiq_pad_priority_from_config(cfg: DictConfig) -> int:
    return _priority_from_config(cfg, "env.robotiq_pad_priority", FRANKA_ROBOTIQ_PAD_PRIORITY)


def _robotiq_pad_solref_from_config(cfg: DictConfig) -> tuple[float, float]:
    return _pair_from_config(cfg, "env.robotiq_pad_solref", FRANKA_ROBOTIQ_PAD_SOLREF)


def _robotiq_pad_solimp_from_config(cfg: DictConfig) -> tuple[float, float, float, float, float]:
    return _five_tuple_from_config(cfg, "env.robotiq_pad_solimp", FRANKA_ROBOTIQ_PAD_SOLIMP)


def _allegro_hand_condim_from_config(cfg: DictConfig) -> int:
    return _condim_from_config(cfg, "env.allegro_hand_condim", 3)


def _allegro_hand_priority_from_config(cfg: DictConfig) -> int:
    return _priority_from_config(cfg, "env.allegro_hand_priority", 0)


def _allegro_hand_solref_from_config(cfg: DictConfig) -> tuple[float, float]:
    return _pair_from_config(cfg, "env.allegro_hand_solref", (0.02, 1.0))


def _allegro_hand_solimp_from_config(cfg: DictConfig) -> tuple[float, float, float, float, float]:
    return _five_tuple_from_config(cfg, "env.allegro_hand_solimp", (0.9, 0.95, 0.001, 0.5, 2.0))


def _allegro_fingertip_friction_from_config(cfg: DictConfig) -> tuple[float, float, float]:
    return _friction_from_config(cfg, "env.allegro_fingertip_pad_friction", (8.0, 0.25, 0.03))


def _allegro_fingertip_radius_from_config(cfg: DictConfig) -> float:
    value = float(
        OmegaConf.select(
            cfg,
            "env.allegro_fingertip_pad_radius",
            default=ALLEGRO_FINGERTIP_PAD_RADIUS,
        )
    )
    if value <= 0.0:
        raise ValueError(f"env.allegro_fingertip_pad_radius must be positive, got {value}")
    return value


def _allegro_fingertip_condim_from_config(cfg: DictConfig) -> int:
    return _condim_from_config(cfg, "env.allegro_fingertip_pad_condim", 6)


def _allegro_fingertip_priority_from_config(cfg: DictConfig) -> int:
    return _priority_from_config(cfg, "env.allegro_fingertip_pad_priority", 2)


def _allegro_fingertip_solref_from_config(cfg: DictConfig) -> tuple[float, float]:
    return _pair_from_config(cfg, "env.allegro_fingertip_pad_solref", (0.002, 1.0))


def _allegro_fingertip_solimp_from_config(cfg: DictConfig) -> tuple[float, float, float, float, float]:
    return _five_tuple_from_config(
        cfg,
        "env.allegro_fingertip_pad_solimp",
        (0.95, 0.99, 0.001, 0.5, 2.0),
    )


def _gripper_pad_patch_grid_from_config(cfg: DictConfig) -> tuple[int, int]:
    value = OmegaConf.select(cfg, "env.gripper_pad_patch_grid", default=(1, 1))
    grid = np.asarray(list(value), dtype=np.int64).reshape(-1)
    if grid.shape != (2,):
        raise ValueError(f"env.gripper_pad_patch_grid must contain 2 values, got shape {grid.shape}")
    if np.any(grid <= 0):
        raise ValueError(f"env.gripper_pad_patch_grid must be positive, got {grid.tolist()}")
    return tuple(int(x) for x in grid)


def _pump_button_joint_range_from_config(cfg: DictConfig) -> tuple[float, float]:
    value = OmegaConf.select(cfg, "pump_bottle.button_joint_range", default=(-0.013, 0.0))
    joint_range = np.asarray(list(value), dtype=np.float64).reshape(-1)
    if joint_range.shape != (2,):
        raise ValueError(f"pump_bottle.button_joint_range must contain 2 values, got shape {joint_range.shape}")
    if not float(joint_range[0]) < float(joint_range[1]):
        raise ValueError(f"pump_bottle.button_joint_range must be increasing [lo, hi], got {joint_range.tolist()}")
    return float(joint_range[0]), float(joint_range[1])


def _pump_bottle_active_id_from_config(cfg: DictConfig) -> str:
    configured = OmegaConf.select(cfg, "pump_bottle.active_id", default=None)
    if configured is not None and str(configured).strip():
        return str(configured).strip()
    mode = str(OmegaConf.select(cfg, "pump_bottle.mode", default="mixed_three")).strip().lower()
    if mode == "single_005":
        return "pump_bottle005"
    if mode != "mixed_three":
        raise ValueError(f"pump_bottle.mode must be 'mixed_three' or 'single_005', got {mode!r}")
    object_ids = OmegaConf.select(
        cfg,
        "pump_bottle.asset_ids",
        default=("SoapDispenser003", "SoapDispenser004", "SoapDispenser007"),
    )
    ids = [str(value).strip() for value in object_ids if str(value).strip()]
    if not ids:
        raise ValueError("pump_bottle.asset_ids must contain at least one ID in mixed_three mode")
    return ids[0]


def _pump_bottle_contact_margin_from_config(cfg: DictConfig) -> float:
    value = float(OmegaConf.select(cfg, "pump_bottle.contact_margin", default=0.0))
    if value < 0.0:
        raise ValueError(f"pump_bottle.contact_margin must be non-negative, got {value}")
    return value


def validate_task_config(cfg: DictConfig) -> None:
    cube_height_ratio = float(OmegaConf.select(cfg, "cube.height_ratio", default=1.0))
    if cube_height_ratio <= 0.0:
        raise ValueError(f"cube.height_ratio must be positive, got {cube_height_ratio}")
    cube_mass = float(OmegaConf.select(cfg, "cube.mass", default=0.08))
    if cube_mass <= 0.0:
        raise ValueError(f"cube.mass must be positive, got {cube_mass}")
    _cube_friction_from_config(cfg)
    max_half = 0.5 * float(cfg.cube.side_length_range[1])
    table_half_extents = _table_half_extents_from_config(cfg)
    x_min = float(TABLE_CENTER[0]) - float(table_half_extents[0]) + max_half
    x_max = float(TABLE_CENTER[0]) + float(table_half_extents[0]) - max_half
    y_min = float(TABLE_CENTER[1]) - float(table_half_extents[1]) + max_half
    y_max = float(TABLE_CENTER[1]) + float(table_half_extents[1]) - max_half
    if float(cfg.cube.x_range[0]) < x_min or float(cfg.cube.x_range[1]) > x_max:
        raise ValueError(f"cube.x_range={cfg.cube.x_range} exceeds table-safe range [{x_min:.4f}, {x_max:.4f}]")
    if float(cfg.cube.y_range[0]) < y_min or float(cfg.cube.y_range[1]) > y_max:
        raise ValueError(f"cube.y_range={cfg.cube.y_range} exceeds table-safe range [{y_min:.4f}, {y_max:.4f}]")
    _cube_lift_success_z_from_config(cfg)


def _cube_lift_success_z_from_config(cfg: DictConfig) -> float:
    value = OmegaConf.select(cfg, "task.success_lift_metric", default=None)
    if value is None:
        value = OmegaConf.select(cfg, "task.sucess_lift_metric", default=0.12)
    lift_metric = float(value)
    if lift_metric <= 0.0:
        raise ValueError(f"task.success_lift_metric must be positive, got {lift_metric}")
    return float(TABLE_TOP_Z) + lift_metric


def validate_pump_bottle_task_config(cfg: DictConfig) -> None:
    mass = float(OmegaConf.select(cfg, "pump_bottle.mass", default=0.18))
    if mass <= 0.0:
        raise ValueError(f"pump_bottle.mass must be positive, got {mass}")
    button_mass = float(OmegaConf.select(cfg, "pump_bottle.button_mass", default=0.05))
    if button_mass <= 0.0:
        raise ValueError(f"pump_bottle.button_mass must be positive, got {button_mass}")
    button_joint_range = _pump_button_joint_range_from_config(cfg)
    button_joint_stiffness = float(OmegaConf.select(cfg, "pump_bottle.button_joint_stiffness", default=35.0))
    if button_joint_stiffness < 0.0:
        raise ValueError(f"pump_bottle.button_joint_stiffness must be non-negative, got {button_joint_stiffness}")
    button_joint_damping = float(OmegaConf.select(cfg, "pump_bottle.button_joint_damping", default=5.0))
    if button_joint_damping < 0.0:
        raise ValueError(f"pump_bottle.button_joint_damping must be non-negative, got {button_joint_damping}")
    button_actuator_kp = float(OmegaConf.select(cfg, "pump_bottle.button_actuator_kp", default=35.0))
    if button_actuator_kp < 0.0:
        raise ValueError(f"pump_bottle.button_actuator_kp must be non-negative, got {button_actuator_kp}")
    mesh_scale = float(OmegaConf.select(cfg, "pump_bottle.mesh_scale", default=1.3))
    if mesh_scale <= 0.0:
        raise ValueError(f"pump_bottle.mesh_scale must be positive, got {mesh_scale}")
    _pump_bottle_contact_margin_from_config(cfg)
    safe_radius = float(OmegaConf.select(cfg, "pump_bottle.safe_radius", default=0.05))
    if safe_radius <= 0.0:
        raise ValueError(f"pump_bottle.safe_radius must be positive, got {safe_radius}")
    button_success_slide = float(OmegaConf.select(cfg, "pump_bottle.button_success_slide", default=-0.012))
    if not (button_joint_range[0] <= button_success_slide <= button_joint_range[1]):
        raise ValueError(
            "pump_bottle.button_success_slide must be within "
            f"button_joint_range={button_joint_range}, got {button_success_slide}"
        )
    button_reset_slide = float(OmegaConf.select(cfg, "pump_bottle.button_reset_slide", default=-0.00700733))
    if not (button_joint_range[0] <= button_reset_slide <= button_joint_range[1]):
        raise ValueError(
            "pump_bottle.button_reset_slide must be within "
            f"button_joint_range={button_joint_range}, got {button_reset_slide}"
        )
    lift_success_delta = float(OmegaConf.select(cfg, "pump_bottle.lift_success_delta", default=0.08))
    if lift_success_delta <= 0.0:
        raise ValueError(f"pump_bottle.lift_success_delta must be positive, got {lift_success_delta}")
    _pump_bottle_friction_from_config(cfg)
    table_half_extents = _table_half_extents_from_config(cfg)
    x_min = float(TABLE_CENTER[0]) - float(table_half_extents[0]) + safe_radius
    x_max = float(TABLE_CENTER[0]) + float(table_half_extents[0]) - safe_radius
    y_min = float(TABLE_CENTER[1]) - float(table_half_extents[1]) + safe_radius
    y_max = float(TABLE_CENTER[1]) + float(table_half_extents[1]) - safe_radius
    if float(cfg.pump_bottle.x_range[0]) < x_min or float(cfg.pump_bottle.x_range[1]) > x_max:
        raise ValueError(
            f"pump_bottle.x_range={cfg.pump_bottle.x_range} exceeds table-safe range [{x_min:.4f}, {x_max:.4f}]"
        )
    if float(cfg.pump_bottle.y_range[0]) < y_min or float(cfg.pump_bottle.y_range[1]) > y_max:
        raise ValueError(
            f"pump_bottle.y_range={cfg.pump_bottle.y_range} exceeds table-safe range [{y_min:.4f}, {y_max:.4f}]"
        )


def validate_keyboard_task_config(cfg: DictConfig) -> None:
    _keyboard_support_box_height_from_config(cfg)
    _keyboard_key_push_distances_from_config(cfg)
    safe_radius = float(OmegaConf.select(cfg, "keyboard.safe_radius", default=0.16))
    if safe_radius <= 0.0:
        raise ValueError(f"keyboard.safe_radius must be positive, got {safe_radius}")
    success_fraction = float(OmegaConf.select(cfg, "keyboard.key_success_fraction", default=0.8))
    if not (0.0 < success_fraction <= 1.0):
        raise ValueError(f"keyboard.key_success_fraction must be in (0, 1], got {success_fraction}")
    target_names = tuple(
        str(x)
        for x in OmegaConf.select(
            cfg,
            "keyboard.target_key_joint_names",
            default=("white_key_00_C2_slide", "white_key_01_D2_slide", "white_key_02_E2_slide"),
        )
    )
    if len(target_names) != 3 or len(set(target_names)) != 3:
        raise ValueError(f"keyboard.target_key_joint_names must contain 3 distinct names, got {target_names}")
    _keyboard_friction_from_config(cfg)
    table_half_extents = _table_half_extents_from_config(cfg)
    x_min = float(TABLE_CENTER[0]) - float(table_half_extents[0]) + safe_radius
    x_max = float(TABLE_CENTER[0]) + float(table_half_extents[0]) - safe_radius
    y_min = float(TABLE_CENTER[1]) - float(table_half_extents[1]) + safe_radius
    y_max = float(TABLE_CENTER[1]) + float(table_half_extents[1]) - safe_radius
    if float(cfg.keyboard.x_range[0]) < x_min or float(cfg.keyboard.x_range[1]) > x_max:
        raise ValueError(
            f"keyboard.x_range={cfg.keyboard.x_range} exceeds table-safe range [{x_min:.4f}, {x_max:.4f}]"
        )
    if float(cfg.keyboard.y_range[0]) < y_min or float(cfg.keyboard.y_range[1]) > y_max:
        raise ValueError(
            f"keyboard.y_range={cfg.keyboard.y_range} exceeds table-safe range [{y_min:.4f}, {y_max:.4f}]"
        )


def _float_sequence(value: Any, expected: int) -> np.ndarray:
    arr = np.asarray(list(value), dtype=np.float64).reshape(-1)
    if arr.shape != (int(expected),):
        raise ValueError(f"Expected {expected} values, got shape {arr.shape}")
    return arr
