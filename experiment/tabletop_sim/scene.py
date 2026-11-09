"""Procedural MuJoCo scene assembly for tabletop manipulation tasks."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from .assets import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CUBE_HALF_EXTENT,
    CUBE_START_POS,
    CUBE_START_QUAT_WXYZ,
    MINI_KEYBOARD_MJCF_PATH,
    MINI_KEYBOARD_ROOT_Z_OFFSET,
    MINI_KEYBOARD_SUPPORT_BOX_HEIGHT,
    PUMP_BOTTLE_MESH_DIR,
    ROBOTIQ_2F85_XML,
    TABLE_CENTER,
    TABLE_HALF_EXTENTS,
    TABLE_TOP_Z,
    require_path,
)
from .perception import DEFAULT_CAMERA_NAMES, HAND_CAMERA_NAME, default_camera_specs
from .mini_keyboard_asset import attach_mini_keyboard_asset
from .pump_bottle_suite import LEGACY_PUMP_BOTTLE_ID, pump_bottle_spec_by_id
from .robot import EmbodimentSpec, SiteAttachmentSpec, get_embodiment_spec
from .rotation import FixedBaseRotationChart
from .robot import (
    ALLEGRO_HAND_JOINT_NAMES,
    ALLEGRO_ROOT_JOINT_NAMES,
    BARRETT_HAND_JOINT_NAMES,
    BARRETT_ROOT_JOINT_NAMES,
    WUJIHAND_HAND_JOINT_NAMES,
    WUJIHAND_ROOT_JOINT_NAMES,
)


ARMLESS_ROOT_SUFFIXES: tuple[str, ...] = ("x", "y", "z", "roll", "pitch", "yaw")
ONLINE_ROLLOUT_RENDER_CAMERA_NAME = "online_rollout_render_cam"
ARMLESS_GRIPPER_ROOT_JOINT_NAMES: tuple[str, ...] = tuple(
    f"robotiq_gripper_{suffix}" for suffix in ARMLESS_ROOT_SUFFIXES
)
ARMLESS_STICK_ROOT_JOINT_NAMES: tuple[str, ...] = tuple(
    f"robotiq_stick_{suffix}" for suffix in ARMLESS_ROOT_SUFFIXES
)
ARMLESS_ROBOTIQ_INTERNAL_JOINT_NAMES: tuple[str, ...] = (
    "g_left_driver_joint",
    "g_left_spring_link_joint",
    "g_left_follower_joint",
    "g_right_driver_joint",
    "g_right_spring_link_joint",
    "g_right_follower_joint",
)
ARMLESS_ROOT_CTRL_RANGES: tuple[tuple[float, float], ...] = (
    (-1.0, 1.0),
    (-1.0, 1.0),
    # Tabletop gripper/stick poses use world Z above 1 m. Keeping the generic
    # [-1, 1] limit here silently clipped their position-actuator targets and
    # pulled a correctly loaded start pose downward on the first control step.
    (-2.0, 2.0),
    (-6.28, 6.28),
    (-6.28, 6.28),
    (-6.28, 6.28),
)
ARMLESS_ROOT_KP: tuple[float, ...] = (300.0, 300.0, 300.0, 40.0, 40.0, 40.0)
ARMLESS_ROOT_KV: tuple[float, ...] = (25.0, 25.0, 25.0, 4.0, 4.0, 4.0)
ARMLESS_ROOT_FORCE_LIMITS: tuple[float, ...] = (30.0, 30.0, 30.0, 2.0, 2.0, 2.0)
ARMLESS_ROBOTIQ_CLOSED_JOINT_VALUE = 0.824
ARMLESS_ROBOTIQ_JOINT_KP = 8.0
ARMLESS_ROBOTIQ_JOINT_KV = 0.25
ARMLESS_ROBOTIQ_JOINT_FORCE_LIMIT = 2.0
ARMLESS_ROBOTIQ_PAD_FRICTION: tuple[float, float, float] = (8.0, 0.25, 0.03)
ARMLESS_ROBOTIQ_PAD_PATCH_GRID: tuple[int, int] = (1, 1)
ARMLESS_ROBOTIQ_PAD_PATCH_FILL = 0.98
BARRETT_FINGERTIP_PAD_RADIUS = 0.03
BARRETT_FINGERTIP_PAD_FRICTION: tuple[float, float, float] = ARMLESS_ROBOTIQ_PAD_FRICTION
BARRETT_FINGERTIP_CAPSULE_RADIUS = 0.014
BARRETT_FINGERTIP_CAPSULE_FROMTO: tuple[float, ...] = (
    0.007451,
    -0.005472,
    0.0006,
    -0.04464,
    0.036259,
    0.0006,
)
BARRETT_FINGERTIP_CAPSULE_FRICTION: tuple[float, float, float] = ARMLESS_ROBOTIQ_PAD_FRICTION
BARRETT_FINGERTIP_COLLISION_MODEL_CAPSULE = "capsule"
BARRETT_FINGERTIP_COLLISION_MODEL_MESH = "mesh"
ALLEGRO_FINGERTIP_PAD_RADIUS = 0.018
ALLEGRO_FINGERTIP_PAD_FRICTION: tuple[float, float, float] = ARMLESS_ROBOTIQ_PAD_FRICTION
ALLEGRO_HAND_FRICTION: tuple[float, float, float] = (1.0, 0.005, 0.0001)
ALLEGRO_HAND_COLLISION_MODEL_FULL = "full"
ALLEGRO_HAND_COLLISION_MODEL_FINGERTIP_MESH_ONLY = "fingertip_mesh_only"
FRANKA_ROBOTIQ_PAD1_FRICTION: tuple[float, float, float] = (0.7, 0.005, 0.0001)
FRANKA_ROBOTIQ_PAD2_FRICTION: tuple[float, float, float] = (0.6, 0.005, 0.0001)
FRANKA_ROBOTIQ_PAD_CONDIM = 3
FRANKA_ROBOTIQ_PAD_PRIORITY = 1
FRANKA_ROBOTIQ_PAD_SOLREF: tuple[float, float] = (0.004, 1.0)
FRANKA_ROBOTIQ_PAD_SOLIMP: tuple[float, float, float, float, float] = (0.95, 0.99, 0.001, 0.5, 2.0)
OBJECT_SUITE_BIN_FRICTION: tuple[float, float, float] = (0.8, 0.005, 0.0001)
ALLEGRO_HAND_CONDIM = 3
ALLEGRO_HAND_PRIORITY = 0
ALLEGRO_HAND_SOLREF: tuple[float, float] = (0.02, 1.0)
ALLEGRO_HAND_SOLIMP: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0)
ALLEGRO_FINGERTIP_PAD_CONDIM = 6
ALLEGRO_FINGERTIP_PAD_PRIORITY = 2
ALLEGRO_FINGERTIP_PAD_SOLREF: tuple[float, float] = (0.002, 1.0)
ALLEGRO_FINGERTIP_PAD_SOLIMP: tuple[float, float, float, float, float] = (0.95, 0.99, 0.001, 0.5, 2.0)
FLOATING_ASSET_VIRTUAL_BODY_MASS = 1.0e-4
FLOATING_ASSET_VIRTUAL_BODY_INERTIA: tuple[float, float, float] = (1.0e-6, 1.0e-6, 1.0e-6)
FLOATING_ROOT_JOINT_ARMATURE = 1.0
FLOATING_ROOT_JOINT_DAMPING = 0.1
FLOATING_HAND_JOINT_ARMATURE = 0.001
FLOATING_HAND_JOINT_DAMPING = 0.02
ALLEGRO_ROOT_CTRL_RANGES: tuple[tuple[float, float], ...] = (
    (-2.0, 2.0),
    (-2.0, 2.0),
    (-2.0, 2.0),
    (-6.28, 6.28),
    (-6.28, 6.28),
    (-6.28, 6.28),
)
ALLEGRO_ROOT_KP: tuple[float, ...] = ARMLESS_ROOT_KP
ALLEGRO_ROOT_KV: tuple[float, ...] = ARMLESS_ROOT_KV
ALLEGRO_ROOT_FORCE_LIMITS: tuple[float, ...] = ARMLESS_ROOT_FORCE_LIMITS
WUJIHAND_LIKE_HAND_KP = 0.5
WUJIHAND_LIKE_HAND_KV = 0.0
WUJIHAND_LIKE_HAND_FORCE_LIMIT = 0.65
ALLEGRO_HAND_KP = 3.0
ALLEGRO_HAND_KV = 2.0
ALLEGRO_HAND_FORCE_LIMIT = 0.65
BARRETT_ROOT_KP: tuple[float, ...] = ARMLESS_ROOT_KP
BARRETT_ROOT_KV: tuple[float, ...] = ARMLESS_ROOT_KV
BARRETT_ROOT_FORCE_LIMITS: tuple[float, ...] = ARMLESS_ROOT_FORCE_LIMITS
BARRETT_HAND_KP = 100.0
BARRETT_HAND_KV = 5.0
BARRETT_HAND_FORCE_LIMIT = 20.0
WUJIHAND_CONTACT_FRICTION: tuple[float, float, float] = (1.0, 0.005, 0.0001)
WUJIHAND_CONTACT_CONDIM = 3
WUJIHAND_HAND_JOINT_ARMATURE = 0.005
WUJIHAND_HAND_JOINT_DAMPING = 0.05
WUJIHAND_HAND_JOINT_FRICTIONLOSS = 0.01
CUBE_TEXTURE_FACE_NAMES: tuple[str, ...] = ("front", "back", "left", "right", "top", "bottom")
CUBE_TEXTURE_FACE_GEOM_NAMES: tuple[str, ...] = tuple(
    f"cube_visual_{face}" for face in CUBE_TEXTURE_FACE_NAMES
)
CUBE_TEXTURE_FACE_MESH_NAMES: tuple[str, ...] = tuple(
    f"cube_visual_{face}_mesh" for face in CUBE_TEXTURE_FACE_NAMES
)
CUBE_TEXTURE_ATLAS_GEOM_NAME = "cube_visual_box"
CUBE_TEXTURE_ATLAS_MESH_NAME = "cube_visual_box_mesh"
CUBE_TEXTURE_FACE_OFFSET = 8.0e-5


@dataclass(frozen=True)
class SceneSpec:
    """Names and paths used by the compiled tabletop scene."""

    robot_xml_path: Path
    embodiment: str = "marvin_wuji"
    object_kind: str = "cube"
    cube_body_name: str = "cube"
    cube_joint_name: str = "cube_free"
    cube_geom_name: str = "cube_geom"
    cube_visual_face_geom_names: tuple[str, ...] = ()
    cube_visual_face_mesh_names: tuple[str, ...] = ()
    cube_visual_box_mesh_name: str | None = None
    cube_visual_face_offset: float = CUBE_TEXTURE_FACE_OFFSET
    object_body_name: str = "task_object"
    object_joint_name: str = "task_object_free"
    object_geom_name: str = "task_object_collision"
    bin_body_name: str = "pick_place_bin"
    bin_floor_geom_name: str = "pick_place_bin_floor"
    pump_bottle_body_name: str = "pump_bottle_bottle_base"
    pump_bottle_free_joint_name: str = "pump_bottle_bottle_free"
    pump_bottle_button_body_name: str = "pump_bottle_button"
    pump_bottle_button_joint_name: str = "pump_bottle_button_slide"
    pump_bottle_button_site_name: str = "pump_bottle_button_force_site"
    pump_bottle_asset_id: str = LEGACY_PUMP_BOTTLE_ID
    mini_keyboard_body_name: str = "mini_keyboard"
    mini_keyboard_key_joint_names: tuple[str, ...] = ()
    robotiq_gripper_root_joint_names: tuple[str, ...] = ARMLESS_GRIPPER_ROOT_JOINT_NAMES
    robotiq_gripper_root_body_name: str = "g_base"
    robotiq_stick_body_name: str = "robotiq_stick"
    robotiq_stick_root_joint_names: tuple[str, ...] = ARMLESS_STICK_ROOT_JOINT_NAMES
    robotiq_stick_geom_name: str = "robotiq_stick_geom"
    table_body_name: str = "table"
    table_geom_name: str = "table_top"
    table_half_extents: tuple[float, float, float] = TABLE_HALF_EXTENTS
    camera_names: tuple[str, ...] = DEFAULT_CAMERA_NAMES


PUMP_BOTTLE_MESHES: tuple[tuple[str, str, tuple[float, float, float, float]], ...] = (
    (
        "pump_bottle_root_E_body_2_P_84d492d7b772968b",
        "root_E_body_2_P_84d492d7b772968b.obj",
        (0.72, 0.78, 0.86, 1.0),
    ),
    (
        "pump_bottle_root_E_body_2_P_54317484ca82968b",
        "root_E_body_2_P_54317484ca82968b.obj",
        (0.72, 0.78, 0.86, 1.0),
    ),
    (
        "pump_bottle_root_E_body_2_E_cap_3_P_c3eef779b677968b",
        "root_E_body_2_E_cap_3_P_c3eef779b677968b.obj",
        (0.72, 0.78, 0.86, 1.0),
    ),
    (
        "pump_bottle_root_E_body_2_E_cap_3_P_bf0b2a047372dadc",
        "root_E_body_2_E_cap_3_P_bf0b2a047372dadc.obj",
        (0.72, 0.78, 0.86, 1.0),
    ),
)
PUMP_BOTTLE_BUTTON_MESH = (
    "pump_bottle_root_E_pump_1_P_5732c6c18672a9fd",
    "root_E_pump_1_P_5732c6c18672a9fd.obj",
    (0.95, 0.36, 0.25, 1.0),
)
PUMP_BOTTLE_BUTTON_EXTENSION_GEOM_NAME = "pump_bottle_button_push_extension"
PUMP_BOTTLE_BUTTON_EXTENSION_LENGTH = 0.052
PUMP_BOTTLE_BUTTON_EXTENSION_OVERLAP = 0.004
PUMP_BOTTLE_BUTTON_EXTENSION_HALF_WIDTH = 0.0105
PUMP_BOTTLE_BUTTON_EXTENSION_HALF_THICKNESS = 0.006
PUMP_BOTTLE_BUTTON_EXTENSION_DIRECTION = -1.0
PUMP_BOTTLE_BODY_MESH_NAME = "pump_bottle_root_E_body_2_P_84d492d7b772968b"
PUMP_BOTTLE_BODY_COLLISION_GEOM_NAME = "pump_bottle_body_collision"
PUMP_BOTTLE_NECK_MESH_NAME = "pump_bottle_root_E_body_2_E_cap_3_P_bf0b2a047372dadc"
PUMP_BOTTLE_NECK_COLLISION_GEOM_NAME = "pump_bottle_neck_collision"
PUMP_BOTTLE_PROXY_RGBA = (0.10, 0.75, 1.00, 0.28)
PUMP_BOTTLE_PROXY_COLLISION_GROUP = 3
PUMP_BOTTLE_COLLISION_MODE_CYLINDER = "cylinder"
PUMP_BOTTLE_COLLISION_MODE_MESH = "mesh"
PUMP_BOTTLE_VISUAL_MESH_SOURCE_VISUAL = "visual"
PUMP_BOTTLE_VISUAL_MESH_SOURCE_COLLISION = "collision"
PUMP_BOTTLE_CONTACT_MARGIN = 0.0
PUMP_BOTTLE_SUITE_COLLISION_PROCESSED_SINGLE_HULL = "processed_single_hull"
PUMP_BOTTLE_SUITE_COLLISION_NVIDIA_CONVEX = "nvidia_convex"

MINI_KEYBOARD_WHITE_NOTES = (
    "C2",
    "D2",
    "E2",
    "F2",
    "G2",
    "A2",
    "B2",
    "C3",
    "D3",
    "E3",
    "F3",
    "G3",
    "A3",
    "B3",
    "C4",
)
MINI_KEYBOARD_BLACK_KEY_SPECS = (
    (0.5, "Cs2"),
    (1.5, "Ds2"),
    (3.5, "Fs2"),
    (4.5, "Gs2"),
    (5.5, "As2"),
    (7.5, "Cs3"),
    (8.5, "Ds3"),
    (10.5, "Fs3"),
    (11.5, "Gs3"),
    (12.5, "As3"),
)
MINI_KEYBOARD_WHITE_KEY_PUSH_DISTANCE = 0.006
MINI_KEYBOARD_BLACK_KEY_PUSH_DISTANCE = 0.005
MINI_KEYBOARD_WHITE_KEY_NAMES = tuple(
    f"white_key_{idx:02d}_{note}_slide" for idx, note in enumerate(MINI_KEYBOARD_WHITE_NOTES)
)
MINI_KEYBOARD_BLACK_KEY_NAMES = tuple(
    f"black_key_{idx:02d}_{note}_slide" for idx, (_, note) in enumerate(MINI_KEYBOARD_BLACK_KEY_SPECS)
)
MINI_KEYBOARD_KEY_JOINT_NAMES = (*MINI_KEYBOARD_WHITE_KEY_NAMES, *MINI_KEYBOARD_BLACK_KEY_NAMES)


def look_at_quat_wxyz(cam_pos: tuple[float, float, float], target: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Return a MuJoCo camera quaternion that looks from ``cam_pos`` to ``target``."""

    fwd = np.asarray(target, dtype=np.float64) - np.asarray(cam_pos, dtype=np.float64)
    fwd /= max(float(np.linalg.norm(fwd)), 1e-9)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(fwd, world_up)
    if float(np.linalg.norm(right)) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rot = np.column_stack([right, up, -fwd])
    trace = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if trace > 0.0:
        scale = 0.5 / math.sqrt(trace + 1.0)
        return (
            0.25 / scale,
            (rot[2, 1] - rot[1, 2]) * scale,
            (rot[0, 2] - rot[2, 0]) * scale,
            (rot[1, 0] - rot[0, 1]) * scale,
        )
    axis = int(np.argmax([rot[0, 0], rot[1, 1], rot[2, 2]]))
    if axis == 0:
        scale = 2.0 * math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        return (
            (rot[2, 1] - rot[1, 2]) / scale,
            0.25 * scale,
            (rot[0, 1] + rot[1, 0]) / scale,
            (rot[0, 2] + rot[2, 0]) / scale,
        )
    if axis == 1:
        scale = 2.0 * math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        return (
            (rot[0, 2] - rot[2, 0]) / scale,
            (rot[0, 1] + rot[1, 0]) / scale,
            0.25 * scale,
            (rot[1, 2] + rot[2, 1]) / scale,
        )
    scale = 2.0 * math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
    return (
        (rot[1, 0] - rot[0, 1]) / scale,
        (rot[0, 2] + rot[2, 0]) / scale,
        (rot[1, 2] + rot[2, 1]) / scale,
        0.25 * scale,
    )


def build_tabletop_spec(
    *,
    add_cameras: bool = True,
    embodiment: str = "marvin_wuji",
    object_kind: str = "cube",
    cube_half_extent: float = CUBE_HALF_EXTENT,
    cube_mass: float = 0.08,
    cube_friction: tuple[float, float, float] = (1.2, 0.05, 0.01),
    object_visual_mesh_path: str | Path | None = None,
    object_collision_mesh_path: str | Path | None = None,
    object_collision_mesh_paths: Sequence[str | Path] | None = None,
    object_texture_path: str | Path | None = None,
    object_mass: float = 0.1,
    object_friction: tuple[float, float, float] = (2.0, 0.05, 0.01),
    object_mesh_min_z: float = 0.0,
    bin_friction: tuple[float, float, float] = OBJECT_SUITE_BIN_FRICTION,
    bin_center_xy: tuple[float, float] = (0.24, 0.30),
    bin_inner_size: tuple[float, float] = (0.22, 0.22),
    bin_wall_height: float = 0.10,
    bin_wall_thickness: float = 0.008,
    timestep: float = 0.002,
    pump_bottle_mass: float = 0.18,
    pump_button_mass: float = 0.05,
    pump_button_joint_range: tuple[float, float] = (-0.013, 0.0),
    pump_button_joint_stiffness: float = 35.0,
    pump_button_joint_damping: float = 5.0,
    pump_button_actuator_kp: float = 35.0,
    pump_button_force_site_visible: bool = True,
    pump_bottle_mesh_scale: float = 1.3,
    pump_bottle_friction: tuple[float, float, float] = (2.0, 0.01, 0.0002),
    pump_bottle_body_collision_mode: str = PUMP_BOTTLE_COLLISION_MODE_CYLINDER,
    pump_bottle_neck_collision_mode: str = PUMP_BOTTLE_COLLISION_MODE_CYLINDER,
    pump_bottle_visual_mesh_source: str = PUMP_BOTTLE_VISUAL_MESH_SOURCE_VISUAL,
    pump_bottle_contact_margin: float = PUMP_BOTTLE_CONTACT_MARGIN,
    pump_bottle_asset_id: str = LEGACY_PUMP_BOTTLE_ID,
    pump_bottle_manifest_path: str | Path | None = None,
    mini_keyboard_friction: tuple[float, float, float] = (5.0, 0.1, 0.01),
    mini_keyboard_support_box_height: float = MINI_KEYBOARD_SUPPORT_BOX_HEIGHT,
    mini_keyboard_white_key_push_distance: float = MINI_KEYBOARD_WHITE_KEY_PUSH_DISTANCE,
    mini_keyboard_black_key_push_distance: float = MINI_KEYBOARD_BLACK_KEY_PUSH_DISTANCE,
    gripper_pad_friction: tuple[float, float, float] = ARMLESS_ROBOTIQ_PAD_FRICTION,
    robotiq_pad1_friction: tuple[float, float, float] = FRANKA_ROBOTIQ_PAD1_FRICTION,
    robotiq_pad2_friction: tuple[float, float, float] = FRANKA_ROBOTIQ_PAD2_FRICTION,
    robotiq_pad_condim: int = FRANKA_ROBOTIQ_PAD_CONDIM,
    robotiq_pad_priority: int = FRANKA_ROBOTIQ_PAD_PRIORITY,
    robotiq_pad_solref: tuple[float, float] = FRANKA_ROBOTIQ_PAD_SOLREF,
    robotiq_pad_solimp: tuple[float, float, float, float, float] = FRANKA_ROBOTIQ_PAD_SOLIMP,
    allegro_hand_friction: tuple[float, float, float] = ALLEGRO_HAND_FRICTION,
    allegro_hand_condim: int = ALLEGRO_HAND_CONDIM,
    allegro_hand_priority: int = ALLEGRO_HAND_PRIORITY,
    allegro_hand_solref: tuple[float, float] = ALLEGRO_HAND_SOLREF,
    allegro_hand_solimp: tuple[float, float, float, float, float] = ALLEGRO_HAND_SOLIMP,
    allegro_hand_collision_model: str = ALLEGRO_HAND_COLLISION_MODEL_FULL,
    allegro_fingertip_friction: tuple[float, float, float] = ALLEGRO_FINGERTIP_PAD_FRICTION,
    allegro_fingertip_radius: float = ALLEGRO_FINGERTIP_PAD_RADIUS,
    allegro_fingertip_condim: int = ALLEGRO_FINGERTIP_PAD_CONDIM,
    allegro_fingertip_priority: int = ALLEGRO_FINGERTIP_PAD_PRIORITY,
    allegro_fingertip_solref: tuple[float, float] = ALLEGRO_FINGERTIP_PAD_SOLREF,
    allegro_fingertip_solimp: tuple[float, float, float, float, float] = ALLEGRO_FINGERTIP_PAD_SOLIMP,
    barrett_fingertip_friction: tuple[float, float, float] = BARRETT_FINGERTIP_CAPSULE_FRICTION,
    barrett_fingertip_collision_model: str = BARRETT_FINGERTIP_COLLISION_MODEL_CAPSULE,
    barrett_fingertip_capsule_radius: float = BARRETT_FINGERTIP_CAPSULE_RADIUS,
    wujihand_contact_friction: tuple[float, float, float] = WUJIHAND_CONTACT_FRICTION,
    gripper_pad_patch_grid: tuple[int, int] = ARMLESS_ROBOTIQ_PAD_PATCH_GRID,
    enable_gripper_stick_collision: bool = False,
    enable_fingertip_collision_pads: bool = True,
    camera_width: int = CAMERA_WIDTH,
    camera_height: int = CAMERA_HEIGHT,
    table_half_extents: tuple[float, float, float] = TABLE_HALF_EXTENTS,
    render_shadows_enabled: bool = True,
    world_visual_style: str = "mujoco_default",
    world_floor_rgba: tuple[float, float, float, float] = (0.78, 0.80, 0.82, 1.0),
    world_wall_rgba: tuple[float, float, float, float] = (0.70, 0.70, 0.68, 1.0),
    world_skybox_rgba: tuple[float, float, float, float] = (0.74, 0.77, 0.79, 1.0),
    table_visual_style: str = "default",
    table_texture_path: str | None = None,
    table_texture_fallback: str = "none",
    table_texrepeat: tuple[float, float] = (1.0, 1.0),
    table_rgba: tuple[float, float, float, float] | None = None,
    table_specular: float = 0.06,
    table_shininess: float = 0.08,
    table_reflectance: float = 0.0,
    cube_texture_path: str | None = None,
    cube_face_texture_paths: dict[str, str | Path] | None = None,
    cube_face_texture_atlas_path: str | Path | None = None,
    cube_visual_face_offset: float = CUBE_TEXTURE_FACE_OFFSET,
    cube_texrepeat: tuple[float, float] = (1.0, 1.0),
    cube_rgba: tuple[float, float, float, float] | None = None,
    cube_specular: float = 0.16,
    cube_shininess: float = 0.18,
    cube_reflectance: float = 0.02,
    wuji_right_hand_rgba: tuple[float, float, float, float] | None = None,
    wuji_right_hand_specular: float = 0.18,
    wuji_right_hand_shininess: float = 0.20,
    wuji_right_hand_reflectance: float = 0.03,
) -> tuple[mujoco.MjSpec, SceneSpec]:
    """Load an embodiment XML and append the shared table, task object, lights, and cameras."""

    table_half_extents = _normalize_table_half_extents(table_half_extents)
    embodiment_key = str(embodiment).strip()
    if embodiment_key == "robotiq_gripper_stick_armless":
        robot_xml = require_path(ROBOTIQ_2F85_XML, label="Robotiq 2F-85 XML")
        spec = mujoco.MjSpec.from_file(str(robot_xml))
        _add_armless_robotiq_stick(
            spec,
            gripper_pad_friction=tuple(float(x) for x in gripper_pad_friction),
            gripper_pad_patch_grid=tuple(int(x) for x in gripper_pad_patch_grid),
            enable_gripper_stick_collision=bool(enable_gripper_stick_collision),
        )
        scene_embodiment = embodiment_key
    else:
        embodiment_spec = get_embodiment_spec(embodiment_key)
        robot_xml = require_path(embodiment_spec.robot_xml_path, label=f"{embodiment_key} robot XML")
        if embodiment_spec.key == "franka_robotiq":
            spec = mujoco.MjSpec.from_file(str(robot_xml))
            _configure_franka_robotiq_pad_contacts(
                spec,
                pad1_friction=tuple(float(x) for x in robotiq_pad1_friction),
                pad2_friction=tuple(float(x) for x in robotiq_pad2_friction),
                condim=int(robotiq_pad_condim),
                priority=int(robotiq_pad_priority),
                solref=tuple(float(x) for x in robotiq_pad_solref),
                solimp=tuple(float(x) for x in robotiq_pad_solimp),
            )
        elif embodiment_spec.key == "allegro_right":
            spec = _build_floating_allegro_spec(
                robot_xml,
                enable_fingertip_collision_pads=bool(enable_fingertip_collision_pads),
                hand_friction=tuple(float(x) for x in allegro_hand_friction),
                hand_condim=int(allegro_hand_condim),
                hand_priority=int(allegro_hand_priority),
                hand_solref=tuple(float(x) for x in allegro_hand_solref),
                hand_solimp=tuple(float(x) for x in allegro_hand_solimp),
                hand_collision_model=str(allegro_hand_collision_model),
                fingertip_friction=tuple(float(x) for x in allegro_fingertip_friction),
                fingertip_radius=float(allegro_fingertip_radius),
                fingertip_condim=int(allegro_fingertip_condim),
                fingertip_priority=int(allegro_fingertip_priority),
                fingertip_solref=tuple(float(x) for x in allegro_fingertip_solref),
                fingertip_solimp=tuple(float(x) for x in allegro_fingertip_solimp),
            )
        elif embodiment_spec.key == "barrett":
            spec = _build_floating_barrett_spec(
                robot_xml,
                enable_fingertip_collision_pads=bool(enable_fingertip_collision_pads),
                fingertip_friction=tuple(float(x) for x in barrett_fingertip_friction),
                fingertip_collision_model=str(barrett_fingertip_collision_model),
                fingertip_capsule_radius=float(barrett_fingertip_capsule_radius),
            )
        elif embodiment_spec.key == "wujihand_right":
            spec = _build_floating_wujihand_spec(
                robot_xml,
                contact_friction=tuple(float(x) for x in wujihand_contact_friction),
            )
        else:
            spec = mujoco.MjSpec.from_file(str(robot_xml))
        _place_robot_root(spec, embodiment_spec)
        _add_site_attachments(spec, embodiment_spec.site_attachments)
        if embodiment_spec.key in {"marvin_wuji", "wujihand_right"} and wuji_right_hand_rgba is not None:
            _apply_wuji_right_hand_visual_material(
                spec,
                rgba=tuple(float(x) for x in wuji_right_hand_rgba),
                specular=float(wuji_right_hand_specular),
                shininess=float(wuji_right_hand_shininess),
                reflectance=float(wuji_right_hand_reflectance),
            )
        scene_embodiment = embodiment_spec.key
    if float(timestep) <= 0.0:
        raise ValueError(f"timestep must be positive, got {timestep}")
    spec.option.timestep = float(timestep)
    spec.visual.global_.offwidth = max(int(camera_width), int(spec.visual.global_.offwidth))
    spec.visual.global_.offheight = max(int(camera_height), int(spec.visual.global_.offheight))
    spec.visual.quality.shadowsize = 4096 if bool(render_shadows_enabled) else 0
    spec.visual.map.shadowclip = 3.0

    _add_world_decor(
        spec,
        visual_style=str(world_visual_style),
        floor_rgba=tuple(float(x) for x in world_floor_rgba),
        wall_rgba=tuple(float(x) for x in world_wall_rgba),
        skybox_rgba=tuple(float(x) for x in world_skybox_rgba),
    )
    _add_table(
        spec,
        table_half_extents=table_half_extents,
        visual_style=str(table_visual_style),
        texture_path=table_texture_path,
        texture_fallback=str(table_texture_fallback),
        texrepeat=tuple(float(x) for x in table_texrepeat),
        rgba=(0.55, 0.40, 0.25, 1.0) if table_rgba is None else tuple(float(x) for x in table_rgba),
        specular=float(table_specular),
        shininess=float(table_shininess),
        reflectance=float(table_reflectance),
    )
    object_kind = str(object_kind).strip().lower()
    if object_kind == "cube":
        _add_cube(
            spec,
            cube_half_extent=float(cube_half_extent),
            cube_mass=float(cube_mass),
            cube_friction=tuple(float(x) for x in cube_friction),
            texture_path=cube_texture_path,
            face_texture_paths=cube_face_texture_paths,
            face_texture_atlas_path=cube_face_texture_atlas_path,
            visual_face_offset=float(cube_visual_face_offset),
            texrepeat=tuple(float(x) for x in cube_texrepeat),
            rgba=(0.92, 0.22, 0.12, 1.0) if cube_rgba is None else tuple(float(x) for x in cube_rgba),
            specular=float(cube_specular),
            shininess=float(cube_shininess),
            reflectance=float(cube_reflectance),
        )
    elif object_kind == "ycb_object":
        _add_ycb_object_and_bin(
            spec,
            visual_mesh_path=object_visual_mesh_path,
            collision_mesh_path=object_collision_mesh_path,
            collision_mesh_paths=object_collision_mesh_paths,
            texture_path=object_texture_path,
            mass=float(object_mass),
            object_friction=tuple(float(x) for x in object_friction),
            mesh_min_z=float(object_mesh_min_z),
            bin_friction=tuple(float(x) for x in bin_friction),
            bin_center_xy=tuple(float(x) for x in bin_center_xy),
            bin_inner_size=tuple(float(x) for x in bin_inner_size),
            bin_wall_height=float(bin_wall_height),
            bin_wall_thickness=float(bin_wall_thickness),
        )
    elif object_kind in {"pump_bottle", "pumpbottle_use"}:
        _add_pump_bottle(
            spec,
            mass=float(pump_bottle_mass),
            button_mass=float(pump_button_mass),
            button_joint_range=tuple(float(x) for x in pump_button_joint_range),
            button_joint_stiffness=float(pump_button_joint_stiffness),
            button_joint_damping=float(pump_button_joint_damping),
            button_actuator_kp=float(pump_button_actuator_kp),
            button_force_site_visible=bool(pump_button_force_site_visible),
            mesh_scale=float(pump_bottle_mesh_scale),
            friction=tuple(float(x) for x in pump_bottle_friction),
            body_collision_mode=str(pump_bottle_body_collision_mode),
            neck_collision_mode=str(pump_bottle_neck_collision_mode),
            visual_mesh_source=str(pump_bottle_visual_mesh_source),
            contact_margin=float(pump_bottle_contact_margin),
            asset_id=str(pump_bottle_asset_id),
            manifest_path=pump_bottle_manifest_path,
        )
        object_kind = "pump_bottle"
    elif object_kind in {"mini_keyboard", "keyboard", "keyboard_press"}:
        _add_mini_keyboard(
            spec,
            friction=tuple(float(x) for x in mini_keyboard_friction),
            support_box_height=float(mini_keyboard_support_box_height),
            white_key_push_distance=float(mini_keyboard_white_key_push_distance),
            black_key_push_distance=float(mini_keyboard_black_key_push_distance),
        )
        object_kind = "mini_keyboard"
    else:
        raise ValueError(f"Unsupported tabletop object_kind={object_kind!r}")
    if add_cameras:
        _add_cameras(spec, camera_width=int(camera_width), camera_height=int(camera_height))
    if not bool(render_shadows_enabled):
        _disable_light_shadows(spec)

    return spec, SceneSpec(
        robot_xml_path=robot_xml,
        embodiment=scene_embodiment,
        object_kind=object_kind,
        cube_visual_face_geom_names=(
            (CUBE_TEXTURE_ATLAS_GEOM_NAME,)
            if cube_face_texture_atlas_path is not None
            else CUBE_TEXTURE_FACE_GEOM_NAMES if cube_face_texture_paths is not None else ()
        ),
        cube_visual_face_mesh_names=(
            CUBE_TEXTURE_FACE_MESH_NAMES
            if cube_face_texture_paths is not None and cube_face_texture_atlas_path is None
            else ()
        ),
        cube_visual_box_mesh_name=(
            CUBE_TEXTURE_ATLAS_MESH_NAME if cube_face_texture_atlas_path is not None else None
        ),
        cube_visual_face_offset=float(cube_visual_face_offset),
        mini_keyboard_key_joint_names=MINI_KEYBOARD_KEY_JOINT_NAMES,
        pump_bottle_asset_id=str(pump_bottle_asset_id),
        table_half_extents=table_half_extents,
    )


def compile_tabletop_scene(
    *,
    add_cameras: bool = True,
    embodiment: str = "marvin_wuji",
    floating_rotation_reference_xyz: tuple[float, float, float] | None = None,
    object_kind: str = "cube",
    cube_half_extent: float = CUBE_HALF_EXTENT,
    cube_mass: float = 0.08,
    cube_friction: tuple[float, float, float] = (1.2, 0.05, 0.01),
    object_visual_mesh_path: str | Path | None = None,
    object_collision_mesh_path: str | Path | None = None,
    object_collision_mesh_paths: Sequence[str | Path] | None = None,
    object_texture_path: str | Path | None = None,
    object_mass: float = 0.1,
    object_friction: tuple[float, float, float] = (2.0, 0.05, 0.01),
    object_mesh_min_z: float = 0.0,
    bin_friction: tuple[float, float, float] = OBJECT_SUITE_BIN_FRICTION,
    bin_center_xy: tuple[float, float] = (0.24, 0.30),
    bin_inner_size: tuple[float, float] = (0.22, 0.22),
    bin_wall_height: float = 0.10,
    bin_wall_thickness: float = 0.008,
    timestep: float = 0.002,
    pump_bottle_mass: float = 0.18,
    pump_button_mass: float = 0.05,
    pump_button_joint_range: tuple[float, float] = (-0.013, 0.0),
    pump_button_joint_stiffness: float = 35.0,
    pump_button_joint_damping: float = 5.0,
    pump_button_actuator_kp: float = 35.0,
    pump_button_force_site_visible: bool = True,
    pump_bottle_mesh_scale: float = 1.3,
    pump_bottle_friction: tuple[float, float, float] = (2.0, 0.01, 0.0002),
    pump_bottle_body_collision_mode: str = PUMP_BOTTLE_COLLISION_MODE_CYLINDER,
    pump_bottle_neck_collision_mode: str = PUMP_BOTTLE_COLLISION_MODE_CYLINDER,
    pump_bottle_visual_mesh_source: str = PUMP_BOTTLE_VISUAL_MESH_SOURCE_VISUAL,
    pump_bottle_contact_margin: float = PUMP_BOTTLE_CONTACT_MARGIN,
    pump_bottle_asset_id: str = LEGACY_PUMP_BOTTLE_ID,
    pump_bottle_manifest_path: str | Path | None = None,
    mini_keyboard_friction: tuple[float, float, float] = (5.0, 0.1, 0.01),
    mini_keyboard_support_box_height: float = MINI_KEYBOARD_SUPPORT_BOX_HEIGHT,
    mini_keyboard_white_key_push_distance: float = MINI_KEYBOARD_WHITE_KEY_PUSH_DISTANCE,
    mini_keyboard_black_key_push_distance: float = MINI_KEYBOARD_BLACK_KEY_PUSH_DISTANCE,
    gripper_pad_friction: tuple[float, float, float] = ARMLESS_ROBOTIQ_PAD_FRICTION,
    robotiq_pad1_friction: tuple[float, float, float] = FRANKA_ROBOTIQ_PAD1_FRICTION,
    robotiq_pad2_friction: tuple[float, float, float] = FRANKA_ROBOTIQ_PAD2_FRICTION,
    robotiq_pad_condim: int = FRANKA_ROBOTIQ_PAD_CONDIM,
    robotiq_pad_priority: int = FRANKA_ROBOTIQ_PAD_PRIORITY,
    robotiq_pad_solref: tuple[float, float] = FRANKA_ROBOTIQ_PAD_SOLREF,
    robotiq_pad_solimp: tuple[float, float, float, float, float] = FRANKA_ROBOTIQ_PAD_SOLIMP,
    allegro_hand_friction: tuple[float, float, float] = ALLEGRO_HAND_FRICTION,
    allegro_hand_condim: int = ALLEGRO_HAND_CONDIM,
    allegro_hand_priority: int = ALLEGRO_HAND_PRIORITY,
    allegro_hand_solref: tuple[float, float] = ALLEGRO_HAND_SOLREF,
    allegro_hand_solimp: tuple[float, float, float, float, float] = ALLEGRO_HAND_SOLIMP,
    allegro_hand_collision_model: str = ALLEGRO_HAND_COLLISION_MODEL_FULL,
    allegro_fingertip_friction: tuple[float, float, float] = ALLEGRO_FINGERTIP_PAD_FRICTION,
    allegro_fingertip_radius: float = ALLEGRO_FINGERTIP_PAD_RADIUS,
    allegro_fingertip_condim: int = ALLEGRO_FINGERTIP_PAD_CONDIM,
    allegro_fingertip_priority: int = ALLEGRO_FINGERTIP_PAD_PRIORITY,
    allegro_fingertip_solref: tuple[float, float] = ALLEGRO_FINGERTIP_PAD_SOLREF,
    allegro_fingertip_solimp: tuple[float, float, float, float, float] = ALLEGRO_FINGERTIP_PAD_SOLIMP,
    barrett_fingertip_friction: tuple[float, float, float] = BARRETT_FINGERTIP_CAPSULE_FRICTION,
    barrett_fingertip_collision_model: str = BARRETT_FINGERTIP_COLLISION_MODEL_CAPSULE,
    barrett_fingertip_capsule_radius: float = BARRETT_FINGERTIP_CAPSULE_RADIUS,
    wujihand_contact_friction: tuple[float, float, float] = WUJIHAND_CONTACT_FRICTION,
    gripper_pad_patch_grid: tuple[int, int] = ARMLESS_ROBOTIQ_PAD_PATCH_GRID,
    enable_gripper_stick_collision: bool = False,
    enable_fingertip_collision_pads: bool = True,
    camera_width: int = CAMERA_WIDTH,
    camera_height: int = CAMERA_HEIGHT,
    table_half_extents: tuple[float, float, float] = TABLE_HALF_EXTENTS,
    render_shadows_enabled: bool = True,
    world_visual_style: str = "mujoco_default",
    world_floor_rgba: tuple[float, float, float, float] = (0.78, 0.80, 0.82, 1.0),
    world_wall_rgba: tuple[float, float, float, float] = (0.70, 0.70, 0.68, 1.0),
    world_skybox_rgba: tuple[float, float, float, float] = (0.74, 0.77, 0.79, 1.0),
    table_visual_style: str = "default",
    table_texture_path: str | None = None,
    table_texture_fallback: str = "none",
    table_texrepeat: tuple[float, float] = (1.0, 1.0),
    table_rgba: tuple[float, float, float, float] | None = None,
    table_specular: float = 0.06,
    table_shininess: float = 0.08,
    table_reflectance: float = 0.0,
    cube_texture_path: str | None = None,
    cube_face_texture_paths: dict[str, str | Path] | None = None,
    cube_face_texture_atlas_path: str | Path | None = None,
    cube_visual_face_offset: float = CUBE_TEXTURE_FACE_OFFSET,
    cube_texrepeat: tuple[float, float] = (1.0, 1.0),
    cube_rgba: tuple[float, float, float, float] | None = None,
    cube_specular: float = 0.16,
    cube_shininess: float = 0.18,
    cube_reflectance: float = 0.02,
    wuji_right_hand_rgba: tuple[float, float, float, float] | None = None,
    wuji_right_hand_specular: float = 0.18,
    wuji_right_hand_shininess: float = 0.20,
    wuji_right_hand_reflectance: float = 0.03,
) -> tuple[mujoco.MjSpec, mujoco.MjModel, mujoco.MjData, SceneSpec]:
    """Build and compile a fresh tabletop scene."""

    spec, scene_spec = build_tabletop_spec(
        add_cameras=add_cameras,
        embodiment=embodiment,
        object_kind=object_kind,
        cube_half_extent=float(cube_half_extent),
        cube_mass=float(cube_mass),
        cube_friction=tuple(float(x) for x in cube_friction),
        object_visual_mesh_path=object_visual_mesh_path,
        object_collision_mesh_path=object_collision_mesh_path,
        object_collision_mesh_paths=object_collision_mesh_paths,
        object_texture_path=object_texture_path,
        object_mass=float(object_mass),
        object_friction=tuple(float(x) for x in object_friction),
        object_mesh_min_z=float(object_mesh_min_z),
        bin_friction=tuple(float(x) for x in bin_friction),
        bin_center_xy=tuple(float(x) for x in bin_center_xy),
        bin_inner_size=tuple(float(x) for x in bin_inner_size),
        bin_wall_height=float(bin_wall_height),
        bin_wall_thickness=float(bin_wall_thickness),
        timestep=float(timestep),
        pump_bottle_mass=float(pump_bottle_mass),
        pump_button_mass=float(pump_button_mass),
        pump_button_joint_range=tuple(float(x) for x in pump_button_joint_range),
        pump_button_joint_stiffness=float(pump_button_joint_stiffness),
        pump_button_joint_damping=float(pump_button_joint_damping),
        pump_button_actuator_kp=float(pump_button_actuator_kp),
        pump_button_force_site_visible=bool(pump_button_force_site_visible),
        pump_bottle_mesh_scale=float(pump_bottle_mesh_scale),
        pump_bottle_friction=tuple(float(x) for x in pump_bottle_friction),
        pump_bottle_body_collision_mode=str(pump_bottle_body_collision_mode),
        pump_bottle_neck_collision_mode=str(pump_bottle_neck_collision_mode),
        pump_bottle_visual_mesh_source=str(pump_bottle_visual_mesh_source),
        pump_bottle_contact_margin=float(pump_bottle_contact_margin),
        pump_bottle_asset_id=str(pump_bottle_asset_id),
        pump_bottle_manifest_path=pump_bottle_manifest_path,
        mini_keyboard_friction=tuple(float(x) for x in mini_keyboard_friction),
        mini_keyboard_support_box_height=float(mini_keyboard_support_box_height),
        mini_keyboard_white_key_push_distance=float(mini_keyboard_white_key_push_distance),
        mini_keyboard_black_key_push_distance=float(mini_keyboard_black_key_push_distance),
        gripper_pad_friction=tuple(float(x) for x in gripper_pad_friction),
        robotiq_pad1_friction=tuple(float(x) for x in robotiq_pad1_friction),
        robotiq_pad2_friction=tuple(float(x) for x in robotiq_pad2_friction),
        robotiq_pad_condim=int(robotiq_pad_condim),
        robotiq_pad_priority=int(robotiq_pad_priority),
        robotiq_pad_solref=tuple(float(x) for x in robotiq_pad_solref),
        robotiq_pad_solimp=tuple(float(x) for x in robotiq_pad_solimp),
        allegro_hand_friction=tuple(float(x) for x in allegro_hand_friction),
        allegro_hand_condim=int(allegro_hand_condim),
        allegro_hand_priority=int(allegro_hand_priority),
        allegro_hand_solref=tuple(float(x) for x in allegro_hand_solref),
        allegro_hand_solimp=tuple(float(x) for x in allegro_hand_solimp),
        allegro_hand_collision_model=str(allegro_hand_collision_model),
        allegro_fingertip_friction=tuple(float(x) for x in allegro_fingertip_friction),
        allegro_fingertip_radius=float(allegro_fingertip_radius),
        allegro_fingertip_condim=int(allegro_fingertip_condim),
        allegro_fingertip_priority=int(allegro_fingertip_priority),
        allegro_fingertip_solref=tuple(float(x) for x in allegro_fingertip_solref),
        allegro_fingertip_solimp=tuple(float(x) for x in allegro_fingertip_solimp),
        barrett_fingertip_friction=tuple(float(x) for x in barrett_fingertip_friction),
        barrett_fingertip_collision_model=str(barrett_fingertip_collision_model),
        barrett_fingertip_capsule_radius=float(barrett_fingertip_capsule_radius),
        wujihand_contact_friction=tuple(float(x) for x in wujihand_contact_friction),
        gripper_pad_patch_grid=tuple(int(x) for x in gripper_pad_patch_grid),
        enable_gripper_stick_collision=bool(enable_gripper_stick_collision),
        enable_fingertip_collision_pads=bool(enable_fingertip_collision_pads),
        camera_width=int(camera_width),
        camera_height=int(camera_height),
        table_half_extents=tuple(float(x) for x in table_half_extents),
        render_shadows_enabled=bool(render_shadows_enabled),
        world_visual_style=str(world_visual_style),
        world_floor_rgba=tuple(float(x) for x in world_floor_rgba),
        world_wall_rgba=tuple(float(x) for x in world_wall_rgba),
        world_skybox_rgba=tuple(float(x) for x in world_skybox_rgba),
        table_visual_style=str(table_visual_style),
        table_texture_path=table_texture_path,
        table_texture_fallback=str(table_texture_fallback),
        table_texrepeat=tuple(float(x) for x in table_texrepeat),
        table_rgba=None if table_rgba is None else tuple(float(x) for x in table_rgba),
        table_specular=float(table_specular),
        table_shininess=float(table_shininess),
        table_reflectance=float(table_reflectance),
        cube_texture_path=cube_texture_path,
        cube_face_texture_paths=cube_face_texture_paths,
        cube_face_texture_atlas_path=cube_face_texture_atlas_path,
        cube_visual_face_offset=float(cube_visual_face_offset),
        cube_texrepeat=tuple(float(x) for x in cube_texrepeat),
        cube_rgba=None if cube_rgba is None else tuple(float(x) for x in cube_rgba),
        cube_specular=float(cube_specular),
        cube_shininess=float(cube_shininess),
        cube_reflectance=float(cube_reflectance),
        wuji_right_hand_rgba=None if wuji_right_hand_rgba is None else tuple(float(x) for x in wuji_right_hand_rgba),
        wuji_right_hand_specular=float(wuji_right_hand_specular),
        wuji_right_hand_shininess=float(wuji_right_hand_shininess),
        wuji_right_hand_reflectance=float(wuji_right_hand_reflectance),
    )
    if floating_rotation_reference_xyz is not None:
        FixedBaseRotationChart(embodiment, floating_rotation_reference_xyz).apply_to_spec(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    return spec, model, data, scene_spec


def _place_robot_root(spec: mujoco.MjSpec, embodiment: EmbodimentSpec) -> None:
    for body in spec.bodies:
        if body.name == embodiment.root_body_name:
            body.pos = list(embodiment.root_body_pos)
            body.quat = list(embodiment.root_body_quat_wxyz)
            return
    raise KeyError(f"Embodiment {embodiment.key!r} root body is missing: {embodiment.root_body_name}")


def _add_site_attachments(spec: mujoco.MjSpec, site_attachments: tuple[SiteAttachmentSpec, ...]) -> None:
    for site in site_attachments:
        for body in spec.bodies:
            if body.name != site.body_name:
                continue
            body.add_site(
                name=site.site_name,
                pos=list(site.pos),
                quat=list(site.quat_wxyz),
                size=list(site.size),
                rgba=list(site.rgba),
            )
            break
        else:
            raise KeyError(f"Cannot attach site {site.site_name!r}; body is missing: {site.body_name}")


def _build_floating_barrett_spec(
    robot_xml: Path,
    *,
    enable_fingertip_collision_pads: bool = True,
    fingertip_friction: tuple[float, float, float] = BARRETT_FINGERTIP_CAPSULE_FRICTION,
    fingertip_collision_model: str = BARRETT_FINGERTIP_COLLISION_MODEL_CAPSULE,
    fingertip_capsule_radius: float = BARRETT_FINGERTIP_CAPSULE_RADIUS,
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(robot_xml))
    fingertip_collision_model = _normalize_barrett_fingertip_collision_model(fingertip_collision_model)
    spec.compiler.balanceinertia = True
    spec.compiler.fusestatic = False
    _regularize_massless_virtual_bodies(spec)
    _configure_joint_dynamics(
        spec,
        BARRETT_ROOT_JOINT_NAMES,
        armature=FLOATING_ROOT_JOINT_ARMATURE,
        damping=FLOATING_ROOT_JOINT_DAMPING,
    )
    _configure_joint_dynamics(
        spec,
        BARRETT_HAND_JOINT_NAMES,
        armature=FLOATING_HAND_JOINT_ARMATURE,
        damping=FLOATING_HAND_JOINT_DAMPING,
    )
    base = _required_body(spec, "bh_base_link")
    existing_joints = {joint.name for joint in spec.joints}
    for name, joint_type, axis, q_range in (
        ("virtual_joint_x", mujoco.mjtJoint.mjJNT_SLIDE, (1.0, 0.0, 0.0), (-2.0, 2.0)),
        ("virtual_joint_y", mujoco.mjtJoint.mjJNT_SLIDE, (0.0, 1.0, 0.0), (-2.0, 2.0)),
        ("virtual_joint_z", mujoco.mjtJoint.mjJNT_SLIDE, (0.0, 0.0, 1.0), (-2.0, 2.0)),
        ("virtual_joint_roll", mujoco.mjtJoint.mjJNT_HINGE, (1.0, 0.0, 0.0), (-6.28, 6.28)),
        ("virtual_joint_pitch", mujoco.mjtJoint.mjJNT_HINGE, (0.0, 1.0, 0.0), (-6.28, 6.28)),
        ("virtual_joint_yaw", mujoco.mjtJoint.mjJNT_HINGE, (0.0, 0.0, 1.0), (-6.28, 6.28)),
    ):
        if name not in existing_joints:
            base.add_joint(
                name=name,
                type=joint_type,
                axis=list(axis),
                limited=True,
                range=list(q_range),
                damping=0.0,
                armature=0.001,
            )

    _add_barrett_sites(
        spec,
        enable_fingertip_collision_pads=bool(enable_fingertip_collision_pads),
        fingertip_friction=tuple(float(x) for x in fingertip_friction),
    )
    if fingertip_collision_model == BARRETT_FINGERTIP_COLLISION_MODEL_CAPSULE:
        _replace_barrett_fingertip_mesh_collision_with_capsules(
            spec,
            fingertip_friction=tuple(float(x) for x in fingertip_friction),
            fingertip_capsule_radius=float(fingertip_capsule_radius),
        )
    else:
        _enable_barrett_fingertip_mesh_collision(
            spec,
            fingertip_friction=tuple(float(x) for x in fingertip_friction),
        )
    _tune_barrett_hand_joint_dynamics(spec)
    existing_actuators = {actuator.name for actuator in spec.actuators}
    for joint_name in BARRETT_ROOT_JOINT_NAMES:
        if joint_name in existing_actuators:
            continue
        root_idx = BARRETT_ROOT_JOINT_NAMES.index(joint_name)
        _add_position_actuator(
            spec,
            name=joint_name,
            joint_name=joint_name,
            kp=float(BARRETT_ROOT_KP[root_idx]),
            kv=float(BARRETT_ROOT_KV[root_idx]),
            ctrlrange=_joint_ctrlrange_from_spec(spec, joint_name),
            force_limit=float(BARRETT_ROOT_FORCE_LIMITS[root_idx]),
        )
    for joint_name in BARRETT_HAND_JOINT_NAMES:
        if joint_name in existing_actuators:
            continue
        _add_position_actuator(
            spec,
            name=joint_name,
            joint_name=joint_name,
            kp=BARRETT_HAND_KP,
            kv=BARRETT_HAND_KV,
            ctrlrange=_joint_ctrlrange_from_spec(spec, joint_name),
            force_limit=BARRETT_HAND_FORCE_LIMIT,
        )
    return spec


def _build_floating_allegro_spec(
    robot_xml: Path,
    *,
    enable_fingertip_collision_pads: bool = True,
    hand_friction: tuple[float, float, float] = ALLEGRO_HAND_FRICTION,
    hand_condim: int = ALLEGRO_HAND_CONDIM,
    hand_priority: int = ALLEGRO_HAND_PRIORITY,
    hand_solref: tuple[float, float] = ALLEGRO_HAND_SOLREF,
    hand_solimp: tuple[float, float, float, float, float] = ALLEGRO_HAND_SOLIMP,
    hand_collision_model: str = ALLEGRO_HAND_COLLISION_MODEL_FULL,
    fingertip_friction: tuple[float, float, float] = ALLEGRO_FINGERTIP_PAD_FRICTION,
    fingertip_radius: float = ALLEGRO_FINGERTIP_PAD_RADIUS,
    fingertip_condim: int = ALLEGRO_FINGERTIP_PAD_CONDIM,
    fingertip_priority: int = ALLEGRO_FINGERTIP_PAD_PRIORITY,
    fingertip_solref: tuple[float, float] = ALLEGRO_FINGERTIP_PAD_SOLREF,
    fingertip_solimp: tuple[float, float, float, float, float] = ALLEGRO_FINGERTIP_PAD_SOLIMP,
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(robot_xml))
    _resolve_allegro_mesh_paths(spec, robot_xml=robot_xml)
    _configure_allegro_hand_contact_geoms(
        spec,
        friction=tuple(float(x) for x in hand_friction),
        condim=int(hand_condim),
        priority=int(hand_priority),
        solref=tuple(float(x) for x in hand_solref),
        solimp=tuple(float(x) for x in hand_solimp),
        collision_model=str(hand_collision_model),
    )
    spec.compiler.balanceinertia = True
    spec.compiler.fusestatic = False
    _set_all_body_gravcomp(spec, gravcomp=1.0)
    _regularize_massless_virtual_bodies(spec)
    _configure_floating_root_joint_ranges(spec, ALLEGRO_ROOT_JOINT_NAMES, ALLEGRO_ROOT_CTRL_RANGES)
    _configure_joint_dynamics(
        spec,
        ALLEGRO_ROOT_JOINT_NAMES,
        armature=FLOATING_ROOT_JOINT_ARMATURE,
        damping=FLOATING_ROOT_JOINT_DAMPING,
    )
    _configure_joint_dynamics(
        spec,
        ALLEGRO_HAND_JOINT_NAMES,
        armature=FLOATING_HAND_JOINT_ARMATURE,
        damping=FLOATING_HAND_JOINT_DAMPING,
    )
    _add_allegro_sites(
        spec,
        enable_fingertip_collision_pads=bool(enable_fingertip_collision_pads),
        fingertip_friction=tuple(float(x) for x in fingertip_friction),
        fingertip_radius=float(fingertip_radius),
        fingertip_condim=int(fingertip_condim),
        fingertip_priority=int(fingertip_priority),
        fingertip_solref=tuple(float(x) for x in fingertip_solref),
        fingertip_solimp=tuple(float(x) for x in fingertip_solimp),
    )

    existing_actuators = {actuator.name for actuator in spec.actuators}
    for joint_name in ALLEGRO_ROOT_JOINT_NAMES:
        if joint_name in existing_actuators:
            continue
        root_idx = ALLEGRO_ROOT_JOINT_NAMES.index(joint_name)
        _add_position_actuator(
            spec,
            name=joint_name,
            joint_name=joint_name,
            kp=float(ALLEGRO_ROOT_KP[root_idx]),
            kv=float(ALLEGRO_ROOT_KV[root_idx]),
            ctrlrange=_joint_ctrlrange_from_spec(spec, joint_name),
            force_limit=float(ALLEGRO_ROOT_FORCE_LIMITS[root_idx]),
        )
    for joint_name in ALLEGRO_HAND_JOINT_NAMES:
        if joint_name in existing_actuators:
            continue
        _add_position_actuator(
            spec,
            name=joint_name,
            joint_name=joint_name,
            kp=ALLEGRO_HAND_KP,
            kv=ALLEGRO_HAND_KV,
            ctrlrange=_joint_ctrlrange_from_spec(spec, joint_name),
            force_limit=ALLEGRO_HAND_FORCE_LIMIT,
        )
    return spec


def _build_floating_wujihand_spec(
    robot_xml: Path,
    *,
    contact_friction: tuple[float, float, float] = WUJIHAND_CONTACT_FRICTION,
) -> mujoco.MjSpec:
    """Load the standalone right WujiHand and give it floating-hand physics."""

    spec = mujoco.MjSpec.from_file(str(robot_xml))
    for camera in list(spec.cameras):
        spec.delete(camera)
    for light in list(spec.lights):
        spec.delete(light)

    spec.compiler.balanceinertia = True
    spec.compiler.fusestatic = False
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    # The standalone asset disables contact and assigns armature=1 in its root
    # defaults. Those values are appropriate for its pre-existing hand geoms
    # and joints, but must not leak into the table, object, bin, or free joint
    # that the shared tabletop builder adds after this function returns.
    spec.default.geom.contype = 1
    spec.default.geom.conaffinity = 1
    spec.default.geom.condim = 3
    spec.default.joint.armature = 0.0
    spec.default.joint.damping = [0.0, 0.0, 0.0]
    spec.default.joint.frictionloss = 0.0
    _set_all_body_gravcomp(spec, gravcomp=1.0)
    _configure_floating_root_joint_ranges(
        spec,
        WUJIHAND_ROOT_JOINT_NAMES,
        ALLEGRO_ROOT_CTRL_RANGES,
    )
    _configure_joint_dynamics(
        spec,
        WUJIHAND_ROOT_JOINT_NAMES,
        armature=FLOATING_ROOT_JOINT_ARMATURE,
        damping=FLOATING_ROOT_JOINT_DAMPING,
    )
    _configure_joint_dynamics(
        spec,
        WUJIHAND_HAND_JOINT_NAMES,
        armature=WUJIHAND_HAND_JOINT_ARMATURE,
        damping=WUJIHAND_HAND_JOINT_DAMPING,
    )
    for joint_name in WUJIHAND_HAND_JOINT_NAMES:
        joint = _find_joint(spec, joint_name)
        if joint is not None:
            joint.frictionloss = WUJIHAND_HAND_JOINT_FRICTIONLOSS
    _configure_wujihand_contact_geoms(spec, friction=contact_friction)
    return spec


def _configure_wujihand_contact_geoms(
    spec: mujoco.MjSpec,
    *,
    friction: tuple[float, float, float] = WUJIHAND_CONTACT_FRICTION,
) -> None:
    """Match the collision masks used by the arm-mounted right WujiHand."""

    found_palm = False
    finger_collision_count = 0
    for geom in spec.geoms:
        name = str(geom.name or "")
        if not name.startswith("collision_hand_right_"):
            continue
        geom.friction = [float(x) for x in friction]
        geom.condim = WUJIHAND_CONTACT_CONDIM
        if name == "collision_hand_right_palm_0":
            found_palm = True
            geom.contype = 1
            geom.conaffinity = 1
            continue
        finger_collision_count += 1
        suffix = name.rsplit("_", 1)[-1]
        if suffix == "4":
            geom.contype = 0
            geom.conaffinity = 0
        elif suffix == "3":
            geom.contype = 2
            geom.conaffinity = 2
        elif suffix in {"0", "1", "2"}:
            geom.contype = 1
            geom.conaffinity = 1
        else:
            raise ValueError(f"Unexpected standalone WujiHand collision geom: {name}")
    if not found_palm or finger_collision_count != 25:
        raise ValueError(
            "Standalone WujiHand must contain one palm and 25 finger collision geoms; "
            f"found palm={found_palm} fingers={finger_collision_count}"
        )


def _resolve_allegro_mesh_paths(spec: mujoco.MjSpec, *, robot_xml: Path) -> None:
    if robot_xml.suffix.lower() != ".urdf":
        return
    spec.compiler.meshdir = str(Path(robot_xml).parent / "meshes" / "collision")
    for mesh in spec.meshes:
        if Path(str(mesh.file)).name == "link_tip.obj":
            mesh.file = "link_tip.obj"


def _configure_franka_robotiq_pad_contacts(
    spec: mujoco.MjSpec,
    *,
    pad1_friction: tuple[float, float, float],
    pad2_friction: tuple[float, float, float],
    condim: int,
    priority: int,
    solref: tuple[float, float],
    solimp: tuple[float, float, float, float, float],
) -> None:
    pad_friction = {
        "g_left_pad1": pad1_friction,
        "g_right_pad1": pad1_friction,
        "g_left_pad2": pad2_friction,
        "g_right_pad2": pad2_friction,
    }
    geoms = {str(geom.name): geom for geom in spec.geoms if str(geom.name) in pad_friction}
    if set(geoms) != set(pad_friction):
        missing = sorted(set(pad_friction) - set(geoms))
        raise KeyError(f"FR3/Robotiq MJCF is missing contact pad geoms: {missing}")
    for name, friction in pad_friction.items():
        geom = geoms[name]
        geom.friction = list(friction)
        geom.condim = int(condim)
        geom.priority = int(priority)
        geom.solref = list(solref)
        geom.solimp = list(solimp)


def _configure_allegro_hand_contact_geoms(
    spec: mujoco.MjSpec,
    *,
    friction: tuple[float, float, float],
    condim: int,
    priority: int,
    solref: tuple[float, float],
    solimp: tuple[float, float, float, float, float],
    collision_model: str = ALLEGRO_HAND_COLLISION_MODEL_FULL,
) -> None:
    collision_model = _normalize_allegro_hand_collision_model(collision_model)
    for geom in spec.geoms:
        if "collision_pad" in str(geom.name):
            continue
        geom.friction = list(friction)
        geom.condim = int(condim)
        geom.priority = int(priority)
        geom.solref = list(solref)
        geom.solimp = list(solimp)
        if collision_model == ALLEGRO_HAND_COLLISION_MODEL_FINGERTIP_MESH_ONLY:
            is_fingertip_mesh = str(geom.parent.name or "").endswith("_tip")
            geom.contype = 1 if is_fingertip_mesh else 0
            geom.conaffinity = 1 if is_fingertip_mesh else 0


def _normalize_allegro_hand_collision_model(value: str | None) -> str:
    text = "full" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "full": ALLEGRO_HAND_COLLISION_MODEL_FULL,
        "full_hand": ALLEGRO_HAND_COLLISION_MODEL_FULL,
        "fingertip_mesh_only": ALLEGRO_HAND_COLLISION_MODEL_FINGERTIP_MESH_ONLY,
        "tip_mesh_only": ALLEGRO_HAND_COLLISION_MODEL_FINGERTIP_MESH_ONLY,
        "fingertip_only": ALLEGRO_HAND_COLLISION_MODEL_FINGERTIP_MESH_ONLY,
    }
    try:
        return aliases[text]
    except KeyError as exc:
        raise ValueError(
            "allegro_hand_collision_model must be 'full' or 'fingertip_mesh_only', "
            f"got {value!r}"
        ) from exc


def _set_all_body_gravcomp(spec: mujoco.MjSpec, *, gravcomp: float) -> None:
    for body in spec.bodies:
        body.gravcomp = float(gravcomp)


def _regularize_massless_virtual_bodies(spec: mujoco.MjSpec) -> None:
    for body in spec.bodies:
        if not str(body.name).startswith("virtual_"):
            continue
        if float(body.mass) <= 0.0:
            body.mass = FLOATING_ASSET_VIRTUAL_BODY_MASS
            body.inertia = list(FLOATING_ASSET_VIRTUAL_BODY_INERTIA)
            body.explicitinertial = True


def _configure_joint_dynamics(
    spec: mujoco.MjSpec,
    joint_names: tuple[str, ...],
    *,
    armature: float,
    damping: float,
) -> None:
    for joint_name in joint_names:
        joint = _find_joint(spec, joint_name)
        if joint is None:
            continue
        joint.armature = float(armature)
        try:
            joint.damping = float(damping)
        except TypeError:
            joint.damping = [float(damping), float(damping), float(damping)]


def _configure_floating_root_joint_ranges(
    spec: mujoco.MjSpec,
    joint_names: tuple[str, ...],
    q_ranges: tuple[tuple[float, float], ...],
) -> None:
    if len(joint_names) != len(q_ranges):
        raise ValueError(f"Expected six floating root joints, got {joint_names}")
    for joint_name, q_range in zip(joint_names, q_ranges):
        joint = _find_joint(spec, joint_name)
        if joint is None:
            continue
        joint.limited = True
        joint.range = list(q_range)


def _joint_ctrlrange_from_spec(spec: mujoco.MjSpec, joint_name: str) -> tuple[float, float]:
    joint = _find_joint(spec, joint_name)
    if joint is None:
        raise KeyError(f"Cannot add actuator; joint is missing: {joint_name}")
    q_range = np.asarray(joint.range, dtype=np.float64).reshape(-1)
    if q_range.size >= 2 and np.isfinite(q_range[:2]).all() and float(q_range[0]) < float(q_range[1]):
        return (float(q_range[0]), float(q_range[1]))
    if joint_name in ALLEGRO_ROOT_JOINT_NAMES[:3]:
        return (-2.0, 2.0)
    if joint_name in ALLEGRO_ROOT_JOINT_NAMES[3:]:
        return (-math.pi, math.pi)
    return (-10.0, 10.0)


def _tune_barrett_hand_joint_dynamics(spec: mujoco.MjSpec) -> None:
    for joint_name in BARRETT_HAND_JOINT_NAMES:
        joint = _find_joint(spec, joint_name)
        if joint is None:
            continue
        damping = np.asarray(joint.damping, dtype=np.float64).reshape(-1).copy()
        if damping.size:
            damping[0] = min(float(damping[0]), 10.0)
            joint.damping = damping


def _add_allegro_sites(
    spec: mujoco.MjSpec,
    *,
    enable_fingertip_collision_pads: bool = True,
    fingertip_friction: tuple[float, float, float] = ALLEGRO_FINGERTIP_PAD_FRICTION,
    fingertip_radius: float = ALLEGRO_FINGERTIP_PAD_RADIUS,
    fingertip_condim: int = ALLEGRO_FINGERTIP_PAD_CONDIM,
    fingertip_priority: int = ALLEGRO_FINGERTIP_PAD_PRIORITY,
    fingertip_solref: tuple[float, float] = ALLEGRO_FINGERTIP_PAD_SOLREF,
    fingertip_solimp: tuple[float, float, float, float, float] = ALLEGRO_FINGERTIP_PAD_SOLIMP,
) -> None:
    base = _required_body(spec, "base_link")
    _add_site_if_missing(
        base,
        name="right_palm",
        site_type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0, 0.0, 0.0),
        size=(0.01, 0.02, 0.03),
        rgba=(1.0, 0.0, 0.0, 0.0),
        update_existing_rgba=True,
    )
    tip_bodies = {
        "index": "link_3.0_tip",
        "middle": "link_7.0_tip",
        "ring": "link_11.0_tip",
        "thumb": "link_15.0_tip",
    }
    for finger_name, body_name in tip_bodies.items():
        body = _required_body(spec, body_name)
        for site_name in (
            f"right_{finger_name}_tip",
            f"track_hand_right_{finger_name}_tip",
            f"trace_hand_right_{finger_name}_tip",
        ):
            _add_site_if_missing(
                body,
                name=site_name,
                site_type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=(0.0, 0.0, 0.0),
                size=(0.006,),
                rgba=(0.0, 1.0, 0.0, 0.0),
                update_existing_rgba=True,
            )
        if enable_fingertip_collision_pads:
            _add_allegro_fingertip_pad_if_missing(
                body,
                name=f"right_{finger_name}_tip_collision_pad",
                pos=(0.0, 0.0, 0.0),
                friction=tuple(float(x) for x in fingertip_friction),
                radius=float(fingertip_radius),
                condim=int(fingertip_condim),
                priority=int(fingertip_priority),
                solref=tuple(float(x) for x in fingertip_solref),
                solimp=tuple(float(x) for x in fingertip_solimp),
            )


def _add_barrett_sites(
    spec: mujoco.MjSpec,
    *,
    enable_fingertip_collision_pads: bool = True,
    fingertip_friction: tuple[float, float, float] = BARRETT_FINGERTIP_PAD_FRICTION,
) -> None:
    base = _required_body(spec, "bh_base_link")
    _add_site_if_missing(
        base,
        name="right_wrist",
        site_type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=(0.0, 0.0, 0.0),
        size=(0.0001,),
        rgba=(1.0, 1.0, 1.0, 0.0),
    )
    _add_site_if_missing(
        base,
        name="right_palm",
        site_type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0, 0.0, 0.06),
        size=(0.01, 0.02, 0.03),
        rgba=(1.0, 0.0, 0.0, 0.0),
        update_existing_rgba=True,
    )
    tip_bodies = {
        "thumb": "bh_finger_33_link",
        "index": "bh_finger_13_link",
        "middle": "bh_finger_23_link",
        "ring": "bh_finger_13_link",
        "pinky": "bh_finger_23_link",
    }
    for finger_name, body_name in tip_bodies.items():
        body = _required_body(spec, body_name)
        pad_pos = (-0.045, 0.035, 0.0006)
        for site_name in (
            f"right_{finger_name}_tip",
            f"track_hand_right_{finger_name}_tip",
            f"trace_hand_right_{finger_name}_tip",
        ):
            _add_site_if_missing(
                body,
                name=site_name,
                site_type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=pad_pos,
                size=(0.01,),
                rgba=(0.0, 1.0, 0.0, 0.0),
                update_existing_rgba=True,
            )
        if enable_fingertip_collision_pads and finger_name in {"thumb", "index", "middle"}:
            _add_barrett_fingertip_pad_if_missing(
                body,
                name=f"right_{finger_name}_tip_collision_pad",
                pos=pad_pos,
                friction=tuple(float(x) for x in fingertip_friction),
            )


def _add_allegro_fingertip_pad_if_missing(
    body,
    *,
    name: str,
    pos: tuple[float, float, float],
    friction: tuple[float, float, float],
    radius: float,
    condim: int,
    priority: int,
    solref: tuple[float, float],
    solimp: tuple[float, float, float, float, float],
) -> None:
    if float(radius) <= 0.0:
        raise ValueError(f"Allegro fingertip collision pad radius must be positive, got {radius}")
    geom = next((geom for geom in body.geoms if geom.name == name), None)
    if geom is None:
        geom = body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=list(pos),
            size=[float(radius), 0.0, 0.0],
            rgba=[0.10, 0.80, 1.00, 0.0],
            group=3,
            mass=1e-8,
        )
    geom.size = [float(radius), 0.0, 0.0]
    geom.friction = list(friction)
    geom.condim = int(condim)
    geom.contype = 3
    geom.conaffinity = 3
    geom.priority = int(priority)
    geom.solref = list(solref)
    geom.solimp = list(solimp)


def _add_barrett_fingertip_pad_if_missing(
    body,
    *,
    name: str,
    pos: tuple[float, float, float],
    friction: tuple[float, float, float] = BARRETT_FINGERTIP_PAD_FRICTION,
) -> None:
    geom = next((geom for geom in body.geoms if geom.name == name), None)
    if geom is None:
        geom = body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=list(pos),
            size=[BARRETT_FINGERTIP_PAD_RADIUS],
            rgba=[0.10, 0.80, 1.00, 0.0],
            group=3,
            mass=1e-8,
            condim=6,
        )
    geom.friction = list(friction)
    geom.contype = 3
    geom.conaffinity = 3
    geom.priority = 2
    geom.solref = [0.002, 1.0]
    geom.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]


def _replace_barrett_fingertip_mesh_collision_with_capsules(
    spec: mujoco.MjSpec,
    *,
    fingertip_friction: tuple[float, float, float] = BARRETT_FINGERTIP_CAPSULE_FRICTION,
    fingertip_capsule_radius: float = BARRETT_FINGERTIP_CAPSULE_RADIUS,
) -> None:
    for finger_name, body_name in (
        ("thumb", "bh_finger_33_link"),
        ("index", "bh_finger_13_link"),
        ("middle", "bh_finger_23_link"),
    ):
        body = _required_body(spec, body_name)
        for geom in body.geoms:
            if int(geom.type) == int(mujoco.mjtGeom.mjGEOM_MESH):
                geom.contype = 0
                geom.conaffinity = 0
        _add_barrett_fingertip_capsule_if_missing(
            body,
            name=f"right_{finger_name}_tip_collision_capsule",
            friction=tuple(float(x) for x in fingertip_friction),
            radius=float(fingertip_capsule_radius),
        )


def _enable_barrett_fingertip_mesh_collision(
    spec: mujoco.MjSpec,
    *,
    fingertip_friction: tuple[float, float, float] = BARRETT_FINGERTIP_CAPSULE_FRICTION,
) -> None:
    for _, body_name in (
        ("thumb", "bh_finger_33_link"),
        ("index", "bh_finger_13_link"),
        ("middle", "bh_finger_23_link"),
    ):
        body = _required_body(spec, body_name)
        found_mesh = False
        for geom in body.geoms:
            name = str(geom.name or "")
            if int(geom.type) == int(mujoco.mjtGeom.mjGEOM_MESH):
                found_mesh = True
                geom.contype = 3
                geom.conaffinity = 3
                geom.friction = list(fingertip_friction)
                geom.condim = 6
                geom.priority = 2
                geom.solref = [0.002, 1.0]
                geom.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
            if name.endswith("_tip_collision_capsule"):
                geom.contype = 0
                geom.conaffinity = 0
        if not found_mesh:
            raise KeyError(f"Barrett fingertip body {body_name!r} has no mesh geom to use for collision")


def _normalize_barrett_fingertip_collision_model(value: str | None) -> str:
    text = "capsule" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "capsule": BARRETT_FINGERTIP_COLLISION_MODEL_CAPSULE,
        "capsules": BARRETT_FINGERTIP_COLLISION_MODEL_CAPSULE,
        "cylinder": BARRETT_FINGERTIP_COLLISION_MODEL_CAPSULE,
        "cylinders": BARRETT_FINGERTIP_COLLISION_MODEL_CAPSULE,
        "proxy": BARRETT_FINGERTIP_COLLISION_MODEL_CAPSULE,
        "primitive": BARRETT_FINGERTIP_COLLISION_MODEL_CAPSULE,
        "mesh": BARRETT_FINGERTIP_COLLISION_MODEL_MESH,
        "original": BARRETT_FINGERTIP_COLLISION_MODEL_MESH,
        "original_mesh": BARRETT_FINGERTIP_COLLISION_MODEL_MESH,
        "native": BARRETT_FINGERTIP_COLLISION_MODEL_MESH,
    }
    try:
        return aliases[text]
    except KeyError as exc:
        raise ValueError(
            "barrett_fingertip_collision_model must be 'capsule' or 'mesh', "
            f"got {value!r}"
        ) from exc


def _add_barrett_fingertip_capsule_if_missing(
    body,
    *,
    name: str,
    friction: tuple[float, float, float] = BARRETT_FINGERTIP_CAPSULE_FRICTION,
    radius: float = BARRETT_FINGERTIP_CAPSULE_RADIUS,
) -> None:
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"Barrett fingertip capsule radius must be positive and finite, got {radius}")
    geom = next((geom for geom in body.geoms if geom.name == name), None)
    if geom is None:
        geom = body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[radius],
            rgba=[0.10, 0.80, 1.00, 0.0],
            group=3,
            mass=1e-8,
            condim=6,
        )
    geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    geom.size = [radius, 0.0, 0.0]
    geom.friction = list(friction)
    geom.fromto = list(BARRETT_FINGERTIP_CAPSULE_FROMTO)
    geom.contype = 3
    geom.conaffinity = 3
    geom.priority = 2
    geom.solref = [0.002, 1.0]
    geom.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]


def _required_body(spec: mujoco.MjSpec, name: str):
    body = _find_first_body(spec, (name,))
    if body is None:
        raise KeyError(f"MuJoCo spec is missing body {name!r}")
    return body


def _find_joint(spec: mujoco.MjSpec, name: str):
    for joint in spec.joints:
        if joint.name == name:
            return joint
    return None


def _add_site_if_missing(
    body,
    *,
    name: str,
    site_type: mujoco.mjtGeom,
    pos: tuple[float, float, float],
    size: tuple[float, ...],
    rgba: tuple[float, float, float, float],
    update_existing_rgba: bool = False,
) -> None:
    for site in body.sites:
        if site.name == name:
            if update_existing_rgba:
                site.rgba = list(rgba)
            return
    body.add_site(
        name=name,
        type=site_type,
        pos=list(pos),
        size=list(size),
        rgba=list(rgba),
    )


def _add_armless_robotiq_stick(
    spec: mujoco.MjSpec,
    *,
    gripper_pad_friction: tuple[float, float, float] = ARMLESS_ROBOTIQ_PAD_FRICTION,
    gripper_pad_patch_grid: tuple[int, int] = ARMLESS_ROBOTIQ_PAD_PATCH_GRID,
    enable_gripper_stick_collision: bool = False,
) -> None:
    gripper_body = _find_first_body(spec, ("g_base",))
    if gripper_body is None:
        raise KeyError("Standalone Robotiq XML is missing root body 'g_base'")
    _add_armless_6d_root_joints(gripper_body, joint_names=ARMLESS_GRIPPER_ROOT_JOINT_NAMES)
    _configure_armless_robotiq_collision(
        spec,
        pad_friction=tuple(float(x) for x in gripper_pad_friction),
        pad_patch_grid=tuple(int(x) for x in gripper_pad_patch_grid),
    )
    _add_armless_robotiq_internal_excludes(spec)

    stick = spec.worldbody.add_body(name="robotiq_stick", pos=[0.0, 0.0, 0.0])
    _add_armless_6d_root_joints(stick, joint_names=ARMLESS_STICK_ROOT_JOINT_NAMES)
    geom = stick.add_geom(
        name="robotiq_stick_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        pos=[0.0, 0.0, 0.04],
        size=[0.008, 0.04],
        rgba=[0.55, 0.55, 0.55, 1.0],
        mass=0.03,
        friction=[0.7, 0.02, 0.001],
        condim=6,
    )
    geom.contype = 3
    geom.conaffinity = 3
    _add_armless_root_position_actuators(spec, joint_names=ARMLESS_GRIPPER_ROOT_JOINT_NAMES)
    _add_armless_root_position_actuators(spec, joint_names=ARMLESS_STICK_ROOT_JOINT_NAMES)
    _add_armless_robotiq_internal_actuators(spec)
    if not bool(enable_gripper_stick_collision):
        _add_armless_robotiq_stick_gripper_excludes(spec)


def _add_armless_6d_root_joints(
    body: mujoco.MjsBody,
    *,
    joint_names: tuple[str, ...],
) -> None:
    if len(joint_names) != 6:
        raise ValueError(f"Expected six root joint names, got {joint_names}")
    specs = (
        (mujoco.mjtJoint.mjJNT_SLIDE, (1.0, 0.0, 0.0), ARMLESS_ROOT_CTRL_RANGES[0]),
        (mujoco.mjtJoint.mjJNT_SLIDE, (0.0, 1.0, 0.0), ARMLESS_ROOT_CTRL_RANGES[1]),
        (mujoco.mjtJoint.mjJNT_SLIDE, (0.0, 0.0, 1.0), ARMLESS_ROOT_CTRL_RANGES[2]),
        (mujoco.mjtJoint.mjJNT_HINGE, (1.0, 0.0, 0.0), ARMLESS_ROOT_CTRL_RANGES[3]),
        (mujoco.mjtJoint.mjJNT_HINGE, (0.0, 1.0, 0.0), ARMLESS_ROOT_CTRL_RANGES[4]),
        (mujoco.mjtJoint.mjJNT_HINGE, (0.0, 0.0, 1.0), ARMLESS_ROOT_CTRL_RANGES[5]),
    )
    for name, (joint_type, axis, q_range) in zip(joint_names, specs):
        body.add_joint(
            name=str(name),
            type=joint_type,
            axis=list(axis),
            limited=True,
            range=list(q_range),
            damping=0.0,
            armature=1.0,
        )


def _add_position_actuator(
    spec: mujoco.MjSpec,
    *,
    name: str,
    joint_name: str,
    kp: float,
    kv: float,
    ctrlrange: tuple[float, float],
    force_limit: float,
) -> None:
    spec.add_actuator(
        name=str(name),
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        target=str(joint_name),
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        gainprm=[float(kp), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        biasprm=[0.0, -float(kp), -float(kv), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ctrllimited=True,
        ctrlrange=list(ctrlrange),
        forcelimited=True,
        forcerange=[-float(force_limit), float(force_limit)],
    )


def _add_armless_root_position_actuators(
    spec: mujoco.MjSpec,
    *,
    joint_names: tuple[str, ...],
) -> None:
    for joint_name, ctrlrange, kp, kv, force_limit in zip(
        joint_names,
        ARMLESS_ROOT_CTRL_RANGES,
        ARMLESS_ROOT_KP,
        ARMLESS_ROOT_KV,
        ARMLESS_ROOT_FORCE_LIMITS,
    ):
        _add_position_actuator(
            spec,
            name=f"{joint_name}_actuator",
            joint_name=joint_name,
            kp=float(kp),
            kv=float(kv),
            ctrlrange=ctrlrange,
            force_limit=float(force_limit),
        )


def _add_armless_robotiq_internal_actuators(spec: mujoco.MjSpec) -> None:
    for joint_name in ARMLESS_ROBOTIQ_INTERNAL_JOINT_NAMES:
        _add_position_actuator(
            spec,
            name=f"{joint_name}_actuator",
            joint_name=joint_name,
            kp=ARMLESS_ROBOTIQ_JOINT_KP,
            kv=ARMLESS_ROBOTIQ_JOINT_KV,
            ctrlrange=(0.0, ARMLESS_ROBOTIQ_CLOSED_JOINT_VALUE),
            force_limit=ARMLESS_ROBOTIQ_JOINT_FORCE_LIMIT,
        )


def _configure_armless_robotiq_collision(
    spec: mujoco.MjSpec,
    *,
    pad_friction: tuple[float, float, float] = ARMLESS_ROBOTIQ_PAD_FRICTION,
    pad_patch_grid: tuple[int, int] = ARMLESS_ROBOTIQ_PAD_PATCH_GRID,
) -> None:
    pad_friction = tuple(float(x) for x in pad_friction)
    if len(pad_friction) != 3 or any(value < 0.0 for value in pad_friction):
        raise ValueError(
            f"gripper_pad_friction must contain three non-negative values, got {pad_friction}"
        )
    pad_patch_grid = tuple(int(x) for x in pad_patch_grid)
    if len(pad_patch_grid) != 2 or any(value <= 0 for value in pad_patch_grid):
        raise ValueError(
            f"gripper_pad_patch_grid must contain two positive integers, got {pad_patch_grid}"
        )
    pad_names = {"g_left_pad1", "g_left_pad2", "g_right_pad1", "g_right_pad2"}
    pad_bodies: dict[str, mujoco.MjsBody] = {}
    for body in spec.bodies:
        body_name = str(body.name or "")
        if body_name in {"g_left_pad", "g_right_pad"}:
            pad_bodies[body_name] = body
        if not body_name.startswith("g_"):
            continue
        for geom in body.geoms:
            geom_name = str(geom.name or "")
            if int(geom.group) != 3 and geom_name not in pad_names:
                continue
            geom.contype = 3
            geom.conaffinity = 3
            geom.group = 3
            geom.condim = 6
            if geom_name in pad_names:
                geom.friction = list(pad_friction)
                geom.priority = 2
                if pad_patch_grid != (1, 1):
                    geom.contype = 0
                    geom.conaffinity = 0
            geom.solref = [0.002, 1.0]
            geom.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
    if pad_patch_grid != (1, 1):
        _add_armless_robotiq_pad_contact_patches(
            pad_bodies,
            pad_friction=pad_friction,
            patch_grid=pad_patch_grid,
        )


def _add_armless_robotiq_pad_contact_patches(
    pad_bodies: dict[str, mujoco.MjsBody],
    *,
    pad_friction: tuple[float, float, float],
    patch_grid: tuple[int, int],
) -> None:
    for body_name, geom_names in (
        ("g_left_pad", ("g_left_pad1", "g_left_pad2")),
        ("g_right_pad", ("g_right_pad1", "g_right_pad2")),
    ):
        body = pad_bodies.get(body_name)
        if body is None:
            raise KeyError(f"Standalone Robotiq XML is missing pad body {body_name!r}")
        original_geoms = {
            str(geom.name or ""): geom
            for geom in body.geoms
            if str(geom.name or "") in set(geom_names)
        }
        if set(original_geoms) != set(geom_names):
            missing = sorted(set(geom_names) - set(original_geoms))
            raise KeyError(f"Standalone Robotiq XML is missing pad geoms {missing}")
        _add_pad_patch_grid(
            body,
            base_name=body_name,
            source_geoms=tuple(original_geoms[name] for name in geom_names),
            pad_friction=pad_friction,
            patch_grid=patch_grid,
        )


def _add_pad_patch_grid(
    body: mujoco.MjsBody,
    *,
    base_name: str,
    source_geoms: tuple[mujoco.MjsGeom, ...],
    pad_friction: tuple[float, float, float],
    patch_grid: tuple[int, int],
) -> None:
    cols, rows = patch_grid
    x_center = float(np.mean([np.asarray(geom.pos, dtype=np.float64)[0] for geom in source_geoms]))
    x_half_extent = max(float(np.asarray(geom.size, dtype=np.float64)[0]) for geom in source_geoms)
    y_min = min(
        float(np.asarray(geom.pos, dtype=np.float64)[1] - np.asarray(geom.size, dtype=np.float64)[1])
        for geom in source_geoms
    )
    y_max = max(
        float(np.asarray(geom.pos, dtype=np.float64)[1] + np.asarray(geom.size, dtype=np.float64)[1])
        for geom in source_geoms
    )
    z_min = min(
        float(np.asarray(geom.pos, dtype=np.float64)[2] - np.asarray(geom.size, dtype=np.float64)[2])
        for geom in source_geoms
    )
    z_max = max(
        float(np.asarray(geom.pos, dtype=np.float64)[2] + np.asarray(geom.size, dtype=np.float64)[2])
        for geom in source_geoms
    )
    patch_y = (y_max - y_min) / float(cols)
    patch_z = (z_max - z_min) / float(rows)
    y_half = 0.5 * patch_y * ARMLESS_ROBOTIQ_PAD_PATCH_FILL
    z_half = 0.5 * patch_z * ARMLESS_ROBOTIQ_PAD_PATCH_FILL
    for col in range(cols):
        y_center = y_min + (float(col) + 0.5) * patch_y
        for row in range(rows):
            z_center = z_min + (float(row) + 0.5) * patch_z
            geom = body.add_geom(
                name=f"{base_name}_patch_{col}_{row}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x_center, y_center, z_center],
                size=[x_half_extent, y_half, z_half],
                rgba=[0.10, 0.80, 1.00, 0.0],
                group=3,
                mass=1e-8,
                friction=list(pad_friction),
                condim=6,
            )
            geom.contype = 3
            geom.conaffinity = 3
            geom.priority = 2
            geom.solref = [0.002, 1.0]
            geom.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]


def _add_armless_robotiq_internal_excludes(spec: mujoco.MjSpec) -> None:
    # The Robotiq asset's same-finger coupler/follower collision meshes overlap
    # in the closed-chain mechanism. Keep those geoms active for object contacts,
    # but suppress their internal contacts with each other.
    spec.add_exclude(
        name="robotiq_left_coupler_follower",
        bodyname1="g_left_coupler",
        bodyname2="g_left_follower",
    )
    spec.add_exclude(
        name="robotiq_right_coupler_follower",
        bodyname1="g_right_coupler",
        bodyname2="g_right_follower",
    )


def _add_armless_robotiq_stick_gripper_excludes(spec: mujoco.MjSpec) -> None:
    body_names = sorted(
        {str(body.name) for body in spec.bodies if str(body.name or "").startswith("g_")}
    )
    for body_name in body_names:
        spec.add_exclude(
            name=f"robotiq_stick_no_contact_{body_name}",
            bodyname1="robotiq_stick",
            bodyname2=body_name,
        )


def _add_world_decor(
    spec: mujoco.MjSpec,
    *,
    visual_style: str = "mujoco_default",
    floor_rgba: tuple[float, float, float, float] = (0.78, 0.80, 0.82, 1.0),
    wall_rgba: tuple[float, float, float, float] = (0.70, 0.70, 0.68, 1.0),
    skybox_rgba: tuple[float, float, float, float] = (0.74, 0.77, 0.79, 1.0),
) -> None:
    style = str(visual_style).strip().lower().replace("-", "_")
    if style in {"", "default", "mujoco", "mujoco_default"}:
        _add_mujoco_world_decor(spec)
        return
    if style in {"indoor", "indoor_room", "studio_indoor", "studio"}:
        _add_indoor_world_decor(
            spec,
            floor_rgba=_validate_rgba_tuple("world floor_rgba", floor_rgba),
            wall_rgba=_validate_rgba_tuple("world wall_rgba", wall_rgba),
            skybox_rgba=_validate_rgba_tuple("world skybox_rgba", skybox_rgba),
        )
        return
    raise ValueError(
        "world visual_style must be 'mujoco_default' or 'indoor', "
        f"got {visual_style!r}"
    )


def _add_mujoco_world_decor(spec: mujoco.MjSpec) -> None:
    world = spec.worldbody
    try:
        spec.add_texture(
            name="skybox",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.3, 0.5, 0.7],
            rgb2=[0.0, 0.0, 0.0],
            width=256,
            height=256,
        )
        spec.add_texture(
            name="groundplane",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            mark=mujoco.mjtMark.mjMARK_EDGE,
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.2, 0.3],
            markrgb=[0.8, 0.8, 0.8],
            width=300,
            height=300,
        )
        spec.add_material(
            name="groundplane",
            textures=_rgb_texture_slots("groundplane"),
            texrepeat=[5, 5],
            reflectance=0.2,
        )
    except Exception:
        pass

    ground = world.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5.0, 5.0, 0.1],
        rgba=[0.78, 0.80, 0.82, 1.0],
    )
    try:
        ground.material = "groundplane"
    except Exception:
        pass

    key_light = world.add_light(
        name="key_light",
        pos=[0.6, 0.4, 2.5],
        dir=[-0.3, -0.3, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    key_light.castshadow = 1
    world.add_light(name="fill_light", pos=[-0.2, -0.5, 1.7], dir=[0.1, 0.3, -1.0])


def _add_indoor_world_decor(
    spec: mujoco.MjSpec,
    *,
    floor_rgba: tuple[float, float, float, float],
    wall_rgba: tuple[float, float, float, float],
    skybox_rgba: tuple[float, float, float, float],
) -> None:
    world = spec.worldbody
    try:
        spec.add_texture(
            name="skybox",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
            rgb1=list(skybox_rgba[:3]),
            rgb2=list(skybox_rgba[:3]),
            width=256,
            height=256,
        )
    except Exception:
        pass
    try:
        spec.add_material(
            name="indoor_floor_material",
            rgba=list(floor_rgba),
            specular=0.0,
            shininess=0.0,
            reflectance=0.0,
        )
    except Exception:
        pass

    ground = world.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[50.0, 50.0, 0.1],
        rgba=list(floor_rgba),
    )
    try:
        ground.material = "indoor_floor_material"
    except Exception:
        pass

    key_light = world.add_light(
        name="key_light",
        pos=[0.6, 0.4, 2.5],
        dir=[-0.3, -0.3, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    key_light.castshadow = 1
    world.add_light(name="fill_light", pos=[-0.4, -0.8, 1.8], dir=[0.2, 0.4, -1.0])


def _disable_light_shadows(spec: mujoco.MjSpec) -> None:
    for light in spec.lights:
        light.castshadow = 0


def _validate_rgba_tuple(label: str, rgba: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    values = tuple(float(x) for x in rgba)
    if len(values) != 4 or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"{label} must contain four values in [0, 1], got {rgba}")
    return values


def _rgb_texture_slots(texture_name: str) -> list[str]:
    slots = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE.value)
    slots[int(mujoco.mjtTextureRole.mjTEXROLE_RGB.value)] = str(texture_name)
    return slots


def _add_table(
    spec: mujoco.MjSpec,
    *,
    table_half_extents: tuple[float, float, float] = TABLE_HALF_EXTENTS,
    visual_style: str = "default",
    texture_path: str | Path | None = None,
    texture_fallback: str = "none",
    texrepeat: tuple[float, float] = (1.0, 1.0),
    rgba: tuple[float, float, float, float] = (0.55, 0.40, 0.25, 1.0),
    specular: float = 0.06,
    shininess: float = 0.08,
    reflectance: float = 0.0,
) -> None:
    table_half_extents = _normalize_table_half_extents(table_half_extents)
    table_center = (float(TABLE_CENTER[0]), float(TABLE_CENTER[1]), float(TABLE_TOP_Z) - table_half_extents[2])
    body = spec.worldbody.add_body(name="table", pos=list(table_center))
    material_name = _add_table_visual_material(
        spec,
        visual_style=visual_style,
        texture_path=texture_path,
        texture_fallback=texture_fallback,
        texrepeat=texrepeat,
        rgba=rgba,
        specular=float(specular),
        shininess=float(shininess),
        reflectance=float(reflectance),
    )
    geom = body.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(table_half_extents),
        rgba=list(rgba),
    )
    if material_name is not None:
        geom.material = material_name
    geom.friction = [1.5, 0.05, 0.01]
    geom.condim = 6
    geom.solref = [0.002, 1.0]
    geom.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]


def _normalize_table_half_extents(
    table_half_extents: tuple[float, float, float],
) -> tuple[float, float, float]:
    values = tuple(float(x) for x in table_half_extents)
    if len(values) != 3:
        raise ValueError(f"table_half_extents must contain 3 values, got {values}")
    if any(value <= 0.0 for value in values):
        raise ValueError(f"table_half_extents must be positive, got {values}")
    return values


def _add_table_visual_material(
    spec: mujoco.MjSpec,
    *,
    visual_style: str,
    texture_path: str | Path | None,
    texture_fallback: str,
    texrepeat: tuple[float, float],
    rgba: tuple[float, float, float, float],
    specular: float,
    shininess: float,
    reflectance: float,
) -> str | None:
    style = str(visual_style).strip().lower().replace("-", "_")
    fallback = str(texture_fallback).strip().lower().replace("-", "_")
    if style not in {"", "none", "default", "wood"}:
        raise ValueError(f"table visual_style must be 'default' or 'wood', got {visual_style!r}")
    if style in {"", "none", "default"} and texture_path is None and fallback in {"", "none", "default"}:
        return None
    if len(tuple(texrepeat)) != 2 or any(float(value) <= 0.0 for value in texrepeat):
        raise ValueError(f"table_texrepeat must contain two positive values, got {texrepeat}")
    texture_name = "table_wood_texture"
    if texture_path is not None:
        path = Path(texture_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"table texture does not exist: {path}")
        spec.add_texture(
            name=texture_name,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            file=str(path),
        )
    elif style == "wood" or fallback in {"builtin_wood", "wood"}:
        spec.add_texture(
            name=texture_name,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.78, 0.52, 0.30],
            rgb2=[0.55, 0.32, 0.16],
            width=512,
            height=512,
        )
    elif fallback not in {"", "none", "default"}:
        raise ValueError(
            "table texture_fallback must be 'none' or 'builtin_wood', "
            f"got {texture_fallback!r}"
        )
    else:
        return None
    material_name = "table_wood_material"
    spec.add_material(
        name=material_name,
        textures=_rgb_texture_slots(texture_name),
        texuniform=True,
        texrepeat=[float(texrepeat[0]), float(texrepeat[1])],
        specular=float(specular),
        shininess=float(shininess),
        reflectance=float(reflectance),
        rgba=list(rgba),
    )
    return material_name


def _apply_wuji_right_hand_visual_material(
    spec: mujoco.MjSpec,
    *,
    rgba: tuple[float, float, float, float],
    specular: float,
    shininess: float,
    reflectance: float,
) -> None:
    if len(tuple(rgba)) != 4 or any(value < 0.0 or value > 1.0 for value in rgba):
        raise ValueError(f"wuji right hand rgba values must be in [0, 1], got {rgba}")
    material_name = "wuji_right_hand_black_material"
    spec.add_material(
        name=material_name,
        rgba=list(rgba),
        specular=float(specular),
        shininess=float(shininess),
        reflectance=float(reflectance),
    )
    for body in spec.bodies:
        body_name = str(body.name or "")
        if body_name not in {"right_wuji_palm", "right_palm"} and not body_name.startswith("right_finger"):
            continue
        for geom in body.geoms:
            geom.rgba = list(rgba)
            geom.material = material_name


def _add_ycb_object_and_bin(
    spec: mujoco.MjSpec,
    *,
    visual_mesh_path: str | Path | None,
    collision_mesh_path: str | Path | None,
    collision_mesh_paths: Sequence[str | Path] | None,
    texture_path: str | Path | None,
    mass: float,
    object_friction: tuple[float, float, float],
    mesh_min_z: float,
    bin_friction: tuple[float, float, float],
    bin_center_xy: tuple[float, float],
    bin_inner_size: tuple[float, float],
    bin_wall_height: float,
    bin_wall_thickness: float,
) -> None:
    if visual_mesh_path is None or collision_mesh_path is None or texture_path is None:
        raise ValueError("ycb_object requires visual, collision, and texture paths")
    visual_path = Path(visual_mesh_path).expanduser().resolve()
    inertial_collision_path = Path(collision_mesh_path).expanduser().resolve()
    if collision_mesh_paths is None:
        raw_collision_paths = (collision_mesh_path,)
    elif isinstance(collision_mesh_paths, (str, Path)):
        raw_collision_paths = (collision_mesh_paths,)
    else:
        raw_collision_paths = tuple(collision_mesh_paths)
    effective_collision_paths = tuple(
        Path(path).expanduser().resolve()
        for path in raw_collision_paths
    )
    if not effective_collision_paths:
        raise ValueError("ycb_object requires at least one collision mesh")
    texture = Path(texture_path).expanduser().resolve()
    required_paths = [
        ("visual mesh", visual_path),
        ("inertial collision mesh", inertial_collision_path),
        ("texture", texture),
    ]
    required_paths.extend(
        (f"collision mesh part {index}", path)
        for index, path in enumerate(effective_collision_paths)
    )
    for label, path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"YCB {label} does not exist: {path}")
    if float(mass) <= 0.0:
        raise ValueError(f"object_mass must be positive, got {mass}")
    if len(object_friction) != 3 or any(float(value) < 0.0 for value in object_friction):
        raise ValueError(
            "object_friction must contain three non-negative values, "
            f"got {object_friction}"
        )
    if len(bin_friction) != 3 or any(float(value) < 0.0 for value in bin_friction):
        raise ValueError(
            "bin_friction must contain three non-negative values, "
            f"got {bin_friction}"
        )

    visual_mesh_name = "task_object_visual_mesh"
    texture_name = "task_object_texture"
    material_name = "task_object_material"
    spec.add_mesh(name=visual_mesh_name, file=str(visual_path))
    use_inertial_geom = len(effective_collision_paths) > 1 or (
        effective_collision_paths[0] != inertial_collision_path
    )
    if use_inertial_geom:
        spec.add_mesh(name="task_object_inertial_mesh", file=str(inertial_collision_path))
    collision_mesh_names: list[str] = []
    for index, path in enumerate(effective_collision_paths):
        mesh_name = (
            "task_object_collision_mesh"
            if index == 0
            else f"task_object_collision_mesh_{index:03d}"
        )
        spec.add_mesh(name=mesh_name, file=str(path))
        collision_mesh_names.append(mesh_name)
    spec.add_texture(
        name=texture_name,
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        file=str(texture),
    )
    spec.add_material(
        name=material_name,
        textures=_rgb_texture_slots(texture_name),
        texuniform=False,
        rgba=[1.0, 1.0, 1.0, 1.0],
        specular=0.12,
        shininess=0.16,
    )

    start_pos = [0.12, -0.08, TABLE_TOP_Z - float(mesh_min_z) + 0.003]
    body = spec.worldbody.add_body(name="task_object", pos=start_pos)
    body.add_joint(name="task_object_free", type=mujoco.mjtJoint.mjJNT_FREE)
    if use_inertial_geom:
        body.add_geom(
            name="task_object_inertial",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="task_object_inertial_mesh",
            mass=float(mass),
            contype=0,
            conaffinity=0,
            rgba=[0.0, 0.0, 0.0, 0.0],
            group=3,
        )
    for index, mesh_name in enumerate(collision_mesh_names):
        collision = body.add_geom(
            name=(
                "task_object_collision"
                if index == 0
                else f"task_object_collision_part_{index:03d}"
            ),
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            mass=0.0 if use_inertial_geom else float(mass),
            friction=list(object_friction),
            rgba=[0.0, 0.0, 0.0, 0.0],
            group=3,
        )
        collision.condim = 6
    visual = body.add_geom(
        name="task_object_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=visual_mesh_name,
        mass=0.0,
        contype=0,
        conaffinity=0,
        group=1,
    )
    visual.material = material_name

    inner_x, inner_y = (float(bin_inner_size[0]), float(bin_inner_size[1]))
    wall_height = float(bin_wall_height)
    wall_thickness = float(bin_wall_thickness)
    if min(inner_x, inner_y, wall_height, wall_thickness) <= 0.0:
        raise ValueError("bin dimensions must be positive")
    center_x, center_y = (float(bin_center_xy[0]), float(bin_center_xy[1]))
    bin_body = spec.worldbody.add_body(name="pick_place_bin", pos=[center_x, center_y, TABLE_TOP_Z])
    floor_thickness = 0.006
    floor = bin_body.add_geom(
        name="pick_place_bin_floor",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.5 * inner_x + wall_thickness, 0.5 * inner_y + wall_thickness, 0.5 * floor_thickness],
        pos=[0.0, 0.0, 0.5 * floor_thickness],
        rgba=[0.22, 0.28, 0.34, 1.0],
        friction=list(bin_friction),
    )
    floor.condim = 6
    wall_z = floor_thickness + 0.5 * wall_height
    for name, size, pos in (
        (
            "pick_place_bin_wall_x_neg",
            [0.5 * wall_thickness, 0.5 * inner_y + wall_thickness, 0.5 * wall_height],
            [-0.5 * inner_x - 0.5 * wall_thickness, 0.0, wall_z],
        ),
        (
            "pick_place_bin_wall_x_pos",
            [0.5 * wall_thickness, 0.5 * inner_y + wall_thickness, 0.5 * wall_height],
            [0.5 * inner_x + 0.5 * wall_thickness, 0.0, wall_z],
        ),
        (
            "pick_place_bin_wall_y_neg",
            [0.5 * inner_x, 0.5 * wall_thickness, 0.5 * wall_height],
            [0.0, -0.5 * inner_y - 0.5 * wall_thickness, wall_z],
        ),
        (
            "pick_place_bin_wall_y_pos",
            [0.5 * inner_x, 0.5 * wall_thickness, 0.5 * wall_height],
            [0.0, 0.5 * inner_y + 0.5 * wall_thickness, wall_z],
        ),
    ):
        wall = bin_body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            pos=pos,
            rgba=[0.28, 0.36, 0.44, 1.0],
            friction=list(bin_friction),
        )
        wall.condim = 6


def _add_cube(
    spec: mujoco.MjSpec,
    *,
    cube_half_extent: float = CUBE_HALF_EXTENT,
    cube_mass: float = 0.08,
    cube_friction: tuple[float, float, float] = (1.2, 0.05, 0.01),
    texture_path: str | Path | None = None,
    face_texture_paths: dict[str, str | Path] | None = None,
    face_texture_atlas_path: str | Path | None = None,
    visual_face_offset: float = CUBE_TEXTURE_FACE_OFFSET,
    texrepeat: tuple[float, float] = (1.0, 1.0),
    rgba: tuple[float, float, float, float] = (0.92, 0.22, 0.12, 1.0),
    specular: float = 0.16,
    shininess: float = 0.18,
    reflectance: float = 0.02,
) -> None:
    cube_half_extent = float(cube_half_extent)
    cube_mass = float(cube_mass)
    cube_friction = tuple(float(x) for x in cube_friction)
    if cube_mass <= 0.0:
        raise ValueError(f"cube_mass must be positive, got {cube_mass}")
    if len(cube_friction) != 3 or any(value < 0.0 for value in cube_friction):
        raise ValueError(f"cube_friction must contain three non-negative values, got {cube_friction}")
    has_face_textures = face_texture_paths is not None or face_texture_atlas_path is not None
    cube_pos = (CUBE_START_POS[0], CUBE_START_POS[1], TABLE_CENTER[2] + TABLE_HALF_EXTENTS[2] + cube_half_extent + 0.003)
    body = spec.worldbody.add_body(name="cube", pos=list(cube_pos), quat=list(CUBE_START_QUAT_WXYZ))
    joint = body.add_joint()
    joint.name = "cube_free"
    joint.type = mujoco.mjtJoint.mjJNT_FREE
    material_name = None
    if not has_face_textures:
        material_name = _add_cube_visual_material(
            spec,
            texture_path=texture_path,
            texrepeat=texrepeat,
            rgba=rgba,
            specular=float(specular),
            shininess=float(shininess),
            reflectance=float(reflectance),
        )
    geom = body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[cube_half_extent, cube_half_extent, cube_half_extent],
        rgba=[0.74, 0.69, 0.59, 1.0] if has_face_textures else list(rgba),
        mass=cube_mass,
    )
    if material_name is not None:
        geom.material = material_name
    geom.friction = list(cube_friction)
    geom.condim = 6
    if has_face_textures:
        _add_cube_face_visuals(
            spec,
            body,
            cube_half_extent=cube_half_extent,
            face_texture_paths=face_texture_paths,
            face_texture_atlas_path=face_texture_atlas_path,
            visual_face_offset=float(visual_face_offset),
            rgba=rgba,
            specular=float(specular),
            shininess=float(shininess),
            reflectance=float(reflectance),
        )


def _add_cube_visual_material(
    spec: mujoco.MjSpec,
    *,
    texture_path: str | Path | None,
    texrepeat: tuple[float, float],
    rgba: tuple[float, float, float, float],
    specular: float,
    shininess: float,
    reflectance: float,
) -> str | None:
    rgba = _validate_rgba_tuple("cube_rgba", tuple(float(x) for x in rgba))
    if texture_path is None:
        return None
    if len(tuple(texrepeat)) != 2 or any(float(value) <= 0.0 for value in texrepeat):
        raise ValueError(f"cube_texrepeat must contain two positive values, got {texrepeat}")
    path = Path(texture_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"cube texture does not exist: {path}")

    texture_name = "cube_box_texture"
    material_name = "cube_box_material"
    spec.add_texture(
        name=texture_name,
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        file=str(path),
    )
    spec.add_material(
        name=material_name,
        textures=_rgb_texture_slots(texture_name),
        texuniform=True,
        texrepeat=[float(texrepeat[0]), float(texrepeat[1])],
        specular=float(specular),
        shininess=float(shininess),
        reflectance=float(reflectance),
        rgba=list(rgba),
    )
    return material_name


def cube_visual_face_vertices(
    face: str,
    half_side: float,
    half_height: float,
    visual_face_offset: float,
) -> list[float]:
    """Return four local vertices for one UV-mapped cube face."""

    face = str(face).strip().lower()
    half_side = float(half_side)
    half_height = float(half_height)
    visual_face_offset = float(visual_face_offset)
    if half_side <= 0.0 or half_height <= 0.0:
        raise ValueError(f"cube visual half extents must be positive, got {(half_side, half_height)}")
    if visual_face_offset < 0.0:
        raise ValueError(f"cube visual_face_offset must be non-negative, got {visual_face_offset}")
    if face == "front":
        center = np.asarray([0.0, half_side + visual_face_offset, 0.0], dtype=np.float64)
        u_axis = np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
        v_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        u_half, v_half = half_side, half_height
    elif face == "back":
        center = np.asarray([0.0, -half_side - visual_face_offset, 0.0], dtype=np.float64)
        u_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        v_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        u_half, v_half = half_side, half_height
    elif face == "left":
        center = np.asarray([-half_side - visual_face_offset, 0.0, 0.0], dtype=np.float64)
        u_axis = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
        v_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        u_half, v_half = half_side, half_height
    elif face == "right":
        center = np.asarray([half_side + visual_face_offset, 0.0, 0.0], dtype=np.float64)
        u_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        v_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        u_half, v_half = half_side, half_height
    elif face == "top":
        center = np.asarray([0.0, 0.0, half_height + visual_face_offset], dtype=np.float64)
        u_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        v_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        u_half, v_half = half_side, half_side
    elif face == "bottom":
        center = np.asarray([0.0, 0.0, -half_height - visual_face_offset], dtype=np.float64)
        u_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        v_axis = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
        u_half, v_half = half_side, half_side
    else:
        raise ValueError(f"Unsupported cube visual face {face!r}; expected one of {CUBE_TEXTURE_FACE_NAMES}")
    points = (
        center - u_axis * u_half - v_axis * v_half,
        center + u_axis * u_half - v_axis * v_half,
        center + u_axis * u_half + v_axis * v_half,
        center - u_axis * u_half + v_axis * v_half,
    )
    return np.asarray(points, dtype=np.float64).reshape(-1).tolist()


def cube_visual_box_vertices(
    half_side: float,
    half_height: float,
    visual_face_offset: float,
) -> list[float]:
    vertices: list[float] = []
    for face in CUBE_TEXTURE_FACE_NAMES:
        vertices.extend(
            cube_visual_face_vertices(
                face,
                half_side=float(half_side),
                half_height=float(half_height),
                visual_face_offset=float(visual_face_offset),
            )
        )
    return vertices


def _cube_visual_box_faces() -> list[int]:
    faces: list[int] = []
    for idx in range(len(CUBE_TEXTURE_FACE_NAMES)):
        base = 4 * idx
        faces.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return faces


def _cube_visual_box_uvs() -> list[float]:
    u_cell = 1.0 / 3.0
    v_cell = 1.0 / 2.0
    uvs: list[float] = []
    for idx in range(len(CUBE_TEXTURE_FACE_NAMES)):
        col = idx % 3
        row = idx // 3
        u0 = col * u_cell
        u1 = (col + 1) * u_cell
        v0 = row * v_cell
        v1 = (row + 1) * v_cell
        uvs.extend([u0, v0, u1, v0, u1, v1, u0, v1])
    return uvs


def _add_cube_face_visuals(
    spec: mujoco.MjSpec,
    body,
    *,
    cube_half_extent: float,
    face_texture_paths: dict[str, str | Path] | None,
    face_texture_atlas_path: str | Path | None,
    visual_face_offset: float,
    rgba: tuple[float, float, float, float],
    specular: float,
    shininess: float,
    reflectance: float,
) -> None:
    rgba = _validate_rgba_tuple("cube_rgba", tuple(float(x) for x in rgba))
    if face_texture_atlas_path is not None:
        _add_cube_atlas_visual(
            spec,
            body,
            cube_half_extent=float(cube_half_extent),
            atlas_path=face_texture_atlas_path,
            visual_face_offset=float(visual_face_offset),
            rgba=rgba,
            specular=float(specular),
            shininess=float(shininess),
            reflectance=float(reflectance),
        )
        return
    if face_texture_paths is None:
        raise ValueError("cube face_texture_paths or face_texture_atlas_path must be set")
    paths = _validate_cube_face_texture_paths(face_texture_paths)

    for face in CUBE_TEXTURE_FACE_NAMES:
        texture_name = f"cube_{face}_texture"
        material_name = f"cube_{face}_material"
        mesh_name = f"cube_visual_{face}_mesh"
        spec.add_texture(
            name=texture_name,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            file=str(paths[face]),
        )
        spec.add_material(
            name=material_name,
            textures=_rgb_texture_slots(texture_name),
            texuniform=False,
            texrepeat=[1.0, 1.0],
            specular=float(specular),
            shininess=float(shininess),
            reflectance=float(reflectance),
            rgba=list(rgba),
        )
        mesh = spec.add_mesh(name=mesh_name)
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        mesh.uservert = cube_visual_face_vertices(
            face,
            half_side=float(cube_half_extent),
            half_height=float(cube_half_extent),
            visual_face_offset=float(visual_face_offset),
        )
        mesh.userface = [0, 1, 2, 0, 2, 3]
        mesh.usertexcoord = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        mesh.userfacetexcoord = [0, 1, 2, 0, 2, 3]
        geom = body.add_geom(
            name=f"cube_visual_{face}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            rgba=list(rgba),
            mass=0.0,
            contype=0,
            conaffinity=0,
            group=1,
        )
        geom.material = material_name


def _validate_cube_face_texture_paths(face_texture_paths: dict[str, str | Path]) -> dict[str, Path]:
    unknown = sorted(set(face_texture_paths).difference(CUBE_TEXTURE_FACE_NAMES))
    missing = [face for face in CUBE_TEXTURE_FACE_NAMES if face not in face_texture_paths]
    if unknown or missing:
        raise ValueError(
            "cube face_texture_paths must contain exactly "
            f"{CUBE_TEXTURE_FACE_NAMES}; missing={missing}, unknown={unknown}"
        )
    paths: dict[str, Path] = {}
    for face in CUBE_TEXTURE_FACE_NAMES:
        path = Path(face_texture_paths[face]).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"cube {face} face texture does not exist: {path}")
        paths[face] = path
    return paths


def _add_cube_atlas_visual(
    spec: mujoco.MjSpec,
    body,
    *,
    cube_half_extent: float,
    atlas_path: str | Path,
    visual_face_offset: float,
    rgba: tuple[float, float, float, float],
    specular: float,
    shininess: float,
    reflectance: float,
) -> None:
    path = Path(atlas_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"cube face texture atlas does not exist: {path}")
    texture_name = "cube_face_atlas_texture"
    material_name = "cube_face_atlas_material"
    spec.add_texture(
        name=texture_name,
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        file=str(path),
    )
    spec.add_material(
        name=material_name,
        textures=_rgb_texture_slots(texture_name),
        texuniform=False,
        texrepeat=[1.0, 1.0],
        specular=float(specular),
        shininess=float(shininess),
        reflectance=float(reflectance),
        rgba=list(rgba),
    )
    mesh = spec.add_mesh(name=CUBE_TEXTURE_ATLAS_MESH_NAME)
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    mesh.uservert = cube_visual_box_vertices(
        half_side=float(cube_half_extent),
        half_height=float(cube_half_extent),
        visual_face_offset=float(visual_face_offset),
    )
    mesh.userface = _cube_visual_box_faces()
    mesh.usertexcoord = _cube_visual_box_uvs()
    mesh.userfacetexcoord = _cube_visual_box_faces()
    geom = body.add_geom(
        name=CUBE_TEXTURE_ATLAS_GEOM_NAME,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=CUBE_TEXTURE_ATLAS_MESH_NAME,
        rgba=list(rgba),
        mass=0.0,
        contype=0,
        conaffinity=0,
        group=1,
    )
    geom.material = material_name


def _add_mini_keyboard(
    spec: mujoco.MjSpec,
    *,
    friction: tuple[float, float, float] = (5.0, 0.1, 0.01),
    support_box_height: float = MINI_KEYBOARD_SUPPORT_BOX_HEIGHT,
    white_key_push_distance: float = MINI_KEYBOARD_WHITE_KEY_PUSH_DISTANCE,
    black_key_push_distance: float = MINI_KEYBOARD_BLACK_KEY_PUSH_DISTANCE,
) -> None:
    attach_mini_keyboard_asset(
        spec,
        mjcf_path=MINI_KEYBOARD_MJCF_PATH,
        body_name="mini_keyboard",
        position=(0.15, 0.0, TABLE_TOP_Z + float(support_box_height)),
        quat_wxyz=(0.70710678, 0.0, 0.0, -0.70710678),
        support_box_height=float(support_box_height),
        root_z_offset=MINI_KEYBOARD_ROOT_Z_OFFSET,
        friction=friction,
        key_joint_names=MINI_KEYBOARD_KEY_JOINT_NAMES,
        white_key_push_distance=float(white_key_push_distance),
        black_key_push_distance=float(black_key_push_distance),
    )


def _add_pump_bottle(
    spec: mujoco.MjSpec,
    *,
    mass: float = 0.18,
    button_mass: float = 0.05,
    button_joint_range: tuple[float, float] = (-0.013, 0.0),
    button_joint_stiffness: float = 35.0,
    button_joint_damping: float = 5.0,
    button_actuator_kp: float = 35.0,
    button_force_site_visible: bool = True,
    mesh_scale: float = 1.3,
    friction: tuple[float, float, float] = (2.0, 0.01, 0.0002),
    body_collision_mode: str = PUMP_BOTTLE_COLLISION_MODE_CYLINDER,
    neck_collision_mode: str = PUMP_BOTTLE_COLLISION_MODE_CYLINDER,
    visual_mesh_source: str = PUMP_BOTTLE_VISUAL_MESH_SOURCE_VISUAL,
    contact_margin: float = PUMP_BOTTLE_CONTACT_MARGIN,
    asset_id: str = LEGACY_PUMP_BOTTLE_ID,
    manifest_path: str | Path | None = None,
) -> None:
    mass = float(mass)
    button_mass = float(button_mass)
    button_joint_range = tuple(float(x) for x in button_joint_range)
    button_joint_stiffness = float(button_joint_stiffness)
    button_joint_damping = float(button_joint_damping)
    button_actuator_kp = float(button_actuator_kp)
    button_force_site_visible = bool(button_force_site_visible)
    mesh_scale = float(mesh_scale)
    friction = tuple(float(x) for x in friction)
    contact_margin = float(contact_margin)
    asset_id = str(asset_id).strip()
    if asset_id != LEGACY_PUMP_BOTTLE_ID:
        _add_suite_pump_bottle(
            spec,
            asset_id=asset_id,
            manifest_path=manifest_path,
            mass=mass,
            button_mass=button_mass,
            button_joint_range=button_joint_range,
            button_joint_stiffness=button_joint_stiffness,
            button_joint_damping=button_joint_damping,
            button_actuator_kp=button_actuator_kp,
            button_force_site_visible=button_force_site_visible,
            friction=friction,
            contact_margin=contact_margin,
            collision_mesh_mode=body_collision_mode,
        )
        return
    body_collision_mode = _normalize_pump_bottle_collision_mode(
        body_collision_mode,
        label="pump_bottle.lower_collision_mode",
    )
    neck_collision_mode = _normalize_pump_bottle_collision_mode(
        neck_collision_mode,
        label="pump_bottle.neck_collision_mode",
    )
    visual_mesh_source = _normalize_pump_bottle_visual_mesh_source(visual_mesh_source)
    use_collision_visual = visual_mesh_source == PUMP_BOTTLE_VISUAL_MESH_SOURCE_COLLISION
    if mass <= 0.0:
        raise ValueError(f"pump_bottle.mass must be positive, got {mass}")
    if button_mass <= 0.0:
        raise ValueError(f"pump_bottle.button_mass must be positive, got {button_mass}")
    if len(button_joint_range) != 2 or not (button_joint_range[0] < button_joint_range[1]):
        raise ValueError(f"pump_bottle.button_joint_range must be increasing [lo, hi], got {button_joint_range}")
    if button_joint_stiffness < 0.0:
        raise ValueError(f"pump_bottle.button_joint_stiffness must be non-negative, got {button_joint_stiffness}")
    if button_joint_damping < 0.0:
        raise ValueError(f"pump_bottle.button_joint_damping must be non-negative, got {button_joint_damping}")
    if button_actuator_kp < 0.0:
        raise ValueError(f"pump_bottle.button_actuator_kp must be non-negative, got {button_actuator_kp}")
    if mesh_scale <= 0.0:
        raise ValueError(f"pump_bottle.mesh_scale must be positive, got {mesh_scale}")
    if len(friction) != 3 or any(value < 0.0 for value in friction):
        raise ValueError(f"pump_bottle.friction must contain three non-negative values, got {friction}")
    if contact_margin < 0.0:
        raise ValueError(f"pump_bottle.contact_margin must be non-negative, got {contact_margin}")

    mesh_dir = PUMP_BOTTLE_MESH_DIR
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Pump bottle mesh directory does not exist: {mesh_dir}")

    for mesh_name, file_name, _ in (*PUMP_BOTTLE_MESHES, PUMP_BOTTLE_BUTTON_MESH):
        spec.add_mesh(
            name=mesh_name,
            content_type="model/obj",
            file=str(mesh_dir / file_name),
            scale=[mesh_scale, mesh_scale, mesh_scale],
        )

    body = spec.worldbody.add_body(
        name="pump_bottle_bottle_base",
        pos=[0.15, 0.0, TABLE_TOP_Z],
        quat=[1.0, 0.0, 0.0, 0.0],
    )
    body.explicitinertial = True
    body.ipos = [0.0, 0.0, 0.104 * mesh_scale / 1.3]
    body.mass = mass
    body.inertia = [0.0009295, 0.0009295, 0.0002028]
    body.add_freejoint(name="pump_bottle_bottle_free")
    for mesh_name, _, rgba in PUMP_BOTTLE_MESHES:
        geom = body.add_geom(
            name=mesh_name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            rgba=list(rgba),
            group=1,
            friction=list(friction),
            condim=6,
        )
        if mesh_name == PUMP_BOTTLE_BODY_MESH_NAME:
            use_mesh_collision = body_collision_mode == PUMP_BOTTLE_COLLISION_MODE_MESH
        elif mesh_name == PUMP_BOTTLE_NECK_MESH_NAME:
            use_mesh_collision = neck_collision_mode == PUMP_BOTTLE_COLLISION_MODE_MESH
        else:
            use_mesh_collision = True
        if use_mesh_collision:
            geom.contype = 3
            geom.conaffinity = 3
        else:
            geom.contype = 0
            geom.conaffinity = 0
            if use_collision_visual:
                geom.group = PUMP_BOTTLE_PROXY_COLLISION_GROUP
                geom.rgba = [float(rgba[0]), float(rgba[1]), float(rgba[2]), 0.0]
        geom.margin = contact_margin
    if body_collision_mode == PUMP_BOTTLE_COLLISION_MODE_CYLINDER:
        _, _, body_rgba = _pump_bottle_mesh_entry(PUMP_BOTTLE_BODY_MESH_NAME)
        _add_pump_bottle_mesh_bounds_cylinder_proxy(
            body,
            mesh_dir=mesh_dir,
            mesh_name=PUMP_BOTTLE_BODY_MESH_NAME,
            geom_name=PUMP_BOTTLE_BODY_COLLISION_GEOM_NAME,
            mesh_scale=mesh_scale,
            friction=friction,
            visual_rgba=body_rgba if use_collision_visual else None,
            contact_margin=contact_margin,
        )
    if neck_collision_mode == PUMP_BOTTLE_COLLISION_MODE_CYLINDER:
        _, _, neck_rgba = _pump_bottle_mesh_entry(PUMP_BOTTLE_NECK_MESH_NAME)
        _add_pump_bottle_neck_collision_proxy(
            body,
            mesh_dir=mesh_dir,
            mesh_scale=mesh_scale,
            friction=friction,
            visual_rgba=neck_rgba if use_collision_visual else None,
            contact_margin=contact_margin,
        )

    button = body.add_body(
        name="pump_bottle_button",
        pos=[0.0001313, -0.00259324, 0.24138],
        quat=[1.0, 0.0, 0.0, 0.0],
    )
    button.explicitinertial = True
    button.ipos = [-0.00273, 0.000455, -0.055185]
    button.mass = button_mass
    button.inertia = [0.0000845, 0.0000845, 0.0000845]
    button.add_joint(
        name="pump_bottle_button_slide",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0.0, 0.0, 1.0],
        stiffness=button_joint_stiffness,
        limited=True,
        range=list(button_joint_range),
        damping=button_joint_damping,
    )
    button_mesh_name, _, button_rgba = PUMP_BOTTLE_BUTTON_MESH
    button_geom = button.add_geom(
        name=button_mesh_name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=button_mesh_name,
        rgba=list(button_rgba),
        group=1,
        friction=list(friction),
        condim=6,
    )
    button_geom.contype = 3
    button_geom.conaffinity = 3
    button_geom.margin = contact_margin
    _add_pump_bottle_button_extension(
        button,
        mesh_dir=mesh_dir,
        mesh_scale=mesh_scale,
        rgba=button_rgba,
        friction=friction,
        contact_margin=contact_margin,
    )
    button.add_site(
        name="pump_bottle_button_force_site",
        pos=[-0.00273, 0.000455, 0.0026],
        size=[0.0078],
        rgba=[0.1, 0.4, 1.0, 1.0 if button_force_site_visible else 0.0],
    )
    spec.add_exclude(
        name="pump_bottle_base_button",
        bodyname1="pump_bottle_bottle_base",
        bodyname2="pump_bottle_button",
    )
    spec.add_actuator(
        name="pump_bottle_button_position",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        target="pump_bottle_button_slide",
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        gainprm=[button_actuator_kp, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        biasprm=[0.0, -button_actuator_kp, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ctrllimited=True,
        ctrlrange=list(button_joint_range),
    )


def _box_inertia(mass: float, extents: np.ndarray) -> list[float]:
    x, y, z = np.maximum(np.asarray(extents, dtype=np.float64).reshape(3), 1.0e-4)
    scale = float(mass) / 12.0
    return [scale * (y * y + z * z), scale * (x * x + z * z), scale * (x * x + y * y)]


def _add_suite_pump_bottle(
    spec: mujoco.MjSpec,
    *,
    asset_id: str,
    manifest_path: str | Path | None,
    mass: float,
    button_mass: float,
    button_joint_range: tuple[float, float],
    button_joint_stiffness: float,
    button_joint_damping: float,
    button_actuator_kp: float,
    button_force_site_visible: bool,
    friction: tuple[float, float, float],
    contact_margin: float,
    collision_mesh_mode: str,
) -> None:
    """Attach one processed NVIDIA bottle using the legacy task-facing names."""

    if mass <= 0.0 or button_mass <= 0.0:
        raise ValueError(f"Pump-bottle body/button masses must be positive, got {mass}, {button_mass}")
    if len(button_joint_range) != 2 or not button_joint_range[0] < button_joint_range[1]:
        raise ValueError(f"pump_bottle.button_joint_range must be increasing, got {button_joint_range}")
    if any(value < 0.0 for value in (button_joint_stiffness, button_joint_damping, button_actuator_kp)):
        raise ValueError("Pump-bottle stiffness, damping, and actuator kp must be non-negative")
    if len(friction) != 3 or any(value < 0.0 for value in friction):
        raise ValueError(f"pump_bottle.friction must contain three non-negative values, got {friction}")

    collision_mesh_mode = _normalize_pump_bottle_suite_collision_mode(collision_mesh_mode)
    bottle = pump_bottle_spec_by_id(asset_id, manifest_path)
    static_mesh_name = "pump_bottle_suite_static_mesh"
    button_mesh_name = "pump_bottle_suite_button_mesh"
    scale = [float(bottle.scale)] * 3
    spec.add_mesh(
        name=static_mesh_name,
        content_type="model/obj",
        file=str(bottle.static_mesh_path),
        scale=scale,
    )

    use_nvidia_convex = collision_mesh_mode == PUMP_BOTTLE_SUITE_COLLISION_NVIDIA_CONVEX
    static_collision_mesh_names: list[str] = []
    button_collision_mesh_names: list[str] = []
    if use_nvidia_convex:
        for index, path in enumerate(bottle.static_collision_mesh_paths):
            mesh_name = f"pump_bottle_suite_static_collision_mesh_{index:03d}"
            spec.add_mesh(name=mesh_name, content_type="model/obj", file=str(path), scale=scale)
            static_collision_mesh_names.append(mesh_name)
        for index, path in enumerate(bottle.button_collision_mesh_paths):
            mesh_name = f"pump_bottle_suite_button_collision_mesh_{index:03d}"
            spec.add_mesh(name=mesh_name, content_type="model/obj", file=str(path), scale=scale)
            button_collision_mesh_names.append(mesh_name)
    spec.add_mesh(
        name=button_mesh_name,
        content_type="model/obj",
        file=str(bottle.button_mesh_path),
        scale=scale,
    )

    body_bounds = np.asarray(bottle.static_bounds, dtype=np.float64) * float(bottle.scale)
    body = spec.worldbody.add_body(
        name="pump_bottle_bottle_base",
        pos=[0.15, 0.0, TABLE_TOP_Z],
        quat=[1.0, 0.0, 0.0, 0.0],
    )
    body.explicitinertial = True
    body.ipos = (0.5 * (body_bounds[0] + body_bounds[1])).tolist()
    body.mass = float(mass)
    body.inertia = _box_inertia(float(mass), bottle.scaled_static_extents)
    body.add_freejoint(name="pump_bottle_bottle_free")
    if use_nvidia_convex:
        body.add_geom(
            name="pump_bottle_body_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=static_mesh_name,
            rgba=list(bottle.body_rgba),
            group=1,
            mass=0.0,
            contype=0,
            conaffinity=0,
        )
        _add_pump_bottle_collision_geoms(
            body,
            mesh_names=static_collision_mesh_names,
            base_name="pump_bottle_body_collision",
            pos=bottle.scaled_collision_rebase_xyz,
            friction=friction,
            contact_margin=contact_margin,
        )
    else:
        body_geom = body.add_geom(
            name="pump_bottle_body_collision",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=static_mesh_name,
            rgba=list(bottle.body_rgba),
            group=1,
            friction=list(friction),
            condim=6,
        )
        body_geom.contype = 3
        body_geom.conaffinity = 3
        body_geom.margin = float(contact_margin)

    button_bounds = np.asarray(bottle.button_bounds, dtype=np.float64) * float(bottle.scale)
    button = body.add_body(name="pump_bottle_button", pos=[0.0, 0.0, 0.0])
    button.explicitinertial = True
    button.ipos = (0.5 * (button_bounds[0] + button_bounds[1])).tolist()
    button.mass = float(button_mass)
    button.inertia = _box_inertia(float(button_mass), bottle.scaled_button_extents)
    button.add_joint(
        name="pump_bottle_button_slide",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0.0, 0.0, 1.0],
        stiffness=float(button_joint_stiffness),
        limited=True,
        range=list(button_joint_range),
        damping=float(button_joint_damping),
    )
    if use_nvidia_convex:
        button.add_geom(
            name="pump_bottle_button_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=button_mesh_name,
            rgba=list(bottle.button_rgba),
            group=1,
            mass=0.0,
            contype=0,
            conaffinity=0,
        )
        _add_pump_bottle_collision_geoms(
            button,
            mesh_names=button_collision_mesh_names,
            base_name="pump_bottle_button_collision",
            pos=bottle.scaled_collision_rebase_xyz,
            friction=friction,
            contact_margin=contact_margin,
        )
    else:
        button_geom = button.add_geom(
            name="pump_bottle_button_collision",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=button_mesh_name,
            rgba=list(bottle.button_rgba),
            group=1,
            friction=list(friction),
            condim=6,
        )
        button_geom.contype = 3
        button_geom.conaffinity = 3
        button_geom.margin = float(contact_margin)
    button.add_site(
        name="pump_bottle_button_force_site",
        pos=list(bottle.press_site),
        size=[0.0078],
        rgba=[0.1, 0.4, 1.0, 1.0 if button_force_site_visible else 0.0],
    )
    spec.add_exclude(
        name="pump_bottle_base_button",
        bodyname1="pump_bottle_bottle_base",
        bodyname2="pump_bottle_button",
    )
    spec.add_actuator(
        name="pump_bottle_button_position",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        target="pump_bottle_button_slide",
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        gainprm=[button_actuator_kp, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        biasprm=[0.0, -button_actuator_kp, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ctrllimited=True,
        ctrlrange=list(button_joint_range),
    )


def _add_pump_bottle_collision_geoms(
    body: mujoco.MjsBody,
    *,
    mesh_names: Sequence[str],
    base_name: str,
    pos: Sequence[float],
    friction: tuple[float, float, float],
    contact_margin: float,
) -> None:
    if not mesh_names:
        raise ValueError(f"{base_name} requires at least one collision mesh")
    for index, mesh_name in enumerate(mesh_names):
        geom = body.add_geom(
            name=base_name if index == 0 else f"{base_name}_part_{index:03d}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            pos=[float(value) for value in pos],
            rgba=[0.0, 0.0, 0.0, 0.0],
            group=3,
            mass=0.0,
            friction=list(friction),
            condim=6,
        )
        geom.contype = 3
        geom.conaffinity = 3
        geom.margin = float(contact_margin)


def _normalize_pump_bottle_suite_collision_mode(value: str) -> str:
    text = str(value).strip().lower().replace("-", "_")
    aliases = {
        "cylinder": PUMP_BOTTLE_SUITE_COLLISION_PROCESSED_SINGLE_HULL,
        "mesh": PUMP_BOTTLE_SUITE_COLLISION_PROCESSED_SINGLE_HULL,
        "original": PUMP_BOTTLE_SUITE_COLLISION_PROCESSED_SINGLE_HULL,
        "original_mesh": PUMP_BOTTLE_SUITE_COLLISION_PROCESSED_SINGLE_HULL,
        "single_mesh": PUMP_BOTTLE_SUITE_COLLISION_PROCESSED_SINGLE_HULL,
        "processed_single_hull": PUMP_BOTTLE_SUITE_COLLISION_PROCESSED_SINGLE_HULL,
        "convex_decomposition": PUMP_BOTTLE_SUITE_COLLISION_NVIDIA_CONVEX,
        "nvidia_convex": PUMP_BOTTLE_SUITE_COLLISION_NVIDIA_CONVEX,
    }
    try:
        return aliases[text]
    except KeyError as exc:
        raise ValueError(
            "Pump-bottle suite collision mode must be 'processed_single_hull' or "
            f"'nvidia_convex', got {value!r}"
        ) from exc


def _normalize_pump_bottle_collision_mode(value: str, *, label: str) -> str:
    text = str(value).strip().lower().replace("-", "_")
    aliases = {
        "cylinder": PUMP_BOTTLE_COLLISION_MODE_CYLINDER,
        "proxy": PUMP_BOTTLE_COLLISION_MODE_CYLINDER,
        "cylinder_proxy": PUMP_BOTTLE_COLLISION_MODE_CYLINDER,
        "mesh": PUMP_BOTTLE_COLLISION_MODE_MESH,
        "original": PUMP_BOTTLE_COLLISION_MODE_MESH,
        "original_mesh": PUMP_BOTTLE_COLLISION_MODE_MESH,
        "obj": PUMP_BOTTLE_COLLISION_MODE_MESH,
    }
    if text not in aliases:
        raise ValueError(f"{label} must be 'cylinder' or 'mesh', got {value!r}")
    return aliases[text]


def _normalize_pump_bottle_visual_mesh_source(value: str) -> str:
    text = str(value).strip().lower().replace("-", "_")
    aliases = {
        "visual": PUMP_BOTTLE_VISUAL_MESH_SOURCE_VISUAL,
        "default": PUMP_BOTTLE_VISUAL_MESH_SOURCE_VISUAL,
        "original": PUMP_BOTTLE_VISUAL_MESH_SOURCE_VISUAL,
        "original_mesh": PUMP_BOTTLE_VISUAL_MESH_SOURCE_VISUAL,
        "collision": PUMP_BOTTLE_VISUAL_MESH_SOURCE_COLLISION,
        "collision_mesh": PUMP_BOTTLE_VISUAL_MESH_SOURCE_COLLISION,
        "collision_geoms": PUMP_BOTTLE_VISUAL_MESH_SOURCE_COLLISION,
    }
    if text not in aliases:
        raise ValueError(
            "pump_bottle.visual_mesh_source must be 'visual' or 'collision', "
            f"got {value!r}"
        )
    return aliases[text]


def _add_pump_bottle_button_extension(
    button: mujoco.MjsBody,
    *,
    mesh_dir: Path,
    mesh_scale: float,
    rgba: tuple[float, float, float, float],
    friction: tuple[float, float, float],
    contact_margin: float,
) -> None:
    _, button_file, _ = PUMP_BOTTLE_BUTTON_MESH
    bounds_min, bounds_max = _obj_vertex_bounds(mesh_dir / button_file)
    bounds_min = bounds_min * float(mesh_scale)
    bounds_max = bounds_max * float(mesh_scale)
    mesh_center = 0.5 * (bounds_min + bounds_max)
    if PUMP_BOTTLE_BUTTON_EXTENSION_DIRECTION >= 0.0:
        x_start = float(bounds_max[0]) - PUMP_BOTTLE_BUTTON_EXTENSION_OVERLAP
    else:
        x_start = float(bounds_min[0]) + PUMP_BOTTLE_BUTTON_EXTENSION_OVERLAP
    x_center = x_start + 0.5 * PUMP_BOTTLE_BUTTON_EXTENSION_DIRECTION * PUMP_BOTTLE_BUTTON_EXTENSION_LENGTH
    geom = button.add_geom(
        name=PUMP_BOTTLE_BUTTON_EXTENSION_GEOM_NAME,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[
            x_center,
            float(mesh_center[1]),
            -PUMP_BOTTLE_BUTTON_EXTENSION_HALF_THICKNESS,
        ],
        size=[
            0.5 * PUMP_BOTTLE_BUTTON_EXTENSION_LENGTH,
            PUMP_BOTTLE_BUTTON_EXTENSION_HALF_WIDTH,
            PUMP_BOTTLE_BUTTON_EXTENSION_HALF_THICKNESS,
        ],
        rgba=[float(rgba[0]), float(rgba[1]), float(rgba[2]), 0.78],
        group=1,
        friction=list(friction),
        condim=6,
    )
    geom.contype = 3
    geom.conaffinity = 3
    geom.margin = float(contact_margin)
    geom.priority = 2
    geom.solref = [0.002, 1.0]
    geom.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]


def _add_pump_bottle_neck_collision_proxy(
    body: mujoco.MjsBody,
    *,
    mesh_dir: Path,
    mesh_scale: float,
    friction: tuple[float, float, float],
    visual_rgba: tuple[float, float, float, float] | None,
    contact_margin: float,
) -> None:
    _add_pump_bottle_mesh_bounds_cylinder_proxy(
        body,
        mesh_dir=mesh_dir,
        mesh_name=PUMP_BOTTLE_NECK_MESH_NAME,
        geom_name=PUMP_BOTTLE_NECK_COLLISION_GEOM_NAME,
        mesh_scale=mesh_scale,
        friction=friction,
        visual_rgba=visual_rgba,
        contact_margin=contact_margin,
    )


def _add_pump_bottle_mesh_bounds_cylinder_proxy(
    body: mujoco.MjsBody,
    *,
    mesh_dir: Path,
    mesh_name: str,
    geom_name: str,
    mesh_scale: float,
    friction: tuple[float, float, float],
    visual_rgba: tuple[float, float, float, float] | None,
    contact_margin: float,
) -> None:
    mesh_entry = _pump_bottle_mesh_entry(mesh_name)
    bounds_min, bounds_max = _obj_vertex_bounds(mesh_dir / mesh_entry[1])
    center = 0.5 * (bounds_min + bounds_max) * float(mesh_scale)
    extents = (bounds_max - bounds_min) * float(mesh_scale)
    radius = 0.5 * max(float(extents[0]), float(extents[1]))
    halfheight = 0.5 * float(extents[2])
    _add_pump_bottle_cylinder_proxy(
        body,
        name=geom_name,
        pos=(float(center[0]), float(center[1]), float(center[2])),
        radius=radius,
        halfheight=halfheight,
        friction=friction,
        visual_rgba=visual_rgba,
        contact_margin=contact_margin,
    )


def _add_pump_bottle_cylinder_proxy(
    body: mujoco.MjsBody,
    *,
    name: str,
    pos: tuple[float, float, float],
    radius: float,
    halfheight: float,
    friction: tuple[float, float, float],
    visual_rgba: tuple[float, float, float, float] | None,
    contact_margin: float,
) -> None:
    rgba = PUMP_BOTTLE_PROXY_RGBA if visual_rgba is None else tuple(float(x) for x in visual_rgba)
    geom = body.add_geom(
        name=str(name),
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=list(pos),
        size=[float(radius), float(halfheight)],
        rgba=list(rgba),
        group=PUMP_BOTTLE_PROXY_COLLISION_GROUP if visual_rgba is None else 1,
        friction=list(friction),
        condim=6,
    )
    geom.contype = 3
    geom.conaffinity = 3
    geom.margin = float(contact_margin)
    geom.priority = 2
    geom.solref = [0.004, 1.0]
    geom.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]


def _pump_bottle_mesh_entry(mesh_name: str) -> tuple[str, str, tuple[float, float, float, float]]:
    for entry in PUMP_BOTTLE_MESHES:
        if entry[0] == str(mesh_name):
            return entry
    raise KeyError(f"Unknown pump bottle mesh {mesh_name!r}")


def _obj_vertex_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mins = np.full((3,), np.inf, dtype=np.float64)
    maxs = np.full((3,), -np.inf, dtype=np.float64)
    count = 0
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            vertex = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            mins = np.minimum(mins, vertex)
            maxs = np.maximum(maxs, vertex)
            count += 1
    if count == 0:
        raise ValueError(f"OBJ mesh has no vertices: {path}")
    return mins, maxs


def _add_cameras(spec: mujoco.MjSpec, *, camera_width: int = CAMERA_WIDTH, camera_height: int = CAMERA_HEIGHT) -> None:
    camera_specs = default_camera_specs(width=int(camera_width), height=int(camera_height))
    for name in ("ext_cam_0", "ext_cam_1"):
        camera_spec = camera_specs[name]
        pos = camera_spec.pos
        lookat = camera_spec.lookat
        if pos is None or lookat is None:
            raise ValueError(f"External camera {name!r} requires pos and lookat")
        spec.worldbody.add_camera(
            name=name,
            pos=list(pos),
            quat=list(look_at_quat_wxyz(pos, lookat)),
            fovy=float(camera_spec.fovy),
        )

    render_pos = (
        float(TABLE_CENTER[0]) + 0.55,
        float(TABLE_CENTER[1]) - 0.60,
        float(TABLE_TOP_Z) + 0.50,
    )
    render_lookat = (
        float(TABLE_CENTER[0]),
        float(TABLE_CENTER[1]),
        float(TABLE_TOP_Z) + 0.08,
    )
    spec.worldbody.add_camera(
        name=ONLINE_ROLLOUT_RENDER_CAMERA_NAME,
        pos=list(render_pos),
        quat=list(look_at_quat_wxyz(render_pos, render_lookat)),
        fovy=45.0,
    )

    hand_spec = camera_specs[HAND_CAMERA_NAME]
    hand_body = _find_first_body(spec, hand_spec.parent_body_candidates)
    if hand_body is None:
        raise KeyError(
            f"Cannot attach {HAND_CAMERA_NAME!r}; none of these bodies exist: {hand_spec.parent_body_candidates}"
        )
    hand_body.add_camera(
        name=HAND_CAMERA_NAME,
        pos=list(hand_spec.pos or (0.0, 0.0, 0.0)),
        quat=list(hand_spec.quat_wxyz or (1.0, 0.0, 0.0, 0.0)),
        fovy=float(hand_spec.fovy),
    )


def _find_first_body(spec: mujoco.MjSpec, names: tuple[str, ...]):
    for candidate in names:
        for body in spec.bodies:
            if body.name == candidate:
                return body
    return None
