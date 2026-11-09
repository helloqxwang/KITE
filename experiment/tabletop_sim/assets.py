"""Asset paths and tabletop constants for the local tabletop simulator."""

from __future__ import annotations

import os
from pathlib import Path


TABLETOP_SIM_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = TABLETOP_SIM_ROOT.parent
CROSS_EMBODIMENT_ROOT = EXPERIMENT_ROOT.parent
PROJECT_ROOT = CROSS_EMBODIMENT_ROOT.parent

SIM_INFRA_ASSETS_ROOT = CROSS_EMBODIMENT_ROOT / "assets"

MARVIN_WUJI_XML = (
    SIM_INFRA_ASSETS_ROOT
    / "robots"
    / "tianji_marvin_ccs_696"
    / "tianji_marvin_CCS_real_limit_wuji_collision.xml"
)
FRANKA_ROBOTIQ_XML = SIM_INFRA_ASSETS_ROOT / "robots" / "droid" / "fr3_robotiq_2f85.xml"
ROBOTIQ_2F85_XML = CROSS_EMBODIMENT_ROOT / "assets" / "composite" / "robotiq_2f85" / "robotiq_2f85.xml"
ALLEGRO_RIGHT_URDF = (
    CROSS_EMBODIMENT_ROOT / "assets" / "ohra_urdf_robot" / "allegro" / "allegro_hand_right_extended.urdf"
)
ALLEGRO_RIGHT_MJCF = (
    CROSS_EMBODIMENT_ROOT
    / "assets"
    / "ohra_urdf_robot"
    / "allegro"
    / "allegro_hand_right_extended_floating_actuated.xml"
)
ALLEGRO_RIGHT_XML = ALLEGRO_RIGHT_MJCF
BARRETT_URDF = CROSS_EMBODIMENT_ROOT / "assets" / "dro_urdf_robot" / "barrett" / "model_extended.urdf"
BARRETT_MJCF = (
    CROSS_EMBODIMENT_ROOT
    / "assets"
    / "dro_urdf_robot"
    / "barrett"
    / "model_extended_floating_actuated.xml"
)
WUJIHAND_RIGHT_XML = (
    CROSS_EMBODIMENT_ROOT / "assets" / "spider_robots" / "wujihand" / "right.xml"
)


PUMP_BOTTLE_ASSET_NAME = "pump_bottle005"
PUMP_BOTTLE_ASSET_DIR = (
    EXPERIMENT_ROOT / "hand_demonstration" / "object_assets" / PUMP_BOTTLE_ASSET_NAME
)
PUMP_BOTTLE_MESH_DIR = PUMP_BOTTLE_ASSET_DIR / "meshes"

MINI_KEYBOARD_ASSET_DIR = (
    CROSS_EMBODIMENT_ROOT / "assets" / "object_assets" / "akai_mpk_mini_mk3_mjcf"
)
MINI_KEYBOARD_MJCF_PATH = MINI_KEYBOARD_ASSET_DIR / "model.xml"
MINI_KEYBOARD_ROOT_Z_OFFSET = 0.010
MINI_KEYBOARD_SUPPORT_BOX_HEIGHT = 0.15

ROBOT_BASE_POS = (-0.45, 0.0, 0.87)

ARM_HOME_Q = {
    "left": (-0.014, -0.551, 0.556, -0.932, 1.098, 0.0, 0.0),
    "right": (-0.172, -0.559, -0.350, -1.153, -0.863, 0.0, 0.0),
}

TABLE_HALF_EXTENTS = (0.30, 0.60, 0.025)
TABLE_TOP_Z = 0.64
TABLE_CENTER = (0.15, 0.0, TABLE_TOP_Z - TABLE_HALF_EXTENTS[2])
FRANKA_ROBOTIQ_BASE_POS = (-0.45, 0.0, TABLE_TOP_Z)

CUBE_HALF_EXTENT = 0.025
CUBE_START_POS = (0.18, -0.14, TABLE_TOP_Z + CUBE_HALF_EXTENT + 0.003)
CUBE_START_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)



CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360


def require_path(path: str | Path, *, label: str = "path") -> Path:
    """Return an absolute path or raise a clear error if it is missing."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved

