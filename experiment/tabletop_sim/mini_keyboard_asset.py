"""Attach the shared articulated mini-keyboard asset to a tabletop scene."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mujoco


SOURCE_ROOT_BODY_NAME = "akai_mpk_mini_mk3"


def attach_mini_keyboard_asset(
    spec: mujoco.MjSpec,
    *,
    mjcf_path: str | Path,
    body_name: str,
    position: Sequence[float],
    quat_wxyz: Sequence[float],
    support_box_height: float,
    root_z_offset: float,
    friction: Sequence[float],
    key_joint_names: Sequence[str],
    white_key_push_distance: float,
    black_key_push_distance: float,
) -> mujoco.MjsBody:
    """Attach the canonical keyboard body and apply task-specific physics."""

    path = Path(mjcf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Mini-keyboard MJCF does not exist: {path}")

    friction_values = tuple(float(value) for value in friction)
    if len(friction_values) != 3 or any(value < 0.0 for value in friction_values):
        raise ValueError(
            "mini_keyboard.friction must contain three non-negative values, "
            f"got {friction_values}"
        )
    support_height = float(support_box_height)
    if support_height < 0.0:
        raise ValueError(
            f"mini_keyboard.support_box_height must be non-negative, got {support_height}"
        )
    white_travel = float(white_key_push_distance)
    black_travel = float(black_key_push_distance)
    if white_travel <= 0.0 or black_travel <= 0.0:
        raise ValueError(
            "mini-keyboard key travel must be positive, got "
            f"white={white_travel}, black={black_travel}"
        )

    keyboard_spec = mujoco.MjSpec.from_file(str(path))
    source_body = keyboard_spec.body(SOURCE_ROOT_BODY_NAME)
    if source_body is None:
        raise KeyError(f"Mini-keyboard MJCF {path} has no body {SOURCE_ROOT_BODY_NAME!r}")

    expected_joints = tuple(str(name) for name in key_joint_names)
    actual_joints = tuple(str(joint.name) for joint in keyboard_spec.joints)
    if actual_joints != expected_joints:
        raise ValueError(
            f"Mini-keyboard joint order in {path} does not match the tabletop task: "
            f"expected={expected_joints}, actual={actual_joints}"
        )

    _configure_keyboard_physics(
        keyboard_spec,
        friction=friction_values,
        key_joint_names=expected_joints,
        white_key_push_distance=white_travel,
        black_key_push_distance=black_travel,
    )

    frame = spec.worldbody.add_frame(
        name=f"{body_name}_asset_frame",
        pos=[float(value) for value in position],
        quat=[float(value) for value in quat_wxyz],
    )
    attached = frame.attach_body(source_body, "", "")
    attached.name = str(body_name)

    if support_height > 1.0e-9:
        support = attached.add_geom(
            name=f"{body_name}_support_box",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.162, 0.090, 0.5 * support_height],
            pos=[0.0, 0.0, -float(root_z_offset) - 0.5 * support_height],
            rgba=[0.42, 0.39, 0.34, 1.0],
        )
        support.contype = 0
        support.conaffinity = 0

    return attached


def _configure_keyboard_physics(
    keyboard_spec: mujoco.MjSpec,
    *,
    friction: tuple[float, float, float],
    key_joint_names: tuple[str, ...],
    white_key_push_distance: float,
    black_key_push_distance: float,
) -> None:
    for geom in keyboard_spec.geoms:
        if str(geom.name) == "table":
            continue
        geom.friction = list(friction)
        geom.condim = 6
        is_key = str(geom.name).startswith(("white_key_", "black_key_"))
        geom.solref = [0.002, 1.0]
        geom.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
        geom.priority = 2 if is_key else 1

    for joint_name in key_joint_names:
        joint = keyboard_spec.joint(joint_name)
        if joint is None:
            raise KeyError(f"Mini-keyboard MJCF is missing joint {joint_name!r}")
        is_white = joint_name.startswith("white_key_")
        joint.range = [
            -float(white_key_push_distance if is_white else black_key_push_distance),
            0.0,
        ]
        try:
            joint.solreflimit = [0.001, 1.0]
            joint.solimplimit = [0.995, 0.999, 0.0001]
        except AttributeError:
            # MuJoCo 3.8 keeps these values on the source defaults but does not
            # expose them through MjsJoint.
            pass

        site_name = f"{joint_name}_site"
        if keyboard_spec.site(site_name) is not None:
            continue
        key_body = keyboard_spec.body(joint_name.removesuffix("_slide"))
        key_geom = keyboard_spec.geom(joint_name.removesuffix("_slide") + "_geom")
        if key_body is None or key_geom is None:
            raise KeyError(f"Mini-keyboard MJCF is missing the body or geom for {joint_name!r}")
        key_body.add_site(
            name=site_name,
            pos=[0.0, 0.0, float(key_geom.size[2]) + 0.001],
            size=[0.0025],
            rgba=[0.1, 0.4, 1.0, 0.0],
        )
