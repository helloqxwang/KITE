"""Embodiment registry and MuJoCo name resolution for tabletop robots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from .assets import (
    ALLEGRO_RIGHT_XML,
    ARM_HOME_Q,
    BARRETT_MJCF,
    FRANKA_ROBOTIQ_BASE_POS,
    FRANKA_ROBOTIQ_XML,
    MARVIN_WUJI_XML,
    ROBOT_BASE_POS,
    WUJIHAND_RIGHT_XML,
)




def arm_joint_names(side: str) -> tuple[str, ...]:
    suffix = "L" if side == "left" else "R"
    return tuple(f"Joint{i}_{suffix}" for i in range(1, 8))


def hand_joint_names(side: str) -> tuple[str, ...]:
    return tuple(f"{side}_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5))


def arm_actuator_names(side: str) -> tuple[str, ...]:
    return tuple(f"{side}_arm_joint{i}_actuator" for i in range(1, 8))


def hand_actuator_names(side: str) -> tuple[str, ...]:
    return tuple(f"{name}_actuator" for name in hand_joint_names(side))


@dataclass(frozen=True)
class SiteAttachmentSpec:
    body_name: str
    site_name: str
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    size: tuple[float, ...] = (0.008,)
    rgba: tuple[float, float, float, float] = (0.95, 0.15, 0.15, 0.75)


@dataclass(frozen=True)
class ControlGroupSpec:
    name: str
    arm_joint_names: tuple[str, ...]
    arm_actuator_names: tuple[str, ...]
    ee_site_name: str
    arm_home_q: tuple[float, ...]
    effector_joint_names: tuple[str, ...] = ()
    effector_actuator_names: tuple[str, ...] = ()
    effector_home_q: tuple[float, ...] = ()
    effector_home_ctrl: tuple[float, ...] = ()
    effector_action: str = "joint_position"
    contact_geom_names: tuple[str, ...] = ()
    ee_closing_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    ee_approach_axis: tuple[float, float, float] = (0.0, 0.0, -1.0)


@dataclass(frozen=True)
class EmbodimentSpec:
    key: str
    robot_xml_path: Path
    groups: tuple[ControlGroupSpec, ...]
    root_body_name: str = "base"
    root_body_pos: tuple[float, float, float] = ROBOT_BASE_POS
    root_body_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    site_attachments: tuple[SiteAttachmentSpec, ...] = ()

    @property
    def group_names(self) -> tuple[str, ...]:
        return tuple(group.name for group in self.groups)

    @property
    def sides(self) -> tuple[str, ...]:
        return self.group_names


@dataclass(frozen=True)
class ResolvedSide:
    side: str
    group_spec: ControlGroupSpec
    arm_joint_names: tuple[str, ...]
    hand_joint_names: tuple[str, ...]
    arm_actuator_names: tuple[str, ...]
    hand_actuator_names: tuple[str, ...]
    wrist_site_name: str
    arm_joint_ids: np.ndarray
    hand_joint_ids: np.ndarray
    arm_qpos_indices: np.ndarray
    hand_qpos_indices: np.ndarray
    arm_qvel_indices: np.ndarray
    hand_qvel_indices: np.ndarray
    arm_actuator_ids: np.ndarray
    hand_actuator_ids: np.ndarray
    arm_ctrl_range: np.ndarray
    hand_ctrl_range: np.ndarray
    arm_q_range: np.ndarray
    hand_q_range: np.ndarray
    arm_home_q: np.ndarray
    hand_home_q: np.ndarray
    hand_home_ctrl: np.ndarray
    wrist_site_id: int
    contact_geom_ids: np.ndarray
    ee_closing_axis: np.ndarray
    ee_approach_axis: np.ndarray

    @property
    def qpos_indices(self) -> np.ndarray:
        return np.concatenate([self.arm_qpos_indices, self.hand_qpos_indices])

    @property
    def qvel_indices(self) -> np.ndarray:
        return np.concatenate([self.arm_qvel_indices, self.hand_qvel_indices])

    @property
    def actuator_ids(self) -> np.ndarray:
        return np.concatenate([self.arm_actuator_ids, self.hand_actuator_ids])

    @property
    def home_q(self) -> np.ndarray:
        return np.concatenate([self.arm_home_q, self.hand_home_q])


    @property
    def arm_dim(self) -> int:
        return int(self.arm_qpos_indices.shape[0])

    @property
    def effector_qpos_dim(self) -> int:
        return int(self.hand_qpos_indices.shape[0])

    @property
    def effector_ctrl_dim(self) -> int:
        return int(self.hand_actuator_ids.shape[0])

    @property
    def effector_action(self) -> str:
        return self.group_spec.effector_action


@dataclass(frozen=True)
class ResolvedRobot:
    spec: EmbodimentSpec
    groups: Mapping[str, ResolvedSide]

    @property
    def group_names(self) -> tuple[str, ...]:
        return tuple(self.groups.keys())

    @property
    def left(self) -> ResolvedSide:
        return self.side("left")

    @property
    def right(self) -> ResolvedSide:
        return self.side("right")

    def has_group(self, side: str) -> bool:
        return side in self.groups

    def side(self, side: str) -> ResolvedSide:
        try:
            return self.groups[side]
        except KeyError as exc:
            raise ValueError(f"Unknown control group {side!r}; available: {self.group_names}") from exc

    def apply_home(
        self,
        data: mujoco.MjData,
        *,
        sides: Iterable[str] | None = None,
        set_qpos: bool = True,
    ) -> None:
        for side_name in self.group_names if sides is None else tuple(sides):
            if not self.has_group(side_name):
                continue
            self.apply_side_q(data, side_name, self.side(side_name).home_q, set_qpos=set_qpos)

    def apply_side_q(self, data: mujoco.MjData, side: str, q: np.ndarray, *, set_qpos: bool = True) -> None:
        side_spec = self.side(side)
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape != side_spec.home_q.shape:
            raise ValueError(f"{side} q must have shape {side_spec.home_q.shape}, got {q.shape}")

        arm_q = np.clip(q[: side_spec.arm_dim], side_spec.arm_q_range[:, 0], side_spec.arm_q_range[:, 1])
        effector_q = np.clip(
            q[side_spec.arm_dim :],
            side_spec.hand_q_range[:, 0],
            side_spec.hand_q_range[:, 1],
        )
        if set_qpos:
            data.qpos[side_spec.arm_qpos_indices] = arm_q
            data.qvel[side_spec.arm_qvel_indices] = 0.0
            if side_spec.effector_qpos_dim:
                data.qpos[side_spec.hand_qpos_indices] = effector_q
                data.qvel[side_spec.hand_qvel_indices] = 0.0
        self.set_side_ctrl(data, side, arm_q=arm_q, effector_ctrl=side_spec.hand_home_ctrl)

    def set_side_ctrl(
        self,
        data: mujoco.MjData,
        side: str,
        *,
        arm_q: np.ndarray | None = None,
        hand_q: np.ndarray | None = None,
        gripper: np.ndarray | float | None = None,
        effector_ctrl: np.ndarray | None = None,
    ) -> None:
        side_spec = self.side(side)
        if arm_q is not None:
            arm_q = np.asarray(arm_q, dtype=np.float64).reshape(-1)
            if arm_q.shape != (side_spec.arm_dim,):
                raise ValueError(f"{side} arm_q must have shape {(side_spec.arm_dim,)}, got {arm_q.shape}")
            data.ctrl[side_spec.arm_actuator_ids] = np.clip(
                arm_q, side_spec.arm_ctrl_range[:, 0], side_spec.arm_ctrl_range[:, 1]
            )

        ctrl_value = effector_ctrl
        if ctrl_value is None and hand_q is not None:
            ctrl_value = hand_q
        if ctrl_value is None and gripper is not None:
            ctrl_value = np.asarray(gripper, dtype=np.float64).reshape(-1)
        if ctrl_value is None or not side_spec.effector_ctrl_dim:
            return

        ctrl = np.asarray(ctrl_value, dtype=np.float64).reshape(-1)
        if ctrl.shape != (side_spec.effector_ctrl_dim,):
            raise ValueError(
                f"{side} effector control must have shape {(side_spec.effector_ctrl_dim,)}, got {ctrl.shape}"
            )
        data.ctrl[side_spec.hand_actuator_ids] = np.clip(
            ctrl,
            side_spec.hand_ctrl_range[:, 0],
            side_spec.hand_ctrl_range[:, 1],
        )

    def wrist_pose(self, data: mujoco.MjData, side: str) -> tuple[np.ndarray, np.ndarray]:
        side_spec = self.side(side)
        pos = np.asarray(data.site_xpos[side_spec.wrist_site_id], dtype=np.float64).copy()
        mat = np.asarray(data.site_xmat[side_spec.wrist_site_id], dtype=np.float64).reshape(3, 3)
        quat_xyzw = R.from_matrix(mat).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
        return pos, quat_wxyz

    def proprio(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {}
        for side_name in self.group_names:
            side = self.side(side_name)
            obs[f"{side_name}_arm_q"] = data.qpos[side.arm_qpos_indices].astype(np.float32).copy()
            if side.effector_qpos_dim:
                effector_q = data.qpos[side.hand_qpos_indices].astype(np.float32).copy()
                obs[f"{side_name}_hand_q"] = effector_q
                obs[f"{side_name}_effector_q"] = effector_q
                if side.effector_action == "gripper":
                    obs[f"{side_name}_gripper_q"] = effector_q
            if side.effector_ctrl_dim:
                effector_ctrl = data.ctrl[side.hand_actuator_ids].astype(np.float32).copy()
                obs[f"{side_name}_effector_ctrl"] = effector_ctrl
                if side.effector_action == "gripper":
                    obs[f"{side_name}_gripper_ctrl"] = effector_ctrl
            pos, quat = self.wrist_pose(data, side_name)
            obs[f"{side_name}_wrist_pos"] = pos.astype(np.float32)
            obs[f"{side_name}_wrist_quat_wxyz"] = quat.astype(np.float32)
        return obs


def _wuji_group(side: str) -> ControlGroupSpec:
    hand_names = hand_joint_names(side)
    return ControlGroupSpec(
        name=side,
        arm_joint_names=arm_joint_names(side),
        arm_actuator_names=arm_actuator_names(side),
        ee_site_name=f"{side}_wrist",
        arm_home_q=tuple(float(x) for x in ARM_HOME_Q[side]),
        effector_joint_names=hand_names,
        effector_actuator_names=hand_actuator_names(side),
        effector_home_q=tuple(0.0 for _ in hand_names),
        effector_home_ctrl=tuple(0.0 for _ in hand_names),
        effector_action="joint_position",
    )


FRANKA_ARM_JOINT_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))
FRANKA_GRIPPER_JOINT_NAMES = (
    "g_left_driver_joint",
    "g_left_spring_link_joint",
    "g_left_follower_joint",
    "g_right_driver_joint",
    "g_right_spring_link_joint",
    "g_right_follower_joint",
)
FRANKA_ROBOTIQ_PAD_GEOMS = ("g_left_pad1", "g_left_pad2", "g_right_pad1", "g_right_pad2")
ALLEGRO_ROOT_JOINT_NAMES = (
    "virtual_joint_x",
    "virtual_joint_y",
    "virtual_joint_z",
    "virtual_joint_roll",
    "virtual_joint_pitch",
    "virtual_joint_yaw",
)
ALLEGRO_HAND_JOINT_NAMES = tuple(f"joint_{idx}.0" for idx in range(16))
BARRETT_ROOT_JOINT_NAMES = (
    "virtual_joint_x",
    "virtual_joint_y",
    "virtual_joint_z",
    "virtual_joint_roll",
    "virtual_joint_pitch",
    "virtual_joint_yaw",
)
BARRETT_HAND_JOINT_NAMES = (
    "bh_j32_joint",
    "bh_j33_joint",
    "bh_j11_joint",
    "bh_j12_joint",
    "bh_j13_joint",
    "bh_j21_joint",
    "bh_j22_joint",
    "bh_j23_joint",
)
WUJIHAND_ROOT_JOINT_NAMES = (
    "right_pos_x",
    "right_pos_y",
    "right_pos_z",
    "right_rot_x",
    "right_rot_y",
    "right_rot_z",
)
WUJIHAND_ROOT_ACTUATOR_NAMES = tuple(f"{name}_position" for name in WUJIHAND_ROOT_JOINT_NAMES)
WUJIHAND_HAND_JOINT_NAMES = hand_joint_names("right")
WUJIHAND_HAND_ACTUATOR_NAMES = tuple(f"{name}_position" for name in WUJIHAND_HAND_JOINT_NAMES)
WUJIHAND_CONTACT_GEOM_NAMES = (
    "collision_hand_right_palm_0",
    *tuple(
        f"collision_hand_right_{finger}_{link}"
        for finger in ("thumb", "index", "middle", "ring", "pinky")
        for link in (3, 2, 1, 0)
    ),
)


EMBODIMENTS = {
    "marvin_wuji": EmbodimentSpec(
        key="marvin_wuji",
        robot_xml_path=MARVIN_WUJI_XML,
        groups=(_wuji_group("left"), _wuji_group("right")),
        root_body_pos=ROBOT_BASE_POS,
    ),
    "franka_robotiq": EmbodimentSpec(
        key="franka_robotiq",
        robot_xml_path=FRANKA_ROBOTIQ_XML,
        groups=(
            ControlGroupSpec(
                name="right",
                arm_joint_names=FRANKA_ARM_JOINT_NAMES,
                arm_actuator_names=FRANKA_ARM_JOINT_NAMES,
                ee_site_name="right_wrist",
                arm_home_q=(0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -1.5707963267948966),
                effector_joint_names=FRANKA_GRIPPER_JOINT_NAMES,
                effector_actuator_names=("g_fingers_actuator",),
                effector_home_q=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                effector_home_ctrl=(0.0,),
                effector_action="gripper",
                contact_geom_names=FRANKA_ROBOTIQ_PAD_GEOMS,
                ee_closing_axis=(1.0, 0.0, 0.0),
                ee_approach_axis=(0.0, 0.0, 1.0),
            ),
        ),
        root_body_pos=FRANKA_ROBOTIQ_BASE_POS,
        site_attachments=(
            SiteAttachmentSpec(
                body_name="g_base",
                site_name="right_wrist",
                pos=(0.0, 0.0, 0.152),
                quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                size=(0.0001,),
                rgba=(1.0, 1.0, 1.0, 0.0),
            ),
        ),
    ),
    "allegro_right": EmbodimentSpec(
        key="allegro_right",
        robot_xml_path=ALLEGRO_RIGHT_XML,
        groups=(
            ControlGroupSpec(
                name="right",
                arm_joint_names=ALLEGRO_ROOT_JOINT_NAMES,
                arm_actuator_names=ALLEGRO_ROOT_JOINT_NAMES,
                ee_site_name="right_palm",
                arm_home_q=(0.0,) * len(ALLEGRO_ROOT_JOINT_NAMES),
                effector_joint_names=ALLEGRO_HAND_JOINT_NAMES,
                effector_actuator_names=ALLEGRO_HAND_JOINT_NAMES,
                effector_home_q=(0.0,) * len(ALLEGRO_HAND_JOINT_NAMES),
                effector_home_ctrl=(0.0,) * len(ALLEGRO_HAND_JOINT_NAMES),
                effector_action="joint_position",
            ),
        ),
        root_body_name="virtual_base_link",
        root_body_pos=(0.0, 0.0, 0.0),
    ),
    "barrett": EmbodimentSpec(
        key="barrett",
        robot_xml_path=BARRETT_MJCF,
        groups=(
            ControlGroupSpec(
                name="right",
                arm_joint_names=BARRETT_ROOT_JOINT_NAMES,
                arm_actuator_names=BARRETT_ROOT_JOINT_NAMES,
                ee_site_name="right_wrist",
                arm_home_q=(0.0,) * len(BARRETT_ROOT_JOINT_NAMES),
                effector_joint_names=BARRETT_HAND_JOINT_NAMES,
                effector_actuator_names=BARRETT_HAND_JOINT_NAMES,
                effector_home_q=(0.0,) * len(BARRETT_HAND_JOINT_NAMES),
                effector_home_ctrl=(0.0,) * len(BARRETT_HAND_JOINT_NAMES),
                effector_action="joint_position",
            ),
        ),
        root_body_name="bh_base_link",
        root_body_pos=(0.0, 0.0, 0.0),
    ),
    "wujihand_right": EmbodimentSpec(
        key="wujihand_right",
        robot_xml_path=WUJIHAND_RIGHT_XML,
        groups=(
            ControlGroupSpec(
                name="right",
                arm_joint_names=WUJIHAND_ROOT_JOINT_NAMES,
                arm_actuator_names=WUJIHAND_ROOT_ACTUATOR_NAMES,
                ee_site_name="right_palm",
                arm_home_q=(0.0,) * len(WUJIHAND_ROOT_JOINT_NAMES),
                effector_joint_names=WUJIHAND_HAND_JOINT_NAMES,
                effector_actuator_names=WUJIHAND_HAND_ACTUATOR_NAMES,
                effector_home_q=(0.0,) * len(WUJIHAND_HAND_JOINT_NAMES),
                effector_home_ctrl=(0.0,) * len(WUJIHAND_HAND_JOINT_NAMES),
                effector_action="joint_position",
                contact_geom_names=WUJIHAND_CONTACT_GEOM_NAMES,
            ),
        ),
        root_body_name="right_palm",
        root_body_pos=(0.0, 0.0, 0.0),
    ),
}


def get_embodiment_spec(key: str = "marvin_wuji") -> EmbodimentSpec:
    try:
        return EMBODIMENTS[str(key)]
    except KeyError as exc:
        raise KeyError(f"Unknown embodiment {key!r}. Available: {sorted(EMBODIMENTS)}") from exc


def resolve_robot(model: mujoco.MjModel, key: str = "marvin_wuji") -> ResolvedRobot:
    spec = get_embodiment_spec(key)
    groups = {group.name: _resolve_group(model, group) for group in spec.groups}
    return ResolvedRobot(spec=spec, groups=groups)


def _resolve_group(model: mujoco.MjModel, group: ControlGroupSpec) -> ResolvedSide:
    arm_jids = _ids(model, mujoco.mjtObj.mjOBJ_JOINT, group.arm_joint_names)
    hand_jids = _ids(model, mujoco.mjtObj.mjOBJ_JOINT, group.effector_joint_names)
    arm_aids = _ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR, group.arm_actuator_names)
    hand_aids = _ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR, group.effector_actuator_names)
    wrist_sid = _required_id(model, mujoco.mjtObj.mjOBJ_SITE, group.ee_site_name)
    contact_geom_ids = _ids(model, mujoco.mjtObj.mjOBJ_GEOM, group.contact_geom_names)

    arm_qpos = np.asarray(model.jnt_qposadr[arm_jids], dtype=np.int32)
    hand_qpos = np.asarray(model.jnt_qposadr[hand_jids], dtype=np.int32)
    arm_qvel = np.asarray(model.jnt_dofadr[arm_jids], dtype=np.int32)
    hand_qvel = np.asarray(model.jnt_dofadr[hand_jids], dtype=np.int32)

    arm_q_range = _joint_ranges(model, arm_jids)
    hand_q_range = _joint_ranges(model, hand_jids)
    arm_ctrl_range = _ctrl_ranges(model, arm_aids, fallback=arm_q_range)
    hand_ctrl_range = _ctrl_ranges(model, hand_aids)

    arm_home = _home_array(group.arm_home_q, len(group.arm_joint_names), label=f"{group.name} arm_home_q")
    arm_home = np.clip(arm_home, arm_ctrl_range[:, 0], arm_ctrl_range[:, 1])
    hand_home = _home_array(group.effector_home_q, len(group.effector_joint_names), label=f"{group.name} effector_home_q")
    if hand_home.size:
        hand_home = np.clip(hand_home, hand_q_range[:, 0], hand_q_range[:, 1])
    hand_home_ctrl = _home_array(
        group.effector_home_ctrl,
        len(group.effector_actuator_names),
        label=f"{group.name} effector_home_ctrl",
    )
    if hand_home_ctrl.size:
        hand_home_ctrl = np.clip(hand_home_ctrl, hand_ctrl_range[:, 0], hand_ctrl_range[:, 1])
    ee_closing_axis = _unit_axis(group.ee_closing_axis, label=f"{group.name} ee_closing_axis")
    ee_approach_axis = _orthogonal_unit_axis(
        group.ee_approach_axis,
        reference=ee_closing_axis,
        label=f"{group.name} ee_approach_axis",
    )

    return ResolvedSide(
        side=group.name,
        group_spec=group,
        arm_joint_names=group.arm_joint_names,
        hand_joint_names=group.effector_joint_names,
        arm_actuator_names=group.arm_actuator_names,
        hand_actuator_names=group.effector_actuator_names,
        wrist_site_name=group.ee_site_name,
        arm_joint_ids=arm_jids,
        hand_joint_ids=hand_jids,
        arm_qpos_indices=arm_qpos,
        hand_qpos_indices=hand_qpos,
        arm_qvel_indices=arm_qvel,
        hand_qvel_indices=hand_qvel,
        arm_actuator_ids=arm_aids,
        hand_actuator_ids=hand_aids,
        arm_ctrl_range=arm_ctrl_range,
        hand_ctrl_range=hand_ctrl_range,
        arm_q_range=arm_q_range,
        hand_q_range=hand_q_range,
        arm_home_q=arm_home,
        hand_home_q=hand_home,
        hand_home_ctrl=hand_home_ctrl,
        wrist_site_id=wrist_sid,
        contact_geom_ids=contact_geom_ids,
        ee_closing_axis=ee_closing_axis,
        ee_approach_axis=ee_approach_axis,
    )


def _required_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = int(mujoco.mj_name2id(model, obj_type, name))
    if obj_id < 0:
        raise KeyError(f"Required MuJoCo object is missing: {name}")
    return obj_id


def _ids(model: mujoco.MjModel, obj_type: mujoco.mjtObj, names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([_required_id(model, obj_type, name) for name in names], dtype=np.int32)


def _home_array(values: tuple[float, ...], expected: int, *, label: str) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).reshape(-1)
    if out.shape != (expected,):
        raise ValueError(f"{label} must have shape {(expected,)}, got {out.shape}")
    return out


def _unit_axis(values: tuple[float, float, float], *, label: str) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-8:
        raise ValueError(f"{label} must be nonzero")
    return axis / norm


def _orthogonal_unit_axis(
    values: tuple[float, float, float],
    *,
    reference: np.ndarray,
    label: str,
) -> np.ndarray:
    axis = _unit_axis(values, label=label)
    dot = float(np.dot(axis, np.asarray(reference, dtype=np.float64).reshape(3)))
    if abs(dot) > 1e-6:
        raise ValueError(f"{label} must be orthogonal to the closing axis; dot={dot:.6g}")
    return axis


def _joint_ranges(model: mujoco.MjModel, joint_ids: np.ndarray) -> np.ndarray:
    if joint_ids.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    ranges = np.asarray(model.jnt_range[joint_ids], dtype=np.float64).copy()
    limited = np.asarray(model.jnt_limited[joint_ids], dtype=bool)
    ranges[~limited, 0] = -np.inf
    ranges[~limited, 1] = np.inf
    ranges[:, 0] = np.where(np.isfinite(ranges[:, 0]), ranges[:, 0], -10.0)
    ranges[:, 1] = np.where(np.isfinite(ranges[:, 1]), ranges[:, 1], 10.0)
    return ranges


def _ctrl_ranges(
    model: mujoco.MjModel,
    actuator_ids: np.ndarray,
    *,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    if actuator_ids.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    ranges = np.asarray(model.actuator_ctrlrange[actuator_ids], dtype=np.float64).copy()
    limited = np.asarray(model.actuator_ctrllimited[actuator_ids], dtype=bool)
    if fallback is None or fallback.shape != ranges.shape:
        fallback = np.tile(np.asarray([-10.0, 10.0], dtype=np.float64), (ranges.shape[0], 1))
    ranges[~limited] = fallback[~limited]
    ranges[:, 0] = np.where(np.isfinite(ranges[:, 0]), ranges[:, 0], fallback[:, 0])
    ranges[:, 1] = np.where(np.isfinite(ranges[:, 1]), ranges[:, 1], fallback[:, 1])
    return ranges
