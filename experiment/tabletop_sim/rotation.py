"""Small rotation helpers shared by tabletop rollout scripts."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""

    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def unwrap_robot_model_q_rotation(q_world: np.ndarray, *, reference: np.ndarray) -> np.ndarray:
    """Choose the nearest equivalent base Euler angles relative to ``reference``."""

    q = np.asarray(q_world, dtype=np.float32).copy()
    ref = np.asarray(reference, dtype=np.float32)
    if q.shape[-1] < 6 or ref.shape[-1] < 6:
        return q
    delta = q[..., 3:6].astype(np.float64) - ref[..., 3:6].astype(np.float64)
    q[..., 3:6] = (ref[..., 3:6].astype(np.float64) + wrap_to_pi(delta)).astype(np.float32)
    return q


class FixedBaseRotationChart:
    """Constant local XYZ coordinates for a floating hand's virtual hinges.

    Decoder configurations keep their original world convention. The simulator
    uses R_world = R_reference @ R_XYZ(native), avoiding a poorly conditioned
    world Euler chart for a calibrated workspace.
    """

    def __init__(self, embodiment: str, reference_euler_xyz):
        bodies = {"barrett": "virtual_link_roll", "allegro_right": "virtual_joint_roll_link"}
        if embodiment not in bodies:
            raise ValueError(f"Fixed base rotation reference is unsupported for {embodiment!r}")
        angles = np.asarray(reference_euler_xyz, dtype=np.float64)
        if angles.shape != (3,) or not np.isfinite(angles).all():
            raise ValueError("Base rotation reference must contain three finite XYZ Euler angles")
        self.body_name = bodies[embodiment]
        self.rotation = Rotation.from_euler("XYZ", angles)

    def apply_to_spec(self, spec) -> None:
        body = next((body for body in spec.bodies if body.name == self.body_name), None)
        if body is None:
            raise ValueError(f"Missing floating rotation body {self.body_name!r}")
        if not np.array_equal(body.pos, [0., 0., 0.]) or not np.array_equal(body.quat, [1., 0., 0., 0.]):
            raise ValueError("Fixed rotation chart requires the original zero-position, identity root frame")
        body.quat = self.rotation.as_quat()[[3, 0, 1, 2]].tolist()

    def to_native(self, q, reference=None) -> np.ndarray:
        result = np.asarray(q, dtype=np.float64).copy()
        angles = (self.rotation.inv() * Rotation.from_euler("XYZ", result[3:6])).as_euler("XYZ")
        if reference is not None:
            angles = np.asarray(reference) + wrap_to_pi(angles - reference)
        result[3:6] = angles
        return result

    def to_world(self, q, reference=None) -> np.ndarray:
        result = np.asarray(q, dtype=np.float64).copy()
        angles = (self.rotation * Rotation.from_euler("XYZ", result[3:6])).as_euler("XYZ")
        if reference is not None:
            angles = np.asarray(reference) + wrap_to_pi(angles - reference)
        result[3:6] = angles
        return result.astype(np.float32)


def robot_model_q_to_native(env, q_world, *, reset: bool = False) -> np.ndarray:
    """Convert at the simulator boundary; retain decoder/world feedback values."""
    chart = getattr(env, "floating_rotation_chart", None)
    if chart is None:
        return q_world
    reference = None if reset else getattr(env, "_native_rotation_reference", None)
    native = chart.to_native(q_world, reference)
    limits = env.robot.side("right").arm_q_range[3:6]
    if np.any(native[3:6] < limits[:, 0]) or np.any(native[3:6] > limits[:, 1]):
        raise ValueError("Local rotation target exceeds the configured virtual-joint limits")
    env._native_rotation_reference = native[3:6].copy()
    env._world_rotation_reference = np.asarray(q_world[3:6]).copy()
    return native.astype(np.float32)


def robot_model_q_to_world(env, q_native) -> np.ndarray:
    chart = getattr(env, "floating_rotation_chart", None)
    if chart is None:
        return q_native
    return chart.to_world(q_native, getattr(env, "_world_rotation_reference", None))
