"""Backend adapters for tabletop simulation."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class RawMujocoBackend:
    """Thin CPU MuJoCo backend used as the default simulator."""

    model: mujoco.MjModel
    data: mujoco.MjData
    frame_skip: int = 10
    backend_name: str = "raw"
    gravity_compensation_dof_indices: np.ndarray | None = None

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def step(self, n_steps: int | None = None) -> None:
        for _ in range(int(self.frame_skip if n_steps is None else n_steps)):
            self._apply_gravity_compensation()
            mujoco.mj_step(self.model, self.data)
        self.data.qfrc_applied[:] = 0.0

    def close(self) -> None:
        return None

    def set_gravity_compensation_dofs(self, dof_indices: np.ndarray | None) -> None:
        if dof_indices is None:
            self.gravity_compensation_dof_indices = None
            self.data.qfrc_applied[:] = 0.0
            return
        dofs = np.asarray(dof_indices, dtype=np.int32).reshape(-1)
        if np.any(dofs < 0) or np.any(dofs >= int(self.model.nv)):
            raise ValueError(f"Gravity compensation DoF indices must be in [0, {self.model.nv})")
        self.gravity_compensation_dof_indices = np.unique(dofs)

    def _apply_gravity_compensation(self) -> None:
        self.data.qfrc_applied[:] = 0.0
        dofs = self.gravity_compensation_dof_indices
        if dofs is None or int(dofs.size) == 0:
            return
        mujoco.mj_forward(self.model, self.data)
        self.data.qfrc_applied[dofs] = self.data.qfrc_bias[dofs]


class MjlabBackend(RawMujocoBackend):
    """Optional mjlab adapter.

    The scene still compiles to an ``MjModel`` so the public environment API
    remains identical. Importing this backend validates that the local mjlab stack
    is installed before the wrapper is used.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, frame_skip: int = 10):
        try:
            import mjlab  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The mjlab backend was requested, but the `mjlab` package is not importable. "
                "Install the pinned mjlab package in the kite environment."
            ) from exc
        super().__init__(model=model, data=data, frame_skip=frame_skip, backend_name="mjlab")


def make_backend(
    backend: str,
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame_skip: int = 10,
) -> RawMujocoBackend:
    key = str(backend).strip().lower()
    if key == "raw":
        return RawMujocoBackend(model=model, data=data, frame_skip=frame_skip)
    if key == "mjlab":
        return MjlabBackend(model=model, data=data, frame_skip=frame_skip)
    raise ValueError("backend must be one of: raw, mjlab")
