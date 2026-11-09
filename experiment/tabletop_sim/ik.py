"""Small raw-MuJoCo damped-least-squares IK solver for the 7-DoF arms."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from .robot import ResolvedSide


@dataclass(frozen=True)
class IKSolverConfig:
    max_iters: int = 40
    damping: float = 0.08
    step_scale: float = 0.65
    max_step_norm: float = 0.12
    pos_tolerance: float = 0.004
    ori_tolerance: float = 0.04
    ori_weight: float = 0.25


def solve_arm_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: ResolvedSide,
    target_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
    *,
    config: IKSolverConfig | None = None,
) -> np.ndarray:
    """Return a clipped 7-joint arm target for a wrist site target pose."""

    cfg = config or IKSolverConfig()
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat_wxyz = np.asarray(target_quat_wxyz, dtype=np.float64).reshape(4)
    quat_norm = float(np.linalg.norm(target_quat_wxyz))
    if quat_norm <= 1e-8:
        raise ValueError("target_quat_wxyz must be nonzero")
    target_quat_wxyz = target_quat_wxyz / quat_norm
    target_rot = R.from_quat([target_quat_wxyz[1], target_quat_wxyz[2], target_quat_wxyz[3], target_quat_wxyz[0]])

    work = mujoco.MjData(model)
    work.qpos[:] = data.qpos
    work.qvel[:] = data.qvel
    work.ctrl[:] = data.ctrl
    q = np.asarray(work.qpos[side.arm_qpos_indices], dtype=np.float64).copy()

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    cols = side.arm_qvel_indices

    for _ in range(int(cfg.max_iters)):
        work.qpos[side.arm_qpos_indices] = q
        work.qvel[side.arm_qvel_indices] = 0.0
        mujoco.mj_forward(model, work)

        cur_pos = np.asarray(work.site_xpos[side.wrist_site_id], dtype=np.float64)
        cur_mat = np.asarray(work.site_xmat[side.wrist_site_id], dtype=np.float64).reshape(3, 3)
        cur_rot = R.from_matrix(cur_mat)
        pos_err = target_pos - cur_pos
        rot_err = (target_rot * cur_rot.inv()).as_rotvec()

        if float(np.linalg.norm(pos_err)) <= cfg.pos_tolerance and float(np.linalg.norm(rot_err)) <= cfg.ori_tolerance:
            break

        mujoco.mj_jacSite(model, work, jacp, jacr, side.wrist_site_id)
        jac = np.vstack([jacp[:, cols], cfg.ori_weight * jacr[:, cols]])
        err = np.concatenate([pos_err, cfg.ori_weight * rot_err])
        lhs = jac @ jac.T + (float(cfg.damping) ** 2) * np.eye(6)
        dq = jac.T @ np.linalg.solve(lhs, err)
        step_norm = float(np.linalg.norm(dq))
        if step_norm > cfg.max_step_norm:
            dq *= float(cfg.max_step_norm) / step_norm
        q = q + float(cfg.step_scale) * dq
        q = np.clip(q, side.arm_q_range[:, 0], side.arm_q_range[:, 1])

    return np.clip(q, side.arm_ctrl_range[:, 0], side.arm_ctrl_range[:, 1])
