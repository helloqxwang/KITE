from __future__ import annotations

from typing import Any, Sequence, Tuple

import torch


# Shared utilities for a residual action representation that swaps the 3 Euler base-rotation
# slots for 6 continuous rot6d slots, so residual_dim = action_dim + 3. xyz and joint residuals
# are plain q2 - q1 differences; the rotation residual is R_residual such that R(q2) = R_residual @ R(q1).


def _normalize(v: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return v / torch.norm(v, dim=-1, keepdim=True).clamp_min(eps)


def euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    # Intrinsic XYZ Euler -> rotation matrix, fully differentiable and batched.
    # Matches scipy's "XYZ" convention used elsewhere in this project.
    shape = euler.shape[:-1]
    e = euler.reshape(-1, 3)
    cx, cy, cz = torch.cos(e[:, 0]), torch.cos(e[:, 1]), torch.cos(e[:, 2])
    sx, sy, sz = torch.sin(e[:, 0]), torch.sin(e[:, 1]), torch.sin(e[:, 2])

    rx = torch.stack([
        torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], dim=-1),
        torch.stack([torch.zeros_like(cx), cx, -sx], dim=-1),
        torch.stack([torch.zeros_like(cx), sx, cx], dim=-1),
    ], dim=-2)
    ry = torch.stack([
        torch.stack([cy, torch.zeros_like(cy), sy], dim=-1),
        torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], dim=-1),
        torch.stack([-sy, torch.zeros_like(cy), cy], dim=-1),
    ], dim=-2)
    rz = torch.stack([
        torch.stack([cz, -sz, torch.zeros_like(cz)], dim=-1),
        torch.stack([sz, cz, torch.zeros_like(cz)], dim=-1),
        torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)], dim=-1),
    ], dim=-2)
    rot = rx @ ry @ rz
    return rot.reshape(*shape, 3, 3)


def matrix_to_euler_xyz(matrix: torch.Tensor) -> torch.Tensor:
    # Inverse of euler_xyz_to_matrix. Backward-stable version of the standard XYZ
    # intrinsic formulas — the naive implementation NaNs backward in two places on
    # this training path:
    #
    #   1) asin(sy) diverges at |sy|=1: gradient is 1/sqrt(1-sy^2). After a matmul
    #      of two orthonormal matrices, float rounding can nudge m[0,2] slightly
    #      above 1; clamp(-1, 1) pins forward to exactly 1 in fp32, making
    #      sqrt(1 - 1·1) = 0 exactly and asin's derivative Inf. We prevent this by
    #      clamping sy to [-1 + eps, 1 - eps] so cy >= sqrt(2·eps) ~ 1.4e-3 and
    #      asin's derivative is bounded by ~700.
    #
    #   2) torch.where(cond, f, g) computes gradients through BOTH branches and
    #      selects per-element at backward time. Near gimbal lock the "unused"
    #      atan2(-m[1,2], m[2,2]) hits atan2(0, 0) whose backward is NaN, and that
    #      NaN contaminates the where output regardless of which branch forward
    #      selected. We replace the unused-branch atan2 arguments with constant
    #      (1, 1) via an inner torch.where — both branches then have finite
    #      backwards, and the outer torch.where selects forward as before.
    shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)
    margin = 1.0e-6
    # `singular` must be evaluated on the RAW sy (before the margin clamp) — the clamp
    # would otherwise pin cy above the singularity threshold and make `singular` always
    # False, pushing all samples into the `normal` branch. At true gimbal lock the
    # normal branch's atan2 args (-m[1,2], m[2,2]) are both zero, and atan2(0, 0)
    # backward is NaN even though forward is 0.
    singular = m[:, 0, 2].abs() >= (1.0 - margin)
    sy = m[:, 0, 2].clamp(-1.0 + margin, 1.0 - margin)
    cy = torch.sqrt(1.0 - sy * sy)
    y = torch.asin(sy)
    ones = torch.ones_like(m[:, 1, 2])
    x_normal = torch.atan2(
        torch.where(singular, ones, -m[:, 1, 2]),
        torch.where(singular, ones,  m[:, 2, 2]),
    )
    z_normal = torch.atan2(
        torch.where(singular, ones, -m[:, 0, 1]),
        torch.where(singular, ones,  m[:, 0, 0]),
    )
    x_singular = torch.atan2(
        torch.where(singular, m[:, 2, 1], ones),
        torch.where(singular, m[:, 1, 1], ones),
    )
    z_singular = torch.zeros_like(z_normal)
    x = torch.where(singular, x_singular, x_normal)
    z = torch.where(singular, z_singular, z_normal)
    euler = torch.stack([x, y, z], dim=-1)
    return euler.reshape(*shape, 3)


def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    # Convention used by _matrix_to_rot6d/_rot6d_to_matrix in robot_model: rot6d stores the
    # first two columns of R stacked, so inverting _rot6d_to_matrix requires grabbing
    # matrix[..., :, 0] and matrix[..., :, 1]. (_matrix_to_rot6d in robot_model does
    # matrix.T.reshape(9)[:6] which is exactly [col0 || col1] in row-major form.)
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    # Matches networks-wide _rot6d_to_matrix: orthogonalize the pair of 3-vectors, then
    # build R whose rows are [x; y; z] (equivalently, columns [x^T, y^T, z^T] in mT form).
    x = _normalize(rot6d[..., 0:3])
    y = _normalize(rot6d[..., 3:6])
    a = _normalize(x + y)
    b = _normalize(x - y)
    x = _normalize(a + b)
    y = _normalize(a - b)
    z = _normalize(torch.cross(x, y, dim=-1))
    return torch.stack([x, y, z], dim=-2).mT


def _joint_indices(
    *,
    dof: int,
    pose_groups: Sequence[dict],
) -> Tuple[int, ...]:
    pose_set: set[int] = set()
    for group in pose_groups:
        pose_set.update(int(i) for i in group["translation_indices"])
        pose_set.update(int(i) for i in group["rotation_indices"])
    return tuple(i for i in range(int(dof)) if i not in pose_set)


def _normalize_pose_groups(
    *,
    dof: int,
    base_translation_indices: Sequence[int],
    base_rotation_indices: Sequence[int],
    pose_groups: Sequence[Any] | None,
) -> Tuple[dict, ...]:
    if pose_groups is None:
        pose_groups = (
            {
                "name": "base",
                "translation_indices": tuple(int(i) for i in base_translation_indices),
                "rotation_indices": tuple(int(i) for i in base_rotation_indices),
            },
        )

    normalized: list[dict] = []
    used: set[int] = set()
    for group_idx, raw_group in enumerate(pose_groups):
        if isinstance(raw_group, dict):
            name = str(raw_group.get("name", f"pose{group_idx}"))
            trans_raw = raw_group.get("translation_indices", raw_group.get("translation", ()))
            rot_raw = raw_group.get("rotation_indices", raw_group.get("rotation", ()))
        elif isinstance(raw_group, (list, tuple)) and len(raw_group) >= 2:
            name = f"pose{group_idx}"
            trans_raw = raw_group[0]
            rot_raw = raw_group[1]
        else:
            raise ValueError(f"Invalid pose group at index {group_idx}: {raw_group!r}")

        trans_idx = tuple(int(i) for i in trans_raw)
        rot_idx = tuple(int(i) for i in rot_raw)
        if len(trans_idx) != 3 or len(rot_idx) != 3:
            raise ValueError(
                f"pose group {name!r} must have exactly 3 translation and 3 rotation indices; "
                f"got translation={trans_idx}, rotation={rot_idx}"
            )
        for idx in trans_idx + rot_idx:
            if idx < 0 or idx >= int(dof):
                raise ValueError(f"pose group {name!r} index {idx} is outside dof={dof}")
            if idx in used:
                raise ValueError(f"pose group {name!r} reuses action index {idx}")
            used.add(idx)
        normalized.append(
            {
                "name": name,
                "translation_indices": trans_idx,
                "rotation_indices": rot_idx,
            }
        )

    if len(normalized) == 0:
        raise ValueError("At least one pose group is required")
    return tuple(normalized)


def build_residual_slot_indices(
    *,
    dof: int,
    base_translation_indices: Sequence[int],
    base_rotation_indices: Sequence[int],
    pose_groups: Sequence[Any] | None = None,
) -> dict:
    # Named index map that downstream code uses to slice residual vectors, normalizer buffers,
    # and mask conversions without re-deriving the slot layout everywhere.
    normalized_pose_groups = _normalize_pose_groups(
        dof=dof,
        base_translation_indices=base_translation_indices,
        base_rotation_indices=base_rotation_indices,
        pose_groups=pose_groups,
    )
    joint_idx = _joint_indices(
        dof=dof,
        pose_groups=normalized_pose_groups,
    )
    pose_slices = []
    cursor = 0
    for group in normalized_pose_groups:
        pose_slices.append(
            {
                "name": group["name"],
                "xyz_slice": (cursor, cursor + 3),
                "rot6d_slice": (cursor + 3, cursor + 9),
            }
        )
        cursor += 9
    residual_dim = cursor + len(joint_idx)
    first_group = normalized_pose_groups[0]
    first_slices = pose_slices[0]
    return {
        "dof": int(dof),
        "residual_dim": int(residual_dim),
        "pose_groups": normalized_pose_groups,
        "pose_residual_slices": tuple(pose_slices),
        "base_translation_indices": tuple(int(i) for i in first_group["translation_indices"]),
        "base_rotation_indices": tuple(int(i) for i in first_group["rotation_indices"]),
        "joint_indices": joint_idx,
        "xyz_slice": tuple(first_slices["xyz_slice"]),
        "rot6d_slice": tuple(first_slices["rot6d_slice"]),
        "joints_slice": (cursor, int(residual_dim)),
    }


def absolute_to_residual(
    q_abs: torch.Tensor,
    q_ref: torch.Tensor,
    *,
    slot: dict,
) -> torch.Tensor:
    # q_abs and q_ref are both (B, dof) Euler-parameterized padded q vectors. Returns
    # residual (B, dof + 3) with xyz / rot6d / joint_residual layout.
    if q_abs.shape != q_ref.shape:
        raise ValueError(f"q_abs and q_ref must share shape, got {tuple(q_abs.shape)} vs {tuple(q_ref.shape)}")
    dof = int(slot["dof"])
    if int(q_abs.shape[-1]) < dof:
        raise ValueError(f"q tensors must have at least dof={dof} trailing elements, got {tuple(q_abs.shape)}")
    q_abs = q_abs[..., :dof]
    q_ref = q_ref[..., :dof]
    joint_idx = list(slot["joint_indices"])

    pose_chunks: list[torch.Tensor] = []
    for group in slot.get("pose_groups", ()):
        trans_idx = list(group["translation_indices"])
        rot_idx = list(group["rotation_indices"])
        xyz_residual = q_abs[..., trans_idx] - q_ref[..., trans_idx]
        r_abs = euler_xyz_to_matrix(q_abs[..., rot_idx])
        r_ref = euler_xyz_to_matrix(q_ref[..., rot_idx])
        r_residual = r_abs @ r_ref.transpose(-1, -2)
        rot6d = matrix_to_rot6d(r_residual)
        pose_chunks.extend([xyz_residual, rot6d])
    joint_residual = q_abs[..., joint_idx] - q_ref[..., joint_idx]
    pose_chunks.append(joint_residual)
    return torch.cat(pose_chunks, dim=-1)


def residual_to_absolute(
    residual: torch.Tensor,
    q_ref: torch.Tensor,
    *,
    slot: dict,
) -> torch.Tensor:
    # Inverse of absolute_to_residual. Output q is Euler-parameterized (B, dof), matching
    # the dataset convention so chamfer/FK paths see what they already expect.
    #
    # The rot6d → matrix → Euler chain has sub-epsilon precision requirements that bf16
    # cannot meet: under AMP, the backward through `torch.cross` and `_normalize` in
    # rot6d_to_matrix produces NaN gradients whenever two intermediate vectors drift
    # close to parallel (|x ± y| near zero), and `matrix_to_euler_xyz`'s asin backward
    # diverges near gimbal lock. We force fp32 for the whole rotation conversion; the
    # autocast scope for the rest of the network is unchanged, and upcasting a 9-wide
    # vector is negligible cost.
    dof = int(slot["dof"])
    if int(q_ref.shape[-1]) < dof:
        raise ValueError(f"q_ref must have at least dof={dof} trailing elements, got {tuple(q_ref.shape)}")
    if int(residual.shape[-1]) != int(slot["residual_dim"]):
        raise ValueError(
            f"residual must have trailing dim {slot['residual_dim']}, got {tuple(residual.shape)}"
        )
    joint_idx = list(slot["joint_indices"])
    pose_slices = list(slot.get("pose_residual_slices", ()))

    out_dtype = residual.dtype
    with torch.amp.autocast(device_type=residual.device.type, enabled=False):
        residual_f = residual.to(torch.float32)
        q_ref_f = q_ref[..., :dof].to(torch.float32)

        q_abs_f = q_ref_f.clone()
        for group, slices in zip(slot.get("pose_groups", ()), pose_slices):
            trans_idx = list(group["translation_indices"])
            rot_idx = list(group["rotation_indices"])
            xyz_start, xyz_end = slices["xyz_slice"]
            rot_start, rot_end = slices["rot6d_slice"]
            xyz_res = residual_f[..., int(xyz_start) : int(xyz_end)]
            rot6d = residual_f[..., int(rot_start) : int(rot_end)]
            r_ref = euler_xyz_to_matrix(q_ref_f[..., rot_idx])
            r_residual = rot6d_to_matrix(rot6d)
            r_abs = r_residual @ r_ref
            euler_abs = matrix_to_euler_xyz(r_abs)
            q_abs_f[..., trans_idx] = q_ref_f[..., trans_idx] + xyz_res
            q_abs_f[..., rot_idx] = euler_abs
        joint_start, joint_end = slot["joints_slice"]
        joint_res = residual_f[..., int(joint_start) : int(joint_end)]
        if len(joint_idx) > 0:
            q_abs_f[..., joint_idx] = q_ref_f[..., joint_idx] + joint_res
    return q_abs_f.to(out_dtype)


def build_closed_form_residual_normalizer(
    *,
    num_robots: int,
    slot: dict,
    xyz_std: float,
    rot_std_rad: float,
    joint_std: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    min_std: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Closed-form residual normalizer derived from the known q1-noise model. Each per-channel
    # scale is 1 / per-channel std so normalized residuals land near unit variance; offsets
    # are zero for xyz / joints (residuals centered at 0) and -identity*scale for rot6d so
    # the identity rotation maps to 0 in normalized space.
    #
    # Assumes: q1 = q2 + Gaussian noise with known per-channel std. Valid only when q1 really
    # follows that noise model (q1_zero_probability = 0, random_q1 = false in the probe dataset).
    residual_dim = int(slot["residual_dim"])
    pose_slices = list(slot.get("pose_residual_slices", ()))
    num_pose_slots = 9 * len(pose_slices)
    num_joint_slots = residual_dim - num_pose_slots
    scale = torch.ones((int(num_robots), residual_dim), dtype=dtype, device=device)
    offset = torch.zeros((int(num_robots), residual_dim), dtype=dtype, device=device)

    safe_xyz = max(float(xyz_std), float(min_std))
    safe_rot = max(float(rot_std_rad), float(min_std))
    identity_rot6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=dtype, device=device)
    for slices in pose_slices:
        xyz_start, xyz_end = slices["xyz_slice"]
        rot_start, rot_end = slices["rot6d_slice"]
        scale[:, int(xyz_start) : int(xyz_end)] = 1.0 / safe_xyz
        scale[:, int(rot_start) : int(rot_end)] = 1.0 / safe_rot
        offset[:, int(rot_start) : int(rot_end)] = -identity_rot6d * scale[:, int(rot_start) : int(rot_end)]

    if num_joint_slots > 0:
        joint_std_t = torch.as_tensor(joint_std, dtype=dtype, device=device).view(-1)
        if int(joint_std_t.shape[0]) != num_joint_slots:
            raise ValueError(
                f"joint_std must have {num_joint_slots} elements matching slot['joint_indices'], "
                f"got {int(joint_std_t.shape[0])}"
            )
        eff = joint_std_t.clamp_min(float(min_std))
        joint_start, _ = slot["joints_slice"]
        scale[:, int(joint_start) : int(joint_start) + num_joint_slots] = (
            (1.0 / eff).unsqueeze(0).expand(int(num_robots), -1)
        )

    return scale, offset


def build_empirical_residual_normalizer(
    residuals: torch.Tensor,
    *,
    slot: dict,
    num_robots: int = 1,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    min_std: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Empirical residual normalizer: scale = 1/std, offset = -mean/std per channel so the
    # normalized distribution is zero-mean / unit-std across the channels that actually vary.
    # residuals: (N, residual_dim) collected from absolute_to_residual(q2, q1) over a
    # representative slice of the training split (run once at train start, cache to meta).
    residual_dim = int(slot["residual_dim"])
    if residuals.ndim != 2 or int(residuals.shape[-1]) != residual_dim:
        raise ValueError(
            f"residuals must have shape (N, {residual_dim}), got {tuple(residuals.shape)}"
        )
    if int(residuals.shape[0]) == 0:
        raise ValueError("empirical residual normalizer needs at least one sample")
    stats = residuals.to(dtype=torch.float64)  # float64 keeps the std accurate at the tail
    mean = stats.mean(dim=0)
    # Population std (unbiased=False) is fine here — we want the normalization to match the
    # distribution we trained on, not an estimator of the true population std.
    std = stats.std(dim=0, unbiased=False).clamp_min(float(min_std))
    per_channel_scale = (1.0 / std).to(dtype=dtype)
    per_channel_offset = (-mean / std).to(dtype=dtype)
    scale = per_channel_scale.unsqueeze(0).expand(int(num_robots), -1).contiguous().to(device=device)
    offset = per_channel_offset.unsqueeze(0).expand(int(num_robots), -1).contiguous().to(device=device)
    return scale, offset


def build_residual_normalizer(
    *,
    absolute_scale: torch.Tensor,
    absolute_offset: torch.Tensor,
    slot: dict,
    rot6d_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Derive a residual-space affine normalizer from the absolute-q normalizer. xyz and joint
    # residuals inherit the absolute *scale* (so the normalized magnitude stays comparable) but
    # get zero offset because residuals are centered at 0. rot6d slots are zero-centered around
    # the identity rotation so the network's regression target is near zero when q1 ≈ q2.
    if absolute_scale.shape != absolute_offset.shape:
        raise ValueError("absolute scale/offset shape mismatch")
    if absolute_scale.ndim != 2:
        raise ValueError(f"absolute_scale must be 2D (num_robots, dof), got {tuple(absolute_scale.shape)}")
    dof = int(slot["dof"])
    if int(absolute_scale.shape[1]) != dof:
        raise ValueError(
            f"absolute_scale trailing dim must match dof={dof}, got {tuple(absolute_scale.shape)}"
        )
    num_robots = int(absolute_scale.shape[0])
    residual_dim = int(slot["residual_dim"])
    residual_scale = torch.ones((num_robots, residual_dim), dtype=absolute_scale.dtype, device=absolute_scale.device)
    residual_offset = torch.zeros((num_robots, residual_dim), dtype=absolute_scale.dtype, device=absolute_scale.device)
    joint_idx = list(slot["joint_indices"])

    # rot6d identity is [1, 0, 0, 0, 1, 0]; subtract it so the identity rotation maps to 0
    # in normalized space — removes the constant learning burden during denoising.
    identity_rot6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=absolute_scale.dtype, device=absolute_scale.device)
    for group, slices in zip(slot.get("pose_groups", ()), slot.get("pose_residual_slices", ())):
        trans_idx = list(group["translation_indices"])
        xyz_start, xyz_end = slices["xyz_slice"]
        rot_start, rot_end = slices["rot6d_slice"]
        if len(trans_idx) > 0:
            residual_scale[:, int(xyz_start) : int(xyz_end)] = absolute_scale[:, trans_idx]
        residual_scale[:, int(rot_start) : int(rot_end)] = float(rot6d_scale)
        residual_offset[:, int(rot_start) : int(rot_end)] = (
            -identity_rot6d * residual_scale[:, int(rot_start) : int(rot_end)]
        )
    if len(joint_idx) > 0:
        joint_start, _ = slot["joints_slice"]
        residual_scale[:, int(joint_start) : int(joint_start) + len(joint_idx)] = absolute_scale[:, joint_idx]
    return residual_scale, residual_offset


def build_residual_mask_from_absolute(
    absolute_mask: torch.Tensor,
    *,
    slot: dict,
) -> torch.Tensor:
    # Convert a (B, dof) boolean/float mask defined over absolute q dimensions into a (B, dof+3)
    # mask over residual slots. The 6 rot6d slots become active if ANY of the 3 Euler rotation
    # slots is active (they share one underlying rotation residual), matching the documented rule.
    if absolute_mask.ndim != 2:
        raise ValueError(f"absolute_mask must be 2D (B, dof), got {tuple(absolute_mask.shape)}")
    dof = int(slot["dof"])
    if int(absolute_mask.shape[1]) != dof:
        raise ValueError(f"absolute_mask trailing dim must be dof={dof}, got {tuple(absolute_mask.shape)}")
    joint_idx = list(slot["joint_indices"])
    batch_size = int(absolute_mask.shape[0])
    residual_dim = int(slot["residual_dim"])
    dtype = absolute_mask.dtype
    residual_mask = torch.zeros((batch_size, residual_dim), dtype=dtype, device=absolute_mask.device)

    for group, slices in zip(slot.get("pose_groups", ()), slot.get("pose_residual_slices", ())):
        trans_idx = list(group["translation_indices"])
        rot_idx = list(group["rotation_indices"])
        xyz_start, xyz_end = slices["xyz_slice"]
        rot_start, rot_end = slices["rot6d_slice"]
        if len(trans_idx) > 0:
            residual_mask[:, int(xyz_start) : int(xyz_end)] = absolute_mask[:, trans_idx]
        if len(rot_idx) > 0:
            any_rot = absolute_mask[:, rot_idx].to(dtype=torch.bool).any(dim=-1, keepdim=True)
            residual_mask[:, int(rot_start) : int(rot_end)] = any_rot.to(dtype=dtype).expand(-1, 6)
    if len(joint_idx) > 0:
        joint_start, _ = slot["joints_slice"]
        residual_mask[:, int(joint_start) : int(joint_start) + len(joint_idx)] = absolute_mask[:, joint_idx]
    return residual_mask
