from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def _random_tangent_basis(normals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ref = (
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=normals.device)
        .unsqueeze(0)
        .repeat(normals.shape[0], 1)
    )
    alt = (
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=normals.device)
        .unsqueeze(0)
        .repeat(normals.shape[0], 1)
    )
    use_alt = torch.abs((normals * ref).sum(dim=1)) > 0.9
    ref[use_alt] = alt[use_alt]
    t1 = torch.cross(normals, ref, dim=1)
    t1 = t1 / torch.norm(t1, dim=1, keepdim=True).clamp_min(1e-8)
    t2 = torch.cross(normals, t1, dim=1)
    t2 = t2 / torch.norm(t2, dim=1, keepdim=True).clamp_min(1e-8)
    return t1, t2


def _broadcast_batch_param(
    value: float | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    out = torch.as_tensor(value, dtype=torch.float32, device=device)
    if out.ndim == 0:
        out = out.repeat(batch_size)
    else:
        out = out.view(-1)
    if int(out.numel()) != int(batch_size):
        raise ValueError(f"{name} must broadcast to batch size {batch_size}, got {tuple(out.shape)}")
    return out


def _sample_virtual_patch_geometry_batch(
    *,
    device: torch.device,
    anchor_points: torch.Tensor,
    anchor_normals: torch.Tensor,
    generator: Optional[torch.Generator],
    anchor_shift_min: float | torch.Tensor,
    anchor_shift_max: float | torch.Tensor,
    max_plane_extent_min: float | torch.Tensor,
    max_plane_extent_max: float | torch.Tensor,
    patch_shift_power: float,
    patch_extent_power: float,
    normal_jitter_max_deg: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, max_anchors, _ = anchor_points.shape
    anchor_shift_min_t = _broadcast_batch_param(
        anchor_shift_min, batch_size=batch_size, device=device, name="anchor_shift_min"
    )
    anchor_shift_max_t = _broadcast_batch_param(
        anchor_shift_max, batch_size=batch_size, device=device, name="anchor_shift_max"
    )
    anchor_shift_max_t = torch.maximum(anchor_shift_max_t, anchor_shift_min_t)
    extent_min_t = _broadcast_batch_param(
        max_plane_extent_min, batch_size=batch_size, device=device, name="max_plane_extent_min"
    )
    extent_max_t = _broadcast_batch_param(
        max_plane_extent_max, batch_size=batch_size, device=device, name="max_plane_extent_max"
    )
    extent_max_t = torch.maximum(extent_max_t, extent_min_t)

    us = torch.rand((batch_size,), device=device, generator=generator).pow(max(1e-3, float(patch_shift_power)))
    ue = torch.rand((batch_size,), device=device, generator=generator).pow(max(1e-3, float(patch_extent_power)))
    anchor_shift_t = anchor_shift_min_t + us * (anchor_shift_max_t - anchor_shift_min_t).clamp_min(1e-8)
    max_plane_extent_t = extent_min_t + ue * (extent_max_t - extent_min_t).clamp_min(1e-8)

    base_t1, base_t2 = _random_tangent_basis(anchor_normals.reshape(-1, 3))
    base_t1 = base_t1.view(batch_size, max_anchors, 3)
    base_t2 = base_t2.view(batch_size, max_anchors, 3)
    max_jitter_rad = math.radians(max(0.0, float(normal_jitter_max_deg)))
    if max_jitter_rad > 0.0:
        jitter_phi = torch.rand((batch_size, max_anchors), device=device, generator=generator) * (2.0 * math.pi)
        jitter_theta = torch.rand((batch_size, max_anchors), device=device, generator=generator) * max_jitter_rad
        jitter_tangent = (
            torch.cos(jitter_phi).unsqueeze(-1) * base_t1
            + torch.sin(jitter_phi).unsqueeze(-1) * base_t2
        )
        perturbed_normals = (
            torch.cos(jitter_theta).unsqueeze(-1) * anchor_normals
            + torch.sin(jitter_theta).unsqueeze(-1) * jitter_tangent
        )
        perturbed_normals = perturbed_normals / perturbed_normals.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    else:
        perturbed_normals = anchor_normals

    flat_t1, flat_t2 = _random_tangent_basis(perturbed_normals.reshape(-1, 3))
    t1 = flat_t1.view(batch_size, max_anchors, 3)
    t2 = flat_t2.view(batch_size, max_anchors, 3)
    centers = anchor_points + anchor_shift_t[:, None, None] * perturbed_normals

    u0 = torch.rand((batch_size, max_anchors), device=device, generator=generator)
    u1 = torch.rand((batch_size, max_anchors), device=device, generator=generator)
    ex = max_plane_extent_t[:, None] * u0.pow(0.32)
    ey = max_plane_extent_t[:, None] * u1.pow(0.32)

    boost_mask = torch.rand((batch_size, max_anchors), device=device, generator=generator) < 0.18
    boost = torch.empty((batch_size, max_anchors), device=device).uniform_(1.2, 1.8, generator=generator)
    ex = torch.where(boost_mask, ex * boost, ex)
    ey = torch.where(boost_mask, ey * boost, ey)

    edge_mask = torch.rand((batch_size, max_anchors), device=device, generator=generator) < 0.35
    edge_scale = torch.empty((batch_size, max_anchors), device=device).uniform_(0.04, 0.25, generator=generator)
    edge_axis = torch.randint(
        0,
        2,
        (batch_size, max_anchors),
        device=device,
        generator=generator,
    ).bool()
    ex = torch.where(edge_mask & edge_axis, ex * edge_scale, ex)
    ey = torch.where(edge_mask & (~edge_axis), ey * edge_scale, ey)
    ex = ex.clamp_min(2e-4)
    ey = ey.clamp_min(2e-4)
    return centers, t1, t2, ex, ey


def sample_object_patches_batch(
    *,
    device: torch.device,
    hand_points: torch.Tensor,
    hand_normals: torch.Tensor,
    anchor_indices: torch.Tensor,
    anchor_valid_mask: torch.Tensor,
    anchor_point_valid_mask: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    anchor_shift_min: float | torch.Tensor = 0.0002,
    anchor_shift_max: float | torch.Tensor = 0.0030,
    max_plane_extent_min: float | torch.Tensor = 0.006,
    max_plane_extent_max: float | torch.Tensor = 0.050,
    patch_shift_power: float = 1.0,
    patch_extent_power: float = 1.0,
    normal_jitter_max_deg: float = 10.0,
    penetration_clearance: float = 0.0001,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if hand_points.ndim != 3 or hand_normals.ndim != 3:
        raise ValueError(
            f"hand_points and hand_normals must have shape (B, N, 3), got {tuple(hand_points.shape)} and {tuple(hand_normals.shape)}"
        )
    if hand_points.shape != hand_normals.shape:
        raise ValueError(
            f"hand_points and hand_normals must have the same shape, got {tuple(hand_points.shape)} vs {tuple(hand_normals.shape)}"
        )
    if anchor_indices.ndim != 2 or anchor_valid_mask.ndim != 2:
        raise ValueError(
            f"anchor_indices and anchor_valid_mask must have shape (B, A), got {tuple(anchor_indices.shape)} and {tuple(anchor_valid_mask.shape)}"
        )
    if anchor_indices.shape != anchor_valid_mask.shape:
        raise ValueError(
            f"anchor_indices and anchor_valid_mask must have the same shape, got {tuple(anchor_indices.shape)} vs {tuple(anchor_valid_mask.shape)}"
        )
    if anchor_point_valid_mask.ndim != 3 or anchor_point_valid_mask.shape[:2] != anchor_indices.shape:
        raise ValueError(
            "anchor_point_valid_mask must have shape (B, A, P) aligned with anchor_indices, "
            f"got {tuple(anchor_point_valid_mask.shape)} for anchors {tuple(anchor_indices.shape)}"
        )

    batch_size, num_points, _ = hand_points.shape
    _, max_anchors = anchor_indices.shape
    max_points_per_anchor = int(anchor_point_valid_mask.shape[-1])
    if int(batch_size) == 0 or max_anchors == 0 or max_points_per_anchor == 0:
        empty_points = torch.zeros(
            (batch_size, max_anchors, max_points_per_anchor, 3),
            dtype=torch.float32,
            device=device,
        )
        empty_mask = torch.zeros(
            (batch_size, max_anchors, max_points_per_anchor),
            dtype=torch.bool,
            device=device,
        )
        return empty_points, empty_mask

    anchor_indices = anchor_indices.to(device=device, dtype=torch.long)
    anchor_valid_mask = anchor_valid_mask.to(device=device).bool()
    anchor_point_valid_mask = anchor_point_valid_mask.to(device=device).bool()
    safe_anchor_indices = anchor_indices.clamp(0, max(0, int(num_points) - 1))
    gather_idx = safe_anchor_indices.unsqueeze(-1).expand(-1, -1, 3)
    anchor_points = torch.gather(hand_points, 1, gather_idx)
    anchor_normals = torch.gather(hand_normals, 1, gather_idx)
    centers, t1, t2, ex, ey = _sample_virtual_patch_geometry_batch(
        device=device,
        anchor_points=anchor_points,
        anchor_normals=anchor_normals,
        generator=generator,
        anchor_shift_min=anchor_shift_min,
        anchor_shift_max=anchor_shift_max,
        max_plane_extent_min=max_plane_extent_min,
        max_plane_extent_max=max_plane_extent_max,
        patch_shift_power=patch_shift_power,
        patch_extent_power=patch_extent_power,
        normal_jitter_max_deg=normal_jitter_max_deg,
    )

    u = (torch.rand((batch_size, max_anchors, max_points_per_anchor), device=device, generator=generator) * 2.0 - 1.0) * ex.unsqueeze(-1)
    v = (torch.rand((batch_size, max_anchors, max_points_per_anchor), device=device, generator=generator) * 2.0 - 1.0) * ey.unsqueeze(-1)
    object_points = (
        centers.unsqueeze(2)
        + u.unsqueeze(-1) * t1.unsqueeze(2)
        + v.unsqueeze(-1) * t2.unsqueeze(2)
    )

    point_valid_mask = anchor_point_valid_mask & anchor_valid_mask.unsqueeze(-1)
    flat_points = object_points.view(batch_size, max_anchors * max_points_per_anchor, 3)
    flat_valid_mask = point_valid_mask.view(batch_size, max_anchors * max_points_per_anchor)
    if bool(flat_valid_mask.any()):
        nearest_idx = torch.cdist(flat_points, hand_points).argmin(dim=-1)
        nearest_gather = nearest_idx.unsqueeze(-1).expand(-1, -1, 3)
        ref_points = torch.gather(hand_points, 1, nearest_gather)
        ref_normals = torch.gather(hand_normals, 1, nearest_gather)
        signed = torch.sum((flat_points - ref_points) * ref_normals, dim=-1)
        flat_valid_mask = flat_valid_mask & (signed >= float(penetration_clearance))
    flat_points = flat_points.masked_fill(~flat_valid_mask.unsqueeze(-1), 0.0)
    return (
        flat_points.view(batch_size, max_anchors, max_points_per_anchor, 3),
        flat_valid_mask.view(batch_size, max_anchors, max_points_per_anchor),
    )
