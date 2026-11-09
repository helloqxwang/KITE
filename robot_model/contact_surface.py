from __future__ import annotations

"""Shared sparse-contact utilities for both articulated robots and MANO."""

from typing import Optional, Tuple

import torch


# Score source surface points against target geometry with the GenDex-style alignment distance.
def compute_gendex_contact_value_source_target_batch(
    *,
    source_points: torch.Tensor,
    source_normals: torch.Tensor,
    target_points: torch.Tensor,
    target_valid_mask: Optional[torch.Tensor] = None,
    source_alignment: Optional[torch.Tensor] = None,
    align_exp_scale: float = 2.0,
    sigmoid_scale: float = 10.0,
) -> torch.Tensor:
    """Compute GenDex-style contact values for batched source/target point sets."""
    if source_points.ndim != 3 or source_normals.ndim != 3 or target_points.ndim != 3:
        raise ValueError(
            "Expected batched tensors with shapes (B, N, 3), (B, N, 3), and (B, K, 3)"
        )
    if source_points.shape != source_normals.shape:
        raise ValueError(
            f"source_points and source_normals must have the same shape, got {tuple(source_points.shape)} vs {tuple(source_normals.shape)}"
        )
    if int(source_points.shape[0]) != int(target_points.shape[0]):
        raise ValueError(
            f"source_points batch and target_points batch must match, got {source_points.shape[0]} vs {target_points.shape[0]}"
        )
    if int(source_points.shape[-1]) != 3 or int(target_points.shape[-1]) != 3:
        raise ValueError("Expected xyz points/normals with last dim = 3")

    batch_size = int(source_points.shape[0])
    num_source = int(source_points.shape[1])
    num_target = int(target_points.shape[1])
    if num_source == 0:
        return torch.zeros((batch_size, 0), dtype=torch.float32, device=source_points.device)
    if num_target == 0:
        return torch.zeros((batch_size, num_source), dtype=torch.float32, device=source_points.device)

    source_points = source_points.float()
    source_normals = source_normals.float()
    target_points = target_points.float()

    if target_valid_mask is not None:
        target_valid_mask = target_valid_mask.to(device=source_points.device).bool()
        if tuple(target_valid_mask.shape) != tuple(target_points.shape[:2]):
            raise ValueError(
                f"target_valid_mask must have shape {tuple(target_points.shape[:2])}, got {tuple(target_valid_mask.shape)}"
            )
    if source_alignment is not None:
        source_alignment = source_alignment.to(device=source_points.device, dtype=torch.float32)
        if tuple(source_alignment.shape) != tuple(source_points.shape[:2]):
            raise ValueError(
                f"source_alignment must have shape {tuple(source_points.shape[:2])}, got {tuple(source_alignment.shape)}"
            )

    ss = (source_points * source_points).sum(dim=2, keepdim=True)
    tt = (target_points * target_points).sum(dim=2).unsqueeze(1)
    st = torch.matmul(source_points, target_points.transpose(1, 2))
    source_target_dist = torch.sqrt((ss + tt - 2.0 * st).clamp_min(0.0))

    if source_alignment is None:
        target_dot_normal = torch.matmul(target_points, source_normals.transpose(1, 2))
        source_dot_normal = (source_points * source_normals).sum(dim=2, keepdim=True)
        source_target_align = target_dot_normal.transpose(1, 2) - source_dot_normal
        source_target_align = source_target_align / (source_target_dist + 1.0e-5)
    else:
        source_target_align = source_alignment.clamp(-1.0, 1.0).unsqueeze(-1).expand_as(source_target_dist)

    source_target_align_dist = source_target_dist * torch.exp(
        float(align_exp_scale) * (1.0 - source_target_align)
    )
    if target_valid_mask is not None:
        source_target_align_dist = source_target_align_dist.masked_fill(
            ~target_valid_mask.unsqueeze(1),
            float("inf"),
        )

    contact_dist = torch.sqrt(source_target_align_dist.min(dim=2).values.clamp_min(0.0))
    contact_value = 1.0 - 2.0 * (torch.sigmoid(float(sigmoid_scale) * contact_dist) - 0.5)
    return contact_value.clamp(0.0, 1.0)


# Turn sorted per-point scores into a boolean top-k support mask.
def build_topk_mask_from_sorted(
    sorted_idx: torch.Tensor,
    k_values: torch.Tensor,
    n_points: int,
) -> torch.Tensor:
    """Convert sorted indices and per-sample K values into boolean masks."""
    if sorted_idx.ndim != 2:
        raise ValueError(f"sorted_idx must have shape (B, N), got {tuple(sorted_idx.shape)}")
    k_values = k_values.to(device=sorted_idx.device, dtype=torch.long).clamp(0, int(n_points))
    rank = torch.arange(int(n_points), device=sorted_idx.device).unsqueeze(0)
    keep_sorted = rank < k_values.unsqueeze(1)
    out = torch.zeros_like(keep_sorted, dtype=torch.bool)
    out.scatter_(1, sorted_idx, keep_sorted)
    return out


# Threshold continuous contact scores and optionally clamp the support size to a target range.
def extract_sparse_contact_support_batch(
    *,
    source_points: torch.Tensor,
    source_normals: torch.Tensor,
    target_points: torch.Tensor,
    target_valid_mask: Optional[torch.Tensor] = None,
    source_alignment: Optional[torch.Tensor] = None,
    threshold: float = 0.4,
    contact_count_range: Optional[torch.Tensor | Tuple[int, int]] = None,
    align_exp_scale: float = 2.0,
    sigmoid_scale: float = 10.0,
) -> dict[str, torch.Tensor | dict[str, int]]:
    """Extract sparse contact masks from point clouds with optional count clamping."""
    if target_points.ndim == 4:
        batch_size = int(target_points.shape[0])
        target_points = target_points.reshape(batch_size, -1, 3)
        if target_valid_mask is not None:
            target_valid_mask = target_valid_mask.reshape(batch_size, -1)
    elif target_points.ndim != 3:
        raise ValueError(
            f"target_points must have shape (B, K, 3) or (B, A, P, 3), got {tuple(target_points.shape)}"
        )

    contact_values = compute_gendex_contact_value_source_target_batch(
        source_points=source_points,
        source_normals=source_normals,
        target_points=target_points,
        target_valid_mask=target_valid_mask,
        source_alignment=source_alignment,
        align_exp_scale=align_exp_scale,
        sigmoid_scale=sigmoid_scale,
    )
    sparse_contact_mask = contact_values >= float(threshold)

    if contact_count_range is None:
        stats = {
            "num_samples": int(contact_values.shape[0]),
            "in_range": int(contact_values.shape[0]),
            "too_low": 0,
            "too_high": 0,
        }
        return {
            "sparse_contact_mask": sparse_contact_mask.bool(),
            "sparse_contact_value": contact_values,
            "contact_count_interval_stats": stats,
        }

    batch_size, num_points = contact_values.shape
    sorted_idx = torch.argsort(contact_values, dim=1, descending=True)
    count_range_t = torch.as_tensor(contact_count_range, dtype=torch.long, device=contact_values.device)
    if count_range_t.ndim == 1:
        if int(count_range_t.numel()) != 2:
            raise ValueError(
                f"contact_count_range must have shape (2,) or (B, 2), got {tuple(count_range_t.shape)}"
            )
        count_range_t = count_range_t.view(1, 2).expand(batch_size, 2)
    elif count_range_t.ndim != 2 or tuple(count_range_t.shape) != (batch_size, 2):
        raise ValueError(
            f"contact_count_range must have shape (2,) or ({batch_size}, 2), got {tuple(count_range_t.shape)}"
        )
    count_lo = count_range_t[:, 0].clamp(1, int(num_points))
    count_hi = torch.maximum(count_lo, count_range_t[:, 1].clamp(1, int(num_points)))
    cur_k = sparse_contact_mask.sum(dim=1).long()
    lo_mask = build_topk_mask_from_sorted(sorted_idx, count_lo, int(num_points))
    hi_mask = build_topk_mask_from_sorted(sorted_idx, count_hi, int(num_points))

    too_low = cur_k < count_lo
    too_high = cur_k > count_hi
    in_range = (~too_low) & (~too_high)
    if bool(too_low.any()):
        sparse_contact_mask[too_low] = lo_mask[too_low]
    if bool(too_high.any()):
        sparse_contact_mask[too_high] = hi_mask[too_high]

    stats = {
        "num_samples": int(batch_size),
        "in_range": int(in_range.sum().item()),
        "too_low": int(too_low.sum().item()),
        "too_high": int(too_high.sum().item()),
    }
    return {
        "sparse_contact_mask": sparse_contact_mask.bool(),
        "sparse_contact_value": contact_values,
        "contact_count_interval_stats": stats,
    }


# Fall back to globally highest-scoring template points when the thresholded support is too small.
def _topk_indices_by_score(
    scores: torch.Tensor,
    *,
    count: int,
) -> torch.Tensor:
    scores_cpu = scores.detach().to(dtype=torch.float32, device="cpu").view(-1)
    ordered = sorted(
        range(int(scores_cpu.numel())),
        key=lambda idx: (-float(scores_cpu[idx].item()), int(idx)),
    )
    return torch.tensor(ordered[: int(count)], dtype=torch.long, device=scores.device)

