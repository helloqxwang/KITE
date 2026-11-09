"""KITE Sec. IV-C: fixed trajectory indices with configurable contact scoring."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans

DEFAULT_CLUSTER_SEED = 7
CONTACT_EXTRACTION_VERSION = "paper_trajectory_contacts_v5"


def offset_contact_positions(
    contact_cloud: np.ndarray,
    real_contact_mask: np.ndarray,
    *,
    maximum_m: float = 0.0,
    phase: np.ndarray | None = None,
    close_phase_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    maximum_m = float(maximum_m)
    if not np.isfinite(maximum_m) or maximum_m < 0:
        raise ValueError("Contact position offset must be finite and nonnegative")
    cloud = np.asarray(contact_cloud, dtype=np.float32)
    real = np.asarray(real_contact_mask, dtype=bool)
    if cloud.ndim != 3 or cloud.shape[-1] != 6 or real.shape != cloud.shape[:2]:
        raise ValueError("Expected a (T, K, 6) contact cloud and (T, K) mask")
    result = cloud.copy()
    applied = np.zeros(real.shape, dtype=np.float32)
    if maximum_m == 0:
        return result, applied
    if phase is None or close_phase_id is None:
        raise ValueError("Nonzero contact offset requires demonstrated close-phase metadata")
    phase = np.asarray(phase)
    if phase.shape != (len(cloud),):
        raise ValueError("Contact offset phase must have one entry per frame")
    close = np.flatnonzero(phase == int(close_phase_id))
    if len(close) < 2 or not np.array_equal(close, np.arange(close[0], close[-1] + 1)):
        raise ValueError("Contact offset requires a contiguous close phase of at least two frames")
    schedule = np.full(len(cloud), maximum_m, dtype=np.float32)
    schedule[:close[0]] = 0
    schedule[close[0]:close[-1] + 1] = np.linspace(0, maximum_m, len(close), dtype=np.float32)
    applied = schedule[:, None] * real
    moving = applied > 0
    normals = cloud[..., 3:6][moving]
    lengths = np.linalg.norm(normals, axis=-1)
    if not np.isfinite(normals).all() or np.any(lengths <= 1e-8):
        raise ValueError("Offset contacts require finite nonzero surface normals")
    direction = normals / lengths[:, None]
    result[..., :3][moving] += applied[moving, None] * direction
    return result, applied


def score_surface_contacts(
    *,
    source_points: torch.Tensor,
    source_normals: torch.Tensor,
    target_points: torch.Tensor,
    target_normals: torch.Tensor | None = None,
    align_exp_scale: float = 2.0,
    sigmoid_scale: float = 10.0,
    target_chunk_size: int = 256,
) -> torch.Tensor:
    """Appendix A2/A3 in displacement coordinates, with bounded working memory.

    Subtract positions before taking norms. Computing squared distance as
    ``p.p + y.y - 2*p.y`` loses precision at millimeter contacts in world
    coordinates and can incorrectly assign a perfect contact score.
    This scorer is local to demonstration extraction; decoder training is
    unaffected.
    """
    if (
        source_points.ndim != 3
        or source_points.shape != source_normals.shape
        or source_points.shape[-1] != 3
        or target_points.ndim != 3
        or target_points.shape[0] != source_points.shape[0]
        or target_points.shape[-1] != 3
        or target_points.shape[1] == 0
        or target_chunk_size < 1
    ):
        raise ValueError("Expected compatible (B, N, 3) source and object clouds")
    points = source_points.float()
    normals = source_normals.float()
    objects = target_points.float()
    minimum = torch.full(
        points.shape[:2], float("inf"), device=points.device, dtype=points.dtype
    )
    nearest_alignment = torch.zeros_like(minimum)
    if target_normals is not None and target_normals.shape != target_points.shape:
        raise ValueError('Object surface normals must match object point shape')
    for start in range(0, objects.shape[1], target_chunk_size):
        displacement = (
            objects[:, None, start : start + target_chunk_size] - points[:, :, None]
        )
        distance = torch.linalg.vector_norm(displacement, dim=-1)
        if target_normals is not None:
            # Compare the two actual outward surface normals at the closest
            # object sample. Unlike a displacement direction, this remains
            # well defined through small simulated interpenetrations.
            nearest_distance, index = distance.min(dim=-1)
            batch = torch.arange(len(points), device=points.device)[:, None]
            object_normal = target_normals[:, start:start + target_chunk_size][batch, index]
            alignment = -(normals * object_normal).sum(-1).clamp(-1, 1)
            better = nearest_distance < minimum
            nearest_alignment = torch.where(better, alignment, nearest_alignment)
            minimum = torch.minimum(minimum, nearest_distance)
            continue
        alignment = (displacement * normals[:, :, None]).sum(-1) / (distance + 1e-5)
        aligned_distance = distance * torch.exp(
            float(align_exp_scale) * (1 - alignment)
        )
        minimum = torch.minimum(minimum, aligned_distance.amin(dim=-1))
    if target_normals is not None:
        minimum = minimum * torch.exp(float(align_exp_scale) * (1 - nearest_alignment))
    return 2 * torch.sigmoid(-float(sigmoid_scale) * minimum.sqrt())


@dataclass(frozen=True)
class SparseContactSlots:
    contact_cloud: np.ndarray
    contact_valid_mask: np.ndarray
    selected_template_indices: np.ndarray
    contact_slot_real_mask: np.ndarray
    contact_slot_supplement_mask: np.ndarray
    contact_slot_source_frame: np.ndarray
    trajectory_candidate_indices: np.ndarray
    trajectory_vertex_cluster_id: np.ndarray
    trajectory_num_clusters: int
    region_counts: np.ndarray


@lru_cache(maxsize=4)
def _surface_graph(neighbors: tuple[tuple[int, ...], ...]) -> csr_matrix:
    """Build each immutable template graph once across demonstration episodes."""
    n = len(neighbors)
    rows, cols = [], []
    for i, row in enumerate(neighbors):
        for j in row:
            if not 0 <= int(j) < n:
                raise ValueError(f"Invalid surface graph edge {i}, {j}")
            rows.append(i)
            cols.append(int(j))
    return csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n, n))


def contact_region_counts(
    active: np.ndarray, neighbors: Sequence[Sequence[int]]
) -> np.ndarray:
    """Count connected active regions, reusing results for identical frame masks."""
    active = np.asarray(active, dtype=bool)
    if active.ndim != 2 or len(neighbors) != active.shape[1]:
        raise ValueError("Surface adjacency must have one row per template point")
    graph = _surface_graph(tuple(tuple(row) for row in neighbors))
    result = np.zeros(len(active), dtype=np.int32)
    counts = {}
    for t, mask in enumerate(active):
        key = mask.tobytes()
        if key not in counts:
            indices = np.flatnonzero(mask)
            counts[key] = (
                connected_components(graph[indices][:, indices], directed=False, return_labels=False)
                if indices.size else 0
            )
        result[t] = counts[key]
    return result


def assemble_trajectory_sparse_contact_slots(
    *,
    surface_points_world: np.ndarray,
    surface_normals_world: np.ndarray,
    surface_points_canonical: np.ndarray,
    surface_neighbors: Sequence[Sequence[int]],
    sparse_contact_value: np.ndarray,
    capacity: int = 6,
    contact_threshold: float = 0.05,
    seed: int = DEFAULT_CLUSTER_SEED,
) -> SparseContactSlots:
    """Cluster candidates, then retain the cumulative-score winner per cluster."""
    points = np.asarray(surface_points_world, dtype=np.float32)
    normals = np.asarray(surface_normals_world, dtype=np.float32)
    canonical = np.asarray(surface_points_canonical, dtype=np.float32)
    scores = np.asarray(sparse_contact_value, dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 3 or not points.shape[0]:
        raise ValueError("Expected a nonempty (T, N, 3) surface trajectory")
    if (
        normals.shape != points.shape
        or canonical.shape != points.shape[1:]
        or scores.shape != points.shape[:2]
    ):
        raise ValueError(
            "Surface positions, normals, canonical template, and scores have inconsistent shapes"
        )
    if not all(np.isfinite(a).all() for a in (points, normals, canonical, scores)):
        raise ValueError("Surface trajectory and contact scores must be finite")
    if not 2 <= int(capacity) <= 6:
        raise ValueError(
            "Paper extraction requires a slot capacity between two and six"
        )
    active = scores >= float(contact_threshold)
    candidates = np.flatnonzero(active.any(axis=0))
    if len(candidates) < 2:
        raise RuntimeError(
            f"Paper contact extraction requires at least two candidates; found {len(candidates)}"
        )
    counts = contact_region_counts(active, surface_neighbors)
    # Average over interaction frames so approach/withdrawal duration does not
    # change the number of represented regions. FK targets still cover all frames.
    # Rounding is unspecified in the paper: nearest integer, halves upward.
    mean_regions = counts[counts > 0].mean()
    k = min(int(capacity), len(candidates), max(2, int(np.floor(mean_regions + 0.5))))
    labels = KMeans(n_clusters=k, n_init=10, random_state=int(seed)).fit_predict(
        canonical[candidates]
    )
    if len(np.unique(labels)) != k:
        raise RuntimeError(
            "Contact candidates cannot form the required number of distinct spatial clusters"
        )
    total_scores = np.where(active, scores, 0.0).sum(axis=0, dtype=np.float64)
    winners = []
    for cluster in range(k):
        indices = candidates[labels == cluster]
        winners.append(
            (int(indices[np.lexsort((indices, -total_scores[indices]))[0]]), cluster)
        )
    winners.sort()
    selected = np.asarray([index for index, _ in winners], dtype=np.int64)
    frames, n = points.shape[:2]
    cloud = np.zeros((frames, capacity, 6), dtype=np.float32)
    cloud[:, :k, :3] = points[:, selected]
    cloud[:, :k, 3:] = normals[:, selected]
    valid = np.zeros((frames, capacity), dtype=bool)
    valid[:, :k] = True
    selected_indices = np.full((frames, capacity), -1, dtype=np.int64)
    selected_indices[:, :k] = selected
    real = np.zeros_like(valid)
    real[:, :k] = active[:, selected]
    # No temporal filling: all motion targets use the current frame's FK.
    source_frame = np.where(valid, np.arange(frames)[:, None], -1)
    vertex_clusters = np.full(n, -1, dtype=np.int64)
    for slot, (_, cluster) in enumerate(winners):
        vertex_clusters[candidates[labels == cluster]] = slot
    return SparseContactSlots(
        cloud,
        valid,
        selected_indices,
        real,
        np.zeros_like(valid),
        source_frame,
        candidates,
        vertex_clusters,
        k,
        counts,
    )


def extract_contact_sequence_from_surface(
    *,
    surface_points_world: np.ndarray,
    surface_normals_world: np.ndarray,
    surface_points_canonical: np.ndarray,
    surface_neighbors: Sequence[Sequence[int]],
    object_points_world: np.ndarray,
    object_normals_world: np.ndarray | None = None,
    contact_threshold: float = 0.05,
    align_exp_scale: float = 2.0,
    sigmoid_scale: float = 10.0,
    contact_mode: str = "trajectory_sparse",
    contact_sparse_capacity: int = 6,
    seed: int = DEFAULT_CLUSTER_SEED,
    score_chunk_frames: int = 4,
    score_device: str = "cpu",
) -> dict[str, object]:
    """Score bounded frame chunks, then select fixed indices over the trajectory."""
    if contact_mode != "trajectory_sparse":
        raise ValueError(
            "KITE demonstration extraction supports contact_mode=trajectory_sparse"
        )
    points = np.asarray(surface_points_world, dtype=np.float32)
    normals = np.asarray(surface_normals_world, dtype=np.float32)
    objects = np.asarray(object_points_world, dtype=np.float32)
    if (
        objects.ndim != 3
        or objects.shape[-1] != 3
        or len(objects) != len(points)
        or not objects.shape[1]
    ):
        raise ValueError("Expected nonempty object points with shape (T, M, 3)")
    if not np.isfinite(objects).all() or int(score_chunk_frames) < 1:
        raise ValueError(
            "Object points must be finite and score_chunk_frames must be positive"
        )
    scores = np.empty(points.shape[:2], dtype=np.float32)
    device = torch.device(
        "cuda"
        if score_device == "auto" and torch.cuda.is_available()
        else "cpu" if score_device == "auto" else score_device
    )
    with torch.inference_mode():
        for start in range(0, len(points), int(score_chunk_frames)):
            stop = start + int(score_chunk_frames)
            scores[start:stop] = (
                score_surface_contacts(
                    source_points=torch.from_numpy(points[start:stop]).to(device),
                    source_normals=torch.from_numpy(normals[start:stop]).to(device),
                    target_points=torch.from_numpy(objects[start:stop]).to(device),
                    target_normals=(None if object_normals_world is None else
                                    torch.from_numpy(np.asarray(object_normals_world[start:stop], dtype=np.float32)).to(device)),
                    align_exp_scale=align_exp_scale,
                    sigmoid_scale=sigmoid_scale,
                )
                .cpu()
                .numpy()
            )
    slots = assemble_trajectory_sparse_contact_slots(
        surface_points_world=points,
        surface_normals_world=normals,
        surface_points_canonical=surface_points_canonical,
        surface_neighbors=surface_neighbors,
        sparse_contact_value=scores,
        capacity=contact_sparse_capacity,
        contact_threshold=contact_threshold,
        seed=seed,
    )
    active = scores >= contact_threshold
    return {
        "contact_extraction_version": CONTACT_EXTRACTION_VERSION,
        "contact_region_average_scope": "active_contact_frames",
        "contact_alignment_mode": ('nearest_surface_normals' if object_normals_world is not None else 'displacement_direction'),
        "template_points_world": points,
        "template_normals_world": normals,
        "real_sparse_mask": active,
        "filled_sparse_mask": active.copy(),
        "source_frame_index": np.arange(len(points), dtype=np.int64),
        "has_real_contact": active.any(axis=1),
        "sparse_contact_value": scores,
        "contact_mode": contact_mode,
        "contact_cloud": slots.contact_cloud,
        "contact_valid_mask": slots.contact_valid_mask,
        "selected_template_indices": slots.selected_template_indices,
        "num_selected_per_frame": slots.contact_valid_mask.sum(axis=1).astype(np.int32),
        "contact_sparse_capacity": contact_sparse_capacity,
        "trajectory_candidate_indices": slots.trajectory_candidate_indices,
        "trajectory_vertex_cluster_id": slots.trajectory_vertex_cluster_id,
        "trajectory_num_clusters": slots.trajectory_num_clusters,
        "contact_region_counts": slots.region_counts,
        "contact_slot_real_mask": slots.contact_slot_real_mask,
        "contact_slot_supplement_mask": slots.contact_slot_supplement_mask,
        "contact_slot_source_frame": slots.contact_slot_source_frame,
    }
