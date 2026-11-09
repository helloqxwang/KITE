from __future__ import annotations

"""Canonical MANO surface templates for synthetic decoder training."""

import hashlib
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_model.contact_surface import extract_sparse_contact_support_batch

if __package__:
    from .mano_utils import load_mano_mesh_and_joints
else:
    from mano_utils import load_mano_mesh_and_joints


MANO_SURFACE_TEMPLATE_VERSION = "mano_surface_template_v1"
MANO_SURFACE_TEMPLATE_SEED = 7
MANO_SURFACE_TEMPLATE_OVERSAMPLE = 8
MANO_VERTEX_TEMPLATE_VERSION = "mano_vertex_template_v1"


# Describe one canonical MANO surface template shared across all sequences.
@dataclass(frozen=True)
class ManoSurfaceTemplate:
    side: str
    num_points: int
    canonical_face_indices: np.ndarray
    canonical_face_vertex_indices: np.ndarray
    canonical_barycentric_coords: np.ndarray
    canonical_points: np.ndarray
    canonical_normals: np.ndarray
    template_hash: str
    version: str


# Compute vertex normals on posed MANO meshes for contact queries and visualization.
def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices_np = np.asarray(vertices, dtype=np.float32)
    faces_np = np.asarray(faces, dtype=np.int64)
    squeeze = False
    if vertices_np.ndim == 2:
        vertices_np = vertices_np[None, ...]
        squeeze = True
    normals = np.zeros_like(vertices_np, dtype=np.float32)
    tris = vertices_np[:, faces_np]
    face_normals = np.cross(
        tris[:, :, 1] - tris[:, :, 0], tris[:, :, 2] - tris[:, :, 0]
    )
    for batch_idx in range(int(vertices_np.shape[0])):
        for corner in range(3):
            np.add.at(normals[batch_idx], faces_np[:, corner], face_normals[batch_idx])
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.clip(norm, 1.0e-8, None)
    if squeeze:
        return normals[0]
    return normals


# Sample barycentric anchors once on the canonical MANO mesh so the template is pose-stable.
def _sample_surface_barycentric(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    num_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices_np = np.asarray(vertices, dtype=np.float32)
    faces_np = np.asarray(faces, dtype=np.int64)
    tris = vertices_np[faces_np]
    face_areas = (
        np.linalg.norm(
            np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1
        )
        * 0.5
    )
    face_prob = face_areas / np.clip(face_areas.sum(), 1.0e-12, None)
    rng = np.random.default_rng(int(seed))
    face_indices = rng.choice(
        len(faces_np), size=int(num_samples), replace=True, p=face_prob
    )
    u = rng.random((int(num_samples), 1), dtype=np.float32)
    v = rng.random((int(num_samples), 1), dtype=np.float32)
    sqrt_u = np.sqrt(u)
    bary = np.concatenate(
        [
            1.0 - sqrt_u,
            sqrt_u * (1.0 - v),
            sqrt_u * v,
        ],
        axis=1,
    ).astype(np.float32)
    face_vertices = tris[face_indices]
    points = (face_vertices * bary[:, :, None]).sum(axis=1)
    return (
        points.astype(np.float32),
        face_indices.astype(np.int64),
        bary.astype(np.float32),
    )


# Reduce an oversampled canonical surface set to a well-spread deterministic template.
def _deterministic_fps(points: np.ndarray, count: int) -> np.ndarray:
    points_np = np.asarray(points, dtype=np.float32)
    num_points = int(points_np.shape[0])
    if count >= num_points:
        return np.arange(num_points, dtype=np.int64)
    centroid = points_np.mean(axis=0, keepdims=True)
    start_dist = np.linalg.norm(points_np - centroid, axis=1)
    start_idx = int(np.argmax(start_dist))
    selected = [start_idx]
    min_dist = np.linalg.norm(points_np - points_np[start_idx : start_idx + 1], axis=1)
    while len(selected) < int(count):
        min_dist[np.asarray(selected, dtype=np.int64)] = -1.0
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        next_dist = np.linalg.norm(
            points_np - points_np[next_idx : next_idx + 1], axis=1
        )
        min_dist = np.minimum(min_dist, next_dist)
    return np.asarray(selected, dtype=np.int64)


# Hash the template contents so payloads can confirm they all use the same surface basis.
def _template_hash(
    *,
    side: str,
    num_points: int,
    face_indices: np.ndarray,
    barycentric_coords: np.ndarray,
    canonical_points: np.ndarray,
    version: str = MANO_SURFACE_TEMPLATE_VERSION,
) -> str:
    payload = [
        str(version).encode("utf-8"),
        str(side).encode("utf-8"),
        np.asarray([int(num_points)], dtype=np.int64).tobytes(),
        np.ascontiguousarray(face_indices.astype(np.int64)).tobytes(),
        np.ascontiguousarray(barycentric_coords.astype(np.float32)).tobytes(),
        np.ascontiguousarray(canonical_points.astype(np.float32)).tobytes(),
    ]
    digest = hashlib.sha1(b"".join(payload)).hexdigest()
    return digest


# Build and cache the canonical MANO surface template used by preprocessing and visualization.
@lru_cache(maxsize=8)
def get_mano_surface_template(
    side: str,
    mano_model_root: str | Path,
    num_points: int,
) -> ManoSurfaceTemplate:
    side_name = str(side).lower()
    mano_root = Path(mano_model_root).expanduser().resolve()
    zero_pose = np.zeros((48,), dtype=np.float32)
    zero_beta = np.zeros((10,), dtype=np.float32)
    zero_trans = np.zeros((3,), dtype=np.float32)
    canonical_vertices, _, faces = load_mano_mesh_and_joints(
        pose=zero_pose,
        beta=zero_beta,
        trans=zero_trans,
        side=side_name,
        mano_model_root=mano_root,
        device="cpu",
    )
    canonical_vertices = canonical_vertices[0]
    canonical_vertex_count = int(canonical_vertices.shape[0])
    requested_points = int(num_points)
    if requested_points > canonical_vertex_count:
        raise ValueError(
            f"MANO surface template size cannot exceed the loaded MANO vertex count: "
            f"requested {requested_points}, vertex_count={canonical_vertex_count}"
        )
    if requested_points == canonical_vertex_count:
        canonical_vertex_normals = _compute_vertex_normals(canonical_vertices, faces)
        face_indices = np.full((canonical_vertex_count,), -1, dtype=np.int64)
        canonical_face_vertex_indices = np.repeat(
            np.arange(canonical_vertex_count, dtype=np.int64)[:, None],
            3,
            axis=1,
        )
        barycentric_coords = np.zeros((canonical_vertex_count, 3), dtype=np.float32)
        barycentric_coords[:, 0] = 1.0
        template_hash = _template_hash(
            side=side_name,
            num_points=canonical_vertex_count,
            face_indices=face_indices,
            barycentric_coords=barycentric_coords,
            canonical_points=canonical_vertices,
            version=MANO_VERTEX_TEMPLATE_VERSION,
        )
        return ManoSurfaceTemplate(
            side=side_name,
            num_points=canonical_vertex_count,
            canonical_face_indices=face_indices,
            canonical_face_vertex_indices=canonical_face_vertex_indices,
            canonical_barycentric_coords=barycentric_coords,
            canonical_points=canonical_vertices.astype(np.float32, copy=False),
            canonical_normals=canonical_vertex_normals.astype(np.float32, copy=False),
            template_hash=template_hash,
            version=MANO_VERTEX_TEMPLATE_VERSION,
        )
    oversample = max(
        int(num_points) * MANO_SURFACE_TEMPLATE_OVERSAMPLE, int(num_points)
    )
    sampled_points, sampled_face_indices, sampled_bary = _sample_surface_barycentric(
        vertices=canonical_vertices,
        faces=faces,
        num_samples=oversample,
        seed=MANO_SURFACE_TEMPLATE_SEED,
    )
    selected = _deterministic_fps(sampled_points, int(num_points))
    face_indices = sampled_face_indices[selected]
    barycentric_coords = sampled_bary[selected]
    canonical_face_vertex_indices = faces[face_indices].astype(np.int64, copy=False)
    canonical_points = sampled_points[selected].astype(np.float32, copy=False)
    canonical_vertex_normals = _compute_vertex_normals(canonical_vertices, faces)
    canonical_normals = (
        canonical_vertex_normals[canonical_face_vertex_indices]
        * barycentric_coords[:, :, None]
    ).sum(axis=1)
    canonical_normals = canonical_normals / np.clip(
        np.linalg.norm(canonical_normals, axis=1, keepdims=True),
        1.0e-8,
        None,
    )
    template_hash = _template_hash(
        side=side_name,
        num_points=int(num_points),
        face_indices=face_indices,
        barycentric_coords=barycentric_coords,
        canonical_points=canonical_points,
    )
    return ManoSurfaceTemplate(
        side=side_name,
        num_points=int(num_points),
        canonical_face_indices=face_indices.astype(np.int64, copy=False),
        canonical_face_vertex_indices=canonical_face_vertex_indices.astype(
            np.int64, copy=False
        ),
        canonical_barycentric_coords=barycentric_coords.astype(np.float32, copy=False),
        canonical_points=canonical_points.astype(np.float32, copy=False),
        canonical_normals=canonical_normals.astype(np.float32, copy=False),
        template_hash=template_hash,
        version=MANO_SURFACE_TEMPLATE_VERSION,
    )
