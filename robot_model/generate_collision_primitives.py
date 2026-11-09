from __future__ import annotations

"""Generate analytic collision primitive sidecars from existing RobotModel assets."""

import argparse
import json
import math
import sys
import warnings
import zlib
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
import trimesh

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is expected in the project env.
    cKDTree = None  # type: ignore[assignment]

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != SCRIPT_DIR]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from robot_model.collision_primitives import (
        SCHEMA_VERSION,
        collision_primitive_spec_path,
        save_collision_primitive_spec,
    )
    from robot_model.robot_model import create_robot_model, discover_robot_assets
except Exception:
    from collision_primitives import (  # type: ignore
        SCHEMA_VERSION,
        collision_primitive_spec_path,
        save_collision_primitive_spec,
    )
    from robot_model import create_robot_model, discover_robot_assets  # type: ignore


def _dedupe_preserve_order(names: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        name_s = str(name).strip()
        if not name_s or name_s in seen:
            continue
        seen.add(name_s)
        out.append(name_s)
    return out


def _load_robot_names(args: argparse.Namespace) -> list[str]:
    if args.robot_names:
        return _dedupe_preserve_order(args.robot_names)

    assets_dir = Path(args.assets_dir)
    if args.robot_list_key == "all":
        return sorted(discover_robot_assets(assets_dir).keys())

    list_file = Path(args.robot_list_file)
    if not list_file.exists():
        raise FileNotFoundError(f"Robot list file not found: {list_file}")
    with open(list_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    names = payload.get(args.robot_list_key)
    if not isinstance(names, list):
        raise KeyError(f"Key '{args.robot_list_key}' is not a list in {list_file}")
    return _dedupe_preserve_order([str(n) for n in names])


def _mesh_points(mesh: Any) -> np.ndarray:
    vertices = np.asarray(getattr(mesh, "vertices", np.zeros((0, 3))), dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    finite = np.isfinite(vertices).all(axis=1)
    return vertices[finite]


def _mesh_vertices_faces(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    vertices = _mesh_points(mesh)
    faces = np.asarray(getattr(mesh, "faces", np.zeros((0, 3), dtype=np.int64)), dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(vertices) == 0:
        return vertices, np.zeros((0, 3), dtype=np.int64)
    valid = (faces >= 0).all(axis=1) & (faces < len(vertices)).all(axis=1)
    return vertices, faces[valid]


def _pca_axes(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    if len(points) < 3 or float(np.linalg.norm(centered)) <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axes = vh.T
    except np.linalg.LinAlgError:
        axes = np.eye(3, dtype=np.float64)
    if axes.shape != (3, 3):
        axes_fixed = np.eye(3, dtype=np.float64)
        axes_fixed[:, : axes.shape[1]] = axes
        axes = axes_fixed
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    return axes.astype(np.float64)


def _obb_from_axes(points: np.ndarray, axes: np.ndarray, *, source: str) -> Dict[str, np.ndarray]:
    origin = points.mean(axis=0)
    coords = (points - origin) @ axes
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center_coords = 0.5 * (mins + maxs)
    half_extents = np.maximum(0.5 * (maxs - mins), 1.0e-8)
    center = origin + axes @ center_coords
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = axes
    tf[:3, 3] = center
    return {
        "origin": origin,
        "axes": axes,
        "coords": coords,
        "mins": mins,
        "maxs": maxs,
        "center": center,
        "half_extents": half_extents,
        "link_from_primitive": tf,
        "source": source,
    }


def _fit_pca_obb(points: np.ndarray) -> Dict[str, np.ndarray]:
    return _obb_from_axes(points, _pca_axes(points), source="pca")


def _fit_tight_obb(points: np.ndarray, angle_digits: int) -> Dict[str, np.ndarray]:
    if len(points) < 4:
        return _fit_pca_obb(points)
    try:
        to_origin, extents = trimesh.bounds.oriented_bounds(
            points,
            angle_digits=int(max(0, angle_digits)),
            ordered=True,
        )
        link_from_primitive = np.linalg.inv(np.asarray(to_origin, dtype=np.float64))
        axes = link_from_primitive[:3, :3]
        if axes.shape != (3, 3) or not np.isfinite(axes).all() or abs(np.linalg.det(axes)) < 1.0e-8:
            return _fit_pca_obb(points)
        # Numerical cleanup; oriented_bounds may return tiny drift from orthonormality.
        u, _, vh = np.linalg.svd(axes)
        axes = u @ vh
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1.0
        center = link_from_primitive[:3, 3]
        coords = (points - center[None, :]) @ axes
        half_extents = np.maximum(0.5 * np.asarray(extents, dtype=np.float64), 1.0e-8)
        tf = np.eye(4, dtype=np.float64)
        tf[:3, :3] = axes
        tf[:3, 3] = center
        return {
            "origin": center,
            "axes": axes,
            "coords": coords,
            "mins": coords.min(axis=0),
            "maxs": coords.max(axis=0),
            "center": center,
            "half_extents": half_extents,
            "link_from_primitive": tf,
            "source": "oriented_bounds",
        }
    except Exception:
        return _fit_pca_obb(points)


def _candidate_obbs(points: np.ndarray, args: argparse.Namespace, *, angle_digits: int | None = None) -> list[Dict[str, np.ndarray]]:
    digits = int(args.obb_angle_digits if angle_digits is None else angle_digits)
    candidates = [_fit_tight_obb(points, angle_digits=digits), _fit_pca_obb(points)]
    unique: list[Dict[str, np.ndarray]] = []
    for candidate in candidates:
        if not np.isfinite(candidate["half_extents"]).all():
            continue
        duplicate = False
        for existing in unique:
            if np.allclose(candidate["link_from_primitive"], existing["link_from_primitive"], atol=1.0e-6):
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    return unique or [_fit_pca_obb(points)]


def _capsule_volume(half_length: float, radius: float) -> float:
    return float(math.pi * radius * radius * (2.0 * half_length) + (4.0 / 3.0) * math.pi * radius**3)


def _rounded_box_volume(half_extents: Sequence[float], radius: float) -> float:
    half = np.asarray(half_extents, dtype=np.float64)
    a, b, c = [float(x) for x in half]
    r = float(radius)
    return float(
        8.0 * a * b * c
        + 8.0 * (a * b + a * c + b * c) * r
        + 2.0 * math.pi * (a + b + c) * r * r
        + (4.0 / 3.0) * math.pi * r**3
    )


def _make_capsule_candidate(
    link_name: str,
    points: np.ndarray,
    *,
    obb: Dict[str, np.ndarray],
    coverage_margin: float,
    min_radius: float,
) -> Dict[str, Any]:
    coords = obb["coords"]
    center_coords = np.array(
        [
            0.5 * (float(coords[:, 0].min()) + float(coords[:, 0].max())),
            float(np.median(coords[:, 1])),
            float(np.median(coords[:, 2])),
        ],
        dtype=np.float64,
    )
    local = coords - center_coords[None, :]
    abs_x = np.abs(local[:, 0])
    radial = np.linalg.norm(local[:, 1:3], axis=1)
    max_abs_x = float(abs_x.max())
    best_half_length = max_abs_x
    best_radius = max(float(radial.max()) + float(coverage_margin), float(min_radius))
    best_volume = _capsule_volume(best_half_length, best_radius)
    for half_length_raw in np.linspace(0.0, max_abs_x, num=32, dtype=np.float64):
        outside_x = np.maximum(abs_x - float(half_length_raw), 0.0)
        required_radius = float(np.sqrt(radial * radial + outside_x * outside_x).max()) + float(coverage_margin)
        radius_i = max(required_radius, float(min_radius))
        volume_i = _capsule_volume(float(half_length_raw), radius_i)
        if volume_i < best_volume:
            best_half_length = float(half_length_raw)
            best_radius = radius_i
            best_volume = volume_i
    radius = best_radius
    half_length = max(best_half_length, 1.0e-8)

    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = obb["axes"]
    tf[:3, 3] = obb["origin"] + obb["axes"] @ center_coords
    return {
        "link_name": link_name,
        "type": "capsule",
        "link_from_primitive": tf.tolist(),
        "params": {
            "half_length": float(half_length),
            "radius": float(radius),
        },
        "fit_metrics": {
            "cluster_vertex_count": int(len(points)),
            "primitive_fitted_volume": best_volume,
            "fit_frame": str(obb.get("source", "unknown")),
        },
    }


def _make_rounded_box_candidate(
    link_name: str,
    points: np.ndarray,
    *,
    obb: Dict[str, np.ndarray],
    coverage_margin: float,
) -> Dict[str, Any]:
    half_extents = np.maximum(obb["half_extents"], 1.0e-8)
    radius = float(max(coverage_margin, 1.0e-8))
    return {
        "link_name": link_name,
        "type": "rounded_box",
        "link_from_primitive": obb["link_from_primitive"].tolist(),
        "params": {
            "half_extents": [float(x) for x in half_extents],
            "radius": radius,
        },
        "fit_metrics": {
            "cluster_vertex_count": int(len(points)),
            "primitive_fitted_volume": _rounded_box_volume(half_extents, radius),
            "fit_frame": str(obb.get("source", "unknown")),
        },
    }


def _primitive_volume(primitive: Dict[str, Any]) -> float:
    return float(primitive.get("fit_metrics", {}).get("primitive_fitted_volume", float("inf")))


def _points_to_primitive(points: np.ndarray, link_from_primitive: np.ndarray) -> np.ndarray:
    rotation = link_from_primitive[:3, :3]
    translation = link_from_primitive[:3, 3]
    return (points - translation[None, :]) @ rotation


def _primitive_sdf_link(points: np.ndarray, primitive: Dict[str, Any]) -> np.ndarray:
    tf = np.asarray(primitive["link_from_primitive"], dtype=np.float64)
    p = _points_to_primitive(points, tf)
    params = primitive["params"]
    if primitive["type"] == "capsule":
        half_length = float(params["half_length"])
        radius = float(params["radius"])
        x = np.clip(p[:, 0], -half_length, half_length)
        closest = np.zeros_like(p)
        closest[:, 0] = x
        return np.linalg.norm(p - closest, axis=1) - radius
    if primitive["type"] == "box":
        half = np.asarray(params["half_extents"], dtype=np.float64)
        q = np.abs(p) - half[None, :]
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside
    if primitive["type"] == "rounded_box":
        half = np.asarray(params["half_extents"], dtype=np.float64)
        q = np.abs(p) - half[None, :]
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside - float(params["radius"])
    raise ValueError(f"Unsupported primitive type: {primitive['type']}")


def _union_sdf_link(points: np.ndarray, primitives: list[Dict[str, Any]]) -> np.ndarray:
    if not primitives:
        return np.full((len(points),), float("inf"), dtype=np.float64)
    return np.stack([_primitive_sdf_link(points, primitive) for primitive in primitives], axis=0).min(axis=0)


def _mesh_surface_samples(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    samples = [vertices]
    if len(faces) > 0:
        tri = vertices[faces]
        samples.extend(
            [
                tri.mean(axis=1),
                0.60 * tri[:, 0] + 0.20 * tri[:, 1] + 0.20 * tri[:, 2],
                0.20 * tri[:, 0] + 0.60 * tri[:, 1] + 0.20 * tri[:, 2],
                0.20 * tri[:, 0] + 0.20 * tri[:, 1] + 0.60 * tri[:, 2],
            ]
        )
    out = np.concatenate(samples, axis=0)
    finite = np.isfinite(out).all(axis=1)
    return out[finite]


def _limit_rows_deterministic(points: np.ndarray, max_count: int, *, seed_key: str) -> np.ndarray:
    max_count = int(max_count)
    if max_count <= 0 or len(points) <= max_count:
        return points
    rng = np.random.default_rng(_stable_seed(seed_key, len(points), max_count))
    indices = np.sort(rng.choice(len(points), size=max_count, replace=False))
    return points[indices]


def _limit_faces_deterministic(faces: np.ndarray, max_count: int, *, seed_key: str) -> np.ndarray:
    max_count = int(max_count)
    if max_count <= 0 or len(faces) <= max_count:
        return faces
    rng = np.random.default_rng(_stable_seed(seed_key, len(faces), max_count))
    indices = np.sort(rng.choice(len(faces), size=max_count, replace=False))
    return faces[indices]


def _nearest_distances(points: np.ndarray, support_points: np.ndarray, tree: Any | None) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0,), dtype=np.float64)
    if tree is not None:
        distances, _ = tree.query(points, k=1)
        return np.asarray(distances, dtype=np.float64)
    diff = points[:, None, :] - support_points[None, :, :]
    return np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))


def _stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int(zlib.crc32(payload) & 0xFFFFFFFF)


def _primitive_sdf_local_np(points_primitive: np.ndarray, primitive: Dict[str, Any]) -> np.ndarray:
    params = primitive["params"]
    if primitive["type"] == "capsule":
        half_length = float(params["half_length"])
        radius = float(params["radius"])
        x = np.clip(points_primitive[:, 0], -half_length, half_length)
        closest = np.zeros_like(points_primitive)
        closest[:, 0] = x
        return np.linalg.norm(points_primitive - closest, axis=1) - radius
    if primitive["type"] == "box":
        half = np.asarray(params["half_extents"], dtype=np.float64)
        q = np.abs(points_primitive) - half[None, :]
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside
    if primitive["type"] == "rounded_box":
        half = np.asarray(params["half_extents"], dtype=np.float64)
        q = np.abs(points_primitive) - half[None, :]
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside - float(params["radius"])
    raise ValueError(f"Unsupported primitive type: {primitive['type']}")


def _primitive_local_bounds(primitive: Dict[str, Any]) -> np.ndarray:
    params = primitive["params"]
    if primitive["type"] == "capsule":
        half_length = float(params["half_length"])
        radius = float(params["radius"])
        return np.array(
            [
                [-(half_length + radius), -radius, -radius],
                [half_length + radius, radius, radius],
            ],
            dtype=np.float64,
        )
    if primitive["type"] == "box":
        half = np.asarray(params["half_extents"], dtype=np.float64)
        return np.stack([-half, half], axis=0)
    if primitive["type"] == "rounded_box":
        half = np.asarray(params["half_extents"], dtype=np.float64) + float(params["radius"])
        return np.stack([-half, half], axis=0)
    raise ValueError(f"Unsupported primitive type: {primitive['type']}")


def _sample_primitive_volume_points(
    primitive: Dict[str, Any],
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    count = int(max(0, count))
    if count == 0:
        return np.zeros((0, 3), dtype=np.float64)
    bounds = _primitive_local_bounds(primitive)
    rng = np.random.default_rng(int(seed))
    accepted: list[np.ndarray] = []
    remaining = count
    attempts = 0
    while remaining > 0 and attempts < 12:
        batch = max(remaining * 4, 256)
        local = rng.uniform(bounds[0], bounds[1], size=(batch, 3))
        local = local[_primitive_sdf_local_np(local, primitive) <= 0.0]
        if len(local) > 0:
            accepted.append(local[:remaining])
            remaining -= min(remaining, len(local))
        attempts += 1
    if not accepted:
        local = np.zeros((count, 3), dtype=np.float64)
    else:
        local = np.concatenate(accepted, axis=0)
        if len(local) < count:
            repeats = np.resize(local, (count, 3))
            local = repeats
    tf = np.asarray(primitive["link_from_primitive"], dtype=np.float64)
    return local @ tf[:3, :3].T + tf[:3, 3][None, :]


def _sample_primitive_union_volume(
    primitives: list[Dict[str, Any]],
    *,
    sample_count: int,
    seed_key: str,
) -> np.ndarray:
    if not primitives or sample_count <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    volumes = np.asarray([max(_primitive_volume(primitive), 1.0e-12) for primitive in primitives], dtype=np.float64)
    raw_counts = np.maximum(8, np.round(float(sample_count) * volumes / volumes.sum()).astype(np.int64))
    if raw_counts.sum() > sample_count:
        scale = float(sample_count) / float(raw_counts.sum())
        raw_counts = np.maximum(8, np.floor(raw_counts * scale).astype(np.int64))
    while raw_counts.sum() < sample_count:
        raw_counts[int(np.argmax(volumes))] += 1
    points = []
    for primitive_idx, (primitive, count) in enumerate(zip(primitives, raw_counts)):
        points.append(
            _sample_primitive_volume_points(
                primitive,
                count=int(count),
                seed=_stable_seed(seed_key, primitive_idx, primitive.get("type", "")),
            )
        )
    return np.concatenate(points, axis=0)


def _empty_space_metrics(
    primitives: list[Dict[str, Any]],
    *,
    mesh_surface_samples: np.ndarray,
    mesh_tree: Any | None,
    link_diag: float,
    tolerance: float,
    sample_count: int,
    seed_key: str,
) -> Dict[str, float | int]:
    samples = _sample_primitive_union_volume(primitives, sample_count=sample_count, seed_key=seed_key)
    if len(samples) == 0 or len(mesh_surface_samples) == 0:
        return {
            "empty_space_score": 0.0,
            "empty_space_score_normalized": 0.0,
            "empty_space_mean": 0.0,
            "empty_space_p90": 0.0,
            "empty_space_p95": 0.0,
            "empty_space_max": 0.0,
            "empty_space_sample_count": int(len(samples)),
        }
    distances = _nearest_distances(samples, mesh_surface_samples, mesh_tree)
    excess = np.maximum(distances - float(tolerance), 0.0)
    mean = float(excess.mean())
    p90 = float(np.quantile(excess, 0.90))
    p95 = float(np.quantile(excess, 0.95))
    max_dist = float(excess.max())
    score = float(0.5 * mean + 0.3 * p90 + 0.2 * p95)
    return {
        "empty_space_score": score,
        "empty_space_score_normalized": score / max(float(link_diag), 1.0e-12),
        "empty_space_mean": mean,
        "empty_space_p90": p90,
        "empty_space_p95": p95,
        "empty_space_max": max_dist,
        "empty_space_sample_count": int(len(samples)),
    }


def _link_matches_split_keyword(link_name: str, args: argparse.Namespace) -> bool:
    keywords = [part.strip().lower() for part in str(args.split_search_link_keywords).split(",")]
    link_l = str(link_name).lower()
    return any(keyword and keyword in link_l for keyword in keywords)


def _add_link_fit_metrics(
    points: np.ndarray,
    primitives: list[Dict[str, Any]],
    *,
    best_single_volume: float,
    best_two_volume: float | None,
    single_empty_metrics: Dict[str, float | int],
    best_two_empty_metrics: Dict[str, float | int] | None,
    accepted_empty_metrics: Dict[str, float | int],
    accepted_split_type: str,
    split_search_reason: str,
    candidate_count: int,
) -> None:
    bounds = np.stack([points.min(axis=0), points.max(axis=0)], axis=0)
    sdf = _union_sdf_link(points, primitives)
    outside = np.maximum(sdf, 0.0)
    fitted_volume = float(sum(_primitive_volume(primitive) for primitive in primitives))
    aabb_volume = float(np.prod(np.maximum(bounds[1] - bounds[0], 0.0)))

    link_metrics = {
        "fit_method": "tight_empty_space_v4",
        "link_vertex_count": int(len(points)),
        "link_primitive_count": int(len(primitives)),
        "original_bounds": bounds.tolist(),
        "original_aabb_volume": aabb_volume,
        "link_fitted_volume": fitted_volume,
        "volume_ratio_to_aabb": fitted_volume / max(aabb_volume, 1.0e-12),
        "best_single_volume": float(best_single_volume),
        "best_two_volume": None if best_two_volume is None else float(best_two_volume),
        "single_empty_space_score": float(single_empty_metrics.get("empty_space_score", 0.0)),
        "single_empty_space_score_normalized": float(single_empty_metrics.get("empty_space_score_normalized", 0.0)),
        "best_two_empty_space_score": None
        if best_two_empty_metrics is None
        else float(best_two_empty_metrics.get("empty_space_score", 0.0)),
        "best_two_empty_space_score_normalized": None
        if best_two_empty_metrics is None
        else float(best_two_empty_metrics.get("empty_space_score_normalized", 0.0)),
        "empty_space_score": float(accepted_empty_metrics.get("empty_space_score", 0.0)),
        "empty_space_score_normalized": float(accepted_empty_metrics.get("empty_space_score_normalized", 0.0)),
        "empty_space_mean": float(accepted_empty_metrics.get("empty_space_mean", 0.0)),
        "empty_space_p90": float(accepted_empty_metrics.get("empty_space_p90", 0.0)),
        "empty_space_p95": float(accepted_empty_metrics.get("empty_space_p95", 0.0)),
        "empty_space_max": float(accepted_empty_metrics.get("empty_space_max", 0.0)),
        "empty_space_sample_count": int(accepted_empty_metrics.get("empty_space_sample_count", 0)),
        "accepted_split_type": str(accepted_split_type),
        "accepted_two_primitive": bool(len(primitives) == 2),
        "split_search_reason": str(split_search_reason),
        "split_candidate_count": int(candidate_count),
        "max_outside_error": float(outside.max()) if len(outside) else 0.0,
        "mean_outside_error": float(outside.mean()) if len(outside) else 0.0,
    }
    for primitive_index, primitive in enumerate(primitives):
        primitive["fit_metrics"].update(link_metrics)
        primitive["fit_metrics"]["primitive_index"] = int(primitive_index)


def _fit_best_primitive(
    link_name: str,
    points: np.ndarray,
    args: argparse.Namespace,
    *,
    angle_digits: int | None = None,
    seed_key: str | None = None,
    max_fit_points: int | None = None,
) -> Dict[str, Any] | None:
    if len(points) < int(args.min_vertices):
        return None
    digits = int(args.obb_angle_digits if angle_digits is None else angle_digits)
    fit_points = _limit_rows_deterministic(
        points,
        int(args.max_fit_points if max_fit_points is None else max_fit_points),
        seed_key=f"{link_name}:fit" if seed_key is None else seed_key,
    )
    tight_obb = _fit_tight_obb(fit_points, angle_digits=digits)
    tight_obb = _obb_from_axes(points, tight_obb["axes"], source=str(tight_obb.get("source", "unknown")))
    longest = float((2.0 * tight_obb["half_extents"]).max())
    if longest < float(args.min_extent):
        return None
    coverage_margin = max(float(args.coverage_abs_margin), float(args.coverage_rel_margin) * longest)
    min_radius = max(float(args.min_extent) * 0.5, 1.0e-6)

    candidates: list[Dict[str, Any]] = []
    for fit_obb in _candidate_obbs(fit_points, args, angle_digits=digits):
        obb = _obb_from_axes(points, fit_obb["axes"], source=str(fit_obb.get("source", "unknown")))
        extents = np.sort(2.0 * obb["half_extents"])[::-1]
        second = float(max(extents[1], 1.0e-8))
        aspect = float(extents[0] / second)
        rounded_box = _make_rounded_box_candidate(
            link_name,
            points,
            obb=obb,
            coverage_margin=coverage_margin,
        )
        rounded_box["fit_metrics"].update(
            {
                "selected_type": "rounded_box",
                "aspect_ratio": aspect,
                "coverage_margin": coverage_margin,
            }
        )
        candidates.append(rounded_box)
        if aspect >= float(args.capsule_aspect_threshold):
            capsule = _make_capsule_candidate(
                link_name,
                points,
                obb=obb,
                coverage_margin=coverage_margin,
                min_radius=min_radius,
            )
            capsule["fit_metrics"].update(
                {
                    "selected_type": "capsule",
                    "aspect_ratio": aspect,
                    "coverage_margin": coverage_margin,
                }
            )
            candidates.append(capsule)

    if not candidates:
        return None
    best = min(candidates, key=_primitive_volume)
    return best


def _face_adjacency_from_faces(faces: np.ndarray) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(len(faces))]
    edge_to_face: dict[tuple[int, int], int] = {}
    for face_idx, face in enumerate(faces):
        for a, b in ((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0]))):
            key = (a, b) if a <= b else (b, a)
            other = edge_to_face.get(key)
            if other is None:
                edge_to_face[key] = face_idx
            else:
                adjacency[face_idx].append(other)
                adjacency[other].append(face_idx)
    return adjacency


def _component_sizes_for_mask(mask: np.ndarray, adjacency: list[list[int]]) -> list[int]:
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros(len(mask), dtype=bool)
    sizes: list[int] = []
    for start in np.flatnonzero(mask):
        if visited[start]:
            continue
        stack = [int(start)]
        visited[start] = True
        size = 0
        while stack:
            face_idx = stack.pop()
            size += 1
            for neigh in adjacency[face_idx]:
                if mask[neigh] and not visited[neigh]:
                    visited[neigh] = True
                    stack.append(int(neigh))
        sizes.append(size)
    return sizes


def _split_mask_valid(mask: np.ndarray, adjacency: list[list[int]], args: argparse.Namespace, *, require_coherent: bool) -> bool:
    mask = np.asarray(mask, dtype=bool)
    face_count = int(mask.shape[0])
    min_faces = max(int(args.min_split_faces), int(math.ceil(float(args.min_split_face_fraction) * face_count)))
    if int(mask.sum()) < min_faces or int((~mask).sum()) < min_faces:
        return False
    if not require_coherent:
        return True
    for side in (mask, ~mask):
        sizes = _component_sizes_for_mask(side, adjacency)
        if not sizes:
            return False
        largest_fraction = float(max(sizes)) / float(max(1, int(side.sum())))
        if len(sizes) > int(args.max_split_components) and largest_fraction < float(args.min_largest_component_fraction):
            return False
    return True


def _append_unique_split(
    splits: list[tuple[str, np.ndarray]],
    seen: set[bytes],
    split_type: str,
    mask: np.ndarray,
    adjacency: list[list[int]],
    args: argparse.Namespace,
    *,
    require_coherent: bool = True,
) -> None:
    mask = np.asarray(mask, dtype=bool)
    if not _split_mask_valid(mask, adjacency, args, require_coherent=require_coherent):
        return
    if int(mask.sum()) > int((~mask).sum()):
        mask = ~mask
    key = np.packbits(mask.astype(np.uint8)).tobytes()
    if key in seen:
        return
    seen.add(key)
    splits.append((split_type, mask.copy()))


def _connected_component_splits(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_centroids: np.ndarray,
    obb: Dict[str, np.ndarray],
    adjacency: list[list[int]],
    args: argparse.Namespace,
    splits: list[tuple[str, np.ndarray]],
    seen: set[bytes],
) -> None:
    all_mask = np.ones(len(faces), dtype=bool)
    components = [np.flatnonzero(cmask) for cmask in _component_masks(all_mask, adjacency)]
    if len(components) < 2:
        return
    if len(components) > int(args.max_connected_component_split_candidates):
        return
    axis_idx = int(np.argmax(obb["half_extents"]))
    axis = obb["axes"][:, axis_idx]
    comp_centers = np.array([face_centroids[c].mean(axis=0) for c in components], dtype=np.float64)
    order = np.argsort(comp_centers @ axis)
    for cut in range(1, len(order)):
        mask = np.zeros(len(faces), dtype=bool)
        for comp_idx in order[:cut]:
            mask[components[int(comp_idx)]] = True
        _append_unique_split(
            splits,
            seen,
            "connected_components",
            mask,
            adjacency,
            args,
            require_coherent=False,
        )


def _component_masks(mask: np.ndarray, adjacency: list[list[int]]) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros(len(mask), dtype=bool)
    components: list[np.ndarray] = []
    for start in np.flatnonzero(mask):
        if visited[start]:
            continue
        comp = np.zeros(len(mask), dtype=bool)
        stack = [int(start)]
        visited[start] = True
        comp[start] = True
        while stack:
            face_idx = stack.pop()
            for neigh in adjacency[face_idx]:
                if mask[neigh] and not visited[neigh]:
                    visited[neigh] = True
                    comp[neigh] = True
                    stack.append(int(neigh))
        components.append(comp)
    return components


def _kmeans_masks(face_centroids: np.ndarray, axes: np.ndarray, args: argparse.Namespace) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for axis_idx in range(3):
        projection = face_centroids @ axes[:, axis_idx]
        lo = face_centroids[int(np.argmin(projection))]
        hi = face_centroids[int(np.argmax(projection))]
        centers = np.stack([lo, hi], axis=0)
        labels = np.zeros(len(face_centroids), dtype=np.int64)
        for _ in range(int(args.kmeans_iterations)):
            distances = ((face_centroids[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for label in (0, 1):
                if np.any(labels == label):
                    centers[label] = face_centroids[labels == label].mean(axis=0)
        masks.append(labels == 0)
    return masks


def _candidate_face_splits(vertices: np.ndarray, faces: np.ndarray, args: argparse.Namespace) -> list[tuple[str, np.ndarray]]:
    if len(faces) < max(2 * int(args.min_split_faces), 2):
        return []
    face_centroids = vertices[faces].mean(axis=1)
    adjacency = _face_adjacency_from_faces(faces)
    obb = _fit_tight_obb(vertices, angle_digits=int(args.split_obb_angle_digits))
    splits: list[tuple[str, np.ndarray]] = []
    seen: set[bytes] = set()

    _connected_component_splits(vertices, faces, face_centroids, obb, adjacency, args, splits, seen)

    spatial_require_coherent = bool(args.require_coherent_spatial_splits)
    for axis_idx in range(3):
        projection = face_centroids @ obb["axes"][:, axis_idx]
        if float(projection.max() - projection.min()) <= 1.0e-10:
            continue
        for quantile in (0.20, 0.35, 0.50, 0.65, 0.80):
            threshold = float(np.quantile(projection, quantile))
            _append_unique_split(
                splits,
                seen,
                f"axis_{axis_idx}_q{quantile:.2f}",
                projection <= threshold,
                adjacency,
                args,
                require_coherent=spatial_require_coherent,
            )

    for km_idx, mask in enumerate(_kmeans_masks(face_centroids, obb["axes"], args)):
        _append_unique_split(
            splits,
            seen,
            f"kmeans_{km_idx}",
            mask,
            adjacency,
            args,
            require_coherent=spatial_require_coherent,
        )

    return splits


def _point_indices_from_face_mask(vertices: np.ndarray, faces: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx0 = np.unique(faces[np.asarray(mask, dtype=bool)].reshape(-1))
    idx1 = np.unique(faces[~np.asarray(mask, dtype=bool)].reshape(-1))
    assigned = np.zeros(len(vertices), dtype=bool)
    assigned[idx0] = True
    assigned[idx1] = True
    missing = np.flatnonzero(~assigned)
    if len(missing) > 0 and len(idx0) > 0 and len(idx1) > 0:
        c0 = vertices[idx0].mean(axis=0)
        c1 = vertices[idx1].mean(axis=0)
        d0 = ((vertices[missing] - c0[None, :]) ** 2).sum(axis=1)
        d1 = ((vertices[missing] - c1[None, :]) ** 2).sum(axis=1)
        idx0 = np.concatenate([idx0, missing[d0 <= d1]])
        idx1 = np.concatenate([idx1, missing[d0 > d1]])
    return np.unique(idx0), np.unique(idx1)


def _fit_two_primitive_candidate(
    link_name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    split_type: str,
    mask: np.ndarray,
    args: argparse.Namespace,
    *,
    mesh_surface_samples: np.ndarray,
    mesh_tree: Any | None,
    link_diag: float,
    empty_tolerance: float,
) -> tuple[list[Dict[str, Any]], float, float, Dict[str, float | int]] | None:
    idx0, idx1 = _point_indices_from_face_mask(vertices, faces, mask)
    if len(idx0) < int(args.min_vertices) or len(idx1) < int(args.min_vertices):
        return None
    prim0 = _fit_best_primitive(
        link_name,
        vertices[idx0],
        args,
        angle_digits=int(args.split_obb_angle_digits),
        seed_key=f"{link_name}:{split_type}:0",
        max_fit_points=int(args.max_split_fit_points),
    )
    prim1 = _fit_best_primitive(
        link_name,
        vertices[idx1],
        args,
        angle_digits=int(args.split_obb_angle_digits),
        seed_key=f"{link_name}:{split_type}:1",
        max_fit_points=int(args.max_split_fit_points),
    )
    if prim0 is None or prim1 is None:
        return None
    primitives = [prim0, prim1]
    for cluster_idx, (primitive, indices, face_mask) in enumerate(((prim0, idx0, mask), (prim1, idx1, ~mask))):
        primitive["fit_metrics"].update(
            {
                "cluster_index": int(cluster_idx),
                "cluster_vertex_count": int(len(indices)),
                "cluster_face_count": int(face_mask.sum()),
                "split_type": str(split_type),
            }
        )
    volume = float(sum(_primitive_volume(primitive) for primitive in primitives))
    max_outside = float(np.maximum(_union_sdf_link(vertices, primitives), 0.0).max())
    if max_outside > float(args.max_outside_error):
        return None
    empty_metrics = _empty_space_metrics(
        primitives,
        mesh_surface_samples=mesh_surface_samples,
        mesh_tree=mesh_tree,
        link_diag=link_diag,
        tolerance=empty_tolerance,
        sample_count=int(args.empty_space_samples),
        seed_key=f"{link_name}:{split_type}",
    )
    primitive_quality = float(empty_metrics["empty_space_score"])
    return primitives, volume, primitive_quality, empty_metrics


def _fit_link_primitives(link_name: str, mesh: Any, args: argparse.Namespace) -> list[Dict[str, Any]]:
    vertices, faces = _mesh_vertices_faces(mesh)
    if len(vertices) < int(args.min_vertices):
        return []

    single = _fit_best_primitive(link_name, vertices, args)
    if single is None:
        return []
    single_volume = _primitive_volume(single)
    mesh_surface_samples = _limit_rows_deterministic(
        _mesh_surface_samples(vertices, faces),
        int(args.max_mesh_surface_samples),
        seed_key=f"{link_name}:surface",
    )
    mesh_tree = cKDTree(mesh_surface_samples) if cKDTree is not None and len(mesh_surface_samples) > 0 else None
    aabb_bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)], axis=0)
    link_diag = float(np.linalg.norm(aabb_bounds[1] - aabb_bounds[0]))
    empty_tolerance = max(float(args.empty_space_abs_tolerance), float(args.empty_space_rel_tolerance) * link_diag)
    single_empty_metrics = _empty_space_metrics(
        [single],
        mesh_surface_samples=mesh_surface_samples,
        mesh_tree=mesh_tree,
        link_diag=link_diag,
        tolerance=empty_tolerance,
        sample_count=int(args.empty_space_samples),
        seed_key=f"{link_name}:single",
    )
    single_empty_normalized = float(single_empty_metrics["empty_space_score_normalized"])

    best_two: tuple[list[Dict[str, Any]], float, float, str, Dict[str, float | int]] | None = None
    single_ratio = single_volume / max(float(np.prod(np.maximum(aabb_bounds[1] - aabb_bounds[0], 0.0))), 1.0e-12)
    split_candidates: list[tuple[str, np.ndarray]] = []
    split_reasons: list[str] = []
    if single_ratio >= float(args.split_search_min_volume_ratio):
        split_reasons.append("volume")
    if single_empty_normalized >= float(args.split_search_min_empty_score):
        split_reasons.append("empty_space")
    if _link_matches_split_keyword(link_name, args):
        split_reasons.append("keyword")
    split_search_reason = "+".join(split_reasons) if split_reasons else "skipped"
    if int(args.max_primitives_per_link) >= 2 and split_reasons:
        split_faces = _limit_faces_deterministic(
            faces,
            int(args.max_split_faces_considered),
            seed_key=f"{link_name}:split_faces",
        )
        split_candidates = _candidate_face_splits(vertices, split_faces, args)

    if int(args.max_primitives_per_link) >= 2 and len(split_candidates) > 0:
        for split_type, mask in split_candidates:
            candidate = _fit_two_primitive_candidate(
                link_name,
                vertices,
                split_faces,
                split_type,
                mask,
                args,
                mesh_surface_samples=mesh_surface_samples,
                mesh_tree=mesh_tree,
                link_diag=link_diag,
                empty_tolerance=empty_tolerance,
            )
            if candidate is None:
                continue
            primitives, volume, empty_score, empty_metrics = candidate
            if best_two is None:
                best_two = (primitives, volume, empty_score, split_type, empty_metrics)
            else:
                best_key = (
                    best_two[2],
                    best_two[1],
                )
                candidate_key = (
                    empty_score,
                    volume,
                )
                if candidate_key < best_key:
                    best_two = (primitives, volume, empty_score, split_type, empty_metrics)

    accept_by_volume = best_two is not None and best_two[1] <= float(args.two_primitive_accept_ratio) * single_volume
    accept_by_empty_space = (
        best_two is not None
        and single_empty_normalized >= float(args.two_primitive_empty_min_score)
        and best_two[2] <= float(args.two_primitive_empty_accept_ratio) * float(single_empty_metrics["empty_space_score"])
        and best_two[1] <= float(args.two_primitive_max_volume_ratio) * single_volume
    )
    if best_two is not None and (accept_by_volume or accept_by_empty_space):
        primitives = best_two[0]
        best_two_volume = best_two[1]
        best_two_empty_metrics = best_two[4]
        accepted_empty_metrics = best_two_empty_metrics
        accepted_split_type = best_two[3]
        if accept_by_empty_space and not accept_by_volume:
            accepted_split_type = f"{accepted_split_type}:empty_space"
    else:
        primitives = [single]
        best_two_volume = None if best_two is None else best_two[1]
        best_two_empty_metrics = None if best_two is None else best_two[4]
        accepted_empty_metrics = single_empty_metrics
        accepted_split_type = "single"

    _add_link_fit_metrics(
        vertices,
        primitives,
        best_single_volume=single_volume,
        best_two_volume=best_two_volume,
        single_empty_metrics=single_empty_metrics,
        best_two_empty_metrics=best_two_empty_metrics,
        accepted_empty_metrics=accepted_empty_metrics,
        accepted_split_type=accepted_split_type,
        split_search_reason=split_search_reason,
        candidate_count=len(split_candidates),
    )
    return primitives


def generate_spec_for_robot(robot_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    model = create_robot_model(
        robot_name=robot_name,
        device=torch.device(args.device),
        num_points=16,
        assets_dir=Path(args.assets_dir),
        load_surface_assets=False,
    )
    if model.__class__.__name__ == "HumanHandModel" or not hasattr(model, "meshes_original"):
        raise RuntimeError(f"{robot_name} resolved to {model.__class__.__name__}; no asset meshes to fit")

    primitives: list[Dict[str, Any]] = []
    skipped_links: list[str] = []
    for link_name, mesh in model.meshes_original.items():
        fitted = _fit_link_primitives(str(link_name), mesh, args)
        if fitted:
            primitives.extend(fitted)
        else:
            skipped_links.append(str(link_name))

    if not primitives:
        raise RuntimeError(f"No collision primitives were fitted for {robot_name}")

    try:
        robot_path = str(Path(model.robot_path).resolve().relative_to(ROOT_DIR))
    except ValueError:
        robot_path = str(Path(model.robot_path).resolve())

    link_metric_by_name: dict[str, Dict[str, Any]] = {}
    for primitive in primitives:
        link_metric_by_name.setdefault(str(primitive["link_name"]), primitive.get("fit_metrics", {}))
    volume_ratios = [
        float(metrics.get("volume_ratio_to_aabb", 0.0))
        for metrics in link_metric_by_name.values()
        if metrics.get("volume_ratio_to_aabb") is not None
    ]
    empty_scores = [
        float(metrics.get("empty_space_score_normalized", 0.0))
        for metrics in link_metric_by_name.values()
        if metrics.get("empty_space_score_normalized") is not None
    ]
    primitive_type_counts = {
        primitive_type: int(sum(1 for primitive in primitives if primitive["type"] == primitive_type))
        for primitive_type in ("capsule", "rounded_box", "box")
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "fit_method": "tight_empty_space_v4",
        "robot_name": str(robot_name),
        "resolved_robot_name": str(getattr(model, "robot_name", robot_name)),
        "robot_path": robot_path,
        "fit_parameters": {
            "capsule_aspect_threshold": float(args.capsule_aspect_threshold),
            "max_primitives_per_link": int(args.max_primitives_per_link),
            "two_primitive_accept_ratio": float(args.two_primitive_accept_ratio),
            "coverage_abs_margin": float(args.coverage_abs_margin),
            "coverage_rel_margin": float(args.coverage_rel_margin),
            "obb_angle_digits": int(args.obb_angle_digits),
            "split_obb_angle_digits": int(args.split_obb_angle_digits),
            "split_search_min_volume_ratio": float(args.split_search_min_volume_ratio),
            "split_search_min_empty_score": float(args.split_search_min_empty_score),
            "split_search_link_keywords": str(args.split_search_link_keywords),
            "max_split_faces_considered": int(args.max_split_faces_considered),
            "max_connected_component_split_candidates": int(args.max_connected_component_split_candidates),
            "max_fit_points": int(args.max_fit_points),
            "max_split_fit_points": int(args.max_split_fit_points),
            "two_primitive_empty_accept_ratio": float(args.two_primitive_empty_accept_ratio),
            "two_primitive_empty_min_score": float(args.two_primitive_empty_min_score),
            "two_primitive_max_volume_ratio": float(args.two_primitive_max_volume_ratio),
            "empty_space_samples": int(args.empty_space_samples),
            "max_mesh_surface_samples": int(args.max_mesh_surface_samples),
            "empty_space_abs_tolerance": float(args.empty_space_abs_tolerance),
            "empty_space_rel_tolerance": float(args.empty_space_rel_tolerance),
            "require_coherent_spatial_splits": bool(args.require_coherent_spatial_splits),
            "max_outside_error": float(args.max_outside_error),
            "min_split_faces": int(args.min_split_faces),
            "min_split_face_fraction": float(args.min_split_face_fraction),
            "min_extent": float(args.min_extent),
            "min_vertices": int(args.min_vertices),
        },
        "fit_summary": {
            "fitted_link_count": int(len({prim["link_name"] for prim in primitives})),
            "skipped_link_count": int(len(skipped_links)),
            "skipped_links": skipped_links,
            "primitive_count": int(len(primitives)),
            "primitive_type_counts": primitive_type_counts,
            "two_primitive_link_count": int(
                sum(1 for metrics in link_metric_by_name.values() if int(metrics.get("link_primitive_count", 0)) == 2)
            ),
            "max_volume_ratio_to_aabb": float(max(volume_ratios)) if volume_ratios else 0.0,
            "mean_volume_ratio_to_aabb": float(np.mean(volume_ratios)) if volume_ratios else 0.0,
            "max_empty_space_score_normalized": float(max(empty_scores)) if empty_scores else 0.0,
            "mean_empty_space_score_normalized": float(np.mean(empty_scores)) if empty_scores else 0.0,
        },
        "primitives": primitives,
    }


def add_collision_primitive_fit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--capsule-aspect-threshold", type=float, default=1.7)
    parser.add_argument("--max-primitives-per-link", type=int, default=2)
    parser.add_argument("--two-primitive-accept-ratio", type=float, default=0.85)
    parser.add_argument("--coverage-abs-margin", type=float, default=5.0e-5)
    parser.add_argument("--coverage-rel-margin", type=float, default=0.0)
    parser.add_argument("--obb-angle-digits", type=int, default=2)
    parser.add_argument("--split-obb-angle-digits", type=int, default=1)
    parser.add_argument("--split-search-min-volume-ratio", type=float, default=1.15)
    parser.add_argument("--split-search-min-empty-score", type=float, default=0.08)
    parser.add_argument("--split-search-link-keywords", type=str, default="palm,wrist")
    parser.add_argument("--max-split-faces-considered", type=int, default=4000)
    parser.add_argument("--max-connected-component-split-candidates", type=int, default=16)
    parser.add_argument("--max-fit-points", type=int, default=8000)
    parser.add_argument("--max-split-fit-points", type=int, default=1500)
    parser.add_argument("--two-primitive-empty-accept-ratio", type=float, default=0.82)
    parser.add_argument("--two-primitive-empty-min-score", type=float, default=0.035)
    parser.add_argument("--two-primitive-max-volume-ratio", type=float, default=1.25)
    parser.add_argument("--empty-space-samples", type=int, default=1024)
    parser.add_argument("--max-mesh-surface-samples", type=int, default=30000)
    parser.add_argument("--empty-space-abs-tolerance", type=float, default=0.0015)
    parser.add_argument("--empty-space-rel-tolerance", type=float, default=0.0)
    parser.add_argument("--require-coherent-spatial-splits", action="store_true")
    parser.add_argument("--max-outside-error", type=float, default=1.0e-6)
    parser.add_argument("--min-split-faces", type=int, default=8)
    parser.add_argument("--min-split-face-fraction", type=float, default=0.08)
    parser.add_argument("--max-split-components", type=int, default=3)
    parser.add_argument("--min-largest-component-fraction", type=float, default=0.80)
    parser.add_argument("--kmeans-iterations", type=int, default=20)
    parser.add_argument("--min-extent", type=float, default=1.0e-5)
    parser.add_argument("--min-vertices", type=int, default=8)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate analytic collision primitive sidecars for RobotModel assets.")
    parser.add_argument("--assets-dir", type=str, default=str(ROOT_DIR / "assets"))
    parser.add_argument("--robot-list-file", type=str, default=str(ROOT_DIR / "assets" / "manipulator_robot_lists.json"))
    parser.add_argument("--robot-list-key", type=str, default="right_hands", choices=["all", "both_hands", "right_hands"])
    parser.add_argument("--robot-names", type=str, nargs="*", default=None)
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "assets" / "collision_primitives"))
    parser.add_argument("--device", type=str, default="cpu")
    add_collision_primitive_fit_args(parser)
    args = parser.parse_args(argv)

    robot_names = _load_robot_names(args)
    if not robot_names:
        raise RuntimeError("No robots selected for collision primitive generation.")

    generated = 0
    failed: dict[str, str] = {}
    for robot_name in robot_names:
        try:
            spec = generate_spec_for_robot(robot_name, args)
            out_path = collision_primitive_spec_path(args.output_dir, robot_name)
            save_collision_primitive_spec(out_path, spec)
            generated += 1
            print(
                f"[generated] {robot_name} -> {out_path} "
                f"(links={spec['fit_summary']['fitted_link_count']}, "
                f"primitives={spec['fit_summary']['primitive_count']})"
            )
        except Exception as exc:
            failed[str(robot_name)] = str(exc)
            warnings.warn(f"[skipped] {robot_name}: {exc}", RuntimeWarning)

    print(f"Generated specs: {generated}/{len(robot_names)}")
    if failed:
        print(f"Skipped/failed specs: {len(failed)}")
        for name, reason in failed.items():
            print(f"  - {name}: {reason}")
    if generated == 0:
        raise RuntimeError("No collision primitive specs were generated.")


if __name__ == "__main__":
    main()
