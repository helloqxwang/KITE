from __future__ import annotations

"""Analytic collision primitive sidecars for articulated ``RobotModel`` assets."""

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import trimesh


SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, 2}
SUPPORTED_PRIMITIVE_TYPES = {"capsule", "box", "rounded_box"}


def safe_robot_name(robot_name: str) -> str:
    """Return a stable filesystem-safe key for a public robot name."""
    name = str(robot_name).strip().strip("/")
    name = name.replace("\\", "/").replace("/", "__")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name or "robot"


def collision_primitive_spec_path(spec_dir: str | Path, robot_name: str) -> Path:
    """Resolve the sidecar path used by the generator and visualizer."""
    return Path(spec_dir) / f"{safe_robot_name(robot_name)}.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_collision_primitive_spec(path: str | Path) -> Dict[str, Any]:
    """Load and minimally validate a collision primitive JSON sidecar."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    schema_version = int(spec.get("schema_version", -1))
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported collision primitive schema_version={schema_version}; "
            f"expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}: {path}"
        )
    primitives = spec.get("primitives")
    if not isinstance(primitives, list):
        raise ValueError(f"Collision primitive spec is missing a primitives list: {path}")
    for idx, prim in enumerate(primitives):
        if not isinstance(prim, dict):
            raise ValueError(f"Primitive {idx} is not an object in {path}")
        if prim.get("type") not in SUPPORTED_PRIMITIVE_TYPES:
            raise ValueError(f"Primitive {idx} has unsupported type={prim.get('type')!r} in {path}")
        if "link_name" not in prim or "link_from_primitive" not in prim or "params" not in prim:
            raise ValueError(f"Primitive {idx} is missing required fields in {path}")
    return spec


def save_collision_primitive_spec(path: str | Path, spec: Dict[str, Any]) -> None:
    """Write a collision primitive JSON sidecar with deterministic formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = dict(spec)
    spec["schema_version"] = int(spec.get("schema_version", SCHEMA_VERSION))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def _as_transform(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    t = torch.as_tensor(value, dtype=dtype, device=device)
    if t.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got shape {tuple(t.shape)}")
    return t


def _as_batched_q(robot_model: Any, q: Optional[torch.Tensor | np.ndarray]) -> torch.Tensor:
    if q is None:
        q = robot_model.get_canonical_q()
    q_t = torch.as_tensor(q, dtype=torch.float32, device=robot_model.device)
    if q_t.ndim == 1:
        q_t = q_t.unsqueeze(0)
    if q_t.ndim != 2:
        raise ValueError(f"Expected q shape (D,) or (B,D), got {tuple(q_t.shape)}")
    return q_t


def _as_batched_points(
    points_world: torch.Tensor | np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    points = torch.as_tensor(points_world, dtype=torch.float32, device=device)
    if points.ndim == 2:
        points = points.unsqueeze(0)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected points shape (N,3) or (B,N,3), got {tuple(points.shape)}")

    point_batch = int(points.shape[0])
    if point_batch == batch_size:
        return points, batch_size
    if point_batch == 1 and batch_size > 1:
        return points.expand(batch_size, -1, -1), batch_size
    if batch_size == 1 and point_batch > 1:
        return points, point_batch
    raise ValueError(f"Cannot broadcast q batch {batch_size} against point batch {point_batch}")


def _capsule_sdf(points_primitive: torch.Tensor, half_length: float, radius: float) -> torch.Tensor:
    x = points_primitive[..., 0].clamp(min=-float(half_length), max=float(half_length))
    closest = torch.stack(
        [
            x,
            torch.zeros_like(points_primitive[..., 1]),
            torch.zeros_like(points_primitive[..., 2]),
        ],
        dim=-1,
    )
    return torch.linalg.norm(points_primitive - closest, dim=-1) - float(radius)


def _box_sdf(points_primitive: torch.Tensor, half_extents: Iterable[float]) -> torch.Tensor:
    half = torch.as_tensor(list(half_extents), dtype=points_primitive.dtype, device=points_primitive.device)
    q = points_primitive.abs() - half.view(1, 1, 3)
    outside = torch.linalg.norm(q.clamp_min(0.0), dim=-1)
    inside = q.max(dim=-1).values.clamp_max(0.0)
    return outside + inside


def _rounded_box_sdf(points_primitive: torch.Tensor, half_extents: Iterable[float], radius: float) -> torch.Tensor:
    return _box_sdf(points_primitive, half_extents) - float(radius)


def _transform_points_world_to_primitive(points_world: torch.Tensor, world_from_primitive: torch.Tensor) -> torch.Tensor:
    rotation = world_from_primitive[:, :3, :3]
    translation = world_from_primitive[:, :3, 3]
    return torch.matmul(points_world - translation[:, None, :], rotation)


def _empty_trimesh() -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.zeros((0, 3), dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int64),
        process=False,
    )


def _merge_meshes(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    meshes = [m for m in meshes if m is not None and len(m.vertices) > 0 and len(m.faces) > 0]
    if not meshes:
        return _empty_trimesh()
    return trimesh.util.concatenate(meshes)


def _rounded_box_mesh(half_extents: Iterable[float], radius: float, sections: int = 16) -> trimesh.Trimesh:
    half = np.asarray(list(half_extents), dtype=np.float64)
    radius = float(max(radius, 1.0e-8))
    outer = half + radius
    max_outer = float(max(outer.max(), radius, 1.0e-8))
    resolution = int(max(12, min(48, sections + 8)))
    pitch = (2.0 * max_outer) / float(max(8, resolution - 4))

    axes = [
        np.arange(-float(outer[i]) - 2.0 * pitch, float(outer[i]) + 2.5 * pitch, pitch, dtype=np.float64)
        for i in range(3)
    ]
    grid = np.meshgrid(*axes, indexing="ij")
    p = np.stack(grid, axis=-1)
    q = np.abs(p) - half.reshape(1, 1, 1, 3)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    sdf = outside + inside - radius
    occupancy = sdf <= 0.0

    try:
        mesh = trimesh.voxel.ops.matrix_to_marching_cubes(occupancy, pitch=pitch)
        mesh.apply_translation([float(axis[0]) for axis in axes])
        return mesh
    except Exception:
        # Visualization fallback only; SDF remains the analytic rounded-box query.
        return trimesh.creation.box(extents=(2.0 * outer).tolist())


def _primitive_local_mesh(primitive: Dict[str, Any], sections: int = 16) -> trimesh.Trimesh:
    params = primitive["params"]
    if primitive["type"] == "capsule":
        radius = float(params["radius"])
        half_length = float(params["half_length"])
        if half_length <= 1.0e-8:
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        else:
            mesh = trimesh.creation.capsule(radius=radius, height=2.0 * half_length, count=[max(8, sections), max(8, sections)])
            mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2.0, [0.0, 1.0, 0.0]))
        return mesh

    if primitive["type"] == "box":
        half_extents = np.asarray(params["half_extents"], dtype=np.float64)
        return trimesh.creation.box(extents=(2.0 * half_extents).tolist())

    if primitive["type"] == "rounded_box":
        half_extents = np.asarray(params["half_extents"], dtype=np.float64)
        return _rounded_box_mesh(half_extents, radius=float(params["radius"]), sections=sections)

    raise ValueError(f"Unsupported primitive type: {primitive['type']}")


class PrimitiveCollisionModel:
    """Analytic SDF and visualization wrapper for generated primitive specs.

    SDF sign convention:
    - positive: query point is outside every primitive by that distance.
    - negative: query point is inside at least one primitive.
    - hand SDF: minimum SDF over all generated link primitives.
    """

    def __init__(self, robot_model: Any, spec: Dict[str, Any]) -> None:
        self.robot_model = robot_model
        self.spec = spec
        self.primitives = list(spec.get("primitives", []))
        self._local_mesh_cache: dict[tuple[int, int], trimesh.Trimesh] = {}
        self._link_from_primitive = [
            _as_transform(
                prim["link_from_primitive"],
                device=torch.device(robot_model.device),
                dtype=torch.float32,
            )
            for prim in self.primitives
        ]

    def query_sdf(
        self,
        q: Optional[torch.Tensor | np.ndarray],
        points_world: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Return batched hand SDF with shape ``(B, N)``."""
        q_t = _as_batched_q(self.robot_model, q)
        points, batch_size = _as_batched_points(
            points_world,
            batch_size=int(q_t.shape[0]),
            device=torch.device(self.robot_model.device),
        )
        if int(q_t.shape[0]) == 1 and batch_size > 1:
            q_t = q_t.expand(batch_size, -1)

        if not self.primitives:
            return torch.full(
                (batch_size, int(points.shape[1])),
                float("inf"),
                dtype=torch.float32,
                device=points.device,
            )

        self.robot_model.update_status(q_t)
        sdf_values: list[torch.Tensor] = []

        for primitive, link_from_primitive in zip(self.primitives, self._link_from_primitive):
            link_name = str(primitive["link_name"])
            if link_name not in self.robot_model.frame_status:
                continue
            world_from_link = self.robot_model.frame_status[link_name].get_matrix().to(points.device, dtype=points.dtype)
            world_from_primitive = world_from_link @ link_from_primitive.to(points.device, dtype=points.dtype).unsqueeze(0)
            points_primitive = _transform_points_world_to_primitive(points, world_from_primitive)

            params = primitive["params"]
            if primitive["type"] == "capsule":
                sdf = _capsule_sdf(
                    points_primitive,
                    half_length=float(params["half_length"]),
                    radius=float(params["radius"]),
                )
            elif primitive["type"] == "box":
                sdf = _box_sdf(points_primitive, params["half_extents"])
            elif primitive["type"] == "rounded_box":
                sdf = _rounded_box_sdf(
                    points_primitive,
                    params["half_extents"],
                    radius=float(params["radius"]),
                )
            else:
                continue
            sdf_values.append(sdf)

        if not sdf_values:
            return torch.full(
                (batch_size, int(points.shape[1])),
                float("inf"),
                dtype=torch.float32,
                device=points.device,
            )
        return torch.stack(sdf_values, dim=0).min(dim=0).values

    def get_trimesh_q(
        self,
        q: Optional[torch.Tensor | np.ndarray] = None,
        *,
        primitive_type: str | None = None,
        sections: int = 16,
    ) -> Dict[str, Any]:
        """Return a RobotModel-style trimesh dictionary for the first q in the batch."""
        q_t = _as_batched_q(self.robot_model, q)
        if q_t.shape[0] > 1:
            q_t = q_t[:1]
        self.robot_model.update_status(q_t)

        meshes: list[trimesh.Trimesh] = []
        parts: Dict[str, trimesh.Trimesh] = {}

        for prim_idx, (primitive, link_from_primitive) in enumerate(zip(self.primitives, self._link_from_primitive)):
            if primitive_type is not None and primitive["type"] != primitive_type:
                continue
            link_name = str(primitive["link_name"])
            if link_name not in self.robot_model.frame_status:
                continue

            cache_key = (prim_idx, int(sections))
            if cache_key not in self._local_mesh_cache:
                self._local_mesh_cache[cache_key] = _primitive_local_mesh(primitive, sections=sections)
            mesh = self._local_mesh_cache[cache_key].copy()
            mesh.apply_transform(link_from_primitive.detach().cpu().numpy())
            world_from_link = self.robot_model.frame_status[link_name].get_matrix()[0].detach().cpu().numpy()
            mesh.apply_transform(world_from_link)
            part_name = f"{link_name}:{prim_idx}:{primitive['type']}"
            parts[part_name] = mesh
            meshes.append(mesh)

        return {"visual": _merge_meshes(meshes), "parts": parts}
