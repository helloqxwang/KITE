from __future__ import annotations

"""Analytic collision primitives for true parametric MANO hands."""

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import trimesh

from robot_model.collision_primitives import (
    _as_batched_points,
    _as_batched_q,
    _as_transform,
    _box_sdf,
    _capsule_sdf,
    _merge_meshes,
    _primitive_local_mesh,
    _rounded_box_sdf,
    _transform_points_world_to_primitive,
    safe_robot_name,
)
from robot_model.human_hand_model import HumanHandModel




def normalize_mano_betas(betas: Optional[Sequence[float] | np.ndarray | torch.Tensor] = None) -> np.ndarray:
    if betas is None:
        return np.zeros((10,), dtype=np.float32)
    arr = np.asarray(betas, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape != (10,):
        raise ValueError(f"MANO betas must have shape (10,), got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def mano_beta_hash(betas: Optional[Sequence[float] | np.ndarray | torch.Tensor] = None) -> str:
    beta_arr = normalize_mano_betas(betas)
    payload = ",".join(f"{float(v):+.8f}" for v in beta_arr).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:10]


def is_zero_mano_beta(betas: Optional[Sequence[float] | np.ndarray | torch.Tensor] = None) -> bool:
    return bool(np.allclose(normalize_mano_betas(betas), 0.0, atol=1.0e-8))


def mano_collision_primitive_spec_path(
    spec_dir: str | Path,
    *,
    side: str = "right",
    betas: Optional[Sequence[float] | np.ndarray | torch.Tensor] = None,
) -> Path:
    side_s = str(side).strip().lower()
    robot_key = f"mano/{side_s}"
    if is_zero_mano_beta(betas):
        return Path(spec_dir) / f"{safe_robot_name(robot_key)}.json"
    return Path(spec_dir) / f"{safe_robot_name(robot_key)}__beta_{mano_beta_hash(betas)}.json"


def _batched_kabsch_from_canonical_to_current(canonical: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    if canonical.ndim != 2 or canonical.shape[-1] != 3:
        raise ValueError(f"Expected canonical support shape (M,3), got {tuple(canonical.shape)}")
    if current.ndim != 3 or current.shape[-1] != 3:
        raise ValueError(f"Expected current support shape (B,M,3), got {tuple(current.shape)}")
    batch_size = int(current.shape[0])
    transform = torch.eye(4, dtype=current.dtype, device=current.device).unsqueeze(0).repeat(batch_size, 1, 1)
    src_mean = canonical.mean(dim=0)
    dst_mean = current.mean(dim=1)
    if int(canonical.shape[0]) < 3:
        transform[:, :3, 3] = dst_mean - src_mean.unsqueeze(0)
        return transform

    src_centered = canonical - src_mean.unsqueeze(0)
    dst_centered = current - dst_mean.unsqueeze(1)
    cov = torch.einsum("mi,bmj->bij", src_centered, dst_centered)
    u, _, vh = torch.linalg.svd(cov)
    v = vh.transpose(-2, -1)
    r = v @ u.transpose(-2, -1)
    det = torch.linalg.det(r)
    if bool((det < 0.0).any()):
        v_fixed = v.clone()
        v_fixed[det < 0.0, :, -1] *= -1.0
        r = v_fixed @ u.transpose(-2, -1)
    transform[:, :3, :3] = r
    transform[:, :3, 3] = dst_mean - torch.matmul(r, src_mean.view(1, 3, 1)).squeeze(-1)
    return transform


class ManoPrimitiveCollisionModel:
    """Fast analytic SDF wrapper for beta-specific MANO primitive specs."""

    def __init__(self, hand_model: HumanHandModel, spec: Dict[str, Any]) -> None:
        if str(spec.get("model_family", "")) != "mano":
            raise ValueError("ManoPrimitiveCollisionModel requires a spec with model_family='mano'")
        self.hand_model = hand_model
        self.spec = spec
        self.primitives = list(spec.get("primitives", []))
        self._local_mesh_cache: dict[tuple[int, int], trimesh.Trimesh] = {}
        spec_betas = torch.as_tensor(normalize_mano_betas(spec.get("betas")), dtype=torch.float32, device=hand_model.device)
        if not torch.allclose(spec_betas, hand_model.betas.to(spec_betas.device), atol=1.0e-5, rtol=1.0e-5):
            raise ValueError(
                "MANO primitive spec betas do not match the HumanHandModel betas. "
                "Create the hand model with betas from the spec."
            )
        self._canonical_vertices = torch.as_tensor(
            hand_model._canonical_vertices_np,
            dtype=torch.float32,
            device=hand_model.device,
        )
        self._canonical_from_primitive = [
            _as_transform(
                prim["link_from_primitive"],
                device=torch.device(hand_model.device),
                dtype=torch.float32,
            )
            for prim in self.primitives
        ]
        self._support_indices = [
            torch.as_tensor(prim.get("mano_support_vertex_indices", []), dtype=torch.long, device=hand_model.device)
            for prim in self.primitives
        ]

    def _world_from_primitives(self, q_t: torch.Tensor) -> list[torch.Tensor]:
        fk = self.hand_model._run_mano_fk(q_t)
        posed_vertices = fk["vertices"]
        transforms: list[torch.Tensor] = []
        for support_indices, canonical_from_primitive in zip(self._support_indices, self._canonical_from_primitive):
            if int(support_indices.numel()) == 0:
                support_indices = torch.arange(self._canonical_vertices.shape[0], dtype=torch.long, device=posed_vertices.device)
            canonical = self._canonical_vertices[support_indices].to(device=posed_vertices.device, dtype=posed_vertices.dtype)
            current = posed_vertices[:, support_indices, :]
            world_from_canonical = _batched_kabsch_from_canonical_to_current(canonical, current)
            transforms.append(world_from_canonical @ canonical_from_primitive.to(posed_vertices.device, posed_vertices.dtype).unsqueeze(0))
        return transforms

    def query_sdf(
        self,
        q: Optional[torch.Tensor | np.ndarray],
        points_world: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        q_t = _as_batched_q(self.hand_model, q)
        points, batch_size = _as_batched_points(
            points_world,
            batch_size=int(q_t.shape[0]),
            device=torch.device(self.hand_model.device),
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

        sdf_values: list[torch.Tensor] = []
        for primitive, world_from_primitive in zip(self.primitives, self._world_from_primitives(q_t)):
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
        q_t = _as_batched_q(self.hand_model, q)
        if q_t.shape[0] > 1:
            q_t = q_t[:1]
        meshes: list[trimesh.Trimesh] = []
        parts: Dict[str, trimesh.Trimesh] = {}
        for prim_idx, (primitive, world_from_primitive) in enumerate(zip(self.primitives, self._world_from_primitives(q_t))):
            if primitive_type is not None and primitive["type"] != primitive_type:
                continue
            cache_key = (prim_idx, int(sections))
            if cache_key not in self._local_mesh_cache:
                self._local_mesh_cache[cache_key] = _primitive_local_mesh(primitive, sections=sections)
            mesh = self._local_mesh_cache[cache_key].copy()
            mesh.apply_transform(world_from_primitive[0].detach().cpu().numpy())
            part_name = f"{primitive.get('mano_region_name', primitive.get('link_name', 'mano'))}:{prim_idx}:{primitive['type']}"
            parts[part_name] = mesh
            meshes.append(mesh)
        return {"visual": _merge_meshes(meshes), "parts": parts}
