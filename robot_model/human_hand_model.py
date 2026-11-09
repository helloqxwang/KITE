from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import trimesh

from robot_model.contact_surface import (
    compute_gendex_contact_value_source_target_batch as _compute_gendex_contact_value_source_target_batch_shared,
    extract_sparse_contact_support_batch as _extract_sparse_contact_support_batch_shared,
)
from robot_model.object_patches import (
    sample_object_patches_batch as _sample_object_patches_shared,
)
from robot_model.robot_model import (
    _farthest_point_sample_indices_local,
    _resolve_focused_scoring_chunk_size,
    _sample_fixed_size_candidate_indices,
    _sample_uniform_so3_euler_xyz,
    _topk_indices_by_score,
)

try:
    from experiment.manipulation.mano_contact import (
        ManoSurfaceTemplate,
        get_mano_surface_template,
    )
    from experiment.manipulation.mano_utils import (
        _get_mano_layer_cached,
        resolve_mano_manotorch_assets_root,
        resolve_mano_model_root,
    )
except ImportError as exc:
    raise ImportError(
        f"Could not import KITE's MANO utilities from {Path(__file__).resolve()}: {exc}. "
        "From the KITE root, run `python scripts/check_mano_installation.py` "
        "to check the active environment and local MANO files."
    ) from exc


MANO_DOF = 51
MANO_POSE_AXIS_ANGLE_DIMS = 48
MANO_NUM_JOINTS = 16
MANO_DEFAULT_BANK_SIZE = 12
MANO_WRIST_JOINT_LIMIT_RAD = math.pi
MANO_TRANSLATION_LIMIT_M = 1.0
_MANO_LBS_WEIGHT_THRESHOLD = 0.05


# Loose axis-angle safety box on MANO's 15 finger joints (radians). Used only by
# ``_clamp_q_to_limits`` as a sanity backstop — natural poses are generated in the
# anatomical Euler basis (see ``_MANO_ANATOMICAL_EULER_BOUNDS_*``) and composed back
# to axis-angle via ``AxisLayerFK.compose``; those axis-angle values are dense across
# all three components of each joint, so this box is intentionally wide.
_MANO_JOINT_AA_BOUNDS: Tuple[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]], ...] = tuple(
    (
        (-math.pi / 2, math.pi / 2),
        (-math.pi / 2, math.pi / 2),
        (-math.pi / 2, math.pi / 2),
    )
    for _ in range(12)
) + (
    # Thumb CMC needs more headroom (opposition sweep > 60°).
    ((-2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0),) * 3,
    ((-math.pi / 2, math.pi / 2),) * 3,
    ((-math.pi / 2, math.pi / 2),) * 3,
)


# Anatomical-Euler (twist, spread, bend) bounds per MANO finger joint, in degrees.
# Rows follow MANO kintree joints 1..15: fingers in order [index, middle, pinky, ring]
# (matches AxisLayerFK's internal order) each (MCP, PIP, DIP); then thumb (CMC, MCP, IP).
# Values from Yang et al. (CPF, ICCV 2021) via manotorch's ``AnatomyConstraintLossEE``.
# Axes that manotorch locks to zero (finger twist, PIP/DIP spread) are widened to ±3°
# per the survey's note so uniform sampling does not collapse onto a measure-zero
# manifold. Composing a triple in this box to MANO axis-angle via AxisLayerFK yields
# an anatomically plausible pose with zero anatomy loss by construction.
_MANO_ANATOMICAL_EULER_BOUNDS_DEG: Tuple[Tuple[Tuple[float, float], ...], ...] = (
    ((-3.0,  3.0),  (-5.0,  5.0),   (0.0, 90.0)),  # index MCP
    ((-3.0,  3.0),  (-3.0,  3.0),   (0.0, 90.0)),  # index PIP
    ((-3.0,  3.0),  (-3.0,  3.0),   (0.0, 90.0)),  # index DIP
    ((-3.0,  3.0),  (-5.0,  5.0),   (0.0, 90.0)),  # middle MCP
    ((-3.0,  3.0),  (-3.0,  3.0),   (0.0, 90.0)),  # middle PIP
    ((-3.0,  3.0),  (-3.0,  3.0),   (0.0, 90.0)),  # middle DIP
    ((-3.0,  3.0),  (-5.0,  5.0),   (0.0, 90.0)),  # pinky MCP
    ((-3.0,  3.0),  (-3.0,  3.0),   (0.0, 90.0)),  # pinky PIP
    ((-3.0,  3.0),  (-3.0,  3.0),   (0.0, 90.0)),  # pinky DIP
    ((-3.0,  3.0),  (-5.0,  5.0),   (0.0, 90.0)),  # ring MCP
    ((-3.0,  3.0),  (-3.0,  3.0),   (0.0, 90.0)),  # ring PIP
    ((-3.0,  3.0),  (-3.0,  3.0),   (0.0, 90.0)),  # ring DIP
    ((-45.0, 45.0), (-15.0, 45.0),  (0.0, 45.0)),  # thumb CMC
    ((-3.0,  3.0),  (-10.0, 10.0),  (0.0, 90.0)),  # thumb MCP
    ((-3.0,  3.0),  (-3.0,  3.0),   (0.0, 90.0)),  # thumb IP
)
_MANO_ANATOMICAL_EULER_BOUNDS_RAD: Tuple[Tuple[Tuple[float, float], ...], ...] = tuple(
    tuple((math.radians(lo), math.radians(hi)) for lo, hi in axes)
    for axes in _MANO_ANATOMICAL_EULER_BOUNDS_DEG
)


class _PkChainShim:
    """Minimal pk_chain surrogate: only `.get_joint_limits()` is ever called externally."""

    def __init__(self, lower: List[float], upper: List[float]) -> None:
        self._lower = [float(x) for x in lower]
        self._upper = [float(x) for x in upper]

    def get_joint_limits(self) -> Tuple[List[float], List[float]]:
        return list(self._lower), list(self._upper)


def _quat_mul_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _euler_xyz_to_rotvec_torch(euler: torch.Tensor) -> torch.Tensor:
    """Intrinsic XYZ Euler -> axis-angle, avoiding acos' singular gradient."""
    if euler.ndim == 1:
        euler = euler.unsqueeze(0)
    half = 0.5 * euler
    cx, sx = torch.cos(half[:, 0]), torch.sin(half[:, 0])
    cy, sy = torch.cos(half[:, 1]), torch.sin(half[:, 1])
    cz, sz = torch.cos(half[:, 2]), torch.sin(half[:, 2])
    zeros = torch.zeros_like(cx)
    qx = torch.stack([cx, sx, zeros, zeros], dim=-1)
    qy = torch.stack([cy, zeros, sy, zeros], dim=-1)
    qz = torch.stack([cz, zeros, zeros, sz], dim=-1)
    quat = _quat_mul_torch(_quat_mul_torch(qx, qy), qz)
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    quat = torch.where(quat[..., :1] < 0.0, -quat, quat)
    w = quat[..., 0].clamp_min(1.0e-8)
    vec = quat[..., 1:]
    vec_norm = vec.norm(dim=-1)
    angle = 2.0 * torch.atan2(vec_norm, w)
    scale = angle / vec_norm.clamp_min(1.0e-8)
    scale = torch.where(vec_norm > 1.0e-6, scale, 2.0 / w)
    return vec * scale.unsqueeze(-1)


def _compute_mano_vertex_normals_torch(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Vectorized per-vertex normals over a batched posed MANO mesh. Shapes (B,V,3) and (F,3)."""
    if vertices.ndim == 2:
        vertices = vertices.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    B, V, _ = vertices.shape
    tri = vertices[:, faces]  # (B, F, 3, 3)
    face_n = torch.cross(tri[..., 1, :] - tri[..., 0, :], tri[..., 2, :] - tri[..., 0, :], dim=-1)
    vertex_normals = torch.zeros_like(vertices)
    for corner in range(3):
        idx = faces[:, corner].view(1, -1, 1).expand(B, -1, 3)
        vertex_normals.scatter_add_(1, idx, face_n)
    vertex_normals = vertex_normals / vertex_normals.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return vertex_normals[0] if squeeze else vertex_normals


class HumanHandModel:
    """Drop-in model for MANO parametric hands inside the contact-policy pipeline.

    Matches the RobotModel API at the level the dataset and visualizer touch it.
    Each canonical surface-template point is its own virtual "link" so ``link_indices``
    arguments into ``transform_attached_points_normals_batch`` carry the template index
    without requiring changes to the dataset code.
    """

    def __init__(
        self,
        *,
        robot_name: str,
        side: str,
        device: torch.device | str,
        betas: Optional[torch.Tensor | np.ndarray | List[float]] = None,
        num_points: int = 256,
        load_surface_assets: bool = True,
        bank_size: int = MANO_DEFAULT_BANK_SIZE,
    ) -> None:
        self.robot_name = str(robot_name)
        self.side = str(side).strip().lower()
        if self.side not in {"right", "left"}:
            raise ValueError(f"Unsupported MANO side={side!r}; expected 'right' or 'left'")
        self.device = torch.device(device)
        self.num_surface_points = int(num_points)
        self.surface_num_points = int(num_points)
        self.bank_size = int(bank_size)
        self.load_surface_assets = bool(load_surface_assets)
        self.betas = self._normalize_betas(betas)

        mano_model_root = resolve_mano_model_root()
        self._mano_model_root = Path(mano_model_root).expanduser().resolve()
        self._mano_layer = _get_mano_layer_cached(self.side, str(self._mano_model_root), str(self.device))

        # Anatomical-Euler sampler backbone. AxisLayerFK composes (twist, spread, bend)
        # Euler triples per joint into MANO-native axis-angle in the flat-hand reference
        # frame. Our manopth ManoLayer runs with flat_hand_mean=True, so compose(zeros)
        # equals zeros and equals the flat hand — no mean-pose offset needed (verified
        # empirically). manotorch hardcodes ``<root>/models/MANO_<SIDE>.pkl``;
        # ``resolve_mano_manotorch_assets_root`` normalizes both nested (``.../mano/models``)
        # and flat (``.../assets/mano_models``) layouts.
        try:
            from manotorch.axislayer import AxisLayerFK  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HumanHandModel now uses manotorch's AxisLayerFK for anatomical pose "
                f"sampling, but the import failed: {exc}. From the KITE root, "
                "install the pinned version with "
                "`python -m pip install --no-deps -r requirements.txt`."
            ) from exc
        self._axis_layer_fk = AxisLayerFK(
            side=self.side,
            mano_assets_root=str(resolve_mano_manotorch_assets_root(self._mano_model_root)),
        ).to(self.device)
        # AxisLayerFK's reference frame is the flat hand (compose(zeros)=zeros); this only
        # matches our ManoLayer's pose=0 when flat_hand_mean=True (i.e. th_hands_mean=0).
        # If that assumption ever breaks, every anatomical sample drifts by the mean-pose
        # offset silently — guard hard.
        _hands_mean_absmax = float(self._mano_layer.th_hands_mean.abs().max().item())
        if _hands_mean_absmax > 1e-6:
            raise RuntimeError(
                "HumanHandModel's anatomical sampler assumes the manopth ManoLayer was "
                "built with flat_hand_mean=True so pose=0 is the flat hand (AxisLayerFK's "
                f"reference). Got th_hands_mean with abs-max={_hands_mean_absmax:.6f}. "
                "Either build the ManoLayer with flat_hand_mean=True, or add a pose-mean "
                "offset inside sample_random_q_anatomical."
            )
        self._anatomical_euler_bounds = torch.as_tensor(
            _MANO_ANATOMICAL_EULER_BOUNDS_RAD,
            dtype=torch.float32,
            device=self.device,
        )  # (15, 3, 2); last dim is (lo, hi)

        template = get_mano_surface_template(self.side, self._mano_model_root, self.num_surface_points)
        self._template: ManoSurfaceTemplate = template

        # q layout: [0..2] wrist rotation (Euler XYZ, converted to axis-angle at FK),
        # [3..47] 15 finger joints (axis-angle, 3 each), [48..50] global translation.
        self.dof = MANO_DOF
        self.joint_orders = self._build_joint_orders()
        self.base_rotation_indices = [0, 1, 2]
        self.base_translation_indices = [48, 49, 50]
        self.base_pose_indices = [0, 1, 2, 48, 49, 50]
        self.fixed_zero_joint_indices: List[int] = []

        self.mesh_link_names: List[str] = [f"tp_{i}" for i in range(self.num_surface_points)]
        self.excluded_contact_link_names: Set[str] = set()
        self.hidden_visual_link_names: Set[str] = set()

        # Canonical surface template (device tensors).
        self.surface_template_points_local = torch.as_tensor(
            template.canonical_points, dtype=torch.float32, device=self.device
        )
        self.surface_template_normals_local = torch.as_tensor(
            template.canonical_normals, dtype=torch.float32, device=self.device
        )
        self.surface_template_link_indices = torch.arange(
            self.num_surface_points, dtype=torch.long, device=self.device
        )
        self.surface_template_points_canonical_world = self.surface_template_points_local.clone()
        self.surface_template_normals_canonical_world = self.surface_template_normals_local.clone()
        self.surface_graph_points_canonical_world = self.surface_template_points_canonical_world.clone()
        self.surface_graph_normals_canonical_world = self.surface_template_normals_canonical_world.clone()

        # Face / bary buffers for batched FK lookup.
        self._template_face_indices = torch.as_tensor(
            np.asarray(template.canonical_face_indices, dtype=np.int64), dtype=torch.long, device=self.device
        )
        self._template_face_vertex_indices = torch.as_tensor(
            np.asarray(template.canonical_face_vertex_indices, dtype=np.int64), dtype=torch.long, device=self.device
        )
        self._template_barycentric = torch.as_tensor(
            np.asarray(template.canonical_barycentric_coords, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        # Canonical vertices / faces cached on CPU for trimesh / SDF path.
        self._hand_faces_np = self._mano_layer.th_faces.detach().cpu().numpy().astype(np.int64, copy=False)
        self._hand_faces_torch = torch.as_tensor(self._hand_faces_np, dtype=torch.long, device=self.device)
        canonical_vertices_t = self._apply_mano_forward(
            pose=torch.zeros((1, MANO_POSE_AXIS_ANGLE_DIMS), dtype=torch.float32, device=self.device),
            trans=torch.zeros((1, 3), dtype=torch.float32, device=self.device),
        )
        self._canonical_vertices_np = canonical_vertices_t[0].detach().cpu().numpy().astype(np.float32, copy=False)
        self.vertices_original = canonical_vertices_t[0].detach().clone()
        canonical_normals_t = _compute_mano_vertex_normals_torch(canonical_vertices_t[0], self._hand_faces_torch)
        template_face_vertices = self._template_face_vertex_indices.to(device=self.device, dtype=torch.long)
        template_barycentric = self._template_barycentric.to(device=self.device, dtype=torch.float32)
        template_tri_points = canonical_vertices_t[0][template_face_vertices]
        template_tri_normals = canonical_normals_t[template_face_vertices]
        self.surface_template_points_local = (template_tri_points * template_barycentric.unsqueeze(-1)).sum(dim=1)
        self.surface_template_normals_local = (template_tri_normals * template_barycentric.unsqueeze(-1)).sum(dim=1)
        self.surface_template_normals_local = self.surface_template_normals_local / self.surface_template_normals_local.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-8)
        self.surface_template_points_canonical_world = self.surface_template_points_local.clone()
        self.surface_template_normals_canonical_world = self.surface_template_normals_local.clone()
        self.surface_graph_points_canonical_world = self.surface_template_points_canonical_world.clone()
        self.surface_graph_normals_canonical_world = self.surface_template_normals_canonical_world.clone()
        self.meshes_original: Dict[str, trimesh.Trimesh] = {
            "mano": trimesh.Trimesh(
                vertices=self._canonical_vertices_np.copy(),
                faces=self._hand_faces_np.copy(),
                process=False,
            )
        }

        # LBS blend weights + kintree for joint-descendant mask.
        self._mano_weights = self._mano_layer.th_weights.detach().to(device=self.device, dtype=torch.float32)
        self._kintree_parents: List[int] = self._load_kintree_parents()
        self.joint_descendant_link_mask = self._build_joint_descendant_link_mask()

        # Expansion bank (per-template-point local candidates on nearby faces).
        self._build_expansion_bank(bank_size=self.bank_size)

        # Surface graph neighbors.
        self.surface_graph_neighbors: List[List[int]] = self._build_surface_graph_neighbors()
        self.surface_graph_neighbors_strict: List[List[int]] = [list(n) for n in self.surface_graph_neighbors]

        # pk_chain shim + cached frame state.
        lower, upper = self._build_joint_limits()
        self.pk_chain = _PkChainShim(lower, upper)
        self.frame_status: Optional[Dict[str, torch.Tensor]] = None

        # Hand FPS template (built on demand).
        self.hand_fps_local_points: Optional[torch.Tensor] = None
        self.hand_fps_local_normals: Optional[torch.Tensor] = None
        self.hand_fps_link_indices: Optional[torch.Tensor] = None
        self.hand_fps_num_points: int = 0

    # ----------------------------------------------------------------- init helpers
    def _normalize_betas(self, betas: Optional[torch.Tensor | np.ndarray | List[float]]) -> torch.Tensor:
        if betas is None:
            return torch.zeros((10,), dtype=torch.float32, device=self.device)
        betas_t = torch.as_tensor(betas, dtype=torch.float32, device=self.device)
        if betas_t.ndim == 2 and int(betas_t.shape[0]) == 1:
            betas_t = betas_t[0]
        if betas_t.shape != (10,):
            raise ValueError(f"MANO betas must have shape (10,), got {tuple(betas_t.shape)}")
        return betas_t.contiguous()

    def _betas_for_batch(
        self,
        betas: Optional[torch.Tensor | np.ndarray | List[float]],
        batch_size: int,
    ) -> torch.Tensor:
        if betas is None:
            betas_t = self.betas
        else:
            betas_t = torch.as_tensor(betas, dtype=torch.float32, device=self.device)
        if betas_t.ndim == 1:
            if betas_t.shape != (10,):
                raise ValueError(f"MANO betas must have shape (10,), got {tuple(betas_t.shape)}")
            return betas_t.unsqueeze(0).expand(int(batch_size), -1).contiguous()
        if betas_t.ndim == 2 and betas_t.shape[-1] == 10:
            if int(betas_t.shape[0]) == int(batch_size):
                return betas_t.contiguous()
            if int(betas_t.shape[0]) == 1:
                return betas_t.expand(int(batch_size), -1).contiguous()
        raise ValueError(f"MANO betas must have shape (10,), (1,10), or (B,10), got {tuple(betas_t.shape)}")

    def _build_joint_orders(self) -> List[str]:
        orders: List[str] = ["root_rx", "root_ry", "root_rz"]
        for j in range(1, MANO_NUM_JOINTS):
            for axis in ("rx", "ry", "rz"):
                orders.append(f"j{j}_{axis}")
        orders.extend(["trans_x", "trans_y", "trans_z"])
        return orders

    def _build_joint_limits(self) -> Tuple[List[float], List[float]]:
        lower: List[float] = []
        upper: List[float] = []
        for i in range(self.dof):
            if i in self.base_rotation_indices:
                lower.append(-MANO_WRIST_JOINT_LIMIT_RAD)
                upper.append(MANO_WRIST_JOINT_LIMIT_RAD)
            elif i in self.base_translation_indices:
                lower.append(-MANO_TRANSLATION_LIMIT_M)
                upper.append(MANO_TRANSLATION_LIMIT_M)
            else:
                # Finger axis-angle: pose index 3..47 → 15 joints × 3 axes.
                joint_idx = (i - 3) // 3
                axis_idx = (i - 3) % 3
                lo, hi = _MANO_JOINT_AA_BOUNDS[joint_idx][axis_idx]
                lower.append(float(lo))
                upper.append(float(hi))
        return lower, upper

    def _load_kintree_parents(self) -> List[int]:
        parents = list(getattr(self._mano_layer, "kintree_parents", []))
        if len(parents) < MANO_NUM_JOINTS:
            raise RuntimeError(
                f"MANO layer returned kintree_parents of length {len(parents)} (expected >= {MANO_NUM_JOINTS})"
            )
        resolved: List[int] = []
        for raw in parents[:MANO_NUM_JOINTS]:
            p = int(raw)
            if p >= MANO_NUM_JOINTS or p < 0:
                p = -1  # root
            resolved.append(p)
        return resolved

    def _build_joint_descendant_link_mask(self) -> torch.Tensor:
        # per-template-point blend weights over MANO's 16 joints via barycentric interp.
        face_vert_w = self._mano_weights[self._template_face_vertex_indices]  # (N, 3, 16)
        w_tp = (face_vert_w * self._template_barycentric.unsqueeze(-1)).sum(dim=1)  # (N, 16)
        ancestors: List[Set[int]] = []
        for jk in range(MANO_NUM_JOINTS):
            cur = {jk}
            p = self._kintree_parents[jk]
            while 0 <= p < MANO_NUM_JOINTS:
                if p in cur:
                    break
                cur.add(p)
                p = self._kintree_parents[p]
            ancestors.append(cur)
        mask = torch.zeros((self.dof, self.num_surface_points), dtype=torch.bool, device=self.device)
        for p_idx in range(MANO_POSE_AXIS_ANGLE_DIMS):
            jk = p_idx // 3
            eligible = torch.zeros((self.num_surface_points,), dtype=torch.bool, device=self.device)
            for kk in range(MANO_NUM_JOINTS):
                if jk in ancestors[kk]:
                    eligible |= (w_tp[:, kk] > _MANO_LBS_WEIGHT_THRESHOLD)
            mask[p_idx] = eligible
        # Wrist rotation (root joint 0) affects the entire hand via LBS chain; translation too.
        mask[0:3] = True
        mask[48:51] = True
        return mask

    def _build_face_adjacency(self) -> List[List[int]]:
        edges: Dict[Tuple[int, int], List[int]] = {}
        faces = self._hand_faces_np
        for f_idx in range(faces.shape[0]):
            a, b, c = int(faces[f_idx, 0]), int(faces[f_idx, 1]), int(faces[f_idx, 2])
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                edges.setdefault(key, []).append(f_idx)
        neighbors: List[List[int]] = [[] for _ in range(faces.shape[0])]
        for key, face_list in edges.items():
            if len(face_list) < 2:
                continue
            for i_f in face_list:
                for j_f in face_list:
                    if i_f != j_f and j_f not in neighbors[i_f]:
                        neighbors[i_f].append(j_f)
        return neighbors

    def _build_expansion_bank(self, *, bank_size: int) -> None:
        rng = np.random.default_rng(7)
        faces = self._hand_faces_np
        face_neighbors = self._build_face_adjacency()

        canonical_verts = self._canonical_vertices_np
        canonical_vnormals = _compute_mano_vertex_normals_torch(
            torch.as_tensor(canonical_verts, dtype=torch.float32, device=self.device),
            self._hand_faces_torch,
        ).detach().cpu().numpy()

        N = self.num_surface_points
        S = int(bank_size)
        bank_face_vertex_indices = np.zeros((N, S, 3), dtype=np.int64)
        bank_barycentric = np.zeros((N, S, 3), dtype=np.float32)
        bank_points = np.zeros((N, S, 3), dtype=np.float32)
        bank_normals = np.zeros((N, S, 3), dtype=np.float32)

        template_face_indices = self._template_face_indices.detach().cpu().numpy()
        for j in range(N):
            parent_face = int(template_face_indices[j])
            if parent_face < 0 or parent_face >= faces.shape[0]:
                # Degenerate: template point came from a direct vertex (not a face sample).
                # Fall back to using the canonical face_vertex_indices[j] as a triangle
                # with barycentric (1,0,0) replicated — effectively uses the same vertex S times.
                bank_face_vertex_indices[j] = self._template_face_vertex_indices[j].detach().cpu().numpy()
                bank_barycentric[j, :, 0] = 1.0
                for s in range(S):
                    bank_points[j, s] = canonical_verts[bank_face_vertex_indices[j, 0, 0]]
                    bank_normals[j, s] = canonical_vnormals[bank_face_vertex_indices[j, 0, 0]]
                continue
            candidate_faces = [parent_face] + list(face_neighbors[parent_face])
            # Slot 0 is the template point itself so exact-template transforms match precisely.
            template_fv = self._template_face_vertex_indices[j].detach().cpu().numpy().astype(np.int64)
            template_bary = self._template_barycentric[j].detach().cpu().numpy().astype(np.float32)
            bank_face_vertex_indices[j, 0] = template_fv
            bank_barycentric[j, 0] = template_bary
            tri_pts0 = canonical_verts[template_fv]
            tri_nrms0 = canonical_vnormals[template_fv]
            bank_points[j, 0] = (tri_pts0 * template_bary[:, None]).sum(axis=0)
            n0 = (tri_nrms0 * template_bary[:, None]).sum(axis=0)
            bank_normals[j, 0] = n0 / (np.linalg.norm(n0) + 1e-8)
            for s in range(1, S):
                face_idx = int(candidate_faces[s % len(candidate_faces)])
                u = float(rng.random())
                v = float(rng.random())
                sqrt_u = math.sqrt(u)
                b0 = 1.0 - sqrt_u
                b1 = sqrt_u * (1.0 - v)
                b2 = sqrt_u * v
                bary = np.asarray([b0, b1, b2], dtype=np.float32)
                fv = faces[face_idx].astype(np.int64)
                bank_face_vertex_indices[j, s] = fv
                bank_barycentric[j, s] = bary
                tri_pts = canonical_verts[fv]
                tri_nrms = canonical_vnormals[fv]
                bank_points[j, s] = (tri_pts * bary[:, None]).sum(axis=0)
                n = (tri_nrms * bary[:, None]).sum(axis=0)
                n_norm = np.linalg.norm(n) + 1e-8
                bank_normals[j, s] = n / n_norm

        self._bank_face_vertex_indices = torch.as_tensor(
            bank_face_vertex_indices, dtype=torch.long, device=self.device
        )
        self._bank_barycentric = torch.as_tensor(bank_barycentric, dtype=torch.float32, device=self.device)
        self.surface_expansion_bank_points_local = torch.as_tensor(
            bank_points, dtype=torch.float32, device=self.device
        )
        self.surface_expansion_bank_normals_local = torch.as_tensor(
            bank_normals, dtype=torch.float32, device=self.device
        )
        self.surface_expansion_bank_link_indices = (
            torch.arange(N, dtype=torch.long, device=self.device).unsqueeze(1).expand(N, S).contiguous()
        )

    def _build_surface_graph_neighbors(self) -> List[List[int]]:
        # Template-point adjacency via shared canonical faces + face-neighbor faces.
        faces = self._hand_faces_np
        face_neighbors = self._build_face_adjacency()
        template_face_indices = self._template_face_indices.detach().cpu().numpy()
        face_to_template: Dict[int, List[int]] = {}
        for j in range(self.num_surface_points):
            f = int(template_face_indices[j])
            face_to_template.setdefault(f, []).append(j)

        neighbors: List[List[int]] = []
        for j in range(self.num_surface_points):
            parent_face = int(template_face_indices[j])
            cand: Set[int] = set()
            if 0 <= parent_face < faces.shape[0]:
                cand.update(face_to_template.get(parent_face, []))
                for nf in face_neighbors[parent_face]:
                    cand.update(face_to_template.get(int(nf), []))
            cand.discard(j)
            cand_list = list(cand)
            if len(cand_list) < 4:
                # k-NN fallback.
                with torch.no_grad():
                    dists = (
                        self.surface_template_points_local - self.surface_template_points_local[j].unsqueeze(0)
                    ).norm(dim=1)
                topk = torch.topk(dists, k=min(5, self.num_surface_points), largest=False).indices.tolist()
                for k in topk:
                    if int(k) != j:
                        cand.add(int(k))
            neighbors.append(sorted(cand))
        return neighbors

    # ------------------------------------------------------------------- FK core
    def _apply_mano_forward(
        self,
        *,
        pose: torch.Tensor,
        trans: torch.Tensor,
        betas: Optional[torch.Tensor | np.ndarray | List[float]] = None,
    ) -> torch.Tensor:
        # Runs the MANO layer on an axis-angle pose (B, 48); returns posed verts in meters with translation.
        batch_size = int(pose.shape[0])
        betas_t = self._betas_for_batch(betas, batch_size)
        # manopth's upstream ManoLayer returns ``(verts, joints)``; forks return a
        # 3-tuple. Slice to the first two to tolerate either.
        verts_mm, _joints_mm = self._mano_layer(pose, betas_t)[:2]
        verts = verts_mm / 1000.0 + trans.unsqueeze(1)
        return verts

    def _run_mano_fk(
        self,
        q: torch.Tensor,
        betas: Optional[torch.Tensor | np.ndarray | List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if int(q.shape[-1]) != self.dof:
            raise ValueError(f"MANO q must have last dim {self.dof}, got {q.shape[-1]}")
        q = q.to(device=self.device, dtype=torch.float32)
        euler_root = q[:, self.base_rotation_indices]  # (B, 3)
        aa_root = _euler_xyz_to_rotvec_torch(euler_root)
        pose48 = torch.cat([aa_root, q[:, 3:MANO_POSE_AXIS_ANGLE_DIMS]], dim=-1)
        trans = q[:, self.base_translation_indices]
        verts = self._apply_mano_forward(pose=pose48, trans=trans, betas=betas)
        vnorm = _compute_mano_vertex_normals_torch(verts, self._hand_faces_torch)
        return {"vertices": verts, "vertex_normals": vnorm}

    def update_status(self, q: torch.Tensor) -> None:
        fk = self._run_mano_fk(q)
        self.frame_status = fk

    # ------------------------------------------------------------------- surface / transform
    def get_surface_points_normals_batch(
        self,
        q: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q is None:
            q = torch.zeros((1, self.dof), dtype=torch.float32, device=self.device)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        fk = self._run_mano_fk(q)
        verts = fk["vertices"]
        vnorm = fk["vertex_normals"]
        B = int(verts.shape[0])
        N = self.num_surface_points
        face_v = self._template_face_vertex_indices.to(device=self.device, dtype=torch.long)  # (N, 3)
        bary = self._template_barycentric.to(device=self.device, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)
        fv_exp = face_v.view(1, N, 3, 1).expand(B, -1, -1, 3)
        # gather: verts[b, fv[n, c], :] for each (b, n, c)
        gathered_pts = torch.gather(
            verts.unsqueeze(1).expand(-1, N, -1, -1),
            2,
            fv_exp,
        )  # (B, N, 3, 3)
        gathered_nrm = torch.gather(
            vnorm.unsqueeze(1).expand(-1, N, -1, -1),
            2,
            fv_exp,
        )  # (B, N, 3, 3)
        pts = (gathered_pts * bary.unsqueeze(-1)).sum(dim=-2)
        nrm = (gathered_nrm * bary.unsqueeze(-1)).sum(dim=-2)
        nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return pts, nrm

    def get_surface_points_normals(
        self,
        q: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pts, nrm = self.get_surface_points_normals_batch(q=q)
        return pts[0], nrm[0]

    def transform_attached_points_normals_batch(
        self,
        *,
        q: torch.Tensor,
        local_points: torch.Tensor,
        link_indices: torch.Tensor,
        local_normals: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        squeeze = False
        if q.ndim == 1:
            q = q.unsqueeze(0)
            squeeze = True
        if local_points.ndim == 2:
            local_points = local_points.unsqueeze(0)
        if link_indices.ndim == 1:
            link_indices = link_indices.unsqueeze(0)
        if local_normals is not None and local_normals.ndim == 2:
            local_normals = local_normals.unsqueeze(0)

        local_points = local_points.to(device=self.device, dtype=torch.float32)
        link_indices = link_indices.to(device=self.device, dtype=torch.long)
        B, K, _ = local_points.shape
        if K == 0:
            empty = torch.zeros((B, 0, 3), dtype=torch.float32, device=self.device)
            if squeeze:
                return empty[0], (None if local_normals is None else empty[0])
            return empty, (None if local_normals is None else empty)

        parent_idx = link_indices.clamp(min=0)  # invalid slots mapped to template 0
        flat_parents = parent_idx.reshape(-1)
        N_t = self.num_surface_points
        # Bank candidates per parent: (B, K, S, 3) canonical, used for nearest-neighbour lookup
        bank_pts_local = self.surface_expansion_bank_points_local.index_select(0, flat_parents)
        bank_pts_local = bank_pts_local.reshape(B, K, self.bank_size, 3)
        diffs = bank_pts_local - local_points.unsqueeze(-2)
        bank_idx = diffs.pow(2).sum(dim=-1).argmin(dim=-1)  # (B, K)

        bank_fv = self._bank_face_vertex_indices.index_select(0, flat_parents).reshape(B, K, self.bank_size, 3)
        bank_bary = self._bank_barycentric.index_select(0, flat_parents).reshape(B, K, self.bank_size, 3)
        gather_idx = bank_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
        fv_sel = bank_fv.gather(2, gather_idx).squeeze(2)  # (B, K, 3) global vertex indices
        bary_sel = bank_bary.gather(2, gather_idx).squeeze(2)  # (B, K, 3)

        fk = self._run_mano_fk(q)
        verts = fk["vertices"]  # (B, 778, 3)
        vnorm = fk["vertex_normals"]  # (B, 778, 3)
        tri_fv = fv_sel.unsqueeze(-1).expand(-1, -1, -1, 3)  # (B, K, 3, 3)
        tri_pts = torch.gather(verts.unsqueeze(1).expand(-1, K, -1, -1), 2, tri_fv)
        tri_nrm = torch.gather(vnorm.unsqueeze(1).expand(-1, K, -1, -1), 2, tri_fv)
        pts_world = (tri_pts * bary_sel.unsqueeze(-1)).sum(dim=-2)  # (B, K, 3)
        nrm_world: Optional[torch.Tensor]
        if local_normals is not None:
            nrm_world = (tri_nrm * bary_sel.unsqueeze(-1)).sum(dim=-2)
            nrm_world = nrm_world / nrm_world.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            nrm_world = None

        if squeeze:
            return pts_world[0], (None if nrm_world is None else nrm_world[0])
        return pts_world, nrm_world


    # ------------------------------------------------------------------- q helpers
    def _apply_fixed_zero_joint_constraints(self, q: torch.Tensor) -> torch.Tensor:
        return q.clone()

    def _zero_base_translation_in_q(self, q: torch.Tensor) -> torch.Tensor:
        q_out = q.clone()
        if q_out.ndim == 1:
            q_out[self.base_translation_indices] = 0.0
        else:
            q_out[..., self.base_translation_indices] = 0.0
        return q_out

    def _zero_base_pose_in_q(self, q: torch.Tensor) -> torch.Tensor:
        q_out = q.clone()
        if q_out.ndim == 1:
            q_out[self.base_pose_indices] = 0.0
        else:
            q_out[..., self.base_pose_indices] = 0.0
        return q_out

    def set_base_pose_in_q(
        self,
        q: torch.Tensor,
        *,
        base_translation: Optional[torch.Tensor] = None,
        base_rotation_euler: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_out = q.clone().to(device=self.device, dtype=torch.float32)
        squeeze = False
        if q_out.ndim == 1:
            q_out = q_out.unsqueeze(0)
            squeeze = True
        if base_translation is not None:
            trans = base_translation.to(device=self.device, dtype=torch.float32)
            if trans.ndim == 1:
                trans = trans.unsqueeze(0).expand(q_out.shape[0], -1)
            q_out[:, self.base_translation_indices] = trans[:, : len(self.base_translation_indices)]
        if base_rotation_euler is not None:
            euler = base_rotation_euler.to(device=self.device, dtype=torch.float32)
            if euler.ndim == 1:
                euler = euler.unsqueeze(0).expand(q_out.shape[0], -1)
            q_out[:, self.base_rotation_indices] = euler[:, : len(self.base_rotation_indices)]
        return q_out[0] if squeeze else q_out

    def sample_uniform_base_rotation_euler_xyz(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        return _sample_uniform_so3_euler_xyz(
            batch_size=int(batch_size),
            device=self.device,
            generator=generator,
        )

    def sample_random_q_anatomical(
        self,
        batch_size: int = 1,
        *,
        include_base_rotation: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Uniform sample over the manotorch anatomical Euler box, composed back to MANO.

        Draws per-joint (twist, spread, bend) triples uniformly inside
        ``_MANO_ANATOMICAL_EULER_BOUNDS_RAD`` (15 × 3), then calls
        ``AxisLayerFK.compose`` to convert into MANO-native axis-angle. Every sample
        is anatomically plausible with zero anatomy loss by construction, and the
        union of the box covers the full biomechanical ROM of a human hand.

        Base rotation and translation are zeroed by default — callers that need
        randomized base pose (e.g. the q2 contact-gen path) overwrite them via
        ``set_base_pose_in_q``.
        """
        B = int(batch_size)
        if B <= 0:
            return torch.zeros((0, self.dof), dtype=torch.float32, device=self.device)

        # Draw uniforms on the generator's device (may differ from self.device, e.g. when
        # the dataloader worker hands us a CPU generator while the model lives on CUDA).
        gen_device = torch.device(generator.device) if generator is not None else self.device
        lo = self._anatomical_euler_bounds[..., 0].to(device=gen_device).unsqueeze(0)
        hi = self._anatomical_euler_bounds[..., 1].to(device=gen_device).unsqueeze(0)
        u = torch.rand((B, 15, 3), dtype=torch.float32, device=gen_device, generator=generator)
        finger_ee = lo + u * (hi - lo)  # (B, 15, 3)

        full_ee = torch.zeros((B, MANO_NUM_JOINTS, 3), dtype=torch.float32, device=self.device)
        full_ee[:, 1:MANO_NUM_JOINTS, :] = finger_ee.to(self.device)
        pose48 = self._axis_layer_fk.compose(full_ee)  # (B, 16, 3) MANO axis-angle

        q = torch.zeros((B, self.dof), dtype=torch.float32, device=self.device)
        q[:, 3:MANO_POSE_AXIS_ANGLE_DIMS] = pose48[:, 1:MANO_NUM_JOINTS, :].reshape(B, 45)
        if include_base_rotation:
            base_rot = _sample_uniform_so3_euler_xyz(
                batch_size=B, device=gen_device, generator=generator
            )
            q[:, self.base_rotation_indices] = base_rot.to(self.device)
        return q[0] if B == 1 else q

    def sample_random_q(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        *,
        sampler: str = "anatomical",
    ) -> torch.Tensor:
        """Random MANO pose sampler.

        ``sampler="anatomical"`` (default) draws from the manotorch anatomical-Euler
        basis so every pose is biomechanically plausible and the sampler covers the
        full hand ROM. ``sampler="uniform_box"`` falls back to the legacy uniform
        sampler over ``_MANO_JOINT_AA_BOUNDS``; kept for A/B comparison only.
        """
        if sampler == "anatomical":
            return self.sample_random_q_anatomical(
                batch_size=int(batch_size), generator=generator
            )
        if sampler != "uniform_box":
            raise ValueError(
                f"Unsupported sampler={sampler!r}; expected 'anatomical' or 'uniform_box'"
            )
        lower, upper = self.pk_chain.get_joint_limits()
        lower_t = torch.tensor(lower, dtype=torch.float32, device=self.device)
        upper_t = torch.tensor(upper, dtype=torch.float32, device=self.device)
        finite = torch.isfinite(lower_t) & torch.isfinite(upper_t)
        u = torch.rand((int(batch_size), self.dof), device=self.device, generator=generator)
        q = torch.zeros((int(batch_size), self.dof), dtype=torch.float32, device=self.device)
        q[:, finite] = lower_t[finite].unsqueeze(0) + u[:, finite] * (
            upper_t[finite] - lower_t[finite]
        ).unsqueeze(0)
        q = self._zero_base_translation_in_q(q)
        if int(batch_size) == 1:
            return q[0]
        return q

    def get_canonical_q(self) -> torch.Tensor:
        return torch.zeros((self.dof,), dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------- contact-mask ops (overrides)
    def get_joint_active_by_contact_mask(self, mask: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
            squeeze = True
        mask = mask.to(device=self.device).bool()
        jd = self.joint_descendant_link_mask.to(device=self.device).bool()
        joint_active = (mask.float() @ jd.float().t()) > 0
        return joint_active[0] if squeeze else joint_active

    def get_inactive_link_mask_by_contact_mask(
        self,
        mask: torch.Tensor,
        *,
        ignore_base_pose: bool = True,
    ) -> torch.Tensor:
        squeeze = False
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
            squeeze = True
        mask = mask.to(device=self.device).bool()
        # Template point == "link" for MANO, so inactive links = not-in-contact template slots.
        inactive = ~mask
        return inactive[0] if squeeze else inactive

    def mix_q_by_contact_mask(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        squeeze = False
        if q1.ndim == 1:
            q1 = q1.unsqueeze(0)
            q2 = q2.unsqueeze(0)
            mask = mask.unsqueeze(0)
            squeeze = True
        q1 = q1.to(device=self.device, dtype=torch.float32)
        q2 = q2.to(device=self.device, dtype=torch.float32)
        joint_active = self.get_joint_active_by_contact_mask(mask)
        q_out = torch.where(joint_active, q2, q1)
        return q_out[0] if squeeze else q_out


    def get_surface_template_hash(self) -> str:
        return str(self._template.template_hash)


    def get_joint_orders(self) -> List[str]:
        return list(self.joint_orders)

    # ------------------------------------------------------------------- contact extraction wrappers
    @staticmethod
    def _compute_gendex_contact_value_source_target_batch(
        source_points: torch.Tensor,
        source_normals: torch.Tensor,
        target_points: torch.Tensor,
        target_valid_mask: Optional[torch.Tensor] = None,
        align_exp_scale: float = 2.0,
        sigmoid_scale: float = 10.0,
    ) -> torch.Tensor:
        return _compute_gendex_contact_value_source_target_batch_shared(
            source_points=source_points,
            source_normals=source_normals,
            target_points=target_points,
            target_valid_mask=target_valid_mask,
            align_exp_scale=align_exp_scale,
            sigmoid_scale=sigmoid_scale,
        )

    def extract_sparse_contact_support_batch(
        self,
        *,
        source_points: torch.Tensor,
        source_normals: torch.Tensor,
        target_points: torch.Tensor,
        target_valid_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.4,
        contact_count_range: Optional[torch.Tensor | Tuple[int, int]] = None,
        align_exp_scale: float = 2.0,
        sigmoid_scale: float = 10.0,
    ) -> Dict[str, torch.Tensor | Dict[str, int]]:
        return _extract_sparse_contact_support_batch_shared(
            source_points=source_points,
            source_normals=source_normals,
            target_points=target_points,
            target_valid_mask=target_valid_mask,
            threshold=threshold,
            contact_count_range=contact_count_range,
            align_exp_scale=align_exp_scale,
            sigmoid_scale=sigmoid_scale,
        )


    # ------------------------------------------------------------------- dense expansion
    def expand_sparse_contact_support_batch(
        self,
        *,
        q: torch.Tensor,
        sparse_contact_mask: torch.Tensor,
        num_contact_points: int,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        squeeze = False
        if q.ndim == 1:
            q = q.unsqueeze(0)
            squeeze = True
        if sparse_contact_mask.ndim == 1:
            sparse_contact_mask = sparse_contact_mask.unsqueeze(0)
        if sparse_contact_mask.shape[0] != q.shape[0]:
            raise ValueError(
                f"sparse_contact_mask batch ({sparse_contact_mask.shape[0]}) != q batch ({q.shape[0]})"
            )
        if int(sparse_contact_mask.shape[1]) != int(self.surface_num_points):
            raise ValueError(
                f"sparse_contact_mask last dim must match surface_num_points={self.surface_num_points}"
            )

        batch_size = int(q.shape[0])
        target_k = int(num_contact_points)
        if target_k <= 0:
            raise ValueError("num_contact_points must be positive")

        sparse_contact_mask = sparse_contact_mask.to(device=self.device).bool()
        template_link_idx = self.surface_template_link_indices.to(device=self.device, dtype=torch.long)
        bank_points_local = self.surface_expansion_bank_points_local.to(device=self.device, dtype=torch.float32)
        bank_normals_local = self.surface_expansion_bank_normals_local.to(device=self.device, dtype=torch.float32)
        bank_size = int(bank_points_local.shape[1])

        contact_local_points = torch.zeros((batch_size, target_k, 3), dtype=torch.float32, device=self.device)
        contact_local_normals = torch.zeros((batch_size, target_k, 3), dtype=torch.float32, device=self.device)
        contact_link_indices = torch.full((batch_size, target_k), -1, dtype=torch.long, device=self.device)
        contact_parent_sparse_indices = torch.full((batch_size, target_k), -1, dtype=torch.long, device=self.device)
        contact_valid_mask = torch.zeros((batch_size, target_k), dtype=torch.bool, device=self.device)

        sparse_points_world, _ = self.get_surface_points_normals_batch(q=q)
        for batch_idx in range(batch_size):
            active_idx = torch.nonzero(sparse_contact_mask[batch_idx], as_tuple=False).view(-1)
            if int(active_idx.numel()) <= 0:
                continue
            selected_parent = _sample_fixed_size_candidate_indices(
                candidate_indices=active_idx,
                candidate_points_world=sparse_points_world[batch_idx].index_select(0, active_idx),
                target_k=target_k,
                generator=generator,
            )
            bank_idx = torch.randint(
                0,
                bank_size,
                (int(selected_parent.numel()),),
                device=self.device,
                generator=generator,
            )
            contact_parent_sparse_indices[batch_idx, : int(selected_parent.numel())] = selected_parent
            contact_link_indices[batch_idx, : int(selected_parent.numel())] = template_link_idx[selected_parent]
            contact_local_points[batch_idx, : int(selected_parent.numel())] = bank_points_local[selected_parent, bank_idx]
            contact_local_normals[batch_idx, : int(selected_parent.numel())] = bank_normals_local[selected_parent, bank_idx]
            contact_valid_mask[batch_idx, : int(selected_parent.numel())] = True

        contact_points_world, contact_normals_world = self.transform_attached_points_normals_batch(
            q=q,
            local_points=contact_local_points,
            link_indices=contact_link_indices,
            local_normals=contact_local_normals,
        )
        if contact_normals_world is None:
            contact_normals_world = torch.zeros_like(contact_points_world)
        contact_cloud = torch.cat([contact_points_world, contact_normals_world], dim=-1)
        contact_cloud = contact_cloud.masked_fill(~contact_valid_mask.unsqueeze(-1), 0.0)
        contact_local_points = contact_local_points.masked_fill(~contact_valid_mask.unsqueeze(-1), 0.0)
        contact_local_normals = contact_local_normals.masked_fill(~contact_valid_mask.unsqueeze(-1), 0.0)
        contact_link_indices = contact_link_indices.masked_fill(~contact_valid_mask, -1)
        contact_parent_sparse_indices = contact_parent_sparse_indices.masked_fill(~contact_valid_mask, -1)

        out = {
            "contact_cloud": contact_cloud,
            "contact_valid_mask": contact_valid_mask,
            "contact_local_points": contact_local_points,
            "contact_local_normals": contact_local_normals,
            "contact_link_indices": contact_link_indices,
            "contact_parent_sparse_indices": contact_parent_sparse_indices,
        }
        if squeeze:
            return {k: v[0] for k, v in out.items()}
        return out


    def expand_sparse_contact_support_focused_batch(
        self,
        *,
        q: torch.Tensor,
        sparse_contact_mask: torch.Tensor,
        target_points: torch.Tensor,
        target_valid_mask: Optional[torch.Tensor] = None,
        num_contact_points: int,
        threshold: float = 0.4,
        align_exp_scale: float = 2.0,
        sigmoid_scale: float = 10.0,
        generator: Optional[torch.Generator] = None,
        warn_on_empty_pruned: bool = True,
    ) -> Dict[str, torch.Tensor]:
        squeeze = False
        if q.ndim == 1:
            q = q.unsqueeze(0)
            squeeze = True
        if sparse_contact_mask.ndim == 1:
            sparse_contact_mask = sparse_contact_mask.unsqueeze(0)
        if sparse_contact_mask.shape[0] != q.shape[0]:
            raise ValueError(
                f"sparse_contact_mask batch ({sparse_contact_mask.shape[0]}) != q batch ({q.shape[0]})"
            )
        if int(sparse_contact_mask.shape[1]) != int(self.surface_num_points):
            raise ValueError(
                f"sparse_contact_mask last dim must match surface_num_points={self.surface_num_points}"
            )

        if target_points.ndim == 2:
            target_points = target_points.unsqueeze(0)
        elif target_points.ndim == 4:
            bt = int(target_points.shape[0])
            target_points = target_points.reshape(bt, -1, 3)
            if target_valid_mask is not None:
                target_valid_mask = target_valid_mask.reshape(bt, -1)
        elif target_points.ndim != 3:
            raise ValueError(f"target_points must have shape (B, K, 3) or (B, A, P, 3), got {tuple(target_points.shape)}")

        target_points = target_points.to(device=self.device, dtype=torch.float32)
        if target_valid_mask is not None:
            target_valid_mask = target_valid_mask.to(device=self.device, dtype=torch.bool)

        batch_size = int(q.shape[0])
        target_k = int(num_contact_points)
        if target_k <= 0:
            raise ValueError("num_contact_points must be positive")

        sparse_contact_mask = sparse_contact_mask.to(device=self.device).bool()
        template_link_idx = self.surface_template_link_indices.to(device=self.device, dtype=torch.long)
        bank_points_local = self.surface_expansion_bank_points_local.to(device=self.device, dtype=torch.float32)
        bank_normals_local = self.surface_expansion_bank_normals_local.to(device=self.device, dtype=torch.float32)
        bank_size = int(bank_points_local.shape[1])

        contact_cloud = torch.zeros((batch_size, target_k, 6), dtype=torch.float32, device=self.device)
        contact_local_points = torch.zeros((batch_size, target_k, 3), dtype=torch.float32, device=self.device)
        contact_local_normals = torch.zeros((batch_size, target_k, 3), dtype=torch.float32, device=self.device)
        contact_link_indices = torch.full((batch_size, target_k), -1, dtype=torch.long, device=self.device)
        contact_parent_sparse_indices = torch.full((batch_size, target_k), -1, dtype=torch.long, device=self.device)
        contact_valid_mask = torch.zeros((batch_size, target_k), dtype=torch.bool, device=self.device)
        fallback_samples = 0
        active_counts = sparse_contact_mask.long().sum(dim=1)
        max_candidates = int(active_counts.max().item()) * bank_size if batch_size > 0 else 0
        num_targets = int(target_points.shape[1])
        focused_chunk_size = _resolve_focused_scoring_chunk_size(
            batch_size=batch_size,
            max_candidates=max_candidates,
            num_targets=num_targets,
        )

        for chunk_start in range(0, batch_size, focused_chunk_size):
            chunk_end = min(batch_size, chunk_start + focused_chunk_size)
            q_chunk = q[chunk_start:chunk_end]
            target_points_chunk = target_points[chunk_start:chunk_end]
            target_valid_mask_chunk = None if target_valid_mask is None else target_valid_mask[chunk_start:chunk_end]
            active_counts_chunk = active_counts[chunk_start:chunk_end]
            chunk_size = int(chunk_end - chunk_start)
            max_active_chunk = int(active_counts_chunk.max().item()) if chunk_size > 0 else 0
            if max_active_chunk <= 0:
                continue

            padded_active_idx = torch.zeros((chunk_size, max_active_chunk), dtype=torch.long, device=self.device)
            active_valid_mask = torch.zeros((chunk_size, max_active_chunk), dtype=torch.bool, device=self.device)
            for local_idx in range(chunk_size):
                count = int(active_counts_chunk[local_idx].item())
                if count <= 0:
                    continue
                active_idx = torch.nonzero(sparse_contact_mask[chunk_start + local_idx], as_tuple=False).view(-1)
                padded_active_idx[local_idx, :count] = active_idx
                active_valid_mask[local_idx, :count] = True

            flat_active_idx = padded_active_idx.reshape(-1)
            candidate_local_points_chunk = bank_points_local.index_select(0, flat_active_idx).reshape(
                chunk_size, max_active_chunk, bank_size, 3
            ).reshape(chunk_size, -1, 3)
            candidate_local_normals_chunk = bank_normals_local.index_select(0, flat_active_idx).reshape(
                chunk_size, max_active_chunk, bank_size, 3
            ).reshape(chunk_size, -1, 3)
            candidate_link_indices_chunk = template_link_idx.index_select(0, flat_active_idx).reshape(
                chunk_size, max_active_chunk, 1
            ).expand(-1, -1, bank_size).reshape(chunk_size, -1)
            candidate_parent_sparse_indices_chunk = padded_active_idx.unsqueeze(-1).expand(
                -1, -1, bank_size
            ).reshape(chunk_size, -1)
            candidate_valid_mask_chunk = active_valid_mask.unsqueeze(-1).expand(-1, -1, bank_size).reshape(chunk_size, -1)

            candidate_world_points_chunk, candidate_world_normals_chunk = self.transform_attached_points_normals_batch(
                q=q_chunk,
                local_points=candidate_local_points_chunk,
                link_indices=candidate_link_indices_chunk,
                local_normals=candidate_local_normals_chunk,
            )
            if candidate_world_normals_chunk is None:
                candidate_world_normals_chunk = torch.zeros_like(candidate_world_points_chunk)

            candidate_scores_chunk = self._compute_gendex_contact_value_source_target_batch(
                source_points=candidate_world_points_chunk,
                source_normals=candidate_world_normals_chunk,
                target_points=target_points_chunk,
                target_valid_mask=target_valid_mask_chunk,
                align_exp_scale=align_exp_scale,
                sigmoid_scale=sigmoid_scale,
            ).masked_fill(~candidate_valid_mask_chunk, float("-inf"))

            for local_idx in range(chunk_size):
                batch_idx = int(chunk_start + local_idx)
                valid_candidate_idx = torch.nonzero(candidate_valid_mask_chunk[local_idx], as_tuple=False).view(-1)
                if int(valid_candidate_idx.numel()) <= 0:
                    continue
                candidate_scores = candidate_scores_chunk[local_idx].index_select(0, valid_candidate_idx)
                pruned_rel_idx = torch.nonzero(candidate_scores >= float(threshold), as_tuple=False).view(-1)
                if int(pruned_rel_idx.numel()) > 0:
                    focused_candidate_idx = valid_candidate_idx.index_select(0, pruned_rel_idx)
                else:
                    fallback_samples += 1
                    fallback_rel_idx = _topk_indices_by_score(
                        candidate_scores,
                        count=min(int(target_k), int(candidate_scores.numel())),
                    )
                    focused_candidate_idx = valid_candidate_idx.index_select(0, fallback_rel_idx)
                if int(focused_candidate_idx.numel()) <= 0:
                    continue
                candidate_world_points = candidate_world_points_chunk[local_idx]
                selected_idx = _sample_fixed_size_candidate_indices(
                    candidate_indices=focused_candidate_idx,
                    candidate_points_world=candidate_world_points.index_select(0, focused_candidate_idx),
                    target_k=target_k,
                    generator=generator,
                )
                contact_cloud[batch_idx, :, :3] = candidate_world_points.index_select(0, selected_idx)
                contact_cloud[batch_idx, :, 3:] = candidate_world_normals_chunk[local_idx].index_select(0, selected_idx)
                contact_local_points[batch_idx] = candidate_local_points_chunk[local_idx].index_select(0, selected_idx)
                contact_local_normals[batch_idx] = candidate_local_normals_chunk[local_idx].index_select(0, selected_idx)
                contact_link_indices[batch_idx] = candidate_link_indices_chunk[local_idx].index_select(0, selected_idx)
                contact_parent_sparse_indices[batch_idx] = candidate_parent_sparse_indices_chunk[local_idx].index_select(
                    0, selected_idx
                )
                contact_valid_mask[batch_idx] = True

        if warn_on_empty_pruned and fallback_samples > 0:
            warnings.warn(
                "Focused contact pruning removed all second-pass candidates for "
                f"{fallback_samples}/{batch_size} samples; falling back to top-scoring focused candidates.",
                RuntimeWarning,
                stacklevel=2,
            )

        out = {
            "contact_cloud": contact_cloud.masked_fill(~contact_valid_mask.unsqueeze(-1), 0.0),
            "contact_valid_mask": contact_valid_mask,
            "contact_local_points": contact_local_points.masked_fill(~contact_valid_mask.unsqueeze(-1), 0.0),
            "contact_local_normals": contact_local_normals.masked_fill(~contact_valid_mask.unsqueeze(-1), 0.0),
            "contact_link_indices": contact_link_indices.masked_fill(~contact_valid_mask, -1),
            "contact_parent_sparse_indices": contact_parent_sparse_indices.masked_fill(~contact_valid_mask, -1),
        }
        if squeeze:
            return {k: v[0] for k, v in out.items()}
        return out


    # ------------------------------------------------------------------- object patches
    def sample_object_patches_batch(
        self,
        *,
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
        return _sample_object_patches_shared(
            device=self.device,
            hand_points=hand_points,
            hand_normals=hand_normals,
            anchor_indices=anchor_indices,
            anchor_valid_mask=anchor_valid_mask,
            anchor_point_valid_mask=anchor_point_valid_mask,
            generator=generator,
            anchor_shift_min=anchor_shift_min,
            anchor_shift_max=anchor_shift_max,
            max_plane_extent_min=max_plane_extent_min,
            max_plane_extent_max=max_plane_extent_max,
            patch_shift_power=patch_shift_power,
            patch_extent_power=patch_extent_power,
            normal_jitter_max_deg=normal_jitter_max_deg,
            penetration_clearance=penetration_clearance,
        )


    def get_trimesh_q(self, q: torch.Tensor, mode: str = "active") -> Dict[str, object]:
        del mode  # MANO mesh has no active/original split
        if q.ndim == 1:
            q = q.unsqueeze(0)
        verts = self._run_mano_fk(q)["vertices"][0].detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices=verts, faces=self._hand_faces_np, process=False)
        return {"visual": mesh, "parts": {"mano": mesh}}

    def build_hand_fps_template(self, n_points: int) -> None:
        n_points = int(n_points)
        if n_points <= 0:
            self.hand_fps_local_points = None
            self.hand_fps_local_normals = None
            self.hand_fps_link_indices = None
            self.hand_fps_num_points = 0
            return
        template_points = self.surface_template_points_local.detach().to(device=self.device, dtype=torch.float32)
        template_normals = self.surface_template_normals_local.detach().to(device=self.device, dtype=torch.float32)
        template_links = self.surface_template_link_indices.detach().to(device=self.device, dtype=torch.long)
        num_template = int(template_points.shape[0])
        n_sample = min(n_points, num_template)
        if n_sample == num_template:
            fps_indices = torch.arange(num_template, dtype=torch.long, device=self.device)
        else:
            fps_indices = _farthest_point_sample_indices_local(template_points, n_sample)
        self.hand_fps_local_points = template_points[fps_indices].contiguous()
        self.hand_fps_local_normals = template_normals[fps_indices].contiguous()
        self.hand_fps_link_indices = template_links[fps_indices].contiguous()
        self.hand_fps_num_points = int(self.hand_fps_local_points.shape[0])

    def transform_hand_fps_template_batch(
        self,
        q: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.hand_fps_local_points is None:
            raise RuntimeError("build_hand_fps_template must be called before transform_hand_fps_template_batch")
        if q.ndim == 1:
            q = q.unsqueeze(0)
        batch_size = int(q.shape[0])
        local_points = self.hand_fps_local_points.unsqueeze(0).expand(batch_size, -1, -1)
        local_normals = self.hand_fps_local_normals.unsqueeze(0).expand(batch_size, -1, -1)
        link_indices = self.hand_fps_link_indices.unsqueeze(0).expand(batch_size, -1)
        points_world, normals_world = self.transform_attached_points_normals_batch(
            q=q,
            local_points=local_points,
            link_indices=link_indices,
            local_normals=local_normals,
        )
        assert normals_world is not None
        return points_world, normals_world
