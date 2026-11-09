from __future__ import annotations

"""Robot asset loading, kinematics helpers, and contact-surface utilities."""

import contextlib
import hashlib
import io
import json
import math
import os
import random
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation

from pytorch3d.ops import sample_farthest_points
from robot_model.contact_surface import (
    build_topk_mask_from_sorted as _build_topk_mask_from_sorted_shared,
    compute_gendex_contact_value_source_target_batch as _compute_gendex_contact_value_source_target_batch_shared,
    extract_sparse_contact_support_batch as _extract_sparse_contact_support_batch_shared,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS_DIR = ROOT_DIR / "assets"
DEFAULT_MANIPULATOR_LISTS_JSON = DEFAULT_ASSETS_DIR / "manipulator_robot_lists.json"

_PUBLIC_ROBOT_NAME_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "ability_right": (
        "spider/ability/right",
        "spider/ability/ability_hand_right_6dof",
    ),
    "ability_left": (
        "spider/ability/left",
        "spider/ability/ability_hand_left_6dof",
    ),
    "allegro": (
        "ohra/allegro/allegro_hand_right_extended",
        "ohra/allegro/allegro_hand_right",
    ),
    "allegro_right": (
        "ohra/allegro/allegro_hand_right_extended",
        "ohra/allegro/allegro_hand_right",
    ),
    "allegro_left": (
        "ohra/allegro/allegro_hand_left_extended",
        "ohra/allegro/allegro_hand_left",
    ),
    "barrett": (
        "dro/barrett/model_extended",
        "dro/barrett/model",
    ),
    "barrett_right": (
        "dro/barrett/model_extended",
        "dro/barrett/model",
    ),
    "ezgripper": (
        "dro/ezgripper/ezgripper_extended",
        "dro/ezgripper/ezgripper",
    ),
    "ezgripper_right": (
        "dro/ezgripper/ezgripper_extended",
        "dro/ezgripper/ezgripper",
    ),
    "inspire_right": ("spider/inspire/right",),
    "inspire_left": ("spider/inspire/left",),
    "leaphand_right": (
        "dro/leaphand/leap_hand_right_extended",
        "dro/leaphand/leap_hand_right",
    ),
    "robotiq_3finger": (
        "dro/urdf/robotiq_3finger_description_extended",
        "dro/urdf/robotiq_3finger_description",
    ),
    "robotiq_3finger_right": (
        "dro/urdf/robotiq_3finger_description_extended",
        "dro/urdf/robotiq_3finger_description",
    ),
    "shadowhand": (
        "dro/shadowhand/shadow_hand_right_extended",
        "dro/shadowhand/shadow_hand_right",
    ),
    "shadowhand_right": (
        "dro/shadowhand/shadow_hand_right_extended",
        "dro/shadowhand/shadow_hand_right",
    ),
    "metahand_right": ("spider/metahand/right",),
    "metahand_left": ("spider/metahand/left",),
    "schunk_right": (
        "spider/schunk/right",
        "spider/schunk/schunk_hand_right_6dof",
    ),
    "schunk_left": (
        "spider/schunk/left",
        "spider/schunk/schunk_hand_left_6dof",
    ),
    "wujihand_right": ("spider/wujihand/right",),
    "wujihand_left": ("spider/wujihand/left",),
    "xhand_right": (
        "spider/xhand/right",
        "spider/xhand/xhand_right_6dof",
    ),
    "xhand_left": (
        "spider/xhand/left",
        "spider/xhand/xhand_left_6dof",
    ),
}


_MANO_ROBOT_NAME_CANDIDATES: Dict[str, str] = {
    "mano_right": "mano/right",
    "mano_left": "mano/left",
    "mano/right": "mano/right",
    "mano/left": "mano/left",
}

_COMPOSITE_ROBOT_NAME_CANDIDATES: Dict[str, str] = {
    "robotiq_gripper_stick": "composite/robotiq_gripper_stick",
    "robotiq_stick": "composite/robotiq_gripper_stick",
    "gripper_stick": "composite/robotiq_gripper_stick",
    "composite/robotiq_gripper_stick": "composite/robotiq_gripper_stick",
}



def _is_mano_robot_name(robot_name: str) -> Optional[str]:
    """Return canonical 'mano/right' or 'mano/left' for MANO public names, else None."""
    return _MANO_ROBOT_NAME_CANDIDATES.get(_normalize_robot_query(str(robot_name)))


def _stable_int_seed(key: str) -> int:
    """Derive a deterministic integer seed from a stable string key."""
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**31 - 1)


def _normalize_robot_query(robot_name: str) -> str:
    return str(robot_name).strip().lower().replace("-", "_").replace(" ", "_")


_BASE_JOINT_NAME_TOKENS = (
    "root",
    "base",
    "world",
    "global",
    "floating",
    "virtual",
    "forearm",
    "free",
)


def _normalize_joint_name(joint_name: str) -> str:
    return str(joint_name).strip().lower().replace("-", "_").replace(" ", "_")


def _has_joint_name_token(normalized_name: str, token: str) -> bool:
    return (
        normalized_name == token
        or normalized_name.startswith(f"{token}_")
        or normalized_name.endswith(f"_{token}")
        or f"_{token}_" in normalized_name
    )


def _has_any_joint_name_token(normalized_name: str, tokens: Sequence[str]) -> bool:
    return any(_has_joint_name_token(normalized_name, token) for token in tokens)


def _is_base_like_joint_name(normalized_name: str) -> bool:
    return _has_any_joint_name_token(normalized_name, _BASE_JOINT_NAME_TOKENS)


def _is_base_translation_joint_name(normalized_name: str) -> bool:
    direct = {
        "rootx",
        "rooty",
        "rootz",
        "root_x",
        "root_y",
        "root_z",
        "base_x",
        "base_y",
        "base_z",
        "world_x",
        "world_y",
        "world_z",
        "global_x",
        "global_y",
        "global_z",
        "trans_x",
        "trans_y",
        "trans_z",
        "translation_x",
        "translation_y",
        "translation_z",
        "floating_x",
        "floating_y",
        "floating_z",
        "virtual_joint_x",
        "virtual_joint_y",
        "virtual_joint_z",
        "virtual_x",
        "virtual_y",
        "virtual_z",
        "right_pos_x",
        "right_pos_y",
        "right_pos_z",
        "left_pos_x",
        "left_pos_y",
        "left_pos_z",
    }
    if normalized_name in direct:
        return True
    axis_tokens = (
        "tx",
        "ty",
        "tz",
        "pos_x",
        "pos_y",
        "pos_z",
        "trans_x",
        "trans_y",
        "trans_z",
        "translation_x",
        "translation_y",
        "translation_z",
        "free_tx",
        "free_ty",
        "free_tz",
    )
    if not _has_any_joint_name_token(normalized_name, axis_tokens):
        return False
    if _has_any_joint_name_token(normalized_name, ("roll", "pitch", "yaw", "rot", "rx", "ry", "rz")):
        return False
    return _is_base_like_joint_name(normalized_name) or normalized_name.startswith(("right_pos_", "left_pos_"))


def _is_base_rotation_joint_name(normalized_name: str) -> bool:
    direct = {
        "root_roll",
        "root_pitch",
        "root_yaw",
        "base_roll",
        "base_pitch",
        "base_yaw",
        "world_roll",
        "world_pitch",
        "world_yaw",
        "global_roll",
        "global_pitch",
        "global_yaw",
        "floating_roll",
        "floating_pitch",
        "floating_yaw",
        "virtual_joint_roll",
        "virtual_joint_pitch",
        "virtual_joint_yaw",
        "virtual_roll",
        "virtual_pitch",
        "virtual_yaw",
        "right_rot_x",
        "right_rot_y",
        "right_rot_z",
        "left_rot_x",
        "left_rot_y",
        "left_rot_z",
    }
    if normalized_name in direct:
        return True
    axis_tokens = (
        "roll",
        "pitch",
        "yaw",
        "rx",
        "ry",
        "rz",
        "rot_x",
        "rot_y",
        "rot_z",
        "free_rx",
        "free_ry",
        "free_rz",
    )
    if not _has_any_joint_name_token(normalized_name, axis_tokens):
        return False
    return _is_base_like_joint_name(normalized_name) or normalized_name.startswith(("right_rot_", "left_rot_"))


@contextlib.contextmanager
def _temporary_global_seed(seed: int):
    """Temporarily override Python, NumPy, and Torch RNG state."""
    np_state = np.random.get_state()
    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    np.random.seed(int(seed) % (2**32 - 1))
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _build_chain_quiet(urdf_data: bytes | str) -> pk.chain.Chain:
    """Build a PK chain while suppressing verbose parser output."""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        return pk.build_chain_from_urdf(urdf_data)


def _as_mesh(scene_or_mesh):
    """Collapse a trimesh scene to a single mesh when possible."""
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            return None
        return trimesh.util.concatenate(
            tuple(
                trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                for g in scene_or_mesh.geometry.values()
            )
        )
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh
    return None


def farthest_point_sampling(
    pos: torch.Tensor,
    n_sampling: int,
    lengths: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a spatially spread subset of points from batched inputs."""
    if pos.ndim == 2:
        pos = pos.unsqueeze(0)
        squeeze = True
    elif pos.ndim == 3:
        squeeze = False
    else:
        raise ValueError("pos must be shape (N, C) or (B, N, C)")

    n_points = pos.shape[1]
    k = min(int(n_sampling), n_points)
    if k <= 0:
        raise ValueError("n_sampling must be positive")

    if lengths is not None:
        lengths = torch.as_tensor(lengths, dtype=torch.long, device=pos.device).view(-1)
        if int(lengths.numel()) != int(pos.shape[0]):
            raise ValueError(
                f"lengths must have one entry per batch element, got {tuple(lengths.shape)} for batch {pos.shape[0]}"
            )
        lengths = lengths.clamp_min(0).clamp_max(int(n_points))

    if sample_farthest_points is not None:
        sampled_xyz, sampled_idx = sample_farthest_points(
            pos[..., :3].contiguous(),
            lengths=lengths,
            K=k,
        )
        if pos.shape[-1] > 3:
            gather_idx = sampled_idx.clamp_min(0)[..., :, None].expand(*sampled_idx.shape, pos.shape[-1])
            sampled = torch.gather(pos, -2, gather_idx)
            sampled = sampled.masked_fill(sampled_idx[..., None] < 0, 0.0)
        else:
            sampled = sampled_xyz
    else:
        sampled_rows = []
        sampled_idx_rows = []
        for batch_idx in range(int(pos.shape[0])):
            cur_len = int(lengths[batch_idx].item()) if lengths is not None else int(n_points)
            cur_idx = torch.full((k,), -1, dtype=torch.long, device=pos.device)
            cur_sample = torch.zeros((k, pos.shape[-1]), dtype=pos.dtype, device=pos.device)
            if cur_len > 0:
                keep = torch.randperm(cur_len, device=pos.device)[: min(k, cur_len)]
                cur_idx[: int(keep.numel())] = keep
                cur_sample[: int(keep.numel())] = pos[batch_idx, keep]
            sampled_rows.append(cur_sample)
            sampled_idx_rows.append(cur_idx)
        sampled = torch.stack(sampled_rows, dim=0)
        sampled_idx = torch.stack(sampled_idx_rows, dim=0)

    if squeeze:
        sampled = sampled.squeeze(0)
        sampled_idx = sampled_idx.squeeze(0)

    return sampled, sampled_idx


def _topk_indices_by_score(
    scores: torch.Tensor,
    *,
    count: int,
) -> torch.Tensor:
    """Return stable score-sorted indices in descending order."""
    if int(count) <= 0 or int(scores.numel()) <= 0:
        return torch.zeros((0,), dtype=torch.long, device=scores.device)
    scores_cpu = scores.detach().to(dtype=torch.float32, device="cpu").view(-1)
    ordered = sorted(
        range(int(scores_cpu.numel())),
        key=lambda idx: (-float(scores_cpu[idx].item()), int(idx)),
    )
    return torch.tensor(ordered[: int(min(int(count), int(scores_cpu.numel())))], dtype=torch.long, device=scores.device)


def _sample_fixed_size_candidate_indices(
    *,
    candidate_indices: torch.Tensor,
    candidate_points_world: torch.Tensor,
    target_k: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample or repeat candidates to a fixed target size while keeping coverage."""
    num_candidates = int(candidate_indices.numel())
    if num_candidates <= 0:
        return torch.zeros((0,), dtype=torch.long, device=candidate_indices.device)
    if num_candidates >= int(target_k):
        shuffled_order = torch.randperm(num_candidates, device=candidate_indices.device, generator=generator)
        shuffled_idx = candidate_indices.index_select(0, shuffled_order)
        shuffled_points = candidate_points_world.index_select(0, shuffled_order)
        _, fps_idx = farthest_point_sampling(shuffled_points, int(target_k))
        return shuffled_idx.index_select(0, fps_idx.long())

    counts = torch.full(
        (num_candidates,),
        int(target_k) // num_candidates,
        dtype=torch.long,
        device=candidate_indices.device,
    )
    remainder = int(target_k) % num_candidates
    if remainder > 0:
        extra = torch.randperm(num_candidates, device=candidate_indices.device, generator=generator)[:remainder]
        counts[extra] += 1
    selected = candidate_indices.repeat_interleave(counts)
    order = torch.randperm(int(selected.numel()), device=candidate_indices.device, generator=generator)
    return selected.index_select(0, order)


def _resolve_focused_scoring_chunk_size(
    *,
    batch_size: int,
    max_candidates: int,
    num_targets: int,
    max_pairwise_entries: int = 96_000_000,
    max_chunk_size: int = 256,
) -> int:
    """Choose a chunk size that keeps the batched score matrix at a manageable size."""
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(max_candidates) <= 0 or int(num_targets) <= 0:
        return int(batch_size)
    per_sample_entries = int(max_candidates) * int(num_targets)
    if per_sample_entries <= 0:
        return int(batch_size)
    chunk = max(1, int(max_pairwise_entries) // int(per_sample_entries))
    return max(1, min(int(batch_size), int(max_chunk_size), int(chunk)))


def _matrix_to_euler_xyz(matrix: torch.Tensor) -> torch.Tensor:
    device = matrix.device
    euler = Rotation.from_matrix(matrix.detach().cpu().numpy()).as_euler("XYZ")
    return torch.tensor(euler, dtype=torch.float32, device=device)


def _farthest_point_sample_indices_local(points: torch.Tensor, n_sample: int) -> torch.Tensor:
    """Greedy FPS over a single (N, 3) cloud used by the hand-surface template."""
    num_points = int(points.shape[0])
    n_sample = int(n_sample)
    if n_sample <= 0 or num_points == 0:
        return torch.zeros((0,), dtype=torch.long, device=points.device)
    if n_sample >= num_points:
        return torch.arange(num_points, dtype=torch.long, device=points.device)
    indices = torch.zeros((n_sample,), dtype=torch.long, device=points.device)
    distances = torch.full((num_points,), float("inf"), dtype=torch.float32, device=points.device)
    last = 0
    indices[0] = last
    for i in range(1, n_sample):
        d = (points - points[last].unsqueeze(0)).pow(2).sum(dim=-1)
        distances = torch.minimum(distances, d)
        last = int(torch.argmax(distances).item())
        indices[i] = last
    return indices


def _euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    device = euler.device
    mat = Rotation.from_euler("XYZ", euler.detach().cpu().numpy()).as_matrix()
    return torch.tensor(mat, dtype=torch.float32, device=device)


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / torch.norm(v, dim=-1, keepdim=True).clamp_min(1e-8)


def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    x = _normalize(rot6d[..., 0:3])
    y = _normalize(rot6d[..., 3:6])
    a = _normalize(x + y)
    b = _normalize(x - y)
    x = _normalize(a + b)
    y = _normalize(a - b)
    z = _normalize(torch.cross(x, y, dim=-1))
    return torch.stack([x, y, z], dim=-2).mT


def _rot6d_to_euler(rot6d: torch.Tensor) -> torch.Tensor:
    return _matrix_to_euler_xyz(_rot6d_to_matrix(rot6d))


def _quat_xyzw_to_matrix_torch(quat_xyzw: torch.Tensor) -> torch.Tensor:
    if quat_xyzw.ndim != 2 or quat_xyzw.shape[-1] != 4:
        raise ValueError(f"quat_xyzw must have shape (B, 4), got {tuple(quat_xyzw.shape)}")
    quat_xyzw = quat_xyzw / torch.norm(quat_xyzw, dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = quat_xyzw.unbind(dim=-1)
    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w
    return torch.stack(
        [
            torch.stack([ww + xx - yy - zz, 2.0 * (xy - zw), 2.0 * (xz + yw)], dim=-1),
            torch.stack([2.0 * (xy + zw), ww - xx + yy - zz, 2.0 * (yz - xw)], dim=-1),
            torch.stack([2.0 * (xz - yw), 2.0 * (yz + xw), ww - xx - yy + zz], dim=-1),
        ],
        dim=-2,
    )


def _sample_uniform_so3_euler_xyz(
    batch_size: int,
    *,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    batch_size = int(batch_size)
    if batch_size <= 0:
        return torch.zeros((0, 3), dtype=torch.float32, device=device)
    u1 = torch.rand((batch_size,), dtype=torch.float32, device=device, generator=generator)
    u2 = torch.rand((batch_size,), dtype=torch.float32, device=device, generator=generator)
    u3 = torch.rand((batch_size,), dtype=torch.float32, device=device, generator=generator)
    two_pi = float(2.0 * math.pi)
    quat_xyzw = torch.stack(
        [
            torch.sqrt(1.0 - u1) * torch.sin(two_pi * u2),
            torch.sqrt(1.0 - u1) * torch.cos(two_pi * u2),
            torch.sqrt(u1) * torch.sin(two_pi * u3),
            torch.sqrt(u1) * torch.cos(two_pi * u3),
        ],
        dim=-1,
    )
    rot_mats = _quat_xyzw_to_matrix_torch(quat_xyzw)
    return _matrix_to_euler_xyz(rot_mats)


def q_rot6d_to_q_euler(q_rot6d: torch.Tensor) -> torch.Tensor:
    if q_rot6d.shape[-1] < 9:
        return q_rot6d
    return torch.cat([q_rot6d[..., :3], _rot6d_to_euler(q_rot6d[..., 3:9]), q_rot6d[..., 9:]], dim=-1)


def _parse_vec(attr_val: Optional[str], n: int, default: List[float]) -> np.ndarray:
    if attr_val is None:
        return np.array(default, dtype=np.float64)
    vals = np.fromstring(attr_val, sep=" ", dtype=np.float64)
    if vals.size != n:
        return np.array(default, dtype=np.float64)
    return vals


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    return Rotation.from_quat(quat_xyzw).as_matrix()


def _parse_pose_element(elem: ET.Element, angle_in_radian: bool = True) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    pos = _parse_vec(elem.attrib.get("pos"), 3, [0.0, 0.0, 0.0])
    t[:3, 3] = pos

    if "quat" in elem.attrib:
        q = _parse_vec(elem.attrib.get("quat"), 4, [1.0, 0.0, 0.0, 0.0])
        t[:3, :3] = _quat_wxyz_to_matrix(q)
    elif "euler" in elem.attrib:
        e = _parse_vec(elem.attrib.get("euler"), 3, [0.0, 0.0, 0.0])
        if not angle_in_radian:
            e = np.deg2rad(e)
        t[:3, :3] = Rotation.from_euler("xyz", e).as_matrix()
    elif "axisangle" in elem.attrib:
        aa = _parse_vec(elem.attrib.get("axisangle"), 4, [1.0, 0.0, 0.0, 0.0])
        axis = aa[:3]
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-8:
            axis = axis / axis_norm
        angle = aa[3]
        if not angle_in_radian:
            angle = np.deg2rad(angle)
        t[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()

    return t


def _resolve_file_path(path_str: str, base_dir: Path) -> Path:
    p = path_str.replace("package://", "").replace("file://", "")
    candidate = Path(p)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _load_mesh_geometry(mesh_path: Path, scale: Optional[np.ndarray] = None) -> Optional[trimesh.Trimesh]:
    if not mesh_path.exists():
        return None
    try:
        mesh = _as_mesh(trimesh.load(str(mesh_path), force="mesh"))
    except Exception:
        return None
    if mesh is None:
        return None
    if scale is not None:
        mesh.apply_scale(scale)
    return mesh


def _load_urdf_link_geometries(urdf_path: Path, link_names: List[str]) -> Dict[str, trimesh.Trimesh]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = urdf_path.parent

    link_meshes: Dict[str, trimesh.Trimesh] = {}

    for link in root.findall("link"):
        link_name = link.attrib.get("name")
        if link_name not in link_names:
            continue

        meshes: List[trimesh.Trimesh] = []
        for visual in link.findall("visual"):
            origin = visual.find("origin")
            origin_tf = np.eye(4, dtype=np.float64)
            if origin is not None:
                xyz = _parse_vec(origin.attrib.get("xyz"), 3, [0.0, 0.0, 0.0])
                rpy = _parse_vec(origin.attrib.get("rpy"), 3, [0.0, 0.0, 0.0])
                origin_tf[:3, 3] = xyz
                origin_tf[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()

            geometry = visual.find("geometry")
            if geometry is None or len(list(geometry)) == 0:
                continue

            geom_node = list(geometry)[0]
            mesh: Optional[trimesh.Trimesh] = None

            if geom_node.tag.endswith("mesh"):
                fname = geom_node.attrib.get("filename", "")
                scale = _parse_vec(geom_node.attrib.get("scale"), 3, [1.0, 1.0, 1.0])
                mesh_path = _resolve_file_path(fname, urdf_dir)
                mesh = _load_mesh_geometry(mesh_path, scale)
            elif geom_node.tag.endswith("box"):
                size = _parse_vec(geom_node.attrib.get("size"), 3, [0.01, 0.01, 0.01])
                mesh = trimesh.creation.box(extents=size)
            elif geom_node.tag.endswith("sphere"):
                radius = float(geom_node.attrib.get("radius", "0.01"))
                mesh = trimesh.creation.icosphere(radius=radius)
            elif geom_node.tag.endswith("cylinder"):
                radius = float(geom_node.attrib.get("radius", "0.01"))
                length = float(geom_node.attrib.get("length", "0.01"))
                mesh = trimesh.creation.cylinder(radius=radius, height=length)

            if mesh is None:
                continue

            mesh = mesh.copy()
            mesh.apply_transform(origin_tf)
            meshes.append(mesh)

        if meshes:
            if len(meshes) == 1:
                link_meshes[link_name] = meshes[0]
            else:
                link_meshes[link_name] = trimesh.util.concatenate(meshes)

    return link_meshes


def _parse_mjcf_defaults(root: ET.Element) -> Tuple[Dict[str, Dict[str, Dict[str, str]]], Dict[str, Dict[str, str]]]:
    class_defaults: Dict[str, Dict[str, Dict[str, str]]] = {}
    base_defaults = {"geom": {}, "joint": {}}

    default_root = root.find("default")
    if default_root is None:
        return class_defaults, base_defaults

    def walk_default(node: ET.Element, inherited: Dict[str, Dict[str, str]]) -> None:
        current = {"geom": dict(inherited["geom"]), "joint": dict(inherited["joint"])}

        geom_node = node.find("geom")
        if geom_node is not None:
            current["geom"].update(geom_node.attrib)

        joint_node = node.find("joint")
        if joint_node is not None:
            current["joint"].update(joint_node.attrib)

        cls = node.attrib.get("class")
        if cls is not None:
            class_defaults[cls] = {"geom": dict(current["geom"]), "joint": dict(current["joint"])}

        for child_default in node.findall("default"):
            walk_default(child_default, current)

    walk_default(default_root, base_defaults)

    geom_node = default_root.find("geom")
    if geom_node is not None:
        base_defaults["geom"].update(geom_node.attrib)

    joint_node = default_root.find("joint")
    if joint_node is not None:
        base_defaults["joint"].update(joint_node.attrib)

    return class_defaults, base_defaults


def _get_effective_attrs(
    elem: ET.Element,
    kind: str,
    active_childclass: Optional[str],
    class_defaults: Dict[str, Dict[str, Dict[str, str]]],
    base_defaults: Dict[str, Dict[str, str]],
) -> Dict[str, str]:
    attrs: Dict[str, str] = dict(base_defaults[kind])

    if active_childclass and active_childclass in class_defaults:
        attrs.update(class_defaults[active_childclass][kind])

    elem_class = elem.attrib.get("class")
    if elem_class and elem_class in class_defaults:
        attrs.update(class_defaults[elem_class][kind])

    attrs.update(elem.attrib)
    return attrs


def _build_mjcf_mesh_assets(root: ET.Element, xml_path: Path) -> Dict[str, Tuple[Path, np.ndarray]]:
    compiler = root.find("compiler")
    meshdir = ""
    if compiler is not None:
        meshdir = compiler.attrib.get("meshdir", "")

    asset = root.find("asset")
    if asset is None:
        return {}

    mesh_map: Dict[str, Tuple[Path, np.ndarray]] = {}
    for mesh in asset.findall("mesh"):
        mesh_name = mesh.attrib.get("name")
        mesh_file = mesh.attrib.get("file")
        if not mesh_name or not mesh_file:
            continue

        scale = _parse_vec(mesh.attrib.get("scale"), 3, [1.0, 1.0, 1.0])
        mesh_path = Path(mesh_file)
        if not mesh_path.is_absolute():
            if meshdir:
                mesh_path = (xml_path.parent / meshdir / mesh_file).resolve()
            else:
                mesh_path = (xml_path.parent / mesh_file).resolve()

        mesh_map[mesh_name] = (mesh_path, scale)

    return mesh_map


def _create_mjcf_primitive_mesh(geom_type: str, attrs: Dict[str, str]) -> Optional[trimesh.Trimesh]:
    if geom_type == "box":
        size = _parse_vec(attrs.get("size"), 3, [0.01, 0.01, 0.01])
        return trimesh.creation.box(extents=2.0 * size)
    if geom_type == "sphere":
        size = _parse_vec(attrs.get("size"), 1, [0.01])
        return trimesh.creation.icosphere(radius=float(size[0]))
    if geom_type == "cylinder":
        size = _parse_vec(attrs.get("size"), 2, [0.01, 0.01])
        radius, half = float(size[0]), float(size[1])
        return trimesh.creation.cylinder(radius=radius, height=2.0 * half)
    if geom_type == "capsule":
        if "fromto" in attrs:
            fromto = _parse_vec(attrs.get("fromto"), 6, [0, 0, 0, 0, 0, 0])
            p0 = fromto[:3]
            p1 = fromto[3:]
            length = float(np.linalg.norm(p1 - p0))
            size = _parse_vec(attrs.get("size"), 1, [0.01])
            radius = float(size[0])
            mesh = trimesh.creation.capsule(radius=radius, height=max(length, 1e-6))

            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            dir_vec = p1 - p0
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-8:
                dir_unit = dir_vec / norm
                axis = np.cross(z_axis, dir_unit)
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-8:
                    axis /= axis_norm
                    angle = math.acos(float(np.clip(np.dot(z_axis, dir_unit), -1.0, 1.0)))
                    rot = Rotation.from_rotvec(axis * angle).as_matrix()
                else:
                    dot = float(np.dot(z_axis, dir_unit))
                    rot = np.eye(3) if dot > 0 else Rotation.from_euler("x", math.pi).as_matrix()
            else:
                rot = np.eye(3)

            tf = np.eye(4)
            tf[:3, :3] = rot
            tf[:3, 3] = 0.5 * (p0 + p1)
            mesh.apply_transform(tf)
            return mesh

        size = _parse_vec(attrs.get("size"), 2, [0.01, 0.01])
        radius, half = float(size[0]), float(size[1])
        return trimesh.creation.capsule(radius=radius, height=2.0 * half)

    return None


def _mjcf_geom_to_trimesh(
    geom_attrs: Dict[str, str],
    mesh_assets: Dict[str, Tuple[Path, np.ndarray]],
    angle_in_radian: bool,
) -> Optional[trimesh.Trimesh]:
    geom_type = geom_attrs.get("type", "sphere")

    if geom_type == "mesh":
        mesh_name = geom_attrs.get("mesh")
        if not mesh_name or mesh_name not in mesh_assets:
            return None
        mesh_path, mesh_scale = mesh_assets[mesh_name]
        mesh = _load_mesh_geometry(mesh_path, mesh_scale)
        if mesh is None:
            return None
    else:
        mesh = _create_mjcf_primitive_mesh(geom_type, geom_attrs)
        if mesh is None:
            return None

    tf = _parse_pose_element(_dict_to_elem(geom_attrs), angle_in_radian=angle_in_radian)
    mesh = mesh.copy()
    mesh.apply_transform(tf)
    return mesh


def _dict_to_elem(attrs: Dict[str, str]) -> ET.Element:
    e = ET.Element("tmp")
    e.attrib.update(attrs)
    return e


def _safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def _matrix_to_xyz_rpy(tf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xyz = tf[:3, 3]
    rpy = Rotation.from_matrix(tf[:3, :3]).as_euler("xyz")
    return xyz, rpy


def _build_pk_and_meshes_from_mjcf(xml_path: Path) -> Tuple[pk.chain.Chain, Dict[str, trimesh.Trimesh]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    angle_mode = "degree"
    if compiler is not None:
        angle_mode = compiler.attrib.get("angle", "degree").lower()
    angle_in_radian = angle_mode == "radian"

    class_defaults, base_defaults = _parse_mjcf_defaults(root)
    mesh_assets = _build_mjcf_mesh_assets(root, xml_path)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF has no <worldbody>: {xml_path}")

    urdf_robot = ET.Element("robot", attrib={"name": root.attrib.get("model", xml_path.stem)})
    ET.SubElement(urdf_robot, "link", attrib={"name": "world"})

    mesh_records: Dict[str, List[trimesh.Trimesh]] = {}
    joint_id = 0

    def add_link(name: str) -> None:
        ET.SubElement(urdf_robot, "link", attrib={"name": name})

    def add_joint(
        name: str,
        parent: str,
        child: str,
        jtype: str,
        origin_tf: np.ndarray,
        axis: Optional[np.ndarray] = None,
        limit: Optional[Tuple[float, float]] = None,
    ) -> None:
        joint_el = ET.SubElement(urdf_robot, "joint", attrib={"name": name, "type": jtype})
        ET.SubElement(joint_el, "parent", attrib={"link": parent})
        ET.SubElement(joint_el, "child", attrib={"link": child})

        xyz, rpy = _matrix_to_xyz_rpy(origin_tf)
        ET.SubElement(
            joint_el,
            "origin",
            attrib={
                "xyz": f"{xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g}",
                "rpy": f"{rpy[0]:.9g} {rpy[1]:.9g} {rpy[2]:.9g}",
            },
        )

        if axis is not None and jtype in ("revolute", "prismatic"):
            ET.SubElement(joint_el, "axis", attrib={"xyz": f"{axis[0]:.9g} {axis[1]:.9g} {axis[2]:.9g}"})

        if jtype in ("revolute", "prismatic"):
            if limit is None:
                limit = (-math.pi, math.pi) if jtype == "revolute" else (-1.0, 1.0)
            lower, upper = float(limit[0]), float(limit[1])
            if lower > upper:
                lower, upper = upper, lower
            ET.SubElement(
                joint_el,
                "limit",
                attrib={
                    "lower": f"{lower:.9g}",
                    "upper": f"{upper:.9g}",
                    "effort": "1000",
                    "velocity": "1000",
                },
            )

    def collect_body_meshes(body: ET.Element, link_name: str, active_childclass: Optional[str]) -> None:
        body_meshes: List[trimesh.Trimesh] = []
        fallback_meshes: List[trimesh.Trimesh] = []

        for geom in body.findall("geom"):
            attrs = _get_effective_attrs(
                geom,
                kind="geom",
                active_childclass=active_childclass,
                class_defaults=class_defaults,
                base_defaults=base_defaults,
            )
            m = _mjcf_geom_to_trimesh(attrs, mesh_assets, angle_in_radian=angle_in_radian)
            if m is None:
                continue

            name_l = attrs.get("name", "").lower()
            is_collision = ("collision" in name_l) or (attrs.get("group") == "3")
            fallback_meshes.append(m)
            if not is_collision:
                body_meshes.append(m)

        use_meshes = body_meshes if body_meshes else fallback_meshes
        if use_meshes:
            mesh_records[link_name] = use_meshes

    def parse_body(body: ET.Element, parent_link: str, inherited_childclass: Optional[str]) -> None:
        nonlocal joint_id

        body_name = _safe_name(body.attrib.get("name", f"body_{joint_id}"))
        body_tf = _parse_pose_element(body, angle_in_radian=angle_in_radian)

        active_childclass = body.attrib.get("childclass", inherited_childclass)

        joints = body.findall("joint")
        freejoint = body.find("freejoint")

        if freejoint is not None:
            free_joint_names = [
                f"{body_name}_free_tx",
                f"{body_name}_free_ty",
                f"{body_name}_free_tz",
                f"{body_name}_free_rx",
                f"{body_name}_free_ry",
                f"{body_name}_free_rz",
            ]
            free_axes = [
                ("prismatic", np.array([1.0, 0.0, 0.0])),
                ("prismatic", np.array([0.0, 1.0, 0.0])),
                ("prismatic", np.array([0.0, 0.0, 1.0])),
                ("revolute", np.array([1.0, 0.0, 0.0])),
                ("revolute", np.array([0.0, 1.0, 0.0])),
                ("revolute", np.array([0.0, 0.0, 1.0])),
            ]

            chain_parent = parent_link
            for idx, (jname, (jtype, axis)) in enumerate(zip(free_joint_names, free_axes)):
                child = body_name if idx == len(free_joint_names) - 1 else f"{body_name}__free_{idx}"
                add_link(child)
                add_joint(
                    name=jname,
                    parent=chain_parent,
                    child=child,
                    jtype=jtype,
                    origin_tf=body_tf if idx == 0 else np.eye(4),
                    axis=axis,
                    limit=(-2.0, 2.0) if jtype == "prismatic" else (-math.pi, math.pi),
                )
                chain_parent = child

            current_link = body_name
        elif not joints:
            add_link(body_name)
            add_joint(
                name=f"{parent_link}_to_{body_name}_fixed",
                parent=parent_link,
                child=body_name,
                jtype="fixed",
                origin_tf=body_tf,
            )
            current_link = body_name
        else:
            supported_joints = []
            for joint in joints:
                attrs = _get_effective_attrs(
                    joint,
                    kind="joint",
                    active_childclass=active_childclass,
                    class_defaults=class_defaults,
                    base_defaults=base_defaults,
                )
                mj_type = attrs.get("type", "hinge")
                if mj_type in ("hinge", "slide"):
                    supported_joints.append((joint, attrs))

            if not supported_joints:
                add_link(body_name)
                add_joint(
                    name=f"{parent_link}_to_{body_name}_fixed",
                    parent=parent_link,
                    child=body_name,
                    jtype="fixed",
                    origin_tf=body_tf,
                )
                current_link = body_name
            else:
                chain_parent = parent_link
                for idx, (_, attrs) in enumerate(supported_joints):
                    mj_type = attrs.get("type", "hinge")
                    jtype = "revolute" if mj_type == "hinge" else "prismatic"
                    jname = _safe_name(attrs.get("name", f"joint_{joint_id}"))
                    joint_id += 1

                    child = body_name if idx == len(supported_joints) - 1 else f"{body_name}__joint_{idx}"
                    add_link(child)

                    axis = _parse_vec(attrs.get("axis"), 3, [0.0, 0.0, 1.0])
                    axis_norm = np.linalg.norm(axis)
                    if axis_norm > 1e-8:
                        axis = axis / axis_norm

                    limit = None
                    if "range" in attrs:
                        rg = _parse_vec(attrs.get("range"), 2, [0.0, 0.0])
                        limit = (float(rg[0]), float(rg[1]))
                        if not angle_in_radian and jtype == "revolute":
                            limit = (math.radians(limit[0]), math.radians(limit[1]))

                    add_joint(
                        name=jname,
                        parent=chain_parent,
                        child=child,
                        jtype=jtype,
                        origin_tf=body_tf if idx == 0 else np.eye(4),
                        axis=axis,
                        limit=limit,
                    )
                    chain_parent = child

                current_link = body_name

        collect_body_meshes(body, current_link, active_childclass)

        for child_body in body.findall("body"):
            parse_body(child_body, current_link, active_childclass)

    top_bodies = worldbody.findall("body")
    if not top_bodies:
        raise ValueError(f"MJCF has no body in worldbody: {xml_path}")

    for top_body in top_bodies:
        parse_body(top_body, "world", None)

    urdf_text = ET.tostring(urdf_robot, encoding="unicode")
    chain = _build_chain_quiet(urdf_text)

    mesh_dict: Dict[str, trimesh.Trimesh] = {}
    link_names = set(chain.get_link_names())
    for link_name, meshes in mesh_records.items():
        if link_name not in link_names or not meshes:
            continue
        if len(meshes) == 1:
            mesh_dict[link_name] = meshes[0]
        else:
            mesh_dict[link_name] = trimesh.util.concatenate(meshes)

    return chain, mesh_dict


def discover_robot_assets(assets_dir: Path = DEFAULT_ASSETS_DIR) -> Dict[str, Dict[str, str]]:
    """Index URDF and MJCF robot assets under the configured assets directory."""
    assets_dir = Path(assets_dir)
    all_assets: Dict[str, Dict[str, str]] = {}

    dro_root = assets_dir / "dro_urdf_robot"
    ohra_root = assets_dir / "ohra_urdf_robot"
    spider_root = assets_dir / "spider_robots"

    if dro_root.exists():
        for urdf in sorted(dro_root.rglob("*.urdf")):
            if urdf.stem.endswith("_nobase") or urdf.stem.endswith("_floating"):
                continue
            rel = urdf.relative_to(assets_dir)
            folder = urdf.parent.name
            key = f"dro/{folder}/{urdf.stem}"
            all_assets[key] = {
                "name": key,
                "format": "urdf",
                "path": str(rel),
                "source": "dro",
            }

    if ohra_root.exists():
        for urdf in sorted(ohra_root.rglob("*.urdf")):
            if urdf.stem.endswith("_nobase") or urdf.stem.endswith("_floating"):
                continue
            rel = urdf.relative_to(assets_dir)
            folder = urdf.parent.name
            key = f"ohra/{folder}/{urdf.stem}"
            all_assets[key] = {
                "name": key,
                "format": "urdf",
                "path": str(rel),
                "source": "ohra",
            }

    if spider_root.exists():
        for urdf in sorted(spider_root.rglob("*.urdf")):
            if urdf.stem.endswith("_nobase") or urdf.stem.endswith("_floating"):
                continue
            rel = urdf.relative_to(assets_dir)
            folder = urdf.parent.name
            key = f"spider/{folder}/{urdf.stem}"
            all_assets[key] = {
                "name": key,
                "format": "urdf",
                "path": str(rel),
                "source": "spider",
            }

        for xml in sorted(spider_root.rglob("*.xml")):
            if xml.name == "bimanual.xml":
                continue
            if xml.stem.endswith("_nobase"):
                continue
            try:
                root = ET.parse(xml).getroot()
                worldbody = root.find("worldbody")
                if worldbody is None or worldbody.find("body") is None:
                    continue
            except Exception:
                continue

            #### HARD CODE: skip wuji xml here.
            if "wuji" in xml.parent.name:
                continue

            rel = xml.relative_to(assets_dir)
            folder = xml.parent.name
            key = f"spider/{folder}/{xml.stem}"
            all_assets[key] = {
                "name": key,
                "format": "xml",
                "path": str(rel),
                "source": "spider",
            }

    return dict(sorted(all_assets.items(), key=lambda kv: kv[0]))


def resolve_robot_model_name(
    robot_name: str,
    *,
    assets_dir: Path = DEFAULT_ASSETS_DIR,
    meta: Dict[str, Dict[str, str]] | None = None,
    hint: str | None = None,
) -> str:
    """Resolve public robot names and short aliases to concrete asset keys."""
    mano_canonical = _is_mano_robot_name(robot_name)
    if mano_canonical is not None:
        return mano_canonical
    composite_canonical = _COMPOSITE_ROBOT_NAME_CANDIDATES.get(_normalize_robot_query(str(robot_name)))
    if composite_canonical is not None:
        return composite_canonical
    meta = discover_robot_assets(assets_dir) if meta is None else meta
    asset_names = set(meta.keys())

    if hint is not None:
        hint_key = _normalize_robot_query(hint)
        if hint_key in asset_names:
            return hint_key

    query = _normalize_robot_query(robot_name)
    if query in asset_names:
        return query

    for candidate in _PUBLIC_ROBOT_NAME_CANDIDATES.get(query, ()):
        if candidate in asset_names:
            return candidate

    suffix_candidates: List[str] = []
    if query.endswith("_right"):
        suffix_candidates.append(f"spider/{query[:-6]}/right")
    elif query.endswith("_left"):
        suffix_candidates.append(f"spider/{query[:-5]}/left")
    for candidate in suffix_candidates:
        if candidate in asset_names:
            return candidate

    candidates = [k for k in asset_names if k.endswith(f"/{query}") or query in k]
    if len(candidates) == 0:
        raise KeyError(
            f"Robot '{robot_name}' not found. Available count={len(meta)}. "
            f"Example keys: {list(meta.keys())[:8]}"
        )
    return _pick_best(candidates, meta) or candidates[0]


def _is_humanoid_asset_name(robot_name: str) -> bool:
    name = robot_name.lower()
    keywords = ["unitree_g1", "unitree_h1", "h1_2", "booster_t1", "fourier_n1", "humanoid"]
    return any(k in name for k in keywords)


def _robot_kind_key(robot_name: str) -> str:
    parts = robot_name.split("/")
    if len(parts) >= 2:
        return parts[1].lower()
    return robot_name.lower()


def _robot_side(robot_name: str) -> str:
    name = robot_name.lower()
    right_markers = ["_right", "/right", " right", "right_", "_r_", "_r", "r_"]
    left_markers = ["_left", "/left", " left", "left_", "_l_", "_l", "l_"]

    if any(m in name for m in right_markers):
        return "right"
    if any(m in name for m in left_markers):
        return "left"
    return "neutral"


def _base_removal_targets(robot_name: str) -> Optional[Dict[str, List[str]]]:
    name = robot_name.lower()
    if "ezgripper" in name:
        return {"urdf_links": ["base_link"], "xml_bodies": ["base_link", "base"]}
    if "shadow" in name:
        return {"urdf_links": ["forearm"], "xml_bodies": ["forearm"]}
    return None


def _recompose_pose(parent_tf: np.ndarray, child_elem: ET.Element, angle_in_radian: bool) -> None:
    child_tf = _parse_pose_element(child_elem, angle_in_radian=angle_in_radian)
    new_tf = parent_tf @ child_tf

    pos = new_tf[:3, 3]
    quat_xyzw = Rotation.from_matrix(new_tf[:3, :3]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)

    child_elem.attrib["pos"] = f"{pos[0]:.9g} {pos[1]:.9g} {pos[2]:.9g}"
    child_elem.attrib["quat"] = f"{quat_wxyz[0]:.9g} {quat_wxyz[1]:.9g} {quat_wxyz[2]:.9g} {quat_wxyz[3]:.9g}"
    child_elem.attrib.pop("euler", None)
    child_elem.attrib.pop("axisangle", None)


def ensure_base_removed_asset(
    robot_name: str,
    robot_path: Path,
    force_override: bool = False,
) -> Path:
    """
    Create and reuse base-removed urdf/xml assets for selected hands.

    Supported targets:
    - ezgripper: remove `base_link`
    - shadow hand: remove `forearm`

    If generated file exists, it is reused unless `force_override=True`.
    """
    robot_path = Path(robot_path)
    targets = _base_removal_targets(robot_name)
    if targets is None:
        return robot_path

    out_path = robot_path.with_name(f"{robot_path.stem}_nobase{robot_path.suffix}")
    if out_path.exists() and (not force_override):
        return out_path

    suffix = robot_path.suffix.lower()
    if suffix == ".urdf":
        remove_links = set(targets["urdf_links"])
        tree = ET.parse(robot_path)
        root = tree.getroot()

        def _joint_parent_child(joint_el: ET.Element) -> Tuple[Optional[str], Optional[str]]:
            parent_el = joint_el.find("parent")
            child_el = joint_el.find("child")
            parent = parent_el.attrib.get("link") if parent_el is not None else None
            child = child_el.attrib.get("link") if child_el is not None else None
            return parent, child

        # Rewire children of removed links to removed-link parent and drop incoming joints.
        changed = True
        while changed:
            changed = False
            for joint_el in list(root.findall("joint")):
                parent, child = _joint_parent_child(joint_el)
                if child not in remove_links:
                    continue

                for other in list(root.findall("joint")):
                    if other is joint_el:
                        continue
                    p2_el = other.find("parent")
                    if p2_el is not None and p2_el.attrib.get("link") == child and parent is not None:
                        p2_el.attrib["link"] = parent

                root.remove(joint_el)
                changed = True

        # Remove joints that still reference removed links.
        for joint_el in list(root.findall("joint")):
            parent, child = _joint_parent_child(joint_el)
            if parent in remove_links or child in remove_links:
                root.remove(joint_el)

        # Remove target links.
        for link_el in list(root.findall("link")):
            if link_el.attrib.get("name") in remove_links:
                root.remove(link_el)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(out_path, encoding="utf-8", xml_declaration=True)
        return out_path

    if suffix == ".xml":
        remove_bodies = {_safe_name(x) for x in targets["xml_bodies"]}
        tree = ET.parse(robot_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            return robot_path

        compiler = root.find("compiler")
        angle_mode = compiler.attrib.get("angle", "degree").lower() if compiler is not None else "degree"
        angle_in_radian = angle_mode == "radian"

        def process_parent(parent_elem: ET.Element) -> None:
            for body in list(parent_elem.findall("body")):
                process_parent(body)
                body_name = _safe_name(body.attrib.get("name", ""))
                if body_name not in remove_bodies:
                    continue

                body_tf = _parse_pose_element(body, angle_in_radian=angle_in_radian)
                for child_body in list(body.findall("body")):
                    _recompose_pose(body_tf, child_body, angle_in_radian=angle_in_radian)
                    parent_elem.append(child_body)
                    body.remove(child_body)

                parent_elem.remove(body)

        process_parent(worldbody)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(out_path, encoding="utf-8", xml_declaration=True)
        return out_path

    return robot_path


def _requires_virtual_floating_base(robot_name: str) -> bool:
    name = _normalize_robot_query(robot_name)
    return "wujihand" in name


def ensure_virtual_floating_base_asset(
    robot_name: str,
    robot_path: Path,
    force_override: bool = False,
    *,
    output_path: Optional[Path] = None,
) -> Path:
    """Wrap selected fixed-base URDF hands with an explicit 6-DoF virtual base."""
    robot_path = Path(robot_path)
    if robot_path.suffix.lower() != ".urdf" or not _requires_virtual_floating_base(robot_name):
        return robot_path
    if robot_path.stem.endswith("_floating"):
        return robot_path

    out_path = (
        Path(output_path) if output_path is not None
        else robot_path.with_name(f"{robot_path.stem}_floating{robot_path.suffix}")
    )
    if out_path.exists() and (not force_override):
        return out_path

    tree = ET.parse(robot_path)
    root = tree.getroot()

    # A wrapper outside the asset directory must keep all mesh references valid.
    if out_path.parent.resolve() != robot_path.parent.resolve():
        for mesh in root.findall(".//mesh"):
            source_mesh = _resolve_file_path(mesh.attrib["filename"], robot_path.parent)
            mesh.set("filename", os.path.relpath(source_mesh, out_path.parent))

    link_names = [
        str(link_el.attrib.get("name"))
        for link_el in root.findall("link")
        if str(link_el.attrib.get("name", "")).strip() != ""
    ]
    child_links = set()
    for joint_el in root.findall("joint"):
        child_el = joint_el.find("child")
        child_name = None if child_el is None else child_el.attrib.get("link")
        if child_name is not None and str(child_name).strip() != "":
            child_links.add(str(child_name))
    root_links = [name for name in link_names if name not in child_links]
    if len(root_links) != 1:
        raise ValueError(
            f"Expected exactly one URDF root link for floating-base wrapping, got {root_links} in {robot_path}"
        )
    original_root_link = str(root_links[0])

    virtual_links = [
        "virtual_base_link",
        "virtual_joint_x_link",
        "virtual_joint_y_link",
        "virtual_joint_z_link",
        "virtual_joint_roll_link",
        "virtual_joint_pitch_link",
    ]
    for link_name in virtual_links:
        if link_name not in link_names:
            ET.SubElement(root, "link", attrib={"name": link_name})

    joint_specs = [
        ("virtual_joint_x", "prismatic", "virtual_base_link", "virtual_joint_x_link", "1 0 0", (-2.0, 2.0)),
        ("virtual_joint_y", "prismatic", "virtual_joint_x_link", "virtual_joint_y_link", "0 1 0", (-2.0, 2.0)),
        ("virtual_joint_z", "prismatic", "virtual_joint_y_link", "virtual_joint_z_link", "0 0 1", (-2.0, 2.0)),
        (
            "virtual_joint_roll",
            "revolute",
            "virtual_joint_z_link",
            "virtual_joint_roll_link",
            "1 0 0",
            (-math.pi, math.pi),
        ),
        (
            "virtual_joint_pitch",
            "revolute",
            "virtual_joint_roll_link",
            "virtual_joint_pitch_link",
            "0 1 0",
            (-math.pi, math.pi),
        ),
        (
            "virtual_joint_yaw",
            "revolute",
            "virtual_joint_pitch_link",
            original_root_link,
            "0 0 1",
            (-math.pi, math.pi),
        ),
    ]

    for joint_name, joint_type, parent_link, child_link, axis_xyz, limit in joint_specs:
        joint_el = ET.SubElement(root, "joint", attrib={"name": joint_name, "type": joint_type})
        ET.SubElement(joint_el, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(joint_el, "parent", attrib={"link": parent_link})
        ET.SubElement(joint_el, "child", attrib={"link": child_link})
        ET.SubElement(joint_el, "axis", attrib={"xyz": axis_xyz})
        ET.SubElement(
            joint_el,
            "limit",
            attrib={
                "lower": f"{float(limit[0]):.9g}",
                "upper": f"{float(limit[1]):.9g}",
                "effort": "1000",
                "velocity": "1000",
            },
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def _candidate_priority(robot_name: str, meta: Dict[str, Dict[str, str]]) -> Tuple[int, int, int, int]:
    item = meta[robot_name]
    fmt = item.get("format", "").lower()
    src = item.get("source", "").lower()
    src_score = 2 if src == "ohra" else 1 if src == "spider" else 0
    extended_score = 1 if src == "ohra" and robot_name.endswith("_extended") else 0
    fmt_score = 1 if fmt == "xml" else 0
    return (src_score, extended_score, fmt_score, -len(robot_name))


def _pick_best(candidates: List[str], meta: Dict[str, Dict[str, str]]) -> Optional[str]:
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda x: (_candidate_priority(x, meta), x), reverse=True)
    return candidates[0]


def _build_link_tree_from_urdf(urdf_path: Path, link_names: List[str]) -> Tuple[Dict[str, Optional[str]], Dict[str, List[str]]]:
    """Recover parent/child link relations for the visualized mesh links."""
    link_set = set(link_names)
    parent_map: Dict[str, Optional[str]] = {name: None for name in link_names}
    children_map: Dict[str, List[str]] = {name: [] for name in link_names}

    root = ET.parse(urdf_path).getroot()
    for joint_el in root.findall("joint"):
        parent_el = joint_el.find("parent")
        child_el = joint_el.find("child")
        if parent_el is None or child_el is None:
            continue
        p = parent_el.attrib.get("link")
        c = child_el.attrib.get("link")
        if c in link_set:
            parent_map[c] = p if p in link_set else None
        if p in link_set and c in link_set:
            children_map[p].append(c)

    return parent_map, children_map


def _build_link_tree_from_mjcf(xml_path: Path, link_names: List[str]) -> Tuple[Dict[str, Optional[str]], Dict[str, List[str]]]:
    link_set = set(link_names)
    parent_map: Dict[str, Optional[str]] = {name: None for name in link_names}
    children_map: Dict[str, List[str]] = {name: [] for name in link_names}

    root = ET.parse(xml_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        return parent_map, children_map

    def rec(parent_body: ET.Element, parent_name: Optional[str]) -> None:
        for body in parent_body.findall("body"):
            name = _safe_name(body.attrib.get("name", ""))
            if name in link_set:
                if parent_name in link_set:
                    parent_map[name] = parent_name
                    children_map[parent_name].append(name)
                else:
                    parent_map[name] = None
                rec(body, name)
            else:
                rec(body, parent_name)

    rec(worldbody, None)
    return parent_map, children_map


def build_manipulator_robot_lists(
    assets_dir: Path = DEFAULT_ASSETS_DIR,
    out_path: Path = DEFAULT_MANIPULATOR_LISTS_JSON,
) -> Dict[str, List[str]]:
    """
    Build two manipulator robot lists:
    - both_hands: one urdf/xml per manipulator kind, include both left and right when both exist.
    - right_hands: one urdf/xml per manipulator kind, prefer right; fallback to neutral/left if needed.
    """
    meta = discover_robot_assets(assets_dir)
    manipulators = [name for name in meta.keys() if not _is_humanoid_asset_name(name)]

    by_kind: Dict[str, List[str]] = {}
    for name in manipulators:
        kind = _robot_kind_key(name)
        by_kind.setdefault(kind, []).append(name)

    list_both: List[str] = []
    list_right: List[str] = []

    for kind in sorted(by_kind.keys()):
        names = by_kind[kind]
        lefts = [n for n in names if _robot_side(n) == "left"]
        rights = [n for n in names if _robot_side(n) == "right"]
        neutrals = [n for n in names if _robot_side(n) == "neutral"]

        left_pick = _pick_best(lefts, meta)
        right_pick = _pick_best(rights, meta)
        neutral_pick = _pick_best(neutrals, meta)

        # List 1: include both hands when both are available.
        if left_pick and right_pick:
            list_both.extend([left_pick, right_pick])
        elif right_pick:
            list_both.append(right_pick)
        elif left_pick:
            list_both.append(left_pick)
        elif neutral_pick:
            list_both.append(neutral_pick)

        # List 2: prefer right hand only for handed robots.
        if right_pick:
            list_right.append(right_pick)
        elif neutral_pick:
            list_right.append(neutral_pick)
        elif left_pick:
            list_right.append(left_pick)

    # Remove duplicates while preserving order.
    def uniq(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for item in seq:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    result = {
        "both_hands": uniq(list_both),
        "right_hands": uniq(list_right),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


class RobotModel:
    """Robot wrapper that couples FK, surface templates, and contact utilities."""

    def __init__(
        self,
        robot_name: str,
        robot_path: Path,
        device: torch.device,
        num_points: int = 512,
        shadowhand_mode: Optional[str] = None,
        load_surface_assets: bool = True,
    ) -> None:
        """Load robot geometry, build surface artifacts, and prepare FK helpers."""
        self.robot_name = robot_name
        self.robot_path = Path(robot_path)
        self.device = device
        self.shadowhand_mode = str(shadowhand_mode or "default").strip().lower()
        self.load_surface_assets = bool(load_surface_assets)

        if self.robot_path.suffix.lower() == ".urdf":
            chain = _build_chain_quiet(self.robot_path.read_bytes())
            chain = chain.to(dtype=torch.float32, device=device)
            meshes = _load_urdf_link_geometries(self.robot_path, chain.get_link_names())
        elif self.robot_path.suffix.lower() == ".xml":
            chain, meshes = _build_pk_and_meshes_from_mjcf(self.robot_path)
            chain = chain.to(dtype=torch.float32, device=device)
            meshes = {k: v for k, v in meshes.items()}
        else:
            raise ValueError(f"Unsupported robot file format: {self.robot_path}")

        self.pk_chain = chain
        self.dof = len(self.pk_chain.get_joint_parameter_names())
        self.joint_orders = [joint.name for joint in self.pk_chain.get_joints()]
        self.base_pose_indices = self._infer_base_pose_indices()
        self.base_translation_indices = self._infer_base_translation_indices()
        self.base_rotation_indices = self._infer_base_rotation_indices()
        self.fixed_zero_joint_indices = self._resolve_fixed_zero_joint_indices()
        self.hidden_visual_link_names = {"wrist"} if self._uses_shadowhand_probe_palm_root_mode() else set()
        self.excluded_contact_link_names = {"wrist"} if self._uses_shadowhand_probe_palm_root_mode() else set()
        self._shadowhand_probe_palm_ref_tf = self._compute_shadowhand_probe_palm_ref_tf()
        if self._shadowhand_probe_palm_ref_tf is not None:
            # Keep the tiny cached probe transforms on CPU so CPU-side SDF queries from
            # DataLoader workers do not touch inherited CUDA storage during q conversion.
            self._shadowhand_probe_palm_ref_tf = self._shadowhand_probe_palm_ref_tf.to(device="cpu", dtype=torch.float32)
            self._shadowhand_probe_palm_ref_inv_tf = torch.linalg.inv(self._shadowhand_probe_palm_ref_tf)
        else:
            self._shadowhand_probe_palm_ref_inv_tf = None
        # NOTE: num_points now means whole-hand surface point count.
        self.link_num_points = int(num_points)
        self.surface_num_points = int(num_points)

        # Keep original-link assets and box-link assets separately.
        self.meshes_original = {k: v.copy() for k, v in meshes.items()}
        self.vertices_original = self._sample_points_from_meshes(self.meshes_original)
        mesh_link_names = list(self.meshes_original.keys())
        self.mesh_link_names = list(mesh_link_names)
        if self.robot_path.suffix.lower() == ".urdf":
            self.link_parent_map, self.link_children_map = _build_link_tree_from_urdf(self.robot_path, mesh_link_names)
        elif self.robot_path.suffix.lower() == ".xml":
            self.link_parent_map, self.link_children_map = _build_link_tree_from_mjcf(self.robot_path, mesh_link_names)
        else:
            self.link_parent_map = {name: None for name in mesh_link_names}
            self.link_children_map = {name: [] for name in mesh_link_names}

        # Whole-hand surface template (local frame) and graph.
        self.surface_template_by_link: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.surface_template_points_local = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        self.surface_template_normals_local = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        self.surface_template_link_indices = torch.zeros((0,), dtype=torch.long, device=self.device)
        self.surface_template_points_canonical_world = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        self.surface_template_normals_canonical_world = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        self.surface_graph_points_canonical_world = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        self.surface_graph_normals_canonical_world = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        self.surface_expansion_bank_points_local = torch.zeros((0, 0, 3), dtype=torch.float32, device=self.device)
        self.surface_expansion_bank_normals_local = torch.zeros((0, 0, 3), dtype=torch.float32, device=self.device)
        self.surface_union_mesh_canonical: Optional[trimesh.Trimesh] = None
        self.surface_graph_neighbors_strict: List[List[int]] = []
        self.surface_graph_neighbors: List[List[int]] = []

        # Contact-mask sampler fitted state.
        self.contact_mask_sampler_state: Optional[Dict[str, torch.Tensor]] = None

        self.frame_status = None
        self._fast_fk_cache: Optional[Dict[str, Any]] = None
        self.surface_template_seed = _stable_int_seed(
            f"{self.robot_name}|{self.surface_num_points}|surface_template_v2"
        )
        self.surface_artifact_path: Optional[str] = None
        self.surface_artifact_built: bool = False
        if self.load_surface_assets:
            with _temporary_global_seed(self.surface_template_seed):
                self._load_or_build_surface_assets(num_points=self.surface_num_points)
        self.joint_descendant_link_mask = self._build_joint_descendant_link_mask()

    def _uses_shadowhand_probe_palm_root_mode(self) -> bool:
        """Check whether ShadowHand palm-root remapping is enabled."""
        return self.shadowhand_mode == "probe_palm_root"

    def _resolve_fixed_zero_joint_indices(self) -> List[int]:
        """Identify probe-only joints that must remain fixed at zero."""
        if not self._uses_shadowhand_probe_palm_root_mode():
            return []
        indices: List[int] = []
        for idx, joint_name in enumerate(self.joint_orders):
            if str(joint_name).upper().startswith("WRJ"):
                indices.append(int(idx))
        return sorted(indices)

    def _compute_shadowhand_probe_palm_ref_tf(self) -> Optional[torch.Tensor]:
        """Precompute the canonical palm transform used by probe-palm-root mode."""
        if not self._uses_shadowhand_probe_palm_root_mode():
            return None
        if len(self.base_translation_indices) != 3 or len(self.base_rotation_indices) != 3:
            raise RuntimeError(
                "ShadowHand probe palm-root mode requires exactly 3 base translation joints "
                f"and 3 base rotation joints, got {self.base_translation_indices} and {self.base_rotation_indices}"
            )
        q_zero = torch.zeros((1, self.dof), dtype=torch.float32, device=self.device)
        if len(self.fixed_zero_joint_indices) > 0:
            q_zero[:, self.fixed_zero_joint_indices] = 0.0
        frame_status = self.pk_chain.forward_kinematics(q_zero)
        if "palm" not in frame_status:
            raise KeyError(f"Robot {self.robot_name} is missing palm frame required for probe_palm_root mode")
        palm_tf = frame_status["palm"].get_matrix()
        if palm_tf.ndim == 3:
            palm_tf = palm_tf[0]
        return palm_tf.to(dtype=torch.float32, device=self.device)

    def _apply_fixed_zero_joint_constraints(self, q: torch.Tensor) -> torch.Tensor:
        """Clamp configured probe joints back to zero after sampling or transforms."""
        if len(self.fixed_zero_joint_indices) == 0:
            return q
        q_out = q.clone()
        if q_out.ndim == 1:
            q_out[self.fixed_zero_joint_indices] = 0.0
        else:
            q_out[..., self.fixed_zero_joint_indices] = 0.0
        return q_out

    def _semantic_shadowhand_probe_q_to_model_q(self, q: torch.Tensor) -> torch.Tensor:
        """Convert palm-root semantic parameters into the model's FK parameterization."""
        if not self._uses_shadowhand_probe_palm_root_mode():
            return self._apply_fixed_zero_joint_constraints(q)

        squeeze = False
        if q.ndim == 1:
            q = q.unsqueeze(0)
            squeeze = True
        q_model = self._apply_fixed_zero_joint_constraints(q)
        if self._shadowhand_probe_palm_ref_inv_tf is None:
            raise RuntimeError("ShadowHand palm-root reference transform is not initialized")

        desired_translation = q_model[:, self.base_translation_indices]
        desired_rotation = q_model[:, self.base_rotation_indices]
        desired_rot_tf = _euler_xyz_to_matrix(desired_rotation)

        ref_inv_tf = self._shadowhand_probe_palm_ref_inv_tf.to(device=q_model.device, dtype=torch.float32)
        ref_inv_rot = ref_inv_tf[:3, :3].unsqueeze(0).expand(int(q_model.shape[0]), -1, -1)
        ref_inv_trans = ref_inv_tf[:3, 3].view(1, 3, 1).expand(int(q_model.shape[0]), -1, -1)

        model_rot_tf = torch.matmul(desired_rot_tf, ref_inv_rot)
        model_trans = desired_translation + torch.matmul(desired_rot_tf, ref_inv_trans).squeeze(-1)
        model_rot_euler = _matrix_to_euler_xyz(model_rot_tf)

        q_model[:, self.base_translation_indices] = model_trans
        q_model[:, self.base_rotation_indices] = model_rot_euler
        if squeeze:
            return q_model[0]
        return q_model


    def _allocate_points_over_meshes(
        self,
        mesh_dict: Dict[str, trimesh.Trimesh],
        total_points: int,
    ) -> Dict[str, int]:
        """Split a global point budget across links in proportion to surface area."""
        if total_points <= 0:
            raise ValueError("total_points must be positive")
        link_names = list(mesh_dict.keys())
        if len(link_names) == 0:
            return {}

        areas = np.array([float(max(mesh_dict[n].area, 1e-8)) for n in link_names], dtype=np.float64)
        areas = areas / np.clip(areas.sum(), 1e-8, None)
        raw = areas * float(total_points)
        counts = np.floor(raw).astype(np.int64)
        remain = int(total_points - counts.sum())
        if remain > 0:
            frac = raw - counts
            order = np.argsort(-frac)
            for i in range(remain):
                counts[order[i % len(order)]] += 1

        return {name: int(count) for name, count in zip(link_names, counts)}

    def _surface_artifact_cache_path(self, num_points: int) -> Path:
        """Resolve the cache path for this model's surface artifact."""
        try:
            from .robot_model_process import get_surface_artifact_path
        except Exception:
            from robot_model_process import get_surface_artifact_path  # type: ignore

        return get_surface_artifact_path(
            robot_path=self.robot_path,
            robot_name=self.robot_name,
            num_points=int(num_points),
        )

    def _load_or_build_surface_assets(self, num_points: int) -> None:
        """Populate the model with cached or freshly built surface tensors."""
        try:
            from .robot_model_process import apply_surface_artifact_to_model, load_or_build_surface_artifact
        except Exception:
            from robot_model_process import apply_surface_artifact_to_model, load_or_build_surface_artifact  # type: ignore

        artifact, cache_path, built = load_or_build_surface_artifact(
            model=self,
            num_points=int(num_points),
            seed=int(self.surface_template_seed),
            cache_path=self._surface_artifact_cache_path(num_points=int(num_points)),
        )
        print(f"Cache path for surface artifact: {cache_path} (built={built})")
        apply_surface_artifact_to_model(self, artifact)
        self.surface_artifact_path = str(cache_path)
        self.surface_artifact_built = bool(built)

    def _sample_mesh_points_normals(
        self,
        mesh: trimesh.Trimesh,
        count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample local mesh points and normals with a vertex fallback."""
        if count <= 0:
            empty = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
            return empty, empty

        try:
            pts_np, face_idx = mesh.sample(int(count), return_index=True)
            nrm_np = mesh.face_normals[face_idx]
            pts = torch.tensor(pts_np, dtype=torch.float32, device=self.device)
            nrms = torch.tensor(nrm_np, dtype=torch.float32, device=self.device)
        except Exception:
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            if verts.shape[0] == 0:
                empty = torch.zeros((count, 3), dtype=torch.float32, device=self.device)
                return empty, empty
            idx = np.random.randint(0, verts.shape[0], size=(count,))
            pts = torch.tensor(verts[idx], dtype=torch.float32, device=self.device)
            nrms = torch.zeros_like(pts)

        nrms = nrms / torch.norm(nrms, dim=1, keepdim=True).clamp_min(1e-8)
        return pts, nrms

    def _get_canonical_link_meshes_world(
        self,
    ) -> Tuple[List[str], Dict[str, trimesh.Trimesh]]:
        """Export link meshes at the canonical pose used for template generation."""
        q_ref = self._zero_base_translation_in_q(self.get_canonical_q().detach().clone())
        self.update_status(q_ref)
        link_names = list(self.meshes_original.keys())
        link_meshes_world: Dict[str, trimesh.Trimesh] = {}
        for link_name in link_names:
            if link_name not in self.frame_status:
                continue
            mesh_local = self.meshes_original.get(link_name, None)
            if mesh_local is None or len(mesh_local.faces) == 0:
                continue
            tf = self.frame_status[link_name].get_matrix()[0].detach().cpu().numpy()
            mesh_world = mesh_local.copy()
            mesh_world.apply_transform(tf)
            if len(mesh_world.faces) == 0:
                continue
            link_meshes_world[link_name] = mesh_world
        if len(link_meshes_world) == 0:
            raise RuntimeError(f"Failed to build canonical meshes for robot: {self.robot_name}")
        return link_names, link_meshes_world

    def _sample_points_from_meshes(self, mesh_dict: Dict[str, trimesh.Trimesh]) -> Dict[str, torch.Tensor]:
        """Sample per-link local points used for coarse visual point clouds."""
        counts = self._allocate_points_over_meshes(mesh_dict, total_points=self.surface_num_points)
        vertices: Dict[str, torch.Tensor] = {}
        for link_name, link_mesh in mesh_dict.items():
            c = int(counts.get(link_name, 0))
            if c <= 0:
                vertices[link_name] = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
                continue
            pts, _ = self._sample_mesh_points_normals(link_mesh, c)
            vertices[link_name] = pts
        return vertices


    def get_surface_template_hash(self) -> str:
        h = hashlib.sha1()
        pts = np.ascontiguousarray(self.surface_template_points_local.detach().cpu().numpy().astype(np.float32))
        nrms = np.ascontiguousarray(self.surface_template_normals_local.detach().cpu().numpy().astype(np.float32))
        idx = np.ascontiguousarray(self.surface_template_link_indices.detach().cpu().numpy().astype(np.int64))
        h.update(pts.tobytes())
        h.update(nrms.tobytes())
        h.update(idx.tobytes())
        return h.hexdigest()


    def _get_visual_assets(self, mode: str = "active") -> Tuple[Dict[str, trimesh.Trimesh], Dict[str, torch.Tensor]]:
        mode_key = str(mode).strip().lower()
        if mode_key not in {"active", "original"}:
            raise ValueError(
                f"Unsupported mode={mode}. RobotModel only supports original meshes in the current pipeline."
            )
        return self.meshes_original, self.vertices_original

    def get_joint_orders(self) -> List[str]:
        return list(self.joint_orders)

    def _infer_base_pose_indices(self) -> List[int]:
        idx: List[int] = []
        for i, name in enumerate(self.joint_orders):
            n = _normalize_joint_name(name)
            if _is_base_translation_joint_name(n) or _is_base_rotation_joint_name(n):
                idx.append(i)
        return sorted(set(idx))

    def _build_joint_descendant_link_mask(self) -> torch.Tensor:
        link_to_idx = {name: i for i, name in enumerate(self.mesh_link_names)}
        joint_to_links: Dict[str, List[str]] = {}
        for joint, child_links in self.pk_chain.get_joints_and_child_links():
            joint_name = str(joint.name)
            descendants: List[str] = []
            for link in child_links:
                link_name = getattr(link, "name", None)
                if link_name in link_to_idx:
                    descendants.append(str(link_name))
            joint_to_links[joint_name] = descendants

        mask = torch.zeros(
            (self.dof, len(self.mesh_link_names)),
            dtype=torch.bool,
            device=self.device,
        )
        for joint_idx, joint_name in enumerate(self.joint_orders):
            for link_name in joint_to_links.get(joint_name, []):
                mask[joint_idx, link_to_idx[link_name]] = True
        if len(self.base_pose_indices) > 0 and len(self.mesh_link_names) > 0:
            mask[self.base_pose_indices] = True
        return mask

    def _infer_base_translation_indices(self) -> List[int]:
        idx: List[int] = []
        for i, name in enumerate(self.joint_orders):
            if _is_base_translation_joint_name(_normalize_joint_name(name)):
                idx.append(i)
        return sorted(set(idx))

    def _infer_base_rotation_indices(self) -> List[int]:
        idx: List[int] = []
        for i, name in enumerate(self.joint_orders):
            if _is_base_rotation_joint_name(_normalize_joint_name(name)):
                idx.append(i)
        return sorted(set(idx))

    def _zero_base_translation_in_q(self, q: torch.Tensor) -> torch.Tensor:
        if len(self.base_translation_indices) == 0:
            return self._apply_fixed_zero_joint_constraints(q)
        q_out = q.clone()
        if q_out.ndim == 1:
            q_out[self.base_translation_indices] = 0.0
        else:
            q_out[..., self.base_translation_indices] = 0.0
        return self._apply_fixed_zero_joint_constraints(q_out)

    def _zero_base_pose_in_q(self, q: torch.Tensor) -> torch.Tensor:
        if len(self.base_pose_indices) == 0:
            return self._apply_fixed_zero_joint_constraints(q)
        q_out = q.clone()
        if q_out.ndim == 1:
            q_out[self.base_pose_indices] = 0.0
        else:
            q_out[..., self.base_pose_indices] = 0.0
        return self._apply_fixed_zero_joint_constraints(q_out)

    def set_base_pose_in_q(
        self,
        q: torch.Tensor,
        *,
        base_translation: Optional[torch.Tensor] = None,
        base_rotation_euler: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        squeeze = False
        if q.ndim == 1:
            q = q.unsqueeze(0)
            squeeze = True
        q_out = q.clone()
        if base_translation is not None and len(self.base_translation_indices) > 0:
            base_translation = torch.as_tensor(base_translation, dtype=torch.float32, device=self.device)
            if base_translation.ndim == 1:
                base_translation = base_translation.unsqueeze(0)
            expected_dim = len(self.base_translation_indices)
            if base_translation.shape != (int(q_out.shape[0]), expected_dim):
                raise ValueError(
                    f"base_translation must have shape ({q_out.shape[0]}, {expected_dim}), "
                    f"got {tuple(base_translation.shape)}"
                )
            q_out[:, self.base_translation_indices] = base_translation
        if base_rotation_euler is not None and len(self.base_rotation_indices) > 0:
            base_rotation_euler = torch.as_tensor(base_rotation_euler, dtype=torch.float32, device=self.device)
            if base_rotation_euler.ndim == 1:
                base_rotation_euler = base_rotation_euler.unsqueeze(0)
            expected_dim = len(self.base_rotation_indices)
            if base_rotation_euler.shape != (int(q_out.shape[0]), expected_dim):
                raise ValueError(
                    f"base_rotation_euler must have shape ({q_out.shape[0]}, {expected_dim}), "
                    f"got {tuple(base_rotation_euler.shape)}"
                )
            q_out[:, self.base_rotation_indices] = base_rotation_euler
        if squeeze:
            return self._apply_fixed_zero_joint_constraints(q_out[0])
        return self._apply_fixed_zero_joint_constraints(q_out)

    def sample_uniform_base_rotation_euler_xyz(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if len(self.base_rotation_indices) == 0:
            return torch.zeros((int(batch_size), 0), dtype=torch.float32, device=self.device)
        if len(self.base_rotation_indices) != 3:
            raise RuntimeError(
                f"Uniform SO(3) sampling requires exactly 3 base rotation joints, got {self.base_rotation_indices}"
            )
        return _sample_uniform_so3_euler_xyz(
            int(batch_size),
            device=self.device,
            generator=generator,
        )

    def _build_active_links_from_contact_mask(self, mask: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
            squeeze = True
        if mask.ndim != 2:
            raise ValueError(f"mask must have shape (B, {self.surface_num_points}), got {tuple(mask.shape)}")
        if int(mask.shape[-1]) != int(self.surface_num_points):
            raise ValueError(
                f"mask last dim must match surface_num_points={self.surface_num_points}, got {mask.shape[-1]}"
            )
        mask = mask.to(self.device).bool()
        active_links = torch.zeros(
            (int(mask.shape[0]), len(self.mesh_link_names)),
            dtype=torch.bool,
            device=self.device,
        )
        link_idx = self.surface_template_link_indices.to(self.device)
        for link_i in range(len(self.mesh_link_names)):
            surface_mask = link_idx == int(link_i)
            if bool(surface_mask.any()):
                active_links[:, link_i] = mask[:, surface_mask].any(dim=1)
        if squeeze:
            return active_links.squeeze(0)
        return active_links

    def get_joint_active_by_contact_mask(self, mask: torch.Tensor) -> torch.Tensor:
        active_links = self._build_active_links_from_contact_mask(mask)
        squeeze = False
        if active_links.ndim == 1:
            active_links = active_links.unsqueeze(0)
            squeeze = True
        joint_active = (
            active_links.unsqueeze(1)
            & self.joint_descendant_link_mask.unsqueeze(0)
        ).any(dim=-1)
        if squeeze:
            return joint_active.squeeze(0)
        return joint_active

    def get_inactive_link_mask_by_contact_mask(
        self,
        mask: torch.Tensor,
        *,
        ignore_base_pose: bool = True,
    ) -> torch.Tensor:
        joint_active = self.get_joint_active_by_contact_mask(mask)
        squeeze = False
        if joint_active.ndim == 1:
            joint_active = joint_active.unsqueeze(0)
            squeeze = True
        if ignore_base_pose and len(self.base_pose_indices) > 0:
            joint_active = joint_active.clone()
            joint_active[:, self.base_pose_indices] = False
        affected_links = (
            joint_active.unsqueeze(-1)
            & self.joint_descendant_link_mask.unsqueeze(0)
        ).any(dim=1)
        inactive_links = ~affected_links
        if squeeze:
            return inactive_links.squeeze(0)
        return inactive_links

    def update_status(self, q: torch.Tensor) -> None:
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if q.shape[-1] == self.dof + 3:
            q = q_rot6d_to_q_euler(q)
        if q.shape[-1] != self.dof:
            raise ValueError(f"Expected joint dim {self.dof}, got {q.shape[-1]}")
        q_model = self._semantic_shadowhand_probe_q_to_model_q(q.to(self.device, dtype=torch.float32))
        self.frame_status = self.pk_chain.forward_kinematics(q_model)


    def _get_fast_fk_cache(self) -> Optional[Dict[str, Any]]:
        """Cache Python-side chain topology used by the attached-point FK fast path."""
        if self._fast_fk_cache is not None:
            return self._fast_fk_cache

        chain = self.pk_chain
        required = (
            "get_all_frame_indices",
            "parents_indices",
            "joint_indices",
            "joint_type_indices",
            "link_offsets",
            "joint_offsets",
            "axes",
            "frame_to_idx",
        )
        if not all(hasattr(chain, name) for name in required):
            return None

        try:
            frame_indices_raw = chain.get_all_frame_indices()
            if torch.is_tensor(frame_indices_raw):
                frame_indices = [int(v) for v in frame_indices_raw.detach().cpu().tolist()]
            else:
                frame_indices = [int(v) for v in frame_indices_raw]

            parents: Dict[int, List[int]] = {}
            for frame_idx in frame_indices:
                if frame_idx < 0 or frame_idx >= len(chain.parents_indices):
                    return None
                parent_raw = chain.parents_indices[frame_idx]
                if torch.is_tensor(parent_raw):
                    parents[frame_idx] = [int(v) for v in parent_raw.detach().cpu().tolist()]
                else:
                    parents[frame_idx] = [int(v) for v in parent_raw]

            joint_indices_raw = chain.joint_indices
            joint_type_raw = chain.joint_type_indices
            if torch.is_tensor(joint_indices_raw):
                joint_indices = [int(v) for v in joint_indices_raw.detach().cpu().tolist()]
            else:
                joint_indices = [int(v) for v in joint_indices_raw]
            if torch.is_tensor(joint_type_raw):
                joint_type_indices = [int(v) for v in joint_type_raw.detach().cpu().tolist()]
            else:
                joint_type_indices = [int(v) for v in joint_type_raw]

            frame_to_idx = {str(k): int(v) for k, v in chain.frame_to_idx.items()}
            mesh_frame_indices = [frame_to_idx.get(str(name), -1) for name in self.mesh_link_names]
        except Exception:
            return None

        self._fast_fk_cache = {
            "frame_indices": frame_indices,
            "parents": parents,
            "joint_indices": joint_indices,
            "joint_type_indices": joint_type_indices,
            "link_offsets": list(chain.link_offsets),
            "joint_offsets": list(chain.joint_offsets),
            "axes": chain.axes,
            "mesh_frame_indices": mesh_frame_indices,
        }
        return self._fast_fk_cache

    @staticmethod
    def _fast_axis_angle_to_matrix_44(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Vectorized copy of pytorch_kinematics' axis-angle 4x4 transform."""
        c = torch.cos(theta)
        one_minus_c = 1.0 - c
        s = torch.sin(theta)
        kx = axis[:, 0].unsqueeze(0)
        ky = axis[:, 1].unsqueeze(0)
        kz = axis[:, 2].unsqueeze(0)

        r00 = c + kx * kx * one_minus_c
        r01 = kx * ky * one_minus_c - kz * s
        r02 = kx * kz * one_minus_c + ky * s
        r10 = ky * kx * one_minus_c + kz * s
        r11 = c + ky * ky * one_minus_c
        r12 = ky * kz * one_minus_c - kx * s
        r20 = kz * kx * one_minus_c - ky * s
        r21 = kz * ky * one_minus_c + kx * s
        r22 = c + kz * kz * one_minus_c

        z = torch.zeros_like(c)
        o = torch.ones_like(c)
        row0 = torch.stack((r00, r01, r02, z), dim=-1)
        row1 = torch.stack((r10, r11, r12, z), dim=-1)
        row2 = torch.stack((r20, r21, r22, z), dim=-1)
        row3 = torch.stack((z, z, z, o), dim=-1)
        return torch.stack((row0, row1, row2, row3), dim=-2)

    @staticmethod
    def _fast_prismatic_to_matrix_44(axis: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Vectorized copy of pytorch_kinematics' prismatic 4x4 transform."""
        z = torch.zeros_like(d)
        o = torch.ones_like(d)
        pos = axis.unsqueeze(0) * d.unsqueeze(-1)
        row0 = torch.stack((o, z, z, pos[..., 0]), dim=-1)
        row1 = torch.stack((z, o, z, pos[..., 1]), dim=-1)
        row2 = torch.stack((z, z, o, pos[..., 2]), dim=-1)
        row3 = torch.stack((z, z, z, o), dim=-1)
        return torch.stack((row0, row1, row2, row3), dim=-2)

    def _fast_fk_matrices_from_model_q(
        self,
        q_model: torch.Tensor,
        cache: Dict[str, Any],
    ) -> Dict[int, torch.Tensor]:
        """Compute raw frame matrices without CUDA scalar syncs from chain topology."""
        batch_size = int(q_model.shape[0])
        axes = cache["axes"].to(device=q_model.device, dtype=q_model.dtype)
        rev_jnt_transform = self._fast_axis_angle_to_matrix_44(axes, q_model)
        pris_jnt_transform = self._fast_prismatic_to_matrix_44(axes, q_model)

        identity = torch.eye(4, dtype=q_model.dtype, device=q_model.device).unsqueeze(0).expand(batch_size, -1, -1)
        frame_transforms: Dict[int, torch.Tensor] = {}
        link_offsets: List[Optional[torch.Tensor]] = cache["link_offsets"]
        joint_offsets: List[Optional[torch.Tensor]] = cache["joint_offsets"]
        joint_indices: List[int] = cache["joint_indices"]
        joint_type_indices: List[int] = cache["joint_type_indices"]

        for frame_idx in cache["frame_indices"]:
            frame_transform = identity
            for chain_idx in cache["parents"].get(frame_idx, []):
                if chain_idx in frame_transforms:
                    frame_transform = frame_transforms[chain_idx]
                    continue

                if chain_idx < len(link_offsets):
                    link_offset = link_offsets[chain_idx]
                    if link_offset is not None:
                        frame_transform = frame_transform @ link_offset.to(device=q_model.device, dtype=q_model.dtype)

                if chain_idx < len(joint_offsets):
                    joint_offset = joint_offsets[chain_idx]
                    if joint_offset is not None:
                        frame_transform = frame_transform @ joint_offset.to(device=q_model.device, dtype=q_model.dtype)

                if chain_idx >= len(joint_indices) or chain_idx >= len(joint_type_indices):
                    continue
                jnt_idx = joint_indices[chain_idx]
                jnt_type = joint_type_indices[chain_idx]
                if jnt_type == 1 and 0 <= jnt_idx < rev_jnt_transform.shape[1]:
                    frame_transform = frame_transform @ rev_jnt_transform[:, jnt_idx]
                elif jnt_type == 2 and 0 <= jnt_idx < pris_jnt_transform.shape[1]:
                    frame_transform = frame_transform @ pris_jnt_transform[:, jnt_idx]

            frame_transforms[int(frame_idx)] = frame_transform

        return frame_transforms

    def _transform_attached_points_normals_batch_fast(
        self,
        *,
        q: torch.Tensor,
        local_points: torch.Tensor,
        link_indices: torch.Tensor,
        local_normals: Optional[torch.Tensor],
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Fast differentiable attached-point FK used by contact-space losses."""
        cache = self._get_fast_fk_cache()
        if cache is None:
            return None

        if q.shape[-1] == self.dof + 3:
            q = q_rot6d_to_q_euler(q)
        if q.shape[-1] != self.dof:
            raise ValueError(f"Expected joint dim {self.dof}, got {q.shape[-1]}")

        q_model = self._semantic_shadowhand_probe_q_to_model_q(q.to(self.device, dtype=torch.float32))
        axes = cache["axes"]
        if not torch.is_tensor(axes) or int(axes.shape[0]) != int(q_model.shape[1]):
            return None
        frame_transforms = self._fast_fk_matrices_from_model_q(q_model, cache)

        batch_size = int(q_model.shape[0])
        num_links = len(self.mesh_link_names)
        local_points = local_points.to(device=self.device, dtype=torch.float32)
        link_indices = link_indices.to(device=self.device, dtype=torch.long)
        if local_normals is not None:
            local_normals = local_normals.to(device=self.device, dtype=torch.float32)

        if num_links <= 0:
            points_world = torch.zeros_like(local_points)
            normals_world = None if local_normals is None else torch.zeros_like(local_normals)
            return points_world, normals_world

        zero_tf = q_model.new_zeros((batch_size, 4, 4))
        mesh_transforms = []
        valid_mesh_links: List[bool] = []
        for frame_idx in cache["mesh_frame_indices"]:
            if int(frame_idx) >= 0 and int(frame_idx) in frame_transforms:
                mesh_transforms.append(frame_transforms[int(frame_idx)])
                valid_mesh_links.append(True)
            else:
                mesh_transforms.append(zero_tf)
                valid_mesh_links.append(False)
        mesh_tf = torch.stack(mesh_transforms, dim=1)

        link_clamped = link_indices.clamp(min=0, max=num_links - 1)
        valid = (link_indices >= 0) & (link_indices < num_links)
        if not all(valid_mesh_links):
            valid_link_tensor = torch.tensor(valid_mesh_links, dtype=torch.bool, device=self.device)
            valid = valid & valid_link_tensor[link_clamped]

        gather_idx = link_clamped.view(batch_size, -1, 1, 1).expand(-1, -1, 4, 4)
        point_tf = torch.gather(mesh_tf, dim=1, index=gather_idx)
        rot = point_tf[..., :3, :3]
        trans = point_tf[..., :3, 3]

        points_world = torch.matmul(local_points.unsqueeze(-2), rot.transpose(-1, -2)).squeeze(-2) + trans
        points_world = torch.where(valid.unsqueeze(-1), points_world, torch.zeros_like(points_world))

        normals_world = None
        if local_normals is not None:
            normals_world = torch.matmul(local_normals.unsqueeze(-2), rot.transpose(-1, -2)).squeeze(-2)
            normals_world = normals_world / torch.norm(normals_world, dim=-1, keepdim=True).clamp_min(1e-8)
            normals_world = torch.where(valid.unsqueeze(-1), normals_world, torch.zeros_like(normals_world))

        return points_world, normals_world


    def get_surface_points_normals_batch(self, q: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform the canonical surface template into world space for a batch of poses."""
        if q is None:
            q = torch.zeros((1, self.dof), dtype=torch.float32, device=self.device)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        self.update_status(q)

        n_total = int(self.surface_template_points_local.shape[0])
        if n_total <= 0:
            empty = torch.zeros((int(q.shape[0]), self.surface_num_points, 3), dtype=torch.float32, device=self.device)
            return empty, empty

        batch_size = int(q.shape[0])
        points = torch.zeros((batch_size, n_total, 3), dtype=torch.float32, device=self.device)
        normals = torch.zeros((batch_size, n_total, 3), dtype=torch.float32, device=self.device)
        link_idx = self.surface_template_link_indices
        link_names = self.mesh_link_names

        for link_i, link_name in enumerate(link_names):
            mask = link_idx == int(link_i)
            if not bool(mask.any()):
                continue
            if link_name not in self.frame_status:
                continue
            pts_local = self.surface_template_points_local[mask]
            nrms_local = self.surface_template_normals_local[mask]

            tf = self.frame_status[link_name].get_matrix().to(self.device)
            if tf.ndim == 2:
                tf = tf.unsqueeze(0)
            rot = tf[:, :3, :3]
            trans = tf[:, :3, 3]
            pts_world = torch.matmul(pts_local.unsqueeze(0), rot.transpose(1, 2)) + trans.unsqueeze(1)
            nrms_world = torch.matmul(nrms_local.unsqueeze(0), rot.transpose(1, 2))
            nrms_world = nrms_world / torch.norm(nrms_world, dim=-1, keepdim=True).clamp_min(1e-8)

            points[:, mask] = pts_world
            normals[:, mask] = nrms_world

        return points, normals

    def get_surface_points_normals(self, q: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-pose convenience wrapper for ``get_surface_points_normals_batch``."""
        points, normals = self.get_surface_points_normals_batch(q=q)
        return points[0], normals[0]

    def transform_attached_points_normals_batch(
        self,
        *,
        q: torch.Tensor,
        local_points: torch.Tensor,
        link_indices: torch.Tensor,
        local_normals: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Move link-local attached points and normals into world coordinates."""
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

        batch_size = int(q.shape[0])
        if local_points.ndim != 3 or tuple(local_points.shape[:2]) != tuple(link_indices.shape):
            raise ValueError(
                f"local_points must have shape (B, K, 3) aligned with link_indices, got {tuple(local_points.shape)} and {tuple(link_indices.shape)}"
            )
        if int(local_points.shape[0]) != batch_size:
            raise ValueError(
                f"local_points batch must match q batch, got {local_points.shape[0]} vs {batch_size}"
            )
        if local_normals is not None and tuple(local_normals.shape) != tuple(local_points.shape):
            raise ValueError(
                f"local_normals must match local_points shape, got {tuple(local_normals.shape)} vs {tuple(local_points.shape)}"
            )

        fast_result = self._transform_attached_points_normals_batch_fast(
            q=q,
            local_points=local_points,
            link_indices=link_indices,
            local_normals=local_normals,
        )
        if fast_result is not None:
            points_world, normals_world = fast_result
            if squeeze:
                return points_world[0], None if normals_world is None else normals_world[0]
            return points_world, normals_world

        self.update_status(q)
        points_world = torch.zeros_like(local_points)
        normals_world = None if local_normals is None else torch.zeros_like(local_normals)
        link_indices = link_indices.to(device=self.device, dtype=torch.long)
        local_points = local_points.to(device=self.device, dtype=torch.float32)
        if local_normals is not None:
            local_normals = local_normals.to(device=self.device, dtype=torch.float32)

        for link_i, link_name in enumerate(self.mesh_link_names):
            if link_name not in self.frame_status:
                continue
            sel = link_indices == int(link_i)
            batch_idx, point_idx = torch.nonzero(sel, as_tuple=True)
            tf = self.frame_status[link_name].get_matrix().to(self.device)
            if tf.ndim == 2:
                tf = tf.unsqueeze(0)
            rot = tf[batch_idx, :3, :3]
            trans = tf[batch_idx, :3, 3]
            pts_local = local_points[batch_idx, point_idx]
            pts_world = torch.bmm(pts_local.unsqueeze(1), rot.transpose(1, 2)).squeeze(1) + trans
            points_world[batch_idx, point_idx] = pts_world
            if normals_world is not None and local_normals is not None:
                nr_local = local_normals[batch_idx, point_idx]
                nr_world = torch.bmm(nr_local.unsqueeze(1), rot.transpose(1, 2)).squeeze(1)
                nr_world = nr_world / torch.norm(nr_world, dim=1, keepdim=True).clamp_min(1e-8)
                normals_world[batch_idx, point_idx] = nr_world

        if squeeze:
            return points_world[0], None if normals_world is None else normals_world[0]
        return points_world, normals_world


    def build_hand_fps_template(self, n_points: int) -> None:
        """Pre-compute a farthest-point-sampled subset of the canonical hand surface.

        The subset is shared across every pose — we only need FK at sample time to place
        the template into world coordinates. Callers store this once and then call
        ``transform_hand_fps_template_batch`` to materialize world-space hand tokens.
        """
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
        if num_template == 0:
            raise RuntimeError(
                f"{self.robot_name}: cannot build hand FPS template — surface template is empty."
            )
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
        """Place the pre-computed hand FPS template into world space for a batch of poses."""
        if getattr(self, "hand_fps_local_points", None) is None:
            raise RuntimeError("build_hand_fps_template must be called before transform_hand_fps_template_batch")
        if q.ndim == 1:
            q = q.unsqueeze(0)
        batch_size = int(q.shape[0])
        local_points = self.hand_fps_local_points.to(device=self.device, dtype=torch.float32).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        local_normals = self.hand_fps_local_normals.to(device=self.device, dtype=torch.float32).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        link_indices = self.hand_fps_link_indices.to(device=self.device, dtype=torch.long).unsqueeze(0).expand(
            batch_size, -1
        )
        points_world, normals_world = self.transform_attached_points_normals_batch(
            q=q,
            local_points=local_points,
            link_indices=link_indices,
            local_normals=local_normals,
        )
        assert normals_world is not None
        return points_world, normals_world

    def expand_sparse_contact_support_batch(
        self,
        *,
        q: torch.Tensor,
        sparse_contact_mask: torch.Tensor,
        num_contact_points: int,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """Expand a sparse surface mask into a fixed-size dense contact set."""
        squeeze = False
        if q.ndim == 1:
            q = q.unsqueeze(0)
            squeeze = True
        if sparse_contact_mask.ndim == 1:
            sparse_contact_mask = sparse_contact_mask.unsqueeze(0)
        if sparse_contact_mask.ndim != 2 or int(sparse_contact_mask.shape[0]) != int(q.shape[0]):
            raise ValueError(
                f"sparse_contact_mask must have shape (B, {self.surface_num_points}), got {tuple(sparse_contact_mask.shape)}"
            )
        if int(sparse_contact_mask.shape[1]) != int(self.surface_num_points):
            raise ValueError(
                f"sparse_contact_mask last dim must match surface_num_points={self.surface_num_points}, got {sparse_contact_mask.shape[1]}"
            )
        if int(self.surface_expansion_bank_points_local.shape[0]) != int(self.surface_num_points):
            raise RuntimeError(
                "Surface expansion bank is missing or inconsistent with the sparse surface template. Rebuild the surface artifact."
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

        # Transform the sparse template first so farthest-point selection happens in world space.
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

            # Draw one precomputed local candidate per selected sparse parent.
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
        """Expand a sparse mask by re-extracting aligned contacts on focused bank neighborhoods."""
        squeeze = False
        if q.ndim == 1:
            q = q.unsqueeze(0)
            squeeze = True
        if sparse_contact_mask.ndim == 1:
            sparse_contact_mask = sparse_contact_mask.unsqueeze(0)
        if sparse_contact_mask.ndim != 2 or int(sparse_contact_mask.shape[0]) != int(q.shape[0]):
            raise ValueError(
                f"sparse_contact_mask must have shape (B, {self.surface_num_points}), got {tuple(sparse_contact_mask.shape)}"
            )
        if int(sparse_contact_mask.shape[1]) != int(self.surface_num_points):
            raise ValueError(
                f"sparse_contact_mask last dim must match surface_num_points={self.surface_num_points}, got {sparse_contact_mask.shape[1]}"
            )
        if int(self.surface_expansion_bank_points_local.shape[0]) != int(self.surface_num_points):
            raise RuntimeError(
                "Surface expansion bank is missing or inconsistent with the sparse surface template. Rebuild the surface artifact."
            )

        if target_points.ndim == 2:
            target_points = target_points.unsqueeze(0)
        elif target_points.ndim == 4:
            batch_size_tp = int(target_points.shape[0])
            target_points = target_points.reshape(batch_size_tp, -1, 3)
            if target_valid_mask is not None:
                target_valid_mask = target_valid_mask.reshape(batch_size_tp, -1)
        elif target_points.ndim != 3:
            raise ValueError(
                f"target_points must have shape (B, K, 3) or (B, A, P, 3), got {tuple(target_points.shape)}"
            )

        if int(target_points.shape[0]) != int(q.shape[0]):
            raise ValueError(
                f"target_points batch must match q batch, got {target_points.shape[0]} vs {q.shape[0]}"
            )
        if int(target_points.shape[-1]) != 3:
            raise ValueError(f"target_points last dim must be 3, got {target_points.shape[-1]}")
        target_points = target_points.to(device=self.device, dtype=torch.float32)
        if target_valid_mask is not None:
            target_valid_mask = target_valid_mask.to(device=self.device, dtype=torch.bool)
            if tuple(target_valid_mask.shape) != tuple(target_points.shape[:2]):
                raise ValueError(
                    f"target_valid_mask must have shape {tuple(target_points.shape[:2])}, got {tuple(target_valid_mask.shape)}"
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

        contact_cloud = torch.zeros((batch_size, target_k, 6), dtype=torch.float32, device=self.device)
        contact_local_points = torch.zeros((batch_size, target_k, 3), dtype=torch.float32, device=self.device)
        contact_local_normals = torch.zeros((batch_size, target_k, 3), dtype=torch.float32, device=self.device)
        contact_link_indices = torch.full((batch_size, target_k), -1, dtype=torch.long, device=self.device)
        contact_parent_sparse_indices = torch.full((batch_size, target_k), -1, dtype=torch.long, device=self.device)
        contact_valid_mask = torch.zeros((batch_size, target_k), dtype=torch.bool, device=self.device)
        fallback_samples = 0
        active_counts = sparse_contact_mask.long().sum(dim=1)
        max_candidates = int(active_counts.max().item()) * bank_size if batch_size > 0 else 0
        num_targets = int(target_points.shape[1]) if int(target_points.ndim) >= 2 else 0
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
                chunk_size,
                max_active_chunk,
                bank_size,
                3,
            ).reshape(chunk_size, -1, 3)
            candidate_local_normals_chunk = bank_normals_local.index_select(0, flat_active_idx).reshape(
                chunk_size,
                max_active_chunk,
                bank_size,
                3,
            ).reshape(chunk_size, -1, 3)
            candidate_link_indices_chunk = template_link_idx.index_select(0, flat_active_idx).reshape(
                chunk_size,
                max_active_chunk,
                1,
            ).expand(-1, -1, bank_size).reshape(chunk_size, -1)
            candidate_parent_sparse_indices_chunk = padded_active_idx.unsqueeze(-1).expand(
                -1,
                -1,
                bank_size,
            ).reshape(chunk_size, -1)
            candidate_valid_mask_chunk = active_valid_mask.unsqueeze(-1).expand(-1, -1, bank_size).reshape(
                chunk_size,
                -1,
            )

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
                    0,
                    selected_idx,
                )
                contact_valid_mask[batch_idx] = True

        if warn_on_empty_pruned and fallback_samples > 0:
            warnings.warn(
                "Focused contact pruning removed all second-pass candidates for "
                f"{fallback_samples}/{batch_size} samples; falling back to top-scoring focused candidates. "
                "This likely indicates a bug or misconfiguration.",
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
            return {key: value[0] for key, value in out.items()}
        return out


    def sample_random_q(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        batch_size = int(batch_size)
        lower, upper = self.pk_chain.get_joint_limits()
        lower_t = torch.tensor(lower, dtype=torch.float32, device=self.device)
        upper_t = torch.tensor(upper, dtype=torch.float32, device=self.device)
        finite = torch.isfinite(lower_t) & torch.isfinite(upper_t)
        q = torch.zeros((batch_size, self.dof), dtype=torch.float32, device=self.device)
        if finite.any():
            u = torch.rand((batch_size, self.dof), device=self.device, generator=generator)
            mode = torch.rand((batch_size, 1), device=self.device, generator=generator)
            correlated = 0.75 * torch.rand((batch_size, 1), device=self.device, generator=generator) + 0.25 * u
            extreme_choice = torch.randint(
                0,
                2,
                (batch_size, 1),
                device=self.device,
                generator=generator,
            ).bool()
            extreme = torch.where(extreme_choice, u.pow(2.5), 1.0 - u.pow(2.5))
            center = 0.5 + 0.5 * (2.0 * u - 1.0).pow(3)
            t = torch.where(
                mode < 0.25,
                u,
                torch.where(mode < 0.55, correlated, torch.where(mode < 0.80, extreme, center)),
            ).clamp(0.0, 1.0)
            finite_mask = finite.unsqueeze(0).expand(batch_size, -1)
            q[finite_mask] = (
                lower_t.unsqueeze(0).expand(batch_size, -1)[finite_mask]
                + t[finite_mask]
                * (upper_t.unsqueeze(0).expand(batch_size, -1)[finite_mask] - lower_t.unsqueeze(0).expand(batch_size, -1)[finite_mask])
            )
        if (~finite).any():
            q[:, ~finite] = (torch.rand((batch_size, int((~finite).sum().item())), device=self.device, generator=generator) * 2.0 - 1.0) * torch.pi
        q = self._zero_base_translation_in_q(self._apply_fixed_zero_joint_constraints(q))
        if batch_size == 1:
            return q[0]
        return q

    def mix_q_by_contact_mask(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        squeeze = False
        if q1.ndim == 1:
            q1 = q1.unsqueeze(0)
            squeeze = True
        if q2.ndim == 1:
            q2 = q2.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)

        if q1.shape != q2.shape:
            raise ValueError(f"q1 and q2 must have the same shape, got {tuple(q1.shape)} vs {tuple(q2.shape)}")
        if q1.ndim != 2:
            raise ValueError(f"Expected q tensors with shape (B, dof), got {tuple(q1.shape)}")
        if q1.shape[-1] != self.dof:
            raise ValueError(f"Expected q last dim {self.dof}, got {q1.shape[-1]}")
        if mask.ndim != 2 or int(mask.shape[0]) != int(q1.shape[0]):
            raise ValueError(
                f"mask must have shape (B, {self.surface_num_points}), got {tuple(mask.shape)} for batch {q1.shape[0]}"
            )
        if int(mask.shape[-1]) != int(self.surface_num_points):
            raise ValueError(
                f"mask last dim must match surface_num_points={self.surface_num_points}, got {mask.shape[-1]}"
            )

        q1 = q1.to(self.device)
        q2 = q2.to(self.device)
        mask = mask.to(self.device).bool()
        joint_active = self.get_joint_active_by_contact_mask(mask)
        mixed = torch.where(joint_active, q2, q1)
        mixed = self._apply_fixed_zero_joint_constraints(mixed)
        if squeeze:
            return mixed.squeeze(0)
        return mixed

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


    @staticmethod
    def _build_topk_mask_from_sorted(
        sorted_idx: torch.Tensor,
        k_values: torch.Tensor,
        n_points: int,
    ) -> torch.Tensor:
        """Convert sorted indices and per-sample K values into boolean masks."""
        return _build_topk_mask_from_sorted_shared(sorted_idx, k_values, n_points)

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
        """Score source surface points against a target cloud and threshold the result."""
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


    def _sample_virtual_patch_geometry_batch(
        self,
        *,
        anchor_points: torch.Tensor,
        anchor_normals: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        anchor_shift_min: float | torch.Tensor,
        anchor_shift_max: float | torch.Tensor,
        max_plane_extent_min: float | torch.Tensor,
        max_plane_extent_max: float | torch.Tensor,
        patch_shift_power: float,
        patch_extent_power: float,
        normal_jitter_max_deg: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from robot_model.object_patches import _sample_virtual_patch_geometry_batch as _shared
        return _shared(
            device=self.device,
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

    def _sample_virtual_object_patches_batch(
        self,
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
        from robot_model.object_patches import sample_object_patches_batch as _shared
        return _shared(
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
        return self._sample_virtual_object_patches_batch(
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


    def get_canonical_q(self) -> torch.Tensor:
        lower, upper = self.pk_chain.get_joint_limits()
        lower_t = torch.tensor(lower, dtype=torch.float32, device=self.device)
        upper_t = torch.tensor(upper, dtype=torch.float32, device=self.device)
        finite = torch.isfinite(lower_t) & torch.isfinite(upper_t)

        canonical = torch.zeros(self.dof, dtype=torch.float32, device=self.device)
        canonical[finite] = lower_t[finite] * 0.75 + upper_t[finite] * 0.25
        return self._apply_fixed_zero_joint_constraints(canonical)


    def get_trimesh_q(self, q: torch.Tensor, mode: str = "active") -> Dict[str, object]:
        self.update_status(q)
        mode_meshes, _ = self._get_visual_assets(mode)

        vertices = []
        faces = []
        vertex_offset = 0
        parts: Dict[str, trimesh.Trimesh] = {}

        for link_name, mesh in mode_meshes.items():
            if link_name in self.hidden_visual_link_names:
                continue
            if link_name not in self.frame_status:
                continue
            tf = self.frame_status[link_name].get_matrix()[0].detach().cpu().numpy()
            part_mesh = mesh.copy()
            part_mesh.apply_transform(tf)
            parts[link_name] = part_mesh

            vertices.append(part_mesh.vertices)
            faces.append(part_mesh.faces + vertex_offset)
            vertex_offset += len(part_mesh.vertices)

        if vertices:
            all_vertices = np.vstack(vertices)
            all_faces = np.vstack(faces)
            visual = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
        else:
            visual = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))

        return {"visual": visual, "parts": parts}


def create_robot_model(
    robot_name: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_points: int = 512,
    assets_dir: Path = DEFAULT_ASSETS_DIR,
    remove_special_bases: bool = True,
    remove_special_bases_override: bool = False,
    shadowhand_mode: Optional[str] = None,
    load_surface_assets: bool = True,
    **robot_model_kwargs: Any,
) -> RobotModel:
    """Resolve a robot asset key and instantiate the corresponding ``RobotModel``."""
    mano_canonical = _is_mano_robot_name(robot_name)
    if mano_canonical is not None:
        from robot_model.human_hand_model import HumanHandModel
        side = "right" if mano_canonical.endswith("right") else "left"
        return HumanHandModel(
            robot_name=mano_canonical,
            side=side,
            device=device,
            num_points=int(num_points),
            load_surface_assets=bool(load_surface_assets),
        )

    composite_canonical = _COMPOSITE_ROBOT_NAME_CANDIDATES.get(_normalize_robot_query(str(robot_name)))
    if composite_canonical is not None:
        from robot_model.composite_gripper_stick_model import RobotiqGripperStickModel
        return RobotiqGripperStickModel(
            device=device,
            num_points=int(num_points),
            load_surface_assets=bool(load_surface_assets),
            **robot_model_kwargs,
        )

    if robot_model_kwargs:
        raise ValueError(
            f"Unsupported robot_model_kwargs for {robot_name!r}: {sorted(robot_model_kwargs.keys())}"
        )


    meta = discover_robot_assets(assets_dir)
    robot_name = resolve_robot_model_name(robot_name, assets_dir=assets_dir, meta=meta)

    robot_rel_path = Path(meta[robot_name]["path"])
    robot_path = Path(assets_dir) / robot_rel_path
    if not robot_path.exists():
        raise FileNotFoundError(f"Robot asset does not exist: {robot_path}")

    if remove_special_bases:
        robot_path = ensure_base_removed_asset(
            robot_name=robot_name,
            robot_path=robot_path,
            force_override=remove_special_bases_override,
        )
    robot_path = ensure_virtual_floating_base_asset(
        robot_name=robot_name,
        robot_path=robot_path,
        force_override=remove_special_bases_override,
    )

    return RobotModel(
        robot_name,
        robot_path,
        device=device,
        num_points=num_points,
        shadowhand_mode=shadowhand_mode,
        load_surface_assets=load_surface_assets,
    )


def dump_robot_assets_index(
    out_path: Path = DEFAULT_ASSETS_DIR / "robot_assets_index.json",
    assets_dir: Path = DEFAULT_ASSETS_DIR,
) -> None:
    meta = discover_robot_assets(assets_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    dump_robot_assets_index()
    build_manipulator_robot_lists()
    print(f"Wrote {DEFAULT_ASSETS_DIR / 'robot_assets_index.json'}")
    print(f"Wrote {DEFAULT_MANIPULATOR_LISTS_JSON}")
