from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh

from networks.residual_action_utils import euler_xyz_to_matrix, matrix_to_euler_xyz


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ROBOTIQ_XML = ROOT_DIR / "assets" / "composite" / "robotiq_2f85" / "robotiq_2f85.xml"
GRIPPER_CONTACT_CANDIDATE_MODES = ("all", "pads", "pads_inner_bottom", "pads_inner_upper", "pads_no_outer")
SELECTED_PAD_FACE_CANDIDATE_MODES = ("pads_inner_bottom", "pads_inner_upper", "pads_no_outer")


def _sample_uniform_so3_euler_xyz(
    batch_size: int,
    *,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    u1 = torch.rand((int(batch_size),), dtype=torch.float32, device=device, generator=generator)
    u2 = torch.rand((int(batch_size),), dtype=torch.float32, device=device, generator=generator)
    u3 = torch.rand((int(batch_size),), dtype=torch.float32, device=device, generator=generator)
    qx = torch.sqrt(1.0 - u1) * torch.sin(2.0 * math.pi * u2)
    qy = torch.sqrt(1.0 - u1) * torch.cos(2.0 * math.pi * u2)
    qz = torch.sqrt(u1) * torch.sin(2.0 * math.pi * u3)
    qw = torch.sqrt(u1) * torch.cos(2.0 * math.pi * u3)
    x2, y2, z2 = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    rot = torch.stack(
        [
            torch.stack([1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1),
            torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)], dim=-1),
            torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)], dim=-1),
        ],
        dim=-2,
    )
    return matrix_to_euler_xyz(rot)


class _CompositeChain:
    def __init__(self) -> None:
        self._joint_limits = (
            np.asarray(
                [
                    -2.0,
                    -2.0,
                    -2.0,
                    -math.pi,
                    -math.pi,
                    -math.pi,
                    0.0,
                    -2.0,
                    -2.0,
                    -2.0,
                    -math.pi,
                    -math.pi,
                    -math.pi,
                ],
                dtype=np.float32,
            ),
            np.asarray(
                [
                    2.0,
                    2.0,
                    2.0,
                    math.pi,
                    math.pi,
                    math.pi,
                    255.0,
                    2.0,
                    2.0,
                    2.0,
                    math.pi,
                    math.pi,
                    math.pi,
                ],
                dtype=np.float32,
            ),
        )

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._joint_limits


def _allocate_counts(total: int, weights: Sequence[float]) -> list[int]:
    total = int(total)
    if total <= 0:
        return [0 for _ in weights]
    weights_np = np.asarray(weights, dtype=np.float64)
    weights_np = weights_np / np.clip(weights_np.sum(), 1.0e-8, None)
    raw = weights_np * float(total)
    counts = np.floor(raw).astype(np.int64)
    counts[counts == 0] = 1
    while int(counts.sum()) > total:
        idx = int(np.argmax(counts))
        counts[idx] -= 1
    remain = total - int(counts.sum())
    if remain > 0:
        frac = raw - np.floor(raw)
        order = np.argsort(-frac)
        for i in range(remain):
            counts[int(order[i % len(order)])] += 1
    return [int(x) for x in counts.tolist()]


def _sample_cylinder_surface(
    *,
    count: int,
    radius: float,
    length: float,
    base_at_end: bool,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    radius = float(radius)
    length = float(length)
    side_area = 2.0 * math.pi * radius * length
    cap_area = math.pi * radius * radius
    probs = torch.tensor([side_area, cap_area, cap_area], dtype=torch.float32)
    probs = probs / probs.sum().clamp_min(1.0e-8)
    part = torch.multinomial(probs, int(count), replacement=True, generator=generator)
    points = torch.zeros((int(count), 3), dtype=torch.float32)
    normals = torch.zeros_like(points)

    side = part == 0
    if bool(side.any()):
        n = int(side.sum().item())
        theta = 2.0 * math.pi * torch.rand((n,), dtype=torch.float32, generator=generator)
        if bool(base_at_end):
            z = torch.rand((n,), dtype=torch.float32, generator=generator) * length
        else:
            z = (torch.rand((n,), dtype=torch.float32, generator=generator) - 0.5) * length
        c, s = torch.cos(theta), torch.sin(theta)
        points[side] = torch.stack([radius * c, radius * s, z], dim=-1)
        normals[side] = torch.stack([c, s, torch.zeros_like(c)], dim=-1)

    caps = ((1, length, 1.0), (2, 0.0, -1.0)) if bool(base_at_end) else (
        (1, length / 2.0, 1.0),
        (2, -length / 2.0, -1.0),
    )
    for part_id, z_value, sign in caps:
        cap = part == int(part_id)
        if not bool(cap.any()):
            continue
        n = int(cap.sum().item())
        theta = 2.0 * math.pi * torch.rand((n,), dtype=torch.float32, generator=generator)
        r = radius * torch.sqrt(torch.rand((n,), dtype=torch.float32, generator=generator))
        points[cap] = torch.stack([r * torch.cos(theta), r * torch.sin(theta), torch.full_like(r, z_value)], dim=-1)
        normals[cap, 2] = float(sign)
    return points, normals


def _sample_mesh_surface(
    mesh: trimesh.Trimesh,
    *,
    count: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    count = int(count)
    if count <= 0:
        empty = torch.zeros((0, 3), dtype=torch.float32, device=device)
        return empty, empty
    try:
        pts_np, face_idx = trimesh.sample.sample_surface(mesh, count, seed=int(seed))
        face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
        nrm_np = face_normals[np.asarray(face_idx, dtype=np.int64)]
    except Exception:
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        if int(verts.shape[0]) == 0:
            pts = torch.zeros((count, 3), dtype=torch.float32, device=device)
            return pts, torch.zeros_like(pts)
        rng = np.random.default_rng(int(seed))
        idx = rng.integers(0, int(verts.shape[0]), size=(count,))
        pts_np = verts[idx]
        nrm_np = np.zeros_like(pts_np, dtype=np.float32)
    pts = torch.as_tensor(pts_np, dtype=torch.float32, device=device)
    nrms = torch.as_tensor(nrm_np, dtype=torch.float32, device=device)
    nrms = nrms / torch.norm(nrms, dim=-1, keepdim=True).clamp_min(1.0e-8)
    return pts, nrms


def _sample_mesh_surface_face_subset(
    mesh: trimesh.Trimesh,
    *,
    face_mask: torch.Tensor,
    count: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    selected = torch.nonzero(face_mask.detach().cpu().bool(), as_tuple=False).view(-1).numpy()
    if int(selected.shape[0]) == 0:
        raise RuntimeError("Cannot sample mesh surface from an empty face subset")
    submesh = mesh.submesh([selected], append=True, repair=False)
    return _sample_mesh_surface(submesh, count=count, seed=seed, device=device)


class RobotiqGripperStickModel:
    robot_name = "composite/robotiq_gripper_stick"

    def __init__(
        self,
        *,
        device: torch.device | str,
        num_points: int = 1024,
        stick_radius: float = 0.008,
        stick_length: float = 0.08,
        stick_base_at_end: bool = True,
        robotiq_xml_path: str | Path | None = None,
        gripper_surface_fraction: float = 0.70,
        gripper_contact_candidate_mode: str = "all",
        gripper_contact_norm_mode: str = "surface",
        gripper_contact_norm_variation_scale: float = 0.20,
        gripper_closed_joint_value: float = 0.824,
        load_surface_assets: bool = True,
        **_: Any,
    ) -> None:
        self.device = torch.device(device)
        self.dof = 13
        self.pk_chain = _CompositeChain()
        self.joint_orders = [
            "gripper_x",
            "gripper_y",
            "gripper_z",
            "gripper_roll",
            "gripper_pitch",
            "gripper_yaw",
            "gripper_cmd",
            "stick_x",
            "stick_y",
            "stick_z",
            "stick_roll",
            "stick_pitch",
            "stick_yaw",
        ]
        self.pose_groups = [
            {"name": "gripper", "translation_indices": (0, 1, 2), "rotation_indices": (3, 4, 5)},
            {"name": "stick", "translation_indices": (7, 8, 9), "rotation_indices": (10, 11, 12)},
        ]
        self.base_translation_indices = [0, 1, 2]
        self.base_rotation_indices = [3, 4, 5]
        self.base_pose_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        self.fixed_zero_joint_indices: list[int] = []
        self.hidden_visual_link_names: set[str] = set()
        self.excluded_contact_link_names: set[str] = set()
        self.surface_num_points = int(num_points)
        self.link_num_points = int(num_points)
        if self.surface_num_points < 16:
            raise ValueError("robotiq_gripper_stick needs at least 16 surface points")
        self.stick_radius = float(stick_radius)
        self.stick_length = float(stick_length)
        self.stick_base_at_end = bool(stick_base_at_end)
        self.gripper_contact_candidate_mode = str(gripper_contact_candidate_mode).strip().lower()
        if self.gripper_contact_candidate_mode not in GRIPPER_CONTACT_CANDIDATE_MODES:
            raise ValueError(
                "gripper_contact_candidate_mode must be one of "
                f"{', '.join(repr(x) for x in GRIPPER_CONTACT_CANDIDATE_MODES)}, "
                f"got {gripper_contact_candidate_mode!r}"
            )
        self.gripper_contact_norm_mode = str(gripper_contact_norm_mode).strip().lower()
        if self.gripper_contact_norm_mode not in {"surface", "pad_center_radial", "pad_face_positional"}:
            raise ValueError(
                "gripper_contact_norm_mode must be 'surface', 'pad_center_radial', or 'pad_face_positional', "
                f"got {gripper_contact_norm_mode!r}"
            )
        self.gripper_contact_norm_variation_scale = float(max(0.0, gripper_contact_norm_variation_scale))
        self.gripper_closed_joint_value = float(gripper_closed_joint_value)
        self.gripper_cmd_to_joint_scale = self.gripper_closed_joint_value / 255.0
        self.robotiq_xml_path = Path(robotiq_xml_path) if robotiq_xml_path is not None else DEFAULT_ROBOTIQ_XML
        self.load_surface_assets = bool(load_surface_assets)

        self._gripper_model = self._build_gripper_model()
        self._gripper_link_names = [str(x) for x in self._gripper_model.mesh_link_names]
        self._gripper_link_count = len(self._gripper_link_names)
        if self._gripper_link_count <= 0:
            raise RuntimeError(f"Robotiq model has no surface mesh links: {self.robotiq_xml_path}")
        self._gripper_dof = int(self._gripper_model.dof)
        if self._gripper_dof <= 0:
            raise RuntimeError(f"Robotiq model has no internal joints: {self.robotiq_xml_path}")

        stick_min = min(32, max(1, self.surface_num_points - self._gripper_link_count))
        requested_stick = int(round(self.surface_num_points * (1.0 - float(gripper_surface_fraction))))
        self.stick_surface_num_points = int(max(stick_min, requested_stick))
        self.stick_surface_num_points = min(
            self.stick_surface_num_points,
            self.surface_num_points - self._gripper_link_count,
        )
        self.gripper_surface_num_points = int(self.surface_num_points - self.stick_surface_num_points)
        if self.gripper_surface_num_points <= 0 or self.stick_surface_num_points <= 0:
            raise ValueError(
                "robotiq_gripper_stick could not allocate positive surface counts to both manipulators "
                f"(total={self.surface_num_points}, gripper_links={self._gripper_link_count})"
            )

        self.mesh_link_names = self._gripper_link_names + ["stick"]
        self._stick_link_index = self._gripper_link_count
        self._gripper_pad_link_indices = tuple(
            i for i, name in enumerate(self._gripper_link_names) if str(name).endswith("_pad")
        )
        if (
            self.gripper_contact_candidate_mode in {"pads", *SELECTED_PAD_FACE_CANDIDATE_MODES}
            and len(self._gripper_pad_link_indices) == 0
        ):
            raise RuntimeError("Robotiq pad-only contact mode requested, but no *_pad links were found")
        self._gripper_contact_link_indices = (
            self._gripper_pad_link_indices
            if self.gripper_contact_candidate_mode in {"pads", *SELECTED_PAD_FACE_CANDIDATE_MODES}
            else tuple(range(self._gripper_link_count))
        )
        self.meshes_original = self._build_meshes()
        self.surface_template_by_link: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._build_surface_template()
        self.vertices_original = {name: pts for name, (pts, _) in self.surface_template_by_link.items()}
        self.surface_template_hash = self.get_surface_template_hash()
        self.surface_graph_neighbors = self._build_surface_graph_neighbors()
        self.surface_graph_neighbors_strict = self.surface_graph_neighbors
        self.surface_expansion_bank_points_local = torch.zeros(
            (self.surface_num_points, 0, 3), dtype=torch.float32, device=self.device
        )
        self.surface_expansion_bank_normals_local = torch.zeros(
            (self.surface_num_points, 0, 3), dtype=torch.float32, device=self.device
        )
        self.joint_descendant_link_mask = self._build_joint_descendant_link_mask()
        self.composite_contact_groups = [
            {
                "name": "gripper",
                "link_indices": self._gripper_contact_link_indices,
                "default_contact_count_range": (1, 3),
                "slot_count": 3,
            },
            {
                "name": "stick",
                "link_indices": (self._stick_link_index,),
                "default_contact_count_range": (1, 3),
                "slot_count": 3,
            },
        ]
        self.hand_fps_local_points = None
        self.hand_fps_local_normals = None
        self.hand_fps_link_indices = None
        self.hand_fps_num_points = 0

    def _build_gripper_model(self) -> Any:
        if not self.robotiq_xml_path.exists():
            raise FileNotFoundError(f"Robotiq 2F-85 MJCF does not exist: {self.robotiq_xml_path}")
        from robot_model.robot_model import RobotModel

        return RobotModel(
            robot_name="robotiq_2f85",
            robot_path=self.robotiq_xml_path,
            device=self.device,
            num_points=int(
                self.gripper_surface_num_points
                if hasattr(self, "gripper_surface_num_points")
                else self.surface_num_points
            ),
            load_surface_assets=False,
        )

    def _build_meshes(self) -> Dict[str, trimesh.Trimesh]:
        meshes = {name: mesh.copy() for name, mesh in self._gripper_model.meshes_original.items()}
        stick = trimesh.creation.cylinder(radius=self.stick_radius, height=self.stick_length, sections=32)
        if self.stick_base_at_end:
            stick.apply_translation([0.0, 0.0, 0.5 * self.stick_length])
        meshes["stick"] = stick
        return meshes

    def _build_surface_template(self) -> None:
        points: list[torch.Tensor] = []
        normals: list[torch.Tensor] = []
        links: list[torch.Tensor] = []

        seed_base = 41_003 + int(self.surface_num_points)
        if self._uses_selected_pad_face_sampling():
            sampled_links = self._sample_selected_pad_faces(
                total_count=int(self.gripper_surface_num_points),
                seed_base=seed_base,
            )
        else:
            gripper_meshes = {name: self.meshes_original[name] for name in self._gripper_link_names}
            weights = [float(max(mesh.area, 1.0e-8)) for mesh in gripper_meshes.values()]
            gripper_counts = _allocate_counts(self.gripper_surface_num_points, weights)
            sampled_links = []
            for link_idx, (name, count) in enumerate(zip(self._gripper_link_names, gripper_counts)):
                pts, nrms = _sample_mesh_surface(
                    gripper_meshes[name],
                    count=int(count),
                    seed=seed_base + link_idx * 104_729,
                    device=self.device,
                )
                sampled_links.append((int(link_idx), str(name), pts, nrms))
        for link_idx, name, pts, nrms in sampled_links:
            self.surface_template_by_link[str(name)] = (pts, nrms)
            points.append(pts)
            normals.append(nrms)
            links.append(torch.full((int(pts.shape[0]),), int(link_idx), dtype=torch.long, device=self.device))

        generator = torch.Generator(device="cpu")
        generator.manual_seed(97_531 + int(self.surface_num_points))
        stick_pts, stick_nrms = _sample_cylinder_surface(
            count=self.stick_surface_num_points,
            radius=self.stick_radius,
            length=self.stick_length,
            base_at_end=self.stick_base_at_end,
            generator=generator,
        )
        stick_pts = stick_pts.to(device=self.device, dtype=torch.float32)
        stick_nrms = stick_nrms.to(device=self.device, dtype=torch.float32)
        self.surface_template_by_link["stick"] = (stick_pts, stick_nrms)
        points.append(stick_pts)
        normals.append(stick_nrms)
        links.append(
            torch.full(
                (int(self.stick_surface_num_points),),
                int(self._stick_link_index),
                dtype=torch.long,
                device=self.device,
            )
        )

        self.surface_template_points_local = torch.cat(points, dim=0).to(self.device, dtype=torch.float32)
        self.surface_template_normals_local = torch.cat(normals, dim=0).to(self.device, dtype=torch.float32)
        self.surface_template_link_indices = torch.cat(links, dim=0).to(self.device, dtype=torch.long)
        if int(self.surface_template_points_local.shape[0]) != int(self.surface_num_points):
            raise RuntimeError(
                f"Surface template size mismatch: got {self.surface_template_points_local.shape[0]}, "
                f"expected {self.surface_num_points}"
            )

        q0 = torch.zeros((1, self.dof), dtype=torch.float32, device=self.device)
        physical_pts_w, physical_nrms_w = self.get_surface_points_normals_batch(q0)
        self.surface_contact_candidate_mask = self._build_surface_contact_candidate_mask(
            physical_pts_w[0],
            physical_nrms_w[0],
        )
        self._apply_contact_normal_mode()
        pts_w, nrms_w = self.get_surface_points_normals_batch(q0)
        self.surface_template_points_canonical_world = pts_w[0].detach().clone()
        self.surface_template_normals_canonical_world = nrms_w[0].detach().clone()
        self.surface_graph_points_canonical_world = self.surface_template_points_canonical_world
        self.surface_graph_normals_canonical_world = self.surface_template_normals_canonical_world

    def _mesh_bounds_center_local(self, link_name: str) -> torch.Tensor:
        mesh = self.meshes_original[str(link_name)]
        bounds = torch.as_tensor(
            np.array(mesh.bounds, dtype=np.float32, copy=True),
            dtype=torch.float32,
            device=self.device,
        )
        return bounds.mean(dim=0)

    def _uses_selected_pad_face_sampling(self) -> bool:
        return self.gripper_contact_candidate_mode in SELECTED_PAD_FACE_CANDIDATE_MODES

    def _pad_centers_canonical_world(self, pad_link_indices: tuple[int, ...]) -> dict[int, torch.Tensor]:
        local_centers = torch.stack(
            [self._mesh_bounds_center_local(self.mesh_link_names[int(idx)]) for idx in pad_link_indices],
            dim=0,
        )
        center_links = torch.as_tensor(pad_link_indices, dtype=torch.long, device=self.device)
        q0 = torch.zeros((self.dof,), dtype=torch.float32, device=self.device)
        centers_world, _ = self.transform_attached_points_normals_batch(
            q=q0,
            local_points=local_centers,
            link_indices=center_links,
            local_normals=None,
        )
        return {int(idx): centers_world[row] for row, idx in enumerate(pad_link_indices)}

    def _selected_pad_face_mask(self, *, pad_link_idx: int, mesh: trimesh.Trimesh) -> torch.Tensor:
        face_centers = torch.as_tensor(
            np.asarray(mesh.triangles_center, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        face_normals = torch.as_tensor(
            np.asarray(mesh.face_normals, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        link_indices = torch.full((int(face_centers.shape[0]),), int(pad_link_idx), dtype=torch.long, device=self.device)
        q0 = torch.zeros((self.dof,), dtype=torch.float32, device=self.device)
        centers_world, normals_world = self.transform_attached_points_normals_batch(
            q=q0,
            local_points=face_centers,
            link_indices=link_indices,
            local_normals=face_normals,
        )
        assert normals_world is not None
        pad_link_indices = tuple(int(x) for x in self._gripper_pad_link_indices)
        if len(pad_link_indices) != 2:
            raise RuntimeError(
                "Robotiq selected-pad mode expects exactly two *_pad links, "
                f"got {len(pad_link_indices)}"
            )
        pad_centers = self._pad_centers_canonical_world(pad_link_indices)
        other_idx = pad_link_indices[1] if int(pad_link_idx) == pad_link_indices[0] else pad_link_indices[0]
        to_other = pad_centers[other_idx].view(1, 3) - centers_world
        to_other = to_other / to_other.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        inner = (normals_world * to_other).sum(dim=-1) > 0.75
        if self.gripper_contact_candidate_mode == "pads_no_outer":
            outer = (normals_world * to_other).sum(dim=-1) < -0.75
            selected = ~outer
        elif self.gripper_contact_candidate_mode == "pads_inner_upper":
            side = (normals_world[:, 2] > 0.75) & (
                centers_world[:, 2] > (pad_centers[int(pad_link_idx)][2] + 1.0e-6)
            )
            selected = inner | side
        else:
            side = (normals_world[:, 2] < -0.75) & (
                centers_world[:, 2] < (pad_centers[int(pad_link_idx)][2] - 1.0e-6)
            )
            selected = inner | side
        if not bool(selected.any()):
            raise RuntimeError(
                f"Robotiq pad link {pad_link_idx} has no selected faces for "
                f"{self.gripper_contact_candidate_mode}"
            )
        return selected

    def _sample_selected_pad_faces(
        self,
        *,
        total_count: int,
        seed_base: int,
    ) -> list[tuple[int, str, torch.Tensor, torch.Tensor]]:
        pad_link_indices = tuple(int(x) for x in self._gripper_pad_link_indices)
        if len(pad_link_indices) == 0:
            raise RuntimeError("Selected pad-face sampling requested, but no *_pad links were found")
        pad_entries: list[tuple[int, str, trimesh.Trimesh, torch.Tensor, float]] = []
        for pad_idx in pad_link_indices:
            name = str(self.mesh_link_names[int(pad_idx)])
            mesh = self.meshes_original[name]
            face_mask = self._selected_pad_face_mask(pad_link_idx=int(pad_idx), mesh=mesh)
            area = float(np.asarray(mesh.area_faces, dtype=np.float64)[face_mask.detach().cpu().numpy()].sum())
            pad_entries.append((int(pad_idx), name, mesh, face_mask, max(area, 1.0e-8)))
        counts = _allocate_counts(int(total_count), [entry[4] for entry in pad_entries])
        sampled: list[tuple[int, str, torch.Tensor, torch.Tensor]] = []
        for entry_idx, ((pad_idx, name, mesh, face_mask, _), count) in enumerate(zip(pad_entries, counts)):
            pts, nrms = _sample_mesh_surface_face_subset(
                mesh,
                face_mask=face_mask,
                count=int(count),
                seed=int(seed_base) + int(pad_idx) * 104_729 + entry_idx,
                device=self.device,
            )
            sampled.append((pad_idx, name, pts, nrms))
        return sampled

    def _build_surface_contact_candidate_mask(
        self,
        physical_points_world: torch.Tensor,
        physical_normals_world: torch.Tensor,
    ) -> torch.Tensor:
        _ = physical_points_world, physical_normals_world
        link_indices = self.surface_template_link_indices.to(device=self.device, dtype=torch.long)
        if self.gripper_contact_candidate_mode == "all":
            return torch.ones((self.surface_num_points,), dtype=torch.bool, device=self.device)

        stick_mask = link_indices == int(self._stick_link_index)
        pad_mask = torch.zeros((self.surface_num_points,), dtype=torch.bool, device=self.device)
        for pad_idx in self._gripper_pad_link_indices:
            pad_mask |= link_indices == int(pad_idx)
        if self.gripper_contact_candidate_mode in {"pads", *SELECTED_PAD_FACE_CANDIDATE_MODES}:
            return stick_mask | pad_mask
        raise RuntimeError(f"Unsupported gripper_contact_candidate_mode={self.gripper_contact_candidate_mode!r}")

    def _apply_contact_normal_mode(self) -> None:
        if self.gripper_contact_norm_mode == "surface":
            return
        normals = self.surface_template_normals_local.clone()
        link_indices = self.surface_template_link_indices.to(device=self.device, dtype=torch.long)
        for pad_idx in self._gripper_pad_link_indices:
            pad_idx = int(pad_idx)
            pad_mask = link_indices == pad_idx
            if not bool(pad_mask.any()):
                continue
            link_name = self.mesh_link_names[pad_idx]
            center = self._mesh_bounds_center_local(str(link_name))
            pad_points = self.surface_template_points_local[pad_mask]
            if self.gripper_contact_norm_mode == "pad_center_radial":
                new_normals = pad_points - center.view(1, 3)
                new_normals = new_normals / new_normals.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
            else:
                new_normals = self._build_pad_face_positional_normals(
                    link_name=str(link_name),
                    points_local=pad_points,
                    base_normals=normals[pad_mask],
                )
            normals[pad_mask] = new_normals
            pts = self.surface_template_points_local[pad_mask]
            self.surface_template_by_link[str(link_name)] = (pts, new_normals)
        self.surface_template_normals_local = normals

    def _build_pad_face_positional_normals(
        self,
        *,
        link_name: str,
        points_local: torch.Tensor,
        base_normals: torch.Tensor,
    ) -> torch.Tensor:
        mesh = self.meshes_original[str(link_name)]
        bounds = torch.as_tensor(
            np.array(mesh.bounds, dtype=np.float32, copy=True),
            dtype=torch.float32,
            device=self.device,
        )
        center = bounds.mean(dim=0)
        half_extent = ((bounds[1] - bounds[0]) * 0.5).clamp_min(1.0e-6)
        rel = ((points_local - center.view(1, 3)) / half_extent.view(1, 3)).clamp(-1.0, 1.0)
        base = base_normals / base_normals.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        tangent = rel - (rel * base).sum(dim=-1, keepdim=True) * base
        varied = base + float(self.gripper_contact_norm_variation_scale) * tangent
        return varied / varied.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)

    def _build_surface_graph_neighbors(self) -> list[list[int]]:
        neighbors: list[list[int]] = [[] for _ in range(self.surface_num_points)]
        links = self.surface_template_link_indices.detach().cpu()
        pts = self.surface_template_points_local.detach().cpu()
        for link_idx in range(len(self.mesh_link_names)):
            idx = torch.nonzero(links == int(link_idx), as_tuple=False).view(-1)
            if int(idx.numel()) <= 1:
                continue
            p = pts.index_select(0, idx)
            k = min(6, int(idx.numel()) - 1)
            dist = torch.cdist(p, p)
            nn = dist.topk(k + 1, largest=False).indices[:, 1:]
            for row, src in enumerate(idx.tolist()):
                neighbors[int(src)] = [int(idx[int(j)].item()) for j in nn[row]]
        return neighbors

    def _build_joint_descendant_link_mask(self) -> torch.Tensor:
        mask = torch.zeros((self.dof, len(self.mesh_link_names)), dtype=torch.bool, device=self.device)
        mask[0:6, : self._gripper_link_count] = True
        inner_mask = self._gripper_model.joint_descendant_link_mask.to(device=self.device, dtype=torch.bool)
        if inner_mask.numel() > 0:
            mask[6, : self._gripper_link_count] = inner_mask.any(dim=0)
        mask[7:13, self._stick_link_index] = True
        return mask

    def _gripper_internal_q(self, q: torch.Tensor) -> torch.Tensor:
        q = self._validate_q(q)
        cmd = q[:, 6].clamp(0.0, 255.0)
        joint_value = cmd * float(self.gripper_cmd_to_joint_scale)
        return joint_value.view(-1, 1).expand(-1, int(self._gripper_dof)).contiguous()

    def _stick_sdf_local(self, points_local: torch.Tensor) -> torch.Tensor:
        radial = torch.linalg.norm(points_local[..., :2], dim=-1) - float(self.stick_radius)
        z = points_local[..., 2]
        z_centered = z - (0.5 * float(self.stick_length) if self.stick_base_at_end else 0.0)
        axial = torch.abs(z_centered) - 0.5 * float(self.stick_length)
        outside = torch.linalg.norm(torch.stack([radial.clamp_min(0.0), axial.clamp_min(0.0)], dim=-1), dim=-1)
        inside = torch.minimum(torch.maximum(radial, axial), torch.zeros_like(radial))
        return outside + inside

    def gripper_surface_to_stick_sdf_batch(self, q: torch.Tensor) -> torch.Tensor:
        q = self._validate_q(q)
        batch = int(q.shape[0])
        link_indices_1d = self.surface_template_link_indices.to(device=self.device, dtype=torch.long)
        gripper_mask = link_indices_1d < int(self._stick_link_index)
        if not bool(gripper_mask.any()):
            return torch.full((batch, 0), float("inf"), dtype=torch.float32, device=self.device)
        local_points = self.surface_template_points_local.index_select(
            0, torch.nonzero(gripper_mask, as_tuple=False).view(-1)
        )
        gripper_link_indices = link_indices_1d[gripper_mask]
        gripper_points_world, _ = self.transform_attached_points_normals_batch(
            q=q,
            local_points=local_points.unsqueeze(0).expand(batch, -1, -1),
            link_indices=gripper_link_indices.unsqueeze(0).expand(batch, -1),
            local_normals=None,
        )
        r_stick = euler_xyz_to_matrix(q[:, 10:13])
        t_stick = q[:, 7:10]
        points_stick = torch.matmul(gripper_points_world - t_stick.unsqueeze(1), r_stick)
        return self._stick_sdf_local(points_stick)

    def gripper_stick_min_sdf_batch(self, q: torch.Tensor) -> torch.Tensor:
        sdf = self.gripper_surface_to_stick_sdf_batch(q)
        if int(sdf.shape[1]) == 0:
            return torch.full((int(sdf.shape[0]),), float("inf"), dtype=torch.float32, device=self.device)
        return sdf.min(dim=1).values

    def gripper_stick_collision_free_mask(self, q: torch.Tensor, *, margin: float = 0.0) -> torch.Tensor:
        return self.gripper_stick_min_sdf_batch(q) >= float(margin)


    def get_surface_template_hash(self) -> str:
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(self.surface_template_points_local.detach().cpu().numpy().astype(np.float32)).tobytes())
        h.update(np.ascontiguousarray(self.surface_template_normals_local.detach().cpu().numpy().astype(np.float32)).tobytes())
        h.update(np.ascontiguousarray(self.surface_template_link_indices.detach().cpu().numpy().astype(np.int64)).tobytes())
        return h.hexdigest()


    def _apply_fixed_zero_joint_constraints(self, q: torch.Tensor) -> torch.Tensor:
        return q

    def _zero_base_translation_in_q(self, q: torch.Tensor) -> torch.Tensor:
        q_out = q.clone()
        q_out[..., [0, 1, 2, 7, 8, 9]] = 0.0
        return q_out

    def _zero_base_pose_in_q(self, q: torch.Tensor) -> torch.Tensor:
        q_out = q.clone()
        q_out[..., self.base_pose_indices] = 0.0
        return q_out

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
        if base_translation is not None:
            base_translation = torch.as_tensor(base_translation, dtype=torch.float32, device=q_out.device)
            if base_translation.ndim == 1:
                base_translation = base_translation.unsqueeze(0)
            q_out[:, self.base_translation_indices] = base_translation[:, :3]
        if base_rotation_euler is not None:
            base_rotation_euler = torch.as_tensor(base_rotation_euler, dtype=torch.float32, device=q_out.device)
            if base_rotation_euler.ndim == 1:
                base_rotation_euler = base_rotation_euler.unsqueeze(0)
            q_out[:, self.base_rotation_indices] = base_rotation_euler[:, :3]
        return q_out[0] if squeeze else q_out

    def sample_uniform_base_rotation_euler_xyz(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        return _sample_uniform_so3_euler_xyz(int(batch_size), device=self.device, generator=generator)

    def sample_random_q(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        q = torch.zeros((int(batch_size), self.dof), dtype=torch.float32, device=self.device)
        q[:, 6] = torch.rand((int(batch_size),), dtype=torch.float32, device=self.device, generator=generator) * 255.0
        return q

    def get_canonical_q(self) -> torch.Tensor:
        return torch.zeros((self.dof,), dtype=torch.float32, device=self.device)

    def _validate_q(self, q: Optional[torch.Tensor]) -> torch.Tensor:
        if q is None:
            q = torch.zeros((1, self.dof), dtype=torch.float32, device=self.device)
        q = q.to(device=self.device, dtype=torch.float32)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if int(q.shape[-1]) != self.dof:
            raise ValueError(f"Expected q trailing dim {self.dof}, got {tuple(q.shape)}")
        return q

    @staticmethod
    def _apply_pose(points: torch.Tensor, normals: Optional[torch.Tensor], rot: torch.Tensor, trans: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        rot_t = rot.transpose(-1, -2).unsqueeze(1)
        points_world = torch.matmul(points.unsqueeze(-2), rot_t).squeeze(-2) + trans.unsqueeze(1)
        normals_world = None
        if normals is not None:
            normals_world = torch.matmul(normals.unsqueeze(-2), rot_t).squeeze(-2)
            normals_world = normals_world / torch.norm(normals_world, dim=-1, keepdim=True).clamp_min(1.0e-8)
        return points_world, normals_world

    def get_surface_points_normals_batch(self, q: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self._validate_q(q)
        batch = int(q.shape[0])
        local_points = self.surface_template_points_local.unsqueeze(0).expand(batch, -1, -1)
        local_normals = self.surface_template_normals_local.unsqueeze(0).expand(batch, -1, -1)
        link_indices = self.surface_template_link_indices.unsqueeze(0).expand(batch, -1)
        points, normals = self.transform_attached_points_normals_batch(
            q=q,
            local_points=local_points,
            link_indices=link_indices,
            local_normals=local_normals,
        )
        assert normals is not None
        return points, normals

    def get_surface_points_normals(self, q: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
        q = self._validate_q(q)
        batch = int(q.shape[0])
        squeeze = False
        if local_points.ndim == 2:
            local_points = local_points.unsqueeze(0)
            squeeze = True
        if link_indices.ndim == 1:
            link_indices = link_indices.unsqueeze(0)
        if local_normals is not None and local_normals.ndim == 2:
            local_normals = local_normals.unsqueeze(0)
        if int(local_points.shape[0]) == 1 and batch > 1:
            local_points = local_points.expand(batch, -1, -1)
        if int(link_indices.shape[0]) == 1 and batch > 1:
            link_indices = link_indices.expand(batch, -1)
        if local_normals is not None and int(local_normals.shape[0]) == 1 and batch > 1:
            local_normals = local_normals.expand(batch, -1, -1)

        local_points = local_points.to(device=self.device, dtype=torch.float32)
        link_indices = link_indices.to(device=self.device, dtype=torch.long)
        local_normals = None if local_normals is None else local_normals.to(device=self.device, dtype=torch.float32)
        if tuple(local_points.shape[:2]) != tuple(link_indices.shape):
            raise ValueError(
                f"local_points and link_indices must align, got {tuple(local_points.shape)} and {tuple(link_indices.shape)}"
            )
        if local_normals is not None and tuple(local_normals.shape) != tuple(local_points.shape):
            raise ValueError(
                f"local_normals must match local_points, got {tuple(local_normals.shape)} and {tuple(local_points.shape)}"
            )

        count = int(local_points.shape[1])
        valid = (link_indices >= 0) & (link_indices < len(self.mesh_link_names))
        gripper_mask = valid & (link_indices < self._gripper_link_count)
        stick_mask = valid & (link_indices == self._stick_link_index)
        points_world = torch.zeros((batch, count, 3), dtype=torch.float32, device=self.device)
        normals_world = None if local_normals is None else torch.zeros_like(points_world)

        if bool(gripper_mask.any()):
            inner_link_indices = link_indices.clamp(min=0, max=self._gripper_link_count - 1)
            gripper_points, gripper_normals = self._gripper_model.transform_attached_points_normals_batch(
                q=self._gripper_internal_q(q),
                local_points=local_points,
                link_indices=inner_link_indices,
                local_normals=local_normals,
            )
            r_gripper = euler_xyz_to_matrix(q[:, 3:6])
            t_gripper = q[:, 0:3]
            gripper_points, gripper_normals = self._apply_pose(gripper_points, gripper_normals, r_gripper, t_gripper)
            points_world = torch.where(gripper_mask.unsqueeze(-1), gripper_points, points_world)
            if normals_world is not None and gripper_normals is not None:
                normals_world = torch.where(gripper_mask.unsqueeze(-1), gripper_normals, normals_world)

        if bool(stick_mask.any()):
            r_stick = euler_xyz_to_matrix(q[:, 10:13])
            t_stick = q[:, 7:10]
            stick_points, stick_normals = self._apply_pose(local_points, local_normals, r_stick, t_stick)
            points_world = torch.where(stick_mask.unsqueeze(-1), stick_points, points_world)
            if normals_world is not None and stick_normals is not None:
                normals_world = torch.where(stick_mask.unsqueeze(-1), stick_normals, normals_world)

        points_world = torch.where(valid.unsqueeze(-1), points_world, torch.zeros_like(points_world))
        if normals_world is not None:
            normals_world = normals_world / torch.norm(normals_world, dim=-1, keepdim=True).clamp_min(1.0e-8)
            normals_world = torch.where(valid.unsqueeze(-1), normals_world, torch.zeros_like(normals_world))
        if squeeze:
            return points_world[0], None if normals_world is None else normals_world[0]
        return points_world, normals_world

    def build_hand_fps_template(self, n_points: int) -> None:
        n = int(max(0, n_points))
        if n <= 0:
            self.hand_fps_local_points = None
            self.hand_fps_local_normals = None
            self.hand_fps_link_indices = None
            self.hand_fps_num_points = 0
            return
        count = min(n, int(self.surface_template_points_local.shape[0]))
        idx = torch.linspace(0, int(self.surface_template_points_local.shape[0]) - 1, count, device=self.device)
        idx = idx.round().to(dtype=torch.long).unique(sorted=True)
        if int(idx.numel()) < count:
            extra = torch.arange(count, dtype=torch.long, device=self.device)
            idx = torch.unique(torch.cat([idx, extra]), sorted=True)[:count]
        self.hand_fps_local_points = self.surface_template_points_local.index_select(0, idx).contiguous()
        self.hand_fps_local_normals = self.surface_template_normals_local.index_select(0, idx).contiguous()
        self.hand_fps_link_indices = self.surface_template_link_indices.index_select(0, idx).contiguous()
        self.hand_fps_num_points = int(idx.numel())

    def transform_hand_fps_template_batch(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.hand_fps_local_points is None or self.hand_fps_local_normals is None or self.hand_fps_link_indices is None:
            raise RuntimeError("build_hand_fps_template must be called before transform_hand_fps_template_batch")
        q = self._validate_q(q)
        batch = int(q.shape[0])
        local_points = self.hand_fps_local_points.unsqueeze(0).expand(batch, -1, -1)
        local_normals = self.hand_fps_local_normals.unsqueeze(0).expand(batch, -1, -1)
        link_indices = self.hand_fps_link_indices.unsqueeze(0).expand(batch, -1)
        points, normals = self.transform_attached_points_normals_batch(
            q=q,
            local_points=local_points,
            link_indices=link_indices,
            local_normals=local_normals,
        )
        assert normals is not None
        return points, normals

    def _build_active_links_from_contact_mask(self, mask: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
            squeeze = True
        mask = mask.to(self.device, dtype=torch.bool)
        active = torch.zeros((int(mask.shape[0]), len(self.mesh_link_names)), dtype=torch.bool, device=self.device)
        link_indices = self.surface_template_link_indices.to(self.device, dtype=torch.long)
        for link_idx in range(len(self.mesh_link_names)):
            cols = link_indices == int(link_idx)
            if bool(cols.any()):
                active[:, link_idx] = mask[:, cols].any(dim=1)
        return active.squeeze(0) if squeeze else active

    def get_joint_active_by_contact_mask(self, mask: torch.Tensor) -> torch.Tensor:
        active_links = self._build_active_links_from_contact_mask(mask)
        squeeze = False
        if active_links.ndim == 1:
            active_links = active_links.unsqueeze(0)
            squeeze = True
        active = (active_links.unsqueeze(1) & self.joint_descendant_link_mask.unsqueeze(0)).any(dim=-1)
        return active.squeeze(0) if squeeze else active

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
        if ignore_base_pose:
            joint_active = joint_active.clone()
            joint_active[:, self.base_pose_indices] = False
        affected = (joint_active.unsqueeze(-1) & self.joint_descendant_link_mask.unsqueeze(0)).any(dim=1)
        inactive = ~affected
        return inactive.squeeze(0) if squeeze else inactive

    def mix_q_by_contact_mask(self, q1: torch.Tensor, q2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if q1.ndim == 1:
            q1 = q1.unsqueeze(0)
            squeeze = True
        if q2.ndim == 1:
            q2 = q2.unsqueeze(0)
        joint_active = self.get_joint_active_by_contact_mask(mask)[:, : self.dof]
        mixed = torch.where(joint_active.to(device=q1.device), q2.to(q1.device), q1)
        return mixed.squeeze(0) if squeeze else mixed

    def get_trimesh_q(self, q: torch.Tensor, mode: str = "active") -> Dict[str, object]:
        q = self._validate_q(q)
        q0 = q[:1]
        parts: Dict[str, trimesh.Trimesh] = {}
        vertices: list[np.ndarray] = []
        faces: list[np.ndarray] = []
        vertex_offset = 0

        gripper_tf = np.eye(4, dtype=np.float64)
        gripper_tf[:3, :3] = euler_xyz_to_matrix(q0[:, 3:6])[0].detach().cpu().numpy()
        gripper_tf[:3, 3] = q0[0, 0:3].detach().cpu().numpy()
        gripper_meshes = self._gripper_model.get_trimesh_q(self._gripper_internal_q(q0)[0], mode=mode)["parts"]
        for name, mesh_obj in gripper_meshes.items():
            mesh = mesh_obj.copy()
            mesh.apply_transform(gripper_tf)
            parts[str(name)] = mesh
            vertices.append(np.asarray(mesh.vertices))
            faces.append(np.asarray(mesh.faces) + vertex_offset)
            vertex_offset += len(mesh.vertices)

        stick_tf = np.eye(4, dtype=np.float64)
        stick_tf[:3, :3] = euler_xyz_to_matrix(q0[:, 10:13])[0].detach().cpu().numpy()
        stick_tf[:3, 3] = q0[0, 7:10].detach().cpu().numpy()
        stick_mesh = self.meshes_original["stick"].copy()
        stick_mesh.apply_transform(stick_tf)
        parts["stick"] = stick_mesh
        vertices.append(np.asarray(stick_mesh.vertices))
        faces.append(np.asarray(stick_mesh.faces) + vertex_offset)

        visual = trimesh.Trimesh(
            vertices=np.vstack(vertices) if vertices else np.zeros((0, 3)),
            faces=np.vstack(faces) if faces else np.zeros((0, 3), dtype=np.int64),
        )
        return {"visual": visual, "parts": parts}
