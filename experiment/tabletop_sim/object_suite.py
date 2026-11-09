"""YCB object-suite metadata and cached GraspGenX grasps."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "experiment" / "assets" / "ycb_object_suite.json"
DEFAULT_CONVEX_DECOMPOSITION_ROOT = (
    PROJECT_ROOT / "experiment" / "assets" / "object_suite_convex_decomposition"
)
FIXED_GRASP_SCHEMA = "object_suite_fixed_grasps_v1"
FIXED_GRASP_TRANSFORM_CONVENTION = "object_from_gripper"


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    name: str
    mass_kg: float
    mesh_bounds_m: np.ndarray
    visual_mesh_path: Path
    collision_mesh_path: Path
    texture_path: Path
    upright_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    friction: tuple[float, float, float] = (0, 0., 0.)
    collision_mesh_paths: tuple[Path, ...] = ()

    @property
    def effective_collision_mesh_paths(self) -> tuple[Path, ...]:
        """Collision meshes attached to the object's single free body."""

        return self.collision_mesh_paths or (self.collision_mesh_path,)

    @property
    def min_bound(self) -> np.ndarray:
        return self.mesh_bounds_m[0]

    @property
    def max_bound(self) -> np.ndarray:
        return self.mesh_bounds_m[1]

    @property
    def extents(self) -> np.ndarray:
        return self.max_bound - self.min_bound

    @property
    def asset_hash(self) -> str:
        digest = hashlib.sha256()
        paths = (
            self.visual_mesh_path,
            self.collision_mesh_path,
            *self.effective_collision_mesh_paths,
            self.texture_path,
        )
        for path in dict.fromkeys(paths):
            if not path.is_file():
                raise FileNotFoundError(f"Missing YCB asset for {self.object_id}: {path}")
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        return digest.hexdigest()

    def require_assets(self) -> None:
        _ = self.asset_hash


def load_object_suite(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    mass_kg_override: float | None = None,
    collision_mesh_mode: str = "single_mesh",
    convex_decomposition_root: str | Path | None = DEFAULT_CONVEX_DECOMPOSITION_ROOT,
    convex_decomposition_subdir: str = "convex_decomposition",
) -> tuple[ObjectSpec, ...]:
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if int(raw.get("version", 0)) != 1:
        raise ValueError(f"Unsupported object-suite manifest version in {manifest_path}")

    effective_mass_override = None if mass_kg_override is None else float(mass_kg_override)
    if effective_mass_override is not None and (
        not np.isfinite(effective_mass_override) or effective_mass_override <= 0.0
    ):
        raise ValueError(
            "mass_kg_override must be a positive finite value when set, "
            f"got {mass_kg_override!r}"
        )

    collision_mode = str(collision_mesh_mode).strip().lower().replace("-", "_")
    collision_mode_aliases = {
        "single": "single_mesh",
        "single_mesh": "single_mesh",
        "original": "single_mesh",
        "original_mesh": "single_mesh",
        "convex": "convex_decomposition",
        "decomposition": "convex_decomposition",
        "convex_decomposition": "convex_decomposition",
        "vhacd": "convex_decomposition",
    }
    if collision_mode not in collision_mode_aliases:
        raise ValueError(
            "collision_mesh_mode must be 'single_mesh' or 'convex_decomposition', "
            f"got {collision_mesh_mode!r}"
        )
    collision_mode = collision_mode_aliases[collision_mode]
    decomposition_subdir = Path(str(convex_decomposition_subdir).strip())
    if (
        not str(decomposition_subdir)
        or decomposition_subdir.is_absolute()
        or ".." in decomposition_subdir.parts
    ):
        raise ValueError(
            "convex_decomposition_subdir must be a non-empty relative path without '..', "
            f"got {convex_decomposition_subdir!r}"
        )
    decomposition_root = None
    if convex_decomposition_root is not None:
        root_text = str(convex_decomposition_root).strip()
        if root_text and root_text.lower() not in {"null", "none"}:
            decomposition_root = Path(root_text).expanduser()
            if not decomposition_root.is_absolute():
                decomposition_root = PROJECT_ROOT / decomposition_root
            decomposition_root = decomposition_root.resolve()

    asset_root = Path(str(raw["asset_root"])).expanduser()
    if not asset_root.is_absolute():
        asset_root = PROJECT_ROOT / asset_root

    objects: list[ObjectSpec] = []
    seen: set[str] = set()
    for entry in raw["objects"]:
        object_id = str(entry["id"])
        if object_id in seen:
            raise ValueError(f"Duplicate object id in {manifest_path}: {object_id}")
        seen.add(object_id)
        object_dir = asset_root / object_id / "google_16k"
        collision_mesh_path = object_dir / "nontextured.stl"
        collision_mesh_paths: tuple[Path, ...] = ()
        if collision_mode == "convex_decomposition":
            decomposition_dir = (
                object_dir / decomposition_subdir
                if decomposition_root is None
                else decomposition_root / object_id
            )
            collision_mesh_paths = tuple(sorted(decomposition_dir.glob("part_*.obj")))
            if not collision_mesh_paths:
                raise FileNotFoundError(
                    f"No convex collision parts found for {object_id} in {decomposition_dir}. "
                    "Install the object assets with `python download_assets.py`."
                )
        bounds = np.asarray(entry["mesh_bounds_m"], dtype=np.float64)
        if bounds.shape != (2, 3) or np.any(bounds[1] <= bounds[0]):
            raise ValueError(f"Invalid mesh_bounds_m for {object_id}: {bounds.tolist()}")
        manifest_mass = float(entry["mass_kg"])
        if not np.isfinite(manifest_mass) or manifest_mass <= 0.0:
            raise ValueError(f"mass_kg must be positive for {object_id}")
        mass = manifest_mass if effective_mass_override is None else effective_mass_override
        upright_quat = np.asarray(
            entry.get("upright_quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        if upright_quat.shape != (4,) or not np.all(np.isfinite(upright_quat)):
            raise ValueError(
                f"upright_quat_wxyz must contain four finite values for {object_id}: "
                f"{upright_quat.tolist()}"
            )
        upright_norm = float(np.linalg.norm(upright_quat))
        if upright_norm <= 1.0e-8:
            raise ValueError(f"upright_quat_wxyz must be nonzero for {object_id}")
        upright_quat = upright_quat / upright_norm
        objects.append(
            ObjectSpec(
                object_id=object_id,
                name=str(entry["name"]),
                mass_kg=mass,
                mesh_bounds_m=bounds,
                visual_mesh_path=object_dir / "textured.obj",
                collision_mesh_path=collision_mesh_path,
                texture_path=object_dir / "texture_map.png",
                upright_quat_wxyz=tuple(float(value) for value in upright_quat),
                collision_mesh_paths=collision_mesh_paths,
            )
        )
    if not objects:
        raise ValueError(f"Object-suite manifest is empty: {manifest_path}")
    return tuple(objects)


def object_spec_by_id(objects: tuple[ObjectSpec, ...], object_id: str) -> ObjectSpec:
    for spec in objects:
        if spec.object_id == str(object_id):
            return spec
    available = ", ".join(spec.object_id for spec in objects)
    raise KeyError(f"Unknown object id {object_id!r}; available: {available}")


@dataclass(frozen=True)
class GraspBank:
    object_id: str
    poses_object: np.ndarray
    scores: np.ndarray
    generator: str
    asset_hash: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_object_id: str,
        expected_asset_hash: str | None = None,
    ) -> "GraspBank":
        bank_path = Path(path).expanduser().resolve()
        if not bank_path.is_file():
            raise FileNotFoundError(
                f"Missing GraspGenX bank for {expected_object_id}: {bank_path}. "
                "GraspGenX banks are not included in this release; use policy.grasp_pose_mode=fixed."
            )
        with np.load(bank_path, allow_pickle=False) as data:
            object_id = str(np.asarray(data["object_id"]).item())
            poses = np.asarray(data["poses_object"], dtype=np.float64)
            scores = np.asarray(data["scores"], dtype=np.float64).reshape(-1)
            generator = str(np.asarray(data["generator"]).item())
            asset_hash = str(np.asarray(data["asset_hash"]).item()) if "asset_hash" in data else ""
        if object_id != str(expected_object_id):
            raise ValueError(f"Grasp bank object_id={object_id!r}, expected {expected_object_id!r}")
        if expected_asset_hash is not None and asset_hash != str(expected_asset_hash):
            raise ValueError(
                f"Grasp bank asset hash does not match {expected_object_id}; regenerate {bank_path}"
            )
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or poses.shape[0] != scores.shape[0]:
            raise ValueError(f"Invalid grasp bank shapes: poses={poses.shape}, scores={scores.shape}")
        if poses.shape[0] == 0 or not np.all(np.isfinite(poses)) or not np.all(np.isfinite(scores)):
            raise ValueError(f"Grasp bank is empty or non-finite: {bank_path}")
        order = np.argsort(scores)[::-1]
        return cls(
            object_id=object_id,
            poses_object=poses[order],
            scores=scores[order],
            generator=generator,
            asset_hash=asset_hash,
        )


@dataclass(frozen=True)
class FixedGraspSet:
    """One hand-selected ``object_from_gripper`` transform per suite object."""

    object_ids: tuple[str, ...]
    asset_hashes: tuple[str, ...]
    poses_object: np.ndarray
    source_candidate_ranks: np.ndarray
    source_candidate_scores: np.ndarray
    source_path: Path | None = None
    file_sha256: str = ""

    def __post_init__(self) -> None:
        object_ids = tuple(str(value) for value in self.object_ids)
        asset_hashes = tuple(str(value) for value in self.asset_hashes)
        poses = np.asarray(self.poses_object, dtype=np.float64)
        ranks = np.asarray(self.source_candidate_ranks, dtype=np.int64).reshape(-1)
        scores = np.asarray(self.source_candidate_scores, dtype=np.float64).reshape(-1)
        _validate_fixed_grasp_arrays(
            object_ids=object_ids,
            asset_hashes=asset_hashes,
            poses_object=poses,
            source_candidate_ranks=ranks,
            source_candidate_scores=scores,
        )
        object.__setattr__(self, "object_ids", object_ids)
        object.__setattr__(self, "asset_hashes", asset_hashes)
        object.__setattr__(self, "poses_object", poses.copy())
        object.__setattr__(self, "source_candidate_ranks", ranks.copy())
        object.__setattr__(self, "source_candidate_scores", scores.copy())
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path).expanduser().resolve())

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_objects: Sequence[ObjectSpec] | None = None,
    ) -> "FixedGraspSet":
        grasp_path = Path(path).expanduser().resolve()
        if not grasp_path.is_file():
            raise FileNotFoundError(f"Missing hand-selected fixed-grasp file: {grasp_path}")
        with np.load(grasp_path, allow_pickle=False) as data:
            schema = str(np.asarray(data["schema"]).item())
            convention = str(np.asarray(data["transform_convention"]).item())
            if schema != FIXED_GRASP_SCHEMA:
                raise ValueError(
                    f"Unsupported fixed-grasp schema {schema!r} in {grasp_path}; "
                    f"expected {FIXED_GRASP_SCHEMA!r}"
                )
            if convention != FIXED_GRASP_TRANSFORM_CONVENTION:
                raise ValueError(
                    f"Unsupported fixed-grasp transform convention {convention!r} in {grasp_path}; "
                    f"expected {FIXED_GRASP_TRANSFORM_CONVENTION!r}"
                )
            loaded = cls(
                object_ids=tuple(str(value) for value in np.asarray(data["object_ids"]).reshape(-1)),
                asset_hashes=tuple(str(value) for value in np.asarray(data["asset_hashes"]).reshape(-1)),
                poses_object=np.asarray(data["poses_object"], dtype=np.float64),
                source_candidate_ranks=np.asarray(data["source_candidate_ranks"], dtype=np.int64),
                source_candidate_scores=np.asarray(data["source_candidate_scores"], dtype=np.float64),
                source_path=grasp_path,
                file_sha256=_file_sha256(grasp_path),
            )
        return loaded.reordered_for(expected_objects) if expected_objects is not None else loaded

    def save(self, path: str | Path) -> Path:
        _validate_fixed_grasp_arrays(
            object_ids=self.object_ids,
            asset_hashes=self.asset_hashes,
            poses_object=np.asarray(self.poses_object, dtype=np.float64),
            source_candidate_ranks=np.asarray(self.source_candidate_ranks, dtype=np.int64),
            source_candidate_scores=np.asarray(self.source_candidate_scores, dtype=np.float64),
        )
        grasp_path = Path(path).expanduser().resolve()
        grasp_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            grasp_path,
            schema=np.asarray(FIXED_GRASP_SCHEMA),
            transform_convention=np.asarray(FIXED_GRASP_TRANSFORM_CONVENTION),
            gripper=np.asarray("robotiq_2f_85"),
            object_ids=np.asarray(self.object_ids),
            asset_hashes=np.asarray(self.asset_hashes),
            poses_object=np.asarray(self.poses_object, dtype=np.float32),
            source_candidate_ranks=np.asarray(self.source_candidate_ranks, dtype=np.int32),
            source_candidate_scores=np.asarray(self.source_candidate_scores, dtype=np.float32),
        )
        return grasp_path

    def reordered_for(self, expected_objects: Sequence[ObjectSpec], *, allow_extra: bool = False) -> "FixedGraspSet":
        expected = tuple(expected_objects)
        expected_ids = tuple(spec.object_id for spec in expected)
        actual_ids = set(self.object_ids)
        missing = [object_id for object_id in expected_ids if object_id not in actual_ids]
        extra = [object_id for object_id in self.object_ids if object_id not in set(expected_ids)]
        if missing or (extra and not allow_extra):
            raise ValueError(
                "Fixed-grasp object coverage does not match the configured manifest: "
                f"missing={missing}, extra={extra}"
            )
        index_by_id = {object_id: index for index, object_id in enumerate(self.object_ids)}
        order = np.asarray([index_by_id[object_id] for object_id in expected_ids], dtype=np.int64)
        hashes = tuple(self.asset_hashes[int(index)] for index in order)
        for spec, saved_hash in zip(expected, hashes):
            expected_hash = spec.asset_hash
            if saved_hash != expected_hash:
                raise ValueError(
                    f"Fixed-grasp asset hash does not match {spec.object_id}; "
                    "open the chooser and resave all grasps for the current assets"
                )
        return FixedGraspSet(
            object_ids=expected_ids,
            asset_hashes=hashes,
            poses_object=self.poses_object[order],
            source_candidate_ranks=self.source_candidate_ranks[order],
            source_candidate_scores=self.source_candidate_scores[order],
            source_path=self.source_path,
            file_sha256=self.file_sha256,
        )

    def index(self, object_id: str) -> int:
        try:
            return self.object_ids.index(str(object_id))
        except ValueError as exc:
            available = ", ".join(self.object_ids)
            raise KeyError(f"No fixed grasp for {object_id!r}; available: {available}") from exc

    def pose_for(self, object_id: str) -> np.ndarray:
        return np.asarray(self.poses_object[self.index(object_id)], dtype=np.float64).copy()

    def source_candidate_for(self, object_id: str) -> tuple[int, float]:
        index = self.index(object_id)
        return (
            int(self.source_candidate_ranks[index]),
            float(self.source_candidate_scores[index]),
        )


def _validate_fixed_grasp_arrays(
    *,
    object_ids: tuple[str, ...],
    asset_hashes: tuple[str, ...],
    poses_object: np.ndarray,
    source_candidate_ranks: np.ndarray,
    source_candidate_scores: np.ndarray,
) -> None:
    count = len(object_ids)
    if count == 0:
        raise ValueError("Fixed-grasp set must contain at least one object")
    if any(not object_id for object_id in object_ids):
        raise ValueError("Fixed-grasp object IDs must be nonempty")
    if len(set(object_ids)) != count:
        raise ValueError(f"Fixed-grasp object IDs must be unique: {object_ids}")
    if len(asset_hashes) != count or any(not value for value in asset_hashes):
        raise ValueError("Fixed-grasp asset_hashes must contain one nonempty hash per object")
    if poses_object.shape != (count, 4, 4):
        raise ValueError(
            f"Fixed-grasp poses_object must have shape {(count, 4, 4)}, got {poses_object.shape}"
        )
    if source_candidate_ranks.shape != (count,) or source_candidate_scores.shape != (count,):
        raise ValueError(
            "Fixed-grasp candidate provenance must contain one rank and score per object; "
            f"got ranks={source_candidate_ranks.shape}, scores={source_candidate_scores.shape}"
        )
    if np.any(source_candidate_ranks < 0) or not np.all(np.isfinite(source_candidate_scores)):
        raise ValueError("Fixed-grasp source candidate ranks/scores must be nonnegative and finite")
    if not np.all(np.isfinite(poses_object)):
        raise ValueError("Fixed-grasp poses_object contains non-finite values")
    expected_bottom = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    for index, pose in enumerate(poses_object):
        if not np.allclose(pose[3], expected_bottom, atol=1.0e-6, rtol=0.0):
            raise ValueError(
                f"Fixed grasp {object_ids[index]} has an invalid homogeneous bottom row"
            )
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-4, rtol=0.0):
            raise ValueError(f"Fixed grasp {object_ids[index]} rotation is not orthonormal")
        determinant = float(np.linalg.det(rotation))
        if abs(determinant - 1.0) > 1.0e-4:
            raise ValueError(
                f"Fixed grasp {object_ids[index]} rotation determinant must be +1, got {determinant}"
            )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
