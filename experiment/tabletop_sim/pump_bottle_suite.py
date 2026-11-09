"""Manifest-backed pump-bottle variants and mixed-dataset helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUMP_BOTTLE_MANIFEST = EXPERIMENT_ROOT / "assets" / "pump_bottle_suite.json"
LEGACY_PUMP_BOTTLE_ID = "pump_bottle005"


@dataclass(frozen=True)
class PumpBottleSpec:
    object_id: str
    name: str
    scale: float
    source_dir: Path
    static_mesh_path: Path
    button_mesh_path: Path
    static_collision_mesh_paths: tuple[Path, ...]
    button_collision_mesh_paths: tuple[Path, ...]
    collision_rebase_xyz: tuple[float, float, float]
    body_rgba: tuple[float, float, float, float]
    button_rgba: tuple[float, float, float, float]
    asset_hash: str
    static_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    button_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]

    @property
    def scaled_collision_rebase_xyz(self) -> np.ndarray:
        return np.asarray(self.collision_rebase_xyz, dtype=np.float64) * float(self.scale)

    @property
    def press_site(self) -> tuple[float, float, float]:
        lo = np.asarray(self.button_bounds[0], dtype=np.float64)
        hi = np.asarray(self.button_bounds[1], dtype=np.float64)
        center = 0.5 * (lo + hi)
        return float(center[0] * self.scale), float(center[1] * self.scale), float(hi[2] * self.scale)

    @property
    def scaled_static_extents(self) -> np.ndarray:
        bounds = np.asarray(self.static_bounds, dtype=np.float64)
        return (bounds[1] - bounds[0]) * float(self.scale)

    @property
    def scaled_button_extents(self) -> np.ndarray:
        bounds = np.asarray(self.button_bounds, dtype=np.float64)
        return (bounds[1] - bounds[0]) * float(self.scale)


def resolve_manifest_path(value: str | Path | None = None) -> Path:
    path = DEFAULT_PUMP_BOTTLE_MANIFEST if value is None else Path(value).expanduser()
    if not path.is_absolute():
        path = (EXPERIMENT_ROOT.parent / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pump-bottle manifest does not exist: {path}")
    return path


def load_pump_bottle_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = resolve_manifest_path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "pump_bottle_suite_v1":
        raise ValueError(f"Unsupported pump-bottle manifest schema in {manifest_path}: {payload.get('schema')!r}")
    return payload


def _manifest_asset_path(manifest_path: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _sha256_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _obj_bounds(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    vertices = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                values = line.split()
                if len(values) >= 4:
                    vertices.append([float(values[1]), float(values[2]), float(values[3])])
    array = np.asarray(vertices, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 4 or not np.isfinite(array).all():
        raise ValueError(f"Invalid or empty OBJ mesh: {path}")
    return tuple(float(x) for x in array.min(axis=0)), tuple(float(x) for x in array.max(axis=0))


def load_pump_bottle_specs(
    manifest_path: str | Path | None = None,
    *,
    object_ids: Sequence[str] | None = None,
) -> tuple[PumpBottleSpec, ...]:
    path = resolve_manifest_path(manifest_path)
    manifest = load_pump_bottle_manifest(path)
    entries = list(manifest.get("objects", []))
    by_id = {str(entry["object_id"]): entry for entry in entries}
    selected = tuple(str(x) for x in (object_ids or manifest.get("default_ids", ())))
    missing = [object_id for object_id in selected if object_id not in by_id]
    if missing:
        raise KeyError(f"Unknown pump-bottle object IDs {missing}; available={sorted(by_id)}")
    specs: list[PumpBottleSpec] = []
    for object_id in selected:
        entry = by_id[object_id]
        static_path = _manifest_asset_path(path, str(entry["static_mesh"]))
        button_path = _manifest_asset_path(path, str(entry["button_mesh"]))
        static_collision_paths = tuple(
            _manifest_asset_path(path, str(value)) for value in entry.get("static_collision_meshes", ())
        )
        button_collision_paths = tuple(
            _manifest_asset_path(path, str(value)) for value in entry.get("button_collision_meshes", ())
        )
        source_dir = _manifest_asset_path(path, str(entry["source_dir"]))
        required_paths = [("static mesh", static_path), ("button mesh", button_path)]
        required_paths.extend(
            (f"static collision mesh {index}", asset_path)
            for index, asset_path in enumerate(static_collision_paths)
        )
        required_paths.extend(
            (f"button collision mesh {index}", asset_path)
            for index, asset_path in enumerate(button_collision_paths)
        )
        for label, asset_path in required_paths:
            if not asset_path.is_file():
                raise FileNotFoundError(
                    f"Missing {label} for {object_id}: {asset_path}. "
                    "Bottles Pumping assets are not included in this release."
                )
        scale = float(entry["scale"])
        if scale <= 0.0:
            raise ValueError(f"Pump-bottle scale must be positive for {object_id}, got {scale}")
        if not static_collision_paths or not button_collision_paths:
            raise ValueError(f"Pump-bottle {object_id} must define static and button collision meshes")
        collision_rebase = np.asarray(entry.get("collision_rebase_xyz", ()), dtype=np.float64)
        if collision_rebase.shape != (3,) or not np.isfinite(collision_rebase).all():
            raise ValueError(
                f"Pump-bottle {object_id} collision_rebase_xyz must contain three finite values, "
                f"got {entry.get('collision_rebase_xyz')!r}"
            )
        specs.append(
            PumpBottleSpec(
                object_id=object_id,
                name=str(entry["name"]),
                scale=scale,
                source_dir=source_dir,
                static_mesh_path=static_path,
                button_mesh_path=button_path,
                static_collision_mesh_paths=static_collision_paths,
                button_collision_mesh_paths=button_collision_paths,
                collision_rebase_xyz=tuple(float(value) for value in collision_rebase),
                body_rgba=tuple(float(x) for x in entry["body_rgba"]),
                button_rgba=tuple(float(x) for x in entry["button_rgba"]),
                asset_hash=_sha256_files((static_path, button_path)),
                static_bounds=_obj_bounds(static_path),
                button_bounds=_obj_bounds(button_path),
            )
        )
    return tuple(specs)


def pump_bottle_spec_by_id(object_id: str, manifest_path: str | Path | None = None) -> PumpBottleSpec:
    return load_pump_bottle_specs(manifest_path, object_ids=(str(object_id),))[0]

