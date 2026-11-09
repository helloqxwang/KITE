from __future__ import annotations

import inspect
from collections import namedtuple
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

# Resolve paths relative to the repo root (this file lives at
# <repo>/experiment/manipulation/mano_utils.py) so the bundled code and models work
# regardless of where the repo is cloned.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANOPTH_ROOT = REPO_ROOT / "third_party" / "manopth"
DEFAULT_MANO_MODEL_DIR = REPO_ROOT / "assets" / "mano_models"
MANO_TIP_JOINT_INDICES = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
MANO_PALM_ROOT_JOINT_INDICES = np.asarray([1, 5, 9, 13, 17], dtype=np.int64)
# The Spider articulated MANO asset places its `*_palm` tracking site at the
# MANO middle-finger MCP, not at the centroid of the palm roots.
MANO_SPIDER_PALM_SITE_JOINT_INDEX = 9


def _ensure_inspect_getargspec() -> None:
    if hasattr(inspect, "getargspec"):
        return
    arg_spec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def _getargspec(func):
        spec = inspect.getfullargspec(func)
        return arg_spec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = _getargspec  # type: ignore[attr-defined]


def _ensure_numpy_legacy_aliases() -> None:
    legacy_aliases = {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in legacy_aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def resolve_mano_model_root(
    mano_model_root: str | Path | None = None,
    manopth_root: str | Path | None = None,
) -> Path:
    # ``manopth_root`` is accepted but unused: model weights are always read from
    # ``mano_model_root``, never from a library location.
    root = Path(mano_model_root or DEFAULT_MANO_MODEL_DIR).expanduser().resolve()
    missing = [name for name in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl")
               if not (root / name).is_file()]
    if not missing:
        return root
    raise FileNotFoundError(
        f"Missing MANO model files in {root}: {', '.join(missing)}. "
        "Download the models from https://mano.is.tue.mpg.de/ and copy "
        f"MANO_LEFT.pkl and MANO_RIGHT.pkl into {root}. "
        "KITE does not search other repositories for MANO weights."
    )


def resolve_mano_manotorch_assets_root(mano_model_root: str | Path) -> Path:
    """Return a directory ``R`` such that manotorch's hardcoded
    ``<R>/models/MANO_<SIDE>.pkl`` layout resolves to the real MANO pickles.

    ``resolve_mano_model_root`` returns the directory that directly contains the
    pickles (manopth convention). That directory may already be named ``models``
    (e.g. the release-local manopth checkout) — in which case manotorch wants the
    parent. Otherwise the layout is flat (e.g. ``<repo>/assets/mano_models``),
    and we lazily materialize a ``models/`` subdir of per-file symlinks so both
    libraries use the same local weights without duplicating them.
    """
    root = Path(mano_model_root).expanduser().resolve()
    if root.name == "models":
        return root.parent
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)
    for side in ("LEFT", "RIGHT"):
        pkl_name = f"MANO_{side}.pkl"
        source = root / pkl_name
        link = models_dir / pkl_name
        if not source.exists():
            continue
        # Replace broken symlinks, e.g. absolute links copied from another machine.
        if link.is_symlink() and not link.exists():
            link.unlink()
        if link.exists():
            continue
        # Use a *relative* target so the subdir is portable across hosts/users.
        try:
            link.symlink_to(Path("..") / pkl_name)
        except OSError:
            import shutil

            shutil.copy2(source, link)
    return root


def _import_manopth_mano_layer():
    """Use this checkout's MANO implementation, independent of pip/PYTHONPATH.

    The qualified name also works if another application already imported an
    incompatible top-level ``manopth`` package in this Python process.
    """
    _ensure_inspect_getargspec()
    _ensure_numpy_legacy_aliases()
    try:
        from third_party.manopth.manopth.manolayer import ManoLayer
    except ImportError as exc:
        raise ImportError(
            f"Could not import KITE's bundled manopth at {DEFAULT_MANOPTH_ROOT}: {exc}. "
            "From the KITE root, install dependencies with "
            "`python -m pip install --no-deps -r requirements.txt`, then run "
            "`python scripts/check_mano_installation.py`."
        ) from exc
    return ManoLayer


@lru_cache(maxsize=8)
def _get_mano_layer_cached(side: str, mano_model_root: str, device_str: str):
    ManoLayer = _import_manopth_mano_layer()

    layer = ManoLayer(
        mano_root=mano_model_root,
        use_pca=False,
        flat_hand_mean=True,
        ncomps=45,
        side=str(side),
        center_idx=0,
    )
    return layer.to(torch.device(device_str))


def load_mano_mesh_and_joints(
    *,
    pose: np.ndarray,
    beta: np.ndarray,
    trans: np.ndarray,
    side: str,
    mano_model_root: str | Path,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose_np = np.asarray(pose, dtype=np.float32)
    trans_np = np.asarray(trans, dtype=np.float32)
    beta_np = np.asarray(beta, dtype=np.float32).reshape(1, 10)
    if pose_np.ndim == 1:
        pose_np = pose_np[None, :]
    if trans_np.ndim == 1:
        trans_np = trans_np[None, :]
    if pose_np.shape[0] != trans_np.shape[0]:
        raise ValueError(
            f"Pose/translation batch mismatch: {pose_np.shape} vs {trans_np.shape}"
        )

    layer = _get_mano_layer_cached(
        str(side), str(Path(mano_model_root).expanduser().resolve()), str(device)
    )
    device_obj = torch.device(device)
    pose_t = torch.from_numpy(pose_np).to(device_obj)
    trans_t = torch.from_numpy(trans_np).to(device_obj)
    beta_t = torch.from_numpy(beta_np).to(device_obj).repeat(pose_t.shape[0], 1)

    # manopth's upstream ``ManoLayer`` returns ``(verts, joints)``; forks (incl. the
    # local MANO layer) return a 3-tuple with an extra value. Slice to the first two.
    verts_t, joints_t = layer(pose_t, beta_t)[:2]
    verts_t = verts_t / 1000.0 + trans_t[:, None, :]
    joints_t = joints_t / 1000.0 + trans_t[:, None, :]
    faces = layer.th_faces.detach().cpu().numpy().astype(np.int32, copy=False)
    return (
        verts_t.detach().cpu().numpy().astype(np.float32, copy=False),
        joints_t.detach().cpu().numpy().astype(np.float32, copy=False),
        faces,
    )
