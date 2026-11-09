from __future__ import annotations

import argparse
import hashlib
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

EXPERIMENT_DIR = Path(__file__).resolve().parent
CROSS_DIR = EXPERIMENT_DIR.parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(CROSS_DIR) not in sys.path:
    sys.path.insert(0, str(CROSS_DIR))

from experiment.embodiments import validate_embodiment_supported  # noqa: E402
from high_level_policy import contact_encoding as contact_encoder  # noqa: E402
from high_level_policy.payload import (  # noqa: E402
    build_common_meta,
    build_training_payload,
    concat_single_manipulator_action,
    contact_points_action_space,
    latent_contact_action_space,
    load_high_level_payload,
    previous_action_proprio,
    save_high_level_payload,
    single_manipulator_payload_to_replay_source,
    single_manipulator_action_space,
    validate_object_only_world_observation,
    with_object_only_world_observation,
)
from high_level_policy.pointcloud_contract import (
    validate_object_only_world_points,
)  # noqa: E402
from high_level_policy.preprocess_utils import pose_mats_to_pos_quat_wxyz  # noqa: E402
from manipulation.sparse_contact_extraction import (  # noqa: E402
    DEFAULT_CLUSTER_SEED,
    CONTACT_EXTRACTION_VERSION,
    extract_contact_sequence_from_surface,
    offset_contact_positions,
)


OBJECT_POINTCLOUD_KEYS = ("obs_point_cloud", "object_pointcloud", "pointcloud_xyzw")
OBJECT_POSE_KEYS = ("object_pose", "cube_pose")
MANIPULATOR_Q_KEYS = ("manipulator_q", "executed_qpos")
FK_CHUNK = 256
DEFAULT_SOURCE_PT_PATH = (
    EXPERIMENT_DIR
    / "data"
    / "high_level_policy"
    / "3600_franka_robotiq_object_suite_pick_place_v2.pt"
)
DEFAULT_OUTPUT_PATH = (
    EXPERIMENT_DIR
    / "data"
    / "high_level_policy"
    / "object_suite_paper_contacts_train.pt"
)
DEFAULT_ACTION_POLICY_CONFIG = CROSS_DIR / "conf" / "kite.yaml"


from paper_config import runtime_config


@dataclass(frozen=True)
class SurfaceTrajectory:
    surface_points_world: np.ndarray
    surface_normals_world: np.ndarray
    surface_points_canonical: np.ndarray
    surface_template_hash: str
    surface_template_size: int
    manipulator_q: np.ndarray
    provider: str
    resolved_manipulator_name: str
    surface_neighbors: tuple[tuple[int, ...], ...]
    surface_graph_hash: str


def _cfg_select(cfg: DictConfig, key: str, default: Any) -> Any:
    value = OmegaConf.select(cfg, key, default=default)
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _resolve_experiment_config_path(
    path_value: Any, *, default: Path | None = None
) -> str | None:
    value = default if path_value is None else path_value
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "null":
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((EXPERIMENT_DIR / path).resolve())


def _resolve_preprocess_action_space(cfg: DictConfig) -> str:
    requested = (
        str(_cfg_select(cfg, "preprocess.action_space", "latent_contact"))
        .strip()
        .lower()
    )
    if requested == "auto":
        policy_action_space = (
            str(_cfg_select(cfg, "policy.action_space", "latent")).strip().lower()
        )
        if policy_action_space == "latent":
            return "latent_contact"
        if policy_action_space == "contact_points":
            return "contact_points"
        return "single_manipulator"
    aliases = {
        "latent": "latent_contact",
        "latent_contact": "latent_contact",
        "contact": "contact_points",
        "contact_points": "contact_points",
        "direct_contact": "contact_points",
        "single_manipulator": "single_manipulator",
        "explicit": "single_manipulator",
    }
    if requested not in aliases:
        raise ValueError(
            f"Unsupported preprocess.action_space={requested!r}; expected latent_contact, contact_points, "
            "single_manipulator, or auto"
        )
    return aliases[requested]


def _namespace_from_hydra_config(cfg: DictConfig) -> argparse.Namespace:
    if OmegaConf.select(cfg, "preprocess", default=None) is None:
        raise KeyError("Expected a preprocess section in the high-level policy config")
    action_policy_config = _cfg_select(
        cfg,
        "preprocess.action_policy_config",
        _cfg_select(
            cfg,
            "preprocess.latent_contact_policy_config",
            str(DEFAULT_ACTION_POLICY_CONFIG),
        ),
    )
    high_level_policy_config = _cfg_select(
        cfg,
        "preprocess.high_level_policy_config",
        str(EXPERIMENT_DIR / "conf" / "train_policy" / "object_suite.yaml"),
    )
    return argparse.Namespace(
        surface_template_points=int(_cfg_select(cfg, "preprocess.surface_template_points", 0)),
        object_surface_points=int(_cfg_select(cfg, "preprocess.object_surface_points", 4096)),
        source_configuration=str(_cfg_select(cfg, "preprocess.source_configuration", "commanded")),
        contact_alignment_mode=str(_cfg_select(cfg, "preprocess.contact_alignment_mode", "displacement_direction")),
        source_format=str(_cfg_select(cfg, "preprocess.source_format", "unified_pt")),
        source_pt=_resolve_experiment_config_path(
            _cfg_select(cfg, "preprocess.source_pt", None),
            default=DEFAULT_SOURCE_PT_PATH,
        ),
        output_path=_resolve_experiment_config_path(
            _cfg_select(cfg, "preprocess.output_path", None),
            default=DEFAULT_OUTPUT_PATH,
        ),
        latent_contact_policy_config=str(action_policy_config),
        high_level_policy_config=_resolve_experiment_config_path(
            high_level_policy_config
        ),
        latent_contact_encoder_checkpoint=_cfg_select(
            cfg, "preprocess.latent_contact_encoder_checkpoint", None
        ),
        latent_encoder_device=str(
            _cfg_select(cfg, "preprocess.latent_encoder_device", "auto")
        ),
        latent_encoder_batch_size=int(
            _cfg_select(cfg, "preprocess.latent_encoder_batch_size", 1024)
        ),
        latent_prefer_ema=bool(_cfg_select(cfg, "preprocess.latent_prefer_ema", True)),
        latent_use_xyz_translation=bool(
            _cfg_select(
                cfg,
                "preprocess.latent_use_xyz_translation",
                _cfg_select(cfg, "policy.latent_use_xyz_translation", True),
            )
        ),
        fps=float(_cfg_select(cfg, "preprocess.fps", 30.0)),
        seed=int(_cfg_select(cfg, "preprocess.seed", _cfg_select(cfg, "seed", 42))),
        max_episodes=int(_cfg_select(cfg, "preprocess.max_episodes", -1)),
        contact_mode=str(
            _cfg_select(cfg, "preprocess.contact_mode", "trajectory_sparse")
        ),
        contact_sparse_capacity=int(
            _cfg_select(cfg, "preprocess.contact_sparse_capacity", -1)
        ),
        contact_threshold=float(_cfg_select(cfg, "preprocess.contact_threshold", 0.1)),
        align_exp_scale=float(_cfg_select(cfg, "preprocess.align_exp_scale", 1.0)),
        sigmoid_scale=float(_cfg_select(cfg, "preprocess.sigmoid_scale", 10.0)),
        contact_position_offset_m=float(
            _cfg_select(cfg, "preprocess.contact_position_offset_m", 0.015)
        ),
        manipulator_name=str(_cfg_select(cfg, "preprocess.manipulator_name", "auto")),
        surface_provider=str(_cfg_select(cfg, "preprocess.surface_provider", "auto")),
        object_pointcloud_frame=str(
            _cfg_select(cfg, "preprocess.object_pointcloud_frame", "world")
        ),
        tabletop_side=str(_cfg_select(cfg, "preprocess.tabletop_side", "auto")),
        skip_latent=bool(_cfg_select(cfg, "preprocess.skip_latent", False)),
        action_space=_resolve_preprocess_action_space(cfg),
        task=str(_cfg_select(cfg, "preprocess.task", "object_suite_pick_place")),
        coordinate_reference_payload=_resolve_experiment_config_path(
            _cfg_select(cfg, "preprocess.coordinate_reference_payload", None)
        ),
        dataset_xyz_translation=_cfg_select(cfg, "preprocess.dataset_xyz_translation", None),
        score_chunk_frames=int(_cfg_select(cfg, "preprocess.score_chunk_frames", 4)),
        score_device=str(_cfg_select(cfg, "preprocess.score_device", "auto")),
    )


def _apply_high_level_preprocess_config(
    args: argparse.Namespace, cfg: DictConfig
) -> None:
    if args.source_format != "unified_pt" or args.task not in {
        "object_suite_pick_place", "keyboard_press", "pumpbottle_use"
    }:
        raise ValueError(
            "KITE preprocessing expects unified_pt tabletop task demonstrations"
        )
    if args.contact_mode != "trajectory_sparse":
        raise ValueError(
            "Set preprocess.contact_mode=trajectory_sparse for paper contact extraction"
        )
    args.latent_use_xyz_translation = bool(
        _cfg_select(cfg, "preprocess.latent_use_xyz_translation", False)
    )


def _dataset_name(args: argparse.Namespace) -> str:
    return _resolve_output_path(args).stem


def _resolve_output_path(args: argparse.Namespace) -> Path:
    path = Path(args.output_path).expanduser().resolve()
    if path.suffix != ".pt":
        path = path.with_suffix(".pt")
    return path


def _source_scalar_str(
    source: dict[str, np.ndarray], key: str, default: str = ""
) -> str:
    if key not in source:
        return str(default)
    value = np.asarray(source[key])
    if value.shape == ():
        return str(value.item())
    if value.size == 1:
        return str(value.reshape(-1)[0])
    return str(default)


def _first_existing_key(
    source: dict[str, np.ndarray], keys: tuple[str, ...], *, label: str
) -> str:
    for key in keys:
        if key in source:
            return key
    raise KeyError(f"Unified .pt source lacks {label}; expected one of {list(keys)}")


def _as_episode_major(array: np.ndarray, *, key: str, min_tail_ndim: int) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim < min_tail_ndim + 1:
        raise ValueError(f"{key} must include a time dimension, got shape {arr.shape}")
    if arr.ndim == min_tail_ndim + 1:
        arr = arr[None, ...]
    return arr


def _episode_count(*arrays: np.ndarray) -> int:
    counts = {int(arr.shape[0]) for arr in arrays}
    if len(counts) != 1:
        raise ValueError(f"Unified .pt episode counts differ: {sorted(counts)}")
    return counts.pop()


def _pose_to_pos_quat_wxyz(pose: np.ndarray) -> np.ndarray:
    pose_np = np.asarray(pose, dtype=np.float32)
    if pose_np.ndim == 2 and pose_np.shape[-1] == 7:
        return pose_np.astype(np.float32, copy=False)
    if pose_np.ndim == 3 and pose_np.shape[-2:] == (4, 4):
        return pose_mats_to_pos_quat_wxyz(pose_np.astype(np.float32, copy=False))
    raise ValueError(
        f"object pose must have shape (T, 7) or (T, 4, 4), got {pose_np.shape}"
    )


def _transform_local_object_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    pose_np = np.asarray(pose, dtype=np.float32)
    if pose_np.ndim == 2 and pose_np.shape[-1] == 7:
        from scipy.spatial.transform import Rotation as R

        quat_wxyz = pose_np[:, 3:7]
        quat_xyzw = np.stack(
            [quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3], quat_wxyz[:, 0]], axis=1
        )
        rot = R.from_quat(quat_xyzw).as_matrix().astype(np.float32)
        return (pts @ np.swapaxes(rot, 1, 2) + pose_np[:, None, :3]).astype(
            np.float32, copy=False
        )
    if pose_np.ndim == 3 and pose_np.shape[-2:] == (4, 4):
        rot = pose_np[:, :3, :3]
        trans = pose_np[:, :3, 3]
        return (pts @ np.swapaxes(rot, 1, 2) + trans[:, None, :]).astype(
            np.float32, copy=False
        )
    raise ValueError(
        f"Cannot transform local object points with pose shape {pose_np.shape}"
    )


def _load_unified_episodes(
    source_pt_path: Path, args: argparse.Namespace
) -> list[dict[str, Any]]:
    payload = load_high_level_payload(source_pt_path)
    source = single_manipulator_payload_to_replay_source(
        payload, source_path=source_pt_path
    )
    source_meta = dict(payload.get("meta", {}))
    configured_task = str(getattr(args, "task", "")).strip()
    source_task = str(source_meta.get("task", "")).strip()
    if configured_task and source_task and configured_task != source_task:
        raise ValueError(
            f"Configured preprocess.task={configured_task!r} does not match "
            f"source meta.task={source_task!r}"
        )
    is_object_suite = (source_task or configured_task) == "object_suite_pick_place"
    if is_object_suite:
        if str(args.object_pointcloud_frame).strip().lower() != "world":
            raise ValueError(
                "Object-suite preprocessing requires preprocess.object_pointcloud_frame=world"
            )
        validate_object_only_world_observation(source_meta, label=str(source_pt_path))
    pc_key = _first_existing_key(
        source, OBJECT_POINTCLOUD_KEYS, label="object point cloud"
    )
    pose_key = _first_existing_key(source, OBJECT_POSE_KEYS, label="object pose")
    q_key = _first_existing_key(
        source, MANIPULATOR_Q_KEYS, label="manipulator trajectory"
    )

    object_pc = _as_episode_major(
        np.asarray(source[pc_key]), key=pc_key, min_tail_ndim=2
    )
    object_pc = np.asarray(object_pc[..., :3], dtype=np.float32)
    object_pose = _as_episode_major(
        np.asarray(source[pose_key]), key=pose_key, min_tail_ndim=1
    )
    manipulator_q = _as_episode_major(
        np.asarray(source[q_key]), key=q_key, min_tail_ndim=1
    )
    count = _episode_count(object_pc, object_pose, manipulator_q)
    pointcloud_occluded = None
    if "pointcloud_occluded" in source:
        pointcloud_occluded = _as_episode_major(
            np.asarray(source["pointcloud_occluded"], dtype=np.bool_),
            key="pointcloud_occluded",
            min_tail_ndim=0,
        )
        if int(pointcloud_occluded.shape[0]) != count:
            raise ValueError(
                "pointcloud_occluded episode count does not match observations"
            )

    if int(args.max_episodes) > 0:
        count = min(count, int(args.max_episodes))

    episodes: list[dict[str, Any]] = []
    source_episode_meta = list(payload.get("episode_meta", []))
    for episode_idx in range(count):
        pc = object_pc[episode_idx].astype(np.float32, copy=False)
        pose_raw = object_pose[episode_idx].astype(np.float32, copy=False)
        q = manipulator_q[episode_idx].astype(np.float32, copy=False)
        if int(pc.shape[0]) != int(q.shape[0]):
            raise ValueError(
                f"Episode {episode_idx}: object point cloud frames={pc.shape[0]} but manipulator frames={q.shape[0]}"
            )
        if int(pose_raw.shape[0]) != int(q.shape[0]):
            raise ValueError(
                f"Episode {episode_idx}: object pose frames={pose_raw.shape[0]} but q frames={q.shape[0]}"
            )
        episode_source_meta = (
            dict(source_episode_meta[episode_idx])
            if episode_idx < len(source_episode_meta)
            and isinstance(source_episode_meta[episode_idx], dict)
            else {}
        )
        if str(args.object_pointcloud_frame) == "local":
            pc = _transform_local_object_points(pc, pose_raw)
        pose_world = _pose_to_pos_quat_wxyz(pose_raw)
        occluded = (
            None if pointcloud_occluded is None else pointcloud_occluded[episode_idx]
        )
        if is_object_suite:
            if episode_source_meta.get("object_bounds_m") is None:
                raise KeyError(
                    f"Episode {episode_idx} is missing episode_meta.object_bounds_m"
                )
            validate_object_only_world_points(
                pc,
                pose_world,
                episode_source_meta["object_bounds_m"],
                occluded_frames=occluded,
                label=f"{source_pt_path} episode {episode_idx}",
            )
        episodes.append(
            {
                "episode_index": episode_idx,
                "object_points_world": pc,
                "object_pose": pose_world,
                "manipulator_q_source": q,
                "source": source,
                "source_meta": source_meta,
                "source_pt_path": source_pt_path,
                "source_trajectory_index": _trajectory_source_index(
                    source, episode_idx
                ),
                "object_pc_key": pc_key,
                "object_pose_key": pose_key,
                "manipulator_q_key": q_key,
                "source_episode_meta": episode_source_meta,
            }
        )
    return episodes


def _trajectory_source_index(source: dict[str, np.ndarray], episode_idx: int) -> int:
    for key in ("source_trajectory_index", "trajectory_index"):
        if key not in source:
            continue
        arr = np.asarray(source[key])
        if arr.ndim >= 1 and int(arr.shape[0]) > int(episode_idx):
            return int(np.asarray(arr[int(episode_idx)]).reshape(-1)[0])
    return int(episode_idx)


def _short_hash_from_arrays(*arrays: np.ndarray, prefix: str = "") -> str:
    h = hashlib.sha1()
    if prefix:
        h.update(str(prefix).encode("utf-8"))
    for arr in arrays:
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _resolve_surface_template_size(
    cfg: Any, manipulator_name: str, *, default: int
) -> int:
    from omegaconf import OmegaConf

    name = str(manipulator_name)
    per_manip = OmegaConf.select(
        cfg, f"contact_candidates.per_manipulator.{name}", default=None
    )
    if per_manip is not None:
        return int(per_manip)
    global_count = OmegaConf.select(
        cfg, "contact_candidates.num_surface_points", default=None
    )
    if global_count is not None:
        return int(global_count)
    hparams_path = OmegaConf.select(cfg, "dataset.hparams_path", default=None)
    if hparams_path:
        resolved = contact_encoder._resolve_cross_path(str(hparams_path))
        if resolved is not None and resolved.is_file():
            hcfg = runtime_config(OmegaConf.load(str(resolved)))
            for key in (name, name.replace("_right", ""), name.replace("_left", "")):
                value = OmegaConf.select(
                    hcfg, f"robots.{key}.num_surface_points", default=None
                )
                if value is not None:
                    return int(value)
    return int(default)


def _resolve_sparse_capacity(cfg: Any, args: argparse.Namespace) -> int:
    capacity = int(args.contact_sparse_capacity)
    if capacity not in {-1, 6}:
        raise ValueError("Paper contact extraction uses six maximum slots")
    return 6


def _resolve_manipulator_name(
    source: dict[str, np.ndarray], args: argparse.Namespace
) -> str:
    if _source_scalar_str(source, "embodiment") == "marvin_wuji":
        return "wujihand_right"
    if _source_scalar_str(source, "embodiment") != "franka_robotiq":
        raise ValueError("KITE objects-placing demonstrations must use franka_robotiq")
    return "tabletop_franka_robotiq_right"


def _infer_embodiment_from_manipulator(manipulator_name: str) -> str:
    name = str(manipulator_name).lower()
    if "wuji" in name:
        return "marvin_wuji"
    if "franka" in name or "robotiq" in name:
        return "franka_robotiq"
    if "mano" in name:
        return "mano"
    return str(manipulator_name)


def _quat_wxyz_to_euler_xyz(quat_wxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    quat = np.asarray(quat_wxyz, dtype=np.float32)
    quat_xyzw = np.stack([quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0]], axis=1)
    return R.from_quat(quat_xyzw).as_euler("XYZ").astype(np.float32)


def _episode_source_array(
    episode: dict[str, Any], key: str, *, min_tail_ndim: int
) -> np.ndarray | None:
    source = episode["source"]
    if key not in source:
        return None
    arr = _as_episode_major(
        np.asarray(source[key]), key=key, min_tail_ndim=min_tail_ndim
    )[int(episode["episode_index"])]
    arr = np.asarray(arr)
    if arr.dtype.kind in {"O", "U", "S"}:
        return None
    return arr


def _single_manipulator_action_from_unified_episode(
    episode: dict[str, Any],
    *,
    manipulator_name: str,
) -> tuple[torch.Tensor, str] | None:
    ee_pos = _episode_source_array(episode, "action_ee_pos", min_tail_ndim=1)
    ee_quat = _episode_source_array(episode, "action_ee_quat_wxyz", min_tail_ndim=1)
    if ee_pos is None or ee_quat is None:
        return None
    joint_key = (
        "action_hand_q" if "wuji" in str(manipulator_name).lower() else "action_gripper"
    )
    joint = _episode_source_array(episode, joint_key, min_tail_ndim=1)
    if joint is None and joint_key != "action_effector_ctrl":
        joint_key = "action_effector_ctrl"
        joint = _episode_source_array(episode, joint_key, min_tail_ndim=1)
    if joint is None:
        return None
    action = concat_single_manipulator_action(
        ee_pos.astype(np.float32), ee_quat.astype(np.float32), joint.astype(np.float32)
    )
    return action, joint_key


def _raw_source_tensors_for_unified_episode(
    episode: dict[str, Any],
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    scalar_frame_keys = {
        "object_index",
        "placement_margin",
        "task_success",
        "phase",
        "pointcloud_occluded",
    }
    for key in (
        "executed_qpos",
        "executed_qvel",
        "executed_ctrl",
        "action_ee_pos",
        "action_ee_quat_wxyz",
        "action_gripper",
        "action_hand_q",
        "action_effector_ctrl",
        "contact_flags",
        "object_index",
        "bin_pose",
        "placement_margin",
        "task_success",
        "phase",
        "pointcloud_occluded",
    ):
        arr = _episode_source_array(
            episode,
            key,
            min_tail_ndim=0 if key in scalar_frame_keys else 1,
        )
        if arr is None:
            continue
        out_key = "source_contact_flags" if key == "contact_flags" else key
        tensors[out_key] = torch.from_numpy(np.asarray(arr))
    return tensors


def _geom_local_mesh(model: Any, geom_id: int) -> Any | None:
    import mujoco
    import trimesh

    geom_type = int(model.geom_type[geom_id])
    if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
        return None
    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        vadr = int(model.mesh_vertadr[mesh_id])
        vnum = int(model.mesh_vertnum[mesh_id])
        fadr = int(model.mesh_faceadr[mesh_id])
        fnum = int(model.mesh_facenum[mesh_id])
        vertices = np.asarray(model.mesh_vert[vadr : vadr + vnum], dtype=np.float64)
        faces = np.asarray(model.mesh_face[fadr : fadr + fnum], dtype=np.int32)
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2.0 * size[:3])
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(
            radius=float(size[0]), height=2.0 * float(size[1])
        )
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(
            radius=float(size[0]), height=2.0 * float(size[1])
        )
    return None


def _allocate_counts(weights: np.ndarray, total: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.ones_like(weights)
    raw = weights / float(weights.sum()) * int(total)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(total) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    return counts


def _sample_mesh_surface(
    mesh: Any, count: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    count = int(count)
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tris = vertices[faces]
    areas = (
        np.linalg.norm(
            np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1
        )
        * 0.5
    )
    prob = areas / np.clip(float(areas.sum()), 1.0e-12, None)
    rng = np.random.default_rng(int(seed))
    face_idx = rng.choice(len(faces), size=count, replace=True, p=prob)
    u = rng.random((count, 1))
    v = rng.random((count, 1))
    sqrt_u = np.sqrt(u)
    bary = np.concatenate([1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v], axis=1)
    pts = (tris[face_idx] * bary[:, :, None]).sum(axis=1)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)[face_idx]
    normals = normals / np.clip(
        np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-8, None
    )
    return pts.astype(np.float32), normals.astype(np.float32)


def _exposed_template_sample_mask(model, data, points, normals, geom_ids):
    """Remove faces buried in other colliders of the same rigid source part.

    The outward epsilon is only a solid-membership query; stored points never
    move. Restricting comparisons to welded bodies makes this pose independent.
    MuJoCo mesh colliders use convex hulls, represented here by half-spaces.
    """
    from scipy.spatial import ConvexHull

    weld = model.body_weldid[model.geom_bodyid[geom_ids]]
    query = points.astype(np.float64) + 1e-6 * normals.astype(np.float64)
    hidden = np.zeros(len(points), dtype=bool)
    for geom in range(model.ngeom):
        body = model.body_weldid[model.geom_bodyid[geom]]
        if body not in weld or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        mesh = _geom_local_mesh(model, geom)
        if mesh is None:
            continue
        local = (query - data.geom_xpos[geom]) @ data.geom_xmat[geom].reshape(3, 3)
        lower, upper = mesh.bounds
        ids = np.flatnonzero(
            (geom_ids != geom) & (weld == body)
            & np.all(local > lower, axis=1) & np.all(local < upper, axis=1)
        )
        if not len(ids):
            continue
        planes = ConvexHull(mesh.vertices).equations
        for start in range(0, len(ids), 256):
            chunk = ids[start:start + 256]
            hidden[chunk] |= np.all(local[chunk] @ planes[:, :3].T + planes[:, 3] < -1e-9, axis=1)
    return ~hidden


def _exterior_surface_neighbors(points, normals, weld_ids):
    """Connect nearby samples across joined pieces of the same rigid surface."""
    from scipy.spatial import cKDTree

    neighbors = [[] for _ in points]
    for body in np.unique(weld_ids):
        ids = np.flatnonzero(weld_ids == body)
        if len(ids) < 2:
            continue
        tree = cKDTree(points[ids])
        spacing = float(np.median(tree.query(points[ids], k=2)[0][:, 1]))
        for i, j in tree.query_pairs(max(0.006, 4 * spacing)):
            if normals[ids[i]] @ normals[ids[j]] >= -0.01:
                neighbors[ids[i]].append(int(ids[j]))
                neighbors[ids[j]].append(int(ids[i]))
    return tuple(tuple(row) for row in neighbors)


@lru_cache(maxsize=4)
def _tabletop_surface_template(
    embodiment: str, side_name: str, num_surface_points: int
):
    """Fixed samples on the two physical distal inner-pad faces.

    The trajectory score selects the contact indices; neither pad has a
    predefined contact anchor.
    """
    import mujoco
    from experiment.tabletop_sim.robot import resolve_robot
    from experiment.tabletop_sim.scene import compile_tabletop_scene
    from experiment.tabletop_sim.object_suite import load_object_suite

    object_spec = load_object_suite(
        CROSS_DIR / "experiment/assets/ycb_object_suite.json"
    )[0]
    _, model, data, _ = compile_tabletop_scene(
        add_cameras=False,
        embodiment=embodiment,
        object_kind="ycb_object",
        object_visual_mesh_path=object_spec.visual_mesh_path,
        object_collision_mesh_path=object_spec.collision_mesh_path,
        object_texture_path=object_spec.texture_path,
    )
    robot = resolve_robot(model, embodiment)
    side = robot.side(side_name)
    meshes = []
    for geom_id in sorted(map(int, side.contact_geom_ids)):
        if not model.geom(geom_id).name.endswith("_pad2"):
            continue
        mesh = _geom_local_mesh(model, geom_id)
        if mesh is None:
            continue
        faces = np.flatnonzero(mesh.face_normals[:, 0] > 0.99)
        if len(faces):
            meshes.append((geom_id, mesh.submesh([faces], append=True, repair=False)))
    if len(meshes) != 2:
        raise RuntimeError(
            f"Expected two physical distal inner pads for {embodiment}/{side_name}"
        )
    counts = _allocate_counts(
        np.asarray([m.area for _, m in meshes]), num_surface_points
    )
    local_points, local_normals, geom_ids = [], [], []
    for (geom_id, mesh), count in zip(meshes, counts):
        if count == 0:
            continue
        pts, nrm = _sample_mesh_surface(
            mesh, int(count), seed=DEFAULT_CLUSTER_SEED + geom_id
        )
        local_points.append(pts)
        local_normals.append(nrm)
        geom_ids.extend([geom_id] * len(pts))
    pts_local = np.concatenate(local_points)
    nrm_local = np.concatenate(local_normals)
    geom_ids = np.asarray(geom_ids, dtype=np.int32)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    canonical = np.empty_like(pts_local)
    canonical_normals = np.empty_like(nrm_local)
    for geom_id in np.unique(geom_ids):
        mask = geom_ids == geom_id
        canonical[mask] = (
            pts_local[mask] @ data.geom_xmat[geom_id].reshape(3, 3).T
            + data.geom_xpos[geom_id]
        )
        canonical_normals[mask] = nrm_local[mask] @ data.geom_xmat[geom_id].reshape(3, 3).T
    exposed = _exposed_template_sample_mask(model, data, canonical, canonical_normals, geom_ids)
    pts_local, nrm_local, geom_ids, canonical, canonical_normals = (
        array[exposed].copy()
        for array in (pts_local, nrm_local, geom_ids, canonical, canonical_normals)
    )
    if len(pts_local) < 2:
        raise RuntimeError("Source template has fewer than two exposed surface samples")
    neighbors = _exterior_surface_neighbors(
        canonical, canonical_normals, model.body_weldid[model.geom_bodyid[geom_ids]]
    )
    edges = np.asarray(
        [(i, j) for i, row in enumerate(neighbors) for j in row], dtype=np.int64
    ).reshape(-1, 2)
    graph_hash = _short_hash_from_arrays(edges, prefix="distal_inner_pad_graph_v1")
    template_hash = _short_hash_from_arrays(
        pts_local,
        nrm_local,
        geom_ids,
        canonical,
        edges,
        prefix=f"{embodiment}:{side_name}:distal_inner_pad_v1",
    )
    return (
        model,
        data,
        side,
        pts_local,
        nrm_local,
        geom_ids,
        canonical,
        tuple(neighbors),
        template_hash,
        graph_hash,
    )


def _tabletop_qpos_surface_trajectory(
    *,
    source: dict[str, np.ndarray],
    qpos: np.ndarray,
    manipulator_name: str,
    side: str,
    num_surface_points: int,
) -> SurfaceTrajectory:
    import mujoco

    embodiment = _source_scalar_str(source, "embodiment")
    side_name = "right" if side == "auto" else side
    (
        model,
        data,
        side_spec,
        local_points,
        local_normals,
        geom_ids,
        canonical,
        neighbors,
        template_hash,
        graph_hash,
    ) = _tabletop_surface_template(embodiment, side_name, int(num_surface_points))
    q = np.asarray(qpos, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != model.nq or not np.isfinite(q).all():
        raise ValueError(
            f"Expected finite source qpos with shape (T, {model.nq}), got {q.shape}"
        )
    points = np.empty((len(q), len(local_points), 3), dtype=np.float32)
    normals = np.empty_like(points)
    for frame, configuration in enumerate(q):
        data.qpos[:] = configuration
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for geom_id in np.unique(geom_ids):
            mask = geom_ids == geom_id
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            points[frame, mask] = (
                local_points[mask] @ rotation.T + data.geom_xpos[geom_id]
            )
            normals[frame, mask] = local_normals[mask] @ rotation.T
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
    return SurfaceTrajectory(
        points,
        normals,
        canonical,
        template_hash,
        len(local_points),
        q[:, side_spec.qpos_indices],
        "tabletop_qpos",
        f"tabletop_{embodiment}_{side_name}",
        neighbors,
        graph_hash,
    )


def _surface_trajectory_for_unified_episode(
    *,
    episode: dict[str, Any],
    args: argparse.Namespace,
    manipulator_name: str,
    num_surface_points: int,
) -> SurfaceTrajectory:
    if episode["source_meta"].get("task") in {"keyboard_press", "pumpbottle_use"}:
        from experiment.manipulation.articulated_source import surface_trajectory

        return surface_trajectory(episode, int(num_surface_points),
                                  int(getattr(args, 'object_surface_points', 4096)),
                                  str(getattr(args, 'source_configuration', 'commanded')))
    if args.surface_provider not in {"auto", "tabletop_qpos"}:
        raise ValueError(
            "Objects placing uses the tabletop_qpos source surface provider"
        )
    return _tabletop_qpos_surface_trajectory(
        source=episode["source"],
        qpos=episode["manipulator_q_source"],
        manipulator_name=manipulator_name,
        side=args.tabletop_side,
        num_surface_points=num_surface_points,
    )


def _real_contact_slot_mask_from_result(contact_result, *, contact_shape, mode_key):
    return np.asarray(contact_result["contact_slot_real_mask"], dtype=bool)


def _encode_latents_if_requested(
    *,
    args: argparse.Namespace,
    latent_encoder: dict[str, object] | None,
    contact_action: np.ndarray,
    contact_action_valid_mask: np.ndarray,
    contact_proprio: np.ndarray,
    contact_proprio_valid_mask: np.ndarray,
    contact_mode: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if latent_encoder is None:
        return None, None
    mode = str(contact_mode).strip().lower()
    uses_sparse_valid_mask = mode in {
        "sparse",
        "trajectory_sparse",
    }
    latent_action = contact_encoder._encode_contact_latent_sequence(
        contact_cloud=contact_action,
        latent_encoder=latent_encoder,
        batch_size=int(args.latent_encoder_batch_size),
        zero_centered=bool(args.latent_use_xyz_translation),
        valid_mask=contact_action_valid_mask if uses_sparse_valid_mask else None,
    )
    latent_proprio = contact_encoder._encode_contact_latent_sequence(
        contact_cloud=contact_proprio,
        latent_encoder=latent_encoder,
        batch_size=int(args.latent_encoder_batch_size),
        zero_centered=bool(args.latent_use_xyz_translation),
        valid_mask=contact_proprio_valid_mask if uses_sparse_valid_mask else None,
    )
    return latent_proprio, latent_action


def _safe_box_from_low_level_cfg(cfg: Any) -> tuple[np.ndarray, np.ndarray]:
    safe_min = np.asarray(
        cfg.dataset.q2_base_translation_min, dtype=np.float32
    ).reshape(-1)
    safe_max = np.asarray(
        cfg.dataset.q2_base_translation_max, dtype=np.float32
    ).reshape(-1)
    if safe_min.shape != (3,) or safe_max.shape != (3,):
        raise ValueError(
            "Low-level cfg.dataset.q2_base_translation_min/max must be length-3 vectors; "
            f"got {safe_min.shape} and {safe_max.shape}"
        )
    if bool((safe_max < safe_min).any()):
        raise ValueError(
            "Low-level cfg.dataset.q2_base_translation_max must be >= q2_base_translation_min elementwise; "
            f"got min={safe_min.tolist()} max={safe_max.tolist()}"
        )
    return safe_min.astype(np.float32, copy=False), safe_max.astype(
        np.float32, copy=False
    )


def _valid_contact_xyz(frame: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    cloud = np.asarray(frame, dtype=np.float32)
    if cloud.ndim != 3 or int(cloud.shape[-1]) < 3:
        raise ValueError(
            f"contact cloud must have shape (T, N, >=3), got {cloud.shape}"
        )
    if mask is None:
        valid = np.isfinite(cloud[..., :3]).all(axis=-1) & (
            np.linalg.norm(cloud[..., :3], axis=-1) > 1.0e-8
        )
    else:
        valid = np.asarray(mask, dtype=np.bool_)
        if valid.shape != cloud.shape[:2]:
            raise ValueError(
                f"contact mask shape {valid.shape} does not match cloud shape {cloud.shape[:2]}"
            )
    xyz = cloud[..., :3][valid]
    if xyz.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return xyz.astype(np.float32, copy=False).reshape(-1, 3)


def _contact_mask_for_shift(
    tensors: dict[str, torch.Tensor], key: str, cloud: np.ndarray
) -> np.ndarray:
    mask_key = f"{key}_valid_mask"
    if mask_key in tensors:
        mask = tensors[mask_key].detach().cpu().numpy().astype(np.bool_)
        if mask.shape == cloud.shape[:2]:
            return mask
    return np.isfinite(cloud[..., :3]).all(axis=-1) & (
        np.linalg.norm(cloud[..., :3], axis=-1) > 1.0e-8
    )


def _shift_contact_tensor_valid_slots(
    tensor: torch.Tensor, mask: np.ndarray, shift: np.ndarray
) -> torch.Tensor:
    arr = tensor.detach().cpu().numpy().astype(np.float32, copy=True)
    if arr.ndim != 3 or int(arr.shape[-1]) < 3:
        raise ValueError(f"contact tensor must have shape (T, N, >=3), got {arr.shape}")
    if mask.shape != arr.shape[:2]:
        raise ValueError(
            f"contact shift mask shape {mask.shape} does not match contact tensor shape {arr.shape[:2]}"
        )
    arr[..., :3][mask] += shift.reshape(1, 3)
    return torch.from_numpy(arr)


def _apply_shift_to_contact_encoder_inputs(
    tensors: dict[str, torch.Tensor], shift: np.ndarray
) -> None:
    for key in ("contact_action", "contact_proprio"):
        if key not in tensors:
            continue
        if f"{key}_world" not in tensors:
            tensors[f"{key}_world"] = tensors[key].detach().cpu().clone()
        cloud = tensors[key].detach().cpu().numpy().astype(np.float32, copy=False)
        mask = _contact_mask_for_shift(tensors, key, cloud)
        tensors[key] = _shift_contact_tensor_valid_slots(tensors[key], mask, shift)


def _reencode_shifted_latents(
    *,
    episode_tensors_all: list[dict[str, torch.Tensor]],
    args: argparse.Namespace,
    latent_encoder: dict[str, object] | None,
) -> None:
    if latent_encoder is None:
        return
    mode_key = str(args.contact_mode_resolved).strip().lower()
    for tensors in episode_tensors_all:
        if "contact_action" not in tensors or "contact_proprio" not in tensors:
            continue
        contact_action = (
            tensors["contact_action"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        contact_proprio = (
            tensors["contact_proprio"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        action_mask = tensors.get("contact_action_valid_mask")
        proprio_mask = tensors.get("contact_proprio_valid_mask")
        latent_proprio, latent_action = _encode_latents_if_requested(
            args=args,
            latent_encoder=latent_encoder,
            contact_action=contact_action,
            contact_action_valid_mask=(
                action_mask.detach().cpu().numpy().astype(np.bool_)
                if action_mask is not None
                else np.ones(contact_action.shape[:2], dtype=np.bool_)
            ),
            contact_proprio=contact_proprio,
            contact_proprio_valid_mask=(
                proprio_mask.detach().cpu().numpy().astype(np.bool_)
                if proprio_mask is not None
                else np.ones(contact_proprio.shape[:2], dtype=np.bool_)
            ),
            contact_mode=mode_key,
        )
        if latent_proprio is not None and latent_action is not None:
            tensors["latent_proprio"] = torch.from_numpy(
                latent_proprio.astype(np.float32, copy=False)
            )
            tensors["latent_action"] = torch.from_numpy(
                latent_action.astype(np.float32, copy=False)
            )


def _fit_and_apply_latent_dataset_translation(
    *,
    episode_tensors_all: list[dict[str, torch.Tensor]],
    args: argparse.Namespace,
    low_level_cfg: Any,
    latent_encoder: dict[str, object] | None,
) -> dict[str, object]:
    action_space_mode = (
        str(getattr(args, "action_space", "latent_contact")).strip().lower()
    )
    if action_space_mode not in {"latent_contact", "contact_points"}:
        return {}

    safe_min, safe_max = _safe_box_from_low_level_cfg(low_level_cfg)
    contact_xyz_chunks: list[np.ndarray] = []
    for tensors in episode_tensors_all:
        for key in ("contact_action", "contact_proprio"):
            if key not in tensors:
                continue
            cloud = tensors[key].detach().cpu().numpy().astype(np.float32, copy=False)
            mask_key = f"{key}_valid_mask"
            mask = (
                tensors[mask_key].detach().cpu().numpy().astype(np.bool_)
                if mask_key in tensors
                else None
            )
            xyz = _valid_contact_xyz(cloud, mask)
            if xyz.size:
                contact_xyz_chunks.append(xyz)
    if not contact_xyz_chunks:
        raise RuntimeError(
            "Cannot fit dataset xyz translation because no valid contact points were extracted"
        )

    xyz_all = np.concatenate(contact_xyz_chunks, axis=0).astype(np.float32, copy=False)
    bbox_min = xyz_all.min(axis=0).astype(np.float32)
    bbox_max = xyz_all.max(axis=0).astype(np.float32)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    safe_center = 0.5 * (safe_min + safe_max)
    shift = (safe_center - bbox_center).astype(np.float32)
    translation_strategy = "center_contact_bbox_in_safe_box"
    if args.coordinate_reference_payload:
        reference = load_high_level_payload(args.coordinate_reference_payload)
        # Reuse the reference payload's training-frame translation so both
        # payloads share decoder coordinates.
        if reference["meta"].get("contact_extraction_version") not in {
            "paper_trajectory_contacts_v1",
            "paper_trajectory_contacts_v2",
            "paper_trajectory_contacts_v3",
            "paper_trajectory_contacts_v4",
            CONTACT_EXTRACTION_VERSION,
        }:
            raise ValueError("Coordinate reference must use paper-extracted contacts")
        shift = np.asarray(
            reference["meta"]["dataset_xyz_translation"], dtype=np.float32
        )
        translation_strategy = "training_payload_reference"
    elif args.dataset_xyz_translation is not None:
        shift = np.asarray(args.dataset_xyz_translation, dtype=np.float32)
        translation_strategy = "fixed_training_calibration"
    if shift.shape != (3,) or not np.isfinite(shift).all():
        raise ValueError("Dataset xyz translation must contain three finite coordinates")
    shifted_min = (bbox_min + shift).astype(np.float32)
    shifted_max = (bbox_max + shift).astype(np.float32)

    tol = 1.0e-5
    if bool((shifted_min < safe_min - tol).any()) or bool(
        (shifted_max > safe_max + tol).any()
    ):
        bbox_extent = (bbox_max - bbox_min).astype(np.float32)
        safe_extent = (safe_max - safe_min).astype(np.float32)
        raise ValueError(
            "Cannot center dataset contact bbox inside the low-level q2_base_translation safe box. "
            f"contact_bbox_min={bbox_min.tolist()} contact_bbox_max={bbox_max.tolist()} "
            f"contact_bbox_extent={bbox_extent.tolist()} safe_min={safe_min.tolist()} "
            f"safe_max={safe_max.tolist()} safe_extent={safe_extent.tolist()} "
            f"centered_shift={shift.tolist()} shifted_min={shifted_min.tolist()} "
            f"shifted_max={shifted_max.tolist()}"
        )

    for tensors in episode_tensors_all:
        _apply_shift_to_contact_encoder_inputs(tensors, shift)
    if action_space_mode == "latent_contact":
        _reencode_shifted_latents(
            episode_tensors_all=episode_tensors_all,
            args=args,
            latent_encoder=latent_encoder,
        )

    print(
        "Applied low-level contact xyz translation: "
        f"shift={shift.tolist()} safe_min={safe_min.tolist()} safe_max={safe_max.tolist()} "
        f"contact_bbox_world_min={bbox_min.tolist()} contact_bbox_world_max={bbox_max.tolist()} "
        f"contact_bbox_shifted_min={shifted_min.tolist()} contact_bbox_shifted_max={shifted_max.tolist()}",
        flush=True,
    )
    return {
        "dataset_xyz_translation": shift.astype(np.float32).tolist(),
        "dataset_xyz_translation_strategy": translation_strategy,
        "dataset_xyz_translation_applied": True,
        "dataset_xyz_translation_applied_to": "contact_cloud_inputs",
        "observation_xyz_translation_applied": False,
        "dataset_xyz_safe_box_min": safe_min.astype(np.float32).tolist(),
        "dataset_xyz_safe_box_max": safe_max.astype(np.float32).tolist(),
        "contact_bbox_world_min": bbox_min.astype(np.float32).tolist(),
        "contact_bbox_world_max": bbox_max.astype(np.float32).tolist(),
        "contact_bbox_shifted_min": shifted_min.astype(np.float32).tolist(),
        "contact_bbox_shifted_max": shifted_max.astype(np.float32).tolist(),
    }


@lru_cache(maxsize=6)
def _object_scoring_surface_template(object_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Sample the demonstration object's surface once, independently of policy observations."""
    import trimesh
    from experiment.tabletop_sim.object_suite import load_object_suite

    specs = {
        spec.object_id: spec
        for spec in load_object_suite(CROSS_DIR / "experiment/assets/ycb_object_suite.json")
    }
    if object_id not in specs:
        raise ValueError(f"Unknown demonstration object for contact scoring: {object_id!r}")
    mesh = trimesh.load(specs[object_id].visual_mesh_path, force="mesh", process=False)
    return _sample_mesh_surface(mesh, 4096, seed=42)


def _object_scoring_template(object_id: str) -> np.ndarray:
    return _object_scoring_surface_template(object_id)[0]


def _object_scoring_normals(episode: dict[str, Any]) -> np.ndarray:
    if 'contact_scoring_normals_world' in episode:
        return episode['contact_scoring_normals_world']
    from scipy.spatial.transform import Rotation
    object_id = str(episode.get('source_episode_meta', {}).get('object_id', ''))
    normals = _object_scoring_surface_template(object_id)[1]
    pose = np.asarray(episode['object_pose'], dtype=np.float32)
    rotation = Rotation.from_quat(pose[:, [4, 5, 6, 3]]).as_matrix().astype(np.float32)
    return normals[None] @ np.swapaxes(rotation, 1, 2)


def _object_scoring_trajectory(episode: dict[str, Any]) -> np.ndarray:
    """Transform the fixed object surface with the measured demonstration poses."""
    if "contact_scoring_points_world" in episode:
        return episode["contact_scoring_points_world"]
    from scipy.spatial.transform import Rotation

    object_id = str(episode.get("source_episode_meta", {}).get("object_id", ""))
    points = _object_scoring_template(object_id)
    pose = np.asarray(episode["object_pose"], dtype=np.float32)
    rotation = Rotation.from_quat(pose[:, [4, 5, 6, 3]]).as_matrix().astype(np.float32)
    return points[None] @ np.swapaxes(rotation, 1, 2) + pose[:, None, :3]


def _prepare_unified_episode(
    *,
    episode: dict[str, Any],
    args: argparse.Namespace,
    latent_encoder: dict[str, object] | None,
    manipulator_name: str,
    surface_template_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    surface = _surface_trajectory_for_unified_episode(
        episode=episode,
        args=args,
        manipulator_name=manipulator_name,
        num_surface_points=int(surface_template_size),
    )
    mode_key = str(args.contact_mode_resolved).strip().lower()
    if args.contact_alignment_mode not in {'displacement_direction', 'nearest_surface_normals'}:
        raise ValueError('Unknown contact alignment mode')
    contact_result = extract_contact_sequence_from_surface(
        surface_points_world=surface.surface_points_world,
        surface_normals_world=surface.surface_normals_world,
        surface_points_canonical=surface.surface_points_canonical,
        object_points_world=_object_scoring_trajectory(episode),
        object_normals_world=(_object_scoring_normals(episode) if args.contact_alignment_mode == 'nearest_surface_normals' else None),
        contact_threshold=float(args.contact_threshold),
        surface_neighbors=surface.surface_neighbors,
        align_exp_scale=float(args.align_exp_scale),
        sigmoid_scale=float(args.sigmoid_scale),
        contact_mode=mode_key,
        contact_sparse_capacity=int(args.contact_sparse_capacity_resolved),
        seed=int(args.seed) + int(episode["episode_index"]),
        score_chunk_frames=args.score_chunk_frames,
        score_device=args.score_device,
    )
    contact_action = np.asarray(contact_result["contact_cloud"], dtype=np.float32)
    contact_action_valid_mask = np.asarray(
        contact_result["contact_valid_mask"], dtype=np.bool_
    )
    contact_real_slot_mask = _real_contact_slot_mask_from_result(
        contact_result,
        contact_shape=contact_action_valid_mask.shape,
        mode_key=mode_key,
    )
    contact_surface_action_world = contact_action.copy()
    offset_maximum = float(getattr(args, "contact_position_offset_m", 0.0))
    source_meta = dict(episode.get("source_meta", {}))
    phase_ids = source_meta.get("phase_ids", {})
    if not phase_ids and isinstance(source_meta.get("source"), dict):
        phase_ids = source_meta["source"].get("phase_ids", {})
    contact_action, contact_position_offset_m = offset_contact_positions(
        contact_action,
        contact_real_slot_mask,
        maximum_m=offset_maximum,
        phase=_episode_source_array(episode, "phase", min_tail_ndim=0) if offset_maximum else None,
        close_phase_id=phase_ids.get("close"),
    )
    contact_proprio = np.zeros_like(contact_action, dtype=np.float32)
    contact_proprio_valid_mask = np.zeros_like(
        contact_action_valid_mask, dtype=np.bool_
    )
    contact_proprio[0] = contact_action[0]
    contact_proprio_valid_mask[0] = contact_action_valid_mask[0]
    if contact_action.shape[0] > 1:
        contact_proprio[1:] = contact_action[:-1]
        contact_proprio_valid_mask[1:] = contact_action_valid_mask[:-1]

    manipulator_action = np.asarray(surface.manipulator_q, dtype=np.float32)
    manipulator_proprio = np.zeros_like(manipulator_action, dtype=np.float32)
    if manipulator_action.shape[0] > 1:
        manipulator_proprio[1:] = manipulator_action[:-1]

    latent_proprio, latent_action = _encode_latents_if_requested(
        args=args,
        latent_encoder=latent_encoder,
        contact_action=contact_action,
        contact_action_valid_mask=contact_action_valid_mask,
        contact_proprio=contact_proprio,
        contact_proprio_valid_mask=contact_proprio_valid_mask,
        contact_mode=mode_key,
    )

    object_points = np.asarray(episode["object_points_world"], dtype=np.float32)
    object_pose = np.asarray(episode["object_pose"], dtype=np.float32)
    tensors: dict[str, torch.Tensor] = {
        "manipulated_point_cloud": torch.from_numpy(object_points.astype(np.float16)),
        "context_point_cloud": torch.zeros(
            (object_points.shape[0], 0, 3), dtype=torch.float16
        ),
        "contact_action_world": torch.from_numpy(contact_action.copy()),
        "contact_surface_action_world": torch.from_numpy(contact_surface_action_world),
        "contact_position_offset_m": torch.from_numpy(contact_position_offset_m),
        "contact_region_counts": torch.from_numpy(
            contact_result["contact_region_counts"]
        ),
        "contact_proprio": torch.from_numpy(contact_proprio),
        "contact_action": torch.from_numpy(contact_action),
        "contact_proprio_valid_mask": torch.from_numpy(contact_proprio_valid_mask),
        "contact_action_valid_mask": torch.from_numpy(contact_action_valid_mask),
        "contact_sparse_mask": torch.from_numpy(
            np.asarray(contact_result["filled_sparse_mask"], dtype=np.bool_)
        ),
        "contact_real_sparse_mask": torch.from_numpy(
            np.asarray(contact_result["real_sparse_mask"], dtype=np.bool_)
        ),
        "contact_source_frame_index": torch.from_numpy(
            np.asarray(contact_result["source_frame_index"], dtype=np.int64)
        ),
        "contact_has_real": torch.from_numpy(
            np.asarray(contact_result["has_real_contact"], dtype=np.bool_)
        ),
        "manipulator_proprio": torch.from_numpy(manipulator_proprio),
        "manipulator_action": torch.from_numpy(manipulator_action),
        "mano_proprio": torch.from_numpy(manipulator_proprio),
        "mano_action": torch.from_numpy(manipulator_action),
        "object_pose": torch.from_numpy(object_pose),
    }
    if mode_key == "patches":
        tensors["contact_patches_template_indices"] = torch.from_numpy(
            np.asarray(contact_result["patches_template_indices"], dtype=np.int64)
        )
    else:
        tensors["contact_selected_template_indices"] = torch.from_numpy(
            np.asarray(contact_result["selected_template_indices"], dtype=np.int64)
        )
        tensors["contact_num_selected_per_frame"] = torch.from_numpy(
            np.asarray(contact_result["num_selected_per_frame"], dtype=np.int32)
        )
        tensors["contact_slot_real_mask"] = torch.from_numpy(
            np.asarray(contact_result["contact_slot_real_mask"], dtype=np.bool_)
        )
        tensors["contact_slot_supplement_mask"] = torch.from_numpy(
            np.asarray(contact_result["contact_slot_supplement_mask"], dtype=np.bool_)
        )
        tensors["contact_slot_source_frame"] = torch.from_numpy(
            np.asarray(contact_result["contact_slot_source_frame"], dtype=np.int64)
        )
    if latent_proprio is not None and latent_action is not None:
        tensors["latent_proprio"] = torch.from_numpy(latent_proprio.astype(np.float32))
        tensors["latent_action"] = torch.from_numpy(latent_action.astype(np.float32))
    single_action = _single_manipulator_action_from_unified_episode(
        episode, manipulator_name=manipulator_name
    )
    if single_action is not None:
        action_t, joint_key = single_action
        tensors["single_manipulator_action"] = action_t
        tensors["single_manipulator_proprio"] = previous_action_proprio(
            action_t,
            torch.as_tensor([int(action_t.shape[0])], dtype=torch.long),
        )
        tensors.update(_raw_source_tensors_for_unified_episode(episode))

    has_real = np.asarray(contact_result["has_real_contact"], dtype=np.bool_)
    real_indices = np.flatnonzero(has_real)
    if real_indices.size == 0:
        raise RuntimeError(
            f"Unified episode {episode['episode_index']}: no real contact frames"
        )
    source_episode_meta = dict(episode.get("source_episode_meta", {}))
    source_meta = dict(episode.get("source_meta", {}))
    meta = {
        "source_format": "unified_pt",
        "source_pt_path": str(Path(args.source_pt).expanduser().resolve()),
        "source_trajectory_index": int(
            episode.get("source_trajectory_index", episode["episode_index"])
        ),
        "object_name": str(
            source_episode_meta.get(
                "object_name", f"trajectory_{int(episode['episode_index']):06d}"
            )
        ),
        "context_object_name": "none",
        "object_pc_key": str(episode["object_pc_key"]),
        "object_pose_key": str(episode["object_pose_key"]),
        "manipulator_q_key": str(episode["manipulator_q_key"]),
        "manipulator_name": str(manipulator_name),
        "resolved_manipulator_name": str(surface.resolved_manipulator_name),
        "surface_provider": str(surface.provider),
        "surface_template_hash": str(surface.surface_template_hash),
        "surface_template_size": int(surface.surface_template_size),
        "contact_template_hash": str(surface.surface_template_hash),
        "contact_template_version": (
            "wuji_collision_surface_fixed_count_v4" if surface.provider.endswith("articulated_hand_fk")
            else "distal_inner_pad_surface_v1"
        ),
        "contact_object_surface_provider": (
            "articulated_collision_surface" if surface.provider.endswith("articulated_hand_fk")
            else "visual_mesh"
        ),
        "contact_object_surface_points": (int(args.object_surface_points)
                                          if surface.provider.endswith("articulated_hand_fk") else 4096),
        "contact_object_surface_seed": 42,
        "contact_alignment_mode": args.contact_alignment_mode,
        "contact_extraction_version": CONTACT_EXTRACTION_VERSION,
        "contact_region_average_scope": "active_contact_frames",
        "contact_position_offset_max_m": offset_maximum,
        "contact_position_offset_schedule": "linear_close_phase_real_slots",
        "surface_graph_hash": surface.surface_graph_hash,
        "num_frames": int(object_points.shape[0]),
        "num_real_contact_frames": int(has_real.sum()),
        "first_real_contact_frame": int(real_indices[0]),
    }
    if source_meta.get("task_objective") is not None:
        meta["task_objective"] = str(source_meta["task_objective"])
    for key in (
        "object_id",
        "object_index",
        "object_asset_hash",
        "object_bounds_m",
        "object_mass_kg",
    ):
        if source_episode_meta.get(key) is not None:
            meta[key] = source_episode_meta[key]
    if single_action is not None:
        meta["single_manipulator_joint_key"] = str(single_action[1])
        meta["single_manipulator_joint_dim"] = int(single_action[0].shape[-1]) - 7
    return tensors, meta


def _build_single_payload(
    *,
    episode_tensors_all: list[dict[str, torch.Tensor]],
    episode_meta_all: list[dict[str, object]],
    args: argparse.Namespace,
    source_format: str,
    manipulator_name: str,
    high_level_config_path: Path,
    low_level_config_path: Path,
    latent_encoder_info: dict[str, object] | None,
    skipped_no_contact: int,
    source_pt_path: Path | None = None,
    dataset_root: Path | None = None,
    mano_model_root: Path | None = None,
    dataset_xyz_translation_meta: dict[str, object] | None = None,
) -> dict[str, object]:
    if len(episode_tensors_all) == 0:
        raise RuntimeError("Cannot build an empty high-level payload")
    data_keys = list(episode_tensors_all[0].keys())
    buffers = {key: [] for key in data_keys}
    episode_meta: list[dict[str, object]] = []
    episode_ends: list[int] = []
    total_frames = 0
    object_distribution: Counter[str] = Counter()
    context_distribution: Counter[str] = Counter()
    template_hashes: set[str] = set()
    template_sizes: set[int] = set()
    template_versions: set[str] = set()
    contact_alignment_modes: set[str] = set()

    for episode_idx, tensors in enumerate(episode_tensors_all):
        missing = set(data_keys).difference(tensors)
        if missing:
            raise KeyError(
                f"Episode {episode_idx} is missing payload fields: {sorted(missing)}"
            )
        meta = dict(episode_meta_all[int(episode_idx)])
        meta.pop("dataset_partition", None)
        for key in data_keys:
            buffers[key].append(tensors[key])
        total_frames += int(meta["num_frames"])
        episode_ends.append(total_frames)
        episode_meta.append(meta)
        object_distribution[
            str(meta.get("object_name", f"episode_{episode_idx:06d}"))
        ] += 1
        context_name = meta.get("context_object_name")
        if context_name is not None:
            context_distribution[str(context_name)] += 1
        template_hash = meta.get(
            "surface_template_hash", meta.get("contact_template_hash")
        )
        if template_hash is not None:
            template_hashes.add(str(template_hash))
        template_size = meta.get(
            "surface_template_size", meta.get("mano_surface_template_size")
        )
        if template_size is not None:
            template_sizes.add(int(template_size))
        template_version = meta.get("contact_template_version")
        if template_version is not None:
            template_versions.add(str(template_version))
        contact_alignment_modes.add(
            str(meta.get("contact_alignment_mode", "nearest_normal"))
        )

    data = {key: torch.cat(value, dim=0) for key, value in buffers.items()}
    latent_dim = (
        int(latent_encoder_info.get("latent_dim", 128))
        if latent_encoder_info is not None
        else 128
    )
    manipulated_count = int(data["manipulated_point_cloud"].shape[1])
    context_count = (
        int(data["context_point_cloud"].shape[1])
        if "context_point_cloud" in data
        else 0
    )
    dataset_name = _dataset_name(args)
    source_format_key = str(source_format)
    contact_mode = str(args.contact_mode_resolved)
    contact_fill_policy = "fixed_trajectory_indices_current_frame_fk"
    meta: dict[str, object] = {
        "version": 9,
        "protocol": "high_level_training_v2",
        "source_format": source_format_key,
        "dataset_name": dataset_name,
        "fps": float(args.fps),
        "manipulated_obs_point_count": manipulated_count,
        "context_obs_point_count": context_count,
        "object_dense_point_count": int(
            getattr(args, "object_dense_point_count", manipulated_count)
        ),
        "object_pose_quat_order": "wxyz",
        "manipulator_name": str(manipulator_name),
        "surface_template_hash": (
            sorted(template_hashes)[0] if len(template_hashes) == 1 else "mixed"
        ),
        "surface_template_hashes": sorted(template_hashes),
        "surface_template_size": (
            sorted(template_sizes)[0] if len(template_sizes) == 1 else -1
        ),
        "surface_template_sizes": sorted(template_sizes),
        "contact_template_version": (
            sorted(template_versions)[0] if len(template_versions) == 1 else "mixed"
        ),
        "contact_mode": contact_mode,
        "contact_sparse_capacity": int(args.contact_sparse_capacity_resolved),
        "contact_dense_count": int(args.contact_count),
        "contact_count": int(args.contact_count),
        "contact_threshold": float(args.contact_threshold),
        "contact_object_surface_provider": "visual_mesh",
        "contact_object_surface_points": 4096,
        "contact_object_surface_seed": 42,
        "contact_alignment_mode": (
            sorted(contact_alignment_modes)[0]
            if len(contact_alignment_modes) == 1
            else "mixed"
        ),
        "contact_alignment_modes": sorted(contact_alignment_modes),
        "align_exp_scale": float(args.align_exp_scale),
        "sigmoid_scale": float(args.sigmoid_scale),
        "contact_position_offset_max_m": float(getattr(args, "contact_position_offset_m", 0.0)),
        "contact_position_offset_schedule": "linear_close_phase_real_slots",
        "contact_fill_policy": contact_fill_policy,
        "contact_extraction_version": CONTACT_EXTRACTION_VERSION,
        "contact_region_average_scope": "active_contact_frames",
        "source_surface_graph_hashes": sorted(
            {m["surface_graph_hash"] for m in episode_meta}
        ),
        "seed": int(args.seed),
        "high_level_policy_config": str(high_level_config_path),
        "action_policy_config": str(low_level_config_path),
        "low_level_config_path": str(low_level_config_path),
        "latent_contact_encoder_checkpoint": (
            None
            if latent_encoder_info is None
            else latent_encoder_info.get("checkpoint_path")
        ),
        "latent_contact_encoder_loaded": bool(
            False
            if latent_encoder_info is None
            else latent_encoder_info.get("loaded", False)
        ),
        "latent_encoder_backend": (
            str(latent_encoder_info.get("backend", "none"))
            if latent_encoder_info is not None
            else "none"
        ),
        "latent_backend": (
            str(latent_encoder_info.get("backend", "none"))
            if latent_encoder_info is not None
            else "none"
        ),
        "latent_encoder_is_cotrain_checkpoint": (
            bool(latent_encoder_info.get("cotrain_checkpoint", False))
            if latent_encoder_info is not None
            else False
        ),
        "latent_encoder_dim": latent_dim,
        "latent_shift_dim": 3 if bool(args.latent_use_xyz_translation) else 0,
        "latent_action_dim": latent_dim
        + (3 if bool(args.latent_use_xyz_translation) else 0),
        "latent_zero_centered": bool(args.latent_use_xyz_translation),
        "latent_use_xyz_translation": bool(args.latent_use_xyz_translation),
        "latent_format": (
            "xyz_shift_plus_zero_centered_contact_only_encoder_cls"
            if bool(args.latent_use_xyz_translation)
            else "contact_only_encoder_cls"
        ),
        "num_episodes": int(len(episode_meta)),
        "num_frames": int(total_frames),
        "object_instance_episode_counts": dict(object_distribution),
        "context_instance_episode_counts": dict(context_distribution),
        "skipped_no_contact_episodes": int(skipped_no_contact),
    }
    if dataset_xyz_translation_meta:
        meta.update(dataset_xyz_translation_meta)
    if source_pt_path is not None:
        meta["source_pt_path"] = str(source_pt_path)
        meta["object_pointcloud_frame"] = str(args.object_pointcloud_frame)
    if dataset_root is not None:
        meta["dataset_root"] = str(dataset_root)
    episode_ends_tensor = torch.tensor(episode_ends, dtype=torch.long)
    action_space_mode = (
        str(getattr(args, "action_space", "latent_contact")).strip().lower()
    )
    if action_space_mode == "latent_contact":
        if "latent_action" not in data or "latent_proprio" not in data:
            raise ValueError(
                "Canonical action_space=latent_contact requires latent_action/latent_proprio. "
                "Remove --skip-latent or choose --action-space=contact_points/single_manipulator."
            )
        canonical_action = data["latent_action"].to(dtype=torch.float32)
        # The canonical high-level contract defines proprio as the previous
        # action within each episode. Encoding the shifted action and proprio
        # clouds in separate batches can introduce small backend-dependent
        # numerical differences, so enforce that contract after encoding.
        canonical_proprio = previous_action_proprio(
            canonical_action, episode_ends_tensor
        )
        data["latent_proprio"] = canonical_proprio
        action_space = latent_contact_action_space(
            dim=int(np.prod(tuple(canonical_action.shape[1:]))),
            latent_encoder_dim=int(meta.get("latent_encoder_dim", 128)),
            latent_use_xyz_translation=bool(
                meta.get("latent_use_xyz_translation", True)
            ),
        )
    elif action_space_mode == "contact_points":
        if "contact_action" not in data or "contact_proprio" not in data:
            raise ValueError(
                "Canonical action_space=contact_points requires contact_action/contact_proprio."
            )
        contact_action = data["contact_action"].to(dtype=torch.float32)
        contact_proprio = data["contact_proprio"].to(dtype=torch.float32)
        if contact_action.ndim != 3 or int(contact_action.shape[-1]) < 6:
            raise ValueError(
                f"contact_action must have shape (T, K, >=6), got {tuple(contact_action.shape)}"
            )
        if (
            tuple(contact_proprio.shape[:2]) != tuple(contact_action.shape[:2])
            or int(contact_proprio.shape[-1]) < 6
        ):
            raise ValueError(
                "contact_proprio must have the same leading (T, K) shape as contact_action "
                f"and at least 6 channels, got {tuple(contact_proprio.shape)} vs {tuple(contact_action.shape)}"
            )
        canonical_action = contact_action[..., :6].contiguous()
        canonical_proprio = contact_proprio[..., :6].contiguous()
        action_space = contact_points_action_space(
            slot_count=int(canonical_action.shape[1]),
            point_dim=int(canonical_action.shape[2]),
        )
    elif action_space_mode == "single_manipulator":
        if (
            "single_manipulator_action" not in data
            or "single_manipulator_proprio" not in data
        ):
            raise ValueError(
                "Canonical action_space=single_manipulator requires source action_ee_pos, "
                "action_ee_quat_wxyz, and action_gripper/action_hand_q fields."
            )
        canonical_action = data["single_manipulator_action"].to(dtype=torch.float32)
        canonical_proprio = data["single_manipulator_proprio"].to(dtype=torch.float32)
        joint_keys = {
            str(item.get("single_manipulator_joint_key", "joint_target"))
            for item in episode_meta
            if item.get("single_manipulator_joint_key") is not None
        }
        joint_dims = {
            int(
                item.get(
                    "single_manipulator_joint_dim", int(canonical_action.shape[-1]) - 7
                )
            )
            for item in episode_meta
        }
        if len(joint_dims) != 1:
            raise ValueError(
                f"Mixed single_manipulator joint dims are not supported: {sorted(joint_dims)}"
            )
        joint_key = sorted(joint_keys)[0] if len(joint_keys) == 1 else "joint_target"
        action_space = single_manipulator_action_space(
            joint_key=joint_key, joint_dim=sorted(joint_dims)[0]
        )
    else:
        raise ValueError(f"Unsupported --action-space={action_space_mode!r}")

    source_indices = [
        int(item["source_trajectory_index"])
        for item in episode_meta
        if str(item.get("source_format", source_format_key)) == "unified_pt"
        and "source_trajectory_index" in item
    ]
    source = {
        "kind": source_format_key,
        "path": None if source_pt_path is None else str(source_pt_path),
        "script": "experiment/encode_contacts.py",
        "source_indices": source_indices,
    }
    common_meta = build_common_meta(
        source=source,
        task=str(getattr(args, "task", "object_suite_pick_place")),
        embodiment=_infer_embodiment_from_manipulator(manipulator_name),
        dt=None if not float(args.fps) else 1.0 / float(args.fps),
        fps=float(args.fps),
        observation_point_count=manipulated_count,
        observation_frame=str(getattr(args, "object_pointcloud_frame", "world")),
        observation_dtype=str(data["manipulated_point_cloud"].dtype).replace(
            "torch.", ""
        ),
        action_space=action_space,
        extra=meta,
    )
    task_objectives = {
        str(item["task_objective"])
        for item in episode_meta
        if item.get("task_objective") is not None
    }
    if len(task_objectives) > 1:
        raise ValueError(
            f"Unified source mixes task_objective values: {sorted(task_objectives)}"
        )
    if task_objectives:
        common_meta["task_objective"] = next(iter(task_objectives))
    if str(getattr(args, "task", "")).strip() == "object_suite_pick_place":
        common_meta = with_object_only_world_observation(common_meta)
    return build_training_payload(
        obs_point_cloud=data["manipulated_point_cloud"].to(dtype=torch.float32),
        action=canonical_action,
        proprio=canonical_proprio,
        episode_ends=episode_ends_tensor,
        episode_meta=episode_meta,
        meta=common_meta,
        extra_data=data,
    )


def _build_latent_encoder(
    args: argparse.Namespace, cfg: Any
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if bool(args.skip_latent):
        return None, None
    encoder = contact_encoder._build_latent_contact_encoder(args, cfg)
    info = {
        "checkpoint_path": encoder.get("checkpoint_path"),
        "loaded": bool(encoder.get("loaded", False)),
        "config_path": str(
            contact_encoder._resolve_cross_path(args.latent_contact_policy_config)
        ),
        "pose_mode": str(cfg.probe.sixd_pose_mode),
        "latent_dim": int(encoder.get("latent_dim", 128)),
        "backend": str(encoder.get("backend", "pointnet")),
        "cotrain_checkpoint": bool(encoder.get("cotrain_checkpoint", False)),
    }
    return encoder, info


def _run_unified_pt(args: argparse.Namespace, high_level_cfg: DictConfig) -> None:
    source_pt_path = Path(args.source_pt).expanduser().resolve()
    if not source_pt_path.is_file():
        raise FileNotFoundError(f"Unified source .pt not found: {source_pt_path}")
    output_path = _resolve_output_path(args)
    _apply_high_level_preprocess_config(args, high_level_cfg)
    latent_probe_cfg = contact_encoder._load_latent_probe_config_from_args(args)
    contact_encoder._apply_contact_preprocess_config(args, latent_probe_cfg)
    args.contact_sparse_capacity_resolved = _resolve_sparse_capacity(
        latent_probe_cfg, args
    )

    episodes = _load_unified_episodes(source_pt_path, args)
    if len(episodes) == 0:
        raise RuntimeError(f"No episodes found in {source_pt_path}")
    manipulator_name = _resolve_manipulator_name(episodes[0]["source"], args)
    default_surface_points = 256
    if str(manipulator_name).startswith("mano"):
        default_surface_points = 778
    surface_template_size = _resolve_surface_template_size(
        latent_probe_cfg,
        manipulator_name,
        default=default_surface_points,
    )
    if args.surface_template_points:
        if args.surface_template_points < 2:
            raise ValueError('surface_template_points must be at least two')
        surface_template_size = args.surface_template_points
    if args.object_surface_points < 2:
        raise ValueError('object_surface_points must be at least two')
    print(
        f"Resolved unified preprocessing: manipulator={manipulator_name} "
        f"surface_template_size={surface_template_size} contact_mode={args.contact_mode_resolved} "
        f"extraction={CONTACT_EXTRACTION_VERSION}",
        flush=True,
    )
    latent_encoder, latent_encoder_info = _build_latent_encoder(args, latent_probe_cfg)

    episode_tensors_all: list[dict[str, torch.Tensor]] = []
    episode_meta_all: list[dict[str, object]] = []
    skipped_no_contact = 0
    for episode in tqdm(episodes, desc="preprocess_high_level_unified_pt"):
        try:
            tensors, meta = _prepare_unified_episode(
                episode=episode,
                args=args,
                latent_encoder=latent_encoder,
                manipulator_name=manipulator_name,
                surface_template_size=int(surface_template_size),
            )
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"Contact extraction failed for episode {episode['episode_index']}: {exc}"
            ) from exc
        episode_tensors_all.append(tensors)
        episode_meta_all.append(meta)
    if len(episode_tensors_all) == 0:
        raise RuntimeError(
            f"No valid unified episodes remained after contact extraction; skipped_no_contact={skipped_no_contact}"
        )
    dataset_xyz_translation_meta = _fit_and_apply_latent_dataset_translation(
        episode_tensors_all=episode_tensors_all,
        args=args,
        low_level_cfg=latent_probe_cfg,
        latent_encoder=latent_encoder,
    )

    high_cfg_path = contact_encoder._resolve_experiment_path(
        args.high_level_policy_config
    )
    low_cfg_path = contact_encoder._resolve_cross_path(
        args.latent_contact_policy_config
    )
    payload = _build_single_payload(
        episode_tensors_all=episode_tensors_all,
        episode_meta_all=episode_meta_all,
        args=args,
        source_format="unified_pt",
        source_pt_path=source_pt_path,
        manipulator_name=manipulator_name,
        high_level_config_path=high_cfg_path,
        low_level_config_path=low_cfg_path,
        latent_encoder_info=latent_encoder_info,
        skipped_no_contact=skipped_no_contact,
        dataset_xyz_translation_meta=dataset_xyz_translation_meta,
    )
    if args.task in {"keyboard_press", "pumpbottle_use"}:
        # Retain reset state, task metadata, and raw observations exactly as in
        # the source. Only contact labels and their encoded actions are replaced.
        original = load_high_level_payload(source_pt_path)
        episode_count = len(episode_tensors_all)
        frame_count = int(payload["episode_ends"][-1])
        original_frames = int(original["episode_ends"][-1])
        original_episodes = len(original["episode_ends"])
        for key, value in original["data"].items():
            if key in payload["data"]:
                continue
            if hasattr(value, "shape") and len(value.shape):
                if value.shape[0] == original_frames:
                    value = value[:frame_count]
                elif value.shape[0] == original_episodes:
                    value = value[:episode_count]
            payload["data"][key] = value
        payload["meta"] = {**original["meta"], **payload["meta"]}
        payload["meta"]["contact_object_surface_provider"] = "articulated_collision_surface"
        payload["meta"]["contact_object_surface_points_requested"] = int(args.object_surface_points)
        payload["meta"]["contact_object_surface_points"] = int(args.object_surface_points)
        payload["meta"]["source_configuration"] = args.source_configuration
        payload["episode_meta"] = [
            {**old, **new} for old, new in zip(original["episode_meta"], payload["episode_meta"])
        ]
    save_high_level_payload(payload, output_path)
    print(f"Wrote high-level unified payload: {output_path}")
    print(
        f"Valid episodes: {len(episode_tensors_all)}; skipped no-contact episodes: {skipped_no_contact}"
    )


@hydra.main(
    version_base="1.2", config_path="conf", config_name="train_policy/object_suite"
)
def main(cfg: DictConfig) -> None:
    cfg = runtime_config(cfg)
    print(OmegaConf.to_yaml(cfg))
    args = _namespace_from_hydra_config(cfg)
    if str(args.source_format) == "unified_pt":
        _run_unified_pt(args, cfg)
        return
    raise ValueError(f"Unsupported source format: {args.source_format}")


if __name__ == "__main__":
    from paper_config import canonicalize_cli
    canonicalize_cli()
    main()
