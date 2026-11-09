from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

EXPERIMENT_DIR = Path(__file__).resolve().parent
CROSS_DIR = EXPERIMENT_DIR.parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(CROSS_DIR) not in sys.path:
    sys.path.insert(0, str(CROSS_DIR))

from high_level_policy.train_utils import prepare_device  # noqa: E402
from experiment.high_level_policy.policy import build_policy_for_payload  # noqa: E402
from robot_model.robot_model import create_robot_model  # noqa: E402
from contact_policy_probe import load_contact_policy_probe_checkpoint  # noqa: E402


DEFAULT_HIGH_LEVEL_CHECKPOINT = (
    EXPERIMENT_DIR / "checkpoints" / "high_level_policy" / "object_suite_paper2400_main_paper_contacts_v1" / "best.pt"
)
DEFAULT_LOW_LEVEL_CONFIG = CROSS_DIR / "conf" / "kite.yaml"


from paper_config import runtime_config


def resolve_project_path(path_str: str | os.PathLike[str] | None) -> Path | None:
    if path_str is None:
        return None
    text = str(path_str).strip()
    if text == "" or text.lower() == "null":
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (CROSS_DIR / path).resolve()


def load_root_hydra_config(config_path: str | Path) -> DictConfig:
    resolved = Path(config_path).expanduser().resolve()
    config_dir = resolved.parent
    config_name = resolved.stem
    for parent in resolved.parents:
        if parent.name == "conf":
            config_dir = parent
            config_name = resolved.relative_to(parent).with_suffix("").as_posix()
            break
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.2", config_dir=str(config_dir)):
        cfg = runtime_config(compose(config_name=config_name))
    GlobalHydra.instance().clear()
    OmegaConf.resolve(cfg)
    return cfg


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def latent_uses_xyz_translation(cfg: DictConfig) -> bool:
    return bool(OmegaConf.select(cfg, "policy.latent_use_xyz_translation", default=True))


def _payload_encoder_dim(payload: dict[str, Any]) -> int:
    # Per-payload encoder dim. Defaults to 128 for legacy payloads (matches PointNet),
    # but a paired_transformer-bottleneck preprocess writes the actual d_model into meta.
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    action_space = meta.get("action_space", {}) if isinstance(meta, dict) else {}
    if isinstance(action_space, dict) and action_space.get("latent_encoder_dim") is not None:
        return int(action_space["latent_encoder_dim"])
    return int(meta.get("latent_encoder_dim", 128))


def _payload_action_space_type(payload: dict[str, Any]) -> str:
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    action_space = meta.get("action_space", {}) if isinstance(meta, dict) else {}
    if isinstance(action_space, dict):
        return str(action_space.get("type", "")).strip().lower()
    return ""


def _payload_latent_use_xyz_translation(payload: dict[str, Any], *, default: bool = True) -> bool:
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    action_space = meta.get("action_space", {}) if isinstance(meta, dict) else {}
    if isinstance(action_space, dict) and action_space.get("latent_use_xyz_translation") is not None:
        return bool(action_space["latent_use_xyz_translation"])
    if isinstance(meta, dict) and meta.get("latent_use_xyz_translation") is not None:
        return bool(meta["latent_use_xyz_translation"])
    return bool(default)


def _canonical_flat_dim(payload: dict[str, Any], key: str, *, use_xyz_translation: bool) -> int:
    tensor = torch.as_tensor(payload["data"][key])
    dim = int(np.prod(tuple(tensor.shape[1:])))
    if _payload_action_space_type(payload) == "latent_contact" and not bool(use_xyz_translation):
        encoder_dim = _payload_encoder_dim(payload)
        if dim == int(encoder_dim) + 3:
            return int(encoder_dim)
    return dim


def _payload_dataset_xyz_translation(payload: dict[str, Any]) -> np.ndarray:
    if _payload_action_space_type(payload) not in {"latent_contact", "contact_points"}:
        return np.zeros((3,), dtype=np.float32)
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    if not isinstance(meta, dict) or "dataset_xyz_translation" not in meta:
        return np.zeros((3,), dtype=np.float32)
    shift = np.asarray(meta["dataset_xyz_translation"], dtype=np.float32).reshape(-1)
    if shift.shape != (3,):
        raise ValueError(f"meta.dataset_xyz_translation must be length 3, got shape {shift.shape}")
    return shift.astype(np.float32, copy=False)


def _effective_latent_tensor(
    tensor: torch.Tensor | np.ndarray,
    *,
    use_xyz_translation: bool,
    encoder_dim: int = 128,
) -> np.ndarray:
    array = _as_numpy(tensor).astype(np.float32, copy=False)
    flat = array.reshape(array.shape[0], -1) if array.ndim > 1 else array.reshape(1, -1)
    if not bool(use_xyz_translation):
        if int(flat.shape[-1]) == int(encoder_dim) + 3:
            flat = flat[:, 3:]
        elif int(flat.shape[-1]) != int(encoder_dim):
            raise ValueError(
                f"Expected {encoder_dim}D or {encoder_dim + 3}D latent when latent_use_xyz_translation=false, "
                f"got {flat.shape[-1]}"
            )
    return flat.astype(np.float32, copy=False)


def _field_dim(payload: dict[str, Any], key: str, *, use_xyz_translation: bool) -> int:
    return _canonical_flat_dim(payload, key, use_xyz_translation=use_xyz_translation)


def _point_cloud_count_for_payload(cfg: DictConfig, payload: dict[str, Any]) -> int:
    stored_count = int(torch.as_tensor(payload["data"]["obs_point_cloud"]).shape[1])
    return int(OmegaConf.select(cfg, "dataset.observation_point_count", default=stored_count))


def _build_high_level_policy(cfg: DictConfig, payload: dict[str, Any]) -> torch.nn.Module:
    payload_type = _payload_action_space_type(payload)
    action_space = str(OmegaConf.select(cfg, "policy.action_space", default="latent")).strip().lower()
    if payload_type == "latent_contact":
        if action_space != "latent":
            raise ValueError(f"Latent high-low pipeline rollout requires policy.action_space='latent', got {action_space!r}")
    elif payload_type == "contact_points":
        if action_space != "contact_points":
            raise ValueError(
                "Direct-contact high-low pipeline rollout requires "
                f"policy.action_space='contact_points', got {action_space!r}"
            )
    else:
        raise ValueError(
            "High-low pipeline rollout requires meta.action_space.type='latent_contact' "
            f"or 'contact_points', got {payload_type!r}"
        )
    use_xyz_translation = latent_uses_xyz_translation(cfg)
    return build_policy_for_payload(
        cfg,
        payload,
        action_dim=_field_dim(payload, "action", use_xyz_translation=use_xyz_translation),
        state_dim=_field_dim(payload, "proprio", use_xyz_translation=use_xyz_translation),
        point_count=_point_cloud_count_for_payload(cfg, payload),
    )


_PAIRED_TRANSFORMER_PIPELINE_VARIANTS = {
    "paired_transformer",
    "paired_transformer_diffusion",
    "paired_transformer_flow_matching",
    "paired_transformer_mlp",
}


def _resolve_low_level_backend(low_cfg_resolved: DictConfig, low_policy: torch.nn.Module) -> str:
    variant = str(low_cfg_resolved.model.variant).strip().lower()
    if variant in {"pointnet_mlp", "pointnet_mlp_m"}:
        return "pointnet"
    if variant in _PAIRED_TRANSFORMER_PIPELINE_VARIANTS:
        if not bool(getattr(low_policy, "contact_bottleneck_enabled", False)):
            raise ValueError(
                f"Latent pipeline requires the paired_transformer low-level policy to have "
                f"contact_bottleneck.enabled=true, got variant={variant} with bottleneck disabled."
            )
        return "paired_transformer"
    raise ValueError(
        f"Latent pipeline requires pointnet_mlp / pointnet_mlp_m or paired_transformer* low-level "
        f"variant, got {variant!r}."
    )


@dataclass
class HighLowPipelineRuntime:
    device: torch.device
    high_level_cfg: DictConfig
    high_level_policy: torch.nn.Module
    high_level_checkpoint_path: str
    low_level_cfg: DictConfig
    low_level_policy: torch.nn.Module
    low_level_checkpoint_path: str
    low_level_meta: dict[str, Any]
    robot_model: Any
    local_dof: int
    action_valid_mask: torch.Tensor
    robot_index: int
    contact_feature_dim: int
    dataset_xyz_translation: np.ndarray
    high_level_action_space_type: str = "latent_contact"
    contact_action_slot_count: int = 0
    contact_action_point_dim: int = 6
    # Selected per low-level checkpoint at load time. Drives the
    # predict_low_level_q_from_contact_latent dispatch and tells callers what shape the
    # contact latent fed by the high-level policy must be.
    encoder_backend: str = "pointnet"
    # Pre-built FPS hand template — only populated for the paired_transformer backend, where
    # the low-level decoder needs world-space hand surface points/normals per call.
    hand_fps_num_points: int = 0
    # Name of the manipulator the low-level head was loaded for. For cotrain checkpoints this
    # is the user-selected head (default wujihand_right); for single-manipulator checkpoints it
    # is the robot baked into the checkpoint's dataset_meta.
    low_level_manipulator: str = ""


# Peek at the sibling dataset_meta.pt to detect whether a low-level checkpoint was produced
# by the cotrain trainer. Returns the raw meta dict only when ``mode == "cotrain"``.
def _peek_cotrain_meta(checkpoint_path: Path) -> dict[str, Any] | None:
    meta_path = Path(checkpoint_path).parent / "dataset_meta.pt"
    if not meta_path.is_file():
        return None
    raw = torch.load(str(meta_path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and str(raw.get("mode", "")).lower() == "cotrain":
        return raw
    return None


# Load a cotrain low-level checkpoint and return (head, ckpt_path, per_manip_meta, build_cfg).
# The returned ``head`` is a ``ContactProbePolicyPairedTransformer`` — the per-manipulator head
# that implements ``predict_action_from_contact_latent``. Non-selected heads share submodules
# (including contact_only_encoder), so only the selected head is returned to the caller.
def _load_cotrain_low_level_policy(
    cfg: DictConfig,
    *,
    checkpoint_path: str,
    device: torch.device,
    prefer_ema: bool,
    selected_manipulator: str,
    cotrain_meta: dict,
) -> tuple[torch.nn.Module, str, dict[str, Any], DictConfig]:
    from networks.contact_probe_policy_paired_cotrain import ContactProbePolicyPairedCoTrain
    from train_kinematic_decoder import _build_cotrain_shared_model_kwargs

    resolved_ckpt_path = str(Path(checkpoint_path).expanduser().resolve())
    ckpt = torch.load(resolved_ckpt_path, map_location=device, weights_only=False)

    build_cfg = cfg
    ckpt_cfg = ckpt.get("config")
    if isinstance(ckpt_cfg, dict):
        try:
            build_cfg = runtime_config(OmegaConf.create(ckpt_cfg))
        except Exception:
            build_cfg = cfg

    manipulator_names = [str(x) for x in cotrain_meta.get("manipulators", [])]
    per_manip = cotrain_meta.get("per_manipulator", {}) or {}
    if selected_manipulator not in manipulator_names:
        raise ValueError(
            f"--low-level-manipulator={selected_manipulator!r} is not among the cotrain "
            f"checkpoint's manipulators={manipulator_names}. Pass one of the listed names."
        )

    manipulator_specs: list[dict[str, Any]] = []
    for name in manipulator_names:
        entry = per_manip.get(name)
        if entry is None:
            raise ValueError(
                f"Cotrain dataset_meta.pt is missing per_manipulator[{name!r}]; checkpoint may be corrupted."
            )
        manipulator_specs.append(
            {
                "name": name,
                "action_dim": int(entry["action_dim"]),
                "action_normalizer_scale": torch.as_tensor(
                    entry["action_normalizer_scale"], dtype=torch.float32
                ).view(-1),
                "action_normalizer_offset": torch.as_tensor(
                    entry["action_normalizer_offset"], dtype=torch.float32
                ).view(-1),
                "residual_normalizer_scale": torch.as_tensor(
                    entry["residual_normalizer_scale"], dtype=torch.float32
                ).view(-1),
                "residual_normalizer_offset": torch.as_tensor(
                    entry["residual_normalizer_offset"], dtype=torch.float32
                ).view(-1),
                "base_translation_indices": [int(x) for x in entry["base_translation_indices"]],
                "base_rotation_indices": [int(x) for x in entry["base_rotation_indices"]],
                "pose_groups": entry.get("pose_groups"),
            }
        )

    shared_kwargs = _build_cotrain_shared_model_kwargs(build_cfg)
    per_manipulator_action_io = bool(
        OmegaConf.select(build_cfg, "co_training.per_manipulator_action_io", default=False)
    )
    model = ContactProbePolicyPairedCoTrain(
        manipulators=manipulator_specs,
        shared_kwargs=shared_kwargs,
        per_manipulator_action_io=per_manipulator_action_io,
    ).to(device)

    state_dict = ckpt["model"]
    if prefer_ema and "ema_model" in ckpt:
        state_dict = ckpt["ema_model"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    head = model.head_for(selected_manipulator)
    # Expose the selected manipulator's per-manipulator probe meta in the flat shape the
    # existing runtime already consumes (surface_num_points, hand_points_fps, robot_model_kwargs
    # etc.). Falls back to the cotrain entry for anything the single-manip path usually reads.
    selected_entry = dict(per_manip[selected_manipulator])
    return head, resolved_ckpt_path, selected_entry, build_cfg


def load_high_low_pipeline_runtime(
    *,
    high_level_payload: dict[str, Any],
    high_level_checkpoint_path: str | Path = DEFAULT_HIGH_LEVEL_CHECKPOINT,
    low_level_config_path: str | Path = DEFAULT_LOW_LEVEL_CONFIG,
    low_level_checkpoint_path: str | Path | None = None,
    low_level_manipulator: str = "wujihand_right",
    device: str | torch.device = "cuda",
    high_level_use_ema: bool = True,
    low_level_use_ema: bool = True,
    preloaded_high_level_checkpoint: dict[str, Any] | None = None,
) -> HighLowPipelineRuntime:
    device_t = prepare_device(str(device)) if not isinstance(device, torch.device) else device
    high_ckpt_path = Path(high_level_checkpoint_path).expanduser().resolve()
    high_ckpt = (
        preloaded_high_level_checkpoint
        if preloaded_high_level_checkpoint is not None
        else torch.load(str(high_ckpt_path), map_location="cpu", weights_only=False)
    )
    if "config" not in high_ckpt:
        raise KeyError(f"High-level checkpoint is missing config: {high_ckpt_path}")
    high_cfg = OmegaConf.create(high_ckpt["config"])
    if OmegaConf.select(high_cfg, "policy.latent_use_xyz_translation", default=None) is None:
        high_cfg.policy.latent_use_xyz_translation = _payload_latent_use_xyz_translation(high_level_payload)
    high_policy = _build_high_level_policy(high_cfg, high_level_payload).to(device_t)
    high_state_key = "ema_model" if bool(high_level_use_ema) and "ema_model" in high_ckpt else "model"
    high_policy.load_state_dict(high_ckpt[high_state_key], strict=True)
    if "normalizer" in high_ckpt:
        high_policy.normalizer.load_state_dict(high_ckpt["normalizer"])
    high_policy = high_policy.to(device=device_t, dtype=torch.float32)
    high_policy.eval()
    # A preloaded mapping is an evaluation-only, consumable handoff. Release
    # its CPU state tensors before the potentially large low-level checkpoint
    # is deserialized; the model and normalizer have already copied them.
    if preloaded_high_level_checkpoint is not None:
        preloaded_high_level_checkpoint.clear()
    del high_ckpt

    low_cfg_path = Path(low_level_config_path).expanduser().resolve()
    low_cfg = load_root_hydra_config(low_cfg_path)
    resolved_low_ckpt = (
        (Path(low_cfg.training.save_dir) / "last.pt").expanduser().resolve()
        if low_level_checkpoint_path is None
        else Path(low_level_checkpoint_path).expanduser().resolve()
    )

    # Detect cotrain checkpoint via dataset_meta.pt. The cotrain path returns one head
    # (ContactProbePolicyPairedTransformer) chosen by ``low_level_manipulator``; the
    # single-manipulator path reuses the existing loader and ignores that argument.
    cotrain_meta = _peek_cotrain_meta(Path(resolved_low_ckpt))
    if cotrain_meta is not None:
        low_policy, low_ckpt_resolved, low_meta, low_cfg_resolved = _load_cotrain_low_level_policy(
            low_cfg,
            checkpoint_path=str(resolved_low_ckpt),
            device=device_t,
            prefer_ema=bool(low_level_use_ema),
            selected_manipulator=str(low_level_manipulator),
            cotrain_meta=cotrain_meta,
        )
    else:
        low_policy, low_ckpt_resolved, low_meta, low_cfg_resolved = load_contact_policy_probe_checkpoint(
            low_cfg,
            device=device_t,
            ckpt_path=str(resolved_low_ckpt),
            prefer_ema=bool(low_level_use_ema),
            return_build_cfg=True,
        )
    encoder_backend = _resolve_low_level_backend(low_cfg_resolved, low_policy)
    if encoder_backend == "pointnet":
        if bool(getattr(low_policy, "use_non_contact_points", True)):
            raise ValueError("Latent pipeline requires low-level remove_non_contact_points=true")
        contact_feature_dim = int(low_cfg_resolved.model.contact_encoder.out_channels)
    else:
        # Paired-transformer variants live on the residual / bottleneck path; they don't
        # carry a non_contact stream at all and the latent dim is the bottleneck d_model.
        contact_feature_dim = int(getattr(low_policy, "d_model"))
    pose_mode = str(low_cfg_resolved.probe.sixd_pose_mode).strip().lower()
    expected_pose_mode = "zero_centered" if latent_uses_xyz_translation(high_cfg) else "movable"
    if pose_mode != expected_pose_mode:
        warnings.warn(
            f"latent_use_xyz_translation={latent_uses_xyz_translation(high_cfg)} usually pairs with "
            f"low-level sixd_pose_mode={expected_pose_mode}, but {low_cfg_path} has sixd_pose_mode={pose_mode}. "
            "Continuing because latent intents are encoded from the shared CLS token.",
            RuntimeWarning,
            stacklevel=2,
        )
    payload_type = _payload_action_space_type(high_level_payload)
    action_space_meta = high_level_payload.get("meta", {}).get("action_space", {})
    if not isinstance(action_space_meta, dict):
        action_space_meta = {}
    contact_action_slot_count = int(action_space_meta.get("slot_count", 0) or 0)
    contact_action_point_dim = int(action_space_meta.get("point_dim", 6) or 6)
    if payload_type == "latent_contact":
        expected_latent_dim = contact_feature_dim + (3 if latent_uses_xyz_translation(high_cfg) else 0)
        if int(high_policy.action_dim) != expected_latent_dim:
            raise ValueError(
                f"High-level action dim {high_policy.action_dim} is incompatible with low-level contact feature dim "
                f"{contact_feature_dim} and latent_use_xyz_translation={latent_uses_xyz_translation(high_cfg)} "
                f"(backend={encoder_backend})"
            )
    elif payload_type == "contact_points":
        if contact_action_slot_count < 1 or contact_action_point_dim < 6:
            raise ValueError(
                "contact_points payload requires action_space.slot_count >= 1 and point_dim >= 6, "
                f"got slot_count={contact_action_slot_count} point_dim={contact_action_point_dim}"
            )
        expected_contact_dim = contact_action_slot_count * contact_action_point_dim
        if int(high_policy.action_dim) != int(expected_contact_dim):
            raise ValueError(
                f"High-level action dim {high_policy.action_dim} does not match contact_points dim "
                f"{expected_contact_dim} ({contact_action_slot_count} slots x {contact_action_point_dim} channels)."
            )
    else:
        raise ValueError(f"Unsupported high-level payload action_space.type={payload_type!r}")
    dataset_xyz_translation = _payload_dataset_xyz_translation(high_level_payload)

    robot_kwargs = dict(low_meta.get("robot_model_kwargs", {}))
    robot_name = str(
        low_meta.get(
            "robot_name",
            low_level_manipulator if cotrain_meta is not None else low_cfg_resolved.probe.robot_name,
        )
    )
    robot_model = create_robot_model(
        robot_name=robot_name,
        device=device_t,
        num_points=int(low_meta.get("surface_num_points", 256)),
        **robot_kwargs,
    )
    local_dof = int(low_meta.get("action_dim", getattr(robot_model, "dof", 0)))
    action_valid_mask = torch.ones((local_dof,), dtype=torch.bool, device=device_t)
    if hasattr(robot_model, "fixed_zero_joint_indices") and len(robot_model.fixed_zero_joint_indices) > 0:
        action_valid_mask[robot_model.fixed_zero_joint_indices] = False

    # Paired-transformer rollout needs world-space hand surface points (FPS-sampled) injected
    # into every batch. Build the template once on the rollout robot model so per-step calls
    # only do the cheap FK transform, not the FPS selection.
    hand_fps_num_points = 0
    if encoder_backend == "paired_transformer":
        hand_fps_num_points = int(low_meta.get("hand_points_fps", 0))
        if hand_fps_num_points <= 0:
            hand_fps_num_points = int(getattr(low_cfg_resolved.model, "hand_points", 32))
        robot_model.build_hand_fps_template(int(hand_fps_num_points))

    return HighLowPipelineRuntime(
        device=device_t,
        high_level_cfg=high_cfg,
        high_level_policy=high_policy,
        high_level_checkpoint_path=str(high_ckpt_path),
        low_level_cfg=low_cfg_resolved,
        low_level_policy=low_policy,
        low_level_checkpoint_path=str(Path(low_ckpt_resolved).expanduser().resolve()),
        low_level_meta=dict(low_meta),
        robot_model=robot_model,
        local_dof=local_dof,
        action_valid_mask=action_valid_mask,
        robot_index=0,
        contact_feature_dim=contact_feature_dim,
        dataset_xyz_translation=dataset_xyz_translation,
        high_level_action_space_type=payload_type,
        contact_action_slot_count=contact_action_slot_count,
        contact_action_point_dim=contact_action_point_dim,
        encoder_backend=encoder_backend,
        hand_fps_num_points=int(hand_fps_num_points),
        low_level_manipulator=robot_name,
    )


def _rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


@torch.inference_mode()
def _predict_low_level_q_from_contact_latent_pointnet(
    *,
    runtime: HighLowPipelineRuntime,
    contact_feature: np.ndarray,
    q_prev: np.ndarray,
    seed: int,
) -> np.ndarray:
    low_policy = runtime.low_level_policy
    device = runtime.device
    feature_np = np.asarray(contact_feature, dtype=np.float32).reshape(-1)
    if int(feature_np.shape[0]) != int(runtime.contact_feature_dim):
        raise ValueError(f"Expected {runtime.contact_feature_dim}D contact latent, got {feature_np.shape[0]}")
    q_prev_np = np.asarray(q_prev, dtype=np.float32).reshape(-1)[: runtime.local_dof]
    feature_t = torch.from_numpy(feature_np).view(1, -1).to(device=device, dtype=torch.float32)
    q_prev_t = torch.from_numpy(q_prev_np).view(1, -1).to(device=device, dtype=torch.float32)
    batch_meta = {
        "q1_padded": q_prev_t,
        "action_valid_mask": runtime.action_valid_mask.view(1, -1),
        "robot_index": torch.tensor([int(runtime.robot_index)], dtype=torch.long, device=device),
        "local_dof": torch.tensor([int(runtime.local_dof)], dtype=torch.long, device=device),
    }
    global_cond = low_policy.obs_mlp(torch.cat([feature_t, low_policy.q_encoder(q_prev_t)], dim=-1))
    action_valid_mask = low_policy._build_action_valid_mask(batch_meta, batch_size=1)  # noqa: SLF001
    norm_scale, norm_offset = low_policy._get_action_normalizer_params(batch_meta, batch_size=1)  # noqa: SLF001

    with torch.random.fork_rng(devices=_rng_devices(device)):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        action = torch.randn((1, 1, int(low_policy.action_dim)), device=device, dtype=torch.float32)
        action = low_policy._apply_action_valid_mask(action, action_valid_mask)  # noqa: SLF001
        low_policy.noise_scheduler.set_timesteps(int(low_policy.num_inference_steps))
        for timestep in low_policy.noise_scheduler.timesteps:
            model_output = low_policy.model(sample=action, timestep=timestep, local_cond=None, global_cond=global_cond)
            action = low_policy.noise_scheduler.step(model_output, timestep, action).prev_sample
            action = low_policy._apply_action_valid_mask(action, action_valid_mask)  # noqa: SLF001
        action = low_policy._unnormalize_action(action, scale=norm_scale, offset=norm_offset)  # noqa: SLF001
        action = low_policy._apply_action_valid_mask(action[:, 0, :], action_valid_mask)  # noqa: SLF001
    q_out = action[0, : runtime.local_dof].detach().cpu().numpy().astype(np.float32, copy=False)
    q_out_t = torch.from_numpy(q_out).to(runtime.robot_model.device, dtype=torch.float32)
    q_out_t = runtime.robot_model._apply_fixed_zero_joint_constraints(q_out_t)  # noqa: SLF001
    return q_out_t.detach().cpu().numpy().astype(np.float32, copy=False)


@torch.inference_mode()
def _predict_low_level_q_from_contact_latent_paired(
    *,
    runtime: HighLowPipelineRuntime,
    contact_feature: np.ndarray,
    q_prev: np.ndarray,
    seed: int,
) -> np.ndarray:
    # Paired-transformer counterpart of the PointNet rollout. The high-level policy emits a
    # contact-bottleneck CLS latent Z; we run hand FK on q_prev to materialize the world-
    # space hand surface tokens, then call the policy's predict_action_from_contact_latent
    # (skipping the contact encoder, since Z is already provided).
    low_policy = runtime.low_level_policy
    device = runtime.device
    feature_np = np.asarray(contact_feature, dtype=np.float32).reshape(-1)
    if int(feature_np.shape[0]) != int(runtime.contact_feature_dim):
        raise ValueError(f"Expected {runtime.contact_feature_dim}D contact latent, got {feature_np.shape[0]}")
    q_prev_np = np.asarray(q_prev, dtype=np.float32).reshape(-1)[: runtime.local_dof]
    z_t = torch.from_numpy(feature_np).view(1, -1).to(device=device, dtype=torch.float32)
    q_prev_t = torch.from_numpy(q_prev_np).view(1, -1).to(device=device, dtype=torch.float32)

    # FK the hand surface for the rollout's q_prev so the bottleneck scene encoder sees the
    # same hand-token format the trainer used (xyz + normals, FPS-sampled).
    hand_points, hand_normals = runtime.robot_model.transform_hand_fps_template_batch(q_prev_t)
    hand_points = hand_points.to(device=device, dtype=torch.float32)
    hand_normals = hand_normals.to(device=device, dtype=torch.float32)

    batch_meta: dict[str, torch.Tensor] = {
        "q1_padded": q_prev_t,
        "hand_points_world": hand_points,
        "hand_normals_world": hand_normals,
        "action_valid_mask": runtime.action_valid_mask.view(1, -1),
        "robot_index": torch.tensor([int(runtime.robot_index)], dtype=torch.long, device=device),
        "local_dof": torch.tensor([int(runtime.local_dof)], dtype=torch.long, device=device),
    }

    with torch.random.fork_rng(devices=_rng_devices(device)):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        out = low_policy.predict_action_from_contact_latent(z_t, batch_meta)
    q_out = out["action_pred"][0, : runtime.local_dof].detach().cpu().numpy().astype(np.float32, copy=False)
    q_out_t = torch.from_numpy(q_out).to(runtime.robot_model.device, dtype=torch.float32)
    q_out_t = runtime.robot_model._apply_fixed_zero_joint_constraints(q_out_t)  # noqa: SLF001
    return q_out_t.detach().cpu().numpy().astype(np.float32, copy=False)


def predict_low_level_q_from_contact_latent(
    *,
    runtime: HighLowPipelineRuntime,
    contact_feature: np.ndarray,
    q_prev: np.ndarray,
    seed: int,
) -> np.ndarray:
    # Backend dispatcher. PointNet path stays bit-identical to the original implementation;
    # paired_transformer path skips the encoder and goes through the bottleneck scene encoder
    # → action decoder pipeline, with hand FK done per call from q_prev.
    if runtime.encoder_backend == "pointnet":
        return _predict_low_level_q_from_contact_latent_pointnet(
            runtime=runtime, contact_feature=contact_feature, q_prev=q_prev, seed=int(seed),
        )
    if runtime.encoder_backend == "paired_transformer":
        return _predict_low_level_q_from_contact_latent_paired(
            runtime=runtime, contact_feature=contact_feature, q_prev=q_prev, seed=int(seed),
        )
    raise ValueError(f"Unknown encoder_backend={runtime.encoder_backend!r}")


def _low_level_translation_groups(runtime: HighLowPipelineRuntime) -> list[tuple[int, int, int]]:
    translation_groups: list[tuple[int, int, int]] = []
    for group in getattr(runtime.robot_model, "pose_groups", []) or []:
        if not isinstance(group, dict):
            continue
        raw = group.get("translation_indices")
        if raw is None:
            continue
        indices = tuple(int(x) for x in raw)
        if len(indices) >= 3 and all(0 <= idx < int(runtime.local_dof) for idx in indices[:3]):
            translation_groups.append(tuple(indices[:3]))
    if not translation_groups:
        raw = tuple(int(x) for x in getattr(runtime.robot_model, "base_translation_indices", []) or [])
        if len(raw) >= 3 and all(0 <= idx < int(runtime.local_dof) for idx in raw[:3]):
            translation_groups.append(tuple(raw[:3]))
    if not translation_groups and int(runtime.local_dof) >= 3:
        translation_groups.append((0, 1, 2))
    return list(dict.fromkeys(translation_groups))


def _shift_low_level_q_for_dataset(runtime: HighLowPipelineRuntime, q_value: np.ndarray, *, sign: float) -> np.ndarray:
    out = np.asarray(q_value, dtype=np.float32).reshape(-1).copy()
    dataset_shift = np.asarray(runtime.dataset_xyz_translation, dtype=np.float32).reshape(3)
    for indices in _low_level_translation_groups(runtime):
        out[list(indices)] += float(sign) * dataset_shift
    return out


@torch.inference_mode()
def _predict_low_level_q_from_contact_cloud(
    *,
    runtime: HighLowPipelineRuntime,
    contact_cloud_shifted: np.ndarray,
    contact_valid_mask: np.ndarray,
    q_prev_shifted: np.ndarray,
    seed: int,
) -> np.ndarray:
    device = runtime.device
    contact_np = np.asarray(contact_cloud_shifted, dtype=np.float32)
    mask_np = np.asarray(contact_valid_mask, dtype=np.bool_)
    q_prev_np = np.asarray(q_prev_shifted, dtype=np.float32)
    if contact_np.ndim != 3 or int(contact_np.shape[-1]) < 6:
        raise ValueError(f"Expected contact cloud shape (B,N,>=6), got {contact_np.shape}")
    if mask_np.shape != contact_np.shape[:2]:
        raise ValueError(f"contact_valid_mask shape {mask_np.shape} does not match {contact_np.shape[:2]}")
    if q_prev_np.ndim != 2 or int(q_prev_np.shape[1]) < int(runtime.local_dof):
        raise ValueError(f"Expected q_prev shape (B,{runtime.local_dof}), got {q_prev_np.shape}")

    q_prev_t = torch.from_numpy(q_prev_np[:, : runtime.local_dof]).to(device=device, dtype=torch.float32)
    batch: dict[str, torch.Tensor] = {
        "contact_cloud": torch.from_numpy(contact_np[..., :6]).to(device=device, dtype=torch.float32),
        "contact_valid_mask": torch.from_numpy(mask_np).to(device=device, dtype=torch.bool),
        "q1_padded": q_prev_t,
        "action_valid_mask": runtime.action_valid_mask.view(1, -1).expand(q_prev_t.shape[0], -1),
        "robot_index": torch.full((q_prev_t.shape[0],), int(runtime.robot_index), dtype=torch.long, device=device),
        "local_dof": torch.full((q_prev_t.shape[0],), int(runtime.local_dof), dtype=torch.long, device=device),
        "non_contact_cloud": torch.zeros((q_prev_t.shape[0], 0, 3), dtype=torch.float32, device=device),
        "non_contact_valid_mask": torch.zeros((q_prev_t.shape[0], 0), dtype=torch.bool, device=device),
        "non_contact_blocker_mask": torch.zeros((q_prev_t.shape[0], 0), dtype=torch.bool, device=device),
    }
    if runtime.encoder_backend == "paired_transformer":
        hand_points, hand_normals = runtime.robot_model.transform_hand_fps_template_batch(q_prev_t)
        batch["hand_points_world"] = hand_points.to(device=device, dtype=torch.float32)
        batch["hand_normals_world"] = hand_normals.to(device=device, dtype=torch.float32)

    with torch.random.fork_rng(devices=_rng_devices(device)):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        out = runtime.low_level_policy.predict_action(batch)
    action = out["action_pred" if "action_pred" in out else "action"]
    q_pred = action[:, : runtime.local_dof].detach().cpu().numpy().astype(np.float32, copy=False)
    q_pred_t = torch.from_numpy(q_pred).to(runtime.robot_model.device, dtype=torch.float32)
    return runtime.robot_model._apply_fixed_zero_joint_constraints(q_pred_t).detach().cpu().numpy().astype(  # noqa: SLF001
        np.float32,
        copy=False,
    )


@torch.inference_mode()
def predict_low_level_q_from_contact_action(
    *,
    runtime: HighLowPipelineRuntime,
    contact_action: np.ndarray,
    q_prev: np.ndarray,
    seed: int,
) -> np.ndarray:
    flat = np.asarray(contact_action, dtype=np.float32).reshape(-1)
    slot_count = int(runtime.contact_action_slot_count)
    point_dim = int(runtime.contact_action_point_dim)
    if slot_count <= 0:
        if flat.shape[0] % 6 != 0:
            raise ValueError(f"Cannot infer contact slot count from flat dim {flat.shape[0]}")
        slot_count = int(flat.shape[0] // 6)
        point_dim = 6
    expected_dim = slot_count * point_dim
    if int(flat.shape[0]) != int(expected_dim):
        raise ValueError(
            f"Expected {expected_dim}D contact action ({slot_count} slots x {point_dim} channels), "
            f"got {flat.shape[0]}"
        )
    if point_dim < 6:
        raise ValueError(f"contact action point_dim must be at least 6, got {point_dim}")
    contact_cloud = flat.reshape(slot_count, point_dim)
    contact_mask = np.ones((slot_count,), dtype=np.bool_)
    q_prev_np = np.asarray(q_prev, dtype=np.float32).reshape(-1)[: runtime.local_dof].copy()
    q_shifted = _shift_low_level_q_for_dataset(runtime, q_prev_np, sign=1.0)
    q_pred_shifted = _predict_low_level_q_from_contact_cloud(
        runtime=runtime,
        contact_cloud_shifted=contact_cloud.reshape(1, *contact_cloud.shape),
        contact_valid_mask=contact_mask.reshape(1, *contact_mask.shape),
        q_prev_shifted=q_shifted.reshape(1, -1),
        seed=int(seed),
    )
    q_pred = _shift_low_level_q_for_dataset(runtime, q_pred_shifted[0], sign=-1.0)
    q_pred_t = torch.from_numpy(q_pred).to(runtime.robot_model.device, dtype=torch.float32)
    q_pred = runtime.robot_model._apply_fixed_zero_joint_constraints(q_pred_t).detach().cpu().numpy()  # noqa: SLF001
    return q_pred.astype(np.float32, copy=False)


@torch.inference_mode()
def predict_low_level_q_from_latent_action(
    *,
    runtime: HighLowPipelineRuntime,
    latent_action: np.ndarray,
    q_prev: np.ndarray,
    seed: int,
) -> np.ndarray:
    use_xyz_translation = latent_uses_xyz_translation(runtime.high_level_cfg)
    latent_np = _effective_latent_tensor(
        np.asarray(latent_action, dtype=np.float32).reshape(1, -1),
        use_xyz_translation=use_xyz_translation,
        encoder_dim=int(runtime.contact_feature_dim),
    )[0]
    expected_dim = int(runtime.high_level_policy.action_dim)
    if int(latent_np.shape[0]) != expected_dim:
        raise ValueError(f"Expected {expected_dim}D latent action, got {latent_np.shape[0]}")

    q_prev_np = np.asarray(q_prev, dtype=np.float32).reshape(-1)[: runtime.local_dof].copy()
    translation_groups: list[tuple[int, int, int]] = []
    for group in getattr(runtime.robot_model, "pose_groups", []) or []:
        if not isinstance(group, dict):
            continue
        raw = group.get("translation_indices")
        if raw is None:
            continue
        indices = tuple(int(x) for x in raw)
        if len(indices) >= 3 and all(0 <= idx < int(runtime.local_dof) for idx in indices[:3]):
            translation_groups.append(tuple(indices[:3]))
    if not translation_groups:
        raw = tuple(int(x) for x in getattr(runtime.robot_model, "base_translation_indices", []) or [])
        if len(raw) >= 3 and all(0 <= idx < int(runtime.local_dof) for idx in raw[:3]):
            translation_groups.append(tuple(raw[:3]))
    if not translation_groups and int(runtime.local_dof) >= 3:
        translation_groups.append((0, 1, 2))
    translation_groups = list(dict.fromkeys(translation_groups))
    dataset_shift = np.asarray(runtime.dataset_xyz_translation, dtype=np.float32).reshape(3)

    def shift_q_for_low_level_frame(q_value: np.ndarray, *, sign: float) -> np.ndarray:
        out = np.asarray(q_value, dtype=np.float32).reshape(-1).copy()
        for indices in translation_groups:
            out[list(indices)] += float(sign) * dataset_shift
        return out

    if use_xyz_translation:
        # Latent xyz translations are stored in the shifted contact-encoder frame.
        # Convert q_prev into that frame before decoding, then convert q_pred back.
        shift = latent_np[:3]
        contact_feature = latent_np[3:]
        q_shifted = shift_q_for_low_level_frame(q_prev_np, sign=1.0)
        q_centered = q_shifted.copy()
        for indices in translation_groups:
            q_centered[list(indices)] -= shift[:3]
        q_pred = predict_low_level_q_from_contact_latent(
            runtime=runtime,
            contact_feature=contact_feature,
            q_prev=q_centered,
            seed=int(seed),
        )
        for indices in translation_groups:
            q_pred[list(indices)] += shift[:3]
        q_pred = shift_q_for_low_level_frame(q_pred, sign=-1.0)
        q_pred_t = torch.from_numpy(q_pred).to(runtime.robot_model.device, dtype=torch.float32)
        q_pred = runtime.robot_model._apply_fixed_zero_joint_constraints(q_pred_t).detach().cpu().numpy()
    else:
        # The high-level action is only the contact latent, but that latent was encoded from
        # contact clouds translated into the low-level policy's safe training region. Decode in
        # that same shifted q frame, then map the predicted q target back to world coordinates.
        q_shifted = shift_q_for_low_level_frame(q_prev_np, sign=1.0)
        q_pred = predict_low_level_q_from_contact_latent(
            runtime=runtime,
            contact_feature=latent_np,
            q_prev=q_shifted,
            seed=int(seed),
        )
        q_pred = shift_q_for_low_level_frame(q_pred, sign=-1.0)
        q_pred_t = torch.from_numpy(q_pred).to(runtime.robot_model.device, dtype=torch.float32)
        q_pred = runtime.robot_model._apply_fixed_zero_joint_constraints(q_pred_t).detach().cpu().numpy()
    return q_pred.astype(np.float32, copy=False)

