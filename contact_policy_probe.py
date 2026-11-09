from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from data_process.contact_policy_probe_dataset import ContactPolicyProbeDataset  # noqa: E402
from networks.contact_probe_policy_paired import ContactProbePolicyPairedTransformer  # noqa: E402
from networks.residual_action_utils import (  # noqa: E402
    absolute_to_residual,
    build_closed_form_residual_normalizer,
    build_empirical_residual_normalizer,
    build_residual_slot_indices,
)
from robot_model.robot_model import create_robot_model  # noqa: E402
from contact_policy_common import (  # noqa: E402
    _autocast_context,
    _build_action_normalizer_meta,
    _build_noise_scheduler,
    _cuda_sync_if_needed,
    _get_amp_dtype,
    _move_batch_to_device,
    _resolve_optional_project_path,
    _resolve_project_path,
    build_contact_policy_common_dataset_kwargs,
)


from paper_config import runtime_config

def _resolve_probe_model_variant(cfg: DictConfig) -> str:
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is None:
        return "paired_transformer_mlp"
    variant = getattr(model_cfg, "variant", None)
    if variant is None:
        return "paired_transformer_mlp"
    return str(variant).strip().lower()


# The paired_transformer architecture comes in three time-conditioned flavors; the plain
# "paired_transformer" alias defaults to flow-matching so configs stay short.
_PAIRED_TRANSFORMER_VARIANTS = {"paired_transformer_mlp"}


def _is_paired_transformer_variant(cfg: DictConfig) -> bool:
    return _resolve_probe_model_variant(cfg) in _PAIRED_TRANSFORMER_VARIANTS


# Build the probe model on top of the same diffusion/normalization utilities as the main contact policy.
def _build_probe_policy(
    cfg: DictConfig,
    *,
    action_dim: int,
    action_normalizer_meta: Dict[str, Any],
    base_translation_indices: Any | None = None,
    base_rotation_indices: Any | None = None,
    pose_groups: Any | None = None,
    residual_normalizer_scale: torch.Tensor | None = None,
    residual_normalizer_offset: torch.Tensor | None = None,
) -> nn.Module:
    variant = _resolve_probe_model_variant(cfg)
    common_kwargs = {
        "action_dim": int(action_dim),
        "robot_names": [str(x) for x in action_normalizer_meta.get("robot_names", [])],
        "action_normalizer_scale": action_normalizer_meta.get("scale"),
        "action_normalizer_offset": action_normalizer_meta.get("offset"),
    }
    if variant in _PAIRED_TRANSFORMER_VARIANTS:
        if base_translation_indices is None or base_rotation_indices is None:
            raise ValueError(
                "paired_transformer requires base_translation_indices and base_rotation_indices. "
                "Pass them from the training dataset's robot model (or the saved dataset_meta.pt)."
            )
        paired_mode = (
            "diffusion"
            if variant == "paired_transformer"
            else variant[len("paired_transformer_") :]
        )
        cb_cfg = OmegaConf.select(cfg.model, "contact_bottleneck", default=None)
        cb_kwargs: Dict[str, Any] = {}
        if cb_cfg is not None:
            slot_count = OmegaConf.select(cb_cfg, "slot_count", default=None)
            slot_dim = OmegaConf.select(cb_cfg, "slot_dim", default=None)
            cb_kwargs = {
                "contact_bottleneck_enabled": bool(getattr(cb_cfg, "enabled", False)),
                "contact_bottleneck_num_encoder_layers": int(
                    getattr(cb_cfg, "num_encoder_layers", 2)
                ),
                "contact_bottleneck_unpack_tokens": int(getattr(cb_cfg, "unpack_tokens", 1)),
                "contact_bottleneck_unpack_mlp_ratio": float(
                    getattr(cb_cfg, "unpack_mlp_ratio", 2.0)
                ),
                "contact_bottleneck_mode": str(getattr(cb_cfg, "mode", "cls_unpack")),
                "contact_bottleneck_slot_count": (
                    int(slot_count) if slot_count is not None else None
                ),
                "contact_bottleneck_slot_dim": (
                    int(slot_dim) if slot_dim is not None else None
                ),
            }
        nc_cfg = OmegaConf.select(cfg.model, "non_contact_context", default=None)
        nc_kwargs: Dict[str, Any] = {}
        if nc_cfg is not None:
            nc_kwargs = {
                "non_contact_context_enabled": bool(getattr(nc_cfg, "enabled", False)),
                "non_contact_context_tokens": int(getattr(nc_cfg, "num_tokens", 4)),
                "non_contact_context_mlp_ratio": float(getattr(nc_cfg, "mlp_ratio", 1.0)),
            }
        return ContactProbePolicyPairedTransformer(
            **common_kwargs,
            variant=paired_mode,
            base_translation_indices=[int(x) for x in base_translation_indices],
            base_rotation_indices=[int(x) for x in base_rotation_indices],
            pose_groups=pose_groups,
            residual_normalizer_scale=residual_normalizer_scale,
            residual_normalizer_offset=residual_normalizer_offset,
            d_model=int(cfg.model.d_model),
            num_heads=int(cfg.model.num_heads),
            num_scene_layers=int(cfg.model.num_scene_layers),
            num_action_layers=int(cfg.model.num_action_layers),
            mlp_ratio=float(cfg.model.mlp_ratio),
            hand_points=int(cfg.model.hand_points),
            use_contact_tokens_in_decoder=bool(cfg.model.use_contact_tokens_in_decoder),
            step_embed_dim=int(cfg.model.step_embed_dim),
            num_inference_steps=int(cfg.model.num_inference_steps),
            flow_time_beta_alpha=float(getattr(cfg.model, "flow_time_beta_alpha", 1.5)),
            flow_time_beta_beta=float(getattr(cfg.model, "flow_time_beta_beta", 1.0)),
            action_tokenization=str(getattr(cfg.model, "action_tokenization", "grouped")),
            use_qref_action_tokens=bool(getattr(cfg.model, "use_qref_action_tokens", False)),
            use_qref_memory_tokens=bool(getattr(cfg.model, "use_qref_memory_tokens", False)),
            action_token_layer_norm=bool(getattr(cfg.model, "action_token_layer_norm", False)),
            noise_scheduler=(
                _build_noise_scheduler(cfg.model.noise_scheduler)
                if paired_mode == "diffusion" and getattr(cfg.model, "noise_scheduler", None) is not None
                else None
            ),
            **cb_kwargs,
            **nc_kwargs,
        )
    raise ValueError(f"Unsupported cfg.model.variant={variant}")


# Reload a trained probe model plus the saved dataset metadata used by visualization.
def load_contact_policy_probe_checkpoint(
    cfg: DictConfig,
    *,
    device: torch.device,
    ckpt_path: str | None = None,
    prefer_ema: bool = True,
    return_build_cfg: bool = False,
) -> tuple[nn.Module, str, Dict[str, Any]] | tuple[nn.Module, str, Dict[str, Any], DictConfig]:
    resolved_ckpt_path = (
        _resolve_project_path(os.path.join(str(cfg.training.save_dir), "last.pt"))
        if ckpt_path is None
        else _resolve_project_path(str(ckpt_path))
    )
    ckpt = torch.load(resolved_ckpt_path, map_location=device, weights_only=False)
    build_cfg = cfg
    ckpt_cfg = ckpt.get("config")
    if isinstance(ckpt_cfg, dict):
        try:
            build_cfg = runtime_config(OmegaConf.create(ckpt_cfg))
        except Exception:
            build_cfg = cfg

    meta_path = os.path.join(os.path.dirname(resolved_ckpt_path), "dataset_meta.pt")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Probe dataset metadata not found: {meta_path}")
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    if not isinstance(meta, dict):
        raise ValueError(f"Invalid probe dataset metadata: {meta_path}")
    if "action_dim" not in meta or "action_normalizer_scale" not in meta or "action_normalizer_offset" not in meta:
        raise ValueError(f"Probe dataset metadata at {meta_path} is missing action normalization tensors")

    action_normalizer_meta = {
        "robot_names": [str(meta.get("robot_name", str(build_cfg.probe.robot_name)))],
        "scale": torch.as_tensor(meta["action_normalizer_scale"], dtype=torch.float32),
        "offset": torch.as_tensor(meta["action_normalizer_offset"], dtype=torch.float32),
    }
    build_variant = _resolve_probe_model_variant(build_cfg)
    # Paired-transformer checkpoints require the robot's base_translation/rotation indices
    # so the residual layout matches what was trained. We saved them into dataset_meta.pt.
    paired_kwargs: Dict[str, Any] = {}
    if build_variant in _PAIRED_TRANSFORMER_VARIANTS:
        if "base_translation_indices" not in meta or "base_rotation_indices" not in meta:
            raise ValueError(
                f"Probe dataset metadata at {meta_path} is missing base_translation_indices/base_rotation_indices. "
                "These are required to rebuild a paired_transformer checkpoint."
            )
        paired_kwargs["base_translation_indices"] = list(meta["base_translation_indices"])
        paired_kwargs["base_rotation_indices"] = list(meta["base_rotation_indices"])
        if meta.get("pose_groups") is not None:
            paired_kwargs["pose_groups"] = meta["pose_groups"]
        # Checkpoints trained before the tuned normalizer existed won't have these; the policy
        # constructor falls back to the legacy absolute-range derivation when both are None.
        if "residual_normalizer_scale" in meta and "residual_normalizer_offset" in meta:
            residual_scale = meta["residual_normalizer_scale"]
            residual_offset = meta["residual_normalizer_offset"]
            if residual_scale is not None and residual_offset is not None:
                paired_kwargs["residual_normalizer_scale"] = torch.as_tensor(
                    residual_scale, dtype=torch.float32
                )
                paired_kwargs["residual_normalizer_offset"] = torch.as_tensor(
                    residual_offset, dtype=torch.float32
                )
    try:
        model = _build_probe_policy(
            build_cfg,
            action_dim=int(meta["action_dim"]),
            action_normalizer_meta=action_normalizer_meta,
            **paired_kwargs,
        ).to(device)
    except Exception as exc:
        raise RuntimeError(
            f"Probe checkpoint config does not define a valid probe policy architecture for model.variant={build_variant}."
        ) from exc
    state_dict = ckpt["model"]
    if prefer_ema and "ema_model" in ckpt:
        state_dict = ckpt["ema_model"]
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Probe checkpoint architecture does not match model.variant={build_variant}. "
            "Use a checkpoint trained with the same probe backbone."
        ) from exc
    model.eval()
    if bool(return_build_cfg):
        return model, resolved_ckpt_path, meta, build_cfg
    return model, resolved_ckpt_path, meta


# Shared contact-policy dataset kwargs are reused, then specialized to one robot for the probe.
def _build_probe_dataset_common_kwargs(cfg: DictConfig) -> Dict[str, Any]:
    common_kwargs = build_contact_policy_common_dataset_kwargs(
        cfg,
        robot_names=[str(cfg.probe.robot_name)],
        global_robot_names=[str(cfg.probe.robot_name)],
    )
    common_kwargs.pop("robot_names", None)
    common_kwargs.pop("global_robot_names", None)
    robot_model_kwargs = OmegaConf.select(cfg, "probe.robot_model_kwargs", default=None)
    if robot_model_kwargs is not None:
        common_kwargs["robot_model_kwargs_by_name"] = {
            str(cfg.probe.robot_name): dict(OmegaConf.to_container(robot_model_kwargs, resolve=True) or {})
        }
    return common_kwargs


# Buffer paths default to the probe save dir, but can still be overridden from config.
def _resolve_probe_buffer_path(path_val: Any, *, save_dir: str, default_name: str) -> str:
    resolved = _resolve_portable_project_path(path_val)
    if resolved is not None:
        return resolved
    return os.path.join(save_dir, default_name)


def _portable_project_path(path_str: str) -> str:
    abs_path = os.path.abspath(str(path_str))
    root_abs = os.path.abspath(ROOT_DIR)
    try:
        rel_path = os.path.relpath(abs_path, root_abs)
    except ValueError:
        return abs_path
    if rel_path == '.' or rel_path == '..' or rel_path.startswith(f'..{os.sep}'):
        return abs_path
    return rel_path


def _resolve_portable_project_path(path_val: Any) -> str | None:
    resolved = _resolve_optional_project_path(path_val)
    if resolved is None:
        return None
    if os.path.exists(resolved):
        return resolved

    raw_path = str(path_val).strip().replace('\\', '/')
    marker = 'crossembodimentaction/'
    marker_idx = raw_path.rfind(marker)
    if marker_idx < 0:
        return resolved
    suffix = raw_path[marker_idx + len(marker) :]
    candidate = os.path.join(ROOT_DIR, suffix)
    if os.path.exists(candidate):
        return candidate
    return candidate


def _optional_int_pair(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    return int(value[0]), int(value[1])


def _meta_sequence_to_list(value: Any) -> list[Any]:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return list(value)


def _float_lists_match(actual: list[Any], expected: list[Any]) -> bool:
    if len(actual) != len(expected):
        return False
    return bool(
        torch.allclose(
            torch.as_tensor(actual, dtype=torch.float64),
            torch.as_tensor(expected, dtype=torch.float64),
            rtol=1.0e-6,
            atol=1.0e-8,
        )
    )


def _probe_buffer_matches_cfg(
    meta: Dict[str, Any],
    cfg: DictConfig,
    *,
    split_key: str,
) -> tuple[bool, str | None]:
    required_meta = {
        "candidate_contact_points": int(cfg.dataset.max_contact_points),
        "contact_cluster_mode": "connected_mask",
        "threshold": float(cfg.dataset.threshold),
        "align_exp_scale": float(cfg.dataset.align_exp_scale),
        "sigmoid_scale": float(cfg.dataset.sigmoid_scale),
        "anchor_temperature": float(cfg.dataset.anchor_temperature),
        "patch_anchor_shift_min": float(cfg.dataset.patch_anchor_shift_min),
        "patch_anchor_shift_max": float(cfg.dataset.patch_anchor_shift_max),
        "patch_extent_min": float(cfg.dataset.patch_extent_min),
        "patch_extent_max": float(cfg.dataset.patch_extent_max),
        "patch_extent_power": float(cfg.dataset.patch_extent_power),
        "patch_shift_power": float(cfg.dataset.patch_shift_power),
        "patch_normal_jitter_max_deg": float(cfg.dataset.patch_normal_jitter_max_deg),
        "patch_points_per_anchor_min": float(cfg.dataset.patch_points_per_anchor_min),
        "patch_points_per_anchor_max": float(cfg.dataset.patch_points_per_anchor_max),
        "patch_penetration_clearance": float(cfg.dataset.patch_penetration_clearance),
    }
    for key, expected in required_meta.items():
        if key not in meta:
            return False, f"buffer meta is missing {key}"
        if key == "contact_cluster_mode":
            if str(meta[key]) != str(expected):
                return False, f"{key} mismatch ({meta[key]} vs {expected})"
        else:
            if not math.isclose(float(meta[key]), float(expected), rel_tol=1.0e-6, abs_tol=1.0e-8):
                return False, f"{key} mismatch ({meta[key]} vs {expected})"

    expected_samples = int(cfg.probe.train_samples if split_key == "train" else cfg.probe.test_samples)
    base_seed = int(cfg.seed)
    expected_build_seed = base_seed if split_key == "train" else base_seed + 10_007
    scalar_checks = {
        "contact_mode": str(cfg.probe.contact_mode),
        "sixd_pose_mode": str(cfg.probe.sixd_pose_mode),
        "robot_name": str(cfg.probe.robot_name),
        "component_sampling_mode": str(cfg.dataset.component_sampling_mode),
        "anchor_sampling_mode": str(cfg.dataset.anchor_sampling_mode),
        "samples": expected_samples,
        "build_seed": expected_build_seed,
        "q1_translation_noise_std": float(cfg.probe.q1_translation_noise_std),
        "q1_rotation_noise_std_deg": float(cfg.probe.q1_rotation_noise_std_deg),
        "q1_joint_noise_scale": float(cfg.probe.q1_joint_noise_scale),
        "q1_zero_probability": float(getattr(cfg.probe, "q1_zero_probability", 0.10)),
        "random_q1": bool(getattr(cfg.probe, "random_q1", False)),
    }
    for key, expected in scalar_checks.items():
        if key not in meta:
            return False, f"buffer meta is missing {key}"
        if str(meta[key]) != str(expected):
            return False, f"{key} mismatch ({meta[key]} vs {expected})"

    expected_sparse_sampling = str(getattr(cfg.probe, "sparse_sampling_mode", "point_uniform"))
    actual_sparse_sampling = str(meta.get("sparse_sampling_mode", "point_uniform"))
    if actual_sparse_sampling != expected_sparse_sampling:
        return False, f"sparse_sampling_mode mismatch ({actual_sparse_sampling} vs {expected_sparse_sampling})"
    expected_q1_sampling = str(getattr(cfg.probe, "q1_sampling_mode", "gaussian"))
    actual_q1_sampling = str(meta.get("q1_sampling_mode", "gaussian"))
    if actual_q1_sampling != expected_q1_sampling:
        return False, f"q1_sampling_mode mismatch ({actual_q1_sampling} vs {expected_q1_sampling})"
    expected_composite_synthesis = str(getattr(cfg.probe, "composite_synthesis_mode", "independent"))
    actual_composite_synthesis = str(meta.get("composite_synthesis_mode", "independent"))
    if actual_composite_synthesis != expected_composite_synthesis:
        return False, f"composite_synthesis_mode mismatch ({actual_composite_synthesis} vs {expected_composite_synthesis})"
    expected_nearby = float(getattr(cfg.probe, "composite_nearby_max_distance", 0.12))
    actual_nearby = float(meta.get("composite_nearby_max_distance", 0.12))
    if not math.isclose(actual_nearby, expected_nearby, rel_tol=1.0e-6, abs_tol=1.0e-8):
        return False, f"composite_nearby_max_distance mismatch ({actual_nearby} vs {expected_nearby})"
    robot_model_kwargs_cfg = OmegaConf.select(cfg, "probe.robot_model_kwargs", default=None)
    expected_robot_model_kwargs = (
        {}
        if robot_model_kwargs_cfg is None
        else dict(OmegaConf.to_container(robot_model_kwargs_cfg, resolve=True) or {})
    )
    actual_robot_model_kwargs = dict(meta.get("robot_model_kwargs", {}))
    if actual_robot_model_kwargs != expected_robot_model_kwargs:
        return False, f"robot_model_kwargs mismatch ({actual_robot_model_kwargs} vs {expected_robot_model_kwargs})"
    expected_collision_enabled = bool(
        getattr(cfg.probe, "composite_collision_filter_enabled", False)
    )
    actual_collision_enabled = bool(meta.get("composite_collision_filter_enabled", False))
    if actual_collision_enabled != expected_collision_enabled:
        return False, (
            "composite_collision_filter_enabled mismatch "
            f"({actual_collision_enabled} vs {expected_collision_enabled})"
        )
    if actual_collision_enabled or expected_collision_enabled:
        expected_collision_margin = float(getattr(cfg.probe, "composite_collision_filter_margin", 0.0))
        actual_collision_margin = float(meta.get("composite_collision_filter_margin", 0.0))
        if not math.isclose(
            actual_collision_margin,
            expected_collision_margin,
            rel_tol=1.0e-6,
            abs_tol=1.0e-8,
        ):
            return False, (
                "composite_collision_filter_margin mismatch "
                f"({actual_collision_margin} vs {expected_collision_margin})"
            )
        expected_collision_attempts = int(getattr(cfg.probe, "composite_collision_filter_max_attempts", 64))
        actual_collision_attempts = int(meta.get("composite_collision_filter_max_attempts", 64))
        if actual_collision_attempts != expected_collision_attempts:
            return False, (
                "composite_collision_filter_max_attempts mismatch "
                f"({actual_collision_attempts} vs {expected_collision_attempts})"
            )
    if expected_q1_sampling == "contact_error_mixture":
        q1_scalar_checks = {
            "q1_contact_error_zero_prob": float(getattr(cfg.probe, "q1_contact_error_zero_prob", 0.08)),
            "q1_contact_error_inactive_scale": float(getattr(cfg.probe, "q1_contact_error_inactive_scale", 0.05)),
            "q1_contact_error_refine_steps": int(getattr(cfg.probe, "q1_contact_error_refine_steps", 2)),
        }
        for key, expected in q1_scalar_checks.items():
            if key not in meta:
                return False, f"buffer meta is missing {key}"
            if str(meta[key]) != str(expected):
                return False, f"{key} mismatch ({meta[key]} vs {expected})"

    list_checks: list[tuple[str, Any]] = [
        ("fixed_contact_links", list(cfg.probe.fixed_contact_links)),
    ]
    component_range = _optional_int_pair(cfg.probe.component_range)
    if component_range is not None:
        list_checks.append(("probe_component_range", list(component_range)))
    contact_count_range = _optional_int_pair(cfg.probe.contact_count_range)
    if contact_count_range is not None:
        list_checks.append(("probe_contact_count_range", list(contact_count_range)))
    if cfg.probe.component_count_values is not None:
        list_checks.append(("probe_component_count_values", [int(x) for x in cfg.probe.component_count_values]))
    if cfg.probe.component_count_distribution is not None:
        list_checks.append(
            ("probe_component_count_distribution", [float(x) for x in cfg.probe.component_count_distribution])
        )
    composite_ranges = getattr(cfg.probe, "composite_contact_count_ranges", None)
    if composite_ranges is not None:
        expected_ranges = {
            str(k): [int(v[0]), int(v[1])]
            for k, v in OmegaConf.to_container(composite_ranges, resolve=True).items()
        }
        actual_ranges = {
            str(k): [int(v[0]), int(v[1])]
            for k, v in dict(meta.get("composite_contact_count_ranges", {})).items()
        }
        if actual_ranges != expected_ranges:
            return False, f"composite_contact_count_ranges mismatch ({actual_ranges} vs {expected_ranges})"
    if getattr(cfg.probe, "q1_sampling_mode", "gaussian") == "contact_error_mixture":
        list_checks.append(
            (
                "q1_contact_error_bins_mm",
                [list(x) for x in getattr(cfg.probe, "q1_contact_error_bins_mm", [])],
            )
        )
    for key, expected in list_checks:
        if key not in meta:
            return False, f"buffer meta is missing {key}"
        actual = _meta_sequence_to_list(meta[key])
        if key == "probe_component_count_distribution":
            matches = _float_lists_match(actual, list(expected))
        else:
            matches = list(actual) == list(expected)
        if not matches:
            return False, f"{key} mismatch ({actual} vs {expected})"

    return True, None


def _load_probe_dataset_from_buffer(
    cfg: DictConfig,
    *,
    buffer_path: str,
    common_kwargs: Dict[str, Any],
    split_key: str,
    deterministic_runtime: bool,
) -> ContactPolicyProbeDataset:
    meta = ContactPolicyProbeDataset.read_buffer_meta(buffer_path)
    meta_ok, reason = _probe_buffer_matches_cfg(meta, cfg, split_key=split_key)
    if not meta_ok:
        raise ValueError(reason)
    return ContactPolicyProbeDataset.from_buffer_file(
        buffer_path,
        contact_input_points=int(cfg.probe.contact_input_points),
        non_contact_input_points=int(cfg.probe.non_contact_input_points),
        non_contact_sampling_radius=float(cfg.probe.non_contact_sampling_radius),
        non_contact_sdf_clearance=float(cfg.probe.non_contact_sdf_clearance),
        non_contact_pattern_mode=str(cfg.probe.non_contact_pattern_mode),
        non_contact_component_range=_optional_int_pair(cfg.probe.non_contact_component_range),
        non_contact_candidate_points=int(getattr(cfg.probe, "non_contact_candidate_points", 128)),
        non_contact_collision_surface_points=int(getattr(cfg.probe, "non_contact_collision_surface_points", 256)),
        non_contact_epoch_batch_size=int(getattr(cfg.probe, "non_contact_epoch_batch_size", 1024)),
        remove_non_contact_points=bool(getattr(cfg.probe, "remove_non_contact_points", False)),
        non_contact_causal_enabled=bool(getattr(cfg.probe, "non_contact_causal_enabled", False)),
        non_contact_empty_probability=float(getattr(cfg.probe, "non_contact_empty_probability", 0.0)),
        non_contact_goal_joint_noise_scale=float(
            getattr(cfg.probe, "non_contact_goal_joint_noise_scale", 0.0)
        ),
        non_contact_goal_joint_select_min=int(
            getattr(cfg.probe, "non_contact_goal_joint_select_min", 0)
        ),
        non_contact_goal_joint_select_max=int(
            getattr(cfg.probe, "non_contact_goal_joint_select_max", -1)
        ),
        non_contact_blocker_fraction=float(getattr(cfg.probe, "non_contact_blocker_fraction", 0.0)),
        non_contact_blocker_candidate_points=int(
            getattr(cfg.probe, "non_contact_blocker_candidate_points", 128)
        ),
        collision_primitive_spec_dir=str(
            getattr(cfg.probe, "collision_primitive_spec_dir", "./assets/collision_primitives")
        ),
        runtime_resample_deterministic=deterministic_runtime,
        q1_translation_noise_std=float(cfg.probe.q1_translation_noise_std),
        q1_rotation_noise_std_deg=float(cfg.probe.q1_rotation_noise_std_deg),
        q1_joint_noise_scale=float(cfg.probe.q1_joint_noise_scale),
        q1_zero_probability=float(getattr(cfg.probe, "q1_zero_probability", 0.10)),
        random_q1=bool(getattr(cfg.probe, "random_q1", False)),
        q1_sampling_mode=str(getattr(cfg.probe, "q1_sampling_mode", "gaussian")),
        q1_contact_error_zero_prob=float(getattr(cfg.probe, "q1_contact_error_zero_prob", 0.08)),
        q1_contact_error_bins_mm=getattr(cfg.probe, "q1_contact_error_bins_mm", None),
        q1_contact_error_inactive_scale=float(getattr(cfg.probe, "q1_contact_error_inactive_scale", 0.05)),
        q1_contact_error_refine_steps=int(getattr(cfg.probe, "q1_contact_error_refine_steps", 2)),
        sparse_sampling_mode=str(getattr(cfg.probe, "sparse_sampling_mode", "point_uniform")),
        composite_synthesis_mode=str(getattr(cfg.probe, "composite_synthesis_mode", "independent")),
        composite_nearby_max_distance=float(getattr(cfg.probe, "composite_nearby_max_distance", 0.12)),
        composite_contact_count_ranges=getattr(cfg.probe, "composite_contact_count_ranges", None),
        composite_collision_filter_enabled=bool(
            getattr(cfg.probe, "composite_collision_filter_enabled", False)
        ),
        composite_collision_filter_margin=float(getattr(cfg.probe, "composite_collision_filter_margin", 0.0)),
        composite_collision_filter_max_attempts=int(
            getattr(cfg.probe, "composite_collision_filter_max_attempts", 64)
        ),
        hand_points_fps=_resolve_hand_points_fps(cfg),
        **common_kwargs,
    )


# Build one fixed split by loading a saved buffer or synthesizing and saving a new one.
def _resolve_hand_points_fps(cfg: DictConfig) -> int:
    # Only paired_transformer variants need hand FPS tokens in the sample. Returning 0 keeps
    # the other variants untouched (no extra FK, no extra batch keys).
    if not _is_paired_transformer_variant(cfg):
        return 0
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is None:
        return 0
    return int(getattr(model_cfg, "hand_points", 32))


def _resolve_residual_normalizer_mode(cfg: DictConfig) -> str:
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is None:
        return "closed_form"
    mode = str(getattr(model_cfg, "residual_normalizer_mode", "closed_form")).strip().lower()
    if mode not in {"closed_form", "empirical"}:
        raise ValueError(
            f"Unsupported cfg.model.residual_normalizer_mode={mode!r}. Expected 'closed_form' or 'empirical'."
        )
    return mode


def _collect_residual_stats_from_dataset(
    dataset: ContactPolicyProbeDataset,
    *,
    slot: Dict[str, Any],
    num_samples: int,
    batch_size: int = 256,
    seed: int = 42,
) -> torch.Tensor:
    # Pull a representative (optionally subsampled) slice of q2 / q1 pairs from the probe
    # dataset and return residuals of shape (N, residual_dim). Uses __getitems__ so runtime
    # q generation runs exactly as during training — the residual distribution captured here
    # matches what the model actually sees.
    dataset_len = int(len(dataset))
    n = dataset_len if int(num_samples) <= 0 else min(int(num_samples), dataset_len)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(dataset_len, generator=generator)[:n].tolist()
    all_residuals: list[torch.Tensor] = []
    progress = tqdm(total=n, desc="residual-stats", leave=False)
    for start in range(0, n, int(batch_size)):
        chunk = perm[start : start + int(batch_size)]
        samples = dataset.__getitems__(chunk)
        q1 = torch.stack([s["q1_padded"] for s in samples], dim=0)
        q2 = torch.stack([s["action"] for s in samples], dim=0)
        residuals = absolute_to_residual(q2.to(dtype=torch.float32), q1.to(dtype=torch.float32), slot=slot)
        all_residuals.append(residuals.detach().cpu())
        progress.update(len(chunk))
    progress.close()
    return torch.cat(all_residuals, dim=0)


def _build_residual_normalizer_meta(
    cfg: DictConfig,
    *,
    train_dataset: ContactPolicyProbeDataset,
    slot: Dict[str, Any],
    num_robots: int,
) -> Dict[str, Any]:
    # Build the residual-space affine normalizer (scale, offset) for the paired policy. The
    # goal is normalized residuals at roughly unit variance per channel so the diffusion
    # target has sensible magnitude at every timestep.
    mode = _resolve_residual_normalizer_mode(cfg)
    if mode == "closed_form":
        model_cfg = getattr(cfg, "model", None)
        allow_non_gaussian = bool(
            getattr(model_cfg, "allow_non_gaussian_closed_form_normalizer", False)
        )
        non_gaussian_q1 = (
            float(train_dataset.q1_zero_probability) > 0.0
            or bool(train_dataset.random_q1)
            or str(getattr(train_dataset, "q1_sampling_mode", "gaussian")) != "gaussian"
        )
        if non_gaussian_q1 and not allow_non_gaussian:
            # The closed-form derivation assumes q1 ~ N(q2, noise_std^2). If the dataset
            # falls back to canonical-zero or random q1, those samples violate the noise
            # model — flag loudly so the user opts in or switches to empirical.
            raise ValueError(
                "residual_normalizer_mode=closed_form requires q1_zero_probability=0 and random_q1=false. "
                "Switch cfg.model.residual_normalizer_mode to 'empirical' when q1 is not purely Gaussian."
            )
        if non_gaussian_q1 and allow_non_gaussian:
            print(
                "[residual_normalizer] WARNING: using closed_form residual normalizer "
                "with a non-Gaussian q1 sampler because "
                "model.allow_non_gaussian_closed_form_normalizer=true. This is intended "
                "only for explicitly configured non-Gaussian sampling.",
                flush=True,
            )
        xyz_std = float(train_dataset.q1_translation_noise_std)
        rot_std_rad = float(train_dataset.q1_rotation_noise_std_deg) * math.pi / 180.0
        joint_std = train_dataset._joint_noise_std.detach().cpu().to(dtype=torch.float32)
        joint_std_for_slots = joint_std[list(slot["joint_indices"])]
        scale, offset = build_closed_form_residual_normalizer(
            num_robots=int(num_robots),
            slot=slot,
            xyz_std=xyz_std,
            rot_std_rad=rot_std_rad,
            joint_std=joint_std_for_slots,
        )
        return {
            "mode": "closed_form",
            "scale": scale,
            "offset": offset,
            "source": {
                "xyz_std": float(xyz_std),
                "rot_std_rad": float(rot_std_rad),
                "joint_std": joint_std_for_slots.tolist(),
            },
        }
    # empirical
    num_samples_cfg = int(getattr(cfg.model, "residual_normalizer_empirical_samples", 0))
    seed = int(cfg.seed) if hasattr(cfg, "seed") else 42
    print(
        f"[residual_normalizer] empirical mode: collecting residual stats over "
        f"{num_samples_cfg if num_samples_cfg > 0 else len(train_dataset)} train samples"
    )
    residuals = _collect_residual_stats_from_dataset(
        train_dataset,
        slot=slot,
        num_samples=num_samples_cfg,
        seed=seed,
    )
    scale, offset = build_empirical_residual_normalizer(
        residuals,
        slot=slot,
        num_robots=int(num_robots),
    )
    mean = residuals.to(dtype=torch.float64).mean(dim=0).to(dtype=torch.float32)
    std = residuals.to(dtype=torch.float64).std(dim=0, unbiased=False).to(dtype=torch.float32)
    return {
        "mode": "empirical",
        "scale": scale,
        "offset": offset,
        "source": {
            "num_samples_collected": int(residuals.shape[0]),
            "residual_mean": mean,
            "residual_std": std,
        },
    }


def _build_probe_dataset(
    cfg: DictConfig,
    *,
    common_kwargs: Dict[str, Any],
    save_dir: str,
    split: str,
    generation_device: torch.device,
    buffer_path_override: str | None = None,
    runtime_resample_deterministic: bool | None = None,
) -> tuple[ContactPolicyProbeDataset, str]:
    split_key = str(split).strip().lower()
    if split_key not in {"train", "test"}:
        raise ValueError(f"Unsupported split={split}")
    num_samples = int(cfg.probe.train_samples if split_key == "train" else cfg.probe.test_samples)
    buffer_path = (
        str(buffer_path_override)
        if buffer_path_override is not None
        else _resolve_probe_buffer_path(
            getattr(cfg.dataset, f"{split_key}_buffer_path", None),
            save_dir=save_dir,
            default_name=f"probe_{split_key}_buffer.pt",
        )
    )
    dataset_kwargs = dict(common_kwargs)
    dataset_kwargs["device"] = str(generation_device)
    # Train and test buffers must use different seeds, otherwise the fully
    # deterministic synthesizer produces bit-identical samples and evaluation
    # collapses to training-set recall.
    base_seed = int(dataset_kwargs.get("seed", 42))
    dataset_kwargs["seed"] = base_seed if split_key == "train" else base_seed + 10_007
    deterministic_runtime = bool(split_key != "train") if runtime_resample_deterministic is None else bool(runtime_resample_deterministic)
    if buffer_path_override is not None:
        if not os.path.isfile(buffer_path):
            raise FileNotFoundError(f"Explicit {split_key} probe buffer not found: {buffer_path}")
        try:
            dataset = _load_probe_dataset_from_buffer(
                cfg,
                buffer_path=buffer_path,
                common_kwargs=dataset_kwargs,
                split_key=split_key,
                deterministic_runtime=deterministic_runtime,
            )
            print(f"Loaded {split_key} probe buffer from {buffer_path}")
            return dataset, buffer_path
        except (KeyError, RuntimeError, ValueError) as exc:
            # Meta mismatch against an explicit path: rebuild the buffer in place.
            print(f"Rebuilding {split_key} probe buffer at {buffer_path}: {exc}")

    use_saved_buffer = (
        buffer_path_override is None
        and bool(getattr(cfg.dataset, "load_saved_buffers", True))
        and os.path.isfile(buffer_path)
    )
    if use_saved_buffer:
        try:
            dataset = _load_probe_dataset_from_buffer(
                cfg,
                buffer_path=buffer_path,
                common_kwargs=dataset_kwargs,
                split_key=split_key,
                deterministic_runtime=deterministic_runtime,
            )
            print(f"Loaded {split_key} probe buffer from {buffer_path}")
            return dataset, buffer_path
        except (KeyError, RuntimeError, ValueError) as exc:
            print(f"Rebuilding {split_key} probe buffer at {buffer_path}: {exc}")

    dataset = ContactPolicyProbeDataset(
        contact_mode=str(cfg.probe.contact_mode),
        sixd_pose_mode=str(cfg.probe.sixd_pose_mode),
        robot_name=str(cfg.probe.robot_name),
        samples=num_samples,
        build_batch_size=int(cfg.dataset.build_batch_size),
        progress_label=f"probe_{split_key}",
        fixed_contact_links=list(cfg.probe.fixed_contact_links),
        probe_component_range=_optional_int_pair(cfg.probe.component_range),
        probe_component_count_values=None
        if cfg.probe.component_count_values is None
        else list(cfg.probe.component_count_values),
        probe_component_count_distribution=None
        if cfg.probe.component_count_distribution is None
        else list(cfg.probe.component_count_distribution),
        probe_contact_count_range=_optional_int_pair(cfg.probe.contact_count_range),
        contact_input_points=int(cfg.probe.contact_input_points),
        non_contact_input_points=int(cfg.probe.non_contact_input_points),
        non_contact_sampling_radius=float(cfg.probe.non_contact_sampling_radius),
        non_contact_sdf_clearance=float(cfg.probe.non_contact_sdf_clearance),
        non_contact_pattern_mode=str(cfg.probe.non_contact_pattern_mode),
        non_contact_component_range=_optional_int_pair(cfg.probe.non_contact_component_range),
        non_contact_candidate_points=int(getattr(cfg.probe, "non_contact_candidate_points", 128)),
        non_contact_collision_surface_points=int(getattr(cfg.probe, "non_contact_collision_surface_points", 256)),
        non_contact_epoch_batch_size=int(getattr(cfg.probe, "non_contact_epoch_batch_size", 1024)),
        remove_non_contact_points=bool(getattr(cfg.probe, "remove_non_contact_points", False)),
        non_contact_causal_enabled=bool(getattr(cfg.probe, "non_contact_causal_enabled", False)),
        non_contact_empty_probability=float(getattr(cfg.probe, "non_contact_empty_probability", 0.0)),
        non_contact_goal_joint_noise_scale=float(
            getattr(cfg.probe, "non_contact_goal_joint_noise_scale", 0.0)
        ),
        non_contact_goal_joint_select_min=int(
            getattr(cfg.probe, "non_contact_goal_joint_select_min", 0)
        ),
        non_contact_goal_joint_select_max=int(
            getattr(cfg.probe, "non_contact_goal_joint_select_max", -1)
        ),
        non_contact_blocker_fraction=float(getattr(cfg.probe, "non_contact_blocker_fraction", 0.0)),
        non_contact_blocker_candidate_points=int(
            getattr(cfg.probe, "non_contact_blocker_candidate_points", 128)
        ),
        collision_primitive_spec_dir=str(
            getattr(cfg.probe, "collision_primitive_spec_dir", "./assets/collision_primitives")
        ),
        runtime_resample_deterministic=deterministic_runtime,
        q1_translation_noise_std=float(cfg.probe.q1_translation_noise_std),
        q1_rotation_noise_std_deg=float(cfg.probe.q1_rotation_noise_std_deg),
        q1_joint_noise_scale=float(cfg.probe.q1_joint_noise_scale),
        q1_zero_probability=float(getattr(cfg.probe, "q1_zero_probability", 0.10)),
        random_q1=bool(getattr(cfg.probe, "random_q1", False)),
        q1_sampling_mode=str(getattr(cfg.probe, "q1_sampling_mode", "gaussian")),
        q1_contact_error_zero_prob=float(getattr(cfg.probe, "q1_contact_error_zero_prob", 0.08)),
        q1_contact_error_bins_mm=getattr(cfg.probe, "q1_contact_error_bins_mm", None),
        q1_contact_error_inactive_scale=float(getattr(cfg.probe, "q1_contact_error_inactive_scale", 0.05)),
        q1_contact_error_refine_steps=int(getattr(cfg.probe, "q1_contact_error_refine_steps", 2)),
        sparse_sampling_mode=str(getattr(cfg.probe, "sparse_sampling_mode", "point_uniform")),
        composite_synthesis_mode=str(getattr(cfg.probe, "composite_synthesis_mode", "independent")),
        composite_nearby_max_distance=float(getattr(cfg.probe, "composite_nearby_max_distance", 0.12)),
        composite_contact_count_ranges=getattr(cfg.probe, "composite_contact_count_ranges", None),
        composite_collision_filter_enabled=bool(
            getattr(cfg.probe, "composite_collision_filter_enabled", False)
        ),
        composite_collision_filter_margin=float(getattr(cfg.probe, "composite_collision_filter_margin", 0.0)),
        composite_collision_filter_max_attempts=int(
            getattr(cfg.probe, "composite_collision_filter_max_attempts", 64)
        ),
        hand_points_fps=_resolve_hand_points_fps(cfg),
        **dataset_kwargs,
    )
    if bool(getattr(cfg.dataset, "save_buffers", True)):
        dataset.save_buffer(buffer_path)
        print(f"Saved {split_key} probe buffer to {buffer_path}")
    return dataset, buffer_path


# Geometry losses are evaluated on the same robot model used by the dataset whenever possible.
def _build_probe_geometry_model(
    dataset: ContactPolicyProbeDataset,
    device: torch.device,
) -> Any:
    if str(getattr(dataset.model, "device", "")) == str(device):
        return dataset.model
    return create_robot_model(
        robot_name=str(dataset.robot_model_name),
        device=device,
        num_points=int(dataset.surface_num_points),
        **dict(getattr(dataset, "robot_model_kwargs", {})),
    )


def _predict_probe_contact_cloud(
    pred_action: torch.Tensor,
    batch: Dict[str, Any],
    geometry_model: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_probe = pred_action[:, : int(geometry_model.dof)].to(geometry_model.device, dtype=torch.float32)
    contact_valid = batch["contact_valid_mask"].to(geometry_model.device).bool()
    pred_points, pred_normals = geometry_model.transform_attached_points_normals_batch(
        q=pred_probe,
        local_points=batch["contact_local_points"].to(geometry_model.device, dtype=torch.float32),
        link_indices=batch["contact_link_indices"].to(geometry_model.device, dtype=torch.long),
        local_normals=batch["contact_local_normals"].to(geometry_model.device, dtype=torch.float32),
    )
    pred_contact = torch.cat([pred_points, pred_normals], dim=-1)
    target_contact = batch["contact_cloud"][..., :6].to(geometry_model.device, dtype=torch.float32)
    return pred_contact, target_contact, contact_valid


_SUPERVISION_MODES = {
    "full_mix",
    "full_mix_contact_margin",
    "point_normal_margin",
    "contact_joints",
}


def _rotate_normals_rodrigues(
    normals: torch.Tensor,
    rvec: torch.Tensor,
) -> torch.Tensor:
    # Rotate unit vectors by a per-point rotation vector via Rodrigues' formula.
    # The clamp keeps axis well-defined at rvec=0; sin(theta)≈theta there so the
    # rotation reduces to the identity smoothly.
    theta = rvec.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    axis = rvec / theta
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    cross = torch.cross(axis, normals, dim=-1)
    dot = (axis * normals).sum(dim=-1, keepdim=True)
    return normals * cos_t + cross * sin_t + axis * dot * (1.0 - cos_t)


def _augment_contact_points(
    batch: Dict[str, Any],
    *,
    pos_scale: float,
    normal_scale: float,
) -> None:
    # In-place: add Gaussian noise to batch['contact_cloud'] xyz and rotate its
    # normals by a Gaussian-sampled tangent-space vector. Invalid (padded)
    # contact slots are left untouched. A clean copy is stashed under
    # batch['contact_cloud_clean'] so FK-space supervision can keep using the
    # unperturbed target.
    if float(pos_scale) <= 0.0 and float(normal_scale) <= 0.0:
        return
    cloud = batch["contact_cloud"]
    if int(cloud.shape[-1]) < 6:
        return
    valid = batch["contact_valid_mask"].to(device=cloud.device).bool().unsqueeze(-1)
    mask_f = valid.to(dtype=cloud.dtype)

    batch["contact_cloud_clean"] = cloud.clone()
    perturbed = cloud.clone()

    if float(pos_scale) > 0.0:
        pos_noise = torch.randn_like(perturbed[..., :3]) * float(pos_scale)
        perturbed[..., :3] = perturbed[..., :3] + pos_noise * mask_f

    if float(normal_scale) > 0.0:
        normals = perturbed[..., 3:6]
        rvec = torch.randn_like(normals) * float(normal_scale)
        rotated = _rotate_normals_rodrigues(normals, rvec)
        perturbed[..., 3:6] = rotated * mask_f + normals * (1.0 - mask_f)

    batch["contact_cloud"] = perturbed


def _resolve_supervision_mode(cfg: DictConfig) -> str:
    mode = getattr(cfg.training, "supervision_mode", "full_mix")
    mode = str(mode).strip().lower()
    if mode not in _SUPERVISION_MODES:
        raise ValueError(
            f"Unsupported cfg.training.supervision_mode={mode!r}. "
            f"Expected one of: {sorted(_SUPERVISION_MODES)}"
        )
    return mode


def _resolve_supervision_kwargs(cfg: DictConfig) -> Dict[str, Any]:
    base_rotation_mode = str(getattr(cfg.training, "contact_fk_base_rotation_mode", "wrap")).strip().lower()
    if base_rotation_mode not in {"wrap", "clamp"}:
        raise ValueError(
            "training.contact_fk_base_rotation_mode must be one of {'wrap', 'clamp'}, "
            f"got {base_rotation_mode!r}"
        )
    return {
        "model_loss_weight": float(getattr(cfg.training, "model_loss_weight", 1.0)),
        "contact_geometry_loss_weight": float(getattr(cfg.training, "contact_geometry_loss_weight", 1.0)),
        "contact_support_geometry_loss_weight": float(
            getattr(cfg.training, "contact_support_geometry_loss_weight", 1.0)
        ),
        "contact_margin_loss_weight": float(getattr(cfg.training, "contact_margin_loss_weight", 1.0)),
        "contact_margin_m": float(getattr(cfg.training, "contact_margin_m", 0.0005)),
        "contact_space_support_points": int(getattr(cfg.training, "contact_space_support_points", 32)),
        "contact_fk_clamp_pred_q": bool(getattr(cfg.training, "contact_fk_clamp_pred_q", False)),
        "contact_fk_base_rotation_mode": base_rotation_mode,
        "contact_fk_batch_limit": int(getattr(cfg.training, "contact_fk_batch_limit", 0)),
    }


def _masked_paired_pos_normal(
    pred_points: torch.Tensor,
    pred_normals: torch.Tensor,
    target_points: torch.Tensor,
    target_normals: torch.Tensor,
    valid: torch.Tensor,
    *,
    with_normals: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Pairing is known: element i of (pred_points, pred_normals) comes from FK of the
    # pred q on (contact_local_points[b, i], contact_link_indices[b, i]), while element
    # i of (target_points, target_normals) comes from FK of the GT q on the SAME
    # local geometry. So slot-for-slot pairing is exact — no chamfer / NN matching
    # needed. Position is squared L2 per contact; rotation of the normal vector uses
    # (1 - cos(pred, target)) after unit-normalizing both (rotation-aware, magnitude-
    # invariant, no acos gradient cliff). Invalid slots are zeroed via the mask.
    float_mask = valid.to(dtype=pred_points.dtype)
    denom = float_mask.sum().clamp_min(1.0)
    valid_vec = valid.unsqueeze(-1)

    pred_points = torch.where(valid_vec, pred_points, torch.zeros_like(pred_points))
    pred_normals = torch.where(valid_vec, pred_normals, torch.zeros_like(pred_normals))
    target_points = torch.where(valid_vec, target_points, torch.zeros_like(target_points))
    target_normals = torch.where(valid_vec, target_normals, torch.zeros_like(target_normals))

    pos_sq_per_contact = (pred_points - target_points).pow(2).sum(dim=-1)
    pos_loss = (pos_sq_per_contact * float_mask).sum() / denom

    if not with_normals:
        return pos_loss, pred_points.new_zeros(())

    pred_unit = pred_normals / pred_normals.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    target_unit = target_normals / target_normals.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    cos_per_contact = (pred_unit * target_unit).sum(dim=-1).clamp(-1.0, 1.0)
    # (1 - cos) ∈ [0, 2]: 0 for aligned, 2 for anti-aligned.
    normal_loss = ((1.0 - cos_per_contact) * float_mask).sum() / denom

    return pos_loss, normal_loss


def _target_contact_cloud_for_supervision(batch: Dict[str, Any], *, device: torch.device) -> torch.Tensor:
    # Contact-noise augmentation stores the original contacts here. Direct FK supervision
    # should learn the clean target, not the perturbed conditioning input.
    target_key = "contact_cloud_clean" if "contact_cloud_clean" in batch else "contact_cloud"
    return batch[target_key][..., :6].to(device, dtype=torch.float32)


def _compact_contact_fk_inputs(
    *,
    batch: Dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    valid_full = batch["contact_valid_mask"].to(device).bool()
    total_slots = int(valid_full.shape[1])
    if total_slots == 0:
        empty_points = torch.empty(
            (int(valid_full.shape[0]), 0, 3),
            dtype=torch.float32,
            device=device,
        )
        empty_normals = torch.empty_like(empty_points)
        empty_links = torch.empty((int(valid_full.shape[0]), 0), dtype=torch.long, device=device)
        empty_cloud = torch.empty(
            (int(valid_full.shape[0]), 0, 6),
            dtype=torch.float32,
            device=device,
        )
        return empty_points, empty_normals, empty_links, empty_cloud, valid_full, None

    max_valid = int(valid_full.sum(dim=1).max().detach().item())
    target_cloud_full = _target_contact_cloud_for_supervision(batch, device=device)
    if max_valid >= total_slots:
        return (
            batch["contact_local_points"].to(device, dtype=torch.float32),
            batch["contact_local_normals"].to(device, dtype=torch.float32),
            batch["contact_link_indices"].to(device, dtype=torch.long),
            target_cloud_full,
            valid_full,
            None,
        )

    slot_idx = torch.topk(
        valid_full.to(dtype=torch.int64),
        k=max_valid,
        dim=1,
        largest=True,
        sorted=False,
    ).indices
    point_idx = slot_idx.unsqueeze(-1).expand(-1, -1, 3)
    cloud_idx = slot_idx.unsqueeze(-1).expand(-1, -1, 6)
    local_points = batch["contact_local_points"].to(device, dtype=torch.float32).gather(1, point_idx)
    local_normals = batch["contact_local_normals"].to(device, dtype=torch.float32).gather(1, point_idx)
    link_indices = batch["contact_link_indices"].to(device, dtype=torch.long).gather(1, slot_idx)
    target_cloud = target_cloud_full.gather(1, cloud_idx)
    valid = valid_full.gather(1, slot_idx)
    return local_points, local_normals, link_indices, target_cloud, valid, slot_idx


def _contact_fk_from_tensors(
    *,
    geometry_model: Any,
    q: torch.Tensor,
    local_points: torch.Tensor,
    link_indices: torch.Tensor,
    local_normals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return geometry_model.transform_attached_points_normals_batch(
        q=q.to(geometry_model.device, dtype=torch.float32),
        local_points=local_points,
        link_indices=link_indices,
        local_normals=local_normals,
    )


def _farthest_fill_indices(
    points: torch.Tensor,
    *,
    count: int,
    initial_indices: list[int] | None = None,
) -> torch.Tensor:
    points_cpu = points.detach().to(device="cpu", dtype=torch.float32)
    n_points = int(points_cpu.shape[0])
    target_count = min(int(count), n_points)
    if target_count <= 0:
        return torch.empty((0,), dtype=torch.long)
    if n_points <= target_count:
        return torch.arange(n_points, dtype=torch.long)

    selected_mask = torch.zeros((n_points,), dtype=torch.bool)
    selected: list[int] = []
    for raw_idx in initial_indices or []:
        idx = int(raw_idx)
        if 0 <= idx < n_points and not bool(selected_mask[idx]):
            selected_mask[idx] = True
            selected.append(idx)
            if len(selected) >= target_count:
                return torch.as_tensor(selected, dtype=torch.long)

    if not selected:
        center = points_cpu.mean(dim=0, keepdim=True)
        first = int((points_cpu - center).pow(2).sum(dim=-1).argmax().item())
        selected_mask[first] = True
        selected.append(first)

    min_dist2 = torch.full((n_points,), float("inf"), dtype=torch.float32)
    for idx in selected:
        dist2 = (points_cpu - points_cpu[idx].view(1, 3)).pow(2).sum(dim=-1)
        min_dist2 = torch.minimum(min_dist2, dist2)
    while len(selected) < target_count:
        min_dist2 = min_dist2.masked_fill(selected_mask, -1.0)
        next_idx = int(min_dist2.argmax().item())
        if bool(selected_mask[next_idx]):
            break
        selected_mask[next_idx] = True
        selected.append(next_idx)
        dist2 = (points_cpu - points_cpu[next_idx].view(1, 3)).pow(2).sum(dim=-1)
        min_dist2 = torch.minimum(min_dist2, dist2)
    return torch.as_tensor(selected, dtype=torch.long)


def _select_support_template_indices(
    *,
    points: torch.Tensor,
    link_indices: torch.Tensor,
    count: int,
) -> torch.Tensor:
    n_points = int(points.shape[0])
    target_count = min(int(count), n_points)
    if target_count <= 0:
        return torch.empty((0,), dtype=torch.long)
    if n_points <= target_count:
        return torch.arange(n_points, dtype=torch.long)

    link_cpu = link_indices.detach().to(device="cpu", dtype=torch.long).view(-1)
    valid_link = link_cpu >= 0
    unique_links = torch.unique(link_cpu[valid_link], sorted=True)
    initial: list[int] = []
    if 0 < int(unique_links.numel()) <= target_count:
        for link_id in unique_links.tolist():
            first = int(torch.nonzero(link_cpu == int(link_id), as_tuple=False)[0].item())
            initial.append(first)
    return _farthest_fill_indices(points, count=target_count, initial_indices=initial)


def _get_support_template(
    geometry_model: Any,
    *,
    num_points: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_count = int(num_points)
    device = geometry_model.device
    if target_count <= 0:
        empty_points = torch.empty((0, 3), dtype=torch.float32, device=device)
        empty_links = torch.empty((0,), dtype=torch.long, device=device)
        return empty_points, empty_points.clone(), empty_links
    cache = getattr(geometry_model, "_contact_support_template_cache", None)
    if cache is None:
        cache = {}
        setattr(geometry_model, "_contact_support_template_cache", cache)
    if target_count in cache:
        return cache[target_count]

    points = geometry_model.surface_template_points_local.detach().to(device=device, dtype=torch.float32)
    normals = geometry_model.surface_template_normals_local.detach().to(device=device, dtype=torch.float32)
    links = geometry_model.surface_template_link_indices.detach().to(device=device, dtype=torch.long)
    if int(points.shape[0]) == 0:
        empty_points = torch.empty((0, 3), dtype=torch.float32, device=device)
        empty_links = torch.empty((0,), dtype=torch.long, device=device)
        template = (empty_points, empty_points.clone(), empty_links)
    else:
        selected = _select_support_template_indices(points=points, link_indices=links, count=target_count).to(
            device=device,
            dtype=torch.long,
        )
        template = (
            points.index_select(0, selected).contiguous(),
            normals.index_select(0, selected).contiguous(),
            links.index_select(0, selected).contiguous(),
        )
    cache[target_count] = template
    return template


def _expand_support_template(
    geometry_model: Any,
    *,
    batch_size: int,
    num_points: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    points, normals, links = _get_support_template(geometry_model, num_points=int(num_points))
    points_b = points.unsqueeze(0).expand(int(batch_size), -1, -1)
    normals_b = normals.unsqueeze(0).expand(int(batch_size), -1, -1)
    links_b = links.unsqueeze(0).expand(int(batch_size), -1)
    valid = torch.ones((int(batch_size), int(points.shape[0])), dtype=torch.bool, device=geometry_model.device)
    return points_b, normals_b, links_b, valid


def _clamp_q_for_contact_fk(
    *,
    geometry_model: Any,
    pred_action: torch.Tensor,
    dof: int,
    enabled: bool,
    base_rotation_mode: str,
) -> torch.Tensor:
    # Direct FK supervision should not feed arbitrarily invalid joint angles into
    # kinematics. Clamp the FK-only copy of articulated q using the geometry model's
    # limits, but leave base translations untouched: they do not create the same
    # numerical risk, and virtual-base translation limits are often arbitrary.
    pred_q = pred_action[:, :dof].to(dtype=torch.float32)
    if not bool(enabled):
        return pred_q
    if not hasattr(geometry_model, "pk_chain"):
        raise RuntimeError("contact FK q clamping requires geometry_model.pk_chain.get_joint_limits()")

    lower, upper = geometry_model.pk_chain.get_joint_limits()
    lower_cpu = torch.as_tensor(lower, dtype=torch.float32).view(-1)
    upper_cpu = torch.as_tensor(upper, dtype=torch.float32).view(-1)
    if int(lower_cpu.numel()) < int(dof) or int(upper_cpu.numel()) < int(dof):
        raise ValueError(
            f"geometry joint limits must cover dof={dof}, got lower={lower_cpu.numel()} upper={upper_cpu.numel()}"
        )
    lower_t = lower_cpu[:dof].to(device=pred_action.device)
    upper_t = upper_cpu[:dof].to(device=pred_action.device)
    q_out = pred_q.clone()

    base_rotation_indices = [
        int(i)
        for i in getattr(geometry_model, "base_rotation_indices", [])
        if 0 <= int(i) < int(dof)
    ]
    base_translation_indices = {
        int(i)
        for i in getattr(geometry_model, "base_translation_indices", [])
        if 0 <= int(i) < int(dof)
    }
    if base_rotation_indices:
        rot_idx = torch.as_tensor(base_rotation_indices, dtype=torch.long, device=pred_action.device)
        if str(base_rotation_mode).strip().lower() == "wrap":
            q_out[:, rot_idx] = torch.remainder(q_out[:, rot_idx] + float(math.pi), float(2.0 * math.pi)) - float(math.pi)
        elif str(base_rotation_mode).strip().lower() == "clamp":
            q_out[:, rot_idx] = q_out[:, rot_idx].clamp(min=-float(math.pi), max=float(math.pi))
        else:
            raise ValueError(f"Unsupported contact_fk_base_rotation_mode={base_rotation_mode!r}")

    finite_mask_cpu = torch.isfinite(lower_cpu[:dof]) & torch.isfinite(upper_cpu[:dof])
    if base_translation_indices:
        finite_mask_cpu[list(base_translation_indices)] = False
    finite_idx_cpu = torch.nonzero(finite_mask_cpu, as_tuple=False).view(-1)
    if int(finite_idx_cpu.numel()) > 0:
        finite_idx = finite_idx_cpu.to(device=pred_action.device)
        q_out[:, finite_idx] = q_out[:, finite_idx].clamp(
            min=lower_t.index_select(0, finite_idx),
            max=upper_t.index_select(0, finite_idx),
        )
    return q_out


def _masked_contact_margin_loss(
    *,
    pred_points: torch.Tensor,
    q1_points: torch.Tensor,
    target_points: torch.Tensor,
    valid: torch.Tensor,
    margin_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Contractive point loss: prediction should be closer to the target than q1 by at
    # least margin_m. This is intentionally position-only; normals remain supervised by
    # the paired cosine term.
    float_mask = valid.to(dtype=pred_points.dtype)
    denom = float_mask.sum().clamp_min(1.0)
    valid_vec = valid.unsqueeze(-1)

    pred_delta = torch.where(valid_vec, pred_points - target_points, torch.zeros_like(pred_points))
    q1_delta = torch.where(valid_vec, q1_points - target_points, torch.zeros_like(q1_points))
    pred_dist = torch.sqrt(pred_delta.pow(2).sum(dim=-1).clamp_min(0.0) + 1.0e-12)
    q1_dist = torch.sqrt(q1_delta.pow(2).sum(dim=-1).clamp_min(0.0) + 1.0e-12)
    required_dist = (q1_dist - float(margin_m)).clamp_min(0.0)
    violation = torch.where(valid, torch.relu(pred_dist - required_dist), torch.zeros_like(pred_dist))
    margin_loss = (violation.pow(2) * float_mask).sum() / denom
    violation_frac = ((violation > 0.0) & valid).to(dtype=pred_points.dtype).sum() / denom

    q1_pos_sq = q1_delta.pow(2).sum(dim=-1)
    q1_pos_mse = (q1_pos_sq * float_mask).sum() / denom
    return margin_loss, violation_frac, q1_pos_mse


def _assert_finite_loss(loss: torch.Tensor, *, name: str, loss_dict: Dict[str, Any]) -> None:
    if torch.isfinite(loss.detach()).all():
        return
    diagnostics = {
        k: (float(v.detach().item()) if isinstance(v, torch.Tensor) else float(v))
        for k, v in loss_dict.items()
    }
    raise FloatingPointError(f"Non-finite {name}: {diagnostics}")


def _index_batch_first_dim(
    batch: Dict[str, Any],
    indices: torch.Tensor,
    *,
    batch_size: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == int(batch_size):
            out[key] = value.index_select(0, indices.to(device=value.device))
        else:
            out[key] = value
    return out


def _compute_supervision_loss(
    *,
    mode: str,
    model: nn.Module,
    batch: Dict[str, Any],
    geometry_model: Any | None,
    normal_weight: float,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    variant: str,
    model_loss_weight: float = 1.0,
    contact_geometry_loss_weight: float = 1.0,
    contact_support_geometry_loss_weight: float = 1.0,
    contact_margin_loss_weight: float = 1.0,
    contact_margin_m: float = 0.0005,
    contact_space_support_points: int = 32,
    contact_fk_clamp_pred_q: bool = False,
    contact_fk_base_rotation_mode: str = "wrap",
    contact_fk_batch_limit: int = 0,
    model_output: tuple[Any, Any, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if model_output is None:
        with _autocast_context(device, use_amp, amp_dtype):
            model_loss, loss_dict, aux = model.compute_loss(batch, return_aux=True)
    else:
        model_loss, loss_dict, aux = model_output
    loss_dict = dict(loss_dict)
    _assert_finite_loss(model_loss, name="model_loss", loss_dict=loss_dict)
    is_paired = variant in _PAIRED_TRANSFORMER_VARIANTS

    if mode == "full_mix":
        loss = model_loss
        _assert_finite_loss(loss, name="supervision_loss", loss_dict=loss_dict)
        loss_dict["supervision_loss"] = loss.detach()
        return loss, model_loss, loss_dict

    pred_action = aux["pred_action"]
    if not pred_action.requires_grad:
        raise RuntimeError(
            f"supervision_mode={mode!r} needs a differentiable pred_action from the policy, "
            "but aux['pred_action'] has requires_grad=False. Check the model's compute_loss implementation."
        )

    if mode == "contact_joints":
        if "contact_joint_mask" not in batch:
            raise KeyError(
                "supervision_mode='contact_joints' requires batch['contact_joint_mask'] from the dataset. "
                "Rebuild probe buffers after updating data_process/contact_policy_probe_dataset.py."
            )
        batch_size = int(pred_action.shape[0])
        if is_paired:
            # Residual-space contact_joints: loss lives in the normalized residual layout,
            # where the 3 Euler rotation slots correspond to 6 rot6d slots (any-active rule
            # applied inside build_residual_mask_from_absolute).
            if "pred_residual_norm" not in aux or "residual_target_norm" not in aux:
                raise RuntimeError(
                    "paired_transformer compute_loss must expose pred_residual_norm and residual_target_norm in aux."
                )
            pred_norm = aux["pred_residual_norm"]
            gt_norm = aux["residual_target_norm"]
            residual_valid_mask = aux["residual_valid_mask"].to(pred_norm.device, dtype=torch.float32)
            residual_joint_mask = model._build_residual_joint_mask(batch, batch_size).to(
                pred_norm.device, dtype=torch.float32
            )
            mask = residual_valid_mask * residual_joint_mask
        else:
            # Absolute-q space contact_joints: per-joint normalized MSE, matched to full_mix.
            norm_scale, norm_offset = model._get_action_normalizer_params(batch, batch_size=batch_size)
            pred_norm = model._normalize_action(
                pred_action.to(dtype=torch.float32),
                scale=norm_scale,
                offset=norm_offset,
            )
            gt_norm = model._normalize_action(
                batch["action"].to(pred_norm.device, dtype=torch.float32),
                scale=norm_scale,
                offset=norm_offset,
            )
            valid_mask = batch["action_valid_mask"].to(pred_norm.device, dtype=torch.float32)
            joint_mask = batch["contact_joint_mask"].to(pred_norm.device, dtype=torch.float32)
            mask = valid_mask * joint_mask
        sq_err = (pred_norm - gt_norm).pow(2) * mask
        loss_per_sample = sq_err.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        loss = loss_per_sample.mean()
        loss_dict["contact_joint_mse"] = loss.detach()
        loss_dict["supervision_loss"] = loss.detach()
        return loss, model_loss, loss_dict

    # FK-space modes use paired geometry supervision. pred_action already lives in
    # absolute q space (residual_to_absolute has re-assembled the transformer's split
    # xyz / rot6d / joint heads back into an Euler-parameterized q), so FK consumes
    # pred_action[:, :dof] directly with no further reshaping.
    if geometry_model is None:
        raise RuntimeError(f"geometry_model is required for supervision_mode={mode!r}")
    batch_size_full = int(pred_action.shape[0])
    fk_limit = int(contact_fk_batch_limit)
    if 0 < fk_limit < batch_size_full:
        # Keep the residual/model loss on the full batch, but estimate the expensive
        # FK-space auxiliary losses on a random subset. Default is disabled, so
        # existing configs keep exact previous behavior.
        subset_idx = torch.randperm(batch_size_full, device=pred_action.device)[:fk_limit]
        pred_action = pred_action.index_select(0, subset_idx)
        batch = _index_batch_first_dim(batch, subset_idx, batch_size=batch_size_full)
    dof = int(geometry_model.dof)
    local_points, local_normals, link_indices, target_cloud, valid, compact_slot_idx = _compact_contact_fk_inputs(
        batch=batch,
        device=geometry_model.device,
    )
    pred_q = _clamp_q_for_contact_fk(
        geometry_model=geometry_model,
        pred_action=pred_action,
        dof=dof,
        enabled=bool(contact_fk_clamp_pred_q),
        base_rotation_mode=str(contact_fk_base_rotation_mode),
    ).to(geometry_model.device, dtype=torch.float32)
    batch_size = int(pred_action.shape[0])
    use_support = mode == "point_normal_margin" and int(contact_space_support_points) > 0
    if use_support:
        support_local_points, support_local_normals, support_link_indices, support_valid = _expand_support_template(
            geometry_model,
            batch_size=batch_size,
            num_points=int(contact_space_support_points),
        )
        fk_local_points = torch.cat([local_points, support_local_points], dim=1)
        fk_local_normals = torch.cat([local_normals, support_local_normals], dim=1)
        fk_link_indices = torch.cat([link_indices, support_link_indices], dim=1)
        contact_slots = int(local_points.shape[1])
    else:
        support_local_points = support_local_normals = support_link_indices = support_valid = None
        fk_local_points = local_points
        fk_local_normals = local_normals
        fk_link_indices = link_indices
        contact_slots = int(local_points.shape[1])

    pred_points, pred_normals = _contact_fk_from_tensors(
        geometry_model=geometry_model,
        q=pred_q,
        local_points=fk_local_points,
        link_indices=fk_link_indices,
        local_normals=fk_local_normals,
    )
    pred_contact_points = pred_points[:, :contact_slots]
    pred_contact_normals = pred_normals[:, :contact_slots]
    target_points = target_cloud[..., :3]
    target_normals = target_cloud[..., 3:6]
    pos_loss, normal_loss = _masked_paired_pos_normal(
        pred_points=pred_contact_points,
        pred_normals=pred_contact_normals,
        target_points=target_points,
        target_normals=target_normals,
        valid=valid,
        with_normals=True,
    )
    if mode == "full_mix_contact_margin":
        if "q1_contact_points_world" in batch:
            q1_points = batch["q1_contact_points_world"].to(geometry_model.device, dtype=torch.float32)
            if compact_slot_idx is not None:
                q1_points = q1_points.gather(
                    1,
                    compact_slot_idx.unsqueeze(-1).expand(-1, -1, 3),
                )
        else:
            q1_q = batch["q1_padded"][:, :dof].to(geometry_model.device, dtype=torch.float32)
            with torch.no_grad():
                q1_points, _ = _contact_fk_from_tensors(
                    geometry_model=geometry_model,
                    q=q1_q,
                    local_points=local_points,
                    link_indices=link_indices,
                    local_normals=local_normals,
                )
        contact_geometry_loss = pos_loss + float(normal_weight) * normal_loss
        margin_loss, violation_frac, q1_pos_mse = _masked_contact_margin_loss(
            pred_points=pred_contact_points,
            q1_points=q1_points,
            target_points=target_points,
            valid=valid,
            margin_m=float(contact_margin_m),
        )
        weighted_model_loss = model_loss * float(model_loss_weight)
        weighted_geometry_loss = contact_geometry_loss * float(contact_geometry_loss_weight)
        weighted_margin_loss = margin_loss * float(contact_margin_loss_weight)
        loss = weighted_model_loss + weighted_geometry_loss + weighted_margin_loss
        _assert_finite_loss(
            loss,
            name="full_mix_contact_margin",
            loss_dict={
                **loss_dict,
                "paired_pos_mse": pos_loss,
                "paired_normal_cos": normal_loss,
                "contact_geometry_loss": contact_geometry_loss,
                "contact_margin_loss": margin_loss,
                "contact_margin_violation_frac": violation_frac,
                "contact_q1_pos_mse": q1_pos_mse,
                "weighted_model_loss": weighted_model_loss,
                "weighted_contact_geometry_loss": weighted_geometry_loss,
                "weighted_contact_margin_loss": weighted_margin_loss,
            },
        )
        loss_dict["paired_pos_mse"] = pos_loss.detach()
        loss_dict["paired_normal_cos"] = normal_loss.detach()
        loss_dict["contact_geometry_loss"] = contact_geometry_loss.detach()
        loss_dict["contact_margin_loss"] = margin_loss.detach()
        loss_dict["contact_margin_violation_frac"] = violation_frac.detach()
        loss_dict["contact_q1_pos_mse"] = q1_pos_mse.detach()
        loss_dict["weighted_model_loss"] = weighted_model_loss.detach()
        loss_dict["weighted_contact_geometry_loss"] = weighted_geometry_loss.detach()
        loss_dict["weighted_contact_margin_loss"] = weighted_margin_loss.detach()
        loss_dict["supervision_loss"] = loss.detach()
        return loss, model_loss, loss_dict

    if mode == "point_normal_margin":
        if support_local_points is None or support_local_normals is None or support_link_indices is None or support_valid is None:
            raise RuntimeError("point_normal_margin requires contact_space_support_points > 0")
        pred_support_points = pred_points[:, contact_slots:]
        pred_support_normals = pred_normals[:, contact_slots:]
        target_q = batch["action"][:, :dof].to(geometry_model.device, dtype=torch.float32)
        with torch.no_grad():
            target_support_points, target_support_normals = _contact_fk_from_tensors(
                geometry_model=geometry_model,
                q=target_q,
                local_points=support_local_points,
                link_indices=support_link_indices,
                local_normals=support_local_normals,
            )
        if "q1_contact_points_world" in batch:
            q1_points = batch["q1_contact_points_world"].to(geometry_model.device, dtype=torch.float32)
            if compact_slot_idx is not None:
                q1_points = q1_points.gather(
                    1,
                    compact_slot_idx.unsqueeze(-1).expand(-1, -1, 3),
                )
        else:
            q1_points = None
        if q1_points is None:
            q1_q = batch["q1_padded"][:, :dof].to(geometry_model.device, dtype=torch.float32)
            with torch.no_grad():
                q1_points, _ = _contact_fk_from_tensors(
                    geometry_model=geometry_model,
                    q=q1_q,
                    local_points=local_points,
                    link_indices=link_indices,
                    local_normals=local_normals,
                )
        contact_geometry_loss = pos_loss + float(normal_weight) * normal_loss
        support_pos_loss, support_normal_loss = _masked_paired_pos_normal(
            pred_points=pred_support_points,
            pred_normals=pred_support_normals,
            target_points=target_support_points,
            target_normals=target_support_normals,
            valid=support_valid,
            with_normals=True,
        )
        support_geometry_loss = support_pos_loss + float(normal_weight) * support_normal_loss
        margin_loss, violation_frac, q1_pos_mse = _masked_contact_margin_loss(
            pred_points=pred_contact_points,
            q1_points=q1_points,
            target_points=target_points,
            valid=valid,
            margin_m=float(contact_margin_m),
        )
        weighted_geometry_loss = contact_geometry_loss * float(contact_geometry_loss_weight)
        weighted_support_loss = support_geometry_loss * float(contact_support_geometry_loss_weight)
        weighted_margin_loss = margin_loss * float(contact_margin_loss_weight)
        loss = weighted_geometry_loss + weighted_support_loss + weighted_margin_loss
        _assert_finite_loss(
            loss,
            name="point_normal_margin",
            loss_dict={
                **loss_dict,
                "contact_paired_pos_mse": pos_loss,
                "contact_paired_normal_cos": normal_loss,
                "contact_geometry_loss": contact_geometry_loss,
                "support_paired_pos_mse": support_pos_loss,
                "support_paired_normal_cos": support_normal_loss,
                "contact_support_geometry_loss": support_geometry_loss,
                "contact_margin_loss": margin_loss,
                "contact_margin_violation_frac": violation_frac,
                "contact_q1_pos_mse": q1_pos_mse,
                "weighted_contact_geometry_loss": weighted_geometry_loss,
                "weighted_contact_support_geometry_loss": weighted_support_loss,
                "weighted_contact_margin_loss": weighted_margin_loss,
            },
        )
        loss_dict["contact_paired_pos_mse"] = pos_loss.detach()
        loss_dict["contact_paired_normal_cos"] = normal_loss.detach()
        loss_dict["contact_geometry_loss"] = contact_geometry_loss.detach()
        loss_dict["support_paired_pos_mse"] = support_pos_loss.detach()
        loss_dict["support_paired_normal_cos"] = support_normal_loss.detach()
        loss_dict["contact_support_geometry_loss"] = support_geometry_loss.detach()
        loss_dict["contact_margin_loss"] = margin_loss.detach()
        loss_dict["contact_margin_violation_frac"] = violation_frac.detach()
        loss_dict["contact_q1_pos_mse"] = q1_pos_mse.detach()
        loss_dict["weighted_contact_geometry_loss"] = weighted_geometry_loss.detach()
        loss_dict["weighted_contact_support_geometry_loss"] = weighted_support_loss.detach()
        loss_dict["weighted_contact_margin_loss"] = weighted_margin_loss.detach()
        loss_dict["supervision_loss"] = loss.detach()
        return loss, model_loss, loss_dict

    raise RuntimeError(f"Unhandled FK-space supervision mode: {mode!r}")


# Probe test metrics; the pointnet_mlp variant also reports its contact-alignment objective.
def _compute_probe_test_metrics(
    policy: nn.Module,
    loader: DataLoader,
    geometry_model: Any | None,
    device: torch.device,
    *,
    prefix: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    variant: str,
    compute_geometric_loss: bool,
) -> Dict[str, float]:
    policy.eval()
    variant_key = str(variant).strip().lower()
    use_pointnet_objective = variant_key == "pointnet_mlp"
    full_action_mse = 0.0
    valid_action_mse = 0.0
    contact_point_mse = 0.0
    contact_normal_mse = 0.0
    model_loss_sum = 0.0
    total_samples = 0
    total_contact_batches = 0

    with torch.no_grad():
        progress = tqdm(total=len(loader), desc=prefix, leave=False)
        for batch in loader:
            batch_device = _move_batch_to_device(batch, device)
            if use_pointnet_objective:
                with _autocast_context(device, use_amp, amp_dtype):
                    model_loss, _, _ = policy.compute_loss(batch_device, return_aux=True)
            else:
                with _autocast_context(device, use_amp, amp_dtype):
                    model_loss, _ = policy.compute_loss(batch_device)
            pred_action = policy.predict_action(batch_device)["action_pred"].detach()
            _cuda_sync_if_needed(device)

            gt_action = batch_device["action"].to(dtype=torch.float32)
            action_valid_mask = batch_device["action_valid_mask"].to(dtype=torch.float32)
            sq_error = (pred_action - gt_action).pow(2)
            full_action_mse += float(sq_error.mean(dim=1).sum().item())
            valid_action_mse += float(
                ((sq_error * action_valid_mask).sum(dim=1) / action_valid_mask.sum(dim=1).clamp_min(1.0)).sum().item()
            )
            total_samples += int(gt_action.shape[0])

            model_loss_sum += float(model_loss.item()) * int(gt_action.shape[0])

            if compute_geometric_loss:
                if geometry_model is None:
                    raise RuntimeError("geometry_model is required when geometry metrics are enabled")
                pred_contact, target_contact, contact_valid = _predict_probe_contact_cloud(
                    pred_action.to(dtype=torch.float32),
                    batch_device,
                    geometry_model,
                )
                contact_mask_xyz = contact_valid.unsqueeze(-1).expand_as(target_contact[..., :3])
                per_batch_contact_mse = (
                    ((pred_contact[..., :3] - target_contact[..., :3]).pow(2) * contact_mask_xyz.float()).sum()
                    / contact_mask_xyz.float().sum().clamp_min(1.0)
                )
                contact_mask_nrm = contact_valid.unsqueeze(-1).expand_as(target_contact[..., 3:6])
                per_batch_contact_normal_mse = (
                    ((pred_contact[..., 3:6] - target_contact[..., 3:6]).pow(2) * contact_mask_nrm.float()).sum()
                    / contact_mask_nrm.float().sum().clamp_min(1.0)
                )
                contact_point_mse += float(per_batch_contact_mse.item()) * int(gt_action.shape[0])
                contact_normal_mse += float(per_batch_contact_normal_mse.item()) * int(gt_action.shape[0])
                total_contact_batches += int(gt_action.shape[0])
            progress.update(1)
        progress.close()

    if total_samples == 0:
        raise RuntimeError("Probe test dataloader returned zero samples")

    metrics = {
        f"{prefix}/action_mse_full": full_action_mse / float(total_samples),
        f"{prefix}/action_mse_valid": valid_action_mse / float(total_samples),
    }

    avg_model_loss = model_loss_sum / float(total_samples)
    metrics[f"{prefix}/loss"] = avg_model_loss
    metrics[f"{prefix}/model_loss"] = avg_model_loss

    if compute_geometric_loss:
        metrics[f"{prefix}/contact_point_mse"] = contact_point_mse / float(max(total_contact_batches, 1))
        metrics[f"{prefix}/contact_normal_mse"] = contact_normal_mse / float(max(total_contact_batches, 1))
    return metrics
