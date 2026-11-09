"""Checkpoint and saved synthetic-data loading helpers for the kinematic decoder viewers."""
from __future__ import annotations

import copy
import os
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from train_kinematic_decoder import _prepare_device, _resolve_project_path
from contact_policy_probe import (
    _build_probe_dataset, _build_probe_dataset_common_kwargs, _optional_int_pair,
    _probe_buffer_matches_cfg, _resolve_portable_project_path, _resolve_probe_buffer_path,
    _resolve_hand_points_fps, load_contact_policy_probe_checkpoint,
)
from data_process.contact_policy_probe_dataset import ContactPolicyProbeDataset


def _optional_cfg_str(cfg: DictConfig, key: str) -> str | None:
    value = OmegaConf.select(cfg, key, default=None)
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "null":
        return None
    return text


def _resolve_visualization_split(cfg: DictConfig) -> str:
    split_name = str(OmegaConf.select(cfg, "visualization.split", default="train")).strip().lower()
    if split_name == "validate":
        return "test"
    if split_name in {"train", "test"}:
        return split_name
    raise ValueError(f"Unsupported visualization.split={split_name}. Expected train or validate.")


def _load_batch_from_dataset(
    dataset: Any,
    *,
    split_key: str,
    start_index: int,
    num_samples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    end_index = min(len(dataset), int(start_index) + int(num_samples))
    indices = list(range(int(start_index), end_index))
    if len(indices) == 0:
        raise IndexError(
            f"No samples available for split={split_key} at start_index={start_index}. "
            f"Dataset length={len(dataset)}."
        )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=len(indices),
        shuffle=False,
        num_workers=0,
    )
    return next(iter(loader)), {"indices": indices, "split": split_key}


def _torch_load_probe_payload_mmap(buffer_path: str) -> dict[str, Any]:
    try:
        payload = torch.load(
            str(buffer_path),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        print(
            "[visualization] WARNING: this PyTorch build does not support torch.load(..., mmap=True); "
            f"loading {buffer_path} may consume substantial RAM.",
            flush=True,
        )
        payload = torch.load(str(buffer_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "meta" not in payload or "buffers" not in payload:
        raise ValueError(f"Invalid probe buffer payload: {buffer_path}")
    return payload


def _slice_probe_payload(
    payload: dict[str, Any],
    *,
    start_index: int,
    num_samples: int,
) -> tuple[dict[str, Any], list[int]]:
    meta = payload["meta"]
    total = int(meta["samples"])
    start = int(max(0, start_index))
    end = min(total, start + int(max(1, num_samples)))
    if start >= total or end <= start:
        raise IndexError(
            f"No samples available in buffer at start_index={start_index}. Buffer length={total}."
        )
    indices = list(range(start, end))
    sliced_meta = copy.deepcopy(meta)
    sliced_meta["samples"] = len(indices)
    sliced_meta["progress_label"] = f"{meta.get('progress_label', 'probe_dataset')}_vis_slice"
    sliced_buffers: dict[str, torch.Tensor] = {}
    for key, value in payload["buffers"].items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == total:
            sliced_buffers[key] = value[start:end].clone()
        elif isinstance(value, torch.Tensor):
            sliced_buffers[key] = value.clone()
        else:
            sliced_buffers[key] = copy.deepcopy(value)
    return {"meta": sliced_meta, "buffers": sliced_buffers}, indices


def _dataset_runtime_kwargs_from_cfg(cfg: DictConfig, *, deterministic_runtime: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "contact_input_points": int(cfg.probe.contact_input_points),
        "non_contact_input_points": int(cfg.probe.non_contact_input_points),
        "non_contact_sampling_radius": float(cfg.probe.non_contact_sampling_radius),
        "non_contact_sdf_clearance": float(cfg.probe.non_contact_sdf_clearance),
        "non_contact_pattern_mode": str(cfg.probe.non_contact_pattern_mode),
        "non_contact_component_range": _optional_int_pair(cfg.probe.non_contact_component_range),
        "non_contact_candidate_points": int(getattr(cfg.probe, "non_contact_candidate_points", 128)),
        "non_contact_collision_surface_points": int(getattr(cfg.probe, "non_contact_collision_surface_points", 256)),
        "non_contact_epoch_batch_size": int(getattr(cfg.probe, "non_contact_epoch_batch_size", 1024)),
        "remove_non_contact_points": bool(getattr(cfg.probe, "remove_non_contact_points", False)),
        "non_contact_causal_enabled": bool(getattr(cfg.probe, "non_contact_causal_enabled", False)),
        "non_contact_empty_probability": float(getattr(cfg.probe, "non_contact_empty_probability", 0.0)),
        "non_contact_goal_joint_noise_scale": float(getattr(cfg.probe, "non_contact_goal_joint_noise_scale", 0.0)),
        "non_contact_goal_joint_select_min": int(getattr(cfg.probe, "non_contact_goal_joint_select_min", 0)),
        "non_contact_goal_joint_select_max": int(getattr(cfg.probe, "non_contact_goal_joint_select_max", -1)),
        "non_contact_blocker_fraction": float(getattr(cfg.probe, "non_contact_blocker_fraction", 0.0)),
        "non_contact_blocker_candidate_points": int(getattr(cfg.probe, "non_contact_blocker_candidate_points", 128)),
        "collision_primitive_spec_dir": str(getattr(cfg.probe, "collision_primitive_spec_dir", "./assets/collision_primitives")),
        "runtime_resample_deterministic": bool(deterministic_runtime),
        "q1_translation_noise_std": float(cfg.probe.q1_translation_noise_std),
        "q1_rotation_noise_std_deg": float(cfg.probe.q1_rotation_noise_std_deg),
        "q1_joint_noise_scale": float(cfg.probe.q1_joint_noise_scale),
        "q1_zero_probability": float(getattr(cfg.probe, "q1_zero_probability", 0.10)),
        "random_q1": bool(getattr(cfg.probe, "random_q1", False)),
        "q1_sampling_mode": str(getattr(cfg.probe, "q1_sampling_mode", "gaussian")),
        "q1_contact_error_zero_prob": float(getattr(cfg.probe, "q1_contact_error_zero_prob", 0.08)),
        "q1_contact_error_bins_mm": getattr(cfg.probe, "q1_contact_error_bins_mm", None),
        "q1_contact_error_inactive_scale": float(getattr(cfg.probe, "q1_contact_error_inactive_scale", 0.05)),
        "q1_contact_error_refine_steps": int(getattr(cfg.probe, "q1_contact_error_refine_steps", 2)),
        "sparse_sampling_mode": str(getattr(cfg.probe, "sparse_sampling_mode", "point_uniform")),
        "composite_synthesis_mode": str(getattr(cfg.probe, "composite_synthesis_mode", "independent")),
        "composite_nearby_max_distance": float(getattr(cfg.probe, "composite_nearby_max_distance", 0.12)),
        "composite_contact_count_ranges": getattr(cfg.probe, "composite_contact_count_ranges", None),
        "composite_collision_filter_enabled": bool(getattr(cfg.probe, "composite_collision_filter_enabled", False)),
        "composite_collision_filter_margin": float(getattr(cfg.probe, "composite_collision_filter_margin", 0.0)),
        "composite_collision_filter_max_attempts": int(getattr(cfg.probe, "composite_collision_filter_max_attempts", 64)),
        "hand_points_fps": _resolve_hand_points_fps(cfg),
    }
    return kwargs


def _load_sliced_probe_dataset_from_buffer(
    cfg: DictConfig,
    *,
    buffer_path: str,
    common_kwargs: dict[str, Any],
    split_key: str,
    generation_device: torch.device,
    start_index: int,
    num_samples: int,
) -> tuple[Any, list[int]]:
    payload = _torch_load_probe_payload_mmap(buffer_path)
    meta_ok, reason = _probe_buffer_matches_cfg(payload["meta"], cfg, split_key=split_key)
    if not meta_ok:
        raise ValueError(f"Saved probe buffer does not match visualization config: {reason}")
    sliced_payload, source_indices = _slice_probe_payload(
        payload,
        start_index=int(start_index),
        num_samples=int(num_samples),
    )
    dataset_kwargs = dict(common_kwargs)
    dataset_kwargs["device"] = str(generation_device)
    dataset = ContactPolicyProbeDataset.from_buffer_payload(
        sliced_payload,
        **_dataset_runtime_kwargs_from_cfg(cfg, deterministic_runtime=True),
        **dataset_kwargs,
    )
    dataset._visualization_source_indices = source_indices
    return dataset, source_indices


def _resolve_single_probe_buffer(
    cfg: DictConfig,
    *,
    ckpt_meta: dict[str, Any],
    split_key: str,
) -> str:
    meta_buffer_path = _resolve_portable_project_path(ckpt_meta.get(f"{split_key}_buffer_path"))
    if meta_buffer_path is not None and os.path.isfile(meta_buffer_path):
        return meta_buffer_path

    save_dir = _resolve_project_path(str(cfg.training.save_dir))
    cfg_buffer_path = _resolve_probe_buffer_path(
        getattr(cfg.dataset, f"{split_key}_buffer_path", None),
        save_dir=save_dir,
        default_name=f"probe_{split_key}_buffer.pt",
    )
    if os.path.isfile(cfg_buffer_path):
        return cfg_buffer_path

    attempted = []
    if meta_buffer_path is not None:
        attempted.append(f"meta={meta_buffer_path}")
    attempted.append(f"config={cfg_buffer_path}")
    raise FileNotFoundError(
        f"Probe visualization buffer for split={split_key} was not found. Tried: {', '.join(attempted)}"
    )


def _load_single_probe_context(
    cfg: DictConfig,
    *,
    device: torch.device,
) -> tuple[Any, Any, dict[str, Any], str, str, str]:
    checkpoint_path = _optional_cfg_str(cfg, "visualization.checkpoint_path")
    policy, ckpt_path, ckpt_meta = load_contact_policy_probe_checkpoint(
        cfg,
        device=device,
        ckpt_path=checkpoint_path,
        prefer_ema=bool(OmegaConf.select(cfg, "visualization.checkpoint_prefer_ema", default=True)),
    )
    split_key = _resolve_visualization_split(cfg)
    buffer_path = _resolve_single_probe_buffer(cfg, ckpt_meta=ckpt_meta, split_key=split_key)
    common_kwargs = _build_probe_dataset_common_kwargs(cfg)
    save_dir = _resolve_project_path(str(cfg.training.save_dir))
    dataset, resolved_buffer_path = _build_probe_dataset(
        cfg,
        common_kwargs=common_kwargs,
        save_dir=save_dir,
        split=split_key,
        generation_device=_prepare_device(str(cfg.dataset.generation_device)),
        buffer_path_override=buffer_path,
        runtime_resample_deterministic=True,
    )
    return policy, dataset, ckpt_meta, ckpt_path, resolved_buffer_path, split_key
