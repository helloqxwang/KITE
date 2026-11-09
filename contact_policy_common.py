"""Shared runtime utilities for contact-policy training and evaluation.

This module contains the small pieces used by both the co-training entry point and
the reusable probe helpers. Keeping them here avoids training scripts importing one
another and makes checkpoint/evaluation imports independent of a Hydra entry point.
"""

from __future__ import annotations

import contextlib
import os
import random
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import DictConfig
from torch.nn.modules.batchnorm import _BatchNorm

from networks.simple_diffusion import SimpleDDIMScheduler


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class EMAModel:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ) -> None:
        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)
        self.update_after_step = int(update_after_step)
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, int(optimization_step) - self.update_after_step - 1)
        value = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model: torch.nn.Module) -> None:
        self.decay = self.get_decay(self.optimization_step)
        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules()):
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                if isinstance(module, _BatchNorm) or not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1.0 - self.decay)
        self.optimization_step += 1

    def state_dict(self) -> Dict[str, float | int]:
        return {
            "decay": float(self.decay),
            "optimization_step": int(self.optimization_step),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.decay = float(state_dict.get("decay", 0.0))
        self.optimization_step = int(state_dict.get("optimization_step", 0))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prepare_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _resolve_project_path(path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.abspath(os.path.join(ROOT_DIR, path_str))


def _resolve_optional_project_path(path_val: Any) -> str | None:
    if path_val is None:
        return None
    path_str = str(path_val).strip()
    if path_str == "" or path_str.lower() == "null":
        return None
    return _resolve_project_path(path_str)


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _get_amp_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if key in {"float16", "fp16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {name}")


def _autocast_context(device: torch.device, enabled: bool, amp_dtype: torch.dtype):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _build_noise_scheduler(cfg: DictConfig) -> SimpleDDIMScheduler:
    return SimpleDDIMScheduler(
        num_train_timesteps=int(cfg.num_train_timesteps),
        beta_start=float(cfg.beta_start),
        beta_end=float(cfg.beta_end),
        beta_schedule=str(cfg.beta_schedule),
        clip_sample=bool(cfg.clip_sample),
        set_alpha_to_one=bool(cfg.set_alpha_to_one),
        steps_offset=int(cfg.steps_offset),
        prediction_type=str(cfg.prediction_type),
    )


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    name: str,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    schedule_name = str(name).lower()
    warmup_steps = max(0, int(num_warmup_steps))
    total_steps = max(1, int(num_training_steps))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if schedule_name == "constant":
            return 1.0
        if schedule_name == "linear":
            return max(0.0, 1.0 - progress)
        if schedule_name == "cosine":
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        raise ValueError(f"Unsupported lr_scheduler: {name}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _extract_wandb_run_id_from_path(path: str) -> str | None:
    real_path = os.path.realpath(path)
    base = os.path.basename(real_path.rstrip(os.sep))
    if not base.startswith("run-"):
        return None
    parts = base.split("-")
    if len(parts) < 3:
        return None
    run_id = parts[-1].strip()
    return run_id or None


def _infer_wandb_run_id(save_dir: str) -> str | None:
    run_id_path = os.path.join(save_dir, "wandb_run_id.txt")
    if os.path.isfile(run_id_path):
        with open(run_id_path, "r", encoding="utf-8") as file:
            run_id = file.read().strip()
        if run_id:
            return run_id

    latest_run_path = os.path.join(ROOT_DIR, "wandb", "latest-run")
    if os.path.exists(latest_run_path):
        run_id = _extract_wandb_run_id_from_path(latest_run_path)
        if run_id is not None:
            return run_id
    return None


def _build_action_normalizer_meta(
    cfg: DictConfig,
    *,
    robot_specs: Dict[str, Dict[str, Any]],
    global_robot_names: list[str],
    action_dim: int,
) -> Dict[str, Any]:
    num_robots = len(global_robot_names)
    scale = torch.ones((num_robots, int(action_dim)), dtype=torch.float32)
    offset = torch.zeros((num_robots, int(action_dim)), dtype=torch.float32)
    translation_min = torch.as_tensor(cfg.dataset.q2_base_translation_min, dtype=torch.float32).view(3)
    translation_max = torch.as_tensor(cfg.dataset.q2_base_translation_max, dtype=torch.float32).view(3)
    for robot_idx, robot_name in enumerate(global_robot_names):
        model = robot_specs[robot_name]["model"]
        lower, upper = model.pk_chain.get_joint_limits()
        lower_t = torch.as_tensor(lower, dtype=torch.float32)
        upper_t = torch.as_tensor(upper, dtype=torch.float32)
        finite = torch.isfinite(lower_t) & torch.isfinite(upper_t)
        lower_t = torch.where(finite, lower_t, torch.full_like(lower_t, -float(np.pi)))
        upper_t = torch.where(finite, upper_t, torch.full_like(upper_t, float(np.pi)))
        pose_groups = getattr(model, "pose_groups", None)
        if pose_groups is None:
            pose_groups = (
                {
                    "translation_indices": tuple(int(i) for i in model.base_translation_indices),
                    "rotation_indices": tuple(int(i) for i in model.base_rotation_indices),
                },
            )
        for group in pose_groups:
            trans_idx = [int(i) for i in group.get("translation_indices", ())]
            rot_idx = [int(i) for i in group.get("rotation_indices", ())]
            if trans_idx:
                lower_t[trans_idx] = translation_min[: len(trans_idx)]
                upper_t[trans_idx] = translation_max[: len(trans_idx)]
            if rot_idx:
                lower_t[rot_idx] = -float(np.pi)
                upper_t[rot_idx] = float(np.pi)
        input_range = upper_t - lower_t
        ignore_dim = input_range.abs() < 1e-6
        input_range = input_range.clone()
        input_range[ignore_dim] = 2.0
        robot_scale = 2.0 / input_range
        robot_offset = -1.0 - robot_scale * lower_t
        robot_scale[ignore_dim] = 1.0
        robot_offset[ignore_dim] = -lower_t[ignore_dim]
        scale[robot_idx, : int(model.dof)] = robot_scale
        offset[robot_idx, : int(model.dof)] = robot_offset
    return {
        "robot_names": list(global_robot_names),
        "scale": scale,
        "offset": offset,
        "q2_base_translation_min": translation_min,
        "q2_base_translation_max": translation_max,
    }


def build_contact_policy_common_dataset_kwargs(
    cfg: DictConfig,
    *,
    robot_names: list[str] | None = None,
    global_robot_names: list[str] | None = None,
) -> Dict[str, Any]:
    resolved_global_robot_names = (
        list(cfg.dataset.robot_names)
        if global_robot_names is None
        else [str(x) for x in global_robot_names]
    )
    real_masks_path_val = getattr(cfg.dataset, "real_masks_path", None)
    return {
        "robot_names": list(cfg.dataset.robot_names) if robot_names is None else [str(x) for x in robot_names],
        "global_robot_names": resolved_global_robot_names,
        "hparams_path": _resolve_project_path(str(cfg.dataset.hparams_path)),
        "real_masks_path": None if real_masks_path_val is None else _resolve_project_path(str(real_masks_path_val)),
        "max_contact_points": int(cfg.dataset.max_contact_points),
        "seed": int(cfg.seed),
        "check_template_hash": bool(cfg.dataset.check_template_hash),
        "threshold": float(cfg.dataset.threshold),
        "align_exp_scale": float(cfg.dataset.align_exp_scale),
        "sigmoid_scale": float(cfg.dataset.sigmoid_scale),
        "component_sampling_mode": str(cfg.dataset.component_sampling_mode),
        "anchor_sampling_mode": str(cfg.dataset.anchor_sampling_mode),
        "anchor_temperature": float(cfg.dataset.anchor_temperature),
        "patch_anchor_shift_min": float(cfg.dataset.patch_anchor_shift_min),
        "patch_anchor_shift_max": float(cfg.dataset.patch_anchor_shift_max),
        "patch_extent_min": float(cfg.dataset.patch_extent_min),
        "patch_extent_max": float(cfg.dataset.patch_extent_max),
        "patch_extent_power": float(cfg.dataset.patch_extent_power),
        "patch_shift_power": float(cfg.dataset.patch_shift_power),
        "patch_normal_jitter_max_deg": float(cfg.dataset.patch_normal_jitter_max_deg),
        "patch_points_per_anchor_min": int(cfg.dataset.patch_points_per_anchor_min),
        "patch_points_per_anchor_max": int(cfg.dataset.patch_points_per_anchor_max),
        "patch_penetration_clearance": float(cfg.dataset.patch_penetration_clearance),
        "q2_base_translation_min": [float(x) for x in cfg.dataset.q2_base_translation_min],
        "q2_base_translation_max": [float(x) for x in cfg.dataset.q2_base_translation_max],
        "q2_base_rotation_mode": str(cfg.dataset.q2_base_rotation_mode),
    }


def _build_loader_kwargs(cfg: DictConfig, *, persistent_workers: bool) -> Dict[str, Any]:
    num_workers = int(cfg.dataset.num_workers)
    kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(cfg.dataset.pin_memory),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(cfg.dataset.prefetch_factor)
    return kwargs
