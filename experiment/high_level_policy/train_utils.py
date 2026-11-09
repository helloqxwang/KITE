from __future__ import annotations

import contextlib
import copy
import random
from typing import Any, Dict

import numpy as np
import torch
from torch.nn.modules.batchnorm import _BatchNorm


# Track an exponential-moving-average copy of the policy for stabler evaluation.
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
        self.averaged_model = copy.deepcopy(model).eval()
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


# Seed Python, NumPy, and Torch so split sampling and training are reproducible.
def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


# Gracefully fall back to CPU when CUDA is requested but unavailable.
def prepare_device(device_str: str) -> torch.device:
    if str(device_str).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(str(device_str))


# Map config strings onto the torch dtype used for mixed precision.
def get_amp_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if key in {"float16", "fp16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {name}")


# Use torch.autocast only when the requested device and dtype actually support it.
def autocast_context(device: torch.device, enabled: bool, amp_dtype: torch.dtype):
    if not bool(enabled) or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


# Recreate the small set of scheduler shapes used by the training config.
def build_lr_scheduler(
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


# Move arbitrarily nested batch structures onto the active device.
def move_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(value, device) for value in batch)
    return batch

