from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


# Shared probe-policy utilities keep action masking and affine normalization identical across all variants.
class ProbeActionPolicyBase(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        robot_names: tuple[str, ...] | list[str] | None = None,
        action_normalizer_scale: torch.Tensor | None = None,
        action_normalizer_offset: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.robot_names = tuple(str(x) for x in (robot_names or []))

        num_robots = (
            int(action_normalizer_scale.shape[0])
            if action_normalizer_scale is not None
            else max(1, len(self.robot_names))
        )
        if action_normalizer_scale is None:
            action_normalizer_scale = torch.ones((num_robots, self.action_dim), dtype=torch.float32)
        if action_normalizer_offset is None:
            action_normalizer_offset = torch.zeros((num_robots, self.action_dim), dtype=torch.float32)
        action_normalizer_scale = torch.as_tensor(action_normalizer_scale, dtype=torch.float32)
        action_normalizer_offset = torch.as_tensor(action_normalizer_offset, dtype=torch.float32)
        if action_normalizer_scale.shape != action_normalizer_offset.shape:
            raise ValueError(
                "action normalizer scale/offset shape mismatch: "
                f"{tuple(action_normalizer_scale.shape)} vs {tuple(action_normalizer_offset.shape)}"
            )
        if action_normalizer_scale.ndim != 2 or int(action_normalizer_scale.shape[1]) != self.action_dim:
            raise ValueError(
                f"action normalizer must have shape (num_robots, {self.action_dim}), got {tuple(action_normalizer_scale.shape)}"
            )
        if len(self.robot_names) not in (0, int(action_normalizer_scale.shape[0])):
            raise ValueError(
                f"robot_names length {len(self.robot_names)} does not match normalizer rows {action_normalizer_scale.shape[0]}"
            )
        self.register_buffer("action_normalizer_scale", action_normalizer_scale)
        self.register_buffer("action_normalizer_offset", action_normalizer_offset)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # A shared head can serve different hands by masking out invalid padded action dimensions.
    def _build_action_valid_mask(self, batch: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        if "action_valid_mask" in batch:
            action_valid_mask = batch["action_valid_mask"].to(self.device).bool()
            if (
                action_valid_mask.ndim == 2
                and int(action_valid_mask.shape[0]) == int(batch_size)
                and int(action_valid_mask.shape[1]) == self.action_dim
            ):
                return action_valid_mask
            raise ValueError(
                f"action_valid_mask must have shape (B, {self.action_dim}), got {tuple(action_valid_mask.shape)}"
            )
        if "local_dof" in batch:
            local_dof = batch["local_dof"].to(self.device, dtype=torch.long).view(-1)
            if int(local_dof.shape[0]) != int(batch_size):
                raise ValueError(f"local_dof batch mismatch: expected {batch_size}, got {tuple(local_dof.shape)}")
            dim_idx = torch.arange(self.action_dim, device=self.device, dtype=torch.long).unsqueeze(0)
            return dim_idx < local_dof.unsqueeze(1)
        return torch.ones((int(batch_size), self.action_dim), dtype=torch.bool, device=self.device)

    def _build_robot_index(self, batch: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        if "robot_index" in batch:
            robot_index = batch["robot_index"].to(self.device, dtype=torch.long).view(-1)
            if int(robot_index.shape[0]) != int(batch_size):
                raise ValueError(f"robot_index batch mismatch: expected {batch_size}, got {tuple(robot_index.shape)}")
        elif int(self.action_normalizer_scale.shape[0]) == 1:
            robot_index = torch.zeros((int(batch_size),), dtype=torch.long, device=self.device)
        else:
            raise ValueError("robot_index is required for multi-robot action normalization")
        num_robots = int(self.action_normalizer_scale.shape[0])
        if torch.compiler.is_compiling():
            # Keep the range check inside compiled graphs without synchronizing the GPU
            # to convert a tensor predicate into a Python bool.
            torch._assert_async(
                ((robot_index >= 0) & (robot_index < num_robots)).all(),
                f"robot_index must be in [0, {num_robots - 1}]",
            )
        elif bool((robot_index < 0).any()) or bool((robot_index >= num_robots).any()):
            raise ValueError(
                f"robot_index must be in [0, {num_robots - 1}], "
                f"got min={int(robot_index.min().item())} max={int(robot_index.max().item())}"
            )
        return robot_index

    def _get_action_normalizer_params(
        self,
        batch: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        robot_index = self._build_robot_index(batch, batch_size=batch_size)
        scale = self.action_normalizer_scale.index_select(0, robot_index)
        offset = self.action_normalizer_offset.index_select(0, robot_index)
        return scale, offset

    # Always zero-out padded joints before sampling noise, computing losses, or returning predictions.
    @staticmethod
    def _apply_action_valid_mask(sample: torch.Tensor, action_valid_mask: torch.Tensor) -> torch.Tensor:
        if sample.ndim == 2:
            return sample * action_valid_mask.to(dtype=sample.dtype)
        if sample.ndim == 3:
            return sample * action_valid_mask.unsqueeze(1).to(dtype=sample.dtype)
        raise ValueError(f"Unsupported sample ndim={sample.ndim}")

    def _normalize_action(
        self,
        action: torch.Tensor,
        *,
        scale: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        if action.ndim == 2:
            return action * scale.to(dtype=action.dtype) + offset.to(dtype=action.dtype)
        if action.ndim == 3:
            scale3 = scale.unsqueeze(1).to(dtype=action.dtype)
            offset3 = offset.unsqueeze(1).to(dtype=action.dtype)
            return action * scale3 + offset3
        raise ValueError(f"Unsupported action ndim={action.ndim}")

    def _unnormalize_action(
        self,
        action: torch.Tensor,
        *,
        scale: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        if action.ndim == 2:
            return (action - offset.to(dtype=action.dtype)) / scale.to(dtype=action.dtype)
        if action.ndim == 3:
            scale3 = scale.unsqueeze(1).to(dtype=action.dtype)
            offset3 = offset.unsqueeze(1).to(dtype=action.dtype)
            return (action - offset3) / scale3
        raise ValueError(f"Unsupported action ndim={action.ndim}")

    # Reduce only over valid joints so robots with fewer active DoFs are not under-weighted by padding.
    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor, action_valid_mask: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
        if pred.ndim == 2:
            mask = action_valid_mask.to(dtype=pred.dtype)
            loss_per_sample = ((pred - target).pow(2) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            return loss_per_sample.mean()
        if pred.ndim == 3:
            mask = action_valid_mask.unsqueeze(1).to(dtype=pred.dtype)
            loss_per_sample = ((pred - target).pow(2) * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)
            return loss_per_sample.mean()
        raise ValueError(f"Unsupported pred ndim={pred.ndim}")
