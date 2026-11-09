from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
CROSS_DIR = EXPERIMENT_DIR.parent
IDP3_REPO_ROOT = CROSS_DIR / "third_party" / "Improved-3D-Diffusion-Policy" / "Improved-3D-Diffusion-Policy"


def ensure_idp3_on_path() -> Path:
    if not IDP3_REPO_ROOT.is_dir():
        raise FileNotFoundError(f"Vendored iDP3 package not found: {IDP3_REPO_ROOT}")
    root = str(IDP3_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return IDP3_REPO_ROOT


ensure_idp3_on_path()

from diffusers.schedulers.scheduling_ddim import DDIMScheduler  # noqa: E402
from diffusion_policy_3d.model.common.normalizer import (  # noqa: E402
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy_3d.policy.diffusion_pointcloud_policy import (  # noqa: E402
    DiffusionPointcloudPolicy,
)


def idp3_model_cfg(cfg: DictConfig) -> DictConfig:
    model_cfg = OmegaConf.select(cfg, "high_level_model", default=None)
    if model_cfg is None:
        model_cfg = OmegaConf.select(cfg, "policy", default=None)
    if model_cfg is None:
        raise KeyError("Expected cfg.high_level_model or legacy cfg.policy model settings.")
    return model_cfg


def _as_int_tuple(values: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in values)


def build_idp3_shape_meta(
    *,
    state_dim: int,
    action_dim: int,
    point_count: int,
    point_channels: int = 3,
) -> DictConfig:
    return OmegaConf.create({
        "obs": {
            "point_cloud": {
                "shape": [int(point_count), int(point_channels)],
                "type": "point_cloud",
            },
            "agent_pos": {
                "shape": [int(state_dim)],
                "type": "low_dim",
            },
        },
        "action": {
            "shape": [int(action_dim)],
        },
    })


def build_idp3_noise_scheduler(model_cfg: DictConfig) -> DDIMScheduler:
    sched_cfg = model_cfg.noise_scheduler
    return DDIMScheduler(
        num_train_timesteps=int(sched_cfg.num_train_timesteps),
        beta_start=float(sched_cfg.beta_start),
        beta_end=float(sched_cfg.beta_end),
        beta_schedule=str(sched_cfg.beta_schedule),
        clip_sample=bool(sched_cfg.clip_sample),
        set_alpha_to_one=bool(sched_cfg.set_alpha_to_one),
        steps_offset=int(sched_cfg.steps_offset),
        prediction_type=str(sched_cfg.prediction_type),
    )


def build_idp3_policy(
    cfg: DictConfig,
    *,
    action_dim: int,
    state_dim: int,
    point_count: int | None = None,
) -> DiffusionPointcloudPolicy:
    model_cfg = idp3_model_cfg(cfg)
    point_cfg = model_cfg.pointcloud_encoder_cfg
    shape_point_count = int(point_cfg.num_points) if point_count is None else int(point_count)
    point_channels = 6 if bool(model_cfg.use_pc_color) else 3
    shape_meta = build_idp3_shape_meta(
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        point_count=shape_point_count,
        point_channels=point_channels,
    )
    return DiffusionPointcloudPolicy(
        shape_meta=shape_meta,
        noise_scheduler=build_idp3_noise_scheduler(model_cfg),
        horizon=int(model_cfg.horizon),
        n_action_steps=int(model_cfg.n_action_steps),
        n_obs_steps=int(model_cfg.n_obs_steps),
        num_inference_steps=int(model_cfg.num_inference_steps),
        obs_as_global_cond=bool(model_cfg.obs_as_global_cond),
        diffusion_step_embed_dim=int(model_cfg.diffusion_step_embed_dim),
        down_dims=_as_int_tuple(model_cfg.down_dims),
        kernel_size=int(model_cfg.kernel_size),
        n_groups=int(model_cfg.n_groups),
        condition_type=str(model_cfg.condition_type),
        use_down_condition=bool(model_cfg.use_down_condition),
        use_mid_condition=bool(model_cfg.use_mid_condition),
        use_up_condition=bool(model_cfg.use_up_condition),
        use_pc_color=bool(model_cfg.use_pc_color),
        pointnet_type=str(model_cfg.pointnet_type),
        pointcloud_encoder_cfg=point_cfg,
        point_downsample=bool(model_cfg.point_downsample),
    )


def build_idp3_normalizer(
    *,
    action_data: torch.Tensor | np.ndarray,
    agent_pos_data: torch.Tensor | np.ndarray | None = None,
    point_cloud_stats: dict[str, torch.Tensor] | None = None,
    normalize_agent_pos: bool = False,
    normalize_point_cloud: bool = False,
) -> LinearNormalizer:
    normalizer = LinearNormalizer()
    normalizer.fit(data={"action": action_data}, last_n_dims=1, mode="limits")
    if bool(normalize_point_cloud):
        if point_cloud_stats is None:
            raise ValueError("point_cloud_stats is required when normalize_point_cloud=True")
        xyz_min = torch.as_tensor(point_cloud_stats["min"], dtype=torch.float32).reshape(3)
        xyz_max = torch.as_tensor(point_cloud_stats["max"], dtype=torch.float32).reshape(3)
        center = 0.5 * (xyz_min + xyz_max)
        radius = 0.5 * (xyz_max - xyz_min).max()
        if not bool(torch.isfinite(center).all()) or not bool(torch.isfinite(radius)) or float(radius) <= 1.0e-6:
            raise ValueError(f"Cannot normalize point cloud with min={xyz_min.tolist()} max={xyz_max.tolist()}")

        # One fixed offset and one scalar scale preserve world-frame positions and
        # Euclidean geometry. This is dataset normalization, not per-cloud centering.
        scale = torch.full((3,), 1.0 / float(radius), dtype=torch.float32)
        offset = -center * scale
        point_cloud_normalizer = SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict={
                key: torch.as_tensor(point_cloud_stats[key], dtype=torch.float32).reshape(3)
                for key in ("min", "max", "mean", "std")
            },
        )
        point_cloud_normalizer.requires_grad_(False)
        normalizer["point_cloud"] = point_cloud_normalizer
        print(
            "Point-cloud normalization: fixed isotropic train-split limits; "
            f"center={center.tolist()} radius={float(radius):.9f}",
            flush=True,
        )
    else:
        normalizer["point_cloud"] = SingleFieldLinearNormalizer.create_identity()
    if bool(normalize_agent_pos):
        if agent_pos_data is None:
            raise ValueError("agent_pos_data is required when normalize_agent_pos=True")
        normalizer["agent_pos"] = SingleFieldLinearNormalizer.create_fit(
            agent_pos_data,
            last_n_dims=1,
            mode="limits",
        )
    else:
        normalizer["agent_pos"] = SingleFieldLinearNormalizer.create_identity()
    return normalizer


def checkpoint_backbone_metadata() -> dict[str, str]:
    return {
        "high_level_backbone": "official_idp3",
        "idp3_source_path": str(IDP3_REPO_ROOT),
    }
