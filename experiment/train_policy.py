from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

EXPERIMENT_DIR = Path(__file__).resolve().parent
CROSS_DIR = EXPERIMENT_DIR.parent
if str(CROSS_DIR) not in sys.path:
    sys.path.insert(0, str(CROSS_DIR))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from high_level_policy.dataset import HighLevelSequenceDataset  # noqa: E402
from high_level_policy.train_utils import (  # noqa: E402
    EMAModel,
    autocast_context,
    build_lr_scheduler,
    get_amp_dtype,
    move_to_device,
    prepare_device,
    set_seed,
)
from experiment.high_level_policy.policy import (  # noqa: E402
    backbone_name,
    build_policy_for_dataset,
    build_sequence_dataset,
    build_train_normalizer,
    checkpoint_metadata,
    model_cfg,
    observation_kind,
)


# Resolve config paths relative to experiment/ so Hydra runs stay location-independent.
from paper_config import runtime_config


def _resolve_path(path_str: str | None) -> str | None:
    if path_str is None:
        return None
    text = str(path_str).strip()
    if text == "" or text.lower() == "null":
        return None
    if os.path.isabs(text):
        return text
    return os.path.abspath(os.path.join(str(EXPERIMENT_DIR), text))


# Build the selected policy from dataset-driven observation/action dimensions.
def _build_policy(cfg: DictConfig, *, dataset: HighLevelSequenceDataset) -> torch.nn.Module:
    return build_policy_for_dataset(cfg, dataset)


def _cfg_action_space(cfg: DictConfig) -> str:
    action_space = OmegaConf.select(cfg, "policy.action_space", default=None)
    if action_space is None or str(action_space).strip() == "":
        raise KeyError("Expected config key policy.action_space, for example 'latent' or 'marvin_wuji'.")
    return str(action_space).strip().lower()


def _resume_signature(cfg: DictConfig) -> dict[str, Any]:
    """Fields that must remain invariant when continuing an optimization run."""

    rgb_store = OmegaConf.select(cfg, "dataset.rgb_store_path", default=None)
    return {
        "backbone": backbone_name(cfg),
        "action_space": _cfg_action_space(cfg),
        "payload_path": _resolve_path(str(cfg.dataset.payload_path)),
        "rgb_store_path": None if rgb_store is None else _resolve_path(str(rgb_store)),
        "test_episode_count": int(OmegaConf.select(cfg, "dataset.test_episode_count", default=20)),
        "split_seed": int(OmegaConf.select(cfg, "dataset.split_seed", default=42)),
        "split_group_key": OmegaConf.select(cfg, "dataset.split_group_key", default=None),
        "high_level_model": OmegaConf.to_container(model_cfg(cfg), resolve=True),
    }


@torch.no_grad()
def _sync_ema_buffers(source: torch.nn.Module, target: torch.nn.Module) -> None:
    """Keep non-parameter state (notably PointNet BatchNorm stats) current."""

    source_buffers = dict(source.named_buffers())
    target_buffers = dict(target.named_buffers())
    if set(source_buffers) != set(target_buffers):
        raise RuntimeError(f"EMA buffer keys differ: {set(source_buffers) ^ set(target_buffers)}")
    for name, source_buffer in source_buffers.items():
        target_buffers[name].copy_(source_buffer.to(target_buffers[name]))


# Keep the inherited iDP3 modules and their custom normalizer in one runtime
# dtype/device. This does not change the architecture; it prevents CUDA dtype
# mismatches in PointNet/Conv1d after normalizer state loading or checkpoint resume.
def _prepare_policy_runtime(policy: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    return policy.to(device=device, dtype=torch.float32)


# Create a DataLoader with the worker settings used by both train and eval splits.
def _build_loader(dataset, *, batch_size: int, num_workers: int, shuffle: bool, pin_memory: bool, persistent_workers: bool):
    use_workers = max(0, int(num_workers))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=use_workers,
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers and use_workers > 0),
        drop_last=False,
    )


# Run one full evaluation pass and return the mean diffusion loss.
def _evaluate(
    policy: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int | None = None,
) -> float:
    policy.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            batch_device = move_to_device(batch, device)
            with autocast_context(device, enabled=use_amp, amp_dtype=amp_dtype):
                loss, _ = policy.compute_loss(batch_device)
            losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


# Runtime implementation shared by the pick-and-place training entrypoint.
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.seed))
    device = prepare_device(str(cfg.training.device))
    payload_path = _resolve_path(str(cfg.dataset.payload_path))
    if payload_path is None or not os.path.isfile(payload_path):
        raise FileNotFoundError(f"Dataset payload not found: {payload_path}")
    action_space = _cfg_action_space(cfg)
    # Build train/test temporal windows from one serialized payload.
    train_dataset = build_sequence_dataset(
        cfg,
        payload_path=payload_path,
        action_space=action_space,
        split="train",
    )
    # Persist the live observation contract independently of the point count
    # stored in an evaluation dataset.
    OmegaConf.update(cfg, "dataset.observation_point_count", int(train_dataset.num_points), force_add=True)
    test_dataset = build_sequence_dataset(
        cfg,
        payload_path=payload_path,
        action_space=action_space,
        split="test",
        allow_empty=True,
    )
    overlap = set(train_dataset.selected_episode_indices.tolist()).intersection(test_dataset.selected_episode_indices.tolist())
    if overlap:
        raise RuntimeError(f"Train/test episode split overlap detected: {sorted(overlap)}")
    print(
        "High-level split: "
        f"train_episodes={len(train_dataset.selected_episode_indices)} "
        f"test_episodes={len(test_dataset.selected_episode_indices)} "
        f"train_windows={len(train_dataset)} test_windows={len(test_dataset)} "
        f"action_dim={train_dataset.action_dim} state_dim={train_dataset.state_dim} "
        f"backbone={backbone_name(cfg)} observation={observation_kind(cfg)} "
        f"points={train_dataset.num_points}",
        flush=True,
    )

    # Wrap datasets in loaders before deriving model dimensions from the train split.
    train_loader = _build_loader(
        train_dataset,
        batch_size=int(cfg.dataset.batch_size),
        num_workers=int(cfg.dataset.num_workers),
        shuffle=True,
        pin_memory=bool(cfg.dataset.pin_memory),
        persistent_workers=bool(cfg.dataset.persistent_workers),
    )
    test_loader = None
    if len(test_dataset) > 0:
        test_loader = _build_loader(
            test_dataset,
            batch_size=int(cfg.dataset.eval_batch_size),
            num_workers=int(cfg.dataset.num_workers),
            shuffle=False,
            pin_memory=bool(cfg.dataset.pin_memory),
            persistent_workers=bool(cfg.dataset.persistent_workers),
        )

    # Instantiate the policy, attach split-specific normalizers, and set up optimization.
    policy = _prepare_policy_runtime(_build_policy(cfg, dataset=train_dataset), device)
    normalizer = build_train_normalizer(cfg, train_dataset)
    policy.set_normalizer(normalizer)
    _prepare_policy_runtime(policy, device)
    optimizer_parameters: Any = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if hasattr(policy, "optimizer_param_groups"):
        optimizer_parameters = policy.optimizer_param_groups(
            lr=float(cfg.training.lr),
            lr_backbone=float(OmegaConf.select(cfg, "training.lr_backbone", default=cfg.training.lr)),
        )
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=float(cfg.training.lr),
        betas=tuple(float(x) for x in cfg.training.betas),
        eps=float(cfg.training.eps),
        weight_decay=float(cfg.training.weight_decay),
    )
    steps_per_epoch = len(train_loader)
    if cfg.training.max_train_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, int(cfg.training.max_train_steps_per_epoch))
    total_steps = max(1, steps_per_epoch * int(cfg.training.epochs))
    lr_scheduler = build_lr_scheduler(
        optimizer,
        name=str(cfg.training.lr_scheduler),
        num_warmup_steps=int(cfg.training.lr_warmup_steps),
        num_training_steps=total_steps,
    )

    ema_helper = None
    ema_policy = None
    if bool(cfg.training.use_ema):
        ema_helper = EMAModel(
            policy,
            update_after_step=int(cfg.training.ema.update_after_step),
            inv_gamma=float(cfg.training.ema.inv_gamma),
            power=float(cfg.training.ema.power),
            min_value=float(cfg.training.ema.min_value),
            max_value=float(cfg.training.ema.max_value),
        )
        ema_policy = ema_helper.averaged_model.to(device)
        ema_policy.load_state_dict(policy.state_dict())

    # Optionally resume the full optimizer/scheduler/model state from the last checkpoint.
    save_dir = Path(_resolve_path(str(cfg.training.save_dir)) or str(EXPERIMENT_DIR / "checkpoints" / "high_level_policy")).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "last.pt"
    start_epoch = 1
    global_step = 0
    best_eval = float("inf")
    if bool(cfg.training.resume) and ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        stored_signature = ckpt.get("resume_signature")
        if stored_signature is None and isinstance(ckpt.get("config"), dict):
            stored_signature = _resume_signature(OmegaConf.create(ckpt["config"]))
        current_signature = _resume_signature(cfg)
        if stored_signature is not None and stored_signature != current_signature:
            raise ValueError(
                "Refusing to resume with incompatible data/model settings.\n"
                f"checkpoint={json.dumps(stored_signature, indent=2, sort_keys=True)}\n"
                f"current={json.dumps(current_signature, indent=2, sort_keys=True)}"
            )
        policy.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        policy.normalizer.load_state_dict(ckpt["normalizer"])
        _prepare_policy_runtime(policy, device)
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt["global_step"])
        best_eval = float(ckpt.get("best_eval_loss", ckpt.get("best_val_loss", float("inf"))))
        if ema_helper is not None and "ema_model" in ckpt:
            ema_policy.load_state_dict(ckpt["ema_model"], strict=True)
            _prepare_policy_runtime(ema_policy, device)
            ema_helper.load_state_dict(ckpt.get("ema_state", {}))

    # Configure AMP and logging after resume so both fresh and resumed runs share one path.
    amp_dtype = get_amp_dtype(str(cfg.training.amp_dtype))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.training.use_amp and device.type == "cuda" and amp_dtype == torch.float16))

    run_name = str(cfg.wandb.name) if cfg.wandb.name is not None else str(cfg.experiment.name)
    if bool(cfg.wandb.enabled):
        wandb.init(
            project=str(cfg.wandb.project),
            entity=None if cfg.wandb.entity is None else str(cfg.wandb.entity),
            name=run_name,
            mode=str(cfg.wandb.mode),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    else:
        wandb.init(mode="disabled")

    # Standard epoch loop: optimize on train windows, periodically evaluate, then checkpoint.
    for epoch in range(start_epoch, int(cfg.training.epochs) + 1):
        policy.train()
        train_losses = []
        progress = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
        max_train_steps = None if cfg.training.max_train_steps_per_epoch is None else int(cfg.training.max_train_steps_per_epoch)
        for batch_idx, batch in enumerate(progress):
            if max_train_steps is not None and batch_idx >= max_train_steps:
                break
            batch_device = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, enabled=bool(cfg.training.use_amp), amp_dtype=amp_dtype):
                loss, loss_dict = policy.compute_loss(batch_device)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if float(cfg.training.grad_clip_norm) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), float(cfg.training.grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if float(cfg.training.grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), float(cfg.training.grad_clip_norm))
                optimizer.step()
            lr_scheduler.step()
            if ema_helper is not None:
                ema_helper.step(policy)
                ema_policy = ema_helper.averaged_model
                if backbone_name(cfg) != "idp3":
                    _sync_ema_buffers(policy, ema_policy)
            global_step += 1
            train_losses.append(float(loss.detach().cpu().item()))
            progress.set_postfix(loss=float(loss.detach().cpu().item()))
            if global_step % int(cfg.training.log_every_n_steps) == 0:
                wandb.log(
                    {
                        "train/loss_step": float(loss.detach().cpu().item()),
                        "train/lr": float(lr_scheduler.get_last_lr()[0]),
                        **{f"train/{key}": float(value) for key, value in loss_dict.items()},
                    },
                    step=global_step,
                )

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        eval_policy = ema_policy if ema_policy is not None else policy
        eval_loss = float("nan")
        did_eval = False
        if test_loader is not None and (epoch % int(cfg.training.eval_every_n_epochs) == 0 or epoch == int(cfg.training.epochs)):
            did_eval = True
            eval_loss = _evaluate(
                eval_policy,
                test_loader,
                device,
                use_amp=bool(cfg.training.use_amp),
                amp_dtype=amp_dtype,
                max_batches=None if cfg.training.max_eval_batches is None else int(cfg.training.max_eval_batches),
            )
            if not np.isfinite(eval_loss):
                print(
                    f"Evaluation returned a non-finite loss at epoch={epoch}. "
                    "This means evaluation actually ran but produced no valid scalar.",
                    flush=True,
                )
        epoch_log = {
            "train/loss_epoch": train_loss,
            "epoch": epoch,
            "global_step": global_step,
        }
        if did_eval:
            if np.isfinite(eval_loss):
                epoch_log["test/loss"] = float(eval_loss)
        wandb.log(epoch_log, step=global_step)

        # Save enough state to resume training or evaluate the EMA model later.
        ckpt_payload: dict[str, Any] = {
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "normalizer": policy.normalizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_eval_loss": float(best_eval),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "resume_signature": _resume_signature(cfg),
            **checkpoint_metadata(cfg),
        }
        if ema_policy is not None:
            ckpt_payload["ema_model"] = ema_policy.state_dict()
            ckpt_payload["ema_state"] = ema_helper.state_dict() if ema_helper is not None else {}
        if epoch % int(cfg.training.save_every_n_epochs) == 0 or epoch == int(cfg.training.epochs):
            torch.save(ckpt_payload, ckpt_path)
        if np.isfinite(eval_loss) and eval_loss < best_eval:
            best_eval = float(eval_loss)
            ckpt_payload["best_eval_loss"] = float(best_eval)
            torch.save(ckpt_payload, save_dir / "best.pt")
            torch.save(ckpt_payload, ckpt_path)

    wandb.finish()


# Hydra entrypoint that wires one payload into deterministic train/test runtime splits.
@hydra.main(version_base="1.2", config_path="conf", config_name="train_policy/object_suite")
def main(cfg: DictConfig) -> None:
    cfg = runtime_config(cfg)
    train(cfg)


if __name__ == "__main__":
    from paper_config import canonicalize_cli
    canonicalize_cli()
    main()
