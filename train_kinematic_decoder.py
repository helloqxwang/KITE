"""Train the shared intent encoder and the embodiment-specific kinematic decoder bank."""
from __future__ import annotations

import copy
import os
import sys
import yaml
from typing import Any, Callable, Dict, List, Optional

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from data_process.contact_policy_cotrain_dataset import (  # noqa: E402
    ContactPolicyCoTrainBundle,
    iter_lockstep,
)
from data_process.contact_policy_dataset import (  # noqa: E402
    OnTheFlySyntheticContactPolicyDataset,
)
from data_process.contact_policy_probe_dataset import ContactPolicyProbeDataset  # noqa: E402
from networks.contact_probe_policy_paired_cotrain import (  # noqa: E402
    ContactProbePolicyPairedCoTrain,
)
from networks.residual_action_utils import build_residual_slot_indices  # noqa: E402
from robot_model.robot_model import create_robot_model, resolve_robot_model_name  # noqa: E402
from contact_policy_common import (  # noqa: E402
    EMAModel,
    _autocast_context,
    _build_action_normalizer_meta,
    _build_loader_kwargs,
    _build_lr_scheduler,
    _build_noise_scheduler,
    _cuda_sync_if_needed,
    _get_amp_dtype,
    _infer_wandb_run_id,
    _move_batch_to_device,
    _prepare_device,
    _resolve_optional_project_path,
    _resolve_project_path,
    _set_seed,
    build_contact_policy_common_dataset_kwargs,
)
from contact_policy_probe import (  # noqa: E402
    _PAIRED_TRANSFORMER_VARIANTS,
    _augment_contact_points,
    _build_probe_dataset,
    _build_probe_dataset_common_kwargs,
    _build_probe_geometry_model,
    _build_residual_normalizer_meta,
    _compute_probe_test_metrics,
    _compute_supervision_loss,
    _is_paired_transformer_variant,
    _portable_project_path,
    _resolve_hand_points_fps,
    _resolve_probe_model_variant,
    _resolve_supervision_kwargs,
    _resolve_supervision_mode,
)


from paper_config import runtime_config


def _resolve_manipulator_names(cfg: DictConfig) -> List[str]:
    manipulator_node = getattr(cfg, "co_training", None)
    if manipulator_node is None or getattr(manipulator_node, "manipulators", None) is None:
        raise ValueError(
            "config is missing co_training.manipulators. Point --config-name at a co-training config."
        )
    names = [str(x) for x in manipulator_node.manipulators]
    if len(names) == 0:
        raise ValueError("co_training.manipulators must list at least one manipulator.")
    if len(set(names)) != len(names):
        raise ValueError(f"co_training.manipulators contains duplicates: {names}")
    return names


def _resolve_per_manipulator_action_io(cfg: DictConfig) -> bool:
    """Read ``co_training.per_manipulator_action_io`` (default False for back-compat).

    When true, noisy_xyz_proj / noisy_rot_proj / head_xyz / head_rot and the learned
    ``action_queries`` Parameter are *not* aliased across heads — each manipulator
    keeps its own copies while the backbone (contact encoder, scene encoder, action
    decoder, time MLP, etc.) stays shared.
    """
    return bool(OmegaConf.select(cfg, "co_training.per_manipulator_action_io", default=False))


def _build_per_manipulator_cfg(
    base_cfg: DictConfig,
    *,
    manipulator_name: str,
    overrides_cfg: Any,
) -> DictConfig:
    """Return a deep copy of ``base_cfg`` with ``probe.robot_name`` set to ``manipulator_name``
    and any per-manipulator overrides from ``co_training.overrides`` applied on top of the
    global ``probe`` block."""
    manip_cfg = copy.deepcopy(base_cfg)
    with open_dict(manip_cfg):
        manip_cfg.probe.robot_name = str(manipulator_name)
    if overrides_cfg is None:
        return manip_cfg
    override_entry = OmegaConf.select(overrides_cfg, str(manipulator_name), default=None)
    if override_entry is None:
        return manip_cfg
    with open_dict(manip_cfg):
        for key, value in OmegaConf.to_container(override_entry, resolve=True).items():
            OmegaConf.update(manip_cfg.probe, str(key), value, force_add=True)
    return manip_cfg


def _maybe_materialize_contact_candidate_hparams(
    cfg: DictConfig,
    *,
    manipulator_names: List[str],
    save_dir: str,
) -> None:
    if not bool(OmegaConf.select(cfg, "contact_candidates.materialize_hparams", default=False)):
        return
    default_points = int(OmegaConf.select(cfg, "contact_candidates.num_surface_points", default=256))
    per_manip = OmegaConf.select(cfg, "contact_candidates.per_manipulator", default={})
    overrides_cfg = OmegaConf.select(cfg, "co_training.overrides", default=None)
    robots: Dict[str, Dict[str, Any]] = {}
    for name in manipulator_names:
        if hasattr(per_manip, "get"):
            num_points = int(per_manip.get(str(name), default_points))
        else:
            num_points = int(default_points)
        contact_range = OmegaConf.select(overrides_cfg, f"{name}.contact_count_range", default=None)
        if contact_range is None:
            contact_range = OmegaConf.select(cfg, "probe.contact_count_range", default=[2, 6])
        component_range = OmegaConf.select(overrides_cfg, f"{name}.component_range", default=None)
        if component_range is None:
            component_range = OmegaConf.select(cfg, "probe.component_range", default=[2, 4])
        component_lo, component_hi = int(component_range[0]), int(component_range[1])
        component_values = list(range(component_lo, component_hi + 1))
        component_prob = [1.0 / float(len(component_values))] * len(component_values)
        robots[str(name)] = {
            "robot_model_name": resolve_robot_model_name(str(name)),
            "num_surface_points": int(num_points),
            "point_prob": [1.0 / float(num_points)] * int(num_points),
            "component_range": [component_lo, component_hi],
            "component_count_values": component_values,
            "component_count_distribution": component_prob,
            "contact_count_range": [int(contact_range[0]), int(contact_range[1])],
        }
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "generated_contact_sampler_hparams.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"robots": robots}, f, sort_keys=True)
    with open_dict(cfg):
        cfg.dataset.hparams_path = out_path
        cfg.dataset.real_masks_path = None
        cfg.dataset.check_template_hash = False
    print(f"[contact_candidates] materialized sampler hparams: {out_path}")


def _build_cotrain_shared_model_kwargs(cfg: DictConfig) -> Dict[str, Any]:
    """Assemble the ``shared_kwargs`` dict consumed by ``ContactProbePolicyPairedCoTrain``.

    Mirrors the paired_transformer branch of ``_build_probe_policy`` but excludes the per-robot
    kwargs (``action_dim``, ``robot_names``, normalizer tensors, base-pose indices). Those are
    added per manipulator inside the cotrain wrapper.
    """
    variant = str(getattr(cfg.model, "variant", "paired_transformer_flow_matching")).strip().lower()
    if variant == "paired_transformer":
        paired_mode = "diffusion"
    elif variant.startswith("paired_transformer_"):
        paired_mode = variant[len("paired_transformer_") :]
    else:
        raise ValueError(
            f"Unsupported cfg.model.variant={variant!r} for co-training. Expected a paired_transformer variant."
        )

    cb_cfg = OmegaConf.select(cfg.model, "contact_bottleneck", default=None)
    cb_kwargs: Dict[str, Any] = {}
    if cb_cfg is not None:
        cb_kwargs = {
            "contact_bottleneck_enabled": bool(getattr(cb_cfg, "enabled", False)),
            "contact_bottleneck_num_encoder_layers": int(getattr(cb_cfg, "num_encoder_layers", 2)),
            "contact_bottleneck_unpack_tokens": int(getattr(cb_cfg, "unpack_tokens", 1)),
            "contact_bottleneck_unpack_mlp_ratio": float(getattr(cb_cfg, "unpack_mlp_ratio", 2.0)),
        }
    nc_cfg = OmegaConf.select(cfg.model, "non_contact_context", default=None)
    nc_kwargs: Dict[str, Any] = {}
    if nc_cfg is not None:
        nc_kwargs = {
            "non_contact_context_enabled": bool(getattr(nc_cfg, "enabled", False)),
            "non_contact_context_tokens": int(getattr(nc_cfg, "num_tokens", 4)),
            "non_contact_context_mlp_ratio": float(getattr(nc_cfg, "mlp_ratio", 1.0)),
        }

    shared_kwargs: Dict[str, Any] = {
        "variant": paired_mode,
        "d_model": int(cfg.model.d_model),
        "num_heads": int(cfg.model.num_heads),
        "num_scene_layers": int(cfg.model.num_scene_layers),
        "num_action_layers": int(cfg.model.num_action_layers),
        "mlp_ratio": float(cfg.model.mlp_ratio),
        "hand_points": int(cfg.model.hand_points),
        "use_contact_tokens_in_decoder": bool(cfg.model.use_contact_tokens_in_decoder),
        "step_embed_dim": int(cfg.model.step_embed_dim),
        "num_inference_steps": int(cfg.model.num_inference_steps),
        "flow_time_beta_alpha": float(getattr(cfg.model, "flow_time_beta_alpha", 1.5)),
        "flow_time_beta_beta": float(getattr(cfg.model, "flow_time_beta_beta", 1.0)),
        "action_tokenization": str(getattr(cfg.model, "action_tokenization", "grouped")),
        "use_qref_action_tokens": bool(getattr(cfg.model, "use_qref_action_tokens", False)),
        "action_token_layer_norm": bool(getattr(cfg.model, "action_token_layer_norm", False)),
        "noise_scheduler": (
            _build_noise_scheduler(cfg.model.noise_scheduler)
            if paired_mode == "diffusion" and getattr(cfg.model, "noise_scheduler", None) is not None
            else None
        ),
    }
    shared_kwargs.update(cb_kwargs)
    shared_kwargs.update(nc_kwargs)
    return shared_kwargs


def _compile_head_losses(
    model: ContactProbePolicyPairedCoTrain,
    *,
    enabled: bool,
    device: torch.device,
) -> Optional[Callable[..., Any]]:
    """Capture all head losses together while leaving the model object untouched."""
    if not enabled:
        return None
    if device.type != "cuda":
        print("[compile] CUDA is unavailable; using eager head losses")
        return None

    names = model.manipulator_names

    def compute_losses(batches: Dict[str, Dict[str, Any]]) -> tuple[Any, ...]:
        return tuple(
            model.head_for(name).compute_loss(batches[name], return_aux=True)
            for name in names
        )

    compiled = torch.compile(
        compute_losses,
        backend="cudagraphs",
        dynamic=False,
        fullgraph=True,
    )
    print(
        "[compile] grouped head losses enabled "
        "(backend=cudagraphs, eager kernels preserved; first full batch captures lazily)"
    )
    return compiled


def _cotrain_compute_supervision(
    *,
    model: ContactProbePolicyPairedCoTrain,
    batches_device: Dict[str, Dict[str, Any]],
    geometry_models: Dict[str, Any],
    supervision_mode: str,
    normal_weight: float,
    model_loss_weight: float,
    contact_geometry_loss_weight: float,
    contact_support_geometry_loss_weight: float,
    contact_margin_loss_weight: float,
    contact_margin_m: float,
    contact_space_support_points: int,
    contact_fk_clamp_pred_q: bool,
    contact_fk_base_rotation_mode: str,
    contact_fk_batch_limit: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    variant: str,
    compiled_head_loss_fn: Optional[Callable[..., Any]] = None,
    compile_batch_size: Optional[int] = None,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Average per-manipulator losses; shared parameters collect every head's gradient."""
    compiled_outputs = None
    if (
        compiled_head_loss_fn is not None
        and compile_batch_size is not None
        and all(
            int(batch["action"].shape[0]) == int(compile_batch_size)
            for batch in batches_device.values()
        )
    ):
        with _autocast_context(device, use_amp, amp_dtype):
            compiled_outputs = dict(
                zip(model.manipulator_names, compiled_head_loss_fn(batches_device))
            )

    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    agg_dict: Dict[str, Any] = {}
    for name, batch in batches_device.items():
        head = model.head_for(name)
        geom = geometry_models[name]
        sub_loss, sub_model_loss, sub_ld = _compute_supervision_loss(
            mode=supervision_mode,
            model=head,
            batch=batch,
            geometry_model=geom,
            normal_weight=normal_weight,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            variant=variant,
            model_loss_weight=model_loss_weight,
            contact_geometry_loss_weight=contact_geometry_loss_weight,
            contact_support_geometry_loss_weight=contact_support_geometry_loss_weight,
            contact_margin_loss_weight=contact_margin_loss_weight,
            contact_margin_m=contact_margin_m,
            contact_space_support_points=contact_space_support_points,
            contact_fk_clamp_pred_q=contact_fk_clamp_pred_q,
            contact_fk_base_rotation_mode=contact_fk_base_rotation_mode,
            contact_fk_batch_limit=contact_fk_batch_limit,
            model_output=None if compiled_outputs is None else compiled_outputs[name],
        )
        total_loss = total_loss + sub_loss
        for key, value in sub_ld.items():
            agg_dict[f"{name}/{key}"] = value
        # Keep scalar losses as GPU tensors here; the main loop batches all .item()
        # calls onto log steps to avoid per-step GPU->CPU syncs.
        agg_dict[f"{name}/model_loss"] = sub_model_loss.detach()
    total_loss = total_loss / float(len(batches_device))
    agg_dict["supervision_loss"] = total_loss.detach()
    return total_loss, agg_dict


def _fill_hand_points_on_device(
    batch: Dict[str, Any],
    *,
    geometry_model: Any,
    local_dof: int,
) -> None:
    """Compute ``hand_points_world`` / ``hand_normals_world`` for one device-side batch.

    Used when the dataset is configured with ``hand_points_fk_device="gpu"`` — dataloader
    workers skip the per-fetch CPU FK and the main thread fills the keys in-place here,
    on CUDA, right before the head's forward pass.
    """
    q_tensor = batch["q1_padded"][..., :int(local_dof)].to(
        geometry_model.device, dtype=torch.float32
    )
    pts, nrm = geometry_model.transform_hand_fps_template_batch(q_tensor)
    batch["hand_points_world"] = pts
    batch["hand_normals_world"] = nrm


@hydra.main(version_base="1.2", config_path="conf", config_name="kite")
def main(cfg: DictConfig) -> None:
    cfg = runtime_config(cfg)
    print("******************************** [Config] ********************************")
    print(OmegaConf.to_yaml(cfg))
    print("******************************** [Config] ********************************")

    _set_seed(int(cfg.seed))
    device = _prepare_device(str(cfg.training.device))
    generation_device = _prepare_device(str(cfg.dataset.generation_device))
    use_amp = bool(cfg.training.use_amp)
    amp_dtype = _get_amp_dtype(str(cfg.training.amp_dtype))
    if bool(OmegaConf.select(cfg, "training.detect_anomaly", default=False)):
        torch.autograd.set_detect_anomaly(True)
        print("[debug] torch autograd anomaly detection enabled")

    manipulator_names = _resolve_manipulator_names(cfg)
    overrides_cfg = OmegaConf.select(cfg, "co_training.overrides", default=None)
    probe_variant = _resolve_probe_model_variant(cfg)
    if not _is_paired_transformer_variant(cfg):
        raise ValueError(
            "Co-training is only implemented for paired_transformer variants; "
            f"got cfg.model.variant={probe_variant!r}."
        )
    supervision_mode = _resolve_supervision_mode(cfg)

    save_dir = _resolve_project_path(str(cfg.training.save_dir))
    os.makedirs(save_dir, exist_ok=True)
    _maybe_materialize_contact_candidate_hparams(
        cfg,
        manipulator_names=manipulator_names,
        save_dir=save_dir,
    )

    # Per-manipulator train/test datasets, each with its own buffer directory.
    # Train datasets defer hand-FPS FK to the main thread / GPU (see main loop below).
    # Test datasets keep the default CPU FK so the untouched probe eval helper
    # ``_compute_probe_test_metrics`` still sees ``hand_points_world`` pre-populated.
    per_manip_cfg: Dict[str, DictConfig] = {}
    train_datasets: Dict[str, ContactPolicyProbeDataset] = {}
    test_datasets: Dict[str, ContactPolicyProbeDataset] = {}
    buffer_paths: Dict[str, Dict[str, str]] = {}
    for name in manipulator_names:
        manip_cfg = _build_per_manipulator_cfg(cfg, manipulator_name=name, overrides_cfg=overrides_cfg)
        per_manip_cfg[name] = manip_cfg
        manip_buffer_dir = os.path.join(save_dir, "buffers", name)
        os.makedirs(manip_buffer_dir, exist_ok=True)
        common_kwargs_base = _build_probe_dataset_common_kwargs(manip_cfg)
        train_common_kwargs = dict(common_kwargs_base)
        # Train datasets delegate runtime hand-FPS FK + q1 noise + contact subset to an
        # epoch-start precompute that runs once on the main-thread GPU per epoch. The
        # dataloader fast path then just indexes into shared-memory buffers.
        train_common_kwargs["hand_points_fk_device"] = "gpu"
        train_common_kwargs["epoch_cache_enabled"] = True
        train_common_kwargs["epoch_cache_q1_contact_points"] = supervision_mode in {
            "full_mix_contact_margin",
            "point_normal_margin",
        }
        test_common_kwargs = dict(common_kwargs_base)
        test_common_kwargs["hand_points_fk_device"] = "cpu"
        test_common_kwargs["epoch_cache_enabled"] = False
        train_ds, train_buf = _build_probe_dataset(
            manip_cfg,
            common_kwargs=train_common_kwargs,
            save_dir=manip_buffer_dir,
            split="train",
            generation_device=generation_device,
        )
        test_ds, test_buf = _build_probe_dataset(
            manip_cfg,
            common_kwargs=test_common_kwargs,
            save_dir=manip_buffer_dir,
            split="test",
            generation_device=generation_device,
        )
        train_datasets[name] = train_ds
        test_datasets[name] = test_ds
        buffer_paths[name] = {"train": train_buf, "test": test_buf}

    train_bundle = ContactPolicyCoTrainBundle(datasets=train_datasets)
    # Keep workers only for the hot training path. Test loaders are first consumed after
    # the model, CUDA graph, and large train buffers already exist; forking five more
    # persistent worker pools there duplicates their address-space page tables and can
    # exhaust host RAM. The test sets are small, so evaluate them in the main process.
    train_loader_kwargs = _build_loader_kwargs(
        cfg,
        persistent_workers=bool(int(cfg.dataset.num_workers) > 0),
    )
    test_loaders: Dict[str, DataLoader] = {}
    for name, dataset in test_datasets.items():
        test_loaders[name] = DataLoader(
            dataset,
            batch_size=int(cfg.dataset.test_batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=bool(cfg.dataset.pin_memory),
        )
    # Build lockstep train loaders once; DataLoader(shuffle=True) reshuffles per-epoch
    # automatically when iter() is called, so rebuilding would only respawn workers.
    train_loaders = train_bundle.build_loaders(
        batch_size=int(cfg.dataset.batch_size),
        shuffle=True,
        drop_last=bool(cfg.dataset.drop_last),
        loader_kwargs=train_loader_kwargs,
    )

    # Per-manipulator action + residual normalizers, action_dim, base-pose indices, geometry model.
    manipulator_specs: List[Dict[str, Any]] = []
    residual_meta_by_manip: Dict[str, Dict[str, Any]] = {}
    geometry_models: Dict[str, Any] = {}
    for name in manipulator_names:
        tr = train_datasets[name]
        manip_cfg = per_manip_cfg[name]
        action_meta = _build_action_normalizer_meta(
            manip_cfg,
            robot_specs=tr.robot_specs,
            global_robot_names=[name],
            action_dim=int(tr.action_dim),
        )
        base_translation_indices = [int(i) for i in tr.model.base_translation_indices]
        base_rotation_indices = [int(i) for i in tr.model.base_rotation_indices]
        pose_groups_raw = getattr(tr.model, "pose_groups", None)
        pose_groups = None if pose_groups_raw is None else [
            {
                "name": str(group["name"]),
                "translation_indices": [int(i) for i in group["translation_indices"]],
                "rotation_indices": [int(i) for i in group["rotation_indices"]],
            }
            for group in pose_groups_raw
        ]
        slot = build_residual_slot_indices(
            dof=int(tr.action_dim),
            base_translation_indices=base_translation_indices,
            base_rotation_indices=base_rotation_indices,
            pose_groups=pose_groups,
        )
        residual_meta = _build_residual_normalizer_meta(
            manip_cfg,
            train_dataset=tr,
            slot=slot,
            num_robots=1,
        )
        residual_meta_by_manip[name] = residual_meta
        manipulator_specs.append(
            {
                "name": name,
                "action_dim": int(tr.action_dim),
                "action_normalizer_scale": action_meta["scale"][0].detach().cpu(),
                "action_normalizer_offset": action_meta["offset"][0].detach().cpu(),
                "residual_normalizer_scale": residual_meta["scale"][0].detach().cpu(),
                "residual_normalizer_offset": residual_meta["offset"][0].detach().cpu(),
                "base_translation_indices": base_translation_indices,
                "base_rotation_indices": base_rotation_indices,
                "pose_groups": pose_groups,
            }
        )
        geometry_models[name] = _build_probe_geometry_model(tr, device)
        # When the dataset delegates hand-FPS FK to the main thread we need the same
        # FPS template built on the GPU-side geometry model so the training loop can
        # call ``transform_hand_fps_template_batch`` without surprise.
        if int(tr.hand_points_fps) > 0 and str(getattr(tr, "hand_points_fk_device", "cpu")) == "gpu":
            geometry_models[name].build_hand_fps_template(int(tr.hand_points_fps))
        print(
            f"[manipulator {name}] action_dim={int(tr.action_dim)} "
            f"residual_mode={residual_meta['mode']} "
            f"scale[xyz]={residual_meta['scale'][0, 0:3].tolist()} "
            f"hand_fk_device={getattr(tr, 'hand_points_fk_device', 'cpu')}"
        )

    shared_model_kwargs = _build_cotrain_shared_model_kwargs(cfg)
    per_manipulator_action_io = _resolve_per_manipulator_action_io(cfg)
    print(f"[cotrain] per_manipulator_action_io={per_manipulator_action_io}")
    model = ContactProbePolicyPairedCoTrain(
        manipulators=manipulator_specs,
        shared_kwargs=shared_model_kwargs,
        per_manipulator_action_io=per_manipulator_action_io,
    ).to(device)

    unique_param_count = sum(p.numel() for p in model.parameters())
    total_param_count = sum(
        p.numel() for name in manipulator_names for p in model.head_for(name).parameters()
    )
    print(
        f"[cotrain] manipulators={manipulator_names} unique_params={unique_param_count} "
        f"sum_over_heads={total_param_count} shared_savings={total_param_count - unique_param_count}"
    )

    contact_point_normal_weight = float(getattr(cfg.training, "contact_point_normal_weight", 1.0))
    supervision_kwargs = _resolve_supervision_kwargs(cfg)
    print(
        f"[supervision] mode={supervision_mode} normal_weight={contact_point_normal_weight} "
        f"kwargs={supervision_kwargs}"
    )

    contact_noise_cfg = getattr(cfg, "contact_noise", None)
    contact_noise_enabled = bool(getattr(contact_noise_cfg, "enabled", False))
    contact_noise_pos_scale = float(getattr(contact_noise_cfg, "pos_scale", 0.0))
    contact_noise_normal_scale = float(getattr(contact_noise_cfg, "normal_scale", 0.0))
    if contact_noise_enabled and contact_noise_pos_scale <= 0.0 and contact_noise_normal_scale <= 0.0:
        contact_noise_enabled = False
    print(
        f"[contact_noise] enabled={contact_noise_enabled} "
        f"pos_scale={contact_noise_pos_scale} normal_scale={contact_noise_normal_scale}"
    )

    ema_model: Optional[ContactProbePolicyPairedCoTrain] = None
    ema: Optional[EMAModel] = None
    if bool(cfg.training.use_ema):
        ema_model = copy.deepcopy(model).to(device)
        ema = EMAModel(
            ema_model,
            update_after_step=int(cfg.training.ema.update_after_step),
            inv_gamma=float(cfg.training.ema.inv_gamma),
            power=float(cfg.training.ema.power),
            min_value=float(cfg.training.ema.min_value),
            max_value=float(cfg.training.ema.max_value),
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        betas=tuple(float(x) for x in cfg.training.betas),
        eps=float(cfg.training.eps),
        weight_decay=float(cfg.training.weight_decay),
    )
    steps_per_epoch = train_bundle.steps_per_epoch(
        batch_size=int(cfg.dataset.batch_size),
        drop_last=bool(cfg.dataset.drop_last),
    )
    total_steps = steps_per_epoch * int(cfg.training.epochs)
    lr_scheduler = _build_lr_scheduler(
        optimizer=optimizer,
        name=str(cfg.training.lr_scheduler),
        num_warmup_steps=int(cfg.training.lr_warmup_steps),
        num_training_steps=int(max(1, total_steps)),
    )

    start_epoch = 1
    global_step = 0
    resume_wandb_run_id: Optional[str] = None
    if bool(getattr(cfg.training, "resume", False)):
        resume_ckpt_path = _resolve_optional_project_path(getattr(cfg.training, "resume_checkpoint_path", None))
        if resume_ckpt_path is None:
            resume_ckpt_path = os.path.join(save_dir, "last.pt")
        if not os.path.isfile(resume_ckpt_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt_path}")
        resume_ckpt = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        ckpt_manipulators = [str(x) for x in resume_ckpt.get("manipulators", [])]
        if ckpt_manipulators != list(manipulator_names):
            raise ValueError(
                f"Resume checkpoint manipulators={ckpt_manipulators} do not match "
                f"current co_training.manipulators={list(manipulator_names)}. Refusing to load."
            )
        model.load_state_dict(resume_ckpt["model"], strict=True)
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        lr_scheduler.load_state_dict(resume_ckpt["lr_scheduler"])
        if ema_model is not None:
            if "ema_model" not in resume_ckpt:
                raise KeyError("Resume checkpoint is missing ema_model but training.use_ema=true")
            ema_model.load_state_dict(resume_ckpt["ema_model"], strict=True)
            if ema is not None and "ema_state" in resume_ckpt:
                ema.load_state_dict(resume_ckpt["ema_state"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
        resume_wandb_run_id = None if cfg.wandb.run_id is None else str(cfg.wandb.run_id)
        if resume_wandb_run_id is None:
            resume_wandb_run_id = resume_ckpt.get("wandb_run_id")
        if resume_wandb_run_id is None:
            resume_wandb_run_id = _infer_wandb_run_id(save_dir)
        print(
            f"Resuming co-train from checkpoint={resume_ckpt_path} epoch={start_epoch} global_step={global_step}"
        )

    compiled_head_loss_fn = _compile_head_losses(
        model,
        enabled=bool(OmegaConf.select(cfg, "training.compile_heads", default=False)),
        device=device,
    )

    run_name = str(cfg.wandb.name) if cfg.wandb.name is not None else str(cfg.experiment.name)
    if bool(cfg.wandb.enabled):
        wandb_init_kwargs = {
            "project": str(cfg.wandb.project),
            "entity": None if cfg.wandb.entity is None else str(cfg.wandb.entity),
            "name": run_name,
            "mode": str(cfg.wandb.mode),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        if bool(getattr(cfg.training, "resume", False)) and resume_wandb_run_id is not None:
            wandb_init_kwargs["id"] = str(resume_wandb_run_id)
            wandb_init_kwargs["resume"] = str(getattr(cfg.wandb, "resume", "allow"))
        wandb.init(**wandb_init_kwargs)
    else:
        wandb.init(mode="disabled")

    active_wandb_run_id: Optional[str] = None
    if bool(cfg.wandb.enabled) and getattr(wandb, "run", None) is not None:
        active_wandb_run_id = str(wandb.run.id)
        with open(os.path.join(save_dir, "wandb_run_id.txt"), "w", encoding="utf-8") as f:
            f.write(active_wandb_run_id)

    # Dataset / normalizer metadata keyed by manipulator name, so viewers/checkpoints can
    # reconstruct the exact co-training setup without re-deriving anything from config.
    per_manip_meta: Dict[str, Dict[str, Any]] = {}
    for name in manipulator_names:
        tr = train_datasets[name]
        manip_cfg = per_manip_cfg[name]
        spec = next(s for s in manipulator_specs if s["name"] == name)
        rm = residual_meta_by_manip[name]
        per_manip_meta[name] = {
            "contact_mode": str(manip_cfg.probe.contact_mode),
            "sixd_pose_mode": str(manip_cfg.probe.sixd_pose_mode),
            "robot_name": name,
            "action_dim": int(tr.action_dim),
            "surface_num_points": int(tr.surface_num_points),
            "max_contact_points": int(tr.candidate_contact_points),
            "candidate_contact_points": int(tr.candidate_contact_points),
            "contact_input_points": int(tr.contact_input_points),
            "non_contact_input_points": int(tr.non_contact_input_points),
            "non_contact_sampling_radius": float(tr.non_contact_sampling_radius),
            "non_contact_sdf_clearance": float(tr.non_contact_sdf_clearance),
            "non_contact_pattern_mode": str(tr.non_contact_pattern_mode),
            "non_contact_component_range": list(tr.non_contact_component_range),
            "non_contact_candidate_points": int(getattr(tr, "non_contact_candidate_points", 0)),
            "non_contact_collision_surface_points": int(getattr(tr, "non_contact_collision_surface_points", 0)),
            "non_contact_epoch_batch_size": int(getattr(tr, "non_contact_epoch_batch_size", 0)),
            "remove_non_contact_points": bool(tr.remove_non_contact_points),
            "non_contact_causal_enabled": bool(getattr(tr, "non_contact_causal_enabled", False)),
            "non_contact_empty_probability": float(getattr(tr, "non_contact_empty_probability", 0.0)),
            "non_contact_goal_joint_noise_scale": float(getattr(tr, "non_contact_goal_joint_noise_scale", 0.0)),
            "non_contact_goal_joint_select_min": int(getattr(tr, "non_contact_goal_joint_select_min", 0)),
            "non_contact_goal_joint_select_max": int(getattr(tr, "non_contact_goal_joint_select_max", -1)),
            "non_contact_blocker_fraction": float(getattr(tr, "non_contact_blocker_fraction", 0.0)),
            "non_contact_blocker_candidate_points": int(
                getattr(tr, "non_contact_blocker_candidate_points", 128)
            ),
            "collision_primitive_spec_dir": str(getattr(tr, "collision_primitive_spec_dir", "")),
            "collision_primitive_spec_path": (
                None
                if getattr(tr, "collision_primitive_spec_path", None) is None
                else _portable_project_path(str(getattr(tr, "collision_primitive_spec_path")))
            ),
            "collision_primitive_fit_method": getattr(tr, "collision_primitive_fit_method", None),
            "contact_cluster_mode": str(tr.contact_cluster_mode),
            "q1_translation_noise_std": float(tr.q1_translation_noise_std),
            "q1_rotation_noise_std_deg": float(tr.q1_rotation_noise_std_deg),
            "q1_joint_noise_scale": float(tr.q1_joint_noise_scale),
            "q1_zero_probability": float(tr.q1_zero_probability),
            "random_q1": bool(tr.random_q1),
            "q1_sampling_mode": str(getattr(tr, "q1_sampling_mode", "gaussian")),
            "q1_contact_error_zero_prob": float(getattr(tr, "q1_contact_error_zero_prob", 0.08)),
            "q1_contact_error_bins_mm": [list(x) for x in getattr(tr, "q1_contact_error_bins_mm", [])],
            "q1_contact_error_inactive_scale": float(getattr(tr, "q1_contact_error_inactive_scale", 0.05)),
            "q1_contact_error_refine_steps": int(getattr(tr, "q1_contact_error_refine_steps", 2)),
            "sparse_sampling_mode": str(getattr(tr, "sparse_sampling_mode", "point_uniform")),
            "action_normalizer_scale": spec["action_normalizer_scale"].view(1, -1).detach().cpu(),
            "action_normalizer_offset": spec["action_normalizer_offset"].view(1, -1).detach().cpu(),
            "robot_model_kwargs": dict(getattr(tr, "robot_model_kwargs", {})),
            "train_buffer_path": _portable_project_path(buffer_paths[name]["train"]),
            "test_buffer_path": _portable_project_path(buffer_paths[name]["test"]),
            "base_translation_indices": list(spec["base_translation_indices"]),
            "base_rotation_indices": list(spec["base_rotation_indices"]),
            "pose_groups": spec.get("pose_groups"),
            "composite_synthesis_mode": str(getattr(tr, "composite_synthesis_mode", "independent")),
            "composite_nearby_max_distance": float(getattr(tr, "composite_nearby_max_distance", 0.12)),
            "composite_contact_count_ranges": {
                str(k): [int(v[0]), int(v[1])]
                for k, v in getattr(tr, "composite_contact_count_ranges", {}).items()
            },
            "hand_points_fps": _resolve_hand_points_fps(cfg),
            "residual_normalizer_mode": str(rm["mode"]),
            "residual_normalizer_scale": spec["residual_normalizer_scale"].view(1, -1).detach().cpu(),
            "residual_normalizer_offset": spec["residual_normalizer_offset"].view(1, -1).detach().cpu(),
        }
        if getattr(tr.model, "asset_provenance", None) is not None:
            per_manip_meta[name]["asset_provenance"] = tr.model.asset_provenance
            per_manip_meta[name]["surface_template_hash"] = tr.model.get_surface_template_hash()

    torch.save(
        {
            "mode": "cotrain",
            "manipulators": list(manipulator_names),
            "model_variant": probe_variant,
            "per_manipulator": per_manip_meta,
        },
        os.path.join(save_dir, "dataset_meta.pt"),
    )

    log_every_n_steps = int(max(1, cfg.training.log_every_n_steps))
    eval_every_n_epochs = int(max(1, cfg.training.eval_every_n_epochs))
    save_every_n_epoch = int(max(1, cfg.training.save_every_n_epoch))

    if start_epoch > int(cfg.training.epochs):
        print(
            f"Checkpoint epoch {start_epoch - 1} already reaches/exceeds configured training.epochs="
            f"{int(cfg.training.epochs)}. Nothing to do."
        )
        wandb.finish()
        return

    last_synced_loss: float = float("nan")
    for epoch in range(start_epoch, int(cfg.training.epochs) + 1):
        model.train()
        # Track epoch mean as a GPU running sum/count so per-step wall cost stays sync-free.
        epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        epoch_loss_count = torch.zeros((), device=device, dtype=torch.float32)

        # Epoch-start precompute: refresh per-sample q1 / action / contact subset /
        # hand-FPS FK on GPU in bulk and publish into each train dataset's shared-memory
        # runtime buffers. DataLoader workers then take the fast path in __getitems__.
        import time as _time_mod
        for name, tr_ds in train_datasets.items():
            if getattr(tr_ds, "epoch_cache_enabled", False):
                t0 = _time_mod.perf_counter()
                tr_ds.prepare_epoch_precomputed(epoch=int(epoch), device=device, chunk_size=32768)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                print(
                    f"[precompute] {name} epoch={epoch} elapsed="
                    f"{_time_mod.perf_counter() - t0:.1f}s"
                )

        progress = tqdm(total=steps_per_epoch, desc=f"cotrain:{epoch}")
        for batch_idx, batches in enumerate(iter_lockstep(train_loaders)):
            batches_device: Dict[str, Dict[str, Any]] = {
                name: _move_batch_to_device(batch, device) for name, batch in batches.items()
            }
            # When a train dataset did NOT opt into the epoch cache but still uses
            # ``hand_points_fk_device="gpu"`` (e.g., a mid-run debugging config), fill hand
            # tokens here per-step. The epoch-cache datasets already carry those keys.
            for name, bd in batches_device.items():
                tr_ds = train_datasets[name]
                if (
                    int(tr_ds.hand_points_fps) <= 0
                    or str(tr_ds.hand_points_fk_device) != "gpu"
                    or getattr(tr_ds, "epoch_cache_enabled", False)
                ):
                    continue
                _fill_hand_points_on_device(
                    bd,
                    geometry_model=geometry_models[name],
                    local_dof=int(tr_ds.local_dof),
                )
            if contact_noise_enabled:
                for bd in batches_device.values():
                    _augment_contact_points(
                        bd,
                        pos_scale=contact_noise_pos_scale,
                        normal_scale=contact_noise_normal_scale,
                    )

            optimizer.zero_grad(set_to_none=True)
            total_loss, agg_loss_dict = _cotrain_compute_supervision(
                model=model,
                batches_device=batches_device,
                geometry_models=geometry_models,
                supervision_mode=supervision_mode,
                normal_weight=contact_point_normal_weight,
                **supervision_kwargs,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                variant=probe_variant,
                compiled_head_loss_fn=compiled_head_loss_fn,
                compile_batch_size=int(cfg.dataset.batch_size),
            )
            if not torch.isfinite(total_loss.detach()).all():
                resolved_agg = {
                    k: (float(v.detach().item()) if isinstance(v, torch.Tensor) else float(v))
                    for k, v in agg_loss_dict.items()
                }
                raise FloatingPointError(f"Non-finite training loss before backward: {resolved_agg}")
            total_loss.backward()
            if float(cfg.training.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(cfg.training.grad_clip_norm),
                    error_if_nonfinite=True,
                )
            optimizer.step()
            lr_scheduler.step()
            if ema is not None:
                ema.step(model)

            # Accumulate epoch loss on GPU; no sync.
            epoch_loss_sum = epoch_loss_sum + total_loss.detach()
            epoch_loss_count = epoch_loss_count + 1.0
            should_log_step = (global_step % log_every_n_steps == 0) or (batch_idx + 1 == steps_per_epoch)
            if should_log_step:
                # Single sync burst resolves total_loss plus any tensor-valued entries
                # in agg_loss_dict. Non-log steps skip the sync entirely.
                loss_value = float(total_loss.detach().item())
                last_synced_loss = loss_value
                resolved_agg: Dict[str, float] = {}
                for k, v in agg_loss_dict.items():
                    resolved_agg[k] = float(v.item()) if isinstance(v, torch.Tensor) else float(v)
                step_log = {
                    "train/loss": loss_value,
                    "train/lr": float(lr_scheduler.get_last_lr()[0]),
                    "global_step": global_step,
                    "epoch": epoch,
                }
                step_log.update({f"train/{k}": v for k, v in resolved_agg.items()})
                wandb.log(step_log, step=global_step)

            progress.set_postfix(loss=f"{last_synced_loss:.4f}")
            global_step += 1
            progress.update(1)
        progress.close()

        # Single end-of-epoch sync: resolve the running GPU sum/count into a Python float.
        num_steps_this_epoch = float(epoch_loss_count.item())
        epoch_mean_loss = (
            float((epoch_loss_sum / epoch_loss_count.clamp_min(1.0)).item())
            if num_steps_this_epoch > 0
            else float("nan")
        )
        epoch_log: Dict[str, Any] = {
            "epoch": epoch,
            "train/loss_epoch": epoch_mean_loss,
        }

        run_test = (epoch % eval_every_n_epochs == 0) or (epoch == int(cfg.training.epochs))
        if run_test:
            eval_policy = ema_model if ema_model is not None else model
            for name in manipulator_names:
                head = eval_policy.head_for(name)
                test_metrics = _compute_probe_test_metrics(
                    head,
                    test_loaders[name],
                    geometry_model=geometry_models[name],
                    device=device,
                    prefix=f"test/{name}",
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    variant=probe_variant,
                    compute_geometric_loss=True,
                )
                epoch_log.update(test_metrics)

                # Mirror the test-set evaluation on a same-sized train slice for each manipulator.
                train_eval_size = min(len(test_datasets[name]), len(train_datasets[name]))
                if train_eval_size > 0:
                    subset_gen = torch.Generator().manual_seed(int(cfg.seed) + epoch)
                    subset_indices = torch.randperm(
                        len(train_datasets[name]), generator=subset_gen
                    )[:train_eval_size].tolist()
                    train_eval_loader = DataLoader(
                        Subset(train_datasets[name], subset_indices),
                        batch_size=int(cfg.dataset.test_batch_size),
                        shuffle=False,
                        drop_last=False,
                        num_workers=0,
                        pin_memory=bool(cfg.dataset.pin_memory),
                    )
                    train_eval_metrics = _compute_probe_test_metrics(
                        head,
                        train_eval_loader,
                        geometry_model=geometry_models[name],
                        device=device,
                        prefix=f"train_eval/{name}",
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        variant=probe_variant,
                        compute_geometric_loss=True,
                    )
                    epoch_log.update(train_eval_metrics)

        wandb.log(epoch_log, step=global_step)

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "wandb_run_id": active_wandb_run_id,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "manipulators": list(manipulator_names),
        }
        if ema_model is not None:
            ckpt["ema_model"] = ema_model.state_dict()
        if ema is not None:
            ckpt["ema_state"] = ema.state_dict()

        torch.save(ckpt, os.path.join(save_dir, "last.pt"))
        if epoch % save_every_n_epoch == 0:
            torch.save(ckpt, os.path.join(save_dir, f"epoch_{epoch}.pt"))

        summary = f"[Epoch {epoch}] train_loss={epoch_log['train/loss_epoch']:.6f}"
        for name in manipulator_names:
            key = f"test/{name}/action_mse_valid"
            if key in epoch_log:
                summary += f" {name}_test_valid={epoch_log[key]:.6f}"
        print(summary)

    wandb.finish()


if __name__ == "__main__":
    from paper_config import canonicalize_cli
    canonicalize_cli()
    torch.set_float32_matmul_precision("high")
    main()


