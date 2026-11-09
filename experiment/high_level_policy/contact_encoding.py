from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
CROSS_DIR = EXPERIMENT_DIR.parent
_PAIRED_TRANSFORMER_LATENT_VARIANTS = {"paired_transformer_mlp"}


from paper_config import runtime_config


def _resolve_cross_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    text = str(path_str).strip()
    if text == "" or text.lower() == "null":
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (CROSS_DIR / path).resolve()


def _resolve_experiment_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    text = str(path_str).strip()
    if text == "" or text.lower() == "null":
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (EXPERIMENT_DIR / path).resolve()


def _load_latent_probe_config(config_path: Path):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    resolved_path = Path(config_path).expanduser().resolve()
    candidates: list[tuple[Path, str]] = []
    for root in (CROSS_DIR / "conf", EXPERIMENT_DIR / "conf"):
        try:
            rel = resolved_path.relative_to(root.resolve())
        except ValueError:
            continue
        candidates.append((root.resolve(), rel.with_suffix("").as_posix()))
    last_error: Exception | None = None

    def _try_compose(config_dir: Path, config_name: str):
        try:
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            with initialize_config_dir(version_base="1.2", config_dir=str(config_dir)):
                cfg = runtime_config(compose(config_name=config_name))
            GlobalHydra.instance().clear()
            OmegaConf.resolve(cfg)
            return cfg
        except Exception as exc:
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            raise exc

    for config_dir, config_name in candidates:
        try:
            return _try_compose(config_dir, config_name)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to compose Hydra config {resolved_path} from candidates "
        f"{[(str(root), name) for root, name in candidates]}"
    ) from last_error


def _load_latent_probe_config_from_args(args: argparse.Namespace):
    config_path = _resolve_cross_path(args.latent_contact_policy_config)
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(f"Latent contact-probe config not found: {config_path}")
    return _load_latent_probe_config(config_path)


def _apply_contact_preprocess_config(args: argparse.Namespace, cfg) -> None:
    # Extraction settings are independent of the frozen decoder's training recipe.
    args.contact_count = int(cfg.probe.contact_input_points)
    args.contact_threshold = float(getattr(args, "contact_threshold", 0.05))
    args.align_exp_scale = float(getattr(args, "align_exp_scale", 2.0))
    args.sigmoid_scale = float(getattr(args, "sigmoid_scale", 10.0))
    if not 0 < args.contact_threshold < 1:
        raise ValueError("Contact threshold must be between zero and one")
    if not all(
        np.isfinite(value) and value > 0
        for value in (args.align_exp_scale, args.sigmoid_scale)
    ):
        raise ValueError("Distance and normal scoring scales must be positive and finite")

    if args.contact_mode != "trajectory_sparse":
        raise ValueError(
            "KITE demonstration extraction requires contact_mode=trajectory_sparse"
        )
    if _resolve_encoder_backend(cfg) != "paired_transformer":
        raise ValueError(
            "Paper contact extraction requires the paired Transformer encoder"
        )
    args.contact_mode_resolved = "trajectory_sparse"
    args.contact_sparse_capacity_resolved = 6
    print(
        "Contact preprocessing: paper trajectory selection, "
        f"threshold={args.contact_threshold:g}, align_exp_scale={args.align_exp_scale:g}, "
        f"sigmoid_scale={args.sigmoid_scale:g}, maximum_slots=6",
        flush=True,
    )


def _resolve_encoder_backend(cfg):
    if str(cfg.model.variant) != "paired_transformer_mlp":
        raise ValueError("KITE requires paired_transformer_mlp.")
    return "paired_transformer"


class _PairedBottleneckEncoderAdapter(torch.nn.Module):
    # Wrap the lifted `_ContactOnlyEncoder` so it presents the same call signature as the
    # PointNet encoder: __call__(points, point_mask=mask) → (B, latent_dim). The encoder's
    # native API is keyword-only and split into (contact_feat_in, contact_xyz, contact_mask),
    # which we hide here so `_encode_contact_latent_sequence` doesn't need to branch on backend.
    def __init__(self, contact_only_encoder: torch.nn.Module) -> None:
        super().__init__()
        self.contact_only_encoder = contact_only_encoder

    def forward(
        self, points: torch.Tensor, point_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if points.ndim != 3 or int(points.shape[-1]) < 6:
            raise ValueError(
                f"Expected contact points (B, N, ≥6), got {tuple(points.shape)}"
            )
        contact_feat_in = points[..., :6]
        contact_xyz = points[..., :3]
        if point_mask is None:
            point_mask = torch.ones(
                points.shape[:2], dtype=torch.bool, device=points.device
            )
        return self.contact_only_encoder(
            contact_feat_in=contact_feat_in,
            contact_xyz=contact_xyz,
            contact_mask=point_mask.to(dtype=torch.bool),
        )


def _load_cotrain_probe_policy(
    cfg,
    *,
    checkpoint_path: str,
    device: torch.device,
    prefer_ema: bool,
    cotrain_meta: dict,
) -> tuple[object, str, object]:
    from omegaconf import OmegaConf

    from networks.contact_probe_policy_paired_cotrain import (
        ContactProbePolicyPairedCoTrain,
    )
    from train_kinematic_decoder import _build_cotrain_shared_model_kwargs

    resolved_ckpt_path = str(checkpoint_path)
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
    if len(manipulator_names) == 0:
        raise ValueError(
            f"Cotrain dataset_meta.pt at {resolved_ckpt_path} has no manipulators listed; "
            "checkpoint may be corrupted."
        )

    manipulator_specs: list[dict] = []
    for name in manipulator_names:
        entry = per_manip.get(name)
        if entry is None:
            raise ValueError(
                f"Cotrain dataset_meta.pt is missing per_manipulator[{name!r}]; checkpoint may be corrupted."
            )
        spec = {
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
            "base_translation_indices": [
                int(x) for x in entry["base_translation_indices"]
            ],
            "base_rotation_indices": [int(x) for x in entry["base_rotation_indices"]],
        }
        if entry.get("pose_groups") is not None:
            spec["pose_groups"] = copy.deepcopy(entry["pose_groups"])
        manipulator_specs.append(spec)

    shared_kwargs = _build_cotrain_shared_model_kwargs(build_cfg)
    per_manipulator_action_io = bool(
        OmegaConf.select(
            build_cfg, "co_training.per_manipulator_action_io", default=False
        )
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
    return model, resolved_ckpt_path, build_cfg


def _build_paired_transformer_latent_encoder(
    args: argparse.Namespace, cfg
) -> dict[str, object]:
    from omegaconf import OmegaConf

    from contact_policy_probe import load_contact_policy_probe_checkpoint

    requested_device = str(args.latent_encoder_device).strip().lower()
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    checkpoint_path = _resolve_cross_path(args.latent_contact_encoder_checkpoint)
    if checkpoint_path is None:
        save_dir = _resolve_cross_path(str(cfg.training.save_dir))
        checkpoint_path = None if save_dir is None else (save_dir / "last.pt").resolve()
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Paired-transformer latent encoder requires a trained checkpoint, not found at: {checkpoint_path}. "
            "Pass --latent-contact-encoder-checkpoint or set cfg.training.save_dir to a directory containing last.pt."
        )

    # Peek at the sibling dataset_meta.pt to detect whether this is a cotrain checkpoint.
    meta_path = Path(str(checkpoint_path)).parent / "dataset_meta.pt"
    cotrain_meta: dict | None = None
    if meta_path.is_file():
        raw_meta = torch.load(str(meta_path), map_location="cpu", weights_only=False)
        if (
            isinstance(raw_meta, dict)
            and str(raw_meta.get("mode", "")).lower() == "cotrain"
        ):
            cotrain_meta = raw_meta

    if cotrain_meta is not None:
        policy, resolved_ckpt_path, build_cfg = _load_cotrain_probe_policy(
            cfg,
            checkpoint_path=str(checkpoint_path),
            device=device,
            prefer_ema=bool(args.latent_prefer_ema),
            cotrain_meta=cotrain_meta,
        )
        # The cotrain wrapper aliases contact_only_encoder across all heads, so reading it
        # from the first head gives the shared weights. Manipulator identity is irrelevant
        # here — the encoder is embodiment-agnostic.
        manipulator_names = [str(x) for x in cotrain_meta.get("manipulators", [])]
        head = policy.head_for(manipulator_names[0])
        if not bool(getattr(head, "contact_bottleneck_enabled", False)):
            raise ValueError(
                f"Cotrain checkpoint at {resolved_ckpt_path} was not trained with "
                "model.contact_bottleneck.enabled=true; the latent pipeline cannot lift its contact-only encoder."
            )
        if getattr(head, "contact_only_encoder", None) is None:
            raise RuntimeError(
                f"Cotrain checkpoint at {resolved_ckpt_path} reports the bottleneck enabled but exposes no "
                "contact_only_encoder module — checkpoint may be corrupted."
            )
        contact_only_encoder = head.contact_only_encoder
        cls_unpacker = getattr(head, "cls_unpacker", None)
        variant_label = str(getattr(head, "variant", ""))
        d_model = int(getattr(head, "d_model"))
    else:
        policy, resolved_ckpt_path, _, build_cfg = load_contact_policy_probe_checkpoint(
            cfg,
            device=device,
            ckpt_path=str(checkpoint_path),
            prefer_ema=bool(args.latent_prefer_ema),
            return_build_cfg=True,
        )
        if not bool(getattr(policy, "contact_bottleneck_enabled", False)):
            raise ValueError(
                f"Paired-transformer checkpoint at {resolved_ckpt_path} was not trained with "
                "model.contact_bottleneck.enabled=true; the latent pipeline cannot lift its contact-only encoder."
            )
        if getattr(policy, "contact_only_encoder", None) is None:
            raise RuntimeError(
                f"Paired-transformer checkpoint at {resolved_ckpt_path} reports the bottleneck enabled but exposes "
                "no contact_only_encoder module — checkpoint may be corrupted."
            )
        contact_only_encoder = policy.contact_only_encoder
        cls_unpacker = getattr(policy, "cls_unpacker", None)
        variant_label = str(getattr(policy, "variant", ""))
        d_model = int(getattr(policy, "d_model"))

    encoder = _PairedBottleneckEncoderAdapter(contact_only_encoder).to(device)
    encoder.eval()
    latent_dim = int(d_model)

    use_xyz_translation = bool(args.latent_use_xyz_translation)
    cfg_pose_mode = str(getattr(build_cfg.probe, "sixd_pose_mode", "")).strip().lower()
    if use_xyz_translation and cfg_pose_mode != "zero_centered":
        print(
            "WARNING: policy.latent_use_xyz_translation=true, but paired-transformer checkpoint trained with "
            f"probe.sixd_pose_mode={cfg_pose_mode}. The bottleneck encoder will still produce its CLS latent.",
            flush=True,
        )
    if not use_xyz_translation and cfg_pose_mode == "zero_centered":
        print(
            "WARNING: policy.latent_use_xyz_translation=false, but paired-transformer checkpoint trained with "
            "probe.sixd_pose_mode=zero_centered. Latents will still be produced from raw contacts.",
            flush=True,
        )
    unpack_tokens = (
        int(getattr(cls_unpacker, "num_tokens", -1)) if cls_unpacker is not None else -1
    )
    source_label = "cotrain" if cotrain_meta is not None else "single_manipulator"
    print(
        f"[latent_encoder] paired_transformer backend loaded: variant={variant_label} "
        f"d_model={latent_dim} unpack_tokens={unpack_tokens} source={source_label}",
        flush=True,
    )
    return {
        "encoder": encoder,
        "device": device,
        "checkpoint_path": str(resolved_ckpt_path),
        "loaded": True,
        "config": OmegaConf.to_container(build_cfg, resolve=True),
        "latent_dim": int(latent_dim),
        "shift_dim": 3 if use_xyz_translation else 0,
        "zero_centered": use_xyz_translation,
        "backend": "paired_transformer",
        "cotrain_checkpoint": bool(cotrain_meta is not None),
    }


def _build_latent_contact_encoder(args, cfg):
    return _build_paired_transformer_latent_encoder(args, cfg)


@torch.inference_mode()
def _encode_contact_latent_sequence(
    *,
    contact_cloud: np.ndarray,
    latent_encoder: dict[str, object],
    batch_size: int,
    zero_centered: bool,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    cloud = np.asarray(contact_cloud, dtype=np.float32)
    if cloud.ndim != 3 or int(cloud.shape[-1]) != 6:
        raise ValueError(
            f"Expected contact cloud with shape (T, N, 6), got {cloud.shape}"
        )
    batch_size = max(1, int(batch_size))

    if valid_mask is None:
        mask_np = np.ones(cloud.shape[:2], dtype=bool)
    else:
        mask_np = np.asarray(valid_mask, dtype=bool)
        if mask_np.shape != cloud.shape[:2]:
            raise ValueError(
                f"valid_mask shape {mask_np.shape} does not match contact cloud shape {cloud.shape[:2]}"
            )

    mask_f = mask_np.astype(np.float32)
    valid_counts = mask_f.sum(axis=1, keepdims=True)
    safe_counts = np.clip(valid_counts, 1.0, None)
    xyz_sum = (cloud[:, :, :3] * mask_f[:, :, None]).sum(axis=1)
    shift = (xyz_sum / safe_counts).astype(np.float32)

    encoded_cloud = cloud.copy()
    if bool(zero_centered):
        # Only shift valid slots; keep padded zeros at their zeroed state (they are masked out
        # in self-attention anyway, so their xyz does not affect the CLS output).
        shift_per_slot = (mask_f[:, :, None] * shift[:, None, :]).astype(np.float32)
        encoded_cloud[:, :, :3] -= shift_per_slot

    encoder = latent_encoder["encoder"]
    device = latent_encoder["device"]
    latents = []
    for start in range(0, int(encoded_cloud.shape[0]), batch_size):
        end = min(int(encoded_cloud.shape[0]), start + batch_size)
        points = torch.from_numpy(encoded_cloud[start:end]).to(
            device=device, dtype=torch.float32
        )
        point_mask = torch.from_numpy(mask_np[start:end]).to(
            device=device, dtype=torch.bool
        )
        latents.append(
            encoder(points, point_mask=point_mask)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
    latent = np.concatenate(latents, axis=0)
    if bool(zero_centered):
        return np.concatenate([shift, latent], axis=-1).astype(np.float32, copy=False)
    return latent.astype(np.float32, copy=False)
