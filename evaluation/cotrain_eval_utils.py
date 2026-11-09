"""Configuration and synthetic-dataset helpers shared by the kinematic decoder viewers."""

from __future__ import annotations

import copy
import os
import sys
import tempfile
from typing import Any, Dict

import numpy as np
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Offset applied when synthesizing the fresh "eval test" buffer. Distinct from the base
# seed (train) and from base + 10_007 (the usual probe test-split seed).
FRESH_TEST_SEED_OFFSET: int = 60_007


# ---------------------------------------------------------------------------
# Hydra config
# ---------------------------------------------------------------------------


def clear_hydra() -> None:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(ROOT_DIR, path))


def apply_tight_q2_translation(cfg: DictConfig, *, lo: float = -0.1, hi: float = 0.1) -> None:
    """Override ``cfg.dataset.q2_base_translation_{min,max}`` in place (tight range).

    Every downstream dataset construction reads these fields, so this must be called
    before ``build_train_datasets_from_buffers`` / ``build_fresh_test_datasets``.
    """
    with open_dict(cfg):
        cfg.dataset.q2_base_translation_min = [float(lo)] * 3
        cfg.dataset.q2_base_translation_max = [float(hi)] * 3


def apply_contact_candidate_count_override(
    cfg: DictConfig,
    *,
    output_dir: str | None = None,
) -> str | None:
    """Materialize the sampler hparams implied by ``contact_candidates``.

    ``dataset.max_contact_points`` controls the fixed contact-cloud slot capacity. The
    number of possible sparse contact candidates is the robot surface-template
    resolution, normally read from ``dataset.hparams_path``. This mirrors the trainer's
    ``_maybe_materialize_contact_candidate_hparams`` path so post-training tools load
    buffers with the same derived hparams that were used during training.
    """
    materialize_flag = OmegaConf.select(cfg, "contact_candidates.materialize_hparams", default=None)
    num_points_val = OmegaConf.select(cfg, "contact_candidates.num_surface_points", default=None)
    if materialize_flag is False:
        return None
    if num_points_val is None:
        if materialize_flag is True:
            raise ValueError("contact_candidates.materialize_hparams=true requires contact_candidates.num_surface_points")
        return None
    default_num_points = int(num_points_val)
    if default_num_points <= 0:
        raise ValueError(
            f"contact_candidates.num_surface_points must be positive, got {default_num_points}"
        )

    import yaml
    from robot_model.robot_model import resolve_robot_model_name

    manipulator_names = [str(x) for x in cfg.co_training.manipulators]
    per_manipulator_points = OmegaConf.select(
        cfg, "contact_candidates.per_manipulator", default={}
    )
    per_manipulator_points = OmegaConf.to_container(per_manipulator_points, resolve=True)
    if per_manipulator_points is None:
        per_manipulator_points = {}
    overrides_cfg = OmegaConf.select(cfg, "co_training.overrides", default=None)
    robots: Dict[str, Dict[str, Any]] = {}
    for name in manipulator_names:
        robot_model_name = resolve_robot_model_name(str(name))
        num_points = int(per_manipulator_points.get(str(name), default_num_points))
        if num_points <= 0:
            raise ValueError(f"{name}: contact candidate point count must be positive, got {num_points}")
        if str(robot_model_name).startswith("mano/") and num_points > 778:
            num_points = 778
        component_range = OmegaConf.select(overrides_cfg, f"{name}.component_range", default=None)
        if component_range is None:
            component_range = OmegaConf.select(cfg, "probe.component_range", default=[2, 4])
        component_lo, component_hi = int(component_range[0]), int(component_range[1])
        component_values = list(range(component_lo, component_hi + 1))
        component_distribution = [1.0 / float(len(component_values))] * len(component_values)
        contact_count_range = OmegaConf.select(overrides_cfg, f"{name}.contact_count_range", default=None)
        if contact_count_range is None:
            contact_count_range = OmegaConf.select(cfg, "probe.contact_count_range", default=[2, 6])
        point_prob = [1.0 / float(num_points)] * int(num_points)
        entry: Dict[str, Any] = {
            "robot_model_name": robot_model_name,
            "num_surface_points": int(num_points),
            "point_prob": point_prob,
            "component_range": [component_lo, component_hi],
            "component_count_values": component_values,
            "component_count_distribution": component_distribution,
            "contact_count_range": [int(contact_count_range[0]), int(contact_count_range[1])],
        }
        robots[str(name)] = entry

    if output_dir is None:
        save_dir = OmegaConf.select(cfg, "training.save_dir", default=None)
        output_dir = (
            resolve_project_path(str(save_dir))
            if save_dir is not None
            else os.path.join(tempfile.gettempdir(), "kite_decoder_hparams")
        )
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "generated_contact_sampler_hparams.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"robots": robots}, f, sort_keys=True)

    with open_dict(cfg):
        cfg.dataset.hparams_path = out_path
        cfg.dataset.real_masks_path = None
        cfg.dataset.check_template_hash = False
    return out_path


# ---------------------------------------------------------------------------
# Per-manipulator cfg + model construction (delegates to the trainer)
# ---------------------------------------------------------------------------


def build_per_manipulator_cfgs(cfg: DictConfig) -> Dict[str, DictConfig]:
    """Reconstruct per-manipulator DictConfigs the same way the trainer does."""
    from train_kinematic_decoder import _build_per_manipulator_cfg, _resolve_manipulator_names

    manipulator_names = _resolve_manipulator_names(cfg)
    overrides_cfg = OmegaConf.select(cfg, "co_training.overrides", default=None)
    out: Dict[str, DictConfig] = {}
    for name in manipulator_names:
        out[name] = _build_per_manipulator_cfg(
            cfg, manipulator_name=str(name), overrides_cfg=overrides_cfg
        )
    return out


# ---------------------------------------------------------------------------
# Dataset materialization (train-first-k / fresh-test)
# ---------------------------------------------------------------------------


def build_train_datasets_from_buffers(
    cfg: DictConfig,
    *,
    dataset_meta: Dict[str, Any],
    manipulator_cfgs: Dict[str, DictConfig],
    generation_device: torch.device,
) -> Dict[str, Any]:
    """Load the existing per-manipulator train probe buffer referenced by ``dataset_meta``.

    Downstream callers can slice the first ``k`` samples themselves; we return the full
    dataset so ``.model``, ``.action_dim``, ``.robot_specs`` etc. remain available.
    """
    from contact_policy_probe import _build_probe_dataset, _build_probe_dataset_common_kwargs

    out: Dict[str, Any] = {}
    for name, manip_cfg in manipulator_cfgs.items():
        per_manip = dataset_meta["per_manipulator"][name]
        buf_path = resolve_project_path(str(per_manip["train_buffer_path"]))
        if not os.path.isfile(buf_path):
            raise FileNotFoundError(
                f"Train buffer for manipulator {name!r} missing: {buf_path}. "
                "Run cotrain training long enough to persist buffers, or point --split test."
            )
        common_kwargs = _build_probe_dataset_common_kwargs(manip_cfg)
        dataset, _ = _build_probe_dataset(
            manip_cfg,
            common_kwargs=common_kwargs,
            save_dir=os.path.dirname(buf_path),
            split="train",
            generation_device=generation_device,
            buffer_path_override=buf_path,
        )
        out[name] = dataset
    return out


def build_fresh_test_datasets(
    cfg: DictConfig,
    *,
    manipulator_cfgs: Dict[str, DictConfig],
    k: int,
    generation_device: torch.device,
    seed_offset: int = FRESH_TEST_SEED_OFFSET,
) -> Dict[str, Any]:
    """Synthesize a fresh k-sample probe buffer per manipulator with a distinct seed.

    Does NOT reuse the existing test buffer referenced in ``dataset_meta``. The seed is
    ``cfg.seed + seed_offset`` so runs are deterministic but distinct from both the
    training buffer (base seed) and the trainer's normal test buffer (base + 10_007).
    """
    from contact_policy_probe import _build_probe_dataset, _build_probe_dataset_common_kwargs

    out: Dict[str, Any] = {}
    for name, manip_cfg in manipulator_cfgs.items():
        # Freeze probe.test_samples to our k. Also force no-save/no-load to avoid clobbering.
        eval_cfg = copy.deepcopy(manip_cfg)
        with open_dict(eval_cfg):
            eval_cfg.probe.test_samples = int(k)
            eval_cfg.dataset.save_buffers = False
            eval_cfg.dataset.load_saved_buffers = False
        common_kwargs = _build_probe_dataset_common_kwargs(eval_cfg)
        # _build_probe_dataset will further add +10_007 to this seed for the test split,
        # giving cfg.seed + seed_offset + 10_007. Still deterministic and distinct.
        common_kwargs["seed"] = int(cfg.seed) + int(seed_offset)
        # Route the synth output to a scratch path that will be discarded (save_buffers=False).
        scratch_dir = os.path.join(
            resolve_project_path(str(cfg.training.save_dir)),
            "eval_scratch",
            name,
        )
        os.makedirs(scratch_dir, exist_ok=True)
        dataset, _ = _build_probe_dataset(
            eval_cfg,
            common_kwargs=common_kwargs,
            save_dir=scratch_dir,
            split="test",
            generation_device=generation_device,
        )
        out[name] = dataset
    return out


