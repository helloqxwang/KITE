from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable
from omegaconf import DictConfig, OmegaConf
from experiment.high_level_policy.dataset import HighLevelSequenceDataset
from experiment.high_level_policy.idp3_adapter import build_idp3_policy, checkpoint_backbone_metadata, idp3_model_cfg


def backbone_name(cfg):
    name = str(OmegaConf.select(cfg, "policy.backbone", default="idp3")).lower()
    if name != "idp3":
        raise ValueError("The KITE pick-and-place policy uses the iDP3 backbone.")
    return name


def model_cfg(cfg):
    return idp3_model_cfg(cfg)


def observation_kind(cfg):
    return "point_cloud"


def build_sequence_dataset(cfg, *, payload_path, action_space, split, allow_empty=False):
    m = model_cfg(cfg)
    return HighLevelSequenceDataset(
        payload_path=payload_path, action_space=action_space,
        horizon=int(m.horizon), pad_before=int(m.n_obs_steps)-1,
        pad_after=int(m.n_action_steps)-1, split=split,
        test_episode_count=int(OmegaConf.select(cfg, "dataset.test_episode_count", default=20)),
        split_seed=int(OmegaConf.select(cfg, "dataset.split_seed", default=42)),
        split_group_key=OmegaConf.select(cfg, "dataset.split_group_key", default=None),
        allow_empty=allow_empty)


def build_policy_for_dataset(cfg, dataset, *, initialize_pretrained_backbone=True):
    return build_idp3_policy(cfg, action_dim=dataset.action_dim,
        state_dim=dataset.state_dim, point_count=dataset.num_points)


def build_policy_for_payload(cfg, payload, *, action_dim, state_dim, point_count, initialize_pretrained_backbone=False):
    return build_idp3_policy(cfg, action_dim=action_dim, state_dim=state_dim, point_count=point_count)


def build_train_normalizer(cfg, dataset):
    m = model_cfg(cfg)
    return dataset.get_normalizer(normalize_agent_pos=bool(getattr(m, "normalize_agent_pos", False)),
        normalize_point_cloud=bool(getattr(m, "normalize_point_cloud", False)))


def checkpoint_metadata(cfg):
    return checkpoint_backbone_metadata()

