"""Shared loading and inference for the independent decoder viewers."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import threading

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from scipy.spatial.transform import Rotation, Slerp

from paper_config import runtime_config
from evaluation.cotrain_eval_utils import (
    apply_contact_candidate_count_override, build_fresh_test_datasets,
    build_per_manipulator_cfgs,
)
from experiment.high_level_policy.contact_encoding import _load_cotrain_probe_policy
from train_kinematic_decoder import _move_batch_to_device, _prepare_device, _resolve_project_path
from visualization._saved_examples import (
    _load_batch_from_dataset, _load_single_probe_context,
    _load_sliced_probe_dataset_from_buffer, _build_probe_dataset_common_kwargs,
)


EMBODIMENT_LABELS = {
    "wujihand_right": "Wuji", "allegro_right": "Allegro", "barrett": "Barrett",
    "robotiq_gripper_stick": "Robotiq+Stick", "mano_right": "Human hand (MANO)",
}


def label(name: str) -> str:
    return EMBODIMENT_LABELS.get(name, name)


@dataclass
class Examples:
    name: str
    dataset: object
    batch: dict
    prediction: torch.Tensor

    @property
    def model(self):
        return self.dataset.model

    @property
    def count(self):
        return int(self.prediction.shape[0])

    def configuration(self, key: str, index: int) -> torch.Tensor:
        return self.batch[key][index, :self.model.dof].detach().cpu().clone()

    def intent(self, index: int) -> torch.Tensor:
        valid = self.batch["contact_valid_mask"][index].bool()
        return self.batch["contact_cloud"][index, valid, :6].detach().cpu().clone()


class DecoderRuntime:
    """Load the checkpoint once; cache only small, lazily built example sets."""

    def __init__(self, cfg):
        self.cfg = runtime_config(cfg)
        self.device = _prepare_device(str(self.cfg.training.device))
        self.lock = threading.RLock()
        self.cache: dict[str, Examples] = {}
        self.num_samples = max(1, int(self.cfg.visualization.num_samples))
        self.start_index = max(0, int(self.cfg.visualization.start_index))
        checkpoint = _resolve_project_path(str(self.cfg.visualization.checkpoint_path))
        self.checkpoint = checkpoint
        self.meta = torch.load(Path(checkpoint).with_name("dataset_meta.pt"), map_location="cpu", weights_only=False)
        self.bank = None
        if self.meta.get("mode") != "cotrain":
            policy, dataset, _, _, _, split = _load_single_probe_context(self.cfg, device=self.device)
            batch, _ = _load_batch_from_dataset(dataset, split_key=split, start_index=self.start_index, num_samples=self.num_samples)
            self.single_policy = policy
            self.names = [str(dataset.robot_name)]
            with torch.inference_mode():
                prediction = policy.predict_action(_move_batch_to_device(batch, self.device))["action_pred"].cpu()
            self.cache[self.names[0]] = Examples(self.names[0], dataset, batch, prediction)
            return
        self.names = [str(name) for name in self.meta["manipulators"]]
        self.bank, _, saved_cfg = _load_cotrain_probe_policy(
            self.cfg, checkpoint_path=checkpoint, device=self.device,
            prefer_ema=bool(self.cfg.visualization.checkpoint_prefer_ema), cotrain_meta=self.meta,
        )
        # Use the selected checkpoint's feature layout and geometry, including
        # composite dimensions. Viewer generation controls remain configurable.
        with open_dict(self.cfg):
            self.cfg.co_training.manipulators = self.names
            self.cfg.model = copy.deepcopy(saved_cfg.model)
            for name in self.names:
                per_meta = self.meta["per_manipulator"][name]
                OmegaConf.update(self.cfg, f"contact_candidates.per_manipulator.{name}",
                                 int(per_meta["surface_num_points"]), force_add=True)
                geometry = per_meta.get("robot_model_kwargs")
                if geometry is not None:
                    OmegaConf.update(self.cfg, f"co_training.overrides.{name}.robot_model_kwargs", geometry, force_add=True)
        apply_contact_candidate_count_override(self.cfg)
        self.embodiment_cfgs = build_per_manipulator_cfgs(self.cfg)

    def head(self, name):
        return self.bank.head_for(name) if self.bank is not None else self.single_policy

    def get(self, name: str) -> Examples:
        if name not in self.names:
            raise ValueError(f"Embodiment {name!r} is absent from this checkpoint")
        with self.lock:
            if name in self.cache:
                return self.cache[name]
            cfg = self.embodiment_cfgs[name]
            source = str(self.cfg.visualization.source)
            if source == "synthetic":
                dataset = build_fresh_test_datasets(
                    self.cfg, manipulator_cfgs={name: cfg}, k=self.start_index + self.num_samples,
                    generation_device=_prepare_device(str(self.cfg.dataset.generation_device)),
                )[name]
                first = self.start_index
            elif source == "buffer":
                split = str(self.cfg.visualization.split)
                split = "test" if split == "validate" else split
                if split not in {"train", "test"}:
                    raise ValueError("visualization.split must be train or test")
                path = _resolve_project_path(self.meta["per_manipulator"][name][f"{split}_buffer_path"])
                dataset, _ = _load_sliced_probe_dataset_from_buffer(
                    cfg, buffer_path=path, common_kwargs=_build_probe_dataset_common_kwargs(cfg),
                    split_key=split, generation_device=_prepare_device(str(self.cfg.dataset.generation_device)),
                    start_index=self.start_index, num_samples=self.num_samples,
                )
                first = 0
            else:
                raise ValueError("visualization.source must be synthetic or buffer")
            batch, _ = _load_batch_from_dataset(dataset, split_key="test", start_index=first, num_samples=self.num_samples)
            with torch.inference_mode():
                prediction = self.head(name).predict_action(_move_batch_to_device(batch, self.device))["action_pred"].cpu()
            if not torch.isfinite(prediction).all():
                raise ValueError(f"Non-finite decoder output for {label(name)}")
            self.cache[name] = Examples(name, dataset, batch, prediction)
            return self.cache[name]

    def encode(self, name: str, intent: torch.Tensor) -> torch.Tensor:
        with self.lock, torch.inference_mode():
            return self.head(name).encode_contact_latent({
                "contact_cloud": intent.unsqueeze(0).to(self.device),
                "contact_valid_mask": torch.ones((1, len(intent)), dtype=torch.bool, device=self.device),
            }).detach().clone()

    def decode(self, name: str, latent: torch.Tensor, q_init: torch.Tensor) -> torch.Tensor:
        """One decoder call. Always recompute the hand features at q_init."""
        with self.lock, torch.inference_mode():
            examples = self.get(name)
            model = examples.model
            # CUDA generation builds a separate CPU model for dataset features.
            # Arbitrary-q decoding uses this model instead, so it needs its own
            # template. Initialize lazily, using the checkpoint's feature count.
            if getattr(model, "hand_fps_local_points", None) is None:
                model.build_hand_fps_template(int(examples.dataset.hand_points_fps))
            q = q_init.to(model.device, dtype=torch.float32).reshape(1, model.dof)
            points, normals = model.transform_hand_fps_template_batch(q)
            batch = {
                "q1_padded": q.to(self.device),
                "hand_points_world": points.to(self.device),
                "hand_normals_world": normals.to(self.device),
                "action_valid_mask": examples.batch["action_valid_mask"][:1].to(self.device),
                "robot_index": torch.zeros(1, dtype=torch.long, device=self.device),
                "local_dof": torch.tensor([model.dof], device=self.device),
            }
            output = self.head(name).predict_action_from_contact_latent(latent, batch)["action_pred"][0, :model.dof].cpu()
            if not torch.isfinite(output).all():
                raise ValueError(f"Non-finite decoder output for {label(name)}")
            return output


def pose_groups(model) -> list[dict]:
    groups = getattr(model, "pose_groups", None)
    groups = list(groups) if groups else [{"name": "base", "translation_indices": model.base_translation_indices,
                                           "rotation_indices": model.base_rotation_indices}]
    return [{**g, "translation_indices": list(g["translation_indices"]),
             "rotation_indices": list(g["rotation_indices"])} for g in groups]


def translate_configuration(model, q, displacement):
    result = torch.as_tensor(q).detach().cpu().clone()
    for group in pose_groups(model):
        result[group["translation_indices"]] += torch.as_tensor(displacement, dtype=result.dtype)
    return result


def aligned_initial(examples: Examples, index: int, intent: torch.Tensor) -> torch.Tensor:
    """Translate every free component equally; do not rotate or fit contacts."""
    delta = intent[:, :3].mean(0) - examples.intent(index)[:, :3].mean(0)
    return translate_configuration(examples.model, examples.configuration("q1_padded", index), delta)


def closest_initial(examples: Examples, intent: torch.Tensor) -> tuple[int, torch.Tensor]:
    """Select a centered synthetic contact pattern as a visible initialization heuristic."""
    desired = intent[:, :3] - intent[:, :3].mean(0)
    scores = []
    for i in range(examples.count):
        candidate = examples.intent(i)[:, :3]
        distances = torch.cdist(desired, candidate - candidate.mean(0))
        scores.append(float(distances.min(0).values.mean() + distances.min(1).values.mean()))
    index = int(np.argmin(scores))
    return index, aligned_initial(examples, index, intent)


def perturb_initial(model, q, *, translation=(0., 0., 0.), rotation_deg=(0., 0., 0.), joint_fraction=0., seed=0):
    """Rigidly perturb all free components, then add bounded joint noise."""
    result = torch.as_tensor(q).detach().cpu().clone()
    groups = pose_groups(model)
    centers = np.stack([result[g["translation_indices"]].numpy() for g in groups])
    pivot = centers.mean(0)
    rotation = Rotation.from_euler("XYZ", rotation_deg, degrees=True)
    rotate = bool(np.any(np.asarray(rotation_deg) != 0))
    roots = set()
    for g, center in zip(groups, centers):
        ti, ri = g["translation_indices"], g["rotation_indices"]
        if rotate:
            result[ti] = torch.as_tensor(pivot + rotation.apply(center - pivot), dtype=result.dtype)
            result[ri] = torch.as_tensor((rotation * Rotation.from_euler("XYZ", result[ri].numpy())).as_euler("XYZ"), dtype=result.dtype)
        # A zero rotation must preserve the actual input coordinates, including
        # noncanonical but equivalent Euler angles. Reset then restores q exactly.
        result[ti] += torch.as_tensor(translation, dtype=result.dtype)
        roots.update(ti + ri)
    lower, upper = model.pk_chain.get_joint_limits()
    lower, upper = torch.as_tensor(lower), torch.as_tensor(upper)
    roots.update(getattr(model, "fixed_zero_joint_indices", ()))
    joints = [i for i in range(model.dof) if i not in roots]
    if joints and joint_fraction:
        generator = torch.Generator().manual_seed(int(seed))
        noise = torch.randn(len(joints), generator=generator) * (upper[joints] - lower[joints]) * float(joint_fraction)
        result[joints] = torch.clamp(result[joints] + noise, lower[joints], upper[joints])
    return result


def interpolate_configuration(model, initial, predicted, fraction):
    """Linear joints/translation and shortest-path base rotations for display."""
    t = float(np.clip(fraction, 0., 1.))
    if t == 0.:
        return initial.clone()
    if t == 1.:
        return predicted.clone()
    result = initial + t * (predicted - initial)
    for group in pose_groups(model):
        ri = group["rotation_indices"]
        if torch.equal(initial[ri], predicted[ri]):
            result[ri] = initial[ri]
            continue
        rotations = Rotation.from_euler("XYZ", torch.stack([initial[ri], predicted[ri]]).cpu().numpy())
        result[ri] = torch.as_tensor(Slerp([0., 1.], rotations)(t).as_euler("XYZ"), dtype=result.dtype, device=result.device)
    return result


def nearest_surface(model, q, intent):
    """Position-nearest template samples; normals evaluated at those same samples."""
    with torch.inference_mode():
        points, normals = model.get_surface_points_normals(q.to(model.device))
        points, normals = points.cpu(), normals.cpu()
        indices = torch.cdist(intent[:, :3].cpu(), points).argmin(1)
        return points[indices], normals[indices], indices


def attached_contacts(examples: Examples, index: int, q):
    batch, model = examples.batch, examples.model
    valid = batch["contact_valid_mask"][index].bool()
    with torch.inference_mode():
        points, normals = model.transform_attached_points_normals_batch(
            q=q.to(model.device).unsqueeze(0),
            local_points=batch["contact_local_points"][index:index+1].to(model.device),
            local_normals=batch["contact_local_normals"][index:index+1].to(model.device),
            link_indices=batch["contact_link_indices"][index:index+1].to(model.device),
        )
    return points[0].cpu()[valid], normals[0].cpu()[valid]


def contact_errors(intent, points, normals):
    distances = torch.linalg.vector_norm(points - intent[:, :3], dim=-1)
    a = torch.nn.functional.normalize(normals, dim=-1)
    b = torch.nn.functional.normalize(intent[:, 3:6], dim=-1)
    angles = torch.rad2deg(torch.acos((a * b).sum(-1).clamp(-1, 1)))
    return float(distances.square().mean().sqrt() * 1000), float(angles.mean())
