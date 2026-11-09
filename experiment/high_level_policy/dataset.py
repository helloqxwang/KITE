from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .payload import SCHEMA, load_high_level_payload, validate_object_only_world_observation

if TYPE_CHECKING:
    from .idp3_adapter import LinearNormalizer


CANONICAL_DATA_KEYS = ("obs_point_cloud", "proprio", "action")
SINGLE_MANIPULATOR_COMPONENTS = ("base_pos", "base_quat_wxyz", "joint_target")


def _flat_tail_dim(tensor: torch.Tensor) -> int:
    return int(np.prod(tuple(tensor.shape[1:])))


def _payload_summary(*, selected_action_space: str, meta: dict[str, Any], action_dim: int, proprio_dim: int) -> str:
    action_space = meta.get("action_space", {})
    payload_type = action_space.get("type", "<missing>") if isinstance(action_space, dict) else type(action_space).__name__
    payload_dim = action_space.get("dim", "<missing>") if isinstance(action_space, dict) else "<missing>"
    return (
        f"selected action_space={selected_action_space!r}; "
        f"payload action_space.type={payload_type!r}; payload action_space.dim={payload_dim!r}; "
        f"payload embodiment={meta.get('embodiment', '<missing>')!r}; "
        f"data.action dim={action_dim}; data.proprio dim={proprio_dim}"
    )


def _raise_action_space_mismatch(message: str, *, summary: str, payload_path: str) -> None:
    raise ValueError(f"{message} Payload mismatch in {payload_path}: {summary}.")


def _validate_single_manipulator_slices(
    action_space: dict[str, Any],
    *,
    action_dim: int,
    summary: str,
    payload_path: str,
) -> None:
    slices = action_space.get("component_slices")
    if not isinstance(slices, dict):
        _raise_action_space_mismatch(
            "single_manipulator payload is missing component_slices.",
            summary=summary,
            payload_path=payload_path,
        )
    for name in SINGLE_MANIPULATOR_COMPONENTS:
        if name not in slices:
            _raise_action_space_mismatch(
                f"single_manipulator payload is missing component slice {name!r}.",
                summary=summary,
                payload_path=payload_path,
            )
        value = slices[name]
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            _raise_action_space_mismatch(
                f"single_manipulator component slice {name!r} must be a two-element range.",
                summary=summary,
                payload_path=payload_path,
            )
        start, end = [int(x) for x in value]
        if start < 0 or end <= start or end > int(action_dim):
            _raise_action_space_mismatch(
                f"single_manipulator component slice {name!r}={value!r} is outside the action dimension.",
                summary=summary,
                payload_path=payload_path,
            )


def _validate_canonical_action_space(
    *,
    payload_path: str,
    meta: dict[str, Any],
    data: dict[str, torch.Tensor],
    selected_action_space: str,
) -> None:
    if meta.get("schema") != SCHEMA:
        raise ValueError(
            f"HighLevelSequenceDataset only accepts canonical {SCHEMA!r} payloads; "
            f"got meta.schema={meta.get('schema')!r} in {payload_path}"
        )
    missing = [key for key in CANONICAL_DATA_KEYS if key not in data]
    if missing:
        raise KeyError(f"Canonical payload {payload_path} is missing data keys: {missing}")
    action = torch.as_tensor(data["action"])
    proprio = torch.as_tensor(data["proprio"])
    action_dim = _flat_tail_dim(action)
    proprio_dim = _flat_tail_dim(proprio)
    summary = _payload_summary(
        selected_action_space=selected_action_space,
        meta=meta,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
    )
    action_space = meta.get("action_space", {})
    if not isinstance(action_space, dict):
        _raise_action_space_mismatch(
            "Canonical payload meta.action_space must be a dictionary.",
            summary=summary,
            payload_path=payload_path,
        )
    payload_type = str(action_space.get("type", "")).strip()
    payload_dim = action_space.get("dim")
    if payload_dim is None:
        _raise_action_space_mismatch(
            "Canonical payload meta.action_space.dim is missing.",
            summary=summary,
            payload_path=payload_path,
        )
    payload_dim_int = int(payload_dim)
    if action_dim != proprio_dim or action_dim != payload_dim_int:
        _raise_action_space_mismatch(
            "Canonical data.action/data.proprio dimensions must match meta.action_space.dim.",
            summary=summary,
            payload_path=payload_path,
        )

    if selected_action_space == "latent":
        if payload_type != "latent_contact":
            _raise_action_space_mismatch(
                "policy.action_space='latent' requires meta.action_space.type='latent_contact'.",
                summary=summary,
                payload_path=payload_path,
            )
        return

    if selected_action_space == "contact_points":
        if payload_type != "contact_points":
            _raise_action_space_mismatch(
                "policy.action_space='contact_points' requires meta.action_space.type='contact_points'.",
                summary=summary,
                payload_path=payload_path,
            )
        point_dim = int(action_space.get("point_dim", 6))
        slot_count = int(action_space.get("slot_count", 0))
        if point_dim < 1 or slot_count < 1 or point_dim * slot_count != payload_dim_int:
            _raise_action_space_mismatch(
                "contact_points payload requires positive slot_count/point_dim whose product matches dim.",
                summary=summary,
                payload_path=payload_path,
            )
        return

    if str(meta.get("embodiment", "")).strip() != selected_action_space:
        _raise_action_space_mismatch(
            "Embodiment action spaces require meta.embodiment to match policy.action_space.",
            summary=summary,
            payload_path=payload_path,
        )
    if payload_type != "single_manipulator":
        _raise_action_space_mismatch(
            "Embodiment action spaces require meta.action_space.type='single_manipulator'.",
            summary=summary,
            payload_path=payload_path,
        )
    _validate_single_manipulator_slices(
        action_space,
        action_dim=action_dim,
        summary=summary,
        payload_path=payload_path,
    )


def _episode_ranges(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for end in episode_ends.reshape(-1):
        end_int = int(end)
        ranges.append((int(start), end_int))
        start = end_int
    return ranges


def _episode_split_indices(
    *,
    episode_count: int,
    test_episode_count: int,
    split_seed: int,
    payload_path: str,
    group_labels: list[str] | tuple[str, ...] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    episode_count = int(episode_count)
    requested = max(0, int(test_episode_count))
    if episode_count <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    labels: np.ndarray | None = None
    if group_labels is not None:
        labels = np.asarray(group_labels, dtype=object).reshape(-1)
        if int(labels.size) != episode_count:
            raise ValueError(
                f"Split group label count {labels.size} does not match episode count {episode_count} in {payload_path}"
            )
        if any(str(value).strip() == "" for value in labels.tolist()):
            raise ValueError(f"Split group labels must be non-empty in {payload_path}")
    if labels is None:
        available_test_count = max(0, episode_count - 1)
    else:
        unique_labels = list(dict.fromkeys(str(value) for value in labels.tolist()))
        group_indices = {
            label: np.flatnonzero(labels == label).astype(np.int64) for label in unique_labels
        }
        # Keep at least one training episode from every represented object.
        available_test_count = sum(max(0, int(indices.size) - 1) for indices in group_indices.values())
    actual_test_count = min(requested, available_test_count)
    if actual_test_count < requested:
        warnings.warn(
            f"Requested {requested} test episodes from {payload_path}, but only {actual_test_count} "
            f"were available after reserving at least one training episode.",
            RuntimeWarning,
            stacklevel=3,
        )
    if actual_test_count == 0:
        return np.arange(episode_count, dtype=np.int64), np.zeros((0,), dtype=np.int64)
    rng = np.random.default_rng(int(split_seed))
    if labels is None:
        test_indices = np.sort(rng.choice(episode_count, size=actual_test_count, replace=False).astype(np.int64))
    else:
        counts = {label: int(indices.size) for label, indices in group_indices.items()}
        capacities = {label: max(0, count - 1) for label, count in counts.items()}
        ideals = {label: actual_test_count * count / episode_count for label, count in counts.items()}
        allocations = {
            label: min(capacities[label], int(np.floor(ideals[label]))) for label in unique_labels
        }
        remaining = actual_test_count - sum(allocations.values())
        tie_break = {label: float(rng.random()) for label in unique_labels}
        while remaining > 0:
            eligible = [label for label in unique_labels if allocations[label] < capacities[label]]
            if not eligible:
                raise RuntimeError("Internal error while allocating stratified test episodes")
            label = max(
                eligible,
                key=lambda value: (ideals[value] - allocations[value], tie_break[value]),
            )
            allocations[label] += 1
            remaining -= 1
        selected = [
            rng.choice(group_indices[label], size=allocations[label], replace=False).astype(np.int64)
            for label in unique_labels
            if allocations[label] > 0
        ]
        test_indices = np.sort(np.concatenate(selected) if selected else np.zeros((0,), dtype=np.int64))
    is_test = np.zeros((episode_count,), dtype=bool)
    is_test[test_indices] = True
    train_indices = np.flatnonzero(~is_test).astype(np.int64)
    return train_indices, test_indices


# Convert episode boundaries into fixed-length training windows with edge padding.
def _create_indices(
    episode_ends: np.ndarray,
    *,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    pad_before = min(max(int(pad_before), 0), int(sequence_length) - 1)
    pad_after = min(max(int(pad_after), 0), int(sequence_length) - 1)
    indices: list[list[int]] = []
    window_episode_indices: list[int] = []
    for episode_idx, (start_idx, end_idx) in enumerate(_episode_ranges(episode_ends)):
        episode_len = end_idx - start_idx
        min_start = -pad_before
        max_start = episode_len - int(sequence_length) + pad_after
        for rel_idx in range(min_start, max_start + 1):
            buffer_start = max(rel_idx, 0) + start_idx
            buffer_end = min(rel_idx + int(sequence_length), episode_len) + start_idx
            start_offset = buffer_start - (rel_idx + start_idx)
            end_offset = (rel_idx + int(sequence_length) + start_idx) - buffer_end
            sample_start = start_offset
            sample_end = int(sequence_length) - end_offset
            indices.append([buffer_start, buffer_end, sample_start, sample_end])
            window_episode_indices.append(int(episode_idx))
    if len(indices) == 0:
        return np.zeros((0, 4), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.asarray(indices, dtype=np.int64), np.asarray(window_episode_indices, dtype=np.int64)


# Materialize one padded temporal window from the flat frame storage.
class _FixedLengthSequenceSampler:
    def __init__(self, data: dict[str, torch.Tensor], indices: np.ndarray, sequence_length: int) -> None:
        self.data = data
        self.indices = indices
        self.sequence_length = int(sequence_length)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def sample_sequence(self, idx: int) -> dict[str, torch.Tensor]:
        buffer_start, buffer_end, sample_start, sample_end = [int(x) for x in self.indices[int(idx)]]
        result: dict[str, torch.Tensor] = {}
        for key, value in self.data.items():
            sample = value[buffer_start:buffer_end]
            if sample_start > 0 or sample_end < self.sequence_length:
                padded = torch.zeros((self.sequence_length,) + tuple(sample.shape[1:]), dtype=value.dtype)
                if sample_start > 0:
                    padded[:sample_start] = sample[0]
                if sample_end < self.sequence_length:
                    padded[sample_end:] = sample[-1]
                padded[sample_start:sample_end] = sample
                sample = padded
            result[key] = sample
        return result


# Bind one canonical payload file to a diffusion-style point-cloud sequence dataset.
class HighLevelSequenceDataset(Dataset):
    def __init__(
        self,
        *,
        payload_path: str | Path,
        action_space: str,
        horizon: int,
        pad_before: int,
        pad_after: int,
        split: str = "all",
        test_episode_count: int = 20,
        split_seed: int = 42,
        split_group_key: str | None = None,
        allow_empty: bool = False,
    ) -> None:
        super().__init__()
        payload = load_high_level_payload(payload_path)
        self.payload_path = str(Path(payload_path).expanduser().resolve())
        self.meta = copy.deepcopy(payload["meta"])
        if str(self.meta.get("task", "")).strip() == "object_suite_pick_place":
            validate_object_only_world_observation(self.meta, label=self.payload_path)
        self.episode_meta = list(payload["episode_meta"])
        self.episode_ends = torch.as_tensor(payload["episode_ends"], dtype=torch.long)
        self.data = {key: torch.as_tensor(value) for key, value in payload["data"].items()}
        self.action_space = str(action_space).strip().lower()
        if self.action_space == "":
            raise ValueError("policy.action_space must be a non-empty string")
        _validate_canonical_action_space(
            payload_path=self.payload_path,
            meta=self.meta,
            data=self.data,
            selected_action_space=self.action_space,
        )
        if self.action_space == "latent":
            starts = torch.cat([self.episode_ends.new_zeros(1), self.episode_ends[:-1]])
            self.data["proprio"] = self.data["proprio"].clone()
            self.data["proprio"][starts] = 0
        self.split = str(split).strip().lower()
        if self.split in {"val", "valid", "validation"}:
            self.split = "test"
        if self.split not in {"train", "test", "all", "none", ""}:
            raise ValueError(f"Unsupported split={split!r}; expected train, test, or all")
        if self.split in {"none", ""}:
            self.split = "all"
        self.test_episode_count = int(test_episode_count)
        self.split_seed = int(split_seed)
        self.split_group_key = None if split_group_key is None else str(split_group_key).strip() or None
        self.obs_point_cloud_key = "obs_point_cloud"
        self.agent_pos_key = "proprio"
        self.action_key = "action"
        self.sequence_length = int(horizon)
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)

        episode_ends_np = self.episode_ends.cpu().numpy()
        train_episode_indices, test_episode_indices = _episode_split_indices(
            episode_count=int(episode_ends_np.shape[0]),
            test_episode_count=self.test_episode_count,
            split_seed=self.split_seed,
            payload_path=self.payload_path,
            group_labels=(
                [str(item.get(self.split_group_key, "")) for item in self.episode_meta]
                if self.split_group_key is not None
                else None
            ),
        )
        self.train_episode_indices = train_episode_indices
        self.test_episode_indices = test_episode_indices
        if self.split == "train":
            selected_episode_indices = train_episode_indices
        elif self.split == "test":
            selected_episode_indices = test_episode_indices
        else:
            selected_episode_indices = np.arange(int(episode_ends_np.shape[0]), dtype=np.int64)
        self.selected_episode_indices = np.asarray(selected_episode_indices, dtype=np.int64)

        indices, window_episode_indices = _create_indices(
            episode_ends_np,
            sequence_length=self.sequence_length,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
        )
        if self.split != "all":
            mask = np.isin(window_episode_indices, self.selected_episode_indices)
            indices = indices[mask]
            window_episode_indices = window_episode_indices[mask]
        self.window_episode_indices = window_episode_indices.astype(np.int64, copy=False)
        if indices.shape[0] == 0 and not bool(allow_empty):
            raise RuntimeError(f"No {self.split} sequence windows available in payload {self.payload_path}")
        sampler_data = {
            self.obs_point_cloud_key: self.data[self.obs_point_cloud_key],
            self.agent_pos_key: self.data[self.agent_pos_key],
            self.action_key: self.data[self.action_key],
        }
        self.sampler = _FixedLengthSequenceSampler(
            data=sampler_data,
            indices=indices,
            sequence_length=self.sequence_length,
        )

    @property
    def num_points(self) -> int:
        return int(self.data[self.obs_point_cloud_key].shape[1])

    @property
    def state_dim(self) -> int:
        return self._effective_flat_dim(self.agent_pos_key)

    @property
    def action_dim(self) -> int:
        return self._effective_flat_dim(self.action_key)

    def __len__(self) -> int:
        return len(self.sampler)

    # Flatten per-step structured state/action tensors into the vector form used by the policy.
    def _effective_flat_dim(self, key: str) -> int:
        tail_shape = self.data[key].shape[1:]
        return int(np.prod(tail_shape))

    def _flatten_tail(self, tensor: torch.Tensor, *, key: str) -> torch.Tensor:
        return tensor.view(tensor.shape[0], -1)

    # Assemble the canonical observation cloud from the stored object point set.
    def _build_point_cloud(self, sample: dict[str, torch.Tensor]) -> torch.Tensor:
        return sample[self.obs_point_cloud_key].to(dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        sample = self.sampler.sample_sequence(int(idx))
        point_cloud = self._build_point_cloud(sample)
        if point_cloud.ndim != 3:
            raise ValueError(f"Expected point_cloud sample with ndim=3, got {tuple(point_cloud.shape)}")
        agent_pos = self._flatten_tail(sample[self.agent_pos_key].to(dtype=torch.float32), key=self.agent_pos_key)
        action = self._flatten_tail(sample[self.action_key].to(dtype=torch.float32), key=self.action_key)
        return {
            "obs": {
                "point_cloud": point_cloud,
                "agent_pos": agent_pos,
            },
            "action": action,
        }

    # Recover the frame ranges behind the payload episodes for normalizer fitting.
    def _frame_ranges(self) -> list[tuple[int, int]]:
        ranges = _episode_ranges(self.episode_ends.cpu().numpy())
        if self.split == "all":
            return ranges
        return [ranges[int(idx)] for idx in self.selected_episode_indices.tolist()]

    def _point_cloud_stats(self) -> dict[str, torch.Tensor]:
        """Compute XYZ statistics over this dataset split without copying the full cloud."""

        xyz_min = torch.full((3,), float("inf"), dtype=torch.float64)
        xyz_max = torch.full((3,), -float("inf"), dtype=torch.float64)
        xyz_sum = torch.zeros(3, dtype=torch.float64)
        xyz_square_sum = torch.zeros(3, dtype=torch.float64)
        point_count = 0
        for start, end in self._frame_ranges():
            xyz = self.data[self.obs_point_cloud_key][start:end].reshape(-1, 3).to(dtype=torch.float64)
            if not bool(torch.isfinite(xyz).all()):
                raise ValueError(f"Point cloud contains non-finite XYZ values in frames [{start}, {end})")
            xyz_min = torch.minimum(xyz_min, xyz.amin(dim=0))
            xyz_max = torch.maximum(xyz_max, xyz.amax(dim=0))
            xyz_sum += xyz.sum(dim=0)
            xyz_square_sum += (xyz * xyz).sum(dim=0)
            point_count += int(xyz.shape[0])
        if point_count == 0:
            raise RuntimeError(f"Cannot fit point-cloud normalization on empty split={self.split!r}")

        mean = xyz_sum / point_count
        variance = (xyz_square_sum / point_count - mean.square()).clamp_min(0.0)
        return {
            "min": xyz_min.to(dtype=torch.float32),
            "max": xyz_max.to(dtype=torch.float32),
            "mean": mean.to(dtype=torch.float32),
            "std": variance.sqrt().to(dtype=torch.float32),
        }

    # Fit every learned input/output transform on this dataset split only.
    def get_normalizer(
        self,
        *,
        normalize_agent_pos: bool = False,
        normalize_point_cloud: bool = False,
    ) -> LinearNormalizer:
        from .idp3_adapter import build_idp3_normalizer

        agent_pos_chunks = []
        action_chunks = []
        for start, end in self._frame_ranges():
            agent_pos_chunks.append(
                self._flatten_tail(self.data[self.agent_pos_key][start:end].to(dtype=torch.float32), key=self.agent_pos_key)
            )
            action_chunks.append(
                self._flatten_tail(self.data[self.action_key][start:end].to(dtype=torch.float32), key=self.action_key)
            )
        agent_pos_data = torch.cat(agent_pos_chunks, dim=0)
        action_data = torch.cat(action_chunks, dim=0)
        return build_idp3_normalizer(
            action_data=action_data,
            agent_pos_data=agent_pos_data,
            point_cloud_stats=self._point_cloud_stats() if bool(normalize_point_cloud) else None,
            normalize_agent_pos=bool(normalize_agent_pos),
            normalize_point_cloud=bool(normalize_point_cloud),
        )
