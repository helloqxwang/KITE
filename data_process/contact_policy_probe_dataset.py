from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from data_process.contact_policy_dataset import (
    _BaseContactPolicyDataset,
    _build_component_distribution_from_range,
    _format_contact_count_interval_summary,
    _init_contact_count_interval_stats,
    _merge_contact_count_interval_stats,
)
from robot_model.object_patches import _random_tangent_basis as _object_random_tangent_basis
from robot_model.collision_primitives import (
    PrimitiveCollisionModel,
    collision_primitive_spec_path,
    load_collision_primitive_spec,
)
from robot_model.mano_collision_primitives import (
    ManoPrimitiveCollisionModel,
    mano_collision_primitive_spec_path,
)
from robot_model.robot_model import create_robot_model, resolve_robot_model_name, _is_mano_robot_name

ROOT_DIR = Path(__file__).resolve().parents[1]


def _sparse_subset_indices_and_valid(
    contact_valid_mask_1d: torch.Tensor,
    *,
    contact_input_points: int,
    max_real_contacts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the runtime contact subset for ``contact_mode='sparse'`` in one sample.

    Semantics:
      - ``contact_input_points == -1``: token width is ``max_real_contacts``; slots
        ``[0, n_active)`` hold the real-contact bank indices (in ascending order,
        which in sparse mode are ``[0..n_active-1]``), remaining slots hold index 0
        with ``valid=False`` — the padding is explicit and the model's attention
        mask hides it.
      - ``contact_input_points > 0``: token width is ``contact_input_points``; each
        real contact is duplicated ``base = k // n_active`` times, with the first
        ``k % n_active`` real contacts getting one extra copy. All slots
        ``valid=True``. Output is not shuffled — the downstream attention is
        xyz-conditioned and order-invariant.

    ``n_active == 0`` → return zeros + all-False, preserving the legacy fallback
    in ``_resample_contact_indices``. Pure function, no RNG.
    """
    mask_cpu = contact_valid_mask_1d.detach().to(device="cpu", dtype=torch.bool).view(-1)
    real_indices = torch.nonzero(mask_cpu, as_tuple=False).view(-1).to(dtype=torch.long)
    n_active = int(real_indices.numel())
    if int(contact_input_points) == -1:
        k_eff = int(max_real_contacts)
    else:
        k_eff = int(contact_input_points)
    if k_eff <= 0:
        raise ValueError(f"resolved token width must be positive, got {k_eff}")
    indices = torch.zeros((k_eff,), dtype=torch.long)
    valid = torch.zeros((k_eff,), dtype=torch.bool)
    if n_active == 0:
        return indices, valid
    if int(contact_input_points) == -1:
        fill = min(n_active, k_eff)
        indices[:fill] = real_indices[:fill]
        valid[:fill] = True
        return indices, valid
    base, rem = divmod(k_eff, n_active)
    offset = 0
    for i in range(n_active):
        reps = base + (1 if i < rem else 0)
        if reps <= 0:
            continue
        indices[offset : offset + reps] = int(real_indices[i].item())
        offset += reps
    valid[:offset] = True
    return indices, valid


# Fixed-buffer single-robot probe dataset for contact-policy diagnostics.
class ContactPolicyProbeDataset(Dataset, _BaseContactPolicyDataset):
    _VALID_CONTACT_MODES = {"tips", "patches", "random_links", "random_tips", "sparse"}
    _VALID_NON_CONTACT_PATTERN_MODES = {"uniform_sphere", "surface_mixture"}
    _VALID_SIXD_POSE_MODES = {"fixed", "xyz_only", "movable", "zero_centered"}
    _VALID_SPARSE_SAMPLING_MODES = {"point_uniform", "region_uniform"}
    _VALID_Q1_SAMPLING_MODES = {"gaussian", "contact_error_mixture"}
    _VALID_COMPOSITE_SYNTHESIS_MODES = {"independent", "nearby"}
    _SPARSE_REGION_NAMES = ("thumb", "index", "middle", "ring", "little", "palm")
    _RUNTIME_NON_CONTACT_SDF_BATCH_SIZE = 64

    def __init__(
        self,
        *,
        contact_mode: str,
        sixd_pose_mode: str,
        robot_name: str = "shadowhand",
        samples: int,
        build_batch_size: int = 1024,
        progress_label: str = "probe_dataset",
        fixed_contact_links: Iterable[str] = (
            "ffdistal",
            "mfdistal",
            "rfdistal",
            "lfdistal",
            "thdistal",
            "palm",
        ),
        probe_component_range: Tuple[int, int] | None = None,
        probe_component_count_values: Any | None = None,
        probe_component_count_distribution: Any | None = None,
        probe_contact_count_range: Tuple[int, int] | None = None,
        contact_input_points: int = 32,
        non_contact_input_points: int = 256,
        non_contact_sampling_radius: float = 0.10,
        non_contact_sdf_clearance: float = 0.002,
        non_contact_pattern_mode: str = "surface_mixture",
        non_contact_component_range: Tuple[int, int] | None = (2, 4),
        non_contact_candidate_points: int = 128,
        non_contact_collision_surface_points: int = 256,
        non_contact_epoch_batch_size: int = 1024,
        remove_non_contact_points: bool = False,
        non_contact_causal_enabled: bool = False,
        non_contact_empty_probability: float = 0.0,
        non_contact_goal_joint_noise_scale: float = 0.0,
        non_contact_goal_joint_select_min: int = 0,
        non_contact_goal_joint_select_max: int = -1,
        non_contact_blocker_fraction: float = 0.0,
        non_contact_blocker_candidate_points: int = 128,
        collision_primitive_spec_dir: str | Path = "./assets/collision_primitives",
        runtime_resample_deterministic: bool = False,
        q1_translation_noise_std: float = 0.005,
        q1_rotation_noise_std_deg: float = 5.0,
        q1_joint_noise_scale: float = 0.05,
        q1_zero_probability: float = 0.10,
        random_q1: bool = False,
        q1_sampling_mode: str = "gaussian",
        q1_contact_error_zero_prob: float = 0.08,
        q1_contact_error_bins_mm: Any | None = None,
        q1_contact_error_inactive_scale: float = 0.05,
        q1_contact_error_refine_steps: int = 2,
        sparse_sampling_mode: str = "point_uniform",
        composite_synthesis_mode: str = "independent",
        composite_nearby_max_distance: float = 0.12,
        composite_contact_count_ranges: Any | None = None,
        composite_collision_filter_enabled: bool = False,
        composite_collision_filter_margin: float = 0.0,
        composite_collision_filter_max_attempts: int = 64,
        hand_points_fps: int = 0,
        hand_points_fk_device: str = "cpu",
        epoch_cache_enabled: bool = False,
        epoch_cache_q1_contact_points: bool = False,
        load_buffer_path: str | Path | None = None,
        load_buffer_payload: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Initialize a single-robot fixed probe buffer and its runtime resampling settings.
        Dataset.__init__(self)

        # The probe is defined by contact source and q2 pose mode; q1 is always provided as proprioception.
        self.contact_mode = self._validate_mode(contact_mode, self._VALID_CONTACT_MODES, "contact_mode")
        self.sixd_pose_mode = self._validate_mode(
            sixd_pose_mode,
            self._VALID_SIXD_POSE_MODES,
            "sixd_pose_mode",
        )
        self.robot_name = str(robot_name)
        self.samples = int(samples)
        if self.samples <= 0:
            raise ValueError("samples must be positive")
        self.build_batch_size = int(max(1, build_batch_size))
        self.progress_label = str(progress_label)
        self.fixed_contact_links = tuple(str(x) for x in fixed_contact_links)
        # Full validation (and resolution of ``contact_input_points_effective``) is deferred
        # until ``probe_contact_count_range`` is resolved, since the ``-1`` sentinel expands
        # to the maximum possible real-contact count per sample.
        self.contact_input_points = int(contact_input_points)
        self.remove_non_contact_points = bool(remove_non_contact_points)
        requested_non_contact_input_points = int(non_contact_input_points)
        if requested_non_contact_input_points <= 0 and not self.remove_non_contact_points:
            raise ValueError("non_contact_input_points must be positive")
        self.non_contact_input_points = 0 if self.remove_non_contact_points else requested_non_contact_input_points
        self.non_contact_sampling_radius = float(max(0.0, non_contact_sampling_radius))
        self.non_contact_sdf_clearance = float(max(0.0, non_contact_sdf_clearance))
        self.non_contact_pattern_mode = self._validate_mode(
            non_contact_pattern_mode,
            self._VALID_NON_CONTACT_PATTERN_MODES,
            "non_contact_pattern_mode",
        )
        self.non_contact_component_range = self._resolve_non_contact_component_range(non_contact_component_range)
        self.non_contact_candidate_points = int(
            max(int(self.non_contact_input_points), int(non_contact_candidate_points))
            if not self.remove_non_contact_points
            else 0
        )
        self.non_contact_collision_surface_points = int(max(0, non_contact_collision_surface_points))
        self.non_contact_epoch_batch_size = int(max(1, non_contact_epoch_batch_size))
        self.non_contact_causal_enabled = bool(non_contact_causal_enabled)
        self.non_contact_empty_probability = min(max(float(non_contact_empty_probability), 0.0), 1.0)
        self.non_contact_goal_joint_noise_scale = float(max(0.0, non_contact_goal_joint_noise_scale))
        self.non_contact_goal_joint_select_min = int(max(0, non_contact_goal_joint_select_min))
        self.non_contact_goal_joint_select_max = int(non_contact_goal_joint_select_max)
        self.non_contact_blocker_fraction = min(max(float(non_contact_blocker_fraction), 0.0), 1.0)
        self.non_contact_blocker_candidate_points = int(max(1, non_contact_blocker_candidate_points))
        self.collision_primitive_spec_dir = self._resolve_collision_primitive_spec_dir(
            collision_primitive_spec_dir
        )
        self.collision_primitive_spec_path: Path | None = None
        self.collision_primitive_spec: Dict[str, Any] | None = None
        self.collision_primitive_fit_method: str | None = None
        self._collision_primitive_model: Any | None = None
        self._cpu_collision_primitive_model: Any | None = None
        self.runtime_resample_deterministic = bool(runtime_resample_deterministic)
        self.contact_cluster_mode = "connected_mask"
        self.q1_translation_noise_std = float(max(0.0, q1_translation_noise_std))
        self.q1_rotation_noise_std_deg = float(max(0.0, q1_rotation_noise_std_deg))
        self.q1_joint_noise_scale = float(max(0.0, q1_joint_noise_scale))
        self.q1_zero_probability = min(max(float(q1_zero_probability), 0.0), 1.0)
        self.random_q1 = bool(random_q1)
        self.q1_sampling_mode = self._validate_mode(
            q1_sampling_mode,
            self._VALID_Q1_SAMPLING_MODES,
            "q1_sampling_mode",
        )
        self.q1_contact_error_zero_prob = min(max(float(q1_contact_error_zero_prob), 0.0), 1.0)
        if q1_contact_error_bins_mm is None:
            q1_contact_error_bins_mm = (
                (0.2, 1.0, 0.32),
                (1.0, 3.0, 0.30),
                (3.0, 8.0, 0.20),
                (8.0, 20.0, 0.10),
            )
        self.q1_contact_error_bins_mm = tuple(
            (float(row[0]), float(row[1]), float(row[2])) for row in q1_contact_error_bins_mm
        )
        self.q1_contact_error_inactive_scale = float(
            min(max(float(q1_contact_error_inactive_scale), 0.0), 1.0)
        )
        self.q1_contact_error_refine_steps = int(max(0, q1_contact_error_refine_steps))
        self.sparse_sampling_mode = self._validate_mode(
            sparse_sampling_mode,
            self._VALID_SPARSE_SAMPLING_MODES,
            "sparse_sampling_mode",
        )
        self.composite_synthesis_mode = self._validate_mode(
            composite_synthesis_mode,
            self._VALID_COMPOSITE_SYNTHESIS_MODES,
            "composite_synthesis_mode",
        )
        self.composite_nearby_max_distance = float(max(0.0, composite_nearby_max_distance))
        self._composite_contact_count_ranges_raw = composite_contact_count_ranges
        self.composite_collision_filter_enabled = bool(composite_collision_filter_enabled)
        self.composite_collision_filter_margin = float(max(0.0, composite_collision_filter_margin))
        self.composite_collision_filter_max_attempts = int(max(0, composite_collision_filter_max_attempts))
        self.hand_points_fps = int(max(0, hand_points_fps))
        fk_dev_choice = str(hand_points_fk_device).lower()
        if fk_dev_choice not in {"cpu", "gpu"}:
            raise ValueError(
                f"hand_points_fk_device must be 'cpu' or 'gpu', got {hand_points_fk_device!r}"
            )
        # When set to "gpu", dataloader workers skip the per-fetch hand FK; the caller must
        # compute hand_points_world / hand_normals_world in the main thread (on GPU) before
        # feeding batches to the model. Default "cpu" preserves the existing probe pipeline.
        self.hand_points_fk_device = fk_dev_choice
        # Epoch-wise precompute cache: opt-in. When enabled, the trainer calls
        # ``prepare_epoch_precomputed(epoch, device=...)`` once at the top of every epoch,
        # which refreshes q1 / action / contact subset / hand-FPS FK in bulk on GPU and
        # pushes the results into shared-memory buffers. The fast ``__getitems__`` path
        # then only indexes into those buffers — dataloader workers do no per-fetch FK
        # and no per-fetch noise sampling. Default False keeps the existing behavior.
        self.epoch_cache_enabled = bool(epoch_cache_enabled)
        self.epoch_cache_q1_contact_points = bool(epoch_cache_q1_contact_points)
        if self.contact_mode in {"tips", "random_tips"} and len(self.fixed_contact_links) == 0:
            raise ValueError(f"{self.contact_mode} contact_mode requires at least one fixed_contact_links entry")

        if load_buffer_path is not None and load_buffer_payload is not None:
            raise ValueError("Only one of load_buffer_path or load_buffer_payload may be provided")

        # Reuse the main contact-policy dataset initialization for robot metadata and patch sampling settings.
        dataset_kwargs = dict(kwargs)
        dataset_kwargs["robot_names"] = [self.robot_name]
        dataset_kwargs["global_robot_names"] = [self.robot_name]
        fallback_robot_cfgs = self._build_probe_robot_fallback_cfgs(Path(dataset_kwargs["hparams_path"]))
        if fallback_robot_cfgs is not None:
            dataset_kwargs["fallback_robot_cfgs"] = fallback_robot_cfgs
        _BaseContactPolicyDataset.__init__(self, **dataset_kwargs)

        # Load the base robot spec first, then swap in the probe-specific template if needed.
        self.spec = dict(self.robot_specs[self.robot_name])
        self._activate_probe_model()
        self.model = self.spec["model"]
        self.robot_model_name = str(self.spec["robot_model_name"])
        self.robot_model_kwargs = dict(self.spec.get("robot_model_kwargs", {}))
        self.local_dof = int(self.model.dof)
        self.candidate_contact_points = int(self.max_contact_points)
        self.composite_contact_count_ranges = self._resolve_composite_contact_count_ranges(
            self._composite_contact_count_ranges_raw
        )
        self.robot_index = 0
        self.action_valid_mask_template = torch.ones((self.action_dim,), dtype=torch.bool)
        if len(self.model.fixed_zero_joint_indices) > 0:
            self.action_valid_mask_template[self.model.fixed_zero_joint_indices] = False

        self.contact_allowed_surface_mask = self._build_contact_allowed_surface_mask()
        self.fixed_contact_point_indices = (
            self._resolve_fixed_contact_point_indices() if self.contact_mode in {"tips", "random_tips"} else None
        )
        self.random_link_point_indices = (
            self._resolve_random_link_point_indices() if self.contact_mode == "random_links" else None
        )
        if self.contact_mode == "sparse":
            self.sparse_candidate_indices = torch.nonzero(
                self.contact_allowed_surface_mask, as_tuple=False
            ).view(-1).to(device=self.device, dtype=torch.long)
            if int(self.sparse_candidate_indices.numel()) < 2:
                raise RuntimeError(
                    f"{self.robot_name}: sparse contact mode needs at least 2 allowed surface points, "
                    f"got {int(self.sparse_candidate_indices.numel())}"
                )
            self.sparse_region_candidate_indices = self._build_sparse_region_candidate_indices()
        else:
            self.sparse_candidate_indices = None
            self.sparse_region_candidate_indices = None
        self.max_sparse_contact_points = self._resolve_max_sparse_contact_points()
        self.probe_component_range = self._resolve_probe_component_range(probe_component_range)
        component_values, component_distribution = _build_component_distribution_from_range(
            self.probe_component_range,
            values=self.spec["component_count_values"] if probe_component_count_values is None else probe_component_count_values,
            probabilities=(
                self.spec["component_count_distribution"]
                if probe_component_count_distribution is None
                else probe_component_count_distribution
            ),
        )
        self.probe_component_values = component_values.to(device=self.device, dtype=torch.long)
        self.probe_component_distribution = component_distribution.to(device=self.device, dtype=torch.float32)
        self.probe_contact_count_range = self._resolve_probe_contact_count_range(probe_contact_count_range)
        # Resolve the runtime token width for contact-cloud subsets. ``-1`` is a sparse-only
        # sentinel meaning "variable length": the width becomes the maximum possible real
        # contact count per sample and padding slots carry ``valid=False``. Any positive
        # value in sparse mode applies equal-duplication across real contacts (so small
        # contact clouds are up-sampled to the requested width with all slots valid).
        self._sparse_max_real_contacts = int(self.probe_contact_count_range[1])
        if self.contact_input_points == -1:
            if self.contact_mode != "sparse":
                raise ValueError(
                    "contact_input_points=-1 is only defined for contact_mode='sparse'"
                )
            self.contact_input_points_effective = self._sparse_max_real_contacts
        elif self.contact_input_points <= 0:
            raise ValueError(
                "contact_input_points must be a positive int or -1 (sparse-only), got "
                f"{self.contact_input_points}"
            )
        else:
            if (
                self.contact_mode == "sparse"
                and self.contact_input_points < self._sparse_max_real_contacts
            ):
                raise ValueError(
                    f"contact_input_points={self.contact_input_points} is less than the "
                    f"maximum possible real contact count "
                    f"({self._sparse_max_real_contacts}); some real contacts would be "
                    "dropped. Either raise contact_input_points or set it to -1."
                )
            self.contact_input_points_effective = self.contact_input_points
        self._joint_limit_lower, self._joint_limit_upper, self._joint_limit_finite = self._build_q_limit_tensors()
        self._joint_noise_std = self._build_q1_joint_noise_std()
        self._cache_runtime_cpu_tensors()
        # Per-worker runtime geometry: build a CPU-device robot model for work that happens
        # inside DataLoader workers. Workers are forked after the main process may have
        # initialized CUDA, so any lazy FK/SDF work in workers must avoid CUDA entirely.
        self._cpu_hand_model = None
        needs_cpu_non_contact_sdf = (
            not self.remove_non_contact_points
            and int(self.non_contact_input_points) > 0
        )
        needs_cpu_runtime_model = (
            self.hand_points_fps > 0
            and self.hand_points_fk_device == "cpu"
            and not self.epoch_cache_enabled
        ) or self.q1_sampling_mode == "contact_error_mixture" or needs_cpu_non_contact_sdf
        if needs_cpu_runtime_model:
            if str(self.model.device) == "cpu":
                # In-place is fine when the base model already lives on CPU.
                self._cpu_hand_model = self.model
            else:
                self._cpu_hand_model = create_robot_model(
                    robot_name=self.robot_model_name,
                    device="cpu",
                    num_points=int(self.surface_num_points),
                    **dict(self.robot_model_kwargs),
                )
            if self.hand_points_fps > 0 and self.hand_points_fk_device == "cpu":
                self._cpu_hand_model.build_hand_fps_template(int(self.hand_points_fps))
        # The epoch-cache path runs hand-FPS FK on ``self.model`` (already on the dataset's
        # generation device) once per epoch. Build the FPS template here so it's ready.
        if self.epoch_cache_enabled and self.hand_points_fps > 0:
            self.model.build_hand_fps_template(int(self.hand_points_fps))
        if self._needs_non_contact_primitives:
            self._init_collision_primitive_models()

        # Fixed-buffer storage keeps train/test splits reusable for later evaluation and visualization.
        self.buffer_contact_cloud = torch.empty(
            (self.samples, self.candidate_contact_points, 6),
            dtype=torch.float32,
            device="cpu",
        )
        self.buffer_contact_valid_mask = torch.empty(
            (self.samples, self.candidate_contact_points),
            dtype=torch.bool,
            device="cpu",
        )
        self.buffer_contact_local_points = torch.empty(
            (self.samples, self.candidate_contact_points, 3),
            dtype=torch.float32,
            device="cpu",
        )
        self.buffer_contact_local_normals = torch.empty(
            (self.samples, self.candidate_contact_points, 3),
            dtype=torch.float32,
            device="cpu",
        )
        self.buffer_contact_link_indices = torch.empty(
            (self.samples, self.candidate_contact_points),
            dtype=torch.int16,
            device="cpu",
        )
        self.buffer_contact_parent_sparse_indices = torch.empty(
            (self.samples, self.candidate_contact_points),
            dtype=torch.int16,
            device="cpu",
        )
        self.buffer_contact_cluster_ids = torch.empty(
            (self.samples, self.candidate_contact_points),
            dtype=torch.int16,
            device="cpu",
        )
        self.buffer_surface_mask = torch.empty(
            (self.samples, self.surface_num_points),
            dtype=torch.bool,
            device="cpu",
        )
        self.buffer_action = torch.empty(
            (self.samples, self.action_dim),
            dtype=torch.float32,
            device="cpu",
        )
        self.buffer_q2_padded = torch.empty(
            (self.samples, self.action_dim),
            dtype=torch.float32,
            device="cpu",
        )
        self.buffer_q1_padded = torch.empty(
            (self.samples, self.action_dim),
            dtype=torch.float32,
            device="cpu",
        )

        # Shared-memory runtime cache for ``epoch_cache_enabled``. These tensors must be
        # allocated BEFORE any DataLoader forks its workers so the shared storage is visible
        # to the forked processes; in-place writes from ``prepare_epoch_precomputed`` then
        # become visible to every worker via the shared backing.
        self._runtime_prepared_epoch = torch.zeros((), dtype=torch.int64)
        self._runtime_q1_padded: torch.Tensor | None = None
        self._runtime_action: torch.Tensor | None = None
        self._runtime_contact_indices: torch.Tensor | None = None
        self._runtime_contact_valid_mask: torch.Tensor | None = None
        self._runtime_q1_contact_points_world: torch.Tensor | None = None
        self._runtime_hand_points_world: torch.Tensor | None = None
        self._runtime_hand_normals_world: torch.Tensor | None = None
        self._runtime_non_contact_cloud: torch.Tensor | None = None
        self._runtime_non_contact_valid_mask: torch.Tensor | None = None
        self._runtime_non_contact_blocker_mask: torch.Tensor | None = None
        if self.epoch_cache_enabled:
            self._runtime_prepared_epoch.share_memory_()
            self._runtime_q1_padded = torch.zeros(
                (self.samples, self.action_dim), dtype=torch.float32, device="cpu"
            )
            self._runtime_action = torch.zeros(
                (self.samples, self.action_dim), dtype=torch.float32, device="cpu"
            )
            self._runtime_contact_indices = torch.zeros(
                (self.samples, self.contact_input_points_effective), dtype=torch.long, device="cpu"
            )
            self._runtime_contact_valid_mask = torch.zeros(
                (self.samples, self.contact_input_points_effective), dtype=torch.bool, device="cpu"
            )
            if self.epoch_cache_q1_contact_points and self.contact_input_points_effective <= 64:
                self._runtime_q1_contact_points_world = torch.zeros(
                    (self.samples, self.contact_input_points_effective, 3),
                    dtype=torch.float32,
                    device="cpu",
                )
            self._runtime_q1_padded.share_memory_()
            self._runtime_action.share_memory_()
            self._runtime_contact_indices.share_memory_()
            self._runtime_contact_valid_mask.share_memory_()
            if self._runtime_q1_contact_points_world is not None:
                self._runtime_q1_contact_points_world.share_memory_()
            if not self.remove_non_contact_points and int(self.non_contact_input_points) > 0:
                self._runtime_non_contact_cloud = torch.zeros(
                    (self.samples, self.non_contact_input_points, 3),
                    dtype=torch.float32,
                    device="cpu",
                )
                self._runtime_non_contact_valid_mask = torch.ones(
                    (self.samples, self.non_contact_input_points),
                    dtype=torch.bool,
                    device="cpu",
                )
                self._runtime_non_contact_blocker_mask = torch.zeros(
                    (self.samples, self.non_contact_input_points),
                    dtype=torch.bool,
                    device="cpu",
                )
                self._runtime_non_contact_cloud.share_memory_()
                self._runtime_non_contact_valid_mask.share_memory_()
                self._runtime_non_contact_blocker_mask.share_memory_()
            if self.hand_points_fps > 0:
                self._runtime_hand_points_world = torch.zeros(
                    (self.samples, self.hand_points_fps, 3), dtype=torch.float32, device="cpu"
                )
                self._runtime_hand_normals_world = torch.zeros(
                    (self.samples, self.hand_points_fps, 3), dtype=torch.float32, device="cpu"
                )
                self._runtime_hand_points_world.share_memory_()
                self._runtime_hand_normals_world.share_memory_()

        # The probe split is either restored from disk or synthesized once and then kept fixed.
        if load_buffer_payload is not None:
            self._load_buffer_payload(load_buffer_payload)
            self.buffer_build_time_s = 0.0
        elif load_buffer_path is not None:
            payload = torch.load(str(Path(load_buffer_path).resolve()), map_location="cpu", weights_only=False)
            self._load_buffer_payload(payload)
            self.buffer_build_time_s = 0.0
        else:
            build_t0 = time.perf_counter()
            self._populate_buffer()
            self.buffer_build_time_s = time.perf_counter() - build_t0

        # Sparse-subset indices/valid are a pure function of the static
        # ``buffer_contact_valid_mask`` plus two config constants, so build the
        # runtime subset once here (rather than recomputing it every epoch).
        if (
            self.epoch_cache_enabled
            and self.contact_mode == "sparse"
            and self._runtime_contact_indices is not None
        ):
            self._build_sparse_runtime_subset_once()

    def _build_sparse_runtime_subset_once(self) -> None:
        for row in range(int(self.samples)):
            if self._is_composite_sparse_contacts() and self.contact_input_points == -1:
                sel_idx = torch.arange(self._sparse_max_real_contacts, dtype=torch.long)
                sel_valid = self.buffer_contact_valid_mask[row, : self._sparse_max_real_contacts].clone()
            else:
                sel_idx, sel_valid = _sparse_subset_indices_and_valid(
                    self.buffer_contact_valid_mask[row],
                    contact_input_points=self.contact_input_points,
                    max_real_contacts=self._sparse_max_real_contacts,
                )
            self._runtime_contact_indices[row].copy_(sel_idx)
            self._runtime_contact_valid_mask[row].copy_(sel_valid)

    @property
    def _needs_non_contact_primitives(self) -> bool:
        return (
            not bool(self.remove_non_contact_points)
            and int(self.non_contact_input_points) > 0
        )

    @staticmethod
    def _resolve_collision_primitive_spec_dir(path_value: str | Path) -> Path:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        cwd_candidate = (Path.cwd() / path).resolve()
        if cwd_candidate.exists():
            return cwd_candidate
        return (ROOT_DIR / path).resolve()

    def _collision_primitive_spec_candidates(self) -> list[Path]:
        candidates: list[Path] = []
        mano_name = _is_mano_robot_name(self.robot_model_name) or _is_mano_robot_name(self.robot_name)
        if mano_name is not None:
            side = "right" if str(mano_name).endswith("right") else "left"
            candidates.append(mano_collision_primitive_spec_path(self.collision_primitive_spec_dir, side=side))
        for name in (self.robot_model_name, self.robot_name, resolve_robot_model_name(self.robot_name)):
            candidates.append(collision_primitive_spec_path(self.collision_primitive_spec_dir, str(name)))
        deduped: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            key = str(path)
            if key not in seen:
                seen.add(key)
                deduped.append(path)
        return deduped

    def _make_collision_primitive_model(self, geometry_model: Any) -> Any:
        if self.collision_primitive_spec is None:
            raise RuntimeError("Collision primitive spec has not been loaded")
        if str(self.collision_primitive_spec.get("model_family", "")) == "mano":
            return ManoPrimitiveCollisionModel(geometry_model, self.collision_primitive_spec)
        return PrimitiveCollisionModel(geometry_model, self.collision_primitive_spec)

    def _init_collision_primitive_models(self) -> None:
        candidates = self._collision_primitive_spec_candidates()
        spec_path = next((path for path in candidates if path.exists()), None)
        if spec_path is None:
            candidate_text = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError(
                f"{self.robot_name}: non-contact synthesis now requires a collision primitive "
                f"sidecar; searched: {candidate_text}"
            )
        spec = load_collision_primitive_spec(spec_path)
        self.collision_primitive_spec_path = spec_path
        self.collision_primitive_spec = spec
        self.collision_primitive_fit_method = (
            str(spec.get("fit_method")) if spec.get("fit_method") is not None else None
        )
        self._collision_primitive_model = self._make_collision_primitive_model(self.model)
        if self._cpu_hand_model is not None and self._cpu_hand_model is not self.model:
            self._cpu_collision_primitive_model = self._make_collision_primitive_model(self._cpu_hand_model)
        else:
            self._cpu_collision_primitive_model = self._collision_primitive_model

    @staticmethod
    def _collision_model_device(collision_model: Any) -> torch.device:
        owner = getattr(collision_model, "robot_model", None)
        if owner is None:
            owner = getattr(collision_model, "hand_model", None)
        return torch.device(getattr(owner, "device", "cpu"))

    def _probe_robot_model_kwargs(self) -> Dict[str, Any]:
        # Select probe-specific robot-model options for robots that need alternate templates.
        if "shadowhand" in str(self.robot_name).strip().lower():
            return {"shadowhand_mode": "probe_palm_root"}
        return {}

    @staticmethod
    def _default_probe_surface_num_points(hparams_path: Path) -> int:
        # Reuse the first hparams surface count when the requested probe robot has no entry.
        try:
            with open(hparams_path, "r", encoding="utf-8") as f:
                hparams = yaml.safe_load(f)
            robot_cfgs = hparams.get("robots", {}) if isinstance(hparams, dict) else {}
            if isinstance(robot_cfgs, dict) and len(robot_cfgs) > 0:
                first_cfg = next(iter(robot_cfgs.values()))
                return int(first_cfg.get("num_surface_points", 256))
        except Exception:
            pass
        return 256

    @classmethod
    def _default_probe_contact_count_range(cls, num_surface_points: int) -> tuple[int, int]:
        # Scale the legacy 256-point probe contact-count range to the active template size.
        scale = float(max(1, int(num_surface_points))) / 256.0
        lo = max(1, int(round(7.0 * scale)))
        hi = max(lo, int(round(34.0 * scale)))
        return lo, hi

    def _build_probe_robot_fallback_cfgs(self, hparams_path: Path) -> Dict[str, Dict[str, Any]] | None:
        # Create a minimal uniform prior config when the probe robot is absent from hparams.
        try:
            with open(hparams_path, "r", encoding="utf-8") as f:
                hparams = yaml.safe_load(f)
            robot_cfgs = hparams.get("robots", {}) if isinstance(hparams, dict) else {}
        except Exception:
            robot_cfgs = {}

        if isinstance(robot_cfgs, dict) and self.robot_name in robot_cfgs:
            return None

        num_surface_points = self._default_probe_surface_num_points(hparams_path)
        contact_count_range = self._default_probe_contact_count_range(num_surface_points)
        component_values = [2, 3, 4]
        component_prob = [1.0 / float(len(component_values))] * len(component_values)
        point_prob = [1.0 / float(num_surface_points)] * int(num_surface_points)
        resolved_robot_model_name = resolve_robot_model_name(self.robot_name)
        print(
            f"[probe_hparams_fallback] {self.robot_name}: no entry in {hparams_path}; "
            f"using uniform point prior with model={resolved_robot_model_name}, "
            f"num_surface_points={num_surface_points}, component_range=[2, 4], "
            f"contact_count_range={list(contact_count_range)}"
        )
        return {
            self.robot_name: {
                "robot_model_name": resolved_robot_model_name,
                "num_surface_points": int(num_surface_points),
                "point_prob": point_prob,
                "component_range": [2, 4],
                "component_count_values": component_values,
                "component_count_distribution": component_prob,
                "contact_count_range": [int(contact_count_range[0]), int(contact_count_range[1])],
            }
        }

    def _activate_probe_model(self) -> None:
        # Swap the base robot model for a probe-specific variant and remap priors if needed.
        probe_kwargs = self._probe_robot_model_kwargs()
        if len(probe_kwargs) == 0:
            return

        # Build the alternate model from the same template size as the base spec.
        base_spec = dict(self.spec)
        base_model = base_spec["model"]
        probe_model = create_robot_model(
            robot_name=str(base_spec["robot_model_name"]),
            device=self.device,
            num_points=int(base_spec["surface_num_points"]),
            **probe_kwargs,
        )

        point_prob = torch.as_tensor(base_spec["point_prob"], dtype=torch.float32, device=self.device).view(-1)
        base_hash = base_model.get_surface_template_hash()
        probe_hash = probe_model.get_surface_template_hash()
        if probe_hash != base_hash:
            # Real masks cannot be safely remapped because their support came from another template.
            if self.real_masks_path is not None:
                raise RuntimeError(
                    f"{self.robot_name}: probe template hash mismatch ({probe_hash} != {base_hash}) with real_masks_path set. Regenerate the real masks for the probe template first."
                )
            point_prob = self._project_point_prob_to_probe_template(base_model, probe_model, point_prob)
            print(
                f"[probe_template] {self.robot_name}: remapped point_prob from {base_hash[:12]} to {probe_hash[:12]}"
            )

        # Replace the stored spec so all later sampling uses the probe template.
        updated_spec = dict(base_spec)
        updated_spec["model"] = probe_model
        updated_spec["robot_model_kwargs"] = dict(probe_kwargs)
        updated_spec["point_prob"] = point_prob
        self.robot_specs[self.robot_name] = updated_spec
        self.spec = updated_spec

    def _project_point_prob_to_probe_template(
        self,
        base_model: Any,
        probe_model: Any,
        point_prob: torch.Tensor,
    ) -> torch.Tensor:
        # Transfer point probabilities between same-link surface templates by nearest neighbor.
        src_prob = torch.as_tensor(point_prob, dtype=torch.float32, device=self.device).view(-1)
        src_points = base_model.surface_template_points_canonical_world.to(device=self.device, dtype=torch.float32)
        src_links = base_model.surface_template_link_indices.to(device=self.device, dtype=torch.long)
        dst_points = probe_model.surface_template_points_canonical_world.to(device=self.device, dtype=torch.float32)
        dst_links = probe_model.surface_template_link_indices.to(device=self.device, dtype=torch.long)
        dst_prob = torch.zeros((int(dst_points.shape[0]),), dtype=torch.float32, device=self.device)

        for link_idx in torch.unique(src_links, sorted=True).tolist():
            src_idx = torch.nonzero(src_links == int(link_idx), as_tuple=False).view(-1)
            dst_idx = torch.nonzero(dst_links == int(link_idx), as_tuple=False).view(-1)
            if int(src_idx.numel()) == 0 or int(dst_idx.numel()) == 0:
                continue
            nearest_dst = torch.cdist(src_points[src_idx], dst_points[dst_idx]).argmin(dim=1)
            dst_prob.scatter_add_(0, dst_idx[nearest_dst], src_prob[src_idx])

        allowed_mask = torch.ones_like(dst_prob, dtype=torch.bool)
        if len(probe_model.excluded_contact_link_names) > 0:
            # Remove probability mass on probe links that should not be used for contacts.
            link_name_to_idx = {str(name): idx for idx, name in enumerate(probe_model.mesh_link_names)}
            for link_name in probe_model.excluded_contact_link_names:
                if link_name in link_name_to_idx:
                    allowed_mask &= dst_links != int(link_name_to_idx[link_name])
            dst_prob = dst_prob * allowed_mask.to(dtype=dst_prob.dtype)

        total = dst_prob.sum()
        if float(total.item()) <= 0.0:
            # Fall back to a uniform allowed-link prior if projection loses all mass.
            dst_prob = allowed_mask.to(dtype=torch.float32)
            total = dst_prob.sum()
        return dst_prob / total.clamp_min(1.0e-8)

    @staticmethod
    def _validate_mode(mode: str, valid_modes: set[str], name: str) -> str:
        # Canonicalize string modes and reject unsupported values early.
        mode_key = str(mode).strip().lower().replace(" ", "_")
        if mode_key not in valid_modes:
            raise ValueError(f"Unsupported {name}={mode}. Expected one of {sorted(valid_modes)}")
        return mode_key

    def _model_pose_groups(self) -> tuple[dict, ...]:
        groups = getattr(self.model, "pose_groups", None)
        if groups is None:
            return (
                {
                    "name": "base",
                    "translation_indices": tuple(int(i) for i in self.model.base_translation_indices),
                    "rotation_indices": tuple(int(i) for i in self.model.base_rotation_indices),
                },
            )
        out: list[dict] = []
        for group_idx, group in enumerate(groups):
            out.append(
                {
                    "name": str(group.get("name", f"pose{group_idx}")),
                    "translation_indices": tuple(int(i) for i in group.get("translation_indices", ())),
                    "rotation_indices": tuple(int(i) for i in group.get("rotation_indices", ())),
                }
            )
        return tuple(out)

    def _pose_translation_indices_flat(self) -> list[int]:
        indices: list[int] = []
        for group in self._model_pose_groups():
            indices.extend(int(i) for i in group["translation_indices"])
        return indices

    def _pose_rotation_indices_flat(self) -> list[int]:
        indices: list[int] = []
        for group in self._model_pose_groups():
            indices.extend(int(i) for i in group["rotation_indices"])
        return indices

    def _pose_translation_bounds_cpu(self, *, device: torch.device | str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        num_groups = len(self._model_pose_groups())
        lo = self._q2_base_translation_min_cpu.to(device=device, dtype=dtype).view(1, 3).expand(num_groups, -1).reshape(-1)
        hi = self._q2_base_translation_max_cpu.to(device=device, dtype=dtype).view(1, 3).expand(num_groups, -1).reshape(-1)
        return lo, hi

    def _pose_translation_bounds_device(self, *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        num_groups = len(self._model_pose_groups())
        lo = self.q2_base_translation_min.to(device=self.device, dtype=dtype).view(1, 3).expand(num_groups, -1).reshape(-1)
        hi = self.q2_base_translation_max.to(device=self.device, dtype=dtype).view(1, 3).expand(num_groups, -1).reshape(-1)
        return lo, hi

    def _has_composite_contact_groups(self) -> bool:
        return bool(getattr(self.model, "composite_contact_groups", None))

    def _is_composite_sparse_contacts(self) -> bool:
        return self.contact_mode == "sparse" and self._has_composite_contact_groups()

    def _resolve_composite_contact_count_ranges(self, raw: Any | None) -> dict[str, tuple[int, int]]:
        groups = getattr(self.model, "composite_contact_groups", None)
        if not groups:
            return {}
        raw_map: dict[str, Any] = {}
        if raw is not None:
            if hasattr(raw, "items"):
                raw_map = {str(k): v for k, v in raw.items()}
            else:
                raw_map = dict(raw)
        resolved: dict[str, tuple[int, int]] = {}
        for group in groups:
            name = str(group["name"])
            default_range = tuple(int(x) for x in group.get("default_contact_count_range", (1, 3)))
            value = raw_map.get(name, default_range)
            lo = int(value[0])
            hi = int(value[1])
            slot_count = int(group.get("slot_count", hi))
            lo = max(1, min(lo, slot_count))
            hi = max(lo, min(hi, slot_count))
            resolved[name] = (lo, hi)
        return resolved

    # Patch probes use all visible hand links except the hidden wrist mount.
    def _build_contact_allowed_surface_mask(self) -> torch.Tensor:
        # Build a template-point mask for links allowed to seed patch contacts.
        mask = torch.ones((self.surface_num_points,), dtype=torch.bool, device=self.device)
        candidate_mask = getattr(self.model, "surface_contact_candidate_mask", None)
        if candidate_mask is not None:
            candidate_mask = torch.as_tensor(candidate_mask, dtype=torch.bool, device=self.device).view(-1)
            if int(candidate_mask.numel()) != int(self.surface_num_points):
                raise RuntimeError(
                    f"{self.robot_name}: surface_contact_candidate_mask has "
                    f"{int(candidate_mask.numel())} entries, expected {int(self.surface_num_points)}"
                )
            mask &= candidate_mask
        if len(self.model.excluded_contact_link_names) == 0:
            return mask
        link_name_to_idx = {str(name): idx for idx, name in enumerate(self.model.mesh_link_names)}
        link_indices = self.model.surface_template_link_indices.to(device=self.device, dtype=torch.long)
        for link_name in self.model.excluded_contact_link_names:
            if link_name in link_name_to_idx:
                mask &= link_indices != int(link_name_to_idx[link_name])
        return mask

    def _resolve_representative_contact_point_indices(
        self,
        link_names: Iterable[str],
    ) -> torch.Tensor:
        # Pick one stable surface point per requested link for tips/random-link modes.
        link_names = tuple(str(name) for name in link_names)
        link_name_to_idx = {str(name): idx for idx, name in enumerate(self.model.mesh_link_names)}
        missing = [name for name in link_names if name not in link_name_to_idx]
        if missing:
            raise KeyError(f"Unknown contact links for {self.robot_name}: {missing}")

        q_zero = torch.zeros((1, self.local_dof), dtype=torch.float32, device=self.device)
        points_zero, _ = self.model.get_surface_points_normals_batch(q=q_zero)
        points_zero = points_zero[0]
        link_indices = self.model.surface_template_link_indices.to(device=self.device, dtype=torch.long)

        if "palm" in link_name_to_idx:
            palm_mask = link_indices == int(link_name_to_idx["palm"])
        else:
            palm_mask = torch.zeros_like(link_indices, dtype=torch.bool)
        palm_centroid = points_zero[palm_mask].mean(dim=0) if bool(palm_mask.any()) else points_zero.mean(dim=0)

        # Use distal-from-palm points for fingers and central points for the palm.
        selected: list[int] = []
        for link_name in link_names:
            link_mask = link_indices == int(link_name_to_idx[link_name])
            if not bool(link_mask.any()):
                raise RuntimeError(f"Robot {self.robot_name} has no surface points on link {link_name}")
            global_indices = torch.nonzero(link_mask, as_tuple=False).view(-1)
            link_points = points_zero[global_indices]
            if link_name == "palm":
                score = -torch.norm(link_points - palm_centroid.view(1, 3), dim=1)
            else:
                score = torch.norm(link_points - palm_centroid.view(1, 3), dim=1)
            selected.append(int(global_indices[int(score.argmax().item())].item()))
        return torch.as_tensor(selected, dtype=torch.long, device=self.device)

    # Tips mode uses one deterministic template point on each requested link.
    def _resolve_fixed_contact_point_indices(self) -> torch.Tensor:
        # Resolve configured fixed-contact links into template point indices.
        return self._resolve_representative_contact_point_indices(self.fixed_contact_links)

    def _resolve_random_link_point_indices(self) -> torch.Tensor:
        # Resolve one representative point on every link that has surface support.
        link_name_to_idx = {str(name): idx for idx, name in enumerate(self.model.mesh_link_names)}
        link_indices = self.model.surface_template_link_indices.to(device=self.device, dtype=torch.long)
        link_names: list[str] = []
        for link_name in self.model.mesh_link_names:
            link_name = str(link_name)
            link_idx = int(link_name_to_idx[link_name])
            if bool((link_indices == link_idx).any()):
                link_names.append(link_name)
        if len(link_names) == 0:
            raise RuntimeError(f"Robot {self.robot_name} has no surface links for random_links contact_mode")
        return self._resolve_representative_contact_point_indices(link_names)

    @staticmethod
    def _region_index_from_text(text: str) -> int | None:
        # Return indices into _SPARSE_REGION_NAMES. The short aliases cover the
        # wujihand/shadow-style ff/mf/rf/lf/th naming convention.
        t = str(text).lower()
        if any(k in t for k in ("thumb", "th_", "_th", "thdistal", "thmiddle", "thproximal", "finger1")):
            return 0
        if any(k in t for k in ("index", "forefinger", "ff_", "_ff", "ffdistal", "ffmiddle", "ffproximal", "finger2")):
            return 1
        if any(k in t for k in ("middle", "mf_", "_mf", "mfdistal", "mfmiddle", "mfproximal", "finger3")):
            return 2
        if any(k in t for k in ("ring", "rf_", "_rf", "rfdistal", "rfmiddle", "rfproximal", "finger4")):
            return 3
        if any(k in t for k in ("little", "pinky", "lf_", "_lf", "lfdistal", "lfmiddle", "lfproximal", "finger5")):
            return 4
        if any(k in t for k in ("palm", "wrist", "base", "metacarpal", "hand")):
            return 5
        return None

    def _build_mano_sparse_region_labels(self) -> torch.Tensor:
        # MANO joints 1..15 follow [index, middle, pinky/little, ring, thumb],
        # each with three anatomical joints. Assign a template point to the finger
        # with the strongest interpolated skinning support; weak/unassigned points
        # are treated as palm.
        labels = torch.full(
            (self.surface_num_points,),
            5,
            dtype=torch.long,
            device=self.device,
        )
        if all(
            hasattr(self.model, attr)
            for attr in ("_mano_weights", "_template_face_vertex_indices", "_template_barycentric")
        ):
            weights = self.model._mano_weights.to(device=self.device, dtype=torch.float32)
            face_vertices = self.model._template_face_vertex_indices.to(device=self.device, dtype=torch.long)
            bary = self.model._template_barycentric.to(device=self.device, dtype=torch.float32)
            face_w = weights.index_select(0, face_vertices.reshape(-1)).reshape(
                int(face_vertices.shape[0]), int(face_vertices.shape[1]), -1
            )
            w_tp = (face_w * bary.unsqueeze(-1)).sum(dim=1)
            groups = (
                (13, 14, 15),  # thumb
                (1, 2, 3),     # index
                (4, 5, 6),     # middle
                (10, 11, 12),  # ring
                (7, 8, 9),     # little / pinky
            )
            group_scores = torch.stack(
                [w_tp[:, list(group)].sum(dim=1) for group in groups],
                dim=1,
            )
            score, region = group_scores.max(dim=1)
            labels = torch.where(score > 0.20, region.to(dtype=torch.long), labels)
            return labels

        jd = self.model.joint_descendant_link_mask.to(device=self.device, dtype=torch.bool)
        groups_q = (
            tuple(range(39, 48)),  # thumb j13..j15
            tuple(range(3, 12)),   # index j1..j3
            tuple(range(12, 21)),  # middle j4..j6
            tuple(range(30, 39)),  # ring j10..j12
            tuple(range(21, 30)),  # little j7..j9
        )
        scores = torch.stack(
            [jd[list(group)].to(dtype=torch.float32).sum(dim=0) for group in groups_q],
            dim=1,
        )
        score, region = scores.max(dim=1)
        return torch.where(score > 0.0, region.to(dtype=torch.long), labels)

    def _build_sparse_region_labels(self) -> torch.Tensor:
        # Label every surface-template point as one of five fingers or palm.
        if _is_mano_robot_name(self.robot_name) is not None:
            return self._build_mano_sparse_region_labels()

        labels = torch.full(
            (self.surface_num_points,),
            5,
            dtype=torch.long,
            device=self.device,
        )
        link_indices = self.model.surface_template_link_indices.to(device=self.device, dtype=torch.long)
        mesh_names = [str(x) for x in self.model.mesh_link_names]
        joint_names = [str(x) for x in getattr(self.model, "joint_orders", [])]
        descendant = self.model.joint_descendant_link_mask.detach().to(device="cpu", dtype=torch.bool)
        link_indices_cpu = link_indices.detach().cpu()
        for point_idx in range(int(self.surface_num_points)):
            link_i = int(link_indices_cpu[point_idx].item())
            text_parts = [mesh_names[link_i] if 0 <= link_i < len(mesh_names) else ""]
            if 0 <= link_i < int(descendant.shape[1]):
                joint_idx = torch.nonzero(descendant[:, link_i], as_tuple=False).view(-1).tolist()
                text_parts.extend(joint_names[int(j)] for j in joint_idx if int(j) < len(joint_names))
            region_text = " ".join(text_parts)
            region = self._region_index_from_text(region_text)
            if region is not None:
                labels[point_idx] = int(region)
        return labels

    def _build_sparse_region_candidate_indices(self) -> list[torch.Tensor]:
        labels = self._build_sparse_region_labels()
        allowed = self.contact_allowed_surface_mask.to(device=self.device, dtype=torch.bool)
        regions: list[torch.Tensor] = []
        for region_idx in range(len(self._SPARSE_REGION_NAMES)):
            idx = torch.nonzero(
                allowed & (labels == int(region_idx)),
                as_tuple=False,
            ).view(-1).to(device=self.device, dtype=torch.long)
            regions.append(idx)
        if sum(int(x.numel()) for x in regions) <= 0:
            raise RuntimeError(f"{self.robot_name}: region-uniform sparse sampling found no candidate points")
        return regions

    def _sparse_region_counts(self) -> list[int]:
        if self.sparse_region_candidate_indices is None:
            return []
        return [int(x.numel()) for x in self.sparse_region_candidate_indices]

    def _resolve_max_sparse_contact_points(self) -> int:
        # Bound sparse support size by the active contact-mode candidate set.
        if self.contact_mode in {"tips", "random_tips"} and self.fixed_contact_point_indices is not None:
            return int(self.fixed_contact_point_indices.numel())
        if self.contact_mode == "random_links" and self.random_link_point_indices is not None:
            return int(self.random_link_point_indices.numel())
        return int(self.surface_num_points)

    def _build_q_limit_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Cache joint lower/upper limits and a finite-limit mask on the generation device.
        lower, upper = self.model.pk_chain.get_joint_limits()
        lower_t = torch.as_tensor(lower, dtype=torch.float32, device=self.device)
        upper_t = torch.as_tensor(upper, dtype=torch.float32, device=self.device)
        finite = torch.isfinite(lower_t) & torch.isfinite(upper_t)
        return lower_t, upper_t, finite

    def _build_q1_joint_noise_std(self) -> torch.Tensor:
        # Convert joint-limit ranges into per-joint q1 noise scales.
        range_t = (self._joint_limit_upper - self._joint_limit_lower).abs()
        fallback = torch.full_like(range_t, float(math.pi))
        std = torch.where(self._joint_limit_finite, range_t, fallback) * float(self.q1_joint_noise_scale)
        pose_indices = self._pose_translation_indices_flat() + self._pose_rotation_indices_flat()
        if len(pose_indices) > 0:
            std[pose_indices] = 0.0
        if len(self.model.fixed_zero_joint_indices) > 0:
            std[self.model.fixed_zero_joint_indices] = 0.0
        return std

    def _cache_runtime_cpu_tensors(self) -> None:
        # Mirror frequently used tensors on CPU for runtime sampling inside dataloader workers.
        self._joint_limit_lower_cpu = self._joint_limit_lower.detach().cpu()
        self._joint_limit_upper_cpu = self._joint_limit_upper.detach().cpu()
        self._joint_limit_finite_cpu = self._joint_limit_finite.detach().cpu()
        self._joint_noise_std_cpu = self._joint_noise_std.detach().cpu()
        self._q2_base_translation_min_cpu = self.q2_base_translation_min.detach().cpu()
        self._q2_base_translation_max_cpu = self.q2_base_translation_max.detach().cpu()
        self._surface_template_link_indices_cpu = self.model.surface_template_link_indices.detach().cpu().to(dtype=torch.long)
        self._joint_descendant_link_mask_cpu = self.model.joint_descendant_link_mask.detach().cpu().to(dtype=torch.bool)

    def _apply_fixed_zero_joints_cpu(self, q: torch.Tensor) -> torch.Tensor:
        # Enforce fixed-zero joint constraints without moving tensors off CPU.
        if len(self.model.fixed_zero_joint_indices) == 0:
            return q
        q_out = q.clone()
        if q_out.ndim == 1:
            q_out[self.model.fixed_zero_joint_indices] = 0.0
        else:
            q_out[..., self.model.fixed_zero_joint_indices] = 0.0
        return q_out

    def _apply_probe_pose_mode_constraints_cpu(
        self,
        q: torch.Tensor,
        *,
        sixd_pose_mode: str,
    ) -> torch.Tensor:
        # Apply pose-mode constraints to CPU q tensors for runtime q1 sampling.
        pose_mode = str(sixd_pose_mode).strip().lower()
        q_out = q.clone()
        if pose_mode == "fixed":
            if len(self.model.base_pose_indices) > 0:
                q_out[..., self.model.base_pose_indices] = 0.0
            return self._apply_fixed_zero_joints_cpu(q_out)
        if pose_mode == "xyz_only":
            # Keep translation movable while forcing base rotation to canonical zero.
            rot_idx = self._pose_rotation_indices_flat()
            trans_idx = self._pose_translation_indices_flat()
            if len(rot_idx) > 0:
                q_out[..., rot_idx] = 0.0
            if len(trans_idx) > 0:
                translation_min, translation_max = self._pose_translation_bounds_cpu(
                    device=q_out.device,
                    dtype=q_out.dtype,
                )
                q_out[..., trans_idx] = q_out[..., trans_idx].clamp(
                    min=translation_min,
                    max=translation_max,
                )
            return self._apply_fixed_zero_joints_cpu(q_out)
        if pose_mode in {"movable", "zero_centered"}:
            return self._apply_fixed_zero_joints_cpu(q_out)
        raise ValueError(f"Unsupported sixd_pose_mode={sixd_pose_mode}")

    def _clamp_q_to_limits_cpu(self, q: torch.Tensor) -> torch.Tensor:
        # Wrap base rotations and clamp finite joints on CPU.
        q_out = q.clone()
        rot_idx = self._pose_rotation_indices_flat()
        if len(rot_idx) > 0:
            q_out[..., rot_idx] = (
                torch.remainder(q_out[..., rot_idx] + float(math.pi), float(2.0 * math.pi))
                - float(math.pi)
            )
        finite = self._joint_limit_finite_cpu
        if bool(finite.any()):
            q_out[..., finite] = q_out[..., finite].clamp(
                min=self._joint_limit_lower_cpu[finite],
                max=self._joint_limit_upper_cpu[finite],
            )
        return self._apply_fixed_zero_joints_cpu(q_out)

    def _build_q1_from_q2_batch_cpu(
        self,
        q2_probe: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        # Create runtime q1 by perturbing q2 in translation, rotation, and joints.
        q1_probe = q2_probe.clone()
        trans_idx = self._pose_translation_indices_flat()
        if float(self.q1_translation_noise_std) > 0.0 and len(trans_idx) > 0:
            translation_noise = torch.randn(
                (int(q2_probe.shape[0]), len(trans_idx)),
                dtype=torch.float32,
                device=q2_probe.device,
                generator=generator,
            )
            q1_probe[:, trans_idx] = (
                q1_probe[:, trans_idx]
                + translation_noise * float(self.q1_translation_noise_std)
            )
        rot_noise_std = math.radians(float(self.q1_rotation_noise_std_deg))
        rot_idx = self._pose_rotation_indices_flat()
        if rot_noise_std > 0.0 and len(rot_idx) > 0:
            rotation_noise = torch.randn(
                (int(q2_probe.shape[0]), len(rot_idx)),
                dtype=torch.float32,
                device=q2_probe.device,
                generator=generator,
            )
            q1_probe[:, rot_idx] = (
                q1_probe[:, rot_idx] + rotation_noise * float(rot_noise_std)
            )
        if float(self.q1_joint_noise_scale) > 0.0:
            joint_noise = torch.randn(
                q1_probe.shape,
                dtype=torch.float32,
                device=q2_probe.device,
                generator=generator,
            )
            q1_probe = q1_probe + joint_noise * self._joint_noise_std_cpu.view(1, -1)
        q1_probe = self._clamp_q_to_limits_cpu(q1_probe)
        return self._apply_probe_pose_mode_constraints_cpu(q1_probe, sixd_pose_mode=self.sixd_pose_mode)

    def _sample_random_q1_batch_cpu(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        # Sample q1 joints independently from q2, with the full 6D base pose fixed at zero.
        batch_size = int(batch_size)
        if batch_size <= 0:
            return torch.zeros((0, self.local_dof), dtype=torch.float32)

        # MANO fast path: delegate to the model's anatomical-Euler sampler so random q1
        # poses cover the full biomechanical ROM instead of uniformly visiting a box in
        # the (non-anatomical) axis-angle frame. The uniform-box blend below is retained
        # for URDF-based manipulators whose axis-angle components *are* anatomically
        # meaningful per-joint.
        if hasattr(self.model, "sample_random_q_anatomical"):
            q_model = self.model.sample_random_q_anatomical(
                batch_size=batch_size,
                include_base_rotation=False,
                generator=generator,
            )
            if q_model.ndim == 1:
                q_model = q_model.unsqueeze(0)
            q = q_model.detach().to(device="cpu", dtype=torch.float32)[:, : self.local_dof].contiguous()
            return self._apply_probe_pose_mode_constraints_cpu(q, sixd_pose_mode="fixed")

        q = torch.zeros((batch_size, self.local_dof), dtype=torch.float32, device="cpu")
        finite = self._joint_limit_finite_cpu
        if bool(finite.any()):
            u = torch.rand((batch_size, self.local_dof), dtype=torch.float32, device="cpu", generator=generator)
            mode = torch.rand((batch_size, 1), dtype=torch.float32, device="cpu", generator=generator)
            correlated = (
                0.75 * torch.rand((batch_size, 1), dtype=torch.float32, device="cpu", generator=generator)
                + 0.25 * u
            )
            extreme_choice = torch.randint(
                0,
                2,
                (batch_size, 1),
                device="cpu",
                generator=generator,
            ).bool()
            extreme = torch.where(extreme_choice, u.pow(2.5), 1.0 - u.pow(2.5))
            center = 0.5 + 0.5 * (2.0 * u - 1.0).pow(3)
            t = torch.where(
                mode < 0.25,
                u,
                torch.where(mode < 0.55, correlated, torch.where(mode < 0.80, extreme, center)),
            ).clamp(0.0, 1.0)
            finite_mask = finite.unsqueeze(0).expand(batch_size, -1)
            lower = self._joint_limit_lower_cpu.unsqueeze(0).expand(batch_size, -1)
            upper = self._joint_limit_upper_cpu.unsqueeze(0).expand(batch_size, -1)
            q[finite_mask] = lower[finite_mask] + t[finite_mask] * (upper[finite_mask] - lower[finite_mask])

        non_finite = ~finite
        if bool(non_finite.any()):
            q[:, non_finite] = (
                torch.rand(
                    (batch_size, int(non_finite.sum().item())),
                    dtype=torch.float32,
                    device="cpu",
                    generator=generator,
                )
                * 2.0
                - 1.0
            ) * float(math.pi)

        return self._apply_probe_pose_mode_constraints_cpu(q, sixd_pose_mode="fixed")

    def _clamp_q_to_limits(self, q: torch.Tensor) -> torch.Tensor:
        # Wrap base rotations and clamp finite joints on the dataset device.
        q_out = q.clone()
        rot_idx = self._pose_rotation_indices_flat()
        if len(rot_idx) > 0:
            q_out[..., rot_idx] = (
                torch.remainder(q_out[..., rot_idx] + float(math.pi), float(2.0 * math.pi))
                - float(math.pi)
            )
        finite = self._joint_limit_finite
        if bool(finite.any()):
            q_out[..., finite] = q_out[..., finite].clamp(
                min=self._joint_limit_lower[finite],
                max=self._joint_limit_upper[finite],
            )
        return self._apply_fixed_zero_joints(q_out)

    def _apply_probe_pose_mode_constraints(
        self,
        q: torch.Tensor,
        *,
        sixd_pose_mode: str,
    ) -> torch.Tensor:
        # Apply pose-mode constraints to generation-device q tensors.
        pose_mode = str(sixd_pose_mode).strip().lower()
        q_out = q.clone()
        if pose_mode == "fixed":
            q_out = self.model._zero_base_pose_in_q(q_out)
            return self._apply_fixed_zero_joints(q_out)
        if pose_mode == "xyz_only":
            # Keep xyz translation movable while canonicalizing base rotation.
            rot_idx = self._pose_rotation_indices_flat()
            trans_idx = self._pose_translation_indices_flat()
            if len(rot_idx) > 0:
                q_out[..., rot_idx] = 0.0
            if len(trans_idx) > 0:
                translation_min, translation_max = self._pose_translation_bounds_device(dtype=q_out.dtype)
                q_out[..., trans_idx] = q_out[..., trans_idx].clamp(
                    min=translation_min,
                    max=translation_max,
                )
            return self._apply_fixed_zero_joints(q_out)
        if pose_mode in {"movable", "zero_centered"}:
            return self._apply_fixed_zero_joints(q_out)
        raise ValueError(f"Unsupported sixd_pose_mode={sixd_pose_mode}")

    def _build_q1_from_q2_batch(
        self,
        q2_probe: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        # Create buffer-time q1 by adding configured noise around q2.
        q1_probe = q2_probe.clone()
        trans_idx = self._pose_translation_indices_flat()
        if float(self.q1_translation_noise_std) > 0.0 and len(trans_idx) > 0:
            translation_noise = torch.randn(
                (int(q2_probe.shape[0]), len(trans_idx)),
                dtype=torch.float32,
                device=self.device,
                generator=generator,
            )
            q1_probe[:, trans_idx] = (
                q1_probe[:, trans_idx]
                + translation_noise * float(self.q1_translation_noise_std)
            )
        rot_noise_std = math.radians(float(self.q1_rotation_noise_std_deg))
        rot_idx = self._pose_rotation_indices_flat()
        if rot_noise_std > 0.0 and len(rot_idx) > 0:
            rotation_noise = torch.randn(
                (int(q2_probe.shape[0]), len(rot_idx)),
                dtype=torch.float32,
                device=self.device,
                generator=generator,
            )
            q1_probe[:, rot_idx] = (
                q1_probe[:, rot_idx] + rotation_noise * float(rot_noise_std)
            )
        if float(self.q1_joint_noise_scale) > 0.0:
            joint_noise = torch.randn(
                q1_probe.shape,
                dtype=torch.float32,
                device=self.device,
                generator=generator,
            )
            q1_probe = q1_probe + joint_noise * self._joint_noise_std.view(1, -1)
        q1_probe = self._clamp_q_to_limits(q1_probe)
        return self._apply_probe_pose_mode_constraints(q1_probe, sixd_pose_mode=self.sixd_pose_mode)

    def _clamp_q_to_limits_on_device(self, q: torch.Tensor) -> torch.Tensor:
        q_out = q.clone()
        device = q_out.device
        rot_idx = self._pose_rotation_indices_flat()
        if len(rot_idx) > 0:
            q_out[..., rot_idx] = (
                torch.remainder(q_out[..., rot_idx] + float(math.pi), float(2.0 * math.pi))
                - float(math.pi)
            )
        finite = self._joint_limit_finite_cpu.to(device=device, dtype=torch.bool)
        if bool(finite.any()):
            lower = self._joint_limit_lower_cpu.to(device=device, dtype=q_out.dtype)
            upper = self._joint_limit_upper_cpu.to(device=device, dtype=q_out.dtype)
            q_out[..., finite] = q_out[..., finite].clamp(
                min=lower[finite],
                max=upper[finite],
            )
        if len(self.model.fixed_zero_joint_indices) > 0:
            q_out[..., self.model.fixed_zero_joint_indices] = 0.0
        return q_out

    def _apply_probe_pose_mode_constraints_on_device(self, q: torch.Tensor) -> torch.Tensor:
        pose_mode = str(self.sixd_pose_mode).strip().lower()
        q_out = q.clone()
        if pose_mode == "fixed":
            if len(self.model.base_pose_indices) > 0:
                q_out[..., self.model.base_pose_indices] = 0.0
        elif pose_mode == "xyz_only":
            rot_idx = self._pose_rotation_indices_flat()
            trans_idx = self._pose_translation_indices_flat()
            if len(rot_idx) > 0:
                q_out[..., rot_idx] = 0.0
            if len(trans_idx) > 0:
                trans_min, trans_max = self._pose_translation_bounds_cpu(
                    device=q_out.device,
                    dtype=q_out.dtype,
                )
                q_out[..., trans_idx] = q_out[..., trans_idx].clamp(
                    min=trans_min,
                    max=trans_max,
                )
        elif pose_mode not in {"movable", "zero_centered"}:
            raise ValueError(f"Unsupported sixd_pose_mode={self.sixd_pose_mode}")
        if len(self.model.fixed_zero_joint_indices) > 0:
            q_out[..., self.model.fixed_zero_joint_indices] = 0.0
        return q_out

    def _build_joint_active_by_contact_mask_on_device(self, mask: torch.Tensor) -> torch.Tensor:
        squeeze = False
        mask = mask.to(dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
            squeeze = True
        device = mask.device
        surface_link_idx = self._surface_template_link_indices_cpu.to(device=device, dtype=torch.long)
        descendant_mask = self._joint_descendant_link_mask_cpu.to(device=device, dtype=torch.bool)
        active_links = torch.zeros(
            (int(mask.shape[0]), int(descendant_mask.shape[1])),
            dtype=torch.bool,
            device=device,
        )
        for link_i in range(int(descendant_mask.shape[1])):
            cols = surface_link_idx == int(link_i)
            if bool(cols.any()):
                active_links[:, link_i] = mask[:, cols].any(dim=1)
        joint_active = (active_links.unsqueeze(1) & descendant_mask.unsqueeze(0)).any(dim=-1)
        if squeeze:
            return joint_active.squeeze(0)
        return joint_active


    def _sample_causal_non_contact_empty_mask(
        self,
        batch_size: int,
        *,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        prob = float(self.non_contact_empty_probability)
        if not self.non_contact_causal_enabled or prob <= 0.0:
            return torch.zeros((int(batch_size),), dtype=torch.bool, device=device)
        if prob >= 1.0:
            return torch.ones((int(batch_size),), dtype=torch.bool, device=device)
        return torch.rand((int(batch_size),), dtype=torch.float32, device=device, generator=generator) < prob

    def _non_contact_goal_noise_allowed_mask(
        self,
        contact_joint_mask: torch.Tensor,
    ) -> torch.Tensor:
        allowed = ~contact_joint_mask[:, : int(self.local_dof)].to(dtype=torch.bool)
        action_valid = self.action_valid_mask_template[: int(self.local_dof)].to(
            device=allowed.device,
            dtype=torch.bool,
        )
        allowed = allowed & action_valid.view(1, -1)
        blocked_indices = set(int(i) for i in getattr(self.model, "base_pose_indices", ()))
        blocked_indices.update(int(i) for i in getattr(self.model, "fixed_zero_joint_indices", ()))
        for idx in blocked_indices:
            if 0 <= idx < int(self.local_dof):
                allowed[:, idx] = False
        return allowed

    def _sample_non_contact_goal_batch(
        self,
        *,
        q_seed: torch.Tensor,
        contact_joint_mask: torch.Tensor,
        empty_mask: torch.Tensor,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, dof = int(q_seed.shape[0]), int(q_seed.shape[1])
        device = q_seed.device
        changed_joint_mask = torch.zeros((batch_size, dof), dtype=torch.bool, device=device)
        if (
            not self.non_contact_causal_enabled
            or float(self.non_contact_goal_joint_noise_scale) <= 0.0
            or batch_size <= 0
        ):
            return q_seed.clone(), changed_joint_mask

        allowed = self._non_contact_goal_noise_allowed_mask(contact_joint_mask.to(device=device))
        allowed = allowed & ~empty_mask.to(device=device, dtype=torch.bool).view(-1, 1)
        available = allowed.sum(dim=1).to(dtype=torch.long)
        max_cfg = int(self.non_contact_goal_joint_select_max)
        if max_cfg < 0:
            max_count = available
        else:
            max_count = torch.minimum(available, torch.full_like(available, max_cfg))
        min_count = torch.minimum(
            max_count,
            torch.full_like(max_count, int(self.non_contact_goal_joint_select_min)),
        )
        span = (max_count - min_count + 1).clamp_min(1)
        draw = torch.floor(
            torch.rand((batch_size,), dtype=torch.float32, device=device, generator=generator)
            * span.to(dtype=torch.float32)
        ).to(dtype=torch.long)
        select_count = min_count + draw

        scores = torch.rand((batch_size, dof), dtype=torch.float32, device=device, generator=generator)
        scores = scores.masked_fill(~allowed, -1.0)
        order = scores.argsort(dim=1, descending=True)
        ranks = torch.empty_like(order)
        rank_values = torch.arange(dof, dtype=torch.long, device=device).view(1, -1).expand(batch_size, -1)
        ranks.scatter_(1, order, rank_values)
        selected = (ranks < select_count.view(-1, 1)) & allowed

        noise_std = self._joint_noise_std_cpu[:dof].to(device=device, dtype=torch.float32)
        noise = torch.randn((batch_size, dof), dtype=torch.float32, device=device, generator=generator)
        q_goal = q_seed + noise * noise_std.view(1, -1) * float(self.non_contact_goal_joint_noise_scale) * selected.to(
            dtype=q_seed.dtype
        )
        q_goal = self._clamp_q_to_limits_on_device(q_goal)
        q_goal = self._apply_probe_pose_mode_constraints_on_device(q_goal)
        q_goal = torch.where(empty_mask.to(device=device).view(-1, 1), q_seed, q_goal)
        changed_joint_mask = selected & ((q_goal - q_seed).abs() > 1.0e-7)
        return q_goal, changed_joint_mask

    def _sample_contact_error_targets_m(
        self,
        batch_size: int,
        *,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        target = torch.zeros((int(batch_size),), dtype=torch.float32, device=device)
        u = torch.rand((int(batch_size),), dtype=torch.float32, device=device, generator=generator)
        cursor = float(self.q1_contact_error_zero_prob)
        for lo_mm, hi_mm, prob in self.q1_contact_error_bins_mm:
            lo_m = max(float(lo_mm), 1.0e-9) * 1.0e-3
            hi_m = max(float(hi_mm), float(lo_mm) + 1.0e-9) * 1.0e-3
            next_cursor = cursor + float(prob)
            sel = (u >= cursor) & (u < next_cursor)
            n_sel = int(sel.sum().item())
            if n_sel > 0:
                r = torch.rand((n_sel,), dtype=torch.float32, device=device, generator=generator)
                target[sel] = torch.exp(
                    math.log(lo_m) + r * (math.log(hi_m) - math.log(lo_m))
                )
            cursor = next_cursor
        tail = u >= cursor
        if bool(tail.any()) and len(self.q1_contact_error_bins_mm) > 0:
            lo_mm, hi_mm, _ = self.q1_contact_error_bins_mm[-1]
            lo_m = max(float(lo_mm), 1.0e-9) * 1.0e-3
            hi_m = max(float(hi_mm), float(lo_mm) + 1.0e-9) * 1.0e-3
            n_tail = int(tail.sum().item())
            r = torch.rand((n_tail,), dtype=torch.float32, device=device, generator=generator)
            target[tail] = torch.exp(math.log(lo_m) + r * (math.log(hi_m) - math.log(lo_m)))
        return target

    def _sample_q1_perturbation_direction(
        self,
        q2_probe: torch.Tensor,
        surface_mask: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        device = q2_probe.device
        direction = torch.zeros_like(q2_probe)
        trans_idx = self._pose_translation_indices_flat()
        if float(self.q1_translation_noise_std) > 0.0 and len(trans_idx) > 0:
            direction[:, trans_idx] = torch.randn(
                (int(q2_probe.shape[0]), len(trans_idx)),
                dtype=torch.float32,
                device=device,
                generator=generator,
            ) * float(self.q1_translation_noise_std)
        rot_noise_std = math.radians(float(self.q1_rotation_noise_std_deg))
        rot_idx = self._pose_rotation_indices_flat()
        if rot_noise_std > 0.0 and len(rot_idx) > 0:
            direction[:, rot_idx] = torch.randn(
                (int(q2_probe.shape[0]), len(rot_idx)),
                dtype=torch.float32,
                device=device,
                generator=generator,
            ) * float(rot_noise_std)
        if float(self.q1_joint_noise_scale) > 0.0:
            joint_noise_std = self._joint_noise_std_cpu.to(device=device, dtype=torch.float32)
            direction = direction + torch.randn(
                direction.shape,
                dtype=torch.float32,
                device=device,
                generator=generator,
            ) * joint_noise_std.view(1, -1)

        joint_active = self._build_joint_active_by_contact_mask_on_device(surface_mask)[:, : int(self.local_dof)]
        inactive_scale = float(self.q1_contact_error_inactive_scale)
        direction = direction * torch.where(
            joint_active,
            torch.ones_like(direction),
            torch.full_like(direction, inactive_scale),
        )
        if len(self.model.fixed_zero_joint_indices) > 0:
            direction[:, self.model.fixed_zero_joint_indices] = 0.0
        return direction

    def _contact_rms_error(
        self,
        *,
        fk_model: Any,
        q_candidate: torch.Tensor,
        target_points: torch.Tensor,
        contact_local_points: torch.Tensor,
        contact_link_indices: torch.Tensor,
        contact_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        pred_points, _ = fk_model.transform_attached_points_normals_batch(
            q=q_candidate,
            local_points=contact_local_points,
            link_indices=contact_link_indices,
            local_normals=None,
        )
        pred_points = pred_points.to(device=q_candidate.device, dtype=torch.float32)
        valid = contact_valid_mask.to(device=q_candidate.device, dtype=torch.bool)
        target_points = target_points.to(device=q_candidate.device, dtype=torch.float32)
        sq = (pred_points - target_points).pow(2).sum(dim=-1)
        denom = valid.sum(dim=1).to(dtype=torch.float32).clamp_min(1.0)
        return torch.sqrt((sq * valid.to(dtype=torch.float32)).sum(dim=1) / denom)

    def _build_q1_contact_error_mixture_batch(
        self,
        *,
        q2_probe: torch.Tensor,
        surface_mask: torch.Tensor,
        contact_local_points: torch.Tensor,
        contact_link_indices: torch.Tensor,
        contact_valid_mask: torch.Tensor,
        target_contact_points: torch.Tensor,
        generator: torch.Generator | None,
        fk_model: Any,
    ) -> torch.Tensor:
        batch_size = int(q2_probe.shape[0])
        if batch_size <= 0:
            return q2_probe.clone()
        device = q2_probe.device
        target_rms = self._sample_contact_error_targets_m(
            batch_size,
            device=device,
            generator=generator,
        )
        nonzero = target_rms > 0.0
        if not bool(nonzero.any()):
            return q2_probe.clone()

        direction = self._sample_q1_perturbation_direction(
            q2_probe,
            surface_mask.to(device=device, dtype=torch.bool),
            generator=generator,
        )
        scale = torch.ones((batch_size,), dtype=torch.float32, device=device)
        q_candidate = q2_probe.clone()
        max_iter = int(self.q1_contact_error_refine_steps) + 1
        for iter_idx in range(max_iter):
            q_candidate = self._clamp_q_to_limits_on_device(q2_probe + direction * scale.view(-1, 1))
            q_candidate = self._apply_probe_pose_mode_constraints_on_device(q_candidate)
            q_candidate = torch.where(nonzero.view(-1, 1), q_candidate, q2_probe)
            rms = self._contact_rms_error(
                fk_model=fk_model,
                q_candidate=q_candidate,
                target_points=target_contact_points,
                contact_local_points=contact_local_points,
                contact_link_indices=contact_link_indices,
                contact_valid_mask=contact_valid_mask,
            )
            if iter_idx >= max_iter - 1:
                break
            adjust = torch.where(
                nonzero,
                target_rms / rms.clamp_min(1.0e-8),
                torch.ones_like(scale),
            )
            scale = (scale * adjust).clamp(0.0, 100.0)
        return q_candidate

    def _build_q1_for_contacts_batch(
        self,
        *,
        q2_probe: torch.Tensor,
        contact_obs: Dict[str, torch.Tensor],
        generator: torch.Generator | None,
        fk_model: Any,
    ) -> torch.Tensor:
        if self.q1_sampling_mode == "contact_error_mixture":
            return self._build_q1_contact_error_mixture_batch(
                q2_probe=q2_probe,
                surface_mask=contact_obs["surface_mask"].to(device=q2_probe.device, dtype=torch.bool),
                contact_local_points=contact_obs["contact_local_points"].to(device=q2_probe.device, dtype=torch.float32),
                contact_link_indices=contact_obs["contact_link_indices"].to(device=q2_probe.device, dtype=torch.long),
                contact_valid_mask=contact_obs["contact_valid_mask"].to(device=q2_probe.device, dtype=torch.bool),
                target_contact_points=contact_obs["contact_cloud"][..., :3].to(device=q2_probe.device, dtype=torch.float32),
                generator=generator,
                fk_model=fk_model,
            )
        return self._build_q1_from_q2_batch(q2_probe, generator=generator)  # type: ignore[arg-type]

    # Probe patch sampling can either use the hparams defaults or explicit config overrides.
    def _resolve_probe_component_range(
        self,
        probe_component_range: Tuple[int, int] | None,
    ) -> Tuple[int, int]:
        # Resolve patch component counts from overrides or base sampler hparams.
        if probe_component_range is None:
            return tuple(int(x) for x in self.spec["component_range"])
        comp_lo = int(probe_component_range[0])
        comp_hi = int(probe_component_range[1])
        max_anchors = int(self.spec["max_anchors"])
        comp_lo = max(1, min(comp_lo, max_anchors))
        comp_hi = max(comp_lo, min(comp_hi, max_anchors))
        return comp_lo, comp_hi

    def _resolve_non_contact_component_range(
        self,
        component_range: Tuple[int, int] | None,
    ) -> Tuple[int, int]:
        # Clamp the non-contact mixture component range to a valid positive interval.
        if component_range is None:
            return 2, 4
        comp_lo = int(component_range[0])
        comp_hi = int(component_range[1])
        comp_lo = max(1, comp_lo)
        comp_hi = max(comp_lo, comp_hi)
        return comp_lo, comp_hi

    def _resolve_probe_contact_count_range(
        self,
        probe_contact_count_range: Tuple[int, int] | None,
    ) -> Tuple[int, int]:
        # Resolve sparse contact count bounds according to the chosen contact mode.
        if self.contact_mode == "tips":
            lo, hi = int(self.max_sparse_contact_points), int(self.max_sparse_contact_points)
        elif probe_contact_count_range is not None:
            lo, hi = int(probe_contact_count_range[0]), int(probe_contact_count_range[1])
        elif self.contact_mode == "random_links":
            lo, hi = 4, 8
        elif self.contact_mode == "random_tips":
            lo, hi = 3, 5
        elif self.contact_mode == "sparse":
            lo, hi = 2, 6
        else:
            base_range = self.spec["contact_count_range"]
            lo, hi = int(base_range[0]), int(base_range[1])
        lo = max(1, min(lo, self.max_sparse_contact_points))
        hi = max(lo, min(hi, self.max_sparse_contact_points))
        return lo, hi

    def _apply_fixed_zero_joints(self, q: torch.Tensor) -> torch.Tensor:
        # Delegate generation-device fixed-joint constraints to the robot model.
        return self.model._apply_fixed_zero_joint_constraints(q)

    # Probe q sampling uses the robot model's public q semantics directly.
    def _sample_probe_q_batch_unfiltered(
        self,
        batch_size: int,
        generator: torch.Generator,
        *,
        sixd_pose_mode: str,
    ) -> torch.Tensor:
        # Start from canonical random q and then apply pose-mode-specific base motion.
        q_probe = self._sample_random_q_batch(
            model=self.model,
            batch_size=int(batch_size),
            generator=generator,
            base_pose_mode="canonical",
        )
        q_probe = self._apply_fixed_zero_joints(q_probe)
        pose_mode = str(sixd_pose_mode).strip().lower()
        if pose_mode not in self._VALID_SIXD_POSE_MODES:
            raise ValueError(f"Unsupported sixd_pose_mode={sixd_pose_mode}")
        pose_groups = self._model_pose_groups()
        if pose_mode in {"movable", "zero_centered", "xyz_only"}:
            if (
                self._has_composite_contact_groups()
                and self.composite_synthesis_mode == "nearby"
                and len(pose_groups) >= 2
            ):
                first_group = pose_groups[0]
                second_group = pose_groups[1]
                u0 = torch.rand((int(batch_size), 3), dtype=torch.float32, device=self.device, generator=generator)
                first_t = self.q2_base_translation_min.view(1, 3) + u0 * (
                    self.q2_base_translation_max.view(1, 3) - self.q2_base_translation_min.view(1, 3)
                )
                direction = torch.randn((int(batch_size), 3), dtype=torch.float32, device=self.device, generator=generator)
                direction = direction / torch.norm(direction, dim=-1, keepdim=True).clamp_min(1.0e-8)
                radius = torch.rand((int(batch_size), 1), dtype=torch.float32, device=self.device, generator=generator).pow(1.0 / 3.0)
                second_t = first_t + direction * radius * float(self.composite_nearby_max_distance)
                second_t = second_t.clamp(
                    min=self.q2_base_translation_min.view(1, 3),
                    max=self.q2_base_translation_max.view(1, 3),
                )
                q_probe[:, list(first_group["translation_indices"])] = first_t
                q_probe[:, list(second_group["translation_indices"])] = second_t
                for group in pose_groups[2:]:
                    u = torch.rand((int(batch_size), 3), dtype=torch.float32, device=self.device, generator=generator)
                    q_probe[:, list(group["translation_indices"])] = self.q2_base_translation_min.view(1, 3) + u * (
                        self.q2_base_translation_max.view(1, 3) - self.q2_base_translation_min.view(1, 3)
                    )
            else:
                for group in pose_groups:
                    trans_idx = list(group["translation_indices"])
                    if len(trans_idx) == 0:
                        continue
                    u = torch.rand((int(batch_size), 3), dtype=torch.float32, device=self.device, generator=generator)
                    q_probe[:, trans_idx] = self.q2_base_translation_min.view(1, 3) + u * (
                        self.q2_base_translation_max.view(1, 3) - self.q2_base_translation_min.view(1, 3)
                    )
        if pose_mode in {"movable", "zero_centered"}:
            # Sample each pose group's rotation independently.
            for group in pose_groups:
                rot_idx = list(group["rotation_indices"])
                if len(rot_idx) == 0:
                    continue
                q_probe[:, rot_idx] = self.model.sample_uniform_base_rotation_euler_xyz(
                    batch_size=int(batch_size),
                    generator=generator,
                )
        return self._apply_probe_pose_mode_constraints(q_probe, sixd_pose_mode=pose_mode)

    def _composite_collision_free_mask(self, q_probe: torch.Tensor) -> torch.Tensor:
        if not self.composite_collision_filter_enabled:
            return torch.ones((int(q_probe.shape[0]),), dtype=torch.bool, device=q_probe.device)
        checker = getattr(self.model, "gripper_stick_collision_free_mask", None)
        if checker is None:
            raise RuntimeError(
                f"{self.robot_name}: composite_collision_filter_enabled=true, but the active "
                "robot model does not expose gripper_stick_collision_free_mask"
            )
        with torch.no_grad():
            valid = checker(q_probe, margin=float(self.composite_collision_filter_margin))
        return valid.to(device=q_probe.device, dtype=torch.bool).view(-1)

    def _resample_composite_collisions(
        self,
        q_probe: torch.Tensor,
        *,
        generator: torch.Generator,
        sixd_pose_mode: str,
    ) -> torch.Tensor:
        if not self.composite_collision_filter_enabled:
            return q_probe
        q_out = q_probe.clone()
        valid = self._composite_collision_free_mask(q_out)
        attempts = 0
        while (not bool(valid.all())) and attempts < int(self.composite_collision_filter_max_attempts):
            rows = torch.nonzero(~valid, as_tuple=False).view(-1)
            replacement = self._sample_probe_q_batch_unfiltered(
                int(rows.numel()),
                generator=generator,
                sixd_pose_mode=sixd_pose_mode,
            )
            q_out.index_copy_(0, rows, replacement)
            valid = self._composite_collision_free_mask(q_out)
            attempts += 1
        if not bool(valid.all()):
            rows = torch.nonzero(~valid, as_tuple=False).view(-1)
            min_sdf_fn = getattr(self.model, "gripper_stick_min_sdf_batch", None)
            min_sdf_msg = ""
            if min_sdf_fn is not None and int(rows.numel()) > 0:
                with torch.no_grad():
                    min_sdf = min_sdf_fn(q_out.index_select(0, rows)).detach().cpu()
                min_sdf_msg = f"; worst min_sdf={float(min_sdf.min().item()):.6f}"
            raise RuntimeError(
                f"{self.robot_name}: failed to sample {int(rows.numel())} collision-free "
                f"gripper-stick actions after {int(self.composite_collision_filter_max_attempts)} "
                f"resampling attempts with margin={float(self.composite_collision_filter_margin):.6f}"
                f"{min_sdf_msg}"
            )
        return q_out

    def _sample_probe_q_batch(
        self,
        batch_size: int,
        generator: torch.Generator,
        *,
        sixd_pose_mode: str,
    ) -> torch.Tensor:
        q_probe = self._sample_probe_q_batch_unfiltered(
            int(batch_size),
            generator=generator,
            sixd_pose_mode=sixd_pose_mode,
        )
        return self._resample_composite_collisions(
            q_probe,
            generator=generator,
            sixd_pose_mode=sixd_pose_mode,
        )

    # Patch mode reuses the main contact-policy synthetic contact sampler on all surface points.
    def _sample_probe_patch_params(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> Dict[str, torch.Tensor]:
        # Override the base spec with probe-specific component priors for patch sampling.
        spec = dict(self.spec)
        spec["component_range"] = self.probe_component_range
        spec["component_count_values"] = self.probe_component_values
        spec["component_count_distribution"] = self.probe_component_distribution
        max_anchors = max(1, min(int(self.probe_component_range[1]), self.surface_num_points))
        spec["max_anchors"] = int(max_anchors)
        return self._sample_synthetic_patch_params(
            spec=spec,
            batch_size=int(batch_size),
            generator=generator,
        )

    # The contact center is computed from the full masked surface set before optional FPS/padding.
    def _compute_contact_center_from_surface_mask(
        self,
        points: torch.Tensor,
        surface_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Compute the mean q2 surface location over active contact points.
        weights = surface_mask.to(device=points.device, dtype=points.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (points * weights).sum(dim=1) / denom

    def _apply_zero_centering(
        self,
        *,
        q2: torch.Tensor,
        contact_cloud: torch.Tensor,
        contact_valid_mask: torch.Tensor,
        contact_center: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Express q2 translation and contact xyz relative to each sample's contact center.
        q2_eff = q2.clone()
        centered_contact = contact_cloud.clone()
        valid_xyz = contact_valid_mask.unsqueeze(-1).to(dtype=centered_contact.dtype)
        centered_contact[..., :3] = centered_contact[..., :3] - contact_center.unsqueeze(1) * valid_xyz
        for group in self._model_pose_groups():
            trans_idx = list(group["translation_indices"])
            if len(trans_idx) > 0:
                q2_eff[:, trans_idx] = q2_eff[:, trans_idx] - contact_center[:, : len(trans_idx)]
        return q2_eff, centered_contact

    def _build_contact_mask_from_indices(
        self,
        batch_size: int,
        point_indices: torch.Tensor,
        point_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Scatter selected template point indices into a dense sparse-contact mask.
        point_indices = point_indices.to(device=self.device, dtype=torch.long)
        if point_valid_mask is None:
            point_valid_mask = torch.ones_like(point_indices, dtype=torch.bool, device=self.device)
        else:
            point_valid_mask = point_valid_mask.to(device=self.device, dtype=torch.bool)
        surface_mask = torch.zeros(
            (int(batch_size), self.surface_num_points),
            dtype=torch.bool,
            device=self.device,
        )
        # Route filler slots to the first valid index per row (always valid since callers
        # guarantee count >= 1), so scatter_ cannot overwrite a real contact at template
        # index 0. Every scatter write is True, and duplicates on the first valid index
        # are idempotent.
        first_valid = point_indices[:, :1].expand_as(point_indices)
        scatter_indices = torch.where(point_valid_mask, point_indices, first_valid)
        surface_mask.scatter_(
            1,
            scatter_indices,
            torch.ones_like(scatter_indices, dtype=torch.bool),
        )
        return surface_mask

    def _sample_random_contact_point_indices_batch(
        self,
        *,
        candidate_point_indices: torch.Tensor,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Randomly choose a variable-size subset from the candidate contact points per sample.
        candidate_point_indices = torch.as_tensor(candidate_point_indices, dtype=torch.long, device=self.device).view(-1)
        num_candidates = int(candidate_point_indices.numel())
        if num_candidates <= 0:
            raise RuntimeError(f"{self.robot_name}: random contact mode has zero candidate points")

        lo = max(1, min(int(self.probe_contact_count_range[0]), num_candidates))
        hi = max(lo, min(int(self.probe_contact_count_range[1]), num_candidates))
        # Shuffle candidates by random score, then keep the first count entries.
        counts = torch.randint(lo, hi + 1, (int(batch_size),), device=self.device, generator=generator)
        scores = torch.rand((int(batch_size), num_candidates), dtype=torch.float32, device=self.device, generator=generator)
        shuffled_idx = scores.argsort(dim=1, descending=True)
        shuffled_points = candidate_point_indices.view(1, -1).expand(int(batch_size), -1).gather(1, shuffled_idx)
        rank = torch.arange(num_candidates, device=self.device).view(1, -1)
        point_valid_mask = rank < counts.unsqueeze(1)
        filler = candidate_point_indices[0].view(1, 1).expand(int(batch_size), num_candidates)
        point_indices = torch.where(point_valid_mask, shuffled_points, filler)
        return point_indices, point_valid_mask

    def _sample_sparse_contact_point_indices_batch(
        self,
        *,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sparse_candidate_indices is None:
            raise RuntimeError("sparse contact sampling requested without sparse candidates")
        if self.sparse_sampling_mode == "point_uniform":
            return self._sample_random_contact_point_indices_batch(
                candidate_point_indices=self.sparse_candidate_indices,
                batch_size=batch_size,
                generator=generator,
            )

        if self.sparse_region_candidate_indices is None:
            raise RuntimeError("region_uniform sparse sampling requested without region candidates")
        nonempty_regions = [idx for idx in self.sparse_region_candidate_indices if int(idx.numel()) > 0]
        if len(nonempty_regions) == 0:
            raise RuntimeError(f"{self.robot_name}: no non-empty sparse contact regions")
        num_candidates = int(self.sparse_candidate_indices.numel())
        lo = max(1, min(int(self.probe_contact_count_range[0]), num_candidates))
        hi = max(lo, min(int(self.probe_contact_count_range[1]), num_candidates))
        batch_size = int(batch_size)
        counts = torch.randint(lo, hi + 1, (batch_size,), device=self.device, generator=generator)
        region_choice = torch.randint(
            0,
            len(nonempty_regions),
            (batch_size, hi),
            device=self.device,
            generator=generator,
        )
        filler = self.sparse_candidate_indices[0]
        point_indices = filler.view(1, 1).expand(batch_size, hi).clone()
        for region_idx, candidates in enumerate(nonempty_regions):
            sel = region_choice == int(region_idx)
            n_sel = int(sel.sum().item())
            if n_sel <= 0:
                continue
            choice = torch.randint(
                0,
                int(candidates.numel()),
                (n_sel,),
                device=self.device,
                generator=generator,
            )
            point_indices[sel] = candidates.index_select(0, choice)
        rank = torch.arange(hi, device=self.device).view(1, -1)
        point_valid_mask = rank < counts.unsqueeze(1)
        point_indices = torch.where(
            point_valid_mask,
            point_indices,
            point_indices[:, :1].expand_as(point_indices),
        )
        return point_indices, point_valid_mask

    def _sample_composite_sparse_contact_point_indices_batch(
        self,
        *,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        groups = list(getattr(self.model, "composite_contact_groups", ()))
        if len(groups) == 0:
            raise RuntimeError("Composite sparse sampling requested without composite_contact_groups")
        template_link_indices = self.model.surface_template_link_indices.to(device=self.device, dtype=torch.long)
        allowed = self.contact_allowed_surface_mask.to(device=self.device, dtype=torch.bool)
        total_slots = sum(int(g.get("slot_count", 0)) for g in groups)
        if total_slots <= 0:
            raise RuntimeError(f"{self.robot_name}: composite contact groups have no slots")
        point_indices = torch.zeros((int(batch_size), total_slots), dtype=torch.long, device=self.device)
        point_valid_mask = torch.zeros((int(batch_size), total_slots), dtype=torch.bool, device=self.device)

        slot_start = 0
        for group in groups:
            name = str(group["name"])
            slot_count = int(group.get("slot_count", 0))
            if slot_count <= 0:
                continue
            link_ids = torch.as_tensor(group.get("link_indices", ()), dtype=torch.long, device=self.device)
            group_allowed = torch.zeros_like(allowed)
            for link_id in link_ids.tolist():
                group_allowed |= template_link_indices == int(link_id)
            candidates = torch.nonzero(allowed & group_allowed, as_tuple=False).view(-1)
            if int(candidates.numel()) <= 0:
                raise RuntimeError(f"{self.robot_name}: composite contact group {name!r} has zero candidate points")
            lo, hi = self.composite_contact_count_ranges.get(
                name,
                tuple(int(x) for x in group.get("default_contact_count_range", (1, min(3, slot_count)))),
            )
            hi = min(int(hi), int(slot_count), int(candidates.numel()))
            lo = max(1, min(int(lo), hi))
            counts = torch.randint(lo, hi + 1, (int(batch_size),), dtype=torch.long, device=self.device, generator=generator)
            scores = torch.rand((int(batch_size), int(candidates.numel())), dtype=torch.float32, device=self.device, generator=generator)
            shuffled = candidates.view(1, -1).expand(int(batch_size), -1).gather(1, scores.argsort(dim=1, descending=True))
            local_rank = torch.arange(slot_count, dtype=torch.long, device=self.device).view(1, -1)
            valid = local_rank < counts.view(-1, 1)
            filler = candidates[0].view(1, 1).expand(int(batch_size), slot_count)
            take = shuffled[:, :slot_count] if int(shuffled.shape[1]) >= slot_count else filler.clone()
            if int(shuffled.shape[1]) < slot_count:
                take[:, : int(shuffled.shape[1])] = shuffled
            point_indices[:, slot_start : slot_start + slot_count] = torch.where(valid, take, filler)
            point_valid_mask[:, slot_start : slot_start + slot_count] = valid
            slot_start += slot_count
        return point_indices, point_valid_mask

    def _build_exact_contact_bank_from_point_slots(
        self,
        *,
        q2_sampled: torch.Tensor,
        point_indices: torch.Tensor,
        point_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = int(q2_sampled.shape[0])
        target_k = int(self.candidate_contact_points)
        slot_k = int(point_indices.shape[1])
        if slot_k > target_k:
            raise RuntimeError(
                f"{self.robot_name}: composite contact slots ({slot_k}) exceed max_contact_points={target_k}"
            )
        point_indices = point_indices.to(device=self.device, dtype=torch.long)
        point_valid_mask = point_valid_mask.to(device=self.device, dtype=torch.bool)
        surface_mask = self._build_contact_mask_from_indices(
            batch_size,
            point_indices,
            point_valid_mask,
        )

        template_points_local = self.model.surface_template_points_local.to(device=self.device, dtype=torch.float32)
        template_normals_local = self.model.surface_template_normals_local.to(device=self.device, dtype=torch.float32)
        template_link_indices = self.model.surface_template_link_indices.to(device=self.device, dtype=torch.long)
        contact_local_points = torch.zeros((batch_size, target_k, 3), dtype=torch.float32, device=self.device)
        contact_local_normals = torch.zeros((batch_size, target_k, 3), dtype=torch.float32, device=self.device)
        contact_link_indices = torch.full((batch_size, target_k), -1, dtype=torch.long, device=self.device)
        contact_parent_sparse_indices = torch.full((batch_size, target_k), -1, dtype=torch.long, device=self.device)
        contact_valid_mask = torch.zeros((batch_size, target_k), dtype=torch.bool, device=self.device)
        contact_local_points[:, :slot_k] = template_points_local.index_select(0, point_indices.reshape(-1)).view(batch_size, slot_k, 3)
        contact_local_normals[:, :slot_k] = template_normals_local.index_select(0, point_indices.reshape(-1)).view(batch_size, slot_k, 3)
        contact_link_indices[:, :slot_k] = template_link_indices.index_select(0, point_indices.reshape(-1)).view(batch_size, slot_k)
        contact_parent_sparse_indices[:, :slot_k] = point_indices
        contact_valid_mask[:, :slot_k] = point_valid_mask
        contact_link_indices = contact_link_indices.masked_fill(~contact_valid_mask, -1)
        contact_parent_sparse_indices = contact_parent_sparse_indices.masked_fill(~contact_valid_mask, -1)

        contact_points_world, contact_normals_world = self.model.transform_attached_points_normals_batch(
            q=q2_sampled,
            local_points=contact_local_points,
            link_indices=contact_link_indices,
            local_normals=contact_local_normals,
        )
        if contact_normals_world is None:
            contact_normals_world = torch.zeros_like(contact_points_world)
        contact_cloud = torch.cat([contact_points_world, contact_normals_world], dim=-1)
        contact_cloud = contact_cloud.masked_fill(~contact_valid_mask.unsqueeze(-1), 0.0)
        contact_local_points = contact_local_points.masked_fill(~contact_valid_mask.unsqueeze(-1), 0.0)
        contact_local_normals = contact_local_normals.masked_fill(~contact_valid_mask.unsqueeze(-1), 0.0)

        surface_cluster_ids = self._build_surface_cluster_ids(surface_mask)
        safe_parent = contact_parent_sparse_indices.clamp_min(0)
        contact_cluster_ids = torch.gather(surface_cluster_ids, 1, safe_parent)
        contact_cluster_ids = contact_cluster_ids.masked_fill(~contact_valid_mask, -1)
        return {
            "contact_cloud": contact_cloud,
            "contact_valid_mask": contact_valid_mask,
            "contact_local_points": contact_local_points,
            "contact_local_normals": contact_local_normals,
            "contact_link_indices": contact_link_indices,
            "contact_parent_sparse_indices": contact_parent_sparse_indices,
            "contact_cluster_ids": contact_cluster_ids,
            "surface_mask": surface_mask,
        }

    def _build_surface_cluster_ids(
        self,
        surface_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Label connected components in the active surface mask using the template graph.
        batch_size = int(surface_mask.shape[0])
        cluster_ids = torch.full(
            (batch_size, self.surface_num_points),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        neighbors = self.model.surface_graph_neighbors
        if len(neighbors) != int(self.surface_num_points):
            raise RuntimeError(
                "Surface connectivity graph is not aligned with the probe surface template. "
                "Rebuild the robot surface artifact."
            )

        for batch_idx in range(batch_size):
            # Run a small DFS over active template points for this sample.
            active = surface_mask[batch_idx].detach().to(device="cpu", dtype=torch.bool).tolist()
            visited = [False] * int(self.surface_num_points)
            cluster_idx = 0
            for root in range(int(self.surface_num_points)):
                if (not active[root]) or visited[root]:
                    continue
                stack = [root]
                visited[root] = True
                members: list[int] = []
                while stack:
                    cur = stack.pop()
                    members.append(cur)
                    for nxt in neighbors[cur]:
                        nxt_i = int(nxt)
                        if active[nxt_i] and (not visited[nxt_i]):
                            visited[nxt_i] = True
                            stack.append(nxt_i)
                cluster_ids[batch_idx, members] = int(cluster_idx)
                cluster_idx += 1
        return cluster_ids

    def _expand_contact_bank_from_surface_mask(
        self,
        *,
        q2_sampled: torch.Tensor,
        surface_mask: torch.Tensor,
        generator: torch.Generator,
        focused_object_points: torch.Tensor | None = None,
        focused_object_valid_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        # Expand a dense surface mask into a contact bank and attach parent/cluster metadata.
        contact_obs = self._expand_contact_cloud_batch(
            model=self.model,
            q=q2_sampled,
            surface_mask=surface_mask,
            generator=generator,
            focused_object_points=focused_object_points,
            focused_object_valid_mask=focused_object_valid_mask,
        )
        if "contact_parent_sparse_indices" not in contact_obs:
            raise ValueError("Probe contact expansion is missing contact_parent_sparse_indices")
        surface_cluster_ids = self._build_surface_cluster_ids(surface_mask)
        contact_parent_sparse_indices = contact_obs["contact_parent_sparse_indices"].to(device=self.device, dtype=torch.long)
        contact_valid_mask = contact_obs["contact_valid_mask"].to(device=self.device, dtype=torch.bool)
        safe_parent = contact_parent_sparse_indices.clamp_min(0)
        contact_cluster_ids = torch.gather(surface_cluster_ids, 1, safe_parent)
        contact_cluster_ids = contact_cluster_ids.masked_fill(~contact_valid_mask, -1)
        contact_obs["surface_mask"] = surface_mask.bool()
        contact_obs["contact_parent_sparse_indices"] = contact_parent_sparse_indices
        contact_obs["contact_cluster_ids"] = contact_cluster_ids
        return contact_obs

    def _build_exact_contact_bank_from_surface_mask(
        self,
        *,
        q2_sampled: torch.Tensor,
        surface_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Truly sparse path: one contact per active template point, no bank expansion.
        # Each active surface-template index becomes exactly one contact slot; the rest of
        # the candidate_contact_points buffer is left invalid so runtime resampling cannot
        # synthesize extra points around each sparse parent.
        batch_size = int(q2_sampled.shape[0])
        target_k = int(self.candidate_contact_points)

        template_points_local = self.model.surface_template_points_local.to(
            device=self.device, dtype=torch.float32
        )
        template_normals_local = self.model.surface_template_normals_local.to(
            device=self.device, dtype=torch.float32
        )
        template_link_indices = self.model.surface_template_link_indices.to(
            device=self.device, dtype=torch.long
        )

        contact_local_points = torch.zeros(
            (batch_size, target_k, 3), dtype=torch.float32, device=self.device
        )
        contact_local_normals = torch.zeros(
            (batch_size, target_k, 3), dtype=torch.float32, device=self.device
        )
        contact_link_indices = torch.full(
            (batch_size, target_k), -1, dtype=torch.long, device=self.device
        )
        contact_parent_sparse_indices = torch.full(
            (batch_size, target_k), -1, dtype=torch.long, device=self.device
        )
        contact_valid_mask = torch.zeros(
            (batch_size, target_k), dtype=torch.bool, device=self.device
        )

        surface_mask_bool = surface_mask.to(device=self.device).bool()
        for batch_idx in range(batch_size):
            active_idx = torch.nonzero(surface_mask_bool[batch_idx], as_tuple=False).view(-1)
            n_active = int(active_idx.numel())
            if n_active <= 0:
                continue
            n_fill = min(n_active, target_k)
            sel = active_idx[:n_fill]
            contact_local_points[batch_idx, :n_fill] = template_points_local.index_select(0, sel)
            contact_local_normals[batch_idx, :n_fill] = template_normals_local.index_select(0, sel)
            contact_link_indices[batch_idx, :n_fill] = template_link_indices.index_select(0, sel)
            contact_parent_sparse_indices[batch_idx, :n_fill] = sel
            contact_valid_mask[batch_idx, :n_fill] = True

        contact_points_world, contact_normals_world = self.model.transform_attached_points_normals_batch(
            q=q2_sampled,
            local_points=contact_local_points,
            link_indices=contact_link_indices,
            local_normals=contact_local_normals,
        )
        if contact_normals_world is None:
            contact_normals_world = torch.zeros_like(contact_points_world)
        contact_cloud = torch.cat([contact_points_world, contact_normals_world], dim=-1)
        contact_cloud = contact_cloud.masked_fill(~contact_valid_mask.unsqueeze(-1), 0.0)
        contact_local_points = contact_local_points.masked_fill(~contact_valid_mask.unsqueeze(-1), 0.0)
        contact_local_normals = contact_local_normals.masked_fill(~contact_valid_mask.unsqueeze(-1), 0.0)

        surface_cluster_ids = self._build_surface_cluster_ids(surface_mask_bool)
        safe_parent = contact_parent_sparse_indices.clamp_min(0)
        contact_cluster_ids = torch.gather(surface_cluster_ids, 1, safe_parent)
        contact_cluster_ids = contact_cluster_ids.masked_fill(~contact_valid_mask, -1)

        return {
            "contact_cloud": contact_cloud,
            "contact_valid_mask": contact_valid_mask,
            "contact_local_points": contact_local_points,
            "contact_local_normals": contact_local_normals,
            "contact_link_indices": contact_link_indices,
            "contact_parent_sparse_indices": contact_parent_sparse_indices,
            "contact_cluster_ids": contact_cluster_ids,
            "surface_mask": surface_mask_bool,
        }

    # Tips/random modes use fixed template candidates; patches mode uses the synthetic contact extractor.
    def _build_contact_observation_batch(
        self,
        *,
        q2_sampled: torch.Tensor,
        q2_points: torch.Tensor,
        q2_normals: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, int]]:
        # Build the fixed contact bank for the configured contact source mode.
        batch_size = int(q2_sampled.shape[0])
        if self.contact_mode == "tips":
            # Fixed tips use every configured representative point in every sample.
            point_indices = self.fixed_contact_point_indices.view(1, -1).expand(batch_size, -1)
            surface_mask = self._build_contact_mask_from_indices(
                batch_size,
                point_indices,
            )
            return (
                self._expand_contact_bank_from_surface_mask(
                    q2_sampled=q2_sampled,
                    surface_mask=surface_mask,
                    generator=generator,
                ),
                _init_contact_count_interval_stats(),
            )
        if self.contact_mode in {"random_links", "random_tips", "sparse"}:
            # Random modes sample subsets from either all links, the configured tip links,
            # or the entire allowed manipulator surface (sparse).
            if self.contact_mode == "random_links":
                candidate_point_indices = self.random_link_point_indices
            elif self.contact_mode == "random_tips":
                candidate_point_indices = self.fixed_contact_point_indices
            else:  # sparse
                candidate_point_indices = self.sparse_candidate_indices
            if self.contact_mode == "sparse":
                if self._has_composite_contact_groups():
                    point_indices, point_valid_mask = self._sample_composite_sparse_contact_point_indices_batch(
                        batch_size=batch_size,
                        generator=generator,
                    )
                else:
                    point_indices, point_valid_mask = self._sample_sparse_contact_point_indices_batch(
                        batch_size=batch_size,
                        generator=generator,
                    )
            else:
                point_indices, point_valid_mask = self._sample_random_contact_point_indices_batch(
                    candidate_point_indices=candidate_point_indices,
                    batch_size=batch_size,
                    generator=generator,
                )
            surface_mask = self._build_contact_mask_from_indices(
                batch_size,
                point_indices,
                point_valid_mask,
            )
            if self.contact_mode == "sparse":
                if self._has_composite_contact_groups():
                    return (
                        self._build_exact_contact_bank_from_point_slots(
                            q2_sampled=q2_sampled,
                            point_indices=point_indices,
                            point_valid_mask=point_valid_mask,
                        ),
                        _init_contact_count_interval_stats(),
                    )
                # Sparse mode must stay truly sparse: one contact per active template
                # point with no bank expansion, so downstream runtime resampling can only
                # duplicate the original parents rather than invent dense neighborhoods.
                return (
                    self._build_exact_contact_bank_from_surface_mask(
                        q2_sampled=q2_sampled,
                        surface_mask=surface_mask,
                    ),
                    _init_contact_count_interval_stats(),
                )
            return (
                self._expand_contact_bank_from_surface_mask(
                    q2_sampled=q2_sampled,
                    surface_mask=surface_mask,
                    generator=generator,
                ),
                _init_contact_count_interval_stats(),
            )

        # Patch mode synthesizes object patches, scores surface contacts, and clamps count range.
        patch_params = self._sample_probe_patch_params(batch_size=batch_size, generator=generator)
        object_points, object_valid_mask = self.model.sample_object_patches_batch(
            hand_points=q2_points,
            hand_normals=q2_normals,
            anchor_indices=patch_params["anchor_indices"],
            anchor_valid_mask=patch_params["anchor_valid_mask"],
            anchor_point_valid_mask=patch_params["anchor_point_valid_mask"],
            generator=generator,
            anchor_shift_min=self.patch_anchor_shift_min,
            anchor_shift_max=self.patch_anchor_shift_max,
            max_plane_extent_min=self.patch_extent_min,
            max_plane_extent_max=self.patch_extent_max,
            patch_shift_power=self.patch_shift_power,
            patch_extent_power=self.patch_extent_power,
            normal_jitter_max_deg=self.patch_normal_jitter_max_deg,
            penetration_clearance=self.patch_penetration_clearance,
        )
        flat_object_points = object_points.view(batch_size, -1, 3)
        flat_object_valid_mask = object_valid_mask.view(batch_size, -1)
        contact_values = self.model._compute_gendex_contact_value_source_target_batch(
            source_points=q2_points,
            source_normals=q2_normals,
            target_points=flat_object_points,
            target_valid_mask=flat_object_valid_mask,
            align_exp_scale=self.align_exp_scale,
            sigmoid_scale=self.sigmoid_scale,
        )
        allowed_mask = self.contact_allowed_surface_mask.view(1, -1).expand(batch_size, -1)
        contact_values = contact_values.masked_fill(~allowed_mask, -1.0e9)
        surface_mask, count_interval_stats = self._build_surface_mask_batch(
            contact_values,
            self.probe_contact_count_range,
        )
        surface_mask = surface_mask & allowed_mask
        contact_obs = self._expand_contact_bank_from_surface_mask(
            q2_sampled=q2_sampled,
            surface_mask=surface_mask,
            generator=generator,
            focused_object_points=object_points,
            focused_object_valid_mask=object_valid_mask,
        )
        return contact_obs, count_interval_stats

    # The fixed buffer stores contact geometry and q2; runtime sampling replaces q1/action in __getitem__.
    @torch.no_grad()
    def _generate_batch(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, int]]:
        # Sample q2 (the contact-conditioned pose source) using the configured probe pose mode.
        q2_probe = self._sample_probe_q_batch(
            batch_size=int(batch_size),
            generator=generator,
            sixd_pose_mode=self.sixd_pose_mode,
        )
        # Query full hand surface points/normals at q2; patch/tips extraction uses these tensors.
        q2_points, q2_normals = self.model.get_surface_points_normals_batch(q=q2_probe)
        # Build contact observation tensors (cloud/masks/local metadata) from q2 geometry.
        obs, count_interval_stats = self._build_contact_observation_batch(
            q2_sampled=q2_probe,
            q2_points=q2_points,
            q2_normals=q2_normals,
            generator=generator,
        )

        # q2_probe_eff is the action-space q2 actually written to the batch (may be recentered below).
        q2_probe_eff = q2_probe.clone()
        contact_cloud = obs["contact_cloud"]
        if self.sixd_pose_mode == "zero_centered":
            # Recenter contact xyz and base translation so coordinates are expressed in contact-centered frame.
            contact_center = self._compute_contact_center_from_surface_mask(
                q2_points,
                obs["surface_mask"],
            )
            q2_probe_eff, contact_cloud = self._apply_zero_centering(
                q2=q2_probe,
                contact_cloud=contact_cloud,
                contact_valid_mask=obs["contact_valid_mask"],
                contact_center=contact_center,
            )

        # Legacy q1/action fields keep older buffer payloads complete; sampled batches rebuild them at runtime.
        obs_for_q1 = dict(obs)
        obs_for_q1["contact_cloud"] = contact_cloud
        q1_probe = self._build_q1_for_contacts_batch(
            q2_probe=q2_probe_eff,
            contact_obs=obs_for_q1,
            generator=generator,
            fk_model=self.model,
        )
        target_q_probe = self.model.mix_q_by_contact_mask(q1_probe, q2_probe_eff, obs["surface_mask"])

        # Assemble the reusable contact/q2 fields shared across all q-only probe models.
        batch: Dict[str, torch.Tensor] = {
            "contact_cloud": contact_cloud,
            "contact_valid_mask": obs["contact_valid_mask"],
            "contact_local_points": obs["contact_local_points"],
            "contact_local_normals": obs["contact_local_normals"],
            "contact_link_indices": obs["contact_link_indices"],
            "contact_parent_sparse_indices": obs["contact_parent_sparse_indices"],
            "contact_cluster_ids": obs["contact_cluster_ids"],
            "surface_mask": obs["surface_mask"],
            "action": self._pad_action_batch(target_q_probe),
            "q2_padded": self._pad_action_batch(q2_probe_eff),
            "q1_padded": self._pad_action_batch(q1_probe),
        }
        return batch, count_interval_stats

    # Build and store the fixed split once, with summary stats for patch-based contacts.
    def _populate_buffer(self) -> None:
        # Fill every probe buffer slot in generation batches using one deterministic generator.
        generator = self._make_generator(self.seed)
        count_interval_stats = _init_contact_count_interval_stats()
        progress = tqdm(total=self.samples, desc=f"{self.progress_label}:build", leave=False)
        offset = 0
        while offset < self.samples:
            cur = min(self.build_batch_size, self.samples - offset)
            batch, batch_stats = self._generate_batch(cur, generator)
            _merge_contact_count_interval_stats(count_interval_stats, batch_stats)
            self._write_batch_to_buffer(slice(offset, offset + cur), batch)
            offset += cur
            progress.update(cur)
        progress.close()
        if self.contact_mode == "patches":
            print(
                _format_contact_count_interval_summary(
                    f"{self.progress_label}:{self.robot_name}",
                    count_interval_stats,
                )
            )

    def _build_runtime_generator(
        self,
        idx: int,
        *,
        resample_variant: int = 0,
        seed_offset: int = 17,
    ) -> torch.Generator | None:
        # Build deterministic per-sample CPU generators when runtime resampling must be repeatable.
        if not self.runtime_resample_deterministic:
            return None
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(self.seed) * 1_000_003
            + int(idx) * 9_176
            + int(max(0, resample_variant)) * 37_153
            + int(seed_offset)
        )
        return generator

    def _resample_contact_indices(
        self,
        *,
        idx: int,
        resample_variant: int = 0,
    ) -> torch.Tensor:
        # Resample a fixed-size contact subset while preserving connected component coverage.
        contact_valid_mask = self.buffer_contact_valid_mask[idx].to(dtype=torch.bool)
        contact_cluster_ids = self.buffer_contact_cluster_ids[idx].to(dtype=torch.long)
        valid_mask = contact_valid_mask & (contact_cluster_ids >= 0)
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).view(-1)
        if int(valid_indices.numel()) <= 0:
            fallback = torch.zeros((self.contact_input_points,), dtype=torch.long)
            return fallback

        candidate_cluster_ids = contact_cluster_ids.index_select(0, valid_indices)
        unique_clusters = torch.unique(candidate_cluster_ids, sorted=True)
        generator = self._build_runtime_generator(idx, resample_variant=resample_variant, seed_offset=17)
        device = valid_indices.device

        if int(unique_clusters.numel()) > self.contact_input_points:
            # Too many clusters: choose one point from a subset of clusters.
            cluster_choice = torch.randperm(int(unique_clusters.numel()), generator=generator, device=device)[
                : self.contact_input_points
            ]
            active_clusters = unique_clusters.index_select(0, cluster_choice)
            cluster_counts = torch.ones((int(active_clusters.numel()),), dtype=torch.long, device=device)
        else:
            # Few clusters: allocate at least one point per cluster, then distribute remaining slots.
            active_clusters = unique_clusters
            cluster_counts = torch.ones((int(active_clusters.numel()),), dtype=torch.long, device=device)
            remaining = int(self.contact_input_points - int(active_clusters.numel()))
            if remaining > 0 and int(active_clusters.numel()) > 0:
                extra_assignments = torch.randint(
                    0,
                    int(active_clusters.numel()),
                    (remaining,),
                    device=device,
                    generator=generator,
                )
                cluster_counts = cluster_counts + torch.bincount(
                    extra_assignments,
                    minlength=int(active_clusters.numel()),
                )

        selected_chunks: list[torch.Tensor] = []
        for cluster_offset, cluster_id in enumerate(active_clusters.tolist()):
            # Draw the requested number of points from each active cluster, with replacement if needed.
            cluster_member_mask = candidate_cluster_ids == int(cluster_id)
            cluster_candidates = valid_indices[cluster_member_mask]
            required = int(cluster_counts[cluster_offset].item())
            if required <= 0 or int(cluster_candidates.numel()) <= 0:
                continue
            if required <= int(cluster_candidates.numel()):
                perm = torch.randperm(int(cluster_candidates.numel()), generator=generator, device=device)[:required]
                chosen = cluster_candidates.index_select(0, perm)
            else:
                perm = torch.randperm(int(cluster_candidates.numel()), generator=generator, device=device)
                chosen = cluster_candidates.index_select(0, perm)
                extra = torch.randint(
                    0,
                    int(cluster_candidates.numel()),
                    (required - int(cluster_candidates.numel()),),
                    device=device,
                    generator=generator,
                )
                chosen = torch.cat([chosen, cluster_candidates.index_select(0, extra)], dim=0)
            selected_chunks.append(chosen)

        if len(selected_chunks) == 0:
            replacement = torch.randint(
                0,
                int(valid_indices.numel()),
                (self.contact_input_points,),
                device=device,
                generator=generator,
            )
            return valid_indices.index_select(0, replacement)

        selected_indices = torch.cat(selected_chunks, dim=0)
        if int(selected_indices.numel()) < self.contact_input_points:
            # Pad short selections from all valid contact points.
            extra = torch.randint(
                0,
                int(valid_indices.numel()),
                (self.contact_input_points - int(selected_indices.numel()),),
                device=device,
                generator=generator,
            )
            selected_indices = torch.cat([selected_indices, valid_indices.index_select(0, extra)], dim=0)
        elif int(selected_indices.numel()) > self.contact_input_points:
            # Trim oversized selections back to the requested contact input width.
            trim = torch.randperm(int(selected_indices.numel()), generator=generator, device=device)[
                : self.contact_input_points
            ]
            selected_indices = selected_indices.index_select(0, trim)
        order = torch.randperm(int(selected_indices.numel()), generator=generator, device=device)
        return selected_indices.index_select(0, order)

    def _build_sparse_contact_observation(self, idx: int) -> Dict[str, torch.Tensor]:
        # Sparse-mode subset builder: shared between the legacy CPU path and the GPU
        # fast path via ``_sparse_subset_indices_and_valid`` so both emit the same
        # contact-cloud distribution (variable-length with invalid pads, or
        # equal-duplication with all-valid slots — see module helper docstring).
        if self._is_composite_sparse_contacts() and self.contact_input_points == -1:
            selected_indices = torch.arange(self._sparse_max_real_contacts, dtype=torch.long)
            selected_valid = self.buffer_contact_valid_mask[idx, : self._sparse_max_real_contacts].clone()
        else:
            selected_indices, selected_valid = _sparse_subset_indices_and_valid(
                self.buffer_contact_valid_mask[idx],
                contact_input_points=self.contact_input_points,
                max_real_contacts=self._sparse_max_real_contacts,
            )
        return {
            "contact_cloud": self.buffer_contact_cloud[idx].index_select(0, selected_indices).clone(),
            "contact_valid_mask": selected_valid.clone(),
            "contact_local_points": self.buffer_contact_local_points[idx].index_select(0, selected_indices).clone(),
            "contact_local_normals": self.buffer_contact_local_normals[idx].index_select(0, selected_indices).clone(),
            "contact_link_indices": self.buffer_contact_link_indices[idx].index_select(0, selected_indices).to(dtype=torch.long),
        }

    def _build_resampled_contact_observation(
        self,
        idx: int,
        *,
        resample_variant: int = 0,
    ) -> Dict[str, torch.Tensor]:
        # Gather the runtime contact subset from the fixed contact bank.
        if self.contact_mode == "sparse":
            return self._build_sparse_contact_observation(idx)
        selected_indices = self._resample_contact_indices(idx=idx, resample_variant=resample_variant)
        return {
            "contact_cloud": self.buffer_contact_cloud[idx].index_select(0, selected_indices).clone(),
            "contact_valid_mask": self.buffer_contact_valid_mask[idx].index_select(0, selected_indices).clone(),
            "contact_local_points": self.buffer_contact_local_points[idx].index_select(0, selected_indices).clone(),
            "contact_local_normals": self.buffer_contact_local_normals[idx].index_select(0, selected_indices).clone(),
            "contact_link_indices": self.buffer_contact_link_indices[idx].index_select(0, selected_indices).to(dtype=torch.long),
        }

    def _build_joint_active_by_contact_mask_cpu(self, mask: torch.Tensor) -> torch.Tensor:
        # Convert active surface points into active joints through descendant-link ownership.
        squeeze = False
        mask = mask.detach().cpu().bool()
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
            squeeze = True
        if mask.ndim != 2 or int(mask.shape[-1]) != int(self.surface_num_points):
            raise ValueError(
                f"mask must have shape (B, {self.surface_num_points}), got {tuple(mask.shape)}"
            )
        active_links = torch.zeros(
            (int(mask.shape[0]), len(self.model.mesh_link_names)),
            dtype=torch.bool,
            device="cpu",
        )
        # Collapse point masks to link masks before expanding links to ancestor joints.
        link_idx = self._surface_template_link_indices_cpu
        for link_i in range(len(self.model.mesh_link_names)):
            surface_mask = link_idx == int(link_i)
            if bool(surface_mask.any()):
                active_links[:, link_i] = mask[:, surface_mask].any(dim=1)
        joint_active = (
            active_links.unsqueeze(1)
            & self._joint_descendant_link_mask_cpu.unsqueeze(0)
        ).any(dim=-1)
        if squeeze:
            return joint_active.squeeze(0)
        return joint_active

    def _mix_q_by_contact_mask_cpu(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # CPU version of contact-aware q mixing for runtime q/action regeneration.
        squeeze = False
        if q1.ndim == 1:
            q1 = q1.unsqueeze(0)
            squeeze = True
        if q2.ndim == 1:
            q2 = q2.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if q1.shape != q2.shape:
            raise ValueError(f"q1 and q2 must have the same shape, got {tuple(q1.shape)} vs {tuple(q2.shape)}")
        if q1.ndim != 2 or int(q1.shape[-1]) != int(self.local_dof):
            raise ValueError(f"Expected q tensors with shape (B, {self.local_dof}), got {tuple(q1.shape)}")
        joint_active = self._build_joint_active_by_contact_mask_cpu(mask)
        mixed = torch.where(joint_active, q2, q1)
        mixed = self._apply_fixed_zero_joints_cpu(mixed)
        if squeeze:
            return mixed.squeeze(0)
        return mixed

    def _pad_runtime_q_batch_cpu(self, q: torch.Tensor) -> torch.Tensor:
        # Pad runtime local q tensors into the shared action dimension on CPU.
        if q.ndim == 1:
            q = q.unsqueeze(0)
        out = torch.zeros((int(q.shape[0]), self.action_dim), dtype=torch.float32, device="cpu")
        out[:, : int(q.shape[-1])] = q.detach().cpu().to(dtype=torch.float32)
        return out

    def _build_runtime_q1_batch(
        self,
        q2_batch: torch.Tensor,
        indices: list[int],
        *,
        resample_variant: int = 0,
    ) -> torch.Tensor:
        # Build one runtime q1 per requested sample, optionally using random or zero-pose starts.
        if self.q1_sampling_mode == "contact_error_mixture":
            if self._cpu_hand_model is None:
                raise RuntimeError("contact_error_mixture runtime q1 sampling requires a CPU robot model")
            surface_mask = self.buffer_surface_mask[indices].to(dtype=torch.bool)
            contact_local_points = self.buffer_contact_local_points[indices].to(dtype=torch.float32)
            contact_link_indices = self.buffer_contact_link_indices[indices].to(dtype=torch.long)
            contact_valid_mask = self.buffer_contact_valid_mask[indices].to(dtype=torch.bool)
            target_points = self.buffer_contact_cloud[indices, :, :3].to(dtype=torch.float32)
            if self.runtime_resample_deterministic:
                q1_rows: list[torch.Tensor] = []
                for row_idx, idx in enumerate(indices):
                    generator = self._build_runtime_generator(
                        int(idx),
                        resample_variant=resample_variant,
                        seed_offset=211,
                    )
                    q1_row = self._build_q1_contact_error_mixture_batch(
                        q2_probe=q2_batch[row_idx : row_idx + 1],
                        surface_mask=surface_mask[row_idx : row_idx + 1],
                        contact_local_points=contact_local_points[row_idx : row_idx + 1],
                        contact_link_indices=contact_link_indices[row_idx : row_idx + 1],
                        contact_valid_mask=contact_valid_mask[row_idx : row_idx + 1],
                        target_contact_points=target_points[row_idx : row_idx + 1],
                        generator=generator,
                        fk_model=self._cpu_hand_model,
                    )
                    q1_rows.append(q1_row[0])
                if len(q1_rows) == 0:
                    return torch.zeros((0, self.local_dof), dtype=torch.float32)
                return torch.stack(q1_rows, dim=0)
            return self._build_q1_contact_error_mixture_batch(
                q2_probe=q2_batch,
                surface_mask=surface_mask,
                contact_local_points=contact_local_points,
                contact_link_indices=contact_link_indices,
                contact_valid_mask=contact_valid_mask,
                target_contact_points=target_points,
                generator=None,
                fk_model=self._cpu_hand_model,
            )

        q1_rows: list[torch.Tensor] = []
        for row_idx, idx in enumerate(indices):
            generator = self._build_runtime_generator(
                int(idx),
                resample_variant=resample_variant,
                seed_offset=211,
            )
            if self.random_q1:
                q1_row = self._sample_random_q1_batch_cpu(1, generator=generator)
            else:
                q2_row = q2_batch[row_idx : row_idx + 1]
                use_zero = False
                if self.q1_zero_probability >= 1.0:
                    use_zero = True
                elif self.q1_zero_probability > 0.0:
                    use_zero = bool(torch.rand((), generator=generator).item() < self.q1_zero_probability)

                if use_zero:
                    # Zero-q augmentation gives the probe a canonical starting pose for some samples.
                    q1_row = self._apply_probe_pose_mode_constraints_cpu(
                        torch.zeros_like(q2_row),
                        sixd_pose_mode=self.sixd_pose_mode,
                    )
                else:
                    q1_row = self._build_q1_from_q2_batch_cpu(q2_row, generator=generator)
            q1_rows.append(q1_row[0])
        if len(q1_rows) == 0:
            return torch.zeros((0, self.local_dof), dtype=torch.float32)
        return torch.stack(q1_rows, dim=0)

    def _build_runtime_q_batch(
        self,
        indices: list[int],
        *,
        resample_variant: int = 0,
    ) -> list[Dict[str, torch.Tensor]]:
        # Regenerate q1/action for a batch of fixed-buffer q2/contact masks at fetch time.
        if len(indices) == 0:
            return []
        q2_batch = self.buffer_q2_padded[indices, : self.local_dof].to(dtype=torch.float32)
        surface_mask = self.buffer_surface_mask[indices].to(dtype=torch.bool)
        q1_batch = self._build_runtime_q1_batch(
            q2_batch,
            indices,
            resample_variant=resample_variant,
        )
        q_seed_batch = self._mix_q_by_contact_mask_cpu(q1_batch, q2_batch, surface_mask)
        action_batch = q_seed_batch
        empty_mask = torch.zeros((len(indices),), dtype=torch.bool)
        changed_joint_mask = torch.zeros((len(indices), self.local_dof), dtype=torch.bool)
        if self.non_contact_causal_enabled:
            contact_joint_mask = self._build_joint_active_by_contact_mask_cpu(surface_mask)[:, : int(self.local_dof)]
            q_goal_rows: list[torch.Tensor] = []
            changed_rows: list[torch.Tensor] = []
            empty_rows: list[torch.Tensor] = []
            for row_idx, idx in enumerate(indices):
                generator = self._build_runtime_generator(
                    int(idx),
                    resample_variant=resample_variant,
                    seed_offset=409,
                )
                empty_row = self._sample_causal_non_contact_empty_mask(
                    1,
                    device=torch.device("cpu"),
                    generator=generator,
                )
                q_goal_row, changed_row = self._sample_non_contact_goal_batch(
                    q_seed=q_seed_batch[row_idx : row_idx + 1],
                    contact_joint_mask=contact_joint_mask[row_idx : row_idx + 1],
                    empty_mask=empty_row,
                    generator=generator,
                )
                q_goal_rows.append(q_goal_row[0])
                changed_rows.append(changed_row[0])
                empty_rows.append(empty_row[0])
            if len(q_goal_rows) > 0:
                action_batch = torch.stack(q_goal_rows, dim=0)
                changed_joint_mask = torch.stack(changed_rows, dim=0)
                empty_mask = torch.stack(empty_rows, dim=0)
        q1_padded = self._pad_runtime_q_batch_cpu(q1_batch)
        q_seed_padded = self._pad_runtime_q_batch_cpu(q_seed_batch)
        action_padded = self._pad_runtime_q_batch_cpu(action_batch)
        changed_padded = torch.zeros((len(indices), self.action_dim), dtype=torch.bool)
        changed_padded[:, : int(self.local_dof)] = changed_joint_mask

        # Hand FK at runtime: CPU via the companion robot model when
        # ``hand_points_fk_device == "cpu"`` (default; dataloader workers emit world-space
        # hand tokens alongside q1). When "gpu" the caller runs FK on the main thread on
        # CUDA after collation, so skip the per-worker FK work entirely.
        hand_points_world = None
        hand_normals_world = None
        if (
            self.hand_points_fps > 0
            and self.hand_points_fk_device == "cpu"
            and self._cpu_hand_model is not None
        ):
            points_batch, normals_batch = self._cpu_hand_model.transform_hand_fps_template_batch(
                q1_batch.to(dtype=torch.float32),
            )
            hand_points_world = points_batch.to(device="cpu", dtype=torch.float32)
            hand_normals_world = normals_batch.to(device="cpu", dtype=torch.float32)

        results: list[Dict[str, torch.Tensor]] = []
        for row_idx in range(len(indices)):
            entry: Dict[str, torch.Tensor] = {
                "q1_padded": q1_padded[row_idx],
                "q_seed_padded": q_seed_padded[row_idx],
                "action": action_padded[row_idx],
                "non_contact_empty": empty_mask[row_idx],
                "non_contact_goal_changed_joint_mask": changed_padded[row_idx],
            }
            if hand_points_world is not None and hand_normals_world is not None:
                entry["hand_points_world"] = hand_points_world[row_idx]
                entry["hand_normals_world"] = hand_normals_world[row_idx]
            results.append(entry)
        return results

    def _build_runtime_q_sample(
        self,
        idx: int,
        *,
        resample_variant: int = 0,
    ) -> Dict[str, torch.Tensor]:
        # Convenience wrapper for building runtime q fields for one sample.
        return self._build_runtime_q_batch([int(idx)], resample_variant=resample_variant)[0]

    @torch.no_grad()
    def prepare_epoch_precomputed(
        self,
        epoch: int,
        *,
        device: torch.device | str = "cpu",
        chunk_size: int = 4096,
    ) -> None:
        """Refresh the epoch-wise runtime cache (q1 / action / hand FK).

        No-op when ``epoch_cache_enabled`` is False. All per-sample work runs on
        ``device`` in chunks: q1 noise + q1/q2 contact-mask mixing + hand-FPS FK are
        fully vectorized over the chunk using the same elementwise formulas as
        ``_build_q1_from_q2_batch_cpu`` / ``_mix_q_by_contact_mask_cpu`` (verified
        equivalent for the ``random_q1=False`` + ``q1_zero_probability=0`` path used
        by the cotrain trainer). The sparse-subset indices/valid mask are a pure
        function of the static ``buffer_contact_valid_mask`` and so are built once
        in ``__init__`` via ``_build_sparse_runtime_subset_once``; they are not
        refreshed here. Results are copied into the shared-memory runtime buffers
        so DataLoader workers see them via the fast ``__getitems__`` path without
        being respawned.

        Constraints: requires ``random_q1=False`` and ``q1_zero_probability==0.0``.
        Those are the only settings used by the cotrain trainer today; if a caller
        opts into epoch caching with richer q1 modes, we fall back to the existing
        CPU helper per-sample (slow but correct). Also restricted to
        ``contact_mode='sparse'`` — the subset helper is sparse-specific, and the
        patches/tips paths must use the legacy per-fetch path.
        """
        if not self.epoch_cache_enabled:
            return
        if self.contact_mode != "sparse":
            raise NotImplementedError(
                "epoch_cache_enabled=True is currently only defined for "
                f"contact_mode='sparse', got {self.contact_mode!r}. Non-sparse modes "
                "must use the legacy per-fetch path (epoch_cache_enabled=False)."
            )
        n = int(self.samples)
        if n <= 0:
            self._runtime_prepared_epoch.fill_(int(epoch))
            return
        chunk = int(max(1, chunk_size))
        fk_device = torch.device(device)
        hand_fps = int(self.hand_points_fps)
        gen_dev = torch.Generator(device=fk_device)
        gen_dev.manual_seed(int(self.seed) * 1_000_003 + int(epoch) * 13_883 + 211)
        # GPU-vectorized path is only valid for the noise-only q1 builder. Everything
        # else (random_q1, nonzero q1_zero_probability) falls back to the legacy
        # per-idx CPU loop.
        use_vectorized_q1 = (
            self.q1_sampling_mode == "gaussian"
            and
            not bool(self.random_q1)
            and float(self.q1_zero_probability) <= 0.0
        )
        use_contact_error_q1 = self.q1_sampling_mode == "contact_error_mixture"

        # Pre-move all static tensors the vectorized path touches onto fk_device. These
        # allocations are tiny (dof-sized) so the one-time setup cost is negligible.
        if use_vectorized_q1 or use_contact_error_q1:
            base_translation_idx = torch.as_tensor(
                self._pose_translation_indices_flat(), dtype=torch.long, device=fk_device
            )
            base_rotation_idx = torch.as_tensor(
                self._pose_rotation_indices_flat(), dtype=torch.long, device=fk_device
            )
            fixed_zero_idx = torch.as_tensor(
                list(self.model.fixed_zero_joint_indices), dtype=torch.long, device=fk_device
            )
            joint_noise_std_dev = self._joint_noise_std_cpu.to(device=fk_device, dtype=torch.float32)
            jl_lower = self._joint_limit_lower_cpu.to(device=fk_device, dtype=torch.float32)
            jl_upper = self._joint_limit_upper_cpu.to(device=fk_device, dtype=torch.float32)
            jl_finite = self._joint_limit_finite_cpu.to(device=fk_device, dtype=torch.bool)
            surface_link_idx = self._surface_template_link_indices_cpu.to(
                device=fk_device, dtype=torch.long
            )
            descendant_mask = self._joint_descendant_link_mask_cpu.to(
                device=fk_device, dtype=torch.bool
            )
            action_valid_mask = self.action_valid_mask_template.to(device=fk_device, dtype=torch.bool)
            rot_noise_std = math.radians(float(self.q1_rotation_noise_std_deg))
            trans_noise_std = float(self.q1_translation_noise_std)
            joint_noise_scale = float(self.q1_joint_noise_scale)
            pose_mode = str(self.sixd_pose_mode).strip().lower()
            num_links = int(descendant_mask.shape[1])

        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            size = end - start
            indices = list(range(start, end))
            if use_vectorized_q1:
                q2_local = self.buffer_q2_padded[start:end, : int(self.local_dof)].to(
                    device=fk_device, dtype=torch.float32
                )
                surface_mask = self.buffer_surface_mask[start:end].to(
                    device=fk_device, dtype=torch.bool
                )
                q1_local = q2_local.clone()
                if trans_noise_std > 0.0 and int(base_translation_idx.numel()) > 0:
                    noise = torch.randn(
                        (size, int(base_translation_idx.numel())),
                        dtype=torch.float32,
                        device=fk_device,
                        generator=gen_dev,
                    )
                    q1_local[:, base_translation_idx] = (
                        q1_local[:, base_translation_idx] + noise * trans_noise_std
                    )
                if rot_noise_std > 0.0 and int(base_rotation_idx.numel()) > 0:
                    noise = torch.randn(
                        (size, int(base_rotation_idx.numel())),
                        dtype=torch.float32,
                        device=fk_device,
                        generator=gen_dev,
                    )
                    q1_local[:, base_rotation_idx] = (
                        q1_local[:, base_rotation_idx] + noise * rot_noise_std
                    )
                if joint_noise_scale > 0.0:
                    noise = torch.randn(
                        (size, int(self.local_dof)),
                        dtype=torch.float32,
                        device=fk_device,
                        generator=gen_dev,
                    )
                    q1_local = q1_local + noise * joint_noise_std_dev.view(1, -1)
                # Clamp base rotations and finite joints (mirrors _clamp_q_to_limits_cpu).
                if int(base_rotation_idx.numel()) > 0:
                    br = q1_local[:, base_rotation_idx]
                    br = torch.remainder(br + math.pi, 2.0 * math.pi) - math.pi
                    q1_local[:, base_rotation_idx] = br
                if bool(jl_finite.any()):
                    finite_cols = torch.nonzero(jl_finite, as_tuple=False).view(-1)
                    q1_local[:, finite_cols] = q1_local[:, finite_cols].clamp(
                        min=jl_lower[finite_cols], max=jl_upper[finite_cols]
                    )
                # Pose-mode constraints.
                if pose_mode == "fixed":
                    if int(base_translation_idx.numel()) > 0:
                        q1_local[:, base_translation_idx] = 0.0
                    if int(base_rotation_idx.numel()) > 0:
                        q1_local[:, base_rotation_idx] = 0.0
                elif pose_mode == "xyz_only":
                    if int(base_rotation_idx.numel()) > 0:
                        q1_local[:, base_rotation_idx] = 0.0
                    if int(base_translation_idx.numel()) > 0:
                        trans_min, trans_max = self._pose_translation_bounds_cpu(
                            device=fk_device,
                            dtype=torch.float32,
                        )
                        q1_local[:, base_translation_idx] = q1_local[:, base_translation_idx].clamp(
                            min=trans_min, max=trans_max
                        )
                if int(fixed_zero_idx.numel()) > 0:
                    q1_local[:, fixed_zero_idx] = 0.0
                # Compute action via contact-mask mixing (mirrors _mix_q_by_contact_mask_cpu +
                # _build_joint_active_by_contact_mask_cpu, vectorized).
                active_links = torch.zeros(
                    (size, num_links), dtype=torch.bool, device=fk_device
                )
                for lk in range(num_links):
                    col = surface_mask[:, surface_link_idx == int(lk)]
                    if col.shape[1] > 0:
                        active_links[:, lk] = col.any(dim=1)
                joint_active_local = (
                    active_links.unsqueeze(1) & descendant_mask.unsqueeze(0)
                ).any(dim=-1)  # (size, action_dim)
                mixed_local = torch.where(
                    joint_active_local[:, : int(self.local_dof)],
                    self.buffer_q2_padded[start:end, : int(self.local_dof)].to(
                        device=fk_device, dtype=torch.float32
                    ),
                    q1_local,
                )
                if int(fixed_zero_idx.numel()) > 0:
                    mixed_local[:, fixed_zero_idx] = 0.0
                q_seed_local = mixed_local
                empty_non_contact = self._sample_causal_non_contact_empty_mask(
                    size,
                    device=fk_device,
                    generator=gen_dev,
                )
                if self.non_contact_causal_enabled:
                    action_local, goal_changed_joint_mask = self._sample_non_contact_goal_batch(
                        q_seed=q_seed_local,
                        contact_joint_mask=joint_active_local[:, : int(self.local_dof)],
                        empty_mask=empty_non_contact,
                        generator=gen_dev,
                    )
                else:
                    action_local = q_seed_local
                    goal_changed_joint_mask = torch.zeros_like(action_local, dtype=torch.bool)

                # Pad to action_dim and copy to shared-memory buffers (via CPU).
                q1_padded_chunk = torch.zeros(
                    (size, self.action_dim), dtype=torch.float32, device=fk_device
                )
                q1_padded_chunk[:, : int(self.local_dof)] = q1_local
                action_padded_chunk = torch.zeros(
                    (size, self.action_dim), dtype=torch.float32, device=fk_device
                )
                action_padded_chunk[:, : int(self.local_dof)] = action_local
                self._runtime_q1_padded[start:end].copy_(q1_padded_chunk.to("cpu"))
                self._runtime_action[start:end].copy_(action_padded_chunk.to("cpu"))
            elif use_contact_error_q1:
                q2_local = self.buffer_q2_padded[start:end, : int(self.local_dof)].to(
                    device=fk_device, dtype=torch.float32
                )
                surface_mask = self.buffer_surface_mask[start:end].to(
                    device=fk_device, dtype=torch.bool
                )
                q1_local = self._build_q1_contact_error_mixture_batch(
                    q2_probe=q2_local,
                    surface_mask=surface_mask,
                    contact_local_points=self.buffer_contact_local_points[start:end].to(
                        device=fk_device, dtype=torch.float32
                    ),
                    contact_link_indices=self.buffer_contact_link_indices[start:end].to(
                        device=fk_device, dtype=torch.long
                    ),
                    contact_valid_mask=self.buffer_contact_valid_mask[start:end].to(
                        device=fk_device, dtype=torch.bool
                    ),
                    target_contact_points=self.buffer_contact_cloud[start:end, :, :3].to(
                        device=fk_device, dtype=torch.float32
                    ),
                    generator=gen_dev,
                    fk_model=self.model,
                )
                joint_active_local = self._build_joint_active_by_contact_mask_on_device(surface_mask)[:, : int(self.local_dof)]
                mixed_local = torch.where(joint_active_local, q2_local, q1_local)
                if int(fixed_zero_idx.numel()) > 0:
                    mixed_local[:, fixed_zero_idx] = 0.0
                q_seed_local = mixed_local
                empty_non_contact = self._sample_causal_non_contact_empty_mask(
                    size,
                    device=fk_device,
                    generator=gen_dev,
                )
                if self.non_contact_causal_enabled:
                    action_local, goal_changed_joint_mask = self._sample_non_contact_goal_batch(
                        q_seed=q_seed_local,
                        contact_joint_mask=joint_active_local,
                        empty_mask=empty_non_contact,
                        generator=gen_dev,
                    )
                else:
                    action_local = q_seed_local
                    goal_changed_joint_mask = torch.zeros_like(action_local, dtype=torch.bool)
                q1_padded_chunk = torch.zeros(
                    (size, self.action_dim), dtype=torch.float32, device=fk_device
                )
                q1_padded_chunk[:, : int(self.local_dof)] = q1_local
                action_padded_chunk = torch.zeros(
                    (size, self.action_dim), dtype=torch.float32, device=fk_device
                )
                action_padded_chunk[:, : int(self.local_dof)] = action_local
                self._runtime_q1_padded[start:end].copy_(q1_padded_chunk.to("cpu"))
                self._runtime_action[start:end].copy_(action_padded_chunk.to("cpu"))
            else:
                # Fallback: the legacy per-idx CPU path (slow but correct for all modes).
                runtime_q_list = self._build_runtime_q_batch(indices, resample_variant=0)
                q1_stack = torch.stack(
                    [entry["q1_padded"] for entry in runtime_q_list], dim=0
                ).to(dtype=torch.float32)
                action_stack = torch.stack(
                    [entry["action"] for entry in runtime_q_list], dim=0
                ).to(dtype=torch.float32)
                self._runtime_q1_padded[start:end].copy_(q1_stack)
                self._runtime_action[start:end].copy_(action_stack)
                q1_local = q1_stack[:, : int(self.local_dof)].to(fk_device, dtype=torch.float32)
                action_local = action_stack[:, : int(self.local_dof)].to(fk_device, dtype=torch.float32)
                q_seed_stack = torch.stack(
                    [entry.get("q_seed_padded", entry["action"]) for entry in runtime_q_list],
                    dim=0,
                ).to(dtype=torch.float32)
                q_seed_local = q_seed_stack[:, : int(self.local_dof)].to(fk_device, dtype=torch.float32)
                empty_non_contact = torch.stack(
                    [
                        entry.get("non_contact_empty", torch.tensor(False, dtype=torch.bool))
                        for entry in runtime_q_list
                    ],
                    dim=0,
                ).to(device=fk_device, dtype=torch.bool)
                goal_changed_stack = torch.stack(
                    [
                        entry.get(
                            "non_contact_goal_changed_joint_mask",
                            torch.zeros((self.action_dim,), dtype=torch.bool),
                        )
                        for entry in runtime_q_list
                    ],
                    dim=0,
                ).to(device=fk_device, dtype=torch.bool)
                goal_changed_joint_mask = goal_changed_stack[:, : int(self.local_dof)]

            contact_idx_for_runtime = self._runtime_contact_indices[start:end].to(device=fk_device, dtype=torch.long)
            ci_u_runtime = contact_idx_for_runtime.unsqueeze(-1)

            if self._runtime_non_contact_cloud is not None and self._runtime_non_contact_valid_mask is not None:
                if self._collision_primitive_model is None:
                    raise RuntimeError(
                        f"{self.robot_name}: collision primitive model is unavailable for epoch-cache non-contact synthesis"
                    )
                full_contact_cloud = self.buffer_contact_cloud[start:end].to(device=fk_device, dtype=torch.float32)
                contact_cloud_runtime = torch.gather(
                    full_contact_cloud,
                    1,
                    ci_u_runtime.expand(-1, -1, 6),
                )
                contact_valid_runtime = self._runtime_contact_valid_mask[start:end].to(
                    device=fk_device,
                    dtype=torch.bool,
                )
                if self.non_contact_causal_enabled:
                    nc_cloud, nc_valid, nc_blocker = self._build_causal_non_contact_batch(
                        contact_cloud=contact_cloud_runtime,
                        contact_valid_mask=contact_valid_runtime,
                        q1_local=q1_local,
                        q_seed_local=q_seed_local,
                        q_goal_local=action_local,
                        changed_joint_mask=goal_changed_joint_mask,
                        empty_mask=empty_non_contact,
                        surface_model=self.model,
                        collision_model=self._collision_primitive_model,
                        generator=gen_dev,
                    )
                else:
                    nc_cloud = self._build_non_contact_approx_batch(
                        contact_cloud=contact_cloud_runtime,
                        contact_valid_mask=contact_valid_runtime,
                        q1_local=q1_local,
                        target_local=action_local,
                        collision_model=self._collision_primitive_model,
                        generator=gen_dev,
                    )
                    nc_valid = torch.ones(
                        (int(nc_cloud.shape[0]), int(nc_cloud.shape[1])),
                        dtype=torch.bool,
                        device=nc_cloud.device,
                    )
                    nc_blocker = torch.zeros_like(nc_valid, dtype=torch.bool)
                self._runtime_non_contact_cloud[start:end].copy_(
                    nc_cloud.detach().to(device="cpu", dtype=torch.float32)
                )
                self._runtime_non_contact_valid_mask[start:end].copy_(nc_valid.detach().to(device="cpu", dtype=torch.bool))
                if self._runtime_non_contact_blocker_mask is not None:
                    self._runtime_non_contact_blocker_mask[start:end].copy_(
                        nc_blocker.detach().to(device="cpu", dtype=torch.bool)
                    )

            if self._runtime_q1_contact_points_world is not None:
                contact_local_points = torch.gather(
                    self.buffer_contact_local_points[start:end].to(device=fk_device, dtype=torch.float32),
                    1,
                    ci_u_runtime.expand(-1, -1, 3),
                )
                contact_link_indices = torch.gather(
                    self.buffer_contact_link_indices[start:end].to(device=fk_device, dtype=torch.long),
                    1,
                    contact_idx_for_runtime,
                )
                with torch.no_grad():
                    q1_contact_points, _ = self.model.transform_attached_points_normals_batch(
                        q=q1_local,
                        local_points=contact_local_points,
                        link_indices=contact_link_indices,
                        local_normals=None,
                    )
                self._runtime_q1_contact_points_world[start:end].copy_(
                    q1_contact_points.detach().to(device="cpu", dtype=torch.float32)
                )

            # Hand-FPS FK in one batched GPU call per chunk.
            if hand_fps > 0 and self._runtime_hand_points_world is not None:
                pts_world, nrm_world = self.model.transform_hand_fps_template_batch(q1_local)
                self._runtime_hand_points_world[start:end].copy_(
                    pts_world.detach().to(device="cpu", dtype=torch.float32)
                )
                self._runtime_hand_normals_world[start:end].copy_(
                    nrm_world.detach().to(device="cpu", dtype=torch.float32)
                )
        self._runtime_prepared_epoch.fill_(int(epoch))


    def _runtime_non_contact_surface_model(self) -> Any:
        # Candidate proposal may run in forked DataLoader workers. Prefer the CPU
        # runtime model built during dataset init; fall back only for CPU-only datasets.
        return self._cpu_hand_model if self._cpu_hand_model is not None else self.model

    def _runtime_non_contact_collision_model(self) -> Any:
        # SDF filtering/ranking is always analytic collision-primitive SDF. It may run in
        # forked DataLoader workers, so prefer the CPU primitive wrapper when available.
        model = (
            self._cpu_collision_primitive_model
            if self._cpu_collision_primitive_model is not None
            else self._collision_primitive_model
        )
        if model is None:
            raise RuntimeError(
                f"{self.robot_name}: collision primitive model is unavailable for non-contact synthesis"
            )
        return model


    def _sample_non_contact_candidates_from_contacts_batch(
        self,
        *,
        contact_cloud: torch.Tensor,
        contact_valid_mask: torch.Tensor,
        count: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        # Fast candidate sampler for epoch-cache non-contact clouds. It samples around
        # valid contact anchors and biases points to the outward side of the hand surface;
        # the analytic primitive SDF filter below remains the final authority.
        batch_size, n_contact, _ = contact_cloud.shape
        device = contact_cloud.device
        k = int(max(1, count))
        valid = contact_valid_mask.to(dtype=torch.bool)
        if int(n_contact) <= 0:
            centers = torch.zeros((batch_size, 1, 3), dtype=torch.float32, device=device)
            normals = torch.zeros_like(centers)
            normals[..., 2] = 1.0
            anchor_idx = torch.zeros((batch_size, k), dtype=torch.long, device=device)
            anchor_points = centers.expand(-1, k, -1)
            anchor_normals = normals.expand(-1, k, -1)
        else:
            safe_valid = valid.clone()
            empty_rows = ~safe_valid.any(dim=1)
            if bool(empty_rows.any()):
                safe_valid[empty_rows, 0] = True
            weights = safe_valid.to(dtype=torch.float32)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            anchor_idx = torch.multinomial(weights, k, replacement=True, generator=generator)
            gather_idx = anchor_idx.unsqueeze(-1).expand(-1, -1, 3)
            anchor_points = torch.gather(contact_cloud[..., :3].to(dtype=torch.float32), 1, gather_idx)
            anchor_normals = torch.gather(contact_cloud[..., 3:6].to(dtype=torch.float32), 1, gather_idx)
            normal_norm = anchor_normals.norm(dim=-1, keepdim=True)
            fallback = torch.zeros_like(anchor_normals)
            fallback[..., 2] = 1.0
            anchor_normals = torch.where(
                normal_norm > 1.0e-8,
                anchor_normals / normal_norm.clamp_min(1.0e-8),
                fallback,
            )

        flat_normals = anchor_normals.reshape(-1, 3)
        t1, t2 = _object_random_tangent_basis(flat_normals)
        t1 = t1.view(batch_size, k, 3)
        t2 = t2.view(batch_size, k, 3)
        theta = torch.rand((batch_size, k, 1), dtype=torch.float32, device=device, generator=generator) * (2.0 * math.pi)
        tangent_dir = torch.cos(theta) * t1 + torch.sin(theta) * t2
        tangent_radius = (
            torch.rand((batch_size, k, 1), dtype=torch.float32, device=device, generator=generator).sqrt()
            * float(self.non_contact_sampling_radius)
            * 0.80
        )
        outward = (
            float(self.non_contact_sdf_clearance)
            + torch.rand((batch_size, k, 1), dtype=torch.float32, device=device, generator=generator)
            * float(self.non_contact_sampling_radius)
            * 0.60
        )
        candidates = anchor_points + tangent_radius * tangent_dir + outward * anchor_normals
        return candidates.to(dtype=torch.float32)

    def _non_contact_primitive_sdf_score(
        self,
        *,
        candidates: torch.Tensor,
        q_local: torch.Tensor,
        collision_model: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_device = self._collision_model_device(collision_model)
        candidates_model = candidates.to(device=model_device, dtype=torch.float32)
        q_model = q_local.to(device=model_device, dtype=torch.float32)
        sdf = collision_model.query_sdf(q_model, candidates_model)
        sdf = sdf.to(device=candidates.device, dtype=torch.float32)
        margin = float(self.non_contact_sdf_clearance)
        return sdf >= margin, sdf

    def _sample_causal_blocker_candidates_batch(
        self,
        *,
        q_seed_local: torch.Tensor,
        changed_joint_mask: torch.Tensor,
        surface_model: Any,
        count: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        batch_size = int(q_seed_local.shape[0])
        k = int(max(1, count))
        device = q_seed_local.device
        seed_points, seed_normals = surface_model.get_surface_points_normals_batch(
            q=q_seed_local.to(device=getattr(surface_model, "device", device), dtype=torch.float32)
        )
        seed_points = seed_points.to(device=device, dtype=torch.float32)
        seed_normals = seed_normals.to(device=device, dtype=torch.float32)
        seed_normals = seed_normals / seed_normals.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)

        descendant = self._joint_descendant_link_mask_cpu[: int(self.local_dof)].to(
            device=device,
            dtype=torch.bool,
        )
        changed_links = (changed_joint_mask.to(device=device).unsqueeze(-1) & descendant.unsqueeze(0)).any(dim=1)
        surface_link_idx = self._surface_template_link_indices_cpu.to(device=device, dtype=torch.long)
        changed_surface = torch.gather(
            changed_links,
            1,
            surface_link_idx.view(1, -1).expand(batch_size, -1),
        )
        has_changed_surface = changed_surface.any(dim=1)
        weights = changed_surface.to(dtype=torch.float32)
        if bool((~has_changed_surface).any()):
            weights = weights.clone()
            weights[~has_changed_surface, 0] = 1.0
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        anchor_idx = torch.multinomial(weights, k, replacement=True, generator=generator)
        gather_idx = anchor_idx.unsqueeze(-1).expand(-1, -1, 3)
        anchor_points = torch.gather(seed_points, 1, gather_idx)
        anchor_normals = torch.gather(seed_normals, 1, gather_idx)

        flat_normals = anchor_normals.reshape(-1, 3)
        t1, t2 = _object_random_tangent_basis(flat_normals)
        t1 = t1.view(batch_size, k, 3)
        t2 = t2.view(batch_size, k, 3)
        theta = torch.rand((batch_size, k, 1), dtype=torch.float32, device=device, generator=generator) * (2.0 * math.pi)
        tangent_dir = torch.cos(theta) * t1 + torch.sin(theta) * t2
        tangent_radius = (
            torch.rand((batch_size, k, 1), dtype=torch.float32, device=device, generator=generator).sqrt()
            * float(self.non_contact_sampling_radius)
            * 0.25
        )
        inward = torch.rand((batch_size, k, 1), dtype=torch.float32, device=device, generator=generator) * max(
            float(self.non_contact_sdf_clearance) * 2.0,
            float(self.non_contact_sampling_radius) * 0.02,
        )
        candidates = anchor_points + tangent_radius * tangent_dir - inward * anchor_normals
        return candidates.to(dtype=torch.float32)

    def _build_causal_blocker_batch(
        self,
        *,
        q_seed_local: torch.Tensor,
        q_goal_local: torch.Tensor,
        changed_joint_mask: torch.Tensor,
        target_count: int,
        surface_model: Any,
        collision_model: Any,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(q_seed_local.shape[0])
        device = q_seed_local.device
        if target_count <= 0 or batch_size <= 0:
            return (
                torch.zeros((batch_size, 0, 3), dtype=torch.float32, device=device),
                torch.zeros((batch_size, 0), dtype=torch.bool, device=device),
            )
        candidate_count = int(max(target_count, self.non_contact_blocker_candidate_points))
        candidates = self._sample_causal_blocker_candidates_batch(
            q_seed_local=q_seed_local,
            changed_joint_mask=changed_joint_mask,
            surface_model=surface_model,
            count=candidate_count,
            generator=generator,
        )
        _, seed_score = self._non_contact_primitive_sdf_score(
            candidates=candidates,
            q_local=q_seed_local,
            collision_model=collision_model,
        )
        goal_accept, goal_score = self._non_contact_primitive_sdf_score(
            candidates=candidates,
            q_local=q_goal_local,
            collision_model=collision_model,
        )
        has_changed_joint = changed_joint_mask.to(device=device, dtype=torch.bool).any(dim=1, keepdim=True)
        inside_seed = seed_score <= 0.0
        blocker_valid = inside_seed & goal_accept & has_changed_joint
        ranked = goal_score - seed_score + blocker_valid.to(dtype=goal_score.dtype) * 1.0e6
        top_idx = torch.topk(ranked, k=int(target_count), dim=1, largest=True).indices
        selected = torch.gather(candidates, 1, top_idx.unsqueeze(-1).expand(-1, -1, 3))
        selected_valid = torch.gather(blocker_valid, 1, top_idx)
        return selected.to(dtype=torch.float32), selected_valid.to(dtype=torch.bool)

    def _build_non_contact_approx_batch(
        self,
        *,
        contact_cloud: torch.Tensor,
        contact_valid_mask: torch.Tensor,
        q1_local: torch.Tensor,
        target_local: torch.Tensor,
        collision_model: Any,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if self.remove_non_contact_points or int(self.non_contact_input_points) <= 0:
            return torch.zeros((int(contact_cloud.shape[0]), 0, 3), dtype=torch.float32, device=contact_cloud.device)

        device = contact_cloud.device
        target_count = int(self.non_contact_input_points)
        candidate_count = int(max(target_count, self.non_contact_candidate_points))
        batch_size = int(contact_cloud.shape[0])
        out_chunks: list[torch.Tensor] = []
        sub_batch = int(max(1, self.non_contact_epoch_batch_size))
        for start in range(0, batch_size, sub_batch):
            end = min(batch_size, start + sub_batch)
            cand = self._sample_non_contact_candidates_from_contacts_batch(
                contact_cloud=contact_cloud[start:end],
                contact_valid_mask=contact_valid_mask[start:end],
                count=candidate_count,
                generator=generator,
            )
            q1_accept, q1_score = self._non_contact_primitive_sdf_score(
                candidates=cand,
                q_local=q1_local[start:end],
                collision_model=collision_model,
            )
            tgt_accept, tgt_score = self._non_contact_primitive_sdf_score(
                candidates=cand,
                q_local=target_local[start:end],
                collision_model=collision_model,
            )
            accept = q1_accept & tgt_accept
            score = torch.minimum(q1_score, tgt_score)
            # Accepted points dominate rejected points; rejected fallback still picks the
            # safest candidates when fewer than target_count candidates pass both filters.
            ranked = score + accept.to(dtype=score.dtype) * 1.0e6
            top_idx = torch.topk(ranked, k=target_count, dim=1, largest=True).indices
            selected = torch.gather(cand, 1, top_idx.unsqueeze(-1).expand(-1, -1, 3))
            shuffle = torch.rand(
                (int(selected.shape[0]), target_count),
                dtype=torch.float32,
                device=selected.device,
                generator=generator,
            ).argsort(dim=1)
            selected = torch.gather(selected, 1, shuffle.unsqueeze(-1).expand(-1, -1, 3))
            out_chunks.append(selected.to(device=device, dtype=torch.float32))
        return torch.cat(out_chunks, dim=0)

    def _build_causal_non_contact_batch(
        self,
        *,
        contact_cloud: torch.Tensor,
        contact_valid_mask: torch.Tensor,
        q1_local: torch.Tensor,
        q_seed_local: torch.Tensor,
        q_goal_local: torch.Tensor,
        changed_joint_mask: torch.Tensor,
        empty_mask: torch.Tensor,
        surface_model: Any,
        collision_model: Any,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_count = int(self.non_contact_input_points)
        batch_size = int(contact_cloud.shape[0])
        device = contact_cloud.device
        if self.remove_non_contact_points or target_count <= 0:
            return (
                torch.zeros((batch_size, 0, 3), dtype=torch.float32, device=device),
                torch.zeros((batch_size, 0), dtype=torch.bool, device=device),
                torch.zeros((batch_size, 0), dtype=torch.bool, device=device),
            )

        cloud = self._build_non_contact_approx_batch(
            contact_cloud=contact_cloud,
            contact_valid_mask=contact_valid_mask,
            q1_local=q1_local,
            target_local=q_goal_local,
            collision_model=collision_model,
            generator=generator,
        )
        valid = torch.ones((batch_size, target_count), dtype=torch.bool, device=device)
        blocker_mask = torch.zeros((batch_size, target_count), dtype=torch.bool, device=device)
        blocker_count = int(round(float(target_count) * float(self.non_contact_blocker_fraction)))
        blocker_count = max(0, min(target_count, blocker_count))
        if blocker_count > 0:
            blocker_cloud, blocker_valid = self._build_causal_blocker_batch(
                q_seed_local=q_seed_local.to(device=getattr(surface_model, "device", device), dtype=torch.float32),
                q_goal_local=q_goal_local.to(device=getattr(surface_model, "device", device), dtype=torch.float32),
                changed_joint_mask=changed_joint_mask.to(device=getattr(surface_model, "device", device), dtype=torch.bool),
                target_count=blocker_count,
                surface_model=surface_model,
                collision_model=collision_model,
                generator=generator,
            )
            blocker_cloud = blocker_cloud.to(device=device, dtype=torch.float32)
            blocker_valid = blocker_valid.to(device=device, dtype=torch.bool)
            replace = blocker_valid.unsqueeze(-1)
            cloud[:, :blocker_count] = torch.where(replace, blocker_cloud, cloud[:, :blocker_count])
            blocker_mask[:, :blocker_count] = blocker_valid

        empty = empty_mask.to(device=device, dtype=torch.bool).view(-1, 1)
        if bool(empty.any()):
            cloud = torch.where(empty.unsqueeze(-1), torch.zeros_like(cloud), cloud)
            valid = valid & ~empty
            blocker_mask = blocker_mask & ~empty
        return cloud, valid, blocker_mask

    def _build_non_contact_observation(
        self,
        idx: int,
        *,
        contact_obs: Dict[str, torch.Tensor],
        q1_padded: torch.Tensor | None = None,
        q_seed_padded: torch.Tensor | None = None,
        target_q_padded: torch.Tensor | None = None,
        non_contact_empty: torch.Tensor | None = None,
        non_contact_goal_changed_joint_mask: torch.Tensor | None = None,
        resample_variant: int = 0,
    ) -> Dict[str, torch.Tensor]:
        # Build one sample through the same approximate batched path used by DataLoader.
        obs = self._build_non_contact_observations_batch(
            [int(idx)],
            contact_obs_list=[contact_obs],
            q1_padded_list=[
                self.buffer_q1_padded[idx] if q1_padded is None else q1_padded
            ],
            q_seed_padded_list=[
                self.buffer_action[idx] if q_seed_padded is None else q_seed_padded
            ],
            target_q_padded_list=[
                self.buffer_action[idx] if target_q_padded is None else target_q_padded
            ],
            non_contact_empty_list=[
                torch.tensor(False, dtype=torch.bool) if non_contact_empty is None else non_contact_empty
            ],
            non_contact_goal_changed_joint_mask_list=[
                torch.zeros((self.action_dim,), dtype=torch.bool)
                if non_contact_goal_changed_joint_mask is None
                else non_contact_goal_changed_joint_mask
            ],
            resample_variant=resample_variant,
        )
        return obs[0]

    @staticmethod
    def _empty_non_contact_observation() -> Dict[str, torch.Tensor]:
        # Return the zero-width non-contact payload used when that input stream is disabled.
        return {
            "non_contact_cloud": torch.zeros((0, 3), dtype=torch.float32),
            "non_contact_valid_mask": torch.zeros((0,), dtype=torch.bool),
            "non_contact_blocker_mask": torch.zeros((0,), dtype=torch.bool),
        }

    def _build_non_contact_observations_batch(
        self,
        indices: list[int],
        *,
        contact_obs_list: list[Dict[str, torch.Tensor]],
        q1_padded_list: list[torch.Tensor] | None = None,
        q_seed_padded_list: list[torch.Tensor] | None = None,
        target_q_padded_list: list[torch.Tensor] | None = None,
        non_contact_empty_list: list[torch.Tensor] | None = None,
        non_contact_goal_changed_joint_mask_list: list[torch.Tensor] | None = None,
        resample_variant: int = 0,
    ) -> list[Dict[str, torch.Tensor]]:
        # Vectorize non-contact generation with the analytic primitive SDF filter:
        # candidate points are sampled around contact anchors, then ranked by primitive
        # SDF clearance for q1/q_seed and the target/action hand.
        if len(indices) == 0:
            return []
        if self.remove_non_contact_points:
            return [self._empty_non_contact_observation() for _ in indices]
        surface_model = self._runtime_non_contact_surface_model()
        collision_model = self._runtime_non_contact_collision_model()
        geom_device = getattr(surface_model, "device", torch.device("cpu"))
        contact_cloud = torch.stack(
            [obs["contact_cloud"].to(dtype=torch.float32) for obs in contact_obs_list],
            dim=0,
        ).to(device=geom_device, dtype=torch.float32)
        contact_valid = torch.stack(
            [obs["contact_valid_mask"].to(dtype=torch.bool) for obs in contact_obs_list],
            dim=0,
        ).to(device=geom_device, dtype=torch.bool)
        if q1_padded_list is None:
            q1_padded_list = [self.buffer_q1_padded[idx] for idx in indices]
        if q_seed_padded_list is None:
            q_seed_padded_list = [self.buffer_action[idx] for idx in indices]
        if target_q_padded_list is None:
            target_q_padded_list = [self.buffer_action[idx] for idx in indices]
        if non_contact_empty_list is None:
            non_contact_empty_list = [torch.tensor(False, dtype=torch.bool) for _ in indices]
        if non_contact_goal_changed_joint_mask_list is None:
            non_contact_goal_changed_joint_mask_list = [
                torch.zeros((self.action_dim,), dtype=torch.bool) for _ in indices
            ]
        q1_local = torch.stack(
            [q[: self.local_dof].to(dtype=torch.float32) for q in q1_padded_list],
            dim=0,
        ).to(device=geom_device, dtype=torch.float32)
        q_seed_local = torch.stack(
            [q[: self.local_dof].to(dtype=torch.float32) for q in q_seed_padded_list],
            dim=0,
        ).to(device=geom_device, dtype=torch.float32)
        target_local = torch.stack(
            [q[: self.local_dof].to(dtype=torch.float32) for q in target_q_padded_list],
            dim=0,
        ).to(device=geom_device, dtype=torch.float32)
        empty_mask = torch.stack(
            [x.to(dtype=torch.bool).view(()) for x in non_contact_empty_list],
            dim=0,
        ).to(device=geom_device, dtype=torch.bool)
        changed_joint_mask = torch.stack(
            [x[: self.local_dof].to(dtype=torch.bool) for x in non_contact_goal_changed_joint_mask_list],
            dim=0,
        ).to(device=geom_device, dtype=torch.bool)
        generator = self._build_runtime_generator(
            int(indices[0]),
            resample_variant=resample_variant,
            seed_offset=101,
        )
        if generator is not None and torch.device(generator.device) != torch.device(geom_device):
            generator = torch.Generator(device=geom_device).manual_seed(int(generator.initial_seed()))
        if self.non_contact_causal_enabled:
            non_contact, valid_batch, blocker_batch = self._build_causal_non_contact_batch(
                contact_cloud=contact_cloud,
                contact_valid_mask=contact_valid,
                q1_local=q1_local,
                q_seed_local=q_seed_local,
                q_goal_local=target_local,
                changed_joint_mask=changed_joint_mask,
                empty_mask=empty_mask,
                surface_model=surface_model,
                collision_model=collision_model,
                generator=generator,
            )
        else:
            non_contact = self._build_non_contact_approx_batch(
                contact_cloud=contact_cloud,
                contact_valid_mask=contact_valid,
                q1_local=q1_local,
                target_local=target_local,
                collision_model=collision_model,
                generator=generator,
            )
            valid_batch = torch.ones(
                (int(non_contact.shape[0]), int(non_contact.shape[1])),
                dtype=torch.bool,
                device=non_contact.device,
            )
            blocker_batch = torch.zeros_like(valid_batch, dtype=torch.bool)
        non_contact = non_contact.detach().to(device="cpu", dtype=torch.float32)
        valid_batch = valid_batch.detach().to(device="cpu", dtype=torch.bool)
        blocker_batch = blocker_batch.detach().to(device="cpu", dtype=torch.bool)
        return [
            {
                "non_contact_cloud": non_contact[i],
                "non_contact_valid_mask": valid_batch[i],
                "non_contact_blocker_mask": blocker_batch[i],
            }
            for i in range(int(non_contact.shape[0]))
        ]

    def _assemble_sample(
        self,
        idx: int,
        *,
        contact_obs: Dict[str, torch.Tensor],
        non_contact_obs: Dict[str, torch.Tensor],
        runtime_q: Dict[str, torch.Tensor] | None = None,
        resample_variant: int = 0,
    ) -> Dict[str, torch.Tensor | str]:
        # Combine fixed-buffer contact/q2 fields with runtime q1/action and non-contact points.
        action = self.buffer_action[idx] if runtime_q is None else runtime_q["action"]
        q1_padded = self.buffer_q1_padded[idx] if runtime_q is None else runtime_q["q1_padded"]
        surface_mask = self.buffer_surface_mask[idx]
        joint_active_local = self._build_joint_active_by_contact_mask_cpu(surface_mask)
        contact_joint_mask = torch.zeros((self.action_dim,), dtype=torch.bool)
        contact_joint_mask[: int(self.local_dof)] = joint_active_local
        contact_joint_mask &= self.action_valid_mask_template
        sample: Dict[str, torch.Tensor | str] = {
            "contact_cloud": contact_obs["contact_cloud"],
            "contact_valid_mask": contact_obs["contact_valid_mask"],
            "contact_local_points": contact_obs["contact_local_points"],
            "contact_local_normals": contact_obs["contact_local_normals"],
            "contact_link_indices": contact_obs["contact_link_indices"],
            "surface_mask": surface_mask,
            "non_contact_cloud": non_contact_obs["non_contact_cloud"],
            "non_contact_valid_mask": non_contact_obs["non_contact_valid_mask"],
            "non_contact_blocker_mask": non_contact_obs.get(
                "non_contact_blocker_mask",
                torch.zeros_like(non_contact_obs["non_contact_valid_mask"], dtype=torch.bool),
            ),
            "action": action,
            "target_q_full": action,
            "q2_padded": self.buffer_q2_padded[idx],
            "action_valid_mask": self.action_valid_mask_template,
            "contact_joint_mask": contact_joint_mask,
            "robot_index": torch.tensor(self.robot_index, dtype=torch.long),
            "local_dof": torch.tensor(self.local_dof, dtype=torch.long),
            "robot_name": self.robot_name,
            "resample_variant": torch.tensor(int(max(0, resample_variant)), dtype=torch.long),
            "q1_padded": q1_padded,
        }
        if runtime_q is not None and "hand_points_world" in runtime_q:
            sample["hand_points_world"] = runtime_q["hand_points_world"]
            sample["hand_normals_world"] = runtime_q["hand_normals_world"]
        return sample

    def get_sample(
        self,
        idx: int,
        *,
        resample_variant: int = 0,
    ) -> Dict[str, torch.Tensor | str]:
        # Public helper that performs all runtime resampling for one dataset index.
        contact_obs = self._build_resampled_contact_observation(idx, resample_variant=resample_variant)
        runtime_q = self._build_runtime_q_sample(idx, resample_variant=resample_variant)
        non_contact_obs = self._build_non_contact_observation(
            idx,
            contact_obs=contact_obs,
            q1_padded=runtime_q["q1_padded"],
            q_seed_padded=runtime_q.get("q_seed_padded", runtime_q["action"]),
            target_q_padded=runtime_q["action"],
            non_contact_empty=runtime_q.get("non_contact_empty"),
            non_contact_goal_changed_joint_mask=runtime_q.get("non_contact_goal_changed_joint_mask"),
            resample_variant=resample_variant,
        )
        return self._assemble_sample(
            idx,
            contact_obs=contact_obs,
            non_contact_obs=non_contact_obs,
            runtime_q=runtime_q,
            resample_variant=resample_variant,
        )

    # The serialized payload mirrors the in-memory probe split and stays easy to reuse later.
    def _write_batch_to_buffer(self, slot_indices: slice, batch: Dict[str, torch.Tensor]) -> None:
        # Copy generated probe tensors into their fixed CPU buffer slots.
        self.buffer_contact_cloud[slot_indices] = batch["contact_cloud"].detach().to(device="cpu", dtype=torch.float32)
        self.buffer_contact_valid_mask[slot_indices] = batch["contact_valid_mask"].detach().to(device="cpu", dtype=torch.bool)
        self.buffer_contact_local_points[slot_indices] = batch["contact_local_points"].detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        self.buffer_contact_local_normals[slot_indices] = batch["contact_local_normals"].detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        self.buffer_contact_link_indices[slot_indices] = batch["contact_link_indices"].detach().to(
            device="cpu",
            dtype=torch.int16,
        )
        self.buffer_contact_parent_sparse_indices[slot_indices] = batch["contact_parent_sparse_indices"].detach().to(
            device="cpu",
            dtype=torch.int16,
        )
        self.buffer_contact_cluster_ids[slot_indices] = batch["contact_cluster_ids"].detach().to(
            device="cpu",
            dtype=torch.int16,
        )
        self.buffer_surface_mask[slot_indices] = batch["surface_mask"].detach().to(device="cpu", dtype=torch.bool)
        self.buffer_action[slot_indices] = batch["action"].detach().to(device="cpu", dtype=torch.float32)
        self.buffer_q2_padded[slot_indices] = batch["q2_padded"].detach().to(device="cpu", dtype=torch.float32)
        self.buffer_q1_padded[slot_indices] = batch["q1_padded"].detach().to(device="cpu", dtype=torch.float32)

    def __len__(self) -> int:
        # Report the number of fixed probe samples in the buffer.
        return self.samples

    # Samples expose exactly the fields required by the q-only probe trainer and viewers.
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        # Fetch one sample with the default runtime resampling variant.
        return self.get_sample(idx, resample_variant=0)

    def __getitems__(self, indices: Iterable[int]) -> list[Dict[str, torch.Tensor | str]]:
        # Batch-fetch samples while sharing runtime q/non-contact work where possible.
        index_list = [int(idx) for idx in indices]
        if len(index_list) == 0:
            return []

        # Fast path: per-sample runtime state was already precomputed on the main thread
        # and parked in shared-memory buffers. Workers just index + gather; no CPU FK,
        # no q1 noise sampling, no contact re-roll.
        if (
            self.epoch_cache_enabled
            and int(self._runtime_prepared_epoch.item()) > 0
        ):
            return self._getitems_from_epoch_cache(index_list)

        contact_obs_list = [
            self._build_resampled_contact_observation(idx, resample_variant=0) for idx in index_list
        ]
        runtime_q_list = self._build_runtime_q_batch(index_list, resample_variant=0)
        non_contact_obs_list = self._build_non_contact_observations_batch(
            index_list,
            contact_obs_list=contact_obs_list,
            q1_padded_list=[runtime_q["q1_padded"] for runtime_q in runtime_q_list],
            q_seed_padded_list=[runtime_q.get("q_seed_padded", runtime_q["action"]) for runtime_q in runtime_q_list],
            target_q_padded_list=[runtime_q["action"] for runtime_q in runtime_q_list],
            non_contact_empty_list=[runtime_q.get("non_contact_empty", torch.tensor(False, dtype=torch.bool)) for runtime_q in runtime_q_list],
            non_contact_goal_changed_joint_mask_list=[
                runtime_q.get(
                    "non_contact_goal_changed_joint_mask",
                    torch.zeros((self.action_dim,), dtype=torch.bool),
                )
                for runtime_q in runtime_q_list
            ],
            resample_variant=0,
        )
        return [
            self._assemble_sample(
                idx,
                contact_obs=contact_obs,
                non_contact_obs=non_contact_obs,
                runtime_q=runtime_q,
                resample_variant=0,
            )
            for idx, contact_obs, non_contact_obs, runtime_q in zip(
                index_list,
                contact_obs_list,
                non_contact_obs_list,
                runtime_q_list,
            )
        ]

    def _getitems_from_epoch_cache(
        self,
        index_list: list[int],
    ) -> list[Dict[str, torch.Tensor | str]]:
        # Assemble samples from the precomputed shared-memory runtime buffers. Contact
        # tensors are gathered per-sample via the epoch's cached subset indices; q1/action/
        # hand-FPS/non-contact outputs come straight from runtime buffers.
        num = len(index_list)
        idx_t = torch.as_tensor(index_list, dtype=torch.long)

        contact_idx = self._runtime_contact_indices.index_select(0, idx_t)               # (B, C)
        # Valid mask comes from the runtime buffer (populated by the sparse helper), not
        # from the bank — padding slots in variable-length mode must report False even
        # though they point to bank slot 0 (which is itself a valid real contact).
        contact_valid_mask = self._runtime_contact_valid_mask.index_select(0, idx_t).clone()
        full_contact_cloud = self.buffer_contact_cloud.index_select(0, idx_t)            # (B, Cmax, 6)
        full_contact_local_points = self.buffer_contact_local_points.index_select(0, idx_t)
        full_contact_local_normals = self.buffer_contact_local_normals.index_select(0, idx_t)
        full_contact_link_indices = self.buffer_contact_link_indices.index_select(0, idx_t)
        surface_mask_batch = self.buffer_surface_mask.index_select(0, idx_t)              # (B, S)
        q2_padded_batch = self.buffer_q2_padded.index_select(0, idx_t)
        q1_padded_batch = self._runtime_q1_padded.index_select(0, idx_t)
        action_batch = self._runtime_action.index_select(0, idx_t)

        ci_u = contact_idx.unsqueeze(-1)  # (B, C, 1)
        contact_cloud = torch.gather(full_contact_cloud, 1, ci_u.expand(-1, -1, 6))
        contact_local_points = torch.gather(full_contact_local_points, 1, ci_u.expand(-1, -1, 3))
        contact_local_normals = torch.gather(full_contact_local_normals, 1, ci_u.expand(-1, -1, 3))
        contact_link_indices = torch.gather(full_contact_link_indices, 1, contact_idx).to(
            dtype=torch.long
        )

        # Contact-joint mask (Python + CPU tensor ops) mirrors ``_assemble_sample``.
        joint_active_local = self._build_joint_active_by_contact_mask_cpu(surface_mask_batch)
        contact_joint_mask = torch.zeros((num, self.action_dim), dtype=torch.bool)
        contact_joint_mask[:, : int(self.local_dof)] = joint_active_local
        contact_joint_mask &= self.action_valid_mask_template.unsqueeze(0)

        # Non-contact is now part of the epoch cache. DataLoader workers only index the
        # shared tensor; synthesis/filtering happens once per epoch in the main process.
        if (
            self.remove_non_contact_points
            or int(self.non_contact_input_points) <= 0
            or self._runtime_non_contact_cloud is None
            or self._runtime_non_contact_valid_mask is None
        ):
            non_contact_cloud_batch = torch.zeros((num, 0, 3), dtype=torch.float32)
            non_contact_valid_batch = torch.zeros((num, 0), dtype=torch.bool)
            non_contact_blocker_batch = torch.zeros((num, 0), dtype=torch.bool)
        else:
            non_contact_cloud_batch = self._runtime_non_contact_cloud.index_select(0, idx_t)
            non_contact_valid_batch = self._runtime_non_contact_valid_mask.index_select(0, idx_t)
            if self._runtime_non_contact_blocker_mask is None:
                non_contact_blocker_batch = torch.zeros_like(non_contact_valid_batch, dtype=torch.bool)
            else:
                non_contact_blocker_batch = self._runtime_non_contact_blocker_mask.index_select(0, idx_t)

        samples: list[Dict[str, torch.Tensor | str]] = []
        has_hand = self.hand_points_fps > 0 and self._runtime_hand_points_world is not None
        has_q1_contact_points = self._runtime_q1_contact_points_world is not None
        for b in range(num):
            sample: Dict[str, torch.Tensor | str] = {
                "contact_cloud": contact_cloud[b].clone(),
                "contact_valid_mask": contact_valid_mask[b].clone(),
                "contact_local_points": contact_local_points[b].clone(),
                "contact_local_normals": contact_local_normals[b].clone(),
                "contact_link_indices": contact_link_indices[b].clone(),
                "surface_mask": surface_mask_batch[b].clone(),
                "non_contact_cloud": non_contact_cloud_batch[b].clone(),
                "non_contact_valid_mask": non_contact_valid_batch[b].clone(),
                "non_contact_blocker_mask": non_contact_blocker_batch[b].clone(),
                "action": action_batch[b].clone(),
                "target_q_full": action_batch[b].clone(),
                "q2_padded": q2_padded_batch[b].clone(),
                "action_valid_mask": self.action_valid_mask_template,
                "contact_joint_mask": contact_joint_mask[b].clone(),
                "robot_index": torch.tensor(self.robot_index, dtype=torch.long),
                "local_dof": torch.tensor(self.local_dof, dtype=torch.long),
                "robot_name": self.robot_name,
                "resample_variant": torch.tensor(0, dtype=torch.long),
                "q1_padded": q1_padded_batch[b].clone(),
            }
            if has_hand:
                sample["hand_points_world"] = self._runtime_hand_points_world[index_list[b]].clone()
                sample["hand_normals_world"] = self._runtime_hand_normals_world[index_list[b]].clone()
            if has_q1_contact_points:
                sample["q1_contact_points_world"] = self._runtime_q1_contact_points_world[index_list[b]].clone()
            samples.append(sample)
        return samples

    @staticmethod
    def _read_buffer_payload(buffer_path: str | Path) -> Dict[str, Any]:
        # Load and validate a serialized probe-buffer payload.
        payload = torch.load(str(Path(buffer_path).resolve()), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "meta" not in payload or "buffers" not in payload:
            raise ValueError(f"Invalid probe buffer payload: {buffer_path}")
        return payload

    @classmethod
    def read_buffer_meta(cls, buffer_path: str | Path) -> Dict[str, Any]:
        # Return only the saved metadata without constructing a dataset instance.
        return cls._read_buffer_payload(buffer_path)["meta"]

    @classmethod
    def from_buffer_file(
        cls,
        buffer_path: str | Path,
        **kwargs: Any,
    ) -> "ContactPolicyProbeDataset":
        # Recreate a probe dataset around a previously saved fixed buffer.
        payload = cls._read_buffer_payload(buffer_path)
        return cls.from_buffer_payload(payload, **kwargs)

    @classmethod
    def from_buffer_payload(
        cls,
        payload: Dict[str, Any],
        **kwargs: Any,
    ) -> "ContactPolicyProbeDataset":
        # Recreate a probe dataset around an in-memory payload, including sliced
        # payloads used by lightweight viewers.
        meta = payload["meta"]
        init_kwargs = dict(kwargs)
        init_kwargs.pop("max_contact_points", None)
        requested_contact_input_points = int(init_kwargs.pop("contact_input_points", meta.get("contact_input_points", 32)))
        requested_non_contact_pattern_mode = str(
            init_kwargs.pop("non_contact_pattern_mode", meta.get("non_contact_pattern_mode", "surface_mixture"))
        )
        requested_non_contact_component_range = tuple(
            init_kwargs.pop("non_contact_component_range", meta.get("non_contact_component_range", (2, 4)))
        )
        requested_remove_non_contact_points = bool(
            init_kwargs.pop("remove_non_contact_points", meta.get("remove_non_contact_points", False))
        )
        requested_non_contact_causal_enabled = bool(
            init_kwargs.pop("non_contact_causal_enabled", meta.get("non_contact_causal_enabled", False))
        )
        requested_non_contact_empty_probability = float(
            init_kwargs.pop("non_contact_empty_probability", meta.get("non_contact_empty_probability", 0.0))
        )
        requested_non_contact_goal_joint_noise_scale = float(
            init_kwargs.pop(
                "non_contact_goal_joint_noise_scale",
                meta.get("non_contact_goal_joint_noise_scale", 0.0),
            )
        )
        requested_non_contact_goal_joint_select_min = int(
            init_kwargs.pop(
                "non_contact_goal_joint_select_min",
                meta.get("non_contact_goal_joint_select_min", 0),
            )
        )
        requested_non_contact_goal_joint_select_max = int(
            init_kwargs.pop(
                "non_contact_goal_joint_select_max",
                meta.get("non_contact_goal_joint_select_max", -1),
            )
        )
        requested_non_contact_blocker_fraction = float(
            init_kwargs.pop("non_contact_blocker_fraction", meta.get("non_contact_blocker_fraction", 0.0))
        )
        requested_non_contact_blocker_candidate_points = int(
            init_kwargs.pop(
                "non_contact_blocker_candidate_points",
                meta.get("non_contact_blocker_candidate_points", 128),
            )
        )
        requested_q1_translation_noise_std = float(
            init_kwargs.pop("q1_translation_noise_std", meta.get("q1_translation_noise_std", 0.0))
        )
        requested_q1_rotation_noise_std_deg = float(
            init_kwargs.pop("q1_rotation_noise_std_deg", meta.get("q1_rotation_noise_std_deg", 0.0))
        )
        requested_q1_joint_noise_scale = float(
            init_kwargs.pop("q1_joint_noise_scale", meta.get("q1_joint_noise_scale", 0.0))
        )
        requested_q1_zero_probability = float(
            init_kwargs.pop("q1_zero_probability", meta.get("q1_zero_probability", 0.10))
        )
        requested_random_q1 = bool(init_kwargs.pop("random_q1", meta.get("random_q1", False)))
        requested_q1_sampling_mode = str(init_kwargs.pop("q1_sampling_mode", meta.get("q1_sampling_mode", "gaussian")))
        requested_q1_contact_error_zero_prob = float(
            init_kwargs.pop("q1_contact_error_zero_prob", meta.get("q1_contact_error_zero_prob", 0.08))
        )
        requested_q1_contact_error_bins_mm = init_kwargs.pop(
            "q1_contact_error_bins_mm",
            meta.get(
                "q1_contact_error_bins_mm",
                [(0.2, 1.0, 0.32), (1.0, 3.0, 0.30), (3.0, 8.0, 0.20), (8.0, 20.0, 0.10)],
            ),
        )
        requested_q1_contact_error_inactive_scale = float(
            init_kwargs.pop(
                "q1_contact_error_inactive_scale",
                meta.get("q1_contact_error_inactive_scale", 0.05),
            )
        )
        requested_q1_contact_error_refine_steps = int(
            init_kwargs.pop(
                "q1_contact_error_refine_steps",
                meta.get("q1_contact_error_refine_steps", 2),
            )
        )
        requested_sparse_sampling_mode = str(
            init_kwargs.pop("sparse_sampling_mode", meta.get("sparse_sampling_mode", "point_uniform"))
        )
        requested_composite_synthesis_mode = str(
            init_kwargs.pop("composite_synthesis_mode", meta.get("composite_synthesis_mode", "independent"))
        )
        requested_composite_nearby_max_distance = float(
            init_kwargs.pop("composite_nearby_max_distance", meta.get("composite_nearby_max_distance", 0.12))
        )
        requested_composite_contact_count_ranges = init_kwargs.pop(
            "composite_contact_count_ranges",
            meta.get("composite_contact_count_ranges", None),
        )
        requested_composite_collision_filter_enabled = bool(
            init_kwargs.pop(
                "composite_collision_filter_enabled",
                meta.get("composite_collision_filter_enabled", False),
            )
        )
        requested_composite_collision_filter_margin = float(
            init_kwargs.pop(
                "composite_collision_filter_margin",
                meta.get("composite_collision_filter_margin", 0.0),
            )
        )
        requested_composite_collision_filter_max_attempts = int(
            init_kwargs.pop(
                "composite_collision_filter_max_attempts",
                meta.get("composite_collision_filter_max_attempts", 64),
            )
        )
        # hand_points_fps is a runtime-only knob (not baked into the buffer), so the caller
        # controls it — we fall back to whatever was saved if the caller didn't override.
        requested_hand_points_fps = int(
            init_kwargs.pop("hand_points_fps", meta.get("hand_points_fps", 0))
        )
        return cls(
            contact_mode=str(meta["contact_mode"]),
            sixd_pose_mode=str(meta["sixd_pose_mode"]),
            robot_name=str(meta["robot_name"]),
            samples=int(meta["samples"]),
            build_batch_size=1,
            progress_label=str(meta.get("progress_label", "probe_dataset")),
            fixed_contact_links=meta["fixed_contact_links"],
            probe_component_range=tuple(meta["probe_component_range"]),
            probe_component_count_values=meta["probe_component_count_values"],
            probe_component_count_distribution=meta["probe_component_count_distribution"],
            probe_contact_count_range=tuple(meta["probe_contact_count_range"]),
            contact_input_points=requested_contact_input_points,
            non_contact_pattern_mode=requested_non_contact_pattern_mode,
            non_contact_component_range=requested_non_contact_component_range,
            remove_non_contact_points=requested_remove_non_contact_points,
            non_contact_causal_enabled=requested_non_contact_causal_enabled,
            non_contact_empty_probability=requested_non_contact_empty_probability,
            non_contact_goal_joint_noise_scale=requested_non_contact_goal_joint_noise_scale,
            non_contact_goal_joint_select_min=requested_non_contact_goal_joint_select_min,
            non_contact_goal_joint_select_max=requested_non_contact_goal_joint_select_max,
            non_contact_blocker_fraction=requested_non_contact_blocker_fraction,
            non_contact_blocker_candidate_points=requested_non_contact_blocker_candidate_points,
            q1_translation_noise_std=requested_q1_translation_noise_std,
            q1_rotation_noise_std_deg=requested_q1_rotation_noise_std_deg,
            q1_joint_noise_scale=requested_q1_joint_noise_scale,
            q1_zero_probability=requested_q1_zero_probability,
            random_q1=requested_random_q1,
            q1_sampling_mode=requested_q1_sampling_mode,
            q1_contact_error_zero_prob=requested_q1_contact_error_zero_prob,
            q1_contact_error_bins_mm=requested_q1_contact_error_bins_mm,
            q1_contact_error_inactive_scale=requested_q1_contact_error_inactive_scale,
            q1_contact_error_refine_steps=requested_q1_contact_error_refine_steps,
            sparse_sampling_mode=requested_sparse_sampling_mode,
            composite_synthesis_mode=requested_composite_synthesis_mode,
            composite_nearby_max_distance=requested_composite_nearby_max_distance,
            composite_contact_count_ranges=requested_composite_contact_count_ranges,
            composite_collision_filter_enabled=requested_composite_collision_filter_enabled,
            composite_collision_filter_margin=requested_composite_collision_filter_margin,
            composite_collision_filter_max_attempts=requested_composite_collision_filter_max_attempts,
            hand_points_fps=requested_hand_points_fps,
            load_buffer_payload=payload,
            max_contact_points=int(meta.get("candidate_contact_points", meta["max_contact_points"])),
            **init_kwargs,
        )

    def _build_buffer_payload(self) -> Dict[str, Any]:
        # Serialize fixed contact/q buffers plus reload-time probe configuration.
        buffers: Dict[str, torch.Tensor] = {
            "contact_cloud": self.buffer_contact_cloud.detach().cpu(),
            "contact_valid_mask": self.buffer_contact_valid_mask.detach().cpu(),
            "contact_local_points": self.buffer_contact_local_points.detach().cpu(),
            "contact_local_normals": self.buffer_contact_local_normals.detach().cpu(),
            "contact_link_indices": self.buffer_contact_link_indices.detach().cpu(),
            "contact_parent_sparse_indices": self.buffer_contact_parent_sparse_indices.detach().cpu(),
            "contact_cluster_ids": self.buffer_contact_cluster_ids.detach().cpu(),
            "surface_mask": self.buffer_surface_mask.detach().cpu(),
            "action": self.buffer_action.detach().cpu(),
            "q2_padded": self.buffer_q2_padded.detach().cpu(),
        }
        buffers["q1_padded"] = self.buffer_q1_padded.detach().cpu()
        return {
            "meta": {
                "contact_mode": self.contact_mode,
                "sixd_pose_mode": self.sixd_pose_mode,
                "robot_name": self.robot_name,
                "robot_model_kwargs": dict(self.robot_model_kwargs),
                "component_sampling_mode": str(self.component_sampling_mode),
                "anchor_sampling_mode": str(self.anchor_sampling_mode),
                "threshold": float(self.threshold),
                "align_exp_scale": float(self.align_exp_scale),
                "sigmoid_scale": float(self.sigmoid_scale),
                "anchor_temperature": float(self.anchor_temperature),
                "patch_anchor_shift_min": float(self.patch_anchor_shift_min),
                "patch_anchor_shift_max": float(self.patch_anchor_shift_max),
                "patch_extent_min": float(self.patch_extent_min),
                "patch_extent_max": float(self.patch_extent_max),
                "patch_extent_power": float(self.patch_extent_power),
                "patch_shift_power": float(self.patch_shift_power),
                "patch_normal_jitter_max_deg": float(self.patch_normal_jitter_max_deg),
                "patch_points_per_anchor_min": int(self.patch_points_per_anchor_min),
                "patch_points_per_anchor_max": int(self.patch_points_per_anchor_max),
                "patch_penetration_clearance": float(self.patch_penetration_clearance),
                "samples": int(self.samples),
                "max_contact_points": int(self.candidate_contact_points),
                "candidate_contact_points": int(self.candidate_contact_points),
                "contact_input_points": int(self.contact_input_points),
                "non_contact_input_points": int(self.non_contact_input_points),
                "non_contact_sampling_radius": float(self.non_contact_sampling_radius),
                "non_contact_sdf_clearance": float(self.non_contact_sdf_clearance),
                "non_contact_pattern_mode": self.non_contact_pattern_mode,
                "non_contact_component_range": list(self.non_contact_component_range),
                "non_contact_candidate_points": int(self.non_contact_candidate_points),
                "non_contact_collision_surface_points": int(self.non_contact_collision_surface_points),
                "non_contact_epoch_batch_size": int(self.non_contact_epoch_batch_size),
                "remove_non_contact_points": bool(self.remove_non_contact_points),
                "non_contact_causal_enabled": bool(self.non_contact_causal_enabled),
                "non_contact_empty_probability": float(self.non_contact_empty_probability),
                "non_contact_goal_joint_noise_scale": float(self.non_contact_goal_joint_noise_scale),
                "non_contact_goal_joint_select_min": int(self.non_contact_goal_joint_select_min),
                "non_contact_goal_joint_select_max": int(self.non_contact_goal_joint_select_max),
                "non_contact_blocker_fraction": float(self.non_contact_blocker_fraction),
                "non_contact_blocker_candidate_points": int(self.non_contact_blocker_candidate_points),
                "collision_primitive_spec_dir": str(self.collision_primitive_spec_dir),
                "collision_primitive_spec_path": (
                    None if self.collision_primitive_spec_path is None else str(self.collision_primitive_spec_path)
                ),
                "collision_primitive_fit_method": self.collision_primitive_fit_method,
                "contact_cluster_mode": self.contact_cluster_mode,
                "q1_translation_noise_std": float(self.q1_translation_noise_std),
                "q1_rotation_noise_std_deg": float(self.q1_rotation_noise_std_deg),
                "q1_joint_noise_scale": float(self.q1_joint_noise_scale),
                "q1_zero_probability": float(self.q1_zero_probability),
                "random_q1": bool(self.random_q1),
                "q1_sampling_mode": str(self.q1_sampling_mode),
                "q1_contact_error_zero_prob": float(self.q1_contact_error_zero_prob),
                "q1_contact_error_bins_mm": [list(x) for x in self.q1_contact_error_bins_mm],
                "q1_contact_error_inactive_scale": float(self.q1_contact_error_inactive_scale),
                "q1_contact_error_refine_steps": int(self.q1_contact_error_refine_steps),
                "sparse_sampling_mode": str(self.sparse_sampling_mode),
                "sparse_region_counts": list(self._sparse_region_counts()),
                "composite_synthesis_mode": str(self.composite_synthesis_mode),
                "composite_nearby_max_distance": float(self.composite_nearby_max_distance),
                "composite_contact_count_ranges": {
                    str(k): [int(v[0]), int(v[1])] for k, v in self.composite_contact_count_ranges.items()
                },
                "composite_collision_filter_enabled": bool(self.composite_collision_filter_enabled),
                "composite_collision_filter_margin": float(self.composite_collision_filter_margin),
                "composite_collision_filter_max_attempts": int(self.composite_collision_filter_max_attempts),
                "pose_groups": [
                    {
                        "name": str(group["name"]),
                        "translation_indices": [int(i) for i in group["translation_indices"]],
                        "rotation_indices": [int(i) for i in group["rotation_indices"]],
                    }
                    for group in self._model_pose_groups()
                ],
                "hand_points_fps": int(self.hand_points_fps),
                "fixed_contact_links": list(self.fixed_contact_links),
                "probe_component_range": list(self.probe_component_range),
                "probe_component_count_values": self.probe_component_values.detach().cpu(),
                "probe_component_count_distribution": self.probe_component_distribution.detach().cpu(),
                "probe_contact_count_range": list(self.probe_contact_count_range),
                "progress_label": self.progress_label,
                "build_seed": int(self.seed),
            },
            "buffers": buffers,
        }

    def save_buffer(self, buffer_path: str | Path) -> None:
        # Persist the fixed probe buffer for deterministic later evaluation.
        path = Path(buffer_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._build_buffer_payload(), str(path))

    # Loading bypasses synthesis completely and restores only the fixed split tensors.
    def _load_buffer_payload(self, payload: Dict[str, Any]) -> None:
        # Validate saved metadata and copy all fixed-buffer tensors into this instance.
        meta = payload["meta"]
        buffers = payload["buffers"]
        if int(meta["samples"]) != int(self.samples):
            raise ValueError(f"Saved probe buffer samples mismatch: {meta['samples']} vs {self.samples}")
        if str(meta["contact_mode"]) != self.contact_mode:
            raise ValueError(f"Saved probe buffer contact_mode mismatch: {meta['contact_mode']} vs {self.contact_mode}")
        if str(meta["sixd_pose_mode"]) != self.sixd_pose_mode:
            raise ValueError(
                f"Saved probe buffer sixd_pose_mode mismatch: {meta['sixd_pose_mode']} vs {self.sixd_pose_mode}"
            )
        if str(meta["robot_name"]) != self.robot_name:
            raise ValueError(f"Saved probe buffer robot mismatch: {meta['robot_name']} vs {self.robot_name}")
        saved_robot_model_kwargs = dict(meta.get("robot_model_kwargs", {}))
        if saved_robot_model_kwargs != dict(self.robot_model_kwargs):
            raise ValueError(
                f"Saved probe buffer robot_model_kwargs mismatch: "
                f"{saved_robot_model_kwargs} vs {dict(self.robot_model_kwargs)}"
            )
        candidate_contact_points = int(meta.get("candidate_contact_points", meta["max_contact_points"]))
        if candidate_contact_points != int(self.candidate_contact_points):
            raise ValueError(
                f"Saved probe buffer candidate_contact_points mismatch: {candidate_contact_points} vs {self.candidate_contact_points}"
            )
        if str(meta.get("contact_cluster_mode", "")) != str(self.contact_cluster_mode):
            raise ValueError(
                f"Saved probe buffer contact_cluster_mode mismatch: {meta.get('contact_cluster_mode')} vs {self.contact_cluster_mode}"
            )
        if str(meta.get("sparse_sampling_mode", "point_uniform")) != str(self.sparse_sampling_mode):
            raise ValueError(
                f"Saved probe buffer sparse_sampling_mode mismatch: "
                f"{meta.get('sparse_sampling_mode', 'point_uniform')} vs {self.sparse_sampling_mode}"
            )
        if self.sparse_sampling_mode == "region_uniform":
            saved_counts = [int(x) for x in meta.get("sparse_region_counts", [])]
            current_counts = list(self._sparse_region_counts())
            if saved_counts != current_counts:
                raise ValueError(
                    f"Saved probe buffer sparse_region_counts mismatch: {saved_counts} vs {current_counts}"
                )
        if str(meta.get("q1_sampling_mode", "gaussian")) != str(self.q1_sampling_mode):
            raise ValueError(
                f"Saved probe buffer q1_sampling_mode mismatch: "
                f"{meta.get('q1_sampling_mode', 'gaussian')} vs {self.q1_sampling_mode}"
            )
        if str(meta.get("composite_synthesis_mode", "independent")) != str(self.composite_synthesis_mode):
            raise ValueError(
                f"Saved probe buffer composite_synthesis_mode mismatch: "
                f"{meta.get('composite_synthesis_mode', 'independent')} vs {self.composite_synthesis_mode}"
            )
        saved_collision_enabled = bool(meta.get("composite_collision_filter_enabled", False))
        if saved_collision_enabled != bool(self.composite_collision_filter_enabled):
            raise ValueError(
                f"Saved probe buffer composite_collision_filter_enabled mismatch: "
                f"{saved_collision_enabled} vs {self.composite_collision_filter_enabled}"
            )
        if saved_collision_enabled or bool(self.composite_collision_filter_enabled):
            saved_collision_margin = float(meta.get("composite_collision_filter_margin", 0.0))
            if not math.isclose(
                saved_collision_margin,
                float(self.composite_collision_filter_margin),
                rel_tol=1.0e-6,
                abs_tol=1.0e-8,
            ):
                raise ValueError(
                    f"Saved probe buffer composite_collision_filter_margin mismatch: "
                    f"{saved_collision_margin} vs {self.composite_collision_filter_margin}"
                )
            saved_collision_attempts = int(meta.get("composite_collision_filter_max_attempts", 64))
            if saved_collision_attempts != int(self.composite_collision_filter_max_attempts):
                raise ValueError(
                    f"Saved probe buffer composite_collision_filter_max_attempts mismatch: "
                    f"{saved_collision_attempts} vs {self.composite_collision_filter_max_attempts}"
                )

        # Restore core contact bank, q2/action, and required metadata tensors.
        self.buffer_contact_cloud.copy_(torch.as_tensor(buffers["contact_cloud"], dtype=torch.float32, device="cpu"))
        self.buffer_contact_valid_mask.copy_(
            torch.as_tensor(buffers["contact_valid_mask"], dtype=torch.bool, device="cpu")
        )
        if (
            "contact_local_points" not in buffers
            or "contact_local_normals" not in buffers
            or "contact_link_indices" not in buffers
            or "contact_parent_sparse_indices" not in buffers
            or "contact_cluster_ids" not in buffers
            or "surface_mask" not in buffers
        ):
            raise ValueError("Saved probe buffer is missing current-format contact metadata")
        self.buffer_contact_local_points.copy_(
            torch.as_tensor(buffers["contact_local_points"], dtype=torch.float32, device="cpu")
        )
        self.buffer_contact_local_normals.copy_(
            torch.as_tensor(buffers["contact_local_normals"], dtype=torch.float32, device="cpu")
        )
        self.buffer_contact_link_indices.copy_(
            torch.as_tensor(buffers["contact_link_indices"], dtype=torch.int16, device="cpu")
        )
        self.buffer_contact_parent_sparse_indices.copy_(
            torch.as_tensor(buffers["contact_parent_sparse_indices"], dtype=torch.int16, device="cpu")
        )
        self.buffer_contact_cluster_ids.copy_(
            torch.as_tensor(buffers["contact_cluster_ids"], dtype=torch.int16, device="cpu")
        )
        self.buffer_surface_mask.copy_(torch.as_tensor(buffers["surface_mask"], dtype=torch.bool, device="cpu"))
        self.buffer_action.copy_(torch.as_tensor(buffers["action"], dtype=torch.float32, device="cpu"))
        self.buffer_q2_padded.copy_(torch.as_tensor(buffers["q2_padded"], dtype=torch.float32, device="cpu"))
        if "q1_padded" not in buffers:
            raise ValueError("Saved probe buffer is missing q1_padded")
        self.buffer_q1_padded.copy_(torch.as_tensor(buffers["q1_padded"], dtype=torch.float32, device="cpu"))


