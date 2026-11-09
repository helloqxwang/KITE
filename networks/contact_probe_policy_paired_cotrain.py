from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from networks.contact_probe_policy_paired import ContactProbePolicyPairedTransformer


# Shape-agnostic submodules that are shared across manipulators regardless of config:
# they carry the backbone representations (hand tokens, contact tokens, scene mixing,
# action decoder, flow/diffusion time embedding) and their semantics are identical
# across manipulators by construction.
_ALWAYS_SHARED_SUBMODULE_NAMES: Tuple[str, ...] = (
    "hand_proj",
    "contact_proj",
    "scene_encoder",
    "contact_only_encoder",
    "contact_slot_encoder",
    "cls_unpacker",
    "slot_to_scene_proj",
    "bottleneck_scene_encoder",
    "non_contact_context_encoder",
    "time_mlp",
    "action_decoder",
)

# Shape-agnostic action-token I/O: these map between world-frame base-pose residual
# channels (xyz / rot6d) and d_model. Their input/output dimensions match across
# manipulators, but the residual channels represent different physical slots and
# different data distributions — so the user may want each manipulator to learn its
# own projection. Controlled by ``per_manipulator_action_io`` on the wrapper.
_ACTION_IO_SUBMODULE_NAMES: Tuple[str, ...] = (
    "noisy_xyz_proj",
    "noisy_rot_proj",
    "head_xyz",
    "head_rot",
)


def _alias_module(
    target: ContactProbePolicyPairedTransformer,
    source: ContactProbePolicyPairedTransformer,
    name: str,
) -> None:
    """Replace ``target.<name>`` with ``source.<name>`` if both exist and are modules."""
    src_mod = getattr(source, name, None)
    tgt_mod = getattr(target, name, None)
    if src_mod is None or tgt_mod is None:
        return
    if not isinstance(src_mod, nn.Module) or not isinstance(tgt_mod, nn.Module):
        return
    src_params = tuple(src_mod.parameters())
    tgt_params = tuple(tgt_mod.parameters())
    if len(src_params) != len(tgt_params) or any(sp.shape != tp.shape for sp, tp in zip(src_params, tgt_params)):
        return
    setattr(target, name, src_mod)


def _alias_action_queries(
    target: ContactProbePolicyPairedTransformer,
    source: ContactProbePolicyPairedTransformer,
) -> None:
    """Alias the learned ``action_queries`` Parameter, preserving the registration slot."""
    src_queries = getattr(source, "action_queries", None)
    tgt_queries = getattr(target, "action_queries", None)
    if not isinstance(src_queries, nn.Parameter) or not isinstance(tgt_queries, nn.Parameter):
        return
    if src_queries.shape != tgt_queries.shape:
        return
    if "action_queries" in target._parameters:
        del target._parameters["action_queries"]
    target.action_queries = src_queries


def _share_submodules(
    target: ContactProbePolicyPairedTransformer,
    source: ContactProbePolicyPairedTransformer,
    *,
    per_manipulator_action_io: bool,
) -> None:
    """Alias shape-agnostic submodules of ``target`` onto ``source``'s copies.

    When ``per_manipulator_action_io=False`` (default), the action-IO projections
    (``noisy_xyz_proj``, ``noisy_rot_proj``, ``head_xyz``, ``head_rot``) and the
    learned ``action_queries`` Parameter are aliased as well. When ``True``, those
    five components remain per-manipulator — the caller wants each head to learn
    its own action-token I/O (e.g. so ``action_queries`` can act as a manipulator-
    identifying prompt).
    """
    for name in _ALWAYS_SHARED_SUBMODULE_NAMES:
        _alias_module(target, source, name)
    if not per_manipulator_action_io:
        for name in _ACTION_IO_SUBMODULE_NAMES:
            _alias_module(target, source, name)
        _alias_action_queries(target, source)


class ContactProbePolicyPairedCoTrain(nn.Module):
    """Co-training wrapper over ``ContactProbePolicyPairedTransformer``.

    Instantiates one probe head per manipulator. The first head is built natively; each
    subsequent head has its shape-agnostic submodules (hand/contact/scene encoders, paired
    transformer blocks, action decoder, time/flow embeddings — and, depending on
    ``per_manipulator_action_io``, the action-IO projections + learned queries) aliased
    to the first head's. Per-manipulator heads always retain their own:

    * ``noisy_joint_proj`` and ``head_joints`` (sized to that manipulator's joint slots)
    * ``action_normalizer_scale/offset`` and ``residual_normalizer_scale/offset``
    * residual slot layout (``_slot``, ``_base_translation_indices``, ``_base_rotation_indices``)

    ``nn.Module.parameters()`` deduplicates Parameter objects by identity, so the optimizer
    sees each shared weight exactly once; gradients summed from per-manipulator losses
    accumulate on shared parameters naturally.

    Checkpoint note: state_dict keys are keyed by ``heads.<name>.<submodule>.*`` whether
    a given submodule is shared or not. A checkpoint saved with one flag therefore still
    loads (``strict=True``) into a model built with the other flag — aliased entries are
    just duplicated identical values on disk.
    """

    def __init__(
        self,
        *,
        manipulators: Sequence[Dict[str, Any]],
        shared_kwargs: Dict[str, Any],
        per_manipulator_action_io: bool = False,
    ) -> None:
        super().__init__()
        if len(manipulators) == 0:
            raise ValueError("At least one manipulator is required for co-training.")

        self._manipulator_names: List[str] = [str(m["name"]) for m in manipulators]
        self._manipulator_to_index: Dict[str, int] = {
            name: idx for idx, name in enumerate(self._manipulator_names)
        }
        self.variant: str = str(shared_kwargs.get("variant", "flow_matching"))
        self.per_manipulator_action_io: bool = bool(per_manipulator_action_io)
        action_tokenization = str(shared_kwargs.get("action_tokenization", "grouped")).strip().lower()
        if action_tokenization == "per_joint" and not self.per_manipulator_action_io:
            raise ValueError(
                "model.action_tokenization='per_joint' requires "
                "co_training.per_manipulator_action_io=true because joint-token counts "
                "can differ across manipulators."
            )

        heads_dict = nn.ModuleDict()
        first_head: Optional[ContactProbePolicyPairedTransformer] = None

        for manip in manipulators:
            name = str(manip["name"])
            head_kwargs = dict(shared_kwargs)
            head_kwargs["action_dim"] = int(manip["action_dim"])
            head_kwargs["robot_names"] = [name]
            head_kwargs["action_normalizer_scale"] = manip["action_normalizer_scale"].view(1, -1).clone()
            head_kwargs["action_normalizer_offset"] = manip["action_normalizer_offset"].view(1, -1).clone()
            head_kwargs["base_translation_indices"] = [int(x) for x in manip["base_translation_indices"]]
            head_kwargs["base_rotation_indices"] = [int(x) for x in manip["base_rotation_indices"]]
            if "pose_groups" in manip and manip["pose_groups"] is not None:
                head_kwargs["pose_groups"] = manip["pose_groups"]
            if "residual_normalizer_scale" in manip and manip["residual_normalizer_scale"] is not None:
                head_kwargs["residual_normalizer_scale"] = manip["residual_normalizer_scale"].view(1, -1).clone()
            if "residual_normalizer_offset" in manip and manip["residual_normalizer_offset"] is not None:
                head_kwargs["residual_normalizer_offset"] = manip["residual_normalizer_offset"].view(1, -1).clone()
            head = ContactProbePolicyPairedTransformer(**head_kwargs)
            if first_head is None:
                first_head = head
            else:
                _share_submodules(
                    head, first_head, per_manipulator_action_io=self.per_manipulator_action_io
                )
            heads_dict[name] = head

        self.heads: nn.ModuleDict = heads_dict

    @property
    def manipulator_names(self) -> List[str]:
        return list(self._manipulator_names)

    def head_for(self, name: str) -> ContactProbePolicyPairedTransformer:
        return self.heads[str(name)]

    def compute_loss(
        self,
        batches: Dict[str, Dict[str, torch.Tensor]],
        *,
        return_aux: bool = False,
    ) -> Any:
        """Compute the mean of per-manipulator probe losses.

        ``batches`` is a mapping from manipulator name to the usual probe batch (same keys
        a single-manipulator trainer would pass). Each batch flows through its own head;
        losses are averaged across manipulators so the overall scale stays comparable to a
        single-manipulator run. Gradients accumulate on shared submodules via autograd.
        """
        if len(batches) == 0:
            raise ValueError("compute_loss requires at least one manipulator batch.")
        device = next(self.parameters()).device
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        agg_dict: Dict[str, torch.Tensor] = {}
        aux_by_name: Dict[str, Dict[str, Any]] = {}
        num_batches = len(batches)

        for name, batch in batches.items():
            head = self.heads[str(name)]
            if return_aux:
                loss, head_loss_dict, aux = head.compute_loss(batch, return_aux=True)
                aux_by_name[str(name)] = aux
            else:
                loss, head_loss_dict = head.compute_loss(batch, return_aux=False)
            total_loss = total_loss + loss
            for key, value in head_loss_dict.items():
                agg_dict[f"{name}/{key}"] = value

        total_loss = total_loss / float(num_batches)
        agg_dict["model_loss"] = total_loss.detach()

        if return_aux:
            return total_loss, agg_dict, aux_by_name
        return total_loss, agg_dict

    @torch.no_grad()
    def predict_action(
        self,
        batches: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, batch in batches.items():
            out[str(name)] = self.heads[str(name)].predict_action(batch)
        return out
