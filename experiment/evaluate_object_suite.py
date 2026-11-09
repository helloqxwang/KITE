"""Evaluate a high-level policy across a mixed YCB object-suite split."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys
import tempfile
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.evaluate_policy import (  # noqa: E402
    _load_hydra_config,
    _merge_result_payloads,
    _merge_training_payloads,
    _print_summary,
    _resolve_experiment_path,
    _resolve_project_path,
    _results_from_payload,
    _save_npz,
    _selected_eval_indices,
    main as eval_main,
)
from experiment.high_level_policy.dataset import _episode_split_indices  # noqa: E402
from experiment.high_level_policy.payload import load_high_level_payload  # noqa: E402
from experiment.tabletop_sim.factory import optional_project_path  # noqa: E402


from paper_config import runtime_config


def _configured_payload(cfg: DictConfig) -> tuple[dict[str, Any], DictConfig, Path]:
    high_cfg_path = _resolve_project_path(
        OmegaConf.select(
            cfg,
            "policy_eval.high_level_config_path",
            default="./experiment/conf/train_policy/object_suite_paper2400.yaml",
        ),
        label="policy_eval.high_level_config_path",
    )
    high_cfg = _load_hydra_config(high_cfg_path)
    override = OmegaConf.select(cfg, "policy_eval.payload_path", default=None)
    payload_value = (
        override
        if override is not None and str(override).strip().lower() not in {"", "null", "none"}
        else OmegaConf.select(high_cfg, "dataset.payload_path", default=None)
    )
    payload_path = _resolve_experiment_path(
        payload_value,
        label="policy_eval.payload_path or high-level dataset.payload_path",
    )
    return load_high_level_payload(payload_path), high_cfg, payload_path


def _selected_object_groups(cfg: DictConfig) -> list[tuple[str, list[int]]]:
    payload, high_cfg, payload_path = _configured_payload(cfg)
    episode_meta = list(payload["episode_meta"])
    episode_count = len(episode_meta)
    train_indices, test_indices = _episode_split_indices(
        episode_count=episode_count,
        test_episode_count=int(OmegaConf.select(high_cfg, "dataset.test_episode_count", default=20)),
        split_seed=int(OmegaConf.select(high_cfg, "dataset.split_seed", default=42)),
        payload_path=str(payload_path),
    )
    split = str(OmegaConf.select(cfg, "policy_eval.split", default="test")).strip().lower()
    if split == "train":
        split_indices = train_indices
    elif split == "test":
        split_indices = test_indices
    elif split == "all":
        split_indices = np.arange(episode_count, dtype=np.int64)
    else:
        raise ValueError(f"policy_eval.split must be train, test, or all; got {split!r}")

    selected = _selected_eval_indices(
        split_indices=split_indices.tolist(),
        source_count=episode_count,
        split=split,
        indices=OmegaConf.select(cfg, "policy_eval.indices", default=None),
        start_index=int(OmegaConf.select(cfg, "policy_eval.start_index", default=0)),
        count=OmegaConf.select(cfg, "policy_eval.count", default=None),
    )
    if not selected:
        raise ValueError("policy_eval selected no object-suite episodes")

    grouped: OrderedDict[str, list[int]] = OrderedDict()
    for source_index in selected:
        meta = episode_meta[int(source_index)]
        object_id = meta.get("object_id") if isinstance(meta, dict) else None
        if object_id is None or not str(object_id).strip():
            raise KeyError(
                f"High-level payload episode_meta[{source_index}] has no object_id; "
                "mixed object-suite evaluation cannot select the matching scene asset."
            )
        grouped.setdefault(str(object_id), []).append(int(source_index))
    return list(grouped.items())


def _object_output_path(value: Any, object_id: str) -> Path | None:
    base_value = optional_project_path(value)
    if base_value is None:
        return None
    base = Path(base_value)
    safe_id = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in object_id)
    return base.with_name(f"{base.stem}__{safe_id}{base.suffix}")


@hydra.main(
    version_base="1.2",
    config_path="conf",
    config_name="evaluate_policy/object_suite_wuji",
)
def main(cfg: DictConfig) -> None:
    cfg = runtime_config(cfg)
    OmegaConf.resolve(cfg)
    if not bool(OmegaConf.select(cfg, "policy_eval.group_object_suite", default=True)):
        eval_main.__wrapped__(cfg)
        return

    groups = _selected_object_groups(cfg)
    selected_count = sum(len(indices) for _, indices in groups)
    print(
        f"object-suite evaluation: selected={selected_count} object_groups={len(groups)} "
        f"objects={[object_id for object_id, _ in groups]}",
        flush=True,
    )

    aggregate_results = optional_project_path(
        OmegaConf.select(cfg, "policy_eval.save_results_npz", default=None)
    )
    aggregate_training = optional_project_path(
        OmegaConf.select(cfg, "policy_eval.save_training_pt", default=None)
    )
    trace_value = OmegaConf.select(cfg, "policy_eval.save_vis_trace_npz", default=None)

    with tempfile.TemporaryDirectory(prefix="object_suite_policy_eval_") as tmp_dir_text:
        tmp_dir = Path(tmp_dir_text)
        result_paths: list[Path] = []
        training_paths: list[Path] = []
        for group_index, (object_id, indices) in enumerate(groups):
            group_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            OmegaConf.update(group_cfg, "objects.active_id", object_id, merge=False)
            OmegaConf.update(group_cfg, "policy_eval.indices", indices, merge=False)
            OmegaConf.update(group_cfg, "policy_eval.start_index", 0, merge=False)
            OmegaConf.update(group_cfg, "policy_eval.count", None, merge=False)

            group_result_path = tmp_dir / f"results_{group_index:03d}.npz"
            OmegaConf.update(
                group_cfg,
                "policy_eval.save_results_npz",
                str(group_result_path),
                merge=False,
            )
            if aggregate_training is not None:
                group_training_path = tmp_dir / f"training_{group_index:03d}.pt"
                OmegaConf.update(
                    group_cfg,
                    "policy_eval.save_training_pt",
                    str(group_training_path),
                    merge=False,
                )
                training_paths.append(group_training_path)
            if trace_value is not None and str(trace_value).strip().lower() not in {"", "null", "none"}:
                group_trace_path = _object_output_path(trace_value, object_id)
                OmegaConf.update(
                    group_cfg,
                    "policy_eval.save_vis_trace_npz",
                    str(group_trace_path),
                    merge=False,
                )

            print(
                f"object group {group_index + 1}/{len(groups)}: "
                f"object_id={object_id} source_indices={indices}",
                flush=True,
            )
            eval_main.__wrapped__(group_cfg)
            result_paths.append(group_result_path)

        merged = _merge_result_payloads(result_paths)
        print("aggregate object-suite result:", flush=True)
        _print_summary(_results_from_payload(merged))
        if aggregate_results is not None:
            out = _save_npz(aggregate_results, merged)
            print(f"saved aggregate object-suite results: {out}", flush=True)
        if aggregate_training is not None:
            out = _merge_training_payloads(training_paths, aggregate_training)
            print(f"saved aggregate object-suite training payload: {out}", flush=True)


if __name__ == "__main__":
    from paper_config import canonicalize_cli
    canonicalize_cli()
    main()
