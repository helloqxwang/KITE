"""Small, config-only switch for deterministic Blender trace capture paths."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf


DEFAULT_OUTPUT_ROOT = "./outputs/traces"

_TASK_SLUGS = {
    "object_suite_pick_place": "object_suite",
}

_EMBODIMENT_SLUGS = {
    "allegro_right": "allegro",
    "barrett": "barrett",
    "wujihand_right": "wuji",
    "marvin_wuji": "wuji",
    "robotiq_gripper_stick_armless": "robotiq_stick",
}


@dataclass(frozen=True)
class VisTraceCapturePaths:
    """Resolved outputs selected by :func:`configure_vis_trace_mode`."""

    task: str
    embodiment: str
    rollout_config: str | None
    source_indices: tuple[int, ...]
    stem: str
    results_path: Path
    trace_path: Path
    hydra_dir: Path
    config_snapshot_path: Path


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if not text:
        raise ValueError("vis_trace task and embodiment names must not be empty")
    return text


def _mode_settings(cfg: DictConfig) -> tuple[bool, str]:
    value = OmegaConf.select(cfg, "online_rollout.vis_trace", default=None)
    if value is None:
        value = OmegaConf.select(cfg, "vis_trace", default=None)
    if value is None:
        return False, DEFAULT_OUTPUT_ROOT

    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        enabled = bool(value.get("enabled", False))
        output_root = str(value.get("output_root", DEFAULT_OUTPUT_ROOT))
        return enabled, output_root
    if isinstance(value, bool):
        return value, DEFAULT_OUTPUT_ROOT
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True, DEFAULT_OUTPUT_ROOT
        if normalized in {"false", "no", "off", "0", "", "none", "null"}:
            return False, DEFAULT_OUTPUT_ROOT
    raise TypeError("online_rollout.vis_trace must be a boolean or a mapping with enabled")


def _parse_explicit_indices(value: Any) -> tuple[int, ...]:
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"}):
        raise ValueError(
            "vis_trace mode requires explicit online_rollout.indices; "
            "use an integer or a list such as [0,1,2,3,4]"
        )

    if isinstance(value, (ListConfig, list, tuple)):
        raw = [int(index) for index in list(value)]
    elif isinstance(value, int) and not isinstance(value, bool):
        raw = [int(value)]
    else:
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if ":" in text and "," not in text:
            pieces = text.split(":")
            if len(pieces) not in {2, 3} or not pieces[0] or not pieces[1]:
                raise ValueError(f"vis_trace requires a closed index slice, got {value!r}")
            start = int(pieces[0])
            stop = int(pieces[1])
            step = 1 if len(pieces) == 2 or not pieces[2] else int(pieces[2])
            if step <= 0:
                raise ValueError("vis_trace index slice step must be positive")
            raw = list(range(start, stop, step))
        else:
            raw = [int(part.strip()) for part in text.split(",") if part.strip()]

    indices = tuple(dict.fromkeys(raw))
    if not indices:
        raise ValueError("vis_trace mode requires at least one online_rollout index")
    if any(index < 0 for index in indices):
        raise ValueError("vis_trace online_rollout.indices must be non-negative")
    return indices


def _source_suffix(indices: tuple[int, ...]) -> str:
    if len(indices) == 1:
        return f"source{indices[0]}"
    runs: list[tuple[int, int]] = []
    for index in indices:
        if runs and index == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], index)
        else:
            runs.append((index, index))
    labels = [str(start) if start == end else f"{start}-{end}" for start, end in runs]
    return "sources" + "_".join(labels)


def _normalized_config_name(value: str) -> str:
    return str(value).strip().removeprefix("/").removesuffix(".yaml")


def _capture_base(
    *,
    cfg: DictConfig,
    project_root: Path,
    rollout_config: str | None,
    task: str,
    embodiment: str,
) -> str:
    """Use the central preset name, otherwise include the rollout config name.

    The official presets intentionally keep short existing names such as
    ``object_suite_allegro``. An unlisted tuning config instead receives its
    unique config-leaf name so it cannot silently overwrite that preset trace.
    Preset-specific capture overrides must also match the effective rollout.
    """

    if rollout_config is None:
        return f"{task}_{embodiment}"
    normalized = _normalized_config_name(rollout_config)
    render_config_path = project_root / "render" / "config.yaml"
    if render_config_path.is_file():
        render_config = yaml.safe_load(render_config_path.read_text(encoding="utf-8"))
        presets = render_config.get("presets", {}) if isinstance(render_config, Mapping) else {}
        task_presets = presets.get(task, {}) if isinstance(presets, Mapping) else {}
        preset = task_presets.get(embodiment) if isinstance(task_presets, Mapping) else None
        if (
            isinstance(preset, Mapping)
            and _normalized_config_name(str(preset.get("rollout_config", ""))) == normalized
            and all(
                OmegaConf.select(cfg, str(key)) == value
                for key, value in preset.get("capture_overrides", {}).items()
            )
        ):
            configured_trace = Path(str(preset.get("trace", ""))).stem
            marker = configured_trace.rfind("_source")
            if marker > 0 and configured_trace.endswith("_trace"):
                return _slug(configured_trace[:marker])
    return _slug(normalized.rsplit("/", 1)[-1])


def configure_vis_trace_mode(
    cfg: DictConfig,
    *,
    project_root: str | Path,
    rollout_config: str | None = None,
) -> VisTraceCapturePaths | None:
    """Apply the optional trace-capture mode and save its resolved config.

    Missing or disabled mode settings are a strict no-op. When enabled, the
    caller must provide explicit ``online_rollout.indices`` so a capture can
    never silently expand to every episode in a source payload.
    """

    enabled, configured_output_root = _mode_settings(cfg)
    if not enabled:
        return None

    indices = _parse_explicit_indices(OmegaConf.select(cfg, "online_rollout.indices", default=None))
    task_name = str(OmegaConf.select(cfg, "task.name", default="task"))
    embodiment_name = str(OmegaConf.select(cfg, "env.embodiment", default="embodiment"))
    task = _TASK_SLUGS.get(task_name, _slug(task_name))
    embodiment = _EMBODIMENT_SLUGS.get(embodiment_name, _slug(embodiment_name))

    project_root_path = Path(project_root).expanduser().resolve()
    root = Path(configured_output_root).expanduser()
    if not root.is_absolute():
        root = project_root_path / root
    root = root.resolve()
    capture_base = _capture_base(
        cfg=cfg,
        project_root=project_root_path,
        rollout_config=rollout_config,
        task=task,
        embodiment=embodiment,
    )
    stem = f"{capture_base}_{_source_suffix(indices)}"
    task_dir = root / task
    explicit_trace = OmegaConf.select(cfg, "online_rollout.vis_trace.trace_path", default=None)
    if explicit_trace:
        selected_trace = Path(str(explicit_trace)).expanduser()
        if not selected_trace.is_absolute():
            selected_trace = project_root_path / selected_trace
        selected_trace = selected_trace.resolve()
        allowed = (project_root_path / "render" / "outputs").resolve()
        if allowed not in selected_trace.parents or not selected_trace.name.endswith("_trace.npz"):
            raise ValueError("vis_trace.trace_path must be a *_trace.npz file below render/outputs")
        task_dir = selected_trace.parent
        stem = selected_trace.name.removesuffix("_trace.npz")
    results_path = task_dir / f"{stem}_results.npz"
    trace_path = task_dir / f"{stem}_trace.npz"
    hydra_dir = task_dir / f"{stem}_hydra"
    config_snapshot_path = hydra_dir / "config.yaml"

    OmegaConf.update(cfg, "online_rollout.render", False, merge=False, force_add=True)
    OmegaConf.update(cfg, "online_rollout.render_images.enabled", False, merge=False, force_add=True)
    OmegaConf.update(
        cfg,
        "online_rollout.save_results_npz",
        str(results_path),
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "online_rollout.save_vis_trace_npz",
        str(trace_path),
        merge=False,
        force_add=True,
    )
    mode_value = OmegaConf.select(cfg, "online_rollout.vis_trace", default=None)
    if rollout_config is not None and isinstance(mode_value, DictConfig):
        OmegaConf.update(
            cfg,
            "online_rollout.vis_trace.resolved_rollout_config",
            _normalized_config_name(rollout_config),
            merge=False,
            force_add=True,
        )

    hydra_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=config_snapshot_path, resolve=True)
    return VisTraceCapturePaths(
        task=task,
        embodiment=embodiment,
        rollout_config=(
            None if rollout_config is None else _normalized_config_name(rollout_config)
        ),
        source_indices=indices,
        stem=stem,
        results_path=results_path,
        trace_path=trace_path,
        hydra_dir=hydra_dir,
        config_snapshot_path=config_snapshot_path,
    )
