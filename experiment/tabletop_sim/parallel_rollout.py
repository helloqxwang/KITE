"""Subprocess sharding helpers for tabletop rollout scripts."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


def configure_torch_threads(cfg: DictConfig, *, section: str) -> None:
    value = OmegaConf.select(cfg, f"{section}.parallel.torch_num_threads", default=None)
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return
    threads = max(1, int(value))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    try:
        import torch

        torch.set_num_threads(threads)
    except Exception:
        return


def parallel_process_count(cfg: DictConfig, *, section: str) -> int:
    num_workers = int(OmegaConf.select(cfg, f"{section}.parallel.num_workers", default=1))
    envs_per_worker = int(OmegaConf.select(cfg, f"{section}.parallel.envs_per_worker", default=1))
    return max(1, num_workers) * max(1, envs_per_worker)


def validate_headless_parallel(
    cfg: DictConfig,
    *,
    section: str,
    render: bool,
    save_vis_trace_npz: str | Path | None,
) -> int:
    process_count = parallel_process_count(cfg, section=section)
    if process_count <= 1:
        return process_count
    if bool(render):
        raise ValueError(f"{section}.render=true is not supported with parallel.num_workers/envs_per_worker > 1")
    if save_vis_trace_npz is not None:
        raise ValueError(f"{section}.save_vis_trace_npz is not supported with parallel.num_workers/envs_per_worker > 1")
    return process_count


def launch_sharded_workers(
    *,
    script_path: Path,
    cfg: DictConfig,
    section: str,
    selected: list[int],
    process_count: int,
    tmpdir: Path,
    result_prefix: str,
    seed_offset_key: str | None = None,
    extra_worker_outputs: dict[str, tuple[str, str]] | None = None,
) -> list[Path]:
    shards = _contiguous_shards(selected, min(int(process_count), max(1, len(selected))))
    result_paths: list[Path] = []
    processes: list[tuple[int, subprocess.Popen[bytes]]] = []
    devices = _devices_from_cfg(cfg, section=section)
    threads = OmegaConf.select(cfg, f"{section}.parallel.torch_num_threads", default=None)

    for worker_idx, (start_pos, shard) in enumerate(shards):
        if not shard:
            continue
        worker_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.update(worker_cfg, f"{section}.parallel.num_workers", 1, force_add=True)
        OmegaConf.update(worker_cfg, f"{section}.parallel.envs_per_worker", 1, force_add=True)
        OmegaConf.update(worker_cfg, f"{section}.indices", _indices_string(shard), force_add=True)
        OmegaConf.update(worker_cfg, f"{section}.start_index", 0, force_add=True)
        OmegaConf.update(worker_cfg, f"{section}.count", None, force_add=True)
        OmegaConf.update(worker_cfg, f"{section}.render", False, force_add=True)
        OmegaConf.update(worker_cfg, f"{section}.save_vis_trace_npz", None, force_add=True)
        if seed_offset_key is not None:
            OmegaConf.update(worker_cfg, seed_offset_key, int(start_pos), force_add=True)
        if devices:
            OmegaConf.update(worker_cfg, f"{section}.device", _device_for_worker(devices, worker_idx), force_add=True)
        result_path = tmpdir / f"{result_prefix}_worker_{worker_idx}.npz"
        OmegaConf.update(worker_cfg, f"{section}.save_results_npz", str(result_path), force_add=True)
        for cfg_key, (path_prefix, suffix) in dict(extra_worker_outputs or {}).items():
            out_path = tmpdir / f"{path_prefix}_worker_{worker_idx}{suffix}"
            OmegaConf.update(worker_cfg, str(cfg_key), str(out_path), force_add=True)
        OmegaConf.update(worker_cfg, "hydra.run.dir", str(tmpdir / f"hydra_worker_{worker_idx}"), force_add=True)
        config_path = tmpdir / f"{result_prefix}_worker_{worker_idx}.yaml"
        OmegaConf.save(config=worker_cfg, f=str(config_path))
        result_paths.append(result_path)

        env = os.environ.copy()
        if threads is not None and str(threads).strip().lower() not in {"", "none", "null"}:
            env["OMP_NUM_THREADS"] = str(max(1, int(threads)))
            env["MKL_NUM_THREADS"] = str(max(1, int(threads)))
        cmd = [
            sys.executable,
            str(script_path),
            "--config-path",
            str(tmpdir),
            "--config-name",
            config_path.stem,
        ]
        processes.append((worker_idx, subprocess.Popen(cmd, env=env)))

    failures: list[tuple[int, int]] = []
    for worker_idx, process in processes:
        return_code = int(process.wait())
        if return_code != 0:
            failures.append((worker_idx, return_code))
    if failures:
        detail = ", ".join(f"worker={idx} returncode={code}" for idx, code in failures)
        raise RuntimeError(f"Parallel rollout workers failed: {detail}")
    missing = [str(path) for path in result_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Parallel rollout worker result files missing: {missing}")
    return result_paths


def _contiguous_shards(values: list[int], process_count: int) -> list[tuple[int, list[int]]]:
    total = len(values)
    count = max(1, int(process_count))
    shards: list[tuple[int, list[int]]] = []
    for worker_idx in range(count):
        start = worker_idx * total // count
        end = (worker_idx + 1) * total // count
        shards.append((start, [int(x) for x in values[start:end]]))
    return shards


def _indices_string(values: list[int]) -> str:
    return ",".join(str(int(x)) for x in values)


def _devices_from_cfg(cfg: DictConfig, *, section: str) -> list[Any]:
    value = OmegaConf.select(cfg, f"{section}.parallel.devices", default=None)
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _device_for_worker(devices: list[Any], worker_idx: int) -> str:
    value = devices[int(worker_idx) % len(devices)]
    text = str(value).strip()
    if text.isdigit():
        return f"cuda:{text}"
    return text
