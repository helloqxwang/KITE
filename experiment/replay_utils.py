"""Replay saved tabletop cube-pick action trajectories with Hydra configuration."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.high_level_policy.payload import (  # noqa: E402
    load_high_level_payload,
    single_manipulator_payload_to_replay_source,
)
from experiment.tabletop_sim.env import TabletopManipulationEnv


def _load_training_pt(path: Path) -> dict[str, np.ndarray]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Replay source does not exist: {path}")
    if path.suffix != ".pt":
        raise ValueError(
            f"Replay sources are single-manipulator .pt payloads only; got {path}"
        )
    payload = load_high_level_payload(path)
    return single_manipulator_payload_to_replay_source(payload, source_path=path)


def _trajectory_count(source: dict[str, np.ndarray]) -> int:
    for key in ("action_ee_pos", "object_pose", "cube_pose", "executed_qpos"):
        if key not in source:
            continue
        arr = np.asarray(source[key])
        if arr.dtype == object and arr.ndim == 1:
            return int(arr.shape[0])
        return int(arr.shape[0]) if arr.ndim >= 3 else 1
    raise KeyError(
        "Replay source must contain at least one of: action_ee_pos, cube_pose, executed_qpos"
    )


def _validate_source_compatible(source: dict[str, np.ndarray], cfg: DictConfig) -> None:
    object_suite = (
        str(_source_scalar(source, "task") or "") == "object_suite_pick_place"
    )
    if not object_suite:
        _sync_cube_height_ratio_from_source(source, cfg)
    actual_embodiment = _source_scalar(source, "embodiment")
    if actual_embodiment is not None and str(actual_embodiment) != str(
        cfg.env.embodiment
    ):
        raise ValueError(
            f"Replay source embodiment={actual_embodiment!r} does not match "
            f"config embodiment={str(cfg.env.embodiment)!r}"
        )

    actual_mode = _source_scalar(source, "mode")
    if actual_mode is not None and not _source_mode_matches_config(
        actual_mode, str(cfg.env.mode)
    ):
        raise ValueError(
            f"Replay source mode={actual_mode!r} does not match config mode={str(cfg.env.mode)!r}"
        )

    frame_skip = _source_scalar(source, "frame_skip")
    if frame_skip is not None and int(frame_skip) != int(cfg.env.frame_skip):
        raise ValueError(
            f"Replay source frame_skip={int(frame_skip)} does not match config env.frame_skip={int(cfg.env.frame_skip)}"
        )
    if object_suite:
        return

    expected_arrays = {
        "cube_x_range": np.asarray(cfg.cube.x_range, dtype=np.float32),
        "cube_y_range": np.asarray(cfg.cube.y_range, dtype=np.float32),
        "cube_side_length_range": np.asarray(
            cfg.cube.side_length_range, dtype=np.float32
        ),
        "cube_height_range": np.asarray(cfg.cube.side_length_range, dtype=np.float32)
        * np.float32(OmegaConf.select(cfg, "cube.height_ratio", default=1.0)),
        "cube_yaw_range_deg": np.asarray(cfg.cube.yaw_range_deg, dtype=np.float32),
    }
    for key, expected in expected_arrays.items():
        if key not in source:
            continue
        actual = np.asarray(source[key], dtype=np.float32)
        if actual.shape == expected.shape and not np.allclose(
            actual, expected, atol=1e-6
        ):
            raise ValueError(
                f"Replay source {key}={actual.tolist()} does not match config {key}={expected.tolist()}"
            )


def _validate_object_suite_source_object(
    source: dict[str, np.ndarray],
    cfg: DictConfig,
    trajectory_index: int,
) -> None:
    if (
        str(OmegaConf.select(cfg, "task.name", default="cube_pick"))
        != "object_suite_pick_place"
    ):
        return
    actual = _trajectory_value(source, "object_id", trajectory_index)
    expected = OmegaConf.select(cfg, "objects.active_id", default=None)
    if actual is not None and expected is not None and str(actual) != str(expected):
        raise ValueError(
            f"Replay trajectory {trajectory_index} uses object_id={str(actual)!r}, but "
            f"objects.active_id={str(expected)!r}. Select the matching object config."
        )


def _sync_cube_height_ratio_from_source(
    source: dict[str, np.ndarray], cfg: DictConfig
) -> float | None:
    ratio = _source_cube_height_ratio(source, cfg)
    if ratio is None:
        return None
    current = float(OmegaConf.select(cfg, "cube.height_ratio", default=1.0))
    if not np.isclose(current, ratio, atol=1e-6):
        OmegaConf.update(cfg, "cube.height_ratio", float(ratio), merge=True)
    return float(ratio)


def _source_cube_height_ratio(
    source: dict[str, np.ndarray], cfg: DictConfig
) -> float | None:
    source_ratio = _source_scalar(source, "cube_height_ratio")
    if source_ratio is not None:
        ratio = float(source_ratio)
        return ratio if np.isfinite(ratio) and ratio > 0.0 else None

    if "cube_height_range" not in source:
        return None
    cfg_side = np.asarray(cfg.cube.side_length_range, dtype=np.float32).reshape(-1)
    source_side = np.asarray(
        source.get("cube_side_length_range", cfg_side), dtype=np.float32
    ).reshape(-1)
    source_height = np.asarray(source["cube_height_range"], dtype=np.float32).reshape(
        -1
    )
    if cfg_side.shape != source_side.shape or source_height.shape != cfg_side.shape:
        return None
    if not np.allclose(source_side, cfg_side, atol=1e-6):
        return None
    if np.any(np.abs(cfg_side) <= 1e-8):
        return None
    ratios = source_height / cfg_side
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0.0):
        return None
    ratio = float(np.mean(ratios))
    if not np.allclose(ratios, ratio, atol=1e-5):
        return None
    return ratio


def _set_cube_state(
    env: TabletopManipulationEnv,
    pose: np.ndarray,
    *,
    side_length: object | None,
    height: object | None = None,
) -> None:
    if side_length is not None and hasattr(env.task, "_set_cube_size"):
        env.task._set_cube_size(
            env.model,
            env.scene,
            side_length=float(np.asarray(side_length).item()),
            height=None if height is None else float(np.asarray(height).item()),
        )
    if hasattr(env.task, "_object_qpos_qvel_adr"):
        qadr, dadr = env.task._object_qpos_qvel_adr(env.model, env.scene)
    else:
        qadr, dadr = env.task._cube_qpos_qvel_adr(env.model, env.scene)
    env.data.qpos[qadr : qadr + 7] = np.asarray(pose, dtype=np.float64).reshape(7)
    env.data.qvel[dadr : dadr + 6] = 0.0
    env.backend.forward()


def _source_mode_matches_config(source_mode: object, config_mode: str) -> bool:
    source_text = str(source_mode)
    config_text = str(config_mode)
    if source_text == config_text:
        return True
    return source_text == "floating_right" and config_text in {
        "single_right",
        "floating",
    }


def _trajectory_array(
    source: dict[str, np.ndarray],
    key: str,
    trajectory_index: int,
    *,
    required: bool = True,
) -> np.ndarray | None:
    if key not in source:
        if required:
            raise KeyError(f"Replay source is missing required key {key!r}")
        return None
    arr = np.asarray(source[key])
    if arr.dtype == object and arr.ndim == 1:
        if int(trajectory_index) >= int(arr.shape[0]):
            raise IndexError(
                f"{key} has {arr.shape[0]} entries; index {trajectory_index} is out of range"
            )
        return np.asarray(arr[int(trajectory_index)])
    if arr.ndim >= 3:
        return np.asarray(arr[int(trajectory_index)])
    if int(trajectory_index) != 0:
        raise IndexError(
            f"{key} is unbatched, so only rollout.trajectory_index=0 is valid"
        )
    return arr


def _trajectory_value(
    source: dict[str, np.ndarray], key: str, trajectory_index: int
) -> object | None:
    if key not in source:
        return None
    arr = np.asarray(source[key])
    if arr.ndim == 0:
        return arr.item()
    if arr.shape[0] == 0:
        return None
    if arr.dtype == object and arr.ndim == 1:
        if int(trajectory_index) >= int(arr.shape[0]):
            raise IndexError(
                f"{key} has {arr.shape[0]} entries; index {trajectory_index} is out of range"
            )
        value = arr[int(trajectory_index)]
        return value.item() if np.asarray(value).ndim == 0 else value
    if arr.ndim >= 1:
        if int(trajectory_index) >= int(arr.shape[0]):
            raise IndexError(
                f"{key} has {arr.shape[0]} entries; index {trajectory_index} is out of range"
            )
        value = arr[int(trajectory_index)]
        return value.item() if np.asarray(value).ndim == 0 else value
    return None


def _source_scalar(source: dict[str, np.ndarray], key: str) -> object | None:
    if key not in source:
        return None
    arr = np.asarray(source[key])
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return None


class ViewerReplayControl:
    KEY_SPACE = 32
    KEY_ESCAPE = 256
    KEY_ENTER = 257
    KEY_N = ord("N")
    KEY_Q = ord("Q")
    KEY_RIGHT = 262

    def __init__(self, *, start_paused: bool = False) -> None:
        self.paused = bool(start_paused)
        self.step_once = False
        self.stop_requested = False

    def key_callback(self, key: int) -> None:
        if int(key) == self.KEY_SPACE:
            self.paused = not self.paused
            self.step_once = False
        elif int(key) in {self.KEY_N, self.KEY_RIGHT, self.KEY_ENTER}:
            self.paused = True
            self.step_once = True
        elif int(key) in {self.KEY_Q, self.KEY_ESCAPE}:
            self.stop_requested = True

    def wait_before_step(
        self,
        env: TabletopManipulationEnv,
        *,
        step_idx: int,
        total_steps: int,
        trajectory_index: int,
    ) -> bool:
        while self.paused and not self.step_once and not self.stop_requested:
            self.update_overlay(
                env,
                step_idx=step_idx,
                total_steps=total_steps,
                trajectory_index=trajectory_index,
            )
            if not env.viewer_is_running():
                return False
            env.render()
            time.sleep(1.0 / 60.0)
        should_step = not self.stop_requested and env.viewer_is_running()
        self.step_once = False
        return should_step

    def update_overlay(
        self,
        env: TabletopManipulationEnv,
        *,
        step_idx: int,
        total_steps: int,
        trajectory_index: int,
    ) -> None:
        state = "paused" if self.paused else "playing"
        title = "Space: play/pause | N/Enter: step | Q/Esc: quit"
        detail = f"{state} trajectory={int(trajectory_index)} step={int(step_idx)}/{int(total_steps)}"
        env.set_viewer_text(title, detail)

    def hold_open(
        self, env: TabletopManipulationEnv, *, completed_steps: int, total_steps: int
    ) -> None:
        self.paused = True
        while not self.stop_requested and env.viewer_is_running():
            env.set_viewer_text(
                "Rollout complete | Q/Esc or close window",
                f"completed {int(completed_steps)}/{int(total_steps)} steps",
            )
            env.render()
            time.sleep(1.0 / 30.0)
