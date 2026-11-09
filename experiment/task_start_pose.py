"""Task-aware start-pose path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment.robotiq_stick_start_pose import DEFAULT_ROBOTIQ_STICK_START_POSE_RELATIVE
from experiment.robot_model_start_pose import (
    DEFAULT_ALLEGRO_RIGHT_START_POSE_RELATIVE,
    DEFAULT_BARRETT_START_POSE_RELATIVE,
    LEGACY_ALLEGRO_ORIGINAL_RIGHT_START_POSE_RELATIVE,
)
from experiment.wuji_start_pose import DEFAULT_WUJI_START_POSE_RELATIVE


EXPERIMENT_DIR = Path(__file__).resolve().parent
CROSS_DIR = EXPERIMENT_DIR.parent
START_POSE_DIR_RELATIVE = "./experiment/data/high_level_policy"

TASK_PUMPBOTTLE_USE = "pumpbottle_use"
TASK_KEYBOARD_PRESS = "keyboard_press"
TASK_CUBE_PICK = "cube_pick"
TASK_START_POSE_CHOICES = (TASK_PUMPBOTTLE_USE, TASK_KEYBOARD_PRESS, TASK_CUBE_PICK)

_TASK_ALIASES = {
    "object_suite_pick_place": TASK_CUBE_PICK,
    "pump": TASK_PUMPBOTTLE_USE,
    "pump_bottle": TASK_PUMPBOTTLE_USE,
    "pumpbottle": TASK_PUMPBOTTLE_USE,
    "pump_bottle_use": TASK_PUMPBOTTLE_USE,
    "pumpbottle_use": TASK_PUMPBOTTLE_USE,
    "keyboard": TASK_KEYBOARD_PRESS,
    "keyboard_press": TASK_KEYBOARD_PRESS,
    "mini_keyboard": TASK_KEYBOARD_PRESS,
    "cube": TASK_CUBE_PICK,
    "cube_pick": TASK_CUBE_PICK,
    "cubepick": TASK_CUBE_PICK,
}

_EMBODIMENT_ALIASES = {
    "wuji": "wuji",
    "wujihand": "wuji",
    "wujihand_right": "wuji",
    "marvin_wuji": "wuji",
    "barrett": "barrett",
    "barrett_right": "barrett",
    "allegro": "allegro_right",
    "allegro_right": "allegro_right",
    "allegro_original_right": "allegro_right",
    "robotiq": "robotiq_gripper_stick",
    "robotiq_stick": "robotiq_gripper_stick",
    "robotiq_gripper_stick": "robotiq_gripper_stick",
    "robotiq_gripper_stick_armless": "robotiq_gripper_stick",
}

_TASK_START_POSE_STEMS = {
    "wuji": "wuji",
    "barrett": "barrett",
    "allegro_right": "allegro_right",
    "robotiq_gripper_stick": "robotiq_gripper_stick",
}

_OLD_START_POSE_RELATIVES = {
    "wuji": (DEFAULT_WUJI_START_POSE_RELATIVE,),
    "barrett": (DEFAULT_BARRETT_START_POSE_RELATIVE,),
    "allegro_right": (
        DEFAULT_ALLEGRO_RIGHT_START_POSE_RELATIVE,
        LEGACY_ALLEGRO_ORIGINAL_RIGHT_START_POSE_RELATIVE,
    ),
    "robotiq_gripper_stick": (DEFAULT_ROBOTIQ_STICK_START_POSE_RELATIVE,),
}


@dataclass(frozen=True)
class StartPoseResolution:
    task: str | None
    embodiment_key: str
    path: Path
    save_path: Path
    fallback_path: Path
    fallback_candidates: tuple[Path, ...]
    task_specific_path: Path | None
    used_task_specific: bool
    explicit: bool
    prefer_task_specific: bool
    fallback_reason: str


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (CROSS_DIR / path).resolve()


def normalize_start_pose_task(task: Any, *, default: str | None = None) -> str | None:
    if task is None:
        return normalize_start_pose_task(default) if default is not None else None
    text = str(task).strip().lower()
    if text in {"", "none", "null", "auto"}:
        return normalize_start_pose_task(default) if default is not None else None
    key = text.replace("-", "_")
    if key not in _TASK_ALIASES:
        raise ValueError(
            f"Unknown start-pose task {task!r}; expected one of {', '.join(TASK_START_POSE_CHOICES)}"
        )
    return _TASK_ALIASES[key]


def normalize_start_pose_embodiment(embodiment: Any) -> str:
    text = str(embodiment).strip().lower().replace("-", "_")
    if text not in _EMBODIMENT_ALIASES:
        raise ValueError(
            f"Unknown start-pose embodiment {embodiment!r}; expected one of "
            f"{', '.join(sorted(_EMBODIMENT_ALIASES))}"
        )
    return _EMBODIMENT_ALIASES[text]


def old_start_pose_paths(embodiment: Any) -> tuple[Path, ...]:
    key = normalize_start_pose_embodiment(embodiment)
    return tuple(resolve_project_path(value) for value in _OLD_START_POSE_RELATIVES[key])


def old_start_pose_path(embodiment: Any) -> Path:
    paths = old_start_pose_paths(embodiment)
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def task_start_pose_path(embodiment: Any, task: Any) -> Path:
    key = normalize_start_pose_embodiment(embodiment)
    task_key = normalize_start_pose_task(task)
    if task_key is None:
        return old_start_pose_path(key)
    stem = _TASK_START_POSE_STEMS[key]
    return resolve_project_path(f"{START_POSE_DIR_RELATIVE}/{stem}_{task_key}_start_pose.pt")


def resolve_start_pose_path(
    embodiment: Any,
    task: Any = None,
    *,
    configured_pose_pt: str | Path | None = None,
    explicit_pose_pt: bool = False,
    prefer_task_specific: bool = True,
    default_task: str | None = None,
) -> StartPoseResolution:
    key = normalize_start_pose_embodiment(embodiment)
    task_key = normalize_start_pose_task(task, default=default_task)
    fallback_candidates = _fallback_candidates(key, configured_pose_pt=configured_pose_pt)
    fallback_path = _first_existing_or_first(fallback_candidates)

    if explicit_pose_pt and configured_pose_pt is not None:
        explicit_path = resolve_project_path(configured_pose_pt)
        return StartPoseResolution(
            task=task_key,
            embodiment_key=key,
            path=explicit_path,
            save_path=explicit_path,
            fallback_path=fallback_path,
            fallback_candidates=fallback_candidates,
            task_specific_path=None if task_key is None else task_start_pose_path(key, task_key),
            used_task_specific=False,
            explicit=True,
            prefer_task_specific=bool(prefer_task_specific),
            fallback_reason="explicit_pose_pt",
        )

    task_path = None if task_key is None else task_start_pose_path(key, task_key)
    use_task_specific = bool(prefer_task_specific) and task_key not in {None, TASK_CUBE_PICK}
    if use_task_specific and task_path is not None:
        if task_path.exists():
            return StartPoseResolution(
                task=task_key,
                embodiment_key=key,
                path=task_path,
                save_path=task_path,
                fallback_path=fallback_path,
                fallback_candidates=fallback_candidates,
                task_specific_path=task_path,
                used_task_specific=True,
                explicit=False,
                prefer_task_specific=True,
                fallback_reason="task_specific_exists",
            )
        return StartPoseResolution(
            task=task_key,
            embodiment_key=key,
            path=fallback_path,
            save_path=task_path,
            fallback_path=fallback_path,
            fallback_candidates=fallback_candidates,
            task_specific_path=task_path,
            used_task_specific=False,
            explicit=False,
            prefer_task_specific=True,
            fallback_reason="task_specific_missing",
        )

    return StartPoseResolution(
        task=task_key,
        embodiment_key=key,
        path=fallback_path,
        save_path=fallback_path,
        fallback_path=fallback_path,
        fallback_candidates=fallback_candidates,
        task_specific_path=task_path,
        used_task_specific=False,
        explicit=False,
        prefer_task_specific=bool(prefer_task_specific),
        fallback_reason="cube_pick_uses_legacy" if task_key == TASK_CUBE_PICK else "task_specific_disabled",
    )


def describe_start_pose_resolution(resolution: StartPoseResolution) -> str:
    task = "none" if resolution.task is None else resolution.task
    mode = "task-specific" if resolution.used_task_specific else "fallback"
    if resolution.explicit:
        mode = "explicit"
    return (
        f"task={task} embodiment={resolution.embodiment_key} mode={mode} "
        f"path={resolution.path} save_path={resolution.save_path}"
    )


def _fallback_candidates(
    embodiment_key: str,
    *,
    configured_pose_pt: str | Path | None,
) -> tuple[Path, ...]:
    out: list[Path] = []
    if configured_pose_pt is not None:
        out.append(resolve_project_path(configured_pose_pt))
    for path in old_start_pose_paths(embodiment_key):
        if path not in out:
            out.append(path)
    return tuple(out)


def _first_existing_or_first(paths: tuple[Path, ...]) -> Path:
    if not paths:
        raise ValueError("At least one fallback start-pose path is required")
    for path in paths:
        if path.exists():
            return path
    return paths[0]
