"""Shared task-success handling for tabletop policy rollouts."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def task_success_from_observation(
    task: Any,
    observation: Mapping[str, Any],
    *,
    max_object_z: float,
) -> bool:
    """Apply the rollout success rule used by GT-contact and high-level eval.

    Placement-style tasks are judged exclusively by their task observation.
    Legacy lift tasks additionally retain the maximum-height fallback used by
    the original cube-pick rollouts.
    """

    success = bool(np.asarray(observation["success"]).reshape(()).item())
    lift_success_z = getattr(task, "lift_success_z", None)
    if lift_success_z is not None:
        success = bool(success or float(max_object_z) >= float(lift_success_z))
    return success


def accumulate_task_success(
    previous_success: bool,
    task: Any,
    observation: Mapping[str, Any],
    *,
    max_object_z: float,
) -> bool:
    """Keep success latched after it is first observed in an episode."""

    return bool(
        previous_success
        or task_success_from_observation(task, observation, max_object_z=max_object_z)
    )
