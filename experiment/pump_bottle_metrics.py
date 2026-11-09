"""Temporal metrics for pump-bottle manipulation rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PumpPressOrderMetrics:
    """Ordering of the first full pump press and the first visible bottle lift."""

    first_button_press_frame: int
    first_lift_frame: int
    lift_clearance_at_first_press: float
    press_after_lift: bool


def pump_press_order_metrics(
    button_slide: object,
    lift_clearance_delta: object,
    *,
    button_success_slide: float,
    lift_success_delta: float,
) -> PumpPressOrderMetrics:
    """Measure whether the first successful pump press occurs after lifting.

    ``press_after_lift`` requires the complete bottle geometry to have cleared its
    reset support height by ``lift_success_delta`` at the first frame where the
    button crosses ``button_success_slide``. Crossing both thresholds in the same
    frame is accepted as pressing during the lift.
    """

    button = np.asarray(button_slide, dtype=np.float64).reshape(-1)
    lift = np.asarray(lift_clearance_delta, dtype=np.float64).reshape(-1)
    if button.shape != lift.shape:
        raise ValueError(
            "button_slide and lift_clearance_delta must have the same shape, "
            f"got {button.shape} and {lift.shape}"
        )

    press_frames = np.flatnonzero(
        np.isfinite(button) & (button <= float(button_success_slide))
    )
    lift_frames = np.flatnonzero(
        np.isfinite(lift) & (lift >= float(lift_success_delta))
    )
    first_press = int(press_frames[0]) if press_frames.size else -1
    first_lift = int(lift_frames[0]) if lift_frames.size else -1
    clearance_at_press = float(lift[first_press]) if first_press >= 0 else float("nan")
    press_after_lift = bool(
        first_press >= 0
        and np.isfinite(clearance_at_press)
        and clearance_at_press >= float(lift_success_delta)
    )
    return PumpPressOrderMetrics(
        first_button_press_frame=first_press,
        first_lift_frame=first_lift,
        lift_clearance_at_first_press=clearance_at_press,
        press_after_lift=press_after_lift,
    )
