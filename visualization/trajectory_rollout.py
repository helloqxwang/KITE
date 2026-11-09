"""Autoregressive kinematic trajectory following, independently of the UI."""
from dataclasses import dataclass

import numpy as np
import torch

from visualization.decoder_runtime import contact_errors, nearest_surface


@dataclass
class TrajectoryRollout:
    initial: torch.Tensor
    states: torch.Tensor
    raw_predictions: torch.Tensor
    position_rms_mm: np.ndarray
    normal_error_deg: np.ndarray
    clipped_coordinates: int


def roll_out(runtime, target, source, initial, *, calls_per_step=1, clamp_limits=True, progress=None):
    """Feed the previous prediction forward; encode each source frame once.

    states[t, k] is the result of call k at source frame t. The final state
    becomes the next frame's input. There is no target reference trajectory.
    """
    if not 1 <= calls_per_step <= 10:
        raise ValueError("calls_per_step must be between 1 and 10")
    model = runtime.get(target).model
    q = torch.as_tensor(initial, dtype=torch.float32).clone().cpu()
    if q.shape != (model.dof,) or not torch.isfinite(q).all():
        raise ValueError("Invalid target initial configuration")
    lower, upper = (torch.as_tensor(x, dtype=torch.float32) for x in model.pk_chain.get_joint_limits())
    if clamp_limits:
        q = q.clamp(lower, upper)
    start = q.clone()
    states, raw, errors, clipped = [], [], [], 0
    with runtime.lock:
        for frame, intent in enumerate(source.intents):
            latent = runtime.encode(source.name, intent)
            step_states, step_raw = [], []
            for _ in range(calls_per_step):
                predicted = runtime.decode(target, latent, q)
                if not torch.isfinite(predicted).all():
                    raise ValueError(f"Non-finite target prediction at frame {frame}")
                q = predicted.clamp(lower, upper) if clamp_limits else predicted.clone()
                clipped += int((q != predicted).sum())
                step_raw.append(predicted.clone()); step_states.append(q.clone())
            states.append(torch.stack(step_states)); raw.append(torch.stack(step_raw))
            points, normals, _ = nearest_surface(model, q, intent)
            errors.append(contact_errors(intent, points, normals))
            if progress is not None:
                progress(frame+1, len(source.intents))
    errors = np.asarray(errors)
    return TrajectoryRollout(start, torch.stack(states), torch.stack(raw), errors[:, 0], errors[:, 1], clipped)
