"""Portable curated lifting motions and initial contact-region alignment."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from visualization.decoder_runtime import interpolate_configuration, translate_configuration


LIBRARY_PATH = Path(__file__).with_name("data") / "lifting_trajectories.json"


def load_library(path=LIBRARY_PATH):
    library = json.loads(Path(path).read_text())
    if library.get("version") != 1:
        raise ValueError("Unsupported lifting trajectory library version")
    return library


def validate_record(model, record):
    if list(model.joint_orders) != record["joint_orders"]:
        raise ValueError("Curated trajectory joint layout does not match this decoder's geometry")
    for key in ("q_start", "q_end"):
        q = np.asarray(record[key])
        if q.shape != (model.dof,) or not np.isfinite(q).all():
            raise ValueError(f"Invalid curated {key}")
    indices = np.asarray(record["contact_indices"], dtype=int)
    if len(indices) != 3 or len(set(indices.tolist())) != 3:
        raise ValueError("Curated lifting motions need three distinct surface contacts")
    if np.any(indices < 0) or np.any(indices >= len(model.surface_template_points_local)):
        raise ValueError("Curated contact indices do not match this surface template")
    actual = model.surface_template_points_local[torch.as_tensor(indices, device=model.device)].cpu().numpy()
    if not np.allclose(actual, record["contact_local_points"], atol=2e-6, rtol=0):
        raise ValueError("Curated contact attachments do not match this surface template")


def attached_intents(model, configurations, contact_indices):
    """Track the same surface samples, including their normals, at every step."""
    with torch.inference_mode():
        points, normals = model.get_surface_points_normals_batch(configurations.to(model.device))
        return torch.cat([points[:, contact_indices], normals[:, contact_indices]], -1).cpu()


@dataclass
class SourceTrajectory:
    name: str
    configurations: torch.Tensor
    intents: torch.Tensor
    contact_indices: list[int]
    contact_labels: list[str]


def build_source_trajectory(model, name, record, *, steps=60, lift_height=.12, finger_motion=1., hold_steps=8):
    if steps < 2 or hold_steps < 0 or hold_steps >= steps-1:
        raise ValueError("Use at least two motion steps after the initial hold")
    if not 0 <= lift_height <= .3 or not 0 <= finger_motion <= 1:
        raise ValueError("Lift height must be 0–0.3 m and finger motion 0–1")
    validate_record(model, record)
    start, end = (torch.tensor(record[key], dtype=torch.float32) for key in ["q_start", "q_end"])
    configurations = []
    for frame in range(steps):
        t = max(0., (frame-hold_steps)/(steps-1-hold_steps))
        s = t*t*(3.-2.*t)
        q = interpolate_configuration(model, start, end, s*finger_motion)
        configurations.append(translate_configuration(model, q, [0., 0., s*lift_height]))
    configurations = torch.stack(configurations)
    indices = record["contact_indices"]
    return SourceTrajectory(name, configurations, attached_intents(model, configurations, indices), indices, record["contact_labels"])
