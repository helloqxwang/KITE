"""Check local MANO code, weights, anatomical sampling, and CPU geometry."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    print(f"Python: {sys.executable}", flush=True)
    print(f"KITE: {ROOT}", flush=True)

    import torch
    from experiment.manipulation.mano_utils import (
        _import_manopth_mano_layer, resolve_mano_model_root,
    )
    from robot_model.robot_model import create_robot_model

    source = Path(inspect.getfile(_import_manopth_mano_layer())).resolve()
    if not source.is_relative_to(ROOT / "third_party" / "manopth"):
        raise RuntimeError(f"Expected KITE's bundled manopth, got {source}")
    print(f"manopth: {source}", flush=True)
    model_root = resolve_mano_model_root()
    for side in ("LEFT", "RIGHT"):
        weights = (model_root / f"MANO_{side}.pkl").resolve()
        if not weights.is_relative_to(ROOT):
            raise RuntimeError(f"Copy MANO weights into assets/mano_models; external link: {weights}")
        print(f"Weights: {weights}", flush=True)

    torch.set_num_threads(1)
    for side in ("left", "right"):
        model = create_robot_model(f"mano_{side}", device="cpu", num_points=778)
        with torch.inference_mode():
            q = model.sample_random_q(batch_size=2, generator=torch.Generator().manual_seed(7))
            points, normals = model.get_surface_points_normals_batch(q)
        if points.shape != (2, 778, 3) or normals.shape != points.shape:
            raise RuntimeError(f"Unexpected {side} MANO geometry: {points.shape}, {normals.shape}")
        if not all(torch.isfinite(value).all() for value in (q, points, normals)):
            raise RuntimeError(f"Non-finite {side} MANO geometry")
        print(f"MANO {side}: anatomical sampling and CPU geometry OK", flush=True)


if __name__ == "__main__":
    main()
