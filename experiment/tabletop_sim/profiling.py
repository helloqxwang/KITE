"""Small profiling helpers for tabletop rollout scripts."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import time
from typing import Iterator


class RolloutProfiler:
    def __init__(self, *, enabled: bool = False, device=None, label: str = "profile") -> None:
        self.enabled = bool(enabled)
        self.device = device
        self.label = str(label)
        self.times: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync_cuda()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync_cuda()
            self.times[str(name)] += time.perf_counter() - start
            self.counts[str(name)] += 1

    def print_summary(self) -> None:
        if not self.enabled:
            return
        total = sum(self.times.values())
        print(f"{self.label}: total_profiled_s={total:.3f}", flush=True)
        for name, elapsed in sorted(self.times.items(), key=lambda item: item[1], reverse=True):
            count = max(1, int(self.counts.get(name, 0)))
            print(
                f"{self.label}: {name} count={count} total_s={elapsed:.3f} "
                f"mean_ms={1000.0 * elapsed / count:.2f}",
                flush=True,
            )

    def _sync_cuda(self) -> None:
        device = self.device
        if device is None or getattr(device, "type", None) != "cuda":
            return
        try:
            import torch

            torch.cuda.synchronize(device)
        except Exception:
            return
