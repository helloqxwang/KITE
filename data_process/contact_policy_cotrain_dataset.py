from __future__ import annotations

from typing import Any, Dict, Iterator, List

from torch.utils.data import DataLoader

from data_process.contact_policy_probe_dataset import ContactPolicyProbeDataset


class ContactPolicyCoTrainBundle:
    """Registry of per-manipulator ``ContactPolicyProbeDataset`` instances for co-training.

    The co-train trainer instantiates one probe dataset per manipulator (each with its own
    buffer) and wraps them in this bundle. The bundle hands back per-manipulator DataLoaders
    that the training loop iterates in lockstep to form stratified batches — one sub-batch
    per manipulator per optimizer step.
    """

    def __init__(self, *, datasets: Dict[str, ContactPolicyProbeDataset]) -> None:
        if not datasets:
            raise ValueError("At least one dataset is required.")
        self.datasets: Dict[str, ContactPolicyProbeDataset] = dict(datasets)
        self.manipulator_names: List[str] = list(self.datasets.keys())

    def __len__(self) -> int:
        # Number of samples in the smallest manipulator buffer; callers that need steps per
        # epoch should use ``steps_per_epoch`` below.
        return min(len(ds) for ds in self.datasets.values())

    def steps_per_epoch(self, *, batch_size: int, drop_last: bool) -> int:
        """Return the number of lockstep batches per epoch for the given sub-batch size.

        Matches ``iter_lockstep``'s stopping rule: the shortest loader determines the count.
        """
        batch_size = int(max(1, batch_size))
        counts: List[int] = []
        for dataset in self.datasets.values():
            n = int(len(dataset))
            if bool(drop_last):
                counts.append(n // batch_size)
            else:
                counts.append((n + batch_size - 1) // batch_size)
        return min(counts) if counts else 0

    def build_loaders(
        self,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        loader_kwargs: Dict[str, Any],
    ) -> Dict[str, DataLoader]:
        loaders: Dict[str, DataLoader] = {}
        for name, dataset in self.datasets.items():
            loaders[name] = DataLoader(
                dataset,
                batch_size=int(batch_size),
                shuffle=bool(shuffle),
                drop_last=bool(drop_last),
                **loader_kwargs,
            )
        return loaders


def iter_lockstep(
    loaders: Dict[str, DataLoader],
) -> Iterator[Dict[str, Dict[str, Any]]]:
    """Iterate per-manipulator DataLoaders in lockstep.

    Yields one dict of per-manipulator batches per step; stops when any loader is
    exhausted. Every step contains one sub-batch per manipulator, with per-manipulator
    sample counts equal to each loader's batch_size.
    """
    iterators = {name: iter(loader) for name, loader in loaders.items()}
    if not iterators:
        return
    num_steps = min(len(loader) for loader in loaders.values())
    for _ in range(num_steps):
        step_batches: Dict[str, Dict[str, Any]] = {}
        for name in loaders:
            step_batches[name] = next(iterators[name])
        yield step_batches
