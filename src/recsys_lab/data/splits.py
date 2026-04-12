from __future__ import annotations

import random
from dataclasses import dataclass

from recsys_lab.data.schemas import InteractionDataset


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train: InteractionDataset
    validation: InteractionDataset
    test: InteractionDataset


def random_split(
    dataset: InteractionDataset,
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetSplit:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0, 1)")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be < 1")

    rng = random.Random(seed)
    interactions = list(dataset.interactions)
    rng.shuffle(interactions)

    train_end = int(len(interactions) * train_ratio)
    validation_end = train_end + int(len(interactions) * validation_ratio)

    def build(name: str, rows: list) -> InteractionDataset:
        return InteractionDataset.from_iterable(
            rows,
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            name=f"{dataset.name}:{name}",
        )

    return DatasetSplit(
        train=build("train", interactions[:train_end]),
        validation=build("validation", interactions[train_end:validation_end]),
        test=build("test", interactions[validation_end:]),
    )
