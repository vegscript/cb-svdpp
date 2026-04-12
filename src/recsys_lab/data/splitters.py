from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recsys_lab.data.processed import RatingsData


@dataclass(frozen=True, slots=True)
class RatingsSplit:
    train: RatingsData
    validation: RatingsData
    test: RatingsData


def random_split_with_train_coverage(
    data: RatingsData,
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> RatingsSplit:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0, 1)")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be < 1")

    rng = np.random.default_rng(seed)
    n_rows = len(data)
    permutation = rng.permutation(n_rows)

    train_end = int(n_rows * train_ratio)
    validation_end = train_end + int(n_rows * validation_ratio)

    train_idx = list(permutation[:train_end])
    validation_idx = list(permutation[train_end:validation_end])
    test_idx = list(permutation[validation_end:])

    train_user_set = set(int(data.user_ids[idx]) for idx in train_idx)
    train_item_set = set(int(data.item_ids[idx]) for idx in train_idx)

    remainder = validation_idx + test_idx
    kept_remainder: list[int] = []

    for idx in remainder:
        user_id = int(data.user_ids[idx])
        item_id = int(data.item_ids[idx])
        if user_id not in train_user_set or item_id not in train_item_set:
            train_idx.append(int(idx))
            train_user_set.add(user_id)
            train_item_set.add(item_id)
        else:
            kept_remainder.append(int(idx))

    original_val_target = len(validation_idx)
    new_validation_idx = np.asarray(kept_remainder[:original_val_target], dtype=np.int64)
    new_test_idx = np.asarray(kept_remainder[original_val_target:], dtype=np.int64)
    new_train_idx = np.asarray(train_idx, dtype=np.int64)

    return RatingsSplit(
        train=data.subset(new_train_idx, name=f"{data.name}:train"),
        validation=data.subset(new_validation_idx, name=f"{data.name}:validation"),
        test=data.subset(new_test_idx, name=f"{data.name}:test"),
    )
