from pathlib import Path

import numpy as np

from recsys_lab.data.processed import RatingsData
from recsys_lab.data.splitters import random_split_with_train_coverage


def test_random_split_preserves_train_user_and_item_coverage() -> None:
    data = RatingsData(
        user_ids=np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int32),
        item_ids=np.asarray([0, 1, 2, 0, 2, 3, 1, 2, 3, 0, 1, 3], dtype=np.int32),
        ratings=np.asarray([4.0, 3.5, 5.0, 2.5, 4.0, 3.0, 4.5, 5.0, 2.0, 3.0, 4.0, 4.5]),
        n_users=4,
        n_items=4,
        name="toy",
        rating_min=2.0,
        rating_max=5.0,
        source_manifest_path=Path("toy_manifest.json"),
    )

    split = random_split_with_train_coverage(
        data,
        train_ratio=0.5,
        validation_ratio=0.25,
        seed=7,
    )

    assert len(split.train) + len(split.validation) + len(split.test) == len(data)
    assert set(split.train.user_ids.tolist()) == {0, 1, 2, 3}
    assert set(split.train.item_ids.tolist()) == {0, 1, 2, 3}
