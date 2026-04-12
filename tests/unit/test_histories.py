from pathlib import Path

import numpy as np
import pytest

from recsys_lab.data.histories import build_user_history_index
from recsys_lab.data.processed import RatingsData


def test_build_user_history_index_deduplicates_user_item_pairs() -> None:
    data = RatingsData(
        user_ids=np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int32),
        item_ids=np.asarray([2, 2, 1, 3, 2, 2], dtype=np.int32),
        ratings=np.asarray([5.0, 4.5, 4.0, 3.0, 2.5, 4.0], dtype=np.float32),
        n_users=3,
        n_items=4,
        name="toy",
        rating_min=2.5,
        rating_max=5.0,
        source_manifest_path=Path("toy_manifest.json"),
    )

    histories = build_user_history_index(data, dtype="float32")

    assert histories.items_for_user(0).tolist() == [1, 2]
    assert histories.items_for_user(1).tolist() == [2, 3]
    assert histories.items_for_user(2).tolist() == [2]
    assert histories.counts.tolist() == [2, 2, 1]
    assert histories.norms[0] == pytest.approx(1.0 / np.sqrt(2.0))
