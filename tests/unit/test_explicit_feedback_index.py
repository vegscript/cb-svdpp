from pathlib import Path

import numpy as np
import pytest

from recsys_lab.data.histories import build_user_explicit_feedback_index
from recsys_lab.data.processed import RatingsData


def test_build_user_explicit_feedback_index_deduplicates_pairs_and_keeps_ratings() -> None:
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

    feedback = build_user_explicit_feedback_index(data, dtype="float32")

    assert feedback.items_for_user(0).tolist() == [1, 2]
    assert feedback.ratings_for_user(0).tolist() == pytest.approx([4.0, 5.0])
    assert feedback.items_for_user(1).tolist() == [2, 3]
    assert feedback.ratings_for_user(1).tolist() == pytest.approx([2.5, 3.0])
    assert feedback.counts.tolist() == [2, 2, 1]
    assert feedback.indptr.dtype == np.int32
    assert feedback.item_indices.dtype == np.int32
    assert feedback.ratings.dtype == np.float32
    assert feedback.counts.dtype == np.int32
    assert feedback.norms.dtype == np.float32
    assert feedback.indptr.flags.c_contiguous
    assert feedback.item_indices.flags.c_contiguous
    assert feedback.ratings.flags.c_contiguous
    assert feedback.counts.flags.c_contiguous
    assert feedback.norms.flags.c_contiguous


def test_build_user_explicit_feedback_index_preserves_configured_float64_dtype() -> None:
    data = RatingsData(
        user_ids=np.asarray([0, 0, 1], dtype=np.int32),
        item_ids=np.asarray([0, 1, 1], dtype=np.int32),
        ratings=np.asarray([5.0, 4.0, 3.0], dtype=np.float32),
        n_users=2,
        n_items=2,
        name="toy",
        rating_min=3.0,
        rating_max=5.0,
        source_manifest_path=Path("toy_manifest.json"),
    )

    feedback = build_user_explicit_feedback_index(data, dtype="float64")

    assert feedback.indptr.dtype == np.int32
    assert feedback.item_indices.dtype == np.int32
    assert feedback.ratings.dtype == np.float64
    assert feedback.counts.dtype == np.int32
    assert feedback.norms.dtype == np.float64
    assert feedback.ratings.flags.c_contiguous
    assert feedback.norms.flags.c_contiguous
