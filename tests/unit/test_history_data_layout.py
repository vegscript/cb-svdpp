from pathlib import Path

import numpy as np
import pytest

from recsys_lab.data.histories import (
    UserClusterCountIndex,
    UserExplicitFeedbackIndex,
    UserHistoryIndex,
    _checked_int32_cumsum,
    build_user_cluster_count_index,
    build_user_explicit_feedback_index,
    build_user_history_index,
    validate_user_cluster_count_index,
    validate_user_explicit_feedback_index,
    validate_user_history_index,
)
from recsys_lab.data.processed import RatingsData


def _toy_data() -> RatingsData:
    return RatingsData(
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


def _valid_history_index() -> UserHistoryIndex:
    return UserHistoryIndex(
        indptr=np.asarray([0, 2, 2, 3], dtype=np.int32),
        item_indices=np.asarray([0, 2, 1], dtype=np.int32),
        counts=np.asarray([2, 0, 1], dtype=np.int32),
        norms=np.asarray([1.0 / np.sqrt(2.0), 0.0, 1.0], dtype=np.float32),
    )


def _valid_explicit_feedback_index() -> UserExplicitFeedbackIndex:
    history = _valid_history_index()
    return UserExplicitFeedbackIndex(
        indptr=history.indptr,
        item_indices=history.item_indices,
        ratings=np.asarray([5.0, 4.0, 3.0], dtype=np.float32),
        counts=history.counts,
        norms=history.norms,
    )


def _valid_cluster_count_index() -> UserClusterCountIndex:
    return UserClusterCountIndex(
        indptr=np.asarray([0, 1, 1, 3], dtype=np.int32),
        cluster_ids=np.asarray([0, 0, 2], dtype=np.int32),
        counts=np.asarray([2, 1, 1], dtype=np.int32),
    )


def _assert_int32_contiguous(array: np.ndarray) -> None:
    assert array.dtype == np.int32
    assert array.flags.c_contiguous


def _assert_float32_contiguous(array: np.ndarray) -> None:
    assert array.dtype == np.float32
    assert array.flags.c_contiguous


def test_build_user_history_index_uses_int32_contiguous_layout() -> None:
    index = build_user_history_index(_toy_data(), dtype="float32")

    _assert_int32_contiguous(index.indptr)
    _assert_int32_contiguous(index.item_indices)
    _assert_int32_contiguous(index.counts)
    _assert_float32_contiguous(index.norms)
    validate_user_history_index(index, n_users=3, n_items=4, dtype="float32")


def test_build_user_explicit_feedback_index_uses_int32_contiguous_layout() -> None:
    index = build_user_explicit_feedback_index(_toy_data(), dtype="float32")

    _assert_int32_contiguous(index.indptr)
    _assert_int32_contiguous(index.item_indices)
    _assert_int32_contiguous(index.counts)
    _assert_float32_contiguous(index.ratings)
    _assert_float32_contiguous(index.norms)
    validate_user_explicit_feedback_index(index, n_users=3, n_items=4, dtype="float32")


def test_build_user_cluster_count_index_uses_int32_contiguous_layout() -> None:
    history = build_user_history_index(_toy_data(), dtype="float32")
    index = build_user_cluster_count_index(
        history,
        np.asarray([0, 1, 1, 0], dtype=np.int32),
        n_clusters=2,
    )

    _assert_int32_contiguous(index.indptr)
    _assert_int32_contiguous(index.cluster_ids)
    _assert_int32_contiguous(index.counts)
    validate_user_cluster_count_index(index, n_users=3, n_clusters=2)


def test_history_index_validator_rejects_int64_indptr() -> None:
    index = _valid_history_index()

    with pytest.raises(ValueError, match="indptr.*int32"):
        validate_user_history_index(
            UserHistoryIndex(
                indptr=index.indptr.astype(np.int64),
                item_indices=index.item_indices,
                counts=index.counts,
                norms=index.norms,
            ),
            n_users=3,
        )


def test_history_index_validator_rejects_non_contiguous_arrays() -> None:
    index = _valid_history_index()

    with pytest.raises(ValueError, match="C-contiguous"):
        validate_user_history_index(
            UserHistoryIndex(
                indptr=index.indptr,
                item_indices=np.asarray([0, 9, 2, 9, 1, 9], dtype=np.int32)[::2],
                counts=index.counts,
                norms=index.norms,
            ),
            n_users=3,
        )


def test_history_index_validator_rejects_non_monotonic_indptr() -> None:
    index = _valid_history_index()

    with pytest.raises(ValueError, match="monotonic"):
        validate_user_history_index(
            UserHistoryIndex(
                indptr=np.asarray([0, 2, 1, 3], dtype=np.int32),
                item_indices=index.item_indices,
                counts=index.counts,
                norms=index.norms,
            ),
            n_users=3,
        )


def test_history_index_validator_rejects_counts_mismatch() -> None:
    index = _valid_history_index()

    with pytest.raises(ValueError, match="counts"):
        validate_user_history_index(
            UserHistoryIndex(
                indptr=index.indptr,
                item_indices=index.item_indices,
                counts=np.asarray([1, 1, 1], dtype=np.int32),
                norms=index.norms,
            ),
            n_users=3,
        )


def test_history_index_validator_rejects_negative_item_ids() -> None:
    index = _valid_history_index()

    with pytest.raises(ValueError, match="non-negative"):
        validate_user_history_index(
            UserHistoryIndex(
                indptr=index.indptr,
                item_indices=np.asarray([0, -1, 1], dtype=np.int32),
                counts=index.counts,
                norms=index.norms,
            ),
            n_users=3,
        )


def test_explicit_feedback_validator_rejects_non_finite_ratings() -> None:
    index = _valid_explicit_feedback_index()

    with pytest.raises(ValueError, match="finite"):
        validate_user_explicit_feedback_index(
            UserExplicitFeedbackIndex(
                indptr=index.indptr,
                item_indices=index.item_indices,
                ratings=np.asarray([5.0, np.inf, 3.0], dtype=np.float32),
                counts=index.counts,
                norms=index.norms,
            ),
            n_users=3,
            dtype="float32",
        )


def test_cluster_count_validator_rejects_invalid_cluster_ids() -> None:
    index = _valid_cluster_count_index()

    with pytest.raises(ValueError, match="bounds"):
        validate_user_cluster_count_index(index, n_users=3, n_clusters=2)


def test_checked_int32_cumsum_rejects_overflow() -> None:
    with pytest.raises(OverflowError, match="indptr"):
        _checked_int32_cumsum(np.asarray([np.iinfo(np.int32).max, 1], dtype=np.int64), name="toy")


def test_empty_history_users_have_zero_norm() -> None:
    index = _valid_history_index()

    assert index.counts[1] == 0
    assert index.norms[1] == np.float32(0.0)
    validate_user_history_index(index, n_users=3, n_items=3, dtype="float32")


def test_nonzero_history_users_have_inverse_sqrt_norm() -> None:
    index = _valid_history_index()

    assert index.norms[0] == pytest.approx(1.0 / np.sqrt(index.counts[0]))
    assert index.norms[2] == pytest.approx(1.0 / np.sqrt(index.counts[2]))
    validate_user_history_index(index, n_users=3, n_items=3, dtype="float32")
