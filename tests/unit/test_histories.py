from pathlib import Path

import numpy as np
import pytest

from recsys_lab.data.histories import (
    UserClusterCountIndex,
    UserExplicitFeedbackIndex,
    UserHistoryIndex,
    _checked_int32_cumsum,
    build_user_history_index,
    ensure_contiguous_1d,
    ensure_float_array,
    ensure_int32_index_array,
    validate_user_cluster_count_index,
    validate_user_explicit_feedback_index,
    validate_user_history_index,
)
from recsys_lab.data.processed import RatingsData


def test_history_layout_helpers_return_contiguous_int32_arrays() -> None:
    source = np.asarray([0, 2, 4, 6, 8, 10], dtype=np.int64)[::2]

    index = ensure_int32_index_array(source, name="toy.index")

    assert index.dtype == np.int32
    assert index.flags.c_contiguous
    np.testing.assert_array_equal(index, np.asarray([0, 4, 8], dtype=np.int32))


def test_history_layout_helpers_reject_invalid_index_arrays() -> None:
    with pytest.raises(ValueError, match="1D"):
        ensure_contiguous_1d(np.zeros((1, 2), dtype=np.int32), name="toy.values")
    with pytest.raises(ValueError, match="object dtype"):
        ensure_int32_index_array(np.asarray([object()], dtype=object), name="toy.index")
    with pytest.raises(ValueError, match="non-negative"):
        ensure_int32_index_array(np.asarray([-1], dtype=np.int64), name="toy.index")
    with pytest.raises(OverflowError, match="int32"):
        ensure_int32_index_array(np.asarray([np.iinfo(np.int32).max + 1], dtype=np.int64), name="toy.index")


def test_history_layout_helpers_return_contiguous_float_arrays() -> None:
    source = np.asarray([0.25, 0.5, 1.0, 2.0], dtype=np.float64)[::2]

    values = ensure_float_array(source, dtype="float32", name="toy.norms")

    assert values.dtype == np.float32
    assert values.flags.c_contiguous
    np.testing.assert_allclose(values, np.asarray([0.25, 1.0], dtype=np.float32))


def test_history_layout_helpers_reject_invalid_float_arrays() -> None:
    with pytest.raises(ValueError, match="floating dtype"):
        ensure_float_array(np.asarray([1, 2], dtype=np.int32), dtype="float32", name="toy.ratings")
    with pytest.raises(ValueError, match="finite"):
        ensure_float_array(np.asarray([np.nan], dtype=np.float32), dtype="float32", name="toy.ratings")
    with pytest.raises(OverflowError, match="float32"):
        ensure_float_array(
            np.asarray([float(np.finfo(np.float32).max) * 2.0], dtype=np.float64),
            dtype="float32",
            name="toy.ratings",
        )


def test_checked_int32_cumsum_returns_contiguous_indptr_and_checks_overflow() -> None:
    indptr = _checked_int32_cumsum(np.asarray([2, 0, 3], dtype=np.int64), name="toy")

    assert indptr.dtype == np.int32
    assert indptr.flags.c_contiguous
    np.testing.assert_array_equal(indptr, np.asarray([0, 2, 2, 5], dtype=np.int32))

    with pytest.raises(OverflowError, match="indptr"):
        _checked_int32_cumsum(np.asarray([np.iinfo(np.int32).max, 1], dtype=np.int64), name="toy")


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
    assert histories.indptr.dtype == np.int32
    assert histories.item_indices.dtype == np.int32
    assert histories.counts.dtype == np.int32
    assert histories.norms.dtype == np.float32
    assert histories.indptr.flags.c_contiguous
    assert histories.item_indices.flags.c_contiguous
    assert histories.counts.flags.c_contiguous
    assert histories.norms.flags.c_contiguous


def test_build_user_history_index_matches_materialized_subset_view() -> None:
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

    subset_view = data.subset(np.asarray([0, 2, 3, 5], dtype=np.int64), name="toy:subset")
    subset_materialized = subset_view.materialize(force_copy=True)

    view_histories = build_user_history_index(subset_view, dtype="float32")
    materialized_histories = build_user_history_index(subset_materialized, dtype="float32")

    np.testing.assert_array_equal(view_histories.indptr, materialized_histories.indptr)
    np.testing.assert_array_equal(view_histories.item_indices, materialized_histories.item_indices)
    np.testing.assert_array_equal(view_histories.counts, materialized_histories.counts)
    np.testing.assert_allclose(view_histories.norms, materialized_histories.norms)


def test_build_user_history_index_preserves_configured_float64_norm_dtype() -> None:
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

    histories = build_user_history_index(data, dtype="float64")

    assert histories.indptr.dtype == np.int32
    assert histories.item_indices.dtype == np.int32
    assert histories.counts.dtype == np.int32
    assert histories.norms.dtype == np.float64
    assert histories.norms.flags.c_contiguous


def _valid_history_index() -> UserHistoryIndex:
    return UserHistoryIndex(
        indptr=np.asarray([0, 2, 2, 3], dtype=np.int32),
        item_indices=np.asarray([0, 2, 1], dtype=np.int32),
        counts=np.asarray([2, 0, 1], dtype=np.int32),
        norms=np.asarray([1.0 / np.sqrt(2.0), 0.0, 1.0], dtype=np.float32),
    )


def _valid_explicit_feedback_index() -> UserExplicitFeedbackIndex:
    return UserExplicitFeedbackIndex(
        indptr=np.asarray([0, 2, 2, 3], dtype=np.int32),
        item_indices=np.asarray([0, 2, 1], dtype=np.int32),
        ratings=np.asarray([5.0, 4.0, 3.0], dtype=np.float32),
        counts=np.asarray([2, 0, 1], dtype=np.int32),
        norms=np.asarray([1.0 / np.sqrt(2.0), 0.0, 1.0], dtype=np.float32),
    )


def _valid_cluster_count_index() -> UserClusterCountIndex:
    return UserClusterCountIndex(
        indptr=np.asarray([0, 1, 1, 3], dtype=np.int32),
        cluster_ids=np.asarray([0, 0, 2], dtype=np.int32),
        counts=np.asarray([2, 1, 1], dtype=np.int32),
    )


def test_validate_user_history_index_enforces_layout_contract() -> None:
    index = _valid_history_index()

    validate_user_history_index(index, n_users=3, n_items=3, dtype="float32")

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
    with pytest.raises(ValueError, match=r"indptr\[0\]"):
        validate_user_history_index(
            UserHistoryIndex(
                indptr=np.asarray([1, 2, 2, 3], dtype=np.int32),
                item_indices=index.item_indices,
                counts=index.counts,
                norms=index.norms,
            ),
            n_users=3,
        )
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
    with pytest.raises(ValueError, match="bounds"):
        validate_user_history_index(index, n_users=3, n_items=2, dtype="float32")
    with pytest.raises(ValueError, match="finite"):
        validate_user_history_index(
            UserHistoryIndex(
                indptr=index.indptr,
                item_indices=index.item_indices,
                counts=index.counts,
                norms=np.asarray([np.nan, 0.0, 1.0], dtype=np.float32),
            ),
            n_users=3,
            dtype="float32",
        )


def test_validate_user_explicit_feedback_index_enforces_layout_contract() -> None:
    index = _valid_explicit_feedback_index()

    validate_user_explicit_feedback_index(index, n_users=3, n_items=3, dtype="float32")

    with pytest.raises(ValueError, match="ratings.*float32 or float64"):
        validate_user_explicit_feedback_index(
            UserExplicitFeedbackIndex(
                indptr=index.indptr,
                item_indices=index.item_indices,
                ratings=index.ratings.astype(np.int32),
                counts=index.counts,
                norms=index.norms,
            ),
            n_users=3,
        )
    with pytest.raises(ValueError, match="configured dtype"):
        validate_user_explicit_feedback_index(index, n_users=3, dtype="float64")
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
    with pytest.raises(ValueError, match="identical shape"):
        validate_user_explicit_feedback_index(
            UserExplicitFeedbackIndex(
                indptr=np.asarray([0, 2, 2, 3], dtype=np.int32),
                item_indices=index.item_indices,
                ratings=np.asarray([5.0, 4.0], dtype=np.float32),
                counts=index.counts,
                norms=index.norms,
            ),
            n_users=3,
            dtype="float32",
        )


def test_validate_user_cluster_count_index_enforces_layout_contract() -> None:
    index = _valid_cluster_count_index()

    validate_user_cluster_count_index(index, n_users=3, n_clusters=3)

    with pytest.raises(ValueError, match="cluster_ids.*int32"):
        validate_user_cluster_count_index(
            UserClusterCountIndex(
                indptr=index.indptr,
                cluster_ids=index.cluster_ids.astype(np.int64),
                counts=index.counts,
            ),
            n_users=3,
        )
    with pytest.raises(ValueError, match="positive"):
        validate_user_cluster_count_index(
            UserClusterCountIndex(
                indptr=index.indptr,
                cluster_ids=index.cluster_ids,
                counts=np.asarray([2, 0, 1], dtype=np.int32),
            ),
            n_users=3,
        )
    with pytest.raises(ValueError, match="bounds"):
        validate_user_cluster_count_index(index, n_users=3, n_clusters=2)
    with pytest.raises(ValueError, match="payload length"):
        validate_user_cluster_count_index(
            UserClusterCountIndex(
                indptr=np.asarray([0, 1, 1, 2], dtype=np.int32),
                cluster_ids=index.cluster_ids,
                counts=index.counts,
            ),
            n_users=3,
        )
