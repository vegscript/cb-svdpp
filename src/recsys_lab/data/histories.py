from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recsys_lab.data.processed import RatingsData

INDEX_DTYPE = np.int32
_INDEX_DTYPE_INFO = np.iinfo(INDEX_DTYPE)


@dataclass(frozen=True, slots=True)
class UserHistoryIndex:
    indptr: np.ndarray
    item_indices: np.ndarray
    counts: np.ndarray
    norms: np.ndarray

    def items_for_user(self, user_id: int) -> np.ndarray:
        start = int(self.indptr[user_id])
        end = int(self.indptr[user_id + 1])
        return self.item_indices[start:end]


@dataclass(frozen=True, slots=True)
class UserExplicitFeedbackIndex:
    indptr: np.ndarray
    item_indices: np.ndarray
    ratings: np.ndarray
    counts: np.ndarray
    norms: np.ndarray

    def items_for_user(self, user_id: int) -> np.ndarray:
        start = int(self.indptr[user_id])
        end = int(self.indptr[user_id + 1])
        return self.item_indices[start:end]

    def ratings_for_user(self, user_id: int) -> np.ndarray:
        start = int(self.indptr[user_id])
        end = int(self.indptr[user_id + 1])
        return self.ratings[start:end]


@dataclass(frozen=True, slots=True)
class UserClusterCountIndex:
    indptr: np.ndarray
    cluster_ids: np.ndarray
    counts: np.ndarray

    def clusters_for_user(self, user_id: int) -> np.ndarray:
        start = int(self.indptr[user_id])
        end = int(self.indptr[user_id + 1])
        return self.cluster_ids[start:end]

    def counts_for_user(self, user_id: int) -> np.ndarray:
        start = int(self.indptr[user_id])
        end = int(self.indptr[user_id + 1])
        return self.counts[start:end]


def ensure_contiguous_1d(array: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(array)
    if values.dtype == np.dtype(object):
        raise ValueError(f"{name} must not use object dtype")
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return np.ascontiguousarray(values)


def ensure_int32_index_array(array: np.ndarray, *, name: str) -> np.ndarray:
    values = ensure_contiguous_1d(array, name=name)
    if values.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must use an integer dtype")
    if values.size:
        if np.any(values < 0):
            raise ValueError(f"{name} must contain non-negative values")
        max_value = int(np.max(values))
        if max_value > int(_INDEX_DTYPE_INFO.max):
            raise OverflowError(f"{name} contains values outside int32 range")
    return np.ascontiguousarray(values.astype(INDEX_DTYPE, copy=False))


def ensure_float_array(array: np.ndarray, *, dtype: str, name: str) -> np.ndarray:
    target_dtype = _history_float_dtype(dtype)
    values = ensure_contiguous_1d(array, name=name)
    if values.dtype.kind != "f":
        raise ValueError(f"{name} must use a floating dtype")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = values.astype(target_dtype, copy=False)
    if not np.all(np.isfinite(converted)):
        raise OverflowError(f"{name} contains values outside {target_dtype.name} finite range")
    return np.ascontiguousarray(converted)


def _checked_int32_cumsum(counts: np.ndarray, *, name: str) -> np.ndarray:
    index_counts = ensure_int32_index_array(counts, name=f"{name}.counts")
    cumulative = np.cumsum(index_counts, dtype=np.int64)
    if cumulative.size and int(cumulative[-1]) > int(_INDEX_DTYPE_INFO.max):
        raise OverflowError(f"{name}.indptr exceeds int32 range")
    indptr = np.empty(index_counts.shape[0] + 1, dtype=INDEX_DTYPE)
    indptr[0] = 0
    indptr[1:] = cumulative.astype(INDEX_DTYPE, copy=False)
    return np.ascontiguousarray(indptr)


def _history_float_dtype(dtype: str) -> np.dtype:
    if dtype not in {"float32", "float64"}:
        raise ValueError("history dtype must be 'float32' or 'float64'")
    return np.dtype(dtype)


def validate_user_history_index(
    history_index: UserHistoryIndex,
    *,
    n_users: int,
    n_items: int | None = None,
    dtype: str | None = None,
    name: str = "user_history_index",
) -> None:
    _validate_int32_array(history_index.item_indices, name=f"{name}.item_indices")
    _validate_int32_array(history_index.counts, name=f"{name}.counts")
    _validate_float_array_contract(history_index.norms, dtype=dtype, name=f"{name}.norms")
    _validate_indptr(
        history_index.indptr,
        n_users=n_users,
        payload_length=history_index.item_indices.shape[0],
        name=name,
    )
    if history_index.counts.shape != (n_users,):
        raise ValueError(f"{name}.counts must have shape ({n_users},)")
    _validate_counts_against_indptr(history_index.counts, history_index.indptr, name=name)
    if history_index.norms.shape != (n_users,):
        raise ValueError(f"{name}.norms must have shape ({n_users},)")
    _validate_norms(history_index.norms, history_index.counts, name=name)
    _validate_id_bounds(history_index.item_indices, upper_bound=n_items, name=f"{name}.item_indices")


def validate_user_explicit_feedback_index(
    explicit_feedback_index: UserExplicitFeedbackIndex,
    *,
    n_users: int,
    n_items: int | None = None,
    dtype: str | None = None,
    name: str = "explicit_feedback_index",
) -> None:
    _validate_int32_array(explicit_feedback_index.item_indices, name=f"{name}.item_indices")
    _validate_int32_array(explicit_feedback_index.counts, name=f"{name}.counts")
    _validate_float_array_contract(explicit_feedback_index.ratings, dtype=dtype, name=f"{name}.ratings")
    _validate_float_array_contract(explicit_feedback_index.norms, dtype=dtype, name=f"{name}.norms")
    _validate_indptr(
        explicit_feedback_index.indptr,
        n_users=n_users,
        payload_length=explicit_feedback_index.item_indices.shape[0],
        name=name,
    )
    if explicit_feedback_index.counts.shape != (n_users,):
        raise ValueError(f"{name}.counts must have shape ({n_users},)")
    _validate_counts_against_indptr(explicit_feedback_index.counts, explicit_feedback_index.indptr, name=name)
    if explicit_feedback_index.norms.shape != (n_users,):
        raise ValueError(f"{name}.norms must have shape ({n_users},)")
    _validate_norms(explicit_feedback_index.norms, explicit_feedback_index.counts, name=name)
    if explicit_feedback_index.item_indices.shape != explicit_feedback_index.ratings.shape:
        raise ValueError(f"{name}.item_indices and ratings must have identical shape")
    _validate_id_bounds(explicit_feedback_index.item_indices, upper_bound=n_items, name=f"{name}.item_indices")


def validate_user_cluster_count_index(
    cluster_count_index: UserClusterCountIndex,
    *,
    n_users: int,
    n_clusters: int | None = None,
    name: str = "user_cluster_count_index",
) -> None:
    _validate_int32_array(cluster_count_index.cluster_ids, name=f"{name}.cluster_ids")
    _validate_int32_array(cluster_count_index.counts, name=f"{name}.counts")
    _validate_indptr(
        cluster_count_index.indptr,
        n_users=n_users,
        payload_length=cluster_count_index.cluster_ids.shape[0],
        name=name,
    )
    if cluster_count_index.cluster_ids.shape != cluster_count_index.counts.shape:
        raise ValueError(f"{name}.cluster_ids and counts must have identical shape")
    if np.any(cluster_count_index.counts <= 0):
        raise ValueError(f"{name}.counts must be positive for stored entries")
    _validate_id_bounds(cluster_count_index.cluster_ids, upper_bound=n_clusters, name=f"{name}.cluster_ids")


def _validate_int32_array(array: np.ndarray, *, name: str) -> None:
    if array.dtype != INDEX_DTYPE:
        raise ValueError(f"{name} must have dtype int32")
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if np.any(array < 0):
        raise ValueError(f"{name} must contain non-negative values")


def _validate_indptr(indptr: np.ndarray, *, n_users: int, payload_length: int, name: str) -> None:
    _validate_int32_array(indptr, name=f"{name}.indptr")
    if indptr.shape != (n_users + 1,):
        raise ValueError(f"{name}.indptr must have shape ({n_users + 1},)")
    if int(indptr[0]) != 0:
        raise ValueError(f"{name}.indptr[0] must be 0")
    if np.any(np.diff(indptr.astype(np.int64, copy=False)) < 0):
        raise ValueError(f"{name}.indptr must be monotonic nondecreasing")
    if int(indptr[-1]) != int(payload_length):
        raise ValueError(f"{name}.indptr[-1] must match payload length")


def _validate_counts_against_indptr(counts: np.ndarray, indptr: np.ndarray, *, name: str) -> None:
    _validate_int32_array(counts, name=f"{name}.counts")
    expected = np.diff(indptr.astype(np.int64, copy=False))
    if not np.array_equal(counts.astype(np.int64, copy=False), expected):
        raise ValueError(f"{name}.counts must equal np.diff(indptr)")


def _validate_float_array_contract(array: np.ndarray, *, dtype: str | None, name: str) -> None:
    if array.dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise ValueError(f"{name} must have dtype float32 or float64")
    if dtype is not None and array.dtype != _history_float_dtype(dtype):
        raise ValueError(f"{name} must match configured dtype {dtype}")
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")


def _validate_norms(norms: np.ndarray, counts: np.ndarray, *, name: str) -> None:
    expected = np.zeros(counts.shape[0], dtype=norms.dtype)
    nonzero = counts > 0
    expected[nonzero] = 1.0 / np.sqrt(counts[nonzero].astype(np.float64))
    if not np.allclose(norms, expected, rtol=1e-6, atol=1e-7):
        raise ValueError(f"{name}.norms are invalid")


def _validate_id_bounds(array: np.ndarray, *, upper_bound: int | None, name: str) -> None:
    if upper_bound is None or array.size == 0:
        return
    if upper_bound < 0:
        raise ValueError(f"{name} upper bound must be non-negative")
    if int(np.max(array)) >= int(upper_bound):
        raise ValueError(f"{name} contains ids outside configured bounds")


def build_user_history_index(
    data: RatingsData,
    *,
    dtype: str = "float32",
) -> UserHistoryIndex:
    history_dtype = _history_float_dtype(dtype)
    if len(data) == 0:
        raise ValueError("cannot build user histories from empty ratings data")

    if data.row_indices is None:
        selected_user_ids = data.base_user_ids
        selected_item_ids = data.base_item_ids
    else:
        selected_user_ids = data.base_user_ids[data.row_indices]
        selected_item_ids = data.base_item_ids[data.row_indices]

    order = np.lexsort((selected_item_ids.astype(np.int64), selected_user_ids.astype(np.int64)))
    sorted_users = selected_user_ids[order]
    sorted_items = selected_item_ids[order]

    is_new_pair = np.ones(sorted_users.shape[0], dtype=bool)
    is_new_pair[1:] = (sorted_users[1:] != sorted_users[:-1]) | (sorted_items[1:] != sorted_items[:-1])

    unique_users = ensure_int32_index_array(sorted_users[is_new_pair], name="user_history.users")
    unique_items = ensure_int32_index_array(sorted_items[is_new_pair], name="user_history.item_indices")

    counts = ensure_int32_index_array(np.bincount(unique_users, minlength=data.n_users), name="user_history.counts")
    indptr = _checked_int32_cumsum(counts, name="user_history")

    norms = np.zeros(data.n_users, dtype=history_dtype)
    nonzero = counts > 0
    norms[nonzero] = 1.0 / np.sqrt(counts[nonzero].astype(np.float64))
    norms = ensure_float_array(norms, dtype=dtype, name="user_history.norms")

    return UserHistoryIndex(
        indptr=indptr,
        item_indices=unique_items,
        counts=counts,
        norms=norms,
    )


def build_user_explicit_feedback_index(
    data: RatingsData,
    *,
    dtype: str = "float32",
) -> UserExplicitFeedbackIndex:
    if dtype not in {"float32", "float64"}:
        raise ValueError("explicit feedback dtype must be 'float32' or 'float64'")
    if len(data) == 0:
        raise ValueError("cannot build explicit feedback index from empty ratings data")

    if data.row_indices is None:
        selected_user_ids = data.base_user_ids
        selected_item_ids = data.base_item_ids
        selected_ratings = data.base_ratings
    else:
        selected_user_ids = data.base_user_ids[data.row_indices]
        selected_item_ids = data.base_item_ids[data.row_indices]
        selected_ratings = data.base_ratings[data.row_indices]

    order = np.lexsort((selected_item_ids.astype(np.int64), selected_user_ids.astype(np.int64)))
    sorted_users = selected_user_ids[order]
    sorted_items = selected_item_ids[order]
    sorted_ratings = selected_ratings[order]

    is_new_pair = np.ones(sorted_users.shape[0], dtype=bool)
    is_new_pair[1:] = (sorted_users[1:] != sorted_users[:-1]) | (sorted_items[1:] != sorted_items[:-1])

    unique_users = ensure_int32_index_array(sorted_users[is_new_pair], name="explicit_feedback.users")
    unique_items = ensure_int32_index_array(sorted_items[is_new_pair], name="explicit_feedback.item_indices")
    unique_ratings = ensure_float_array(sorted_ratings[is_new_pair], dtype=dtype, name="explicit_feedback.ratings")

    counts = ensure_int32_index_array(
        np.bincount(unique_users, minlength=data.n_users),
        name="explicit_feedback.counts",
    )
    indptr = _checked_int32_cumsum(counts, name="explicit_feedback")

    norms = np.zeros(data.n_users, dtype=np.dtype(dtype))
    nonzero = counts > 0
    norms[nonzero] = 1.0 / np.sqrt(counts[nonzero].astype(np.float64))
    norms = ensure_float_array(norms, dtype=dtype, name="explicit_feedback.norms")

    return UserExplicitFeedbackIndex(
        indptr=indptr,
        item_indices=unique_items,
        ratings=unique_ratings,
        counts=counts,
        norms=norms,
    )


def build_user_cluster_count_index(
    history_index: UserHistoryIndex,
    item_clusters: np.ndarray,
    *,
    n_clusters: int,
) -> UserClusterCountIndex:
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")

    item_cluster_ids = ensure_int32_index_array(item_clusters, name="item_clusters")

    n_users = history_index.counts.shape[0]
    user_cluster_lists: list[np.ndarray] = []
    user_count_lists: list[np.ndarray] = []
    user_cluster_counts = np.zeros(n_users, dtype=INDEX_DTYPE)
    running_total = 0

    for user_id in range(n_users):
        items = history_index.items_for_user(user_id)
        if items.size == 0:
            user_cluster_lists.append(np.empty(0, dtype=np.int32))
            user_count_lists.append(np.empty(0, dtype=np.int32))
            continue

        cluster_counts = np.bincount(
            item_cluster_ids[items].astype(np.int64, copy=False),
            minlength=n_clusters,
        )
        active_clusters = ensure_int32_index_array(np.flatnonzero(cluster_counts), name="cluster_history.cluster_ids")
        active_counts = ensure_int32_index_array(cluster_counts[active_clusters], name="cluster_history.counts")

        user_cluster_lists.append(active_clusters)
        user_count_lists.append(active_counts)
        running_total += int(active_clusters.shape[0])
        if running_total > int(_INDEX_DTYPE_INFO.max):
            raise OverflowError("cluster_history.indptr exceeds int32 range")
        user_cluster_counts[user_id] = int(active_clusters.shape[0])

    indptr = _checked_int32_cumsum(user_cluster_counts, name="cluster_history")

    if running_total == 0:
        cluster_ids = np.empty(0, dtype=np.int32)
        counts = np.empty(0, dtype=np.int32)
    else:
        cluster_ids = ensure_int32_index_array(np.concatenate(user_cluster_lists), name="cluster_history.cluster_ids")
        counts = ensure_int32_index_array(np.concatenate(user_count_lists), name="cluster_history.counts")

    return UserClusterCountIndex(
        indptr=indptr,
        cluster_ids=cluster_ids,
        counts=counts,
    )
