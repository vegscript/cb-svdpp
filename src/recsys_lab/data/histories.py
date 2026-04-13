from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recsys_lab.data.processed import RatingsData


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


def build_user_history_index(
    data: RatingsData,
    *,
    dtype: str = "float32",
) -> UserHistoryIndex:
    if dtype not in {"float32", "float64"}:
        raise ValueError("history dtype must be 'float32' or 'float64'")
    if len(data) == 0:
        raise ValueError("cannot build user histories from empty ratings data")

    order = np.lexsort((data.item_ids.astype(np.int64), data.user_ids.astype(np.int64)))
    sorted_users = data.user_ids[order]
    sorted_items = data.item_ids[order]

    is_new_pair = np.ones(sorted_users.shape[0], dtype=bool)
    is_new_pair[1:] = (sorted_users[1:] != sorted_users[:-1]) | (
        sorted_items[1:] != sorted_items[:-1]
    )

    unique_users = sorted_users[is_new_pair].astype(np.int32, copy=False)
    unique_items = sorted_items[is_new_pair].astype(np.int32, copy=False)

    counts = np.bincount(unique_users, minlength=data.n_users).astype(np.int32, copy=False)
    indptr = np.zeros(data.n_users + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts, dtype=np.int64)

    norms = np.zeros(data.n_users, dtype=np.dtype(dtype))
    nonzero = counts > 0
    norms[nonzero] = 1.0 / np.sqrt(counts[nonzero].astype(np.float64))
    norms = norms.astype(np.dtype(dtype), copy=False)

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

    order = np.lexsort((data.item_ids.astype(np.int64), data.user_ids.astype(np.int64)))
    sorted_users = data.user_ids[order]
    sorted_items = data.item_ids[order]
    sorted_ratings = data.ratings[order]

    is_new_pair = np.ones(sorted_users.shape[0], dtype=bool)
    is_new_pair[1:] = (sorted_users[1:] != sorted_users[:-1]) | (
        sorted_items[1:] != sorted_items[:-1]
    )

    unique_users = sorted_users[is_new_pair].astype(np.int32, copy=False)
    unique_items = sorted_items[is_new_pair].astype(np.int32, copy=False)
    unique_ratings = sorted_ratings[is_new_pair].astype(np.dtype(dtype), copy=False)

    counts = np.bincount(unique_users, minlength=data.n_users).astype(np.int32, copy=False)
    indptr = np.zeros(data.n_users + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts, dtype=np.int64)

    norms = np.zeros(data.n_users, dtype=np.dtype(dtype))
    nonzero = counts > 0
    norms[nonzero] = 1.0 / np.sqrt(counts[nonzero].astype(np.float64))
    norms = norms.astype(np.dtype(dtype), copy=False)

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

    item_cluster_ids = np.asarray(item_clusters, dtype=np.int32)
    if item_cluster_ids.ndim != 1:
        raise ValueError("item_clusters must be a 1D array")

    n_users = history_index.counts.shape[0]
    user_cluster_lists: list[np.ndarray] = []
    user_count_lists: list[np.ndarray] = []
    indptr = np.zeros(n_users + 1, dtype=np.int64)
    running_total = 0

    for user_id in range(n_users):
        items = history_index.items_for_user(user_id)
        if items.size == 0:
            user_cluster_lists.append(np.empty(0, dtype=np.int32))
            user_count_lists.append(np.empty(0, dtype=np.int32))
            indptr[user_id + 1] = running_total
            continue

        cluster_counts = np.bincount(
            item_cluster_ids[items].astype(np.int64, copy=False),
            minlength=n_clusters,
        ).astype(np.int32, copy=False)
        active_clusters = np.flatnonzero(cluster_counts).astype(np.int32, copy=False)
        active_counts = cluster_counts[active_clusters].astype(np.int32, copy=False)

        user_cluster_lists.append(active_clusters)
        user_count_lists.append(active_counts)
        running_total += int(active_clusters.shape[0])
        indptr[user_id + 1] = running_total

    if running_total == 0:
        cluster_ids = np.empty(0, dtype=np.int32)
        counts = np.empty(0, dtype=np.int32)
    else:
        cluster_ids = np.concatenate(user_cluster_lists).astype(np.int32, copy=False)
        counts = np.concatenate(user_count_lists).astype(np.int32, copy=False)

    return UserClusterCountIndex(
        indptr=indptr,
        cluster_ids=cluster_ids,
        counts=counts,
    )
