from __future__ import annotations

from typing import Callable

import numpy as np


def build_unique_user_context_batch(
    *,
    user_ids: np.ndarray,
    context_dim: int,
    build_user_context: Callable[[int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    users = np.asarray(user_ids, dtype=np.int64)
    unique_users, inverse = np.unique(users, return_inverse=True)
    contexts = np.empty((unique_users.shape[0], context_dim), dtype=np.float64)

    for position, user_id in enumerate(unique_users):
        contexts[position] = np.asarray(build_user_context(int(user_id)), dtype=np.float64)

    return users, inverse.astype(np.int64, copy=False), contexts


def build_user_context_cache(
    *,
    n_users: int,
    context_dim: int,
    build_user_context: Callable[[int], np.ndarray],
) -> np.ndarray:
    cache = np.empty((n_users, context_dim), dtype=np.float64)
    for user_id in range(n_users):
        cache[user_id] = np.asarray(build_user_context(user_id), dtype=np.float64)
    return cache
