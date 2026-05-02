from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from recsys_lab.experiments.benchmarking import summarize_scalar_samples


def build_repeated_sorted_prefix_query(
    *,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    prefix_rows: int,
    repeat_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    if prefix_rows <= 0:
        raise ValueError("prefix_rows must be positive")
    if repeat_factor <= 0:
        raise ValueError("repeat_factor must be positive")

    users = np.asarray(user_ids, dtype=np.int64)
    items = np.asarray(item_ids, dtype=np.int64)
    if users.shape != items.shape:
        raise ValueError("user_ids and item_ids must have identical shape")
    if users.size == 0:
        raise ValueError("query source must contain at least one row")

    order = np.argsort(users, kind="stable")
    prefix_size = min(prefix_rows, users.size)
    prefix_users = users[order][:prefix_size]
    prefix_items = items[order][:prefix_size]
    return (
        np.tile(prefix_users, repeat_factor).astype(np.int64, copy=False),
        np.tile(prefix_items, repeat_factor).astype(np.int64, copy=False),
    )


def time_inference_variant(
    *,
    predict_many_fn: Any,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    repeats: int,
) -> tuple[list[float], np.ndarray]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    warmup_predictions = np.asarray(predict_many_fn(user_ids, item_ids, clip=False), dtype=np.float64)
    timings: list[float] = []
    final_predictions = warmup_predictions

    for _ in range(repeats):
        started = perf_counter()
        final_predictions = np.asarray(predict_many_fn(user_ids, item_ids, clip=False), dtype=np.float64)
        timings.append(perf_counter() - started)
    return timings, final_predictions


def summarize_inference_variant(query_rows: int, timings: list[float]) -> dict[str, Any]:
    return {
        "aggregate": {
            "inference_wall_clock_seconds": summarize_scalar_samples(timings),
            "ratings_per_second_inference": summarize_scalar_samples([query_rows / value for value in timings]),
        }
    }
