from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

KERNEL_PROFILE_VERSION = "kernel_cost_anatomy_v1"

DEFAULT_TOUCH_FACTORS: dict[str, float] = {
    "base_rating_touch_factor": 4.0,
    "implicit_touch_factor": 2.0,
    "explicit_touch_factor": 2.0,
    "cluster_touch_factor": 2.0,
}


def build_kernel_profile_payload(
    *,
    run_id: str,
    dataset: str,
    model: str,
    epochs: int,
    latent_dim: int,
    train_rows: int,
    train_user_ids: np.ndarray,
    epoch_durations_seconds: list[float] | tuple[float, ...] | np.ndarray | None,
    fit_artifacts: object | None = None,
    touch_factors: Mapping[str, float] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    epoch_durations = _float_list(epoch_durations_seconds)
    factors = _touch_factors(touch_factors)
    implicit_index = getattr(fit_artifacts, "user_history_index", None)
    explicit_index = getattr(fit_artifacts, "explicit_feedback_index", None)
    cluster_index = getattr(fit_artifacts, "user_cluster_history_index", None)
    kernel_work = estimate_kernel_work(
        train_user_ids=train_user_ids,
        train_rows=train_rows,
        epochs=epochs,
        latent_dim=latent_dim,
        implicit_history_index=implicit_index,
        explicit_feedback_index=explicit_index,
        cluster_history_index=cluster_index,
        touch_factors=factors,
    )
    total_fit_seconds = float(sum(epoch_durations))
    profile_notes = _kernel_profile_notes(
        epoch_durations=epoch_durations,
        epochs=epochs,
        supplied_notes=notes,
    )
    return {
        "profile_version": KERNEL_PROFILE_VERSION,
        "run_id": str(run_id),
        "dataset": str(dataset),
        "model": str(model),
        "epochs": int(epochs),
        "latent_dim": int(latent_dim),
        "train_rows": int(train_rows),
        "epoch_durations_seconds": epoch_durations,
        "ratings_per_second_by_epoch": ratings_per_second_by_epoch(
            epoch_durations_seconds=epoch_durations,
            train_rows=train_rows,
        ),
        "history_structure": {
            "implicit": summarize_history_index(implicit_index),
            "explicit": summarize_history_index(explicit_index),
            "cluster": summarize_history_index(cluster_index),
        },
        "estimated_kernel_work": kernel_work,
        "cost_ratios": {
            "fit_seconds_per_epoch_mean": _safe_divide(total_fit_seconds, len(epoch_durations)),
            "fit_seconds_per_million_ratings": _safe_divide(
                total_fit_seconds,
                float(train_rows) * float(epochs) / 1_000_000.0,
            ),
            "fit_seconds_per_million_estimated_factor_touches": _safe_divide(
                total_fit_seconds,
                float(kernel_work["estimated_factor_touches"]) / 1_000_000.0,
            ),
        },
        "notes": profile_notes,
    }


def summarize_history_index(history_index: object | None) -> dict[str, Any]:
    if history_index is None:
        return {}

    indptr = getattr(history_index, "indptr", None)
    if indptr is None:
        raise TypeError("history index must expose an indptr array")
    lengths = _history_lengths_from_indptr(indptr)
    if lengths.size == 0:
        return _empty_history_summary()

    return {
        "users_with_history": int((lengths > 0).sum()),
        "total_edges": int(lengths.sum(dtype=np.int64)),
        "mean_len": float(lengths.mean()),
        "p50_len": float(np.percentile(lengths, 50)),
        "p90_len": float(np.percentile(lengths, 90)),
        "p95_len": float(np.percentile(lengths, 95)),
        "max_len": int(lengths.max()),
    }


def estimate_kernel_work(
    *,
    train_user_ids: np.ndarray,
    train_rows: int,
    epochs: int,
    latent_dim: int,
    implicit_history_index: object | None = None,
    explicit_feedback_index: object | None = None,
    cluster_history_index: object | None = None,
    touch_factors: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    factors = _touch_factors(touch_factors)
    train_users = np.asarray(train_user_ids, dtype=np.int64)
    train_rows_value = int(train_rows)
    epochs_value = int(epochs)
    latent_dim_value = int(latent_dim)

    implicit_visits_per_epoch = _train_row_weighted_history_visits(train_users, implicit_history_index)
    explicit_visits_per_epoch = _train_row_weighted_history_visits(train_users, explicit_feedback_index)
    cluster_visits_per_epoch = _train_row_weighted_history_visits(train_users, cluster_history_index)

    rating_updates = train_rows_value * epochs_value
    implicit_history_visits = implicit_visits_per_epoch * epochs_value
    explicit_history_visits = explicit_visits_per_epoch * epochs_value
    cluster_history_visits = cluster_visits_per_epoch * epochs_value
    estimated_factor_touches = int(
        round(
            latent_dim_value
            * (
                factors["base_rating_touch_factor"] * train_rows_value * epochs_value
                + factors["implicit_touch_factor"] * implicit_visits_per_epoch * epochs_value
                + factors["explicit_touch_factor"] * explicit_visits_per_epoch * epochs_value
                + factors["cluster_touch_factor"] * cluster_visits_per_epoch * epochs_value
            )
        )
    )

    return {
        "rating_updates": int(rating_updates),
        "implicit_history_visits": int(implicit_history_visits),
        "explicit_history_visits": int(explicit_history_visits),
        "cluster_history_visits": int(cluster_history_visits),
        "estimated_factor_touches": estimated_factor_touches,
        "rating_updates_per_epoch": train_rows_value,
        "implicit_history_visits_per_epoch": int(implicit_visits_per_epoch),
        "explicit_history_visits_per_epoch": int(explicit_visits_per_epoch),
        "cluster_history_visits_per_epoch": int(cluster_visits_per_epoch),
        "touch_factors": dict(factors),
    }


def ratings_per_second_by_epoch(
    *,
    epoch_durations_seconds: list[float] | tuple[float, ...] | np.ndarray | None,
    train_rows: int,
) -> list[float]:
    rows = int(train_rows)
    rates: list[float] = []
    for duration in _float_list(epoch_durations_seconds):
        rates.append(_safe_divide(float(rows), duration))
    return rates


def _train_row_weighted_history_visits(train_user_ids: np.ndarray, history_index: object | None) -> int:
    if history_index is None:
        return 0
    lengths = _history_lengths_from_indptr(getattr(history_index, "indptr", None))
    if train_user_ids.size == 0:
        return 0
    if int(train_user_ids.max()) >= int(lengths.shape[0]) or int(train_user_ids.min()) < 0:
        raise ValueError("train_user_ids contain user ids outside the history index range")
    return int(lengths[train_user_ids].sum(dtype=np.int64))


def _history_lengths_from_indptr(indptr: object) -> np.ndarray:
    if indptr is None:
        raise TypeError("history index must expose an indptr array")
    values = np.asarray(indptr, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("history index indptr must be 1D")
    if values.size == 0:
        raise ValueError("history index indptr must not be empty")
    if values.size == 1:
        return np.zeros(0, dtype=np.int64)
    lengths = np.diff(values)
    if np.any(lengths < 0):
        raise ValueError("history index indptr must be monotonic nondecreasing")
    return lengths.astype(np.int64, copy=False)


def _empty_history_summary() -> dict[str, Any]:
    return {
        "users_with_history": 0,
        "total_edges": 0,
        "mean_len": 0.0,
        "p50_len": 0.0,
        "p90_len": 0.0,
        "p95_len": 0.0,
        "max_len": 0,
    }


def _touch_factors(touch_factors: Mapping[str, float] | None) -> dict[str, float]:
    factors = dict(DEFAULT_TOUCH_FACTORS)
    if touch_factors is not None:
        factors.update({str(key): float(value) for key, value in touch_factors.items()})
    for key, value in factors.items():
        if value < 0.0:
            raise ValueError(f"{key} must be nonnegative")
    return factors


def _float_list(values: list[float] | tuple[float, ...] | np.ndarray | None) -> list[float]:
    if values is None:
        return []
    return [float(value) for value in values]


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _kernel_profile_notes(
    *,
    epoch_durations: list[float],
    epochs: int,
    supplied_notes: list[str] | None,
) -> list[str]:
    notes = [] if supplied_notes is None else [str(note) for note in supplied_notes]
    notes.append(
        "estimated_factor_touches is a structural heuristic for comparable kernel anatomy, "
        "not a CPU instruction count"
    )
    notes.append(
        "history visit counts are train-row-weighted across all epochs; per-epoch counts are included for audit"
    )
    if len(epoch_durations) != int(epochs):
        notes.append(
            "epoch_durations_seconds length differs from configured epochs; profile uses recorded durations for "
            "timing ratios"
        )
    return notes
