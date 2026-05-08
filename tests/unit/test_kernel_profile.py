from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from recsys_lab.data.histories import UserExplicitFeedbackIndex, UserHistoryIndex
from recsys_lab.experiments.kernel_profile import (
    build_kernel_profile_payload,
    estimate_kernel_work,
    ratings_per_second_by_epoch,
    summarize_history_index,
)


def _history_index(lengths: list[int]) -> UserHistoryIndex:
    counts = np.asarray(lengths, dtype=np.int32)
    indptr = np.zeros(counts.shape[0] + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(counts, dtype=np.int64).astype(np.int32, copy=False)
    return UserHistoryIndex(
        indptr=indptr,
        item_indices=np.arange(int(counts.sum()), dtype=np.int32),
        counts=counts,
        norms=np.ones(counts.shape[0], dtype=np.float32),
    )


def _explicit_index(lengths: list[int]) -> UserExplicitFeedbackIndex:
    history = _history_index(lengths)
    return UserExplicitFeedbackIndex(
        indptr=history.indptr,
        item_indices=history.item_indices,
        ratings=np.ones(history.item_indices.shape[0], dtype=np.float32),
        counts=history.counts,
        norms=history.norms,
    )


def test_summarize_history_index_empty() -> None:
    summary = summarize_history_index(_history_index([]))

    assert summary == {
        "users_with_history": 0,
        "total_edges": 0,
        "mean_len": 0.0,
        "p50_len": 0.0,
        "p90_len": 0.0,
        "p95_len": 0.0,
        "max_len": 0,
    }


def test_summarize_history_index_non_empty() -> None:
    summary = summarize_history_index(_history_index([2, 0, 3]))

    assert summary["users_with_history"] == 2
    assert summary["total_edges"] == 5
    assert summary["mean_len"] == pytest.approx(5 / 3)
    assert summary["p50_len"] == 2.0
    assert summary["p90_len"] == pytest.approx(2.8)
    assert summary["p95_len"] == pytest.approx(2.9)
    assert summary["max_len"] == 3


def test_estimate_kernel_work_biased_mf_has_no_history_visits() -> None:
    work = estimate_kernel_work(
        train_user_ids=np.asarray([0, 1, 0, 2], dtype=np.int64),
        train_rows=4,
        epochs=3,
        latent_dim=5,
    )

    assert work["rating_updates"] == 12
    assert work["implicit_history_visits"] == 0
    assert work["explicit_history_visits"] == 0
    assert work["cluster_history_visits"] == 0
    assert work["estimated_factor_touches"] == 5 * 4.0 * 4 * 3


def test_estimate_kernel_work_svdpp_counts_implicit_visits() -> None:
    work = estimate_kernel_work(
        train_user_ids=np.asarray([0, 1, 0], dtype=np.int64),
        train_rows=3,
        epochs=2,
        latent_dim=4,
        implicit_history_index=_history_index([2, 1]),
    )

    assert work["rating_updates"] == 6
    assert work["implicit_history_visits_per_epoch"] == 5
    assert work["implicit_history_visits"] == 10
    assert work["explicit_history_visits"] == 0
    assert work["cluster_history_visits"] == 0
    assert work["estimated_factor_touches"] == 4 * ((4.0 * 3 * 2) + (2.0 * 5 * 2))


def test_estimate_kernel_work_asvdpp_counts_explicit_and_implicit_visits() -> None:
    work = estimate_kernel_work(
        train_user_ids=np.asarray([0, 1, 2, 0], dtype=np.int64),
        train_rows=4,
        epochs=2,
        latent_dim=3,
        implicit_history_index=_history_index([2, 1, 0]),
        explicit_feedback_index=_explicit_index([1, 3, 2]),
    )

    assert work["implicit_history_visits_per_epoch"] == 5
    assert work["implicit_history_visits"] == 10
    assert work["explicit_history_visits_per_epoch"] == 7
    assert work["explicit_history_visits"] == 14
    assert work["cluster_history_visits"] == 0


def test_kernel_profile_payload_has_required_fields() -> None:
    fit_artifacts = SimpleNamespace(
        user_history_index=_history_index([2, 1]),
        explicit_feedback_index=None,
        user_cluster_history_index=None,
    )

    payload = build_kernel_profile_payload(
        run_id="run-1",
        dataset="ml_latest_small",
        model="svdpp",
        epochs=2,
        latent_dim=4,
        train_rows=3,
        train_user_ids=np.asarray([0, 1, 0], dtype=np.int64),
        epoch_durations_seconds=[0.5, 1.0],
        fit_artifacts=fit_artifacts,
    )

    assert payload["profile_version"] == "kernel_cost_anatomy_v1"
    assert payload["run_id"] == "run-1"
    assert payload["dataset"] == "ml_latest_small"
    assert payload["model"] == "svdpp"
    assert payload["epochs"] == 2
    assert payload["latent_dim"] == 4
    assert payload["train_rows"] == 3
    assert payload["epoch_durations_seconds"] == [0.5, 1.0]
    assert payload["ratings_per_second_by_epoch"] == [6.0, 3.0]
    assert payload["history_structure"]["implicit"]["total_edges"] == 3
    assert payload["history_structure"]["explicit"] == {}
    assert payload["estimated_kernel_work"]["implicit_history_visits"] == 10
    assert payload["cost_ratios"]["fit_seconds_per_epoch_mean"] == 0.75
    assert payload["notes"]


def test_ratings_per_second_by_epoch() -> None:
    assert ratings_per_second_by_epoch(epoch_durations_seconds=[0.5, 2.0, 0.0], train_rows=10) == [
        20.0,
        5.0,
        0.0,
    ]
