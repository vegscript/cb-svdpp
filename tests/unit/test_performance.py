from __future__ import annotations

import numpy as np
import pytest

from recsys_lab.experiments.performance import (
    StageProfiler,
    build_performance_profile_payload,
    build_system_metrics,
    estimate_numpy_payload_nbytes,
)


class _SharedArrayHolder:
    def __init__(self, shared_array: np.ndarray) -> None:
        self.first = shared_array
        self.second = shared_array
        self.nested = {"again": shared_array}


def test_estimate_numpy_payload_nbytes_deduplicates_shared_arrays() -> None:
    shared_array = np.zeros((4, 5), dtype=np.float32)
    holder = _SharedArrayHolder(shared_array)

    assert estimate_numpy_payload_nbytes(holder) == shared_array.nbytes


def test_build_system_metrics_uses_training_and_inference_contract() -> None:
    model = _SharedArrayHolder(np.zeros((2, 3), dtype=np.float32))

    metrics = build_system_metrics(
        train_rows=10,
        epochs=4,
        training_wall_clock_seconds=2.0,
        inference_rows=5,
        inference_wall_clock_seconds=0.5,
        peak_memory_bytes=8 * 1024 * 1024,
        baseline_memory_bytes=2 * 1024 * 1024,
        model=model,
        epoch_durations_seconds=[0.4, 0.5, 0.5, 0.6],
        train_time_total_seconds=3.0,
    )

    assert metrics["train_time_total"] == 3.0
    assert metrics["train_time_per_epoch"] == 0.5
    assert metrics["train_ratings_processed"] == 40
    assert metrics["ratings_per_second_train"] == 20.0
    assert metrics["ratings_per_second_inference"] == 10.0
    assert metrics["peak_memory_mb"] == 8.0
    assert metrics["peak_memory_delta_mb"] == 6.0
    assert metrics["model_size_mb"] > 0.0
    assert metrics["epoch_durations_seconds"] == [0.4, 0.5, 0.5, 0.6]


def test_stage_profiler_records_completed_stage() -> None:
    profiler = StageProfiler()

    with profiler.stage("synthetic_stage", metadata={"input": "toy"}) as metadata:
        metadata["rows"] = 12

    payload = profiler.to_payload()
    stage = payload["stages"][0]

    assert payload["profile_version"] == "stage_profile_v1"
    assert payload["stage_count"] == 1
    assert payload["total_profiled_wall_clock_seconds"] >= 0.0
    assert stage["name"] == "synthetic_stage"
    assert stage["status"] == "completed"
    assert stage["wall_clock_seconds"] >= 0.0
    assert stage["rss_start_mb"] > 0.0
    assert stage["rss_end_mb"] > 0.0
    assert stage["metadata"] == {"input": "toy", "rows": 12}


def test_stage_profiler_rejects_empty_stage_name() -> None:
    profiler = StageProfiler()

    with pytest.raises(ValueError, match="stage name must not be empty"):
        with profiler.stage(""):
            pass


def test_stage_profiler_records_failed_stage_and_reraises() -> None:
    profiler = StageProfiler()

    with pytest.raises(RuntimeError, match="synthetic failure"):
        with profiler.stage("failing_stage"):
            raise RuntimeError("synthetic failure")

    payload = profiler.to_payload()
    stage = payload["stages"][0]
    assert stage["name"] == "failing_stage"
    assert stage["status"] == "failed"
    assert stage["exception_type"] == "RuntimeError"
    profile = build_performance_profile_payload(
        stage_profile=payload,
        run_id="failed-run",
        dataset="ml_latest_small",
        model="biased_mf",
        device_profile="local_test",
        split_family="benchmark_random_v1",
        split_seed=1,
        model_seed=1,
    )
    assert profile["stages"][0]["status"] == "failed"
    assert profile["stages"][0]["exception_type"] == "RuntimeError"


def test_stage_profiler_payload_has_consistent_stage_count() -> None:
    profiler = StageProfiler()

    with profiler.stage("first_stage"):
        pass
    with profiler.stage("second_stage"):
        pass

    payload = profiler.to_payload()
    assert payload["stage_count"] == len(payload["stages"])


def test_performance_profile_hotspots_are_sorted() -> None:
    profile = build_performance_profile_payload(
        stage_profile={
            "profile_version": "stage_profile_v1",
            "stage_count": 3,
            "total_profiled_wall_clock_seconds": 6.0,
            "stages": [
                {
                    "name": "short_stage",
                    "status": "completed",
                    "wall_clock_seconds": 1.0,
                    "rss_start_mb": 10.0,
                    "rss_end_mb": 11.0,
                    "rss_delta_mb": 1.0,
                    "metadata": {},
                },
                {
                    "name": "main_training",
                    "status": "completed",
                    "wall_clock_seconds": 4.0,
                    "rss_start_mb": 11.0,
                    "rss_end_mb": 13.0,
                    "rss_delta_mb": 2.0,
                    "metadata": {},
                },
                {
                    "name": "medium_stage",
                    "status": "completed",
                    "wall_clock_seconds": 2.0,
                    "rss_start_mb": 13.0,
                    "rss_end_mb": 14.0,
                    "rss_delta_mb": 1.0,
                    "metadata": {},
                },
            ],
        },
        run_id="synthetic",
        dataset="ml_latest_small",
        model="biased_mf",
        device_profile="local_test",
        split_family="benchmark_random_v1",
        split_seed=1,
        model_seed=2,
    )

    assert profile["profile_version"] == "performance_forensics_v1"
    assert profile["stage_count"] == len(profile["stages"])
    assert [stage["name"] for stage in profile["stages"]] == [
        "short_stage",
        "fit_model",
        "medium_stage",
    ]
    hotspot_seconds = [hotspot["wall_clock_seconds"] for hotspot in profile["hotspots"]]
    assert hotspot_seconds == sorted(hotspot_seconds, reverse=True)
    assert profile["hotspots"][0]["name"] == "fit_model"
