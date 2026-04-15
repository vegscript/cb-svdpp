from __future__ import annotations

import numpy as np

from recsys_lab.experiments.performance import build_system_metrics, estimate_numpy_payload_nbytes


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
