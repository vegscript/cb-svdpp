from __future__ import annotations

import pytest

from recsys_lab.experiments.benchmarking import (
    build_benchmark_measurement,
    summarize_scalar_samples,
)


def test_summarize_scalar_samples_reports_count_median_and_dispersion() -> None:
    summary = summarize_scalar_samples([2.0, 4.0, 6.0])

    assert summary["count"] == 3
    assert summary["mean"] == 4.0
    assert summary["median"] == 4.0
    assert summary["min"] == 2.0
    assert summary["max"] == 6.0
    assert summary["std"] > 0.0
    assert summary["coefficient_of_variation"] > 0.0


def test_build_benchmark_measurement_requires_valid_counts() -> None:
    with pytest.raises(ValueError, match="measured_sample_count"):
        build_benchmark_measurement(
            time_metric="training_wall_clock_seconds",
            time_metric_semantics="test",
            sample_unit="fold_run",
            measured_sample_count=0,
        )

    with pytest.raises(ValueError, match="warmup_sample_count"):
        build_benchmark_measurement(
            time_metric="training_wall_clock_seconds",
            time_metric_semantics="test",
            sample_unit="fold_run",
            measured_sample_count=1,
            warmup_sample_count=-1,
        )


def test_build_benchmark_measurement_returns_machine_readable_contract() -> None:
    payload = build_benchmark_measurement(
        time_metric="training_wall_clock_seconds",
        time_metric_semantics="Fit time in seconds.",
        sample_unit="official_fold_run",
        measured_sample_count=5,
        warmup_policy="none",
        warmup_sample_count=0,
        notes=["Each fold is measured once."],
    )

    assert payload == {
        "time_metric": "training_wall_clock_seconds",
        "time_metric_unit": "seconds",
        "time_metric_semantics": "Fit time in seconds.",
        "sample_unit": "official_fold_run",
        "warmup_policy": "none",
        "warmup_sample_count": 0,
        "measured_sample_count": 5,
        "notes": ["Each fold is measured once."],
    }
