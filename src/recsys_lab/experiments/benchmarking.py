from __future__ import annotations

import statistics
from typing import Any


def summarize_scalar_samples(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize empty value list")

    mean = float(statistics.fmean(values))
    std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    coefficient_of_variation = 0.0 if mean == 0.0 else float(abs(std / mean))
    return {
        "count": len(values),
        "mean": mean,
        "std": std,
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "coefficient_of_variation": coefficient_of_variation,
    }


def build_benchmark_measurement(
    *,
    time_metric: str,
    time_metric_semantics: str,
    sample_unit: str,
    measured_sample_count: int,
    warmup_policy: str = "none",
    warmup_sample_count: int = 0,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    if measured_sample_count < 1:
        raise ValueError("measured_sample_count must be at least 1")
    if warmup_sample_count < 0:
        raise ValueError("warmup_sample_count must be non-negative")
    if warmup_policy not in {"none", "separate_unmeasured", "included_in_measurement"}:
        raise ValueError(f"unsupported warmup_policy: {warmup_policy}")

    payload: dict[str, Any] = {
        "time_metric": time_metric,
        "time_metric_unit": "seconds",
        "time_metric_semantics": time_metric_semantics,
        "sample_unit": sample_unit,
        "warmup_policy": warmup_policy,
        "warmup_sample_count": warmup_sample_count,
        "measured_sample_count": measured_sample_count,
    }
    if notes:
        payload["notes"] = [str(note) for note in notes]
    return payload
