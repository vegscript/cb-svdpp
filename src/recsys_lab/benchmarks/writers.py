from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

KERNEL_BENCHMARK_JSON = "kernel_benchmark.json"

SUMMARY_FIELDS = (
    "benchmark_id",
    "kernel_name",
    "model",
    "dataset_profile",
    "dtype",
    "latent_dim",
    "train_rows",
    "epochs_per_repeat",
    "warmup_repeats",
    "timed_repeats",
    "mean_wall_seconds",
    "median_wall_seconds",
    "std_wall_seconds",
    "min_wall_seconds",
    "max_wall_seconds",
    "ratings_per_second_mean",
    "estimated_factor_touches",
    "seconds_per_million_estimated_factor_touches",
    "finite_parameters_after",
    "mutated_array_count",
    "claim_boundary",
)


def write_kernel_benchmark_json(payload: dict[str, Any], output_dir: Path) -> Path:
    safe_payload = _json_safe_payload(payload)
    benchmark_id = _benchmark_id(safe_payload)
    output_path = Path(output_dir) / benchmark_id / KERNEL_BENCHMARK_JSON
    _atomic_write_text(
        output_path,
        json.dumps(safe_payload, indent=2, sort_keys=True) + "\n",
    )
    return output_path


def write_kernel_benchmark_summary_csv(payloads: list[dict[str, Any]], output_path: Path) -> Path:
    safe_payloads = [_json_safe_payload(payload) for payload in payloads]
    rows = [_summary_row(payload) for payload in safe_payloads]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)
    return path


def _summary_row(payload: dict[str, Any]) -> dict[str, Any]:
    state_checks = payload.get("state_checks", {})
    if not isinstance(state_checks, dict):
        raise TypeError("payload state_checks must be a mapping")
    row = {field: payload.get(field, "") for field in SUMMARY_FIELDS}
    row["finite_parameters_after"] = state_checks.get("finite_parameters_after", "")
    row["mutated_array_count"] = state_checks.get("mutated_array_count", "")
    return row


def _benchmark_id(payload: dict[str, Any]) -> str:
    benchmark_id = payload.get("benchmark_id")
    if not isinstance(benchmark_id, str) or not benchmark_id:
        raise ValueError("payload benchmark_id must be a non-empty string")
    return benchmark_id


def _json_safe_payload(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        raise TypeError("kernel benchmark payloads must not contain numpy arrays")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_payload(item) for item in value]
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temp_path, path)
