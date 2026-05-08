from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

COMPARISON_FIELDS = (
    "model",
    "before_benchmark_id",
    "after_benchmark_id",
    "kernel_name",
    "dataset_profile",
    "dtype",
    "latent_dim",
    "train_rows",
    "epochs_per_repeat",
    "warmup_repeats",
    "timed_repeats",
    "before_mean_wall_seconds",
    "after_mean_wall_seconds",
    "delta_mean_wall_seconds",
    "after_over_before_mean_wall_seconds",
    "before_median_wall_seconds",
    "after_median_wall_seconds",
    "delta_median_wall_seconds",
    "after_over_before_median_wall_seconds",
    "before_ratings_per_second_mean",
    "after_ratings_per_second_mean",
    "delta_ratings_per_second_mean",
    "after_over_before_ratings_per_second_mean",
    "before_seconds_per_million_estimated_factor_touches",
    "after_seconds_per_million_estimated_factor_touches",
    "delta_seconds_per_million_estimated_factor_touches",
    "after_over_before_seconds_per_million_estimated_factor_touches",
    "metadata_match",
)

METADATA_FIELDS = (
    "kernel_name",
    "dataset_profile",
    "dtype",
    "latent_dim",
    "train_rows",
    "epochs_per_repeat",
    "warmup_repeats",
    "timed_repeats",
    "estimated_factor_touches",
    "mutated_array_count",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    before_rows = _read_summary(args.before)
    after_rows = _read_summary(args.after)
    comparison_rows = _compare_rows(before_rows=before_rows, after_rows=after_rows)
    _write_comparison(comparison_rows, args.output)
    print(args.output)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two kernel benchmark summary CSV files.")
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _read_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_model: dict[str, dict[str, str]] = {}
    for row in rows:
        model = row.get("model", "")
        if not model:
            raise ValueError(f"{path} contains a row without model")
        if model in by_model:
            raise ValueError(f"{path} contains duplicate model row: {model}")
        by_model[model] = row
    return by_model


def _compare_rows(
    *,
    before_rows: dict[str, dict[str, str]],
    after_rows: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    if set(before_rows) != set(after_rows):
        missing_after = sorted(set(before_rows) - set(after_rows))
        missing_before = sorted(set(after_rows) - set(before_rows))
        raise ValueError(f"summary model sets differ: missing_after={missing_after}, missing_before={missing_before}")

    rows: list[dict[str, Any]] = []
    for model in sorted(before_rows):
        before = before_rows[model]
        after = after_rows[model]
        rows.append(_compare_model(model=model, before=before, after=after))
    return rows


def _compare_model(
    *,
    model: str,
    before: dict[str, str],
    after: dict[str, str],
) -> dict[str, Any]:
    before_mean = _float(before, "mean_wall_seconds")
    after_mean = _float(after, "mean_wall_seconds")
    before_median = _float(before, "median_wall_seconds")
    after_median = _float(after, "median_wall_seconds")
    before_rps = _float(before, "ratings_per_second_mean")
    after_rps = _float(after, "ratings_per_second_mean")
    before_touch_seconds = _float(before, "seconds_per_million_estimated_factor_touches")
    after_touch_seconds = _float(after, "seconds_per_million_estimated_factor_touches")

    return {
        "model": model,
        "before_benchmark_id": before["benchmark_id"],
        "after_benchmark_id": after["benchmark_id"],
        "kernel_name": after["kernel_name"],
        "dataset_profile": after["dataset_profile"],
        "dtype": after["dtype"],
        "latent_dim": after["latent_dim"],
        "train_rows": after["train_rows"],
        "epochs_per_repeat": after["epochs_per_repeat"],
        "warmup_repeats": after["warmup_repeats"],
        "timed_repeats": after["timed_repeats"],
        "before_mean_wall_seconds": before_mean,
        "after_mean_wall_seconds": after_mean,
        "delta_mean_wall_seconds": after_mean - before_mean,
        "after_over_before_mean_wall_seconds": _ratio(after_mean, before_mean),
        "before_median_wall_seconds": before_median,
        "after_median_wall_seconds": after_median,
        "delta_median_wall_seconds": after_median - before_median,
        "after_over_before_median_wall_seconds": _ratio(after_median, before_median),
        "before_ratings_per_second_mean": before_rps,
        "after_ratings_per_second_mean": after_rps,
        "delta_ratings_per_second_mean": after_rps - before_rps,
        "after_over_before_ratings_per_second_mean": _ratio(after_rps, before_rps),
        "before_seconds_per_million_estimated_factor_touches": before_touch_seconds,
        "after_seconds_per_million_estimated_factor_touches": after_touch_seconds,
        "delta_seconds_per_million_estimated_factor_touches": after_touch_seconds - before_touch_seconds,
        "after_over_before_seconds_per_million_estimated_factor_touches": _ratio(
            after_touch_seconds,
            before_touch_seconds,
        ),
        "metadata_match": all(before[field] == after[field] for field in METADATA_FIELDS),
    }


def _write_comparison(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, str], field: str) -> float:
    return float(row[field])


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
