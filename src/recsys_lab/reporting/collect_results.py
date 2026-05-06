from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

RESULT_COLUMNS = (
    "dataset",
    "model",
    "validation_rmse",
    "test_rmse",
    "validation_mae",
    "test_mae",
    "abs_error_p90",
    "prediction_out_of_range_rate",
    "train_time",
    "peak_memory",
    "alpha",
    "cb_claim_eligible",
    "diagnostic_claim_ready",
    "status",
)

DATASET_ORDER = {
    "ml100k": 0,
    "ml1m": 1,
    "ml10m": 2,
    "ml20m": 3,
}

MODEL_ORDER = {
    "biased_mf": 0,
    "svdpp": 1,
    "asymmetric_svd": 2,
    "asvdpp": 3,
    "cb_svdpp": 4,
    "cb_asvdpp": 5,
}


@dataclass(frozen=True)
class ResultRow:
    dataset: str
    model: str
    validation_rmse: float | None
    test_rmse: float | None
    validation_mae: float | None
    test_mae: float | None
    abs_error_p90: float | None
    prediction_out_of_range_rate: float | None
    train_time: float | None
    peak_memory: float | None
    alpha: float | None
    cb_claim_eligible: bool | None
    diagnostic_claim_ready: bool | None
    status: str
    run_id: str
    generated_at_utc: str

    def table_payload(self) -> dict[str, Any]:
        return {column: getattr(self, column) for column in RESULT_COLUMNS}


def collect_result_rows(
    runs_dir: Path,
    *,
    latest_per_dataset_model: bool = True,
    datasets: set[str] | None = None,
) -> list[ResultRow]:
    rows = [
        row
        for run_dir in sorted(runs_dir.iterdir())
        if run_dir.is_dir() and (row := result_row_from_run_dir(run_dir)) is not None
    ]
    if datasets is not None:
        rows = [row for row in rows if row.dataset in datasets]
    if latest_per_dataset_model:
        rows = latest_rows(rows)
    return sorted(rows, key=_row_sort_key)


def result_row_from_run_dir(run_dir: Path) -> ResultRow | None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return None

    manifest = _load_json(manifest_path)
    metrics_path = run_dir / "metrics.json"
    metrics = _load_json(metrics_path) if metrics_path.is_file() else None
    metric_values = metrics.get("metrics", {}) if isinstance(metrics, dict) else {}

    dataset = _string(_get(manifest, "dataset.short_name") or _get(metrics, "dataset.short_name"))
    model = _string(_get(manifest, "model.name") or _get(metrics, "model.name"))
    manifest_status = _string(manifest.get("status") or "unknown")
    status = manifest_status if metrics is not None else f"{manifest_status}_missing_metrics"

    return ResultRow(
        dataset=dataset,
        model=model,
        validation_rmse=_number(_metric(metric_values, "validation", "rmse")),
        test_rmse=_number(_metric(metric_values, "test", "rmse")),
        validation_mae=_number(_metric(metric_values, "validation", "mae")),
        test_mae=_number(_metric(metric_values, "test", "mae")),
        abs_error_p90=_number(_preferred_metric(metric_values, "abs_error_p90")),
        prediction_out_of_range_rate=_number(_preferred_metric(metric_values, "prediction_out_of_range_rate")),
        train_time=_number(_train_time(metrics, manifest)),
        peak_memory=_number(_peak_memory(metrics, manifest)),
        alpha=_number(
            _get(metrics, "cb_diagnostics.alpha")
            or _get(metrics, "cb_semantics.alpha")
            or _get(manifest, "cb_semantics.alpha")
        ),
        cb_claim_eligible=_bool(
            _get(metrics, "cb_diagnostics.cb_claim_eligible")
            if metrics is not None and _get(metrics, "cb_diagnostics.cb_claim_eligible") is not None
            else _get(metrics, "cb_semantics.cb_claim_eligible")
            if metrics is not None and _get(metrics, "cb_semantics.cb_claim_eligible") is not None
            else _get(manifest, "cb_semantics.cb_claim_eligible")
        ),
        diagnostic_claim_ready=_bool(_get(metrics, "cb_diagnostics.diagnostic_claim_ready")),
        status=status,
        run_id=_string(manifest.get("run_id") or run_dir.name),
        generated_at_utc=_string(manifest.get("generated_at_utc") or run_dir.name),
    )


def latest_rows(rows: Iterable[ResultRow]) -> list[ResultRow]:
    latest: dict[tuple[str, str], ResultRow] = {}
    for row in rows:
        key = (row.dataset, row.model)
        current = latest.get(key)
        if current is None or (row.generated_at_utc, row.run_id) > (current.generated_at_utc, current.run_id):
            latest[key] = row
    return list(latest.values())


def render_markdown_table(rows: Sequence[ResultRow]) -> str:
    lines = [
        "| " + " | ".join(RESULT_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in RESULT_COLUMNS) + " |",
    ]
    for row in rows:
        payload = row.table_payload()
        lines.append("| " + " | ".join(_format_cell(payload[column]) for column in RESULT_COLUMNS) + " |")
    return "\n".join(lines)


def write_csv(rows: Sequence[ResultRow], output: TextIO) -> None:
    writer = csv.DictWriter(output, fieldnames=list(RESULT_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_cell(value) for key, value in row.table_payload().items()})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect report result tables from run artifacts.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("artifacts/runs"),
        help="Directory containing run folders.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Emit every manifest-backed run instead of the latest run per dataset/model.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Restrict output to a dataset. Can be passed multiple times.",
    )
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    args = parser.parse_args(argv)

    rows = collect_result_rows(
        args.runs_dir,
        latest_per_dataset_model=not args.all_runs,
        datasets=set(args.datasets) if args.datasets else None,
    )
    if args.format == "csv":
        write_csv(rows, sys.stdout)
    else:
        print(render_markdown_table(rows))
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get(payload: Any, dotted_path: str) -> Any:
    current = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _metric(metric_values: dict[str, Any], split: str, metric_name: str) -> Any:
    nested = _get(metric_values, f"{split}.{metric_name}")
    if nested is not None:
        return nested
    return metric_values.get(f"{split}_{metric_name}")


def _preferred_metric(metric_values: dict[str, Any], metric_name: str) -> Any:
    test_value = _metric(metric_values, "test", metric_name)
    if test_value is not None:
        return test_value
    return _metric(metric_values, "validation", metric_name)


def _train_time(metrics: dict[str, Any] | None, manifest: dict[str, Any]) -> Any:
    if metrics is None:
        return None
    return (
        _get(metrics, "system_metrics.train_time_total")
        or _get(metrics, "timing.training_wall_clock_seconds")
        or _get(manifest, "timing.training_wall_clock_seconds")
        or _profiling_stage_seconds(metrics, "main_training")
        or _get(metrics, "profiling.total_profiled_wall_clock_seconds")
    )


def _peak_memory(metrics: dict[str, Any] | None, manifest: dict[str, Any]) -> Any:
    if metrics is None:
        return None
    return (
        _get(metrics, "system_metrics.peak_memory_mb")
        or _max_stage_value(_get(metrics, "profiling.stages"), "rss_end_mb")
        or _max_stage_value(_get(manifest, "profiling.stages"), "rss_end_mb")
    )


def _profiling_stage_seconds(metrics: dict[str, Any], stage_name: str) -> float | None:
    stages = _get(metrics, "profiling.stages")
    if not isinstance(stages, list):
        return None
    for stage in stages:
        if isinstance(stage, dict) and stage.get("name") == stage_name:
            return _number(stage.get("wall_clock_seconds"))
    return None


def _max_stage_value(stages: Any, field: str) -> float | None:
    if not isinstance(stages, list):
        return None
    values = [_number(stage.get(field)) for stage in stages if isinstance(stage, dict)]
    numeric_values = [value for value in values if value is not None]
    return max(numeric_values) if numeric_values else None


def _row_sort_key(row: ResultRow) -> tuple[int, str, int, str, str]:
    return (
        DATASET_ORDER.get(row.dataset, 999),
        row.dataset,
        MODEL_ORDER.get(row.model, 999),
        row.model,
        row.generated_at_utc,
    )


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _string(value: Any) -> str:
    return "" if value is None else str(value)


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    return value


if __name__ == "__main__":
    raise SystemExit(main())
