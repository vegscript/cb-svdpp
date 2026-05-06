from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KERNEL_COST_COLUMNS = (
    "dataset",
    "model",
    "run_id",
    "epochs",
    "latent_dim",
    "train_rows",
    "fit_seconds_total",
    "fit_seconds_per_epoch_mean",
    "implicit_history_visits",
    "explicit_history_visits",
    "cluster_history_visits",
    "estimated_factor_touches",
    "fit_seconds_per_million_estimated_factor_touches",
)


@dataclass(frozen=True, slots=True)
class KernelCostRow:
    dataset: str
    model: str
    run_id: str
    epochs: int
    latent_dim: int
    train_rows: int
    fit_seconds_total: float
    fit_seconds_per_epoch_mean: float
    implicit_history_visits: int
    explicit_history_visits: int
    cluster_history_visits: int
    estimated_factor_touches: int
    fit_seconds_per_million_estimated_factor_touches: float

    def table_payload(self) -> dict[str, Any]:
        return {column: getattr(self, column) for column in KERNEL_COST_COLUMNS}


def collect_kernel_profile_rows(runs_dir: Path) -> list[KernelCostRow]:
    rows = [_kernel_cost_row(_load_json(profile_path)) for profile_path in _iter_kernel_profile_paths(runs_dir)]
    return sorted(rows, key=lambda row: (row.dataset, row.model, row.run_id))


def write_kernel_profile_report(
    *,
    runs_dir: Path,
    output_dir: Path,
) -> Path:
    rows = collect_kernel_profile_rows(runs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "kernel_cost_anatomy.csv"
    _write_csv(rows, columns=KERNEL_COST_COLUMNS, output_path=output_path)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect kernel cost anatomy CSV report from run artifacts.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("artifacts/runs"),
        help="Directory containing run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports"),
        help="Directory for generated CSV reports.",
    )
    args = parser.parse_args(argv)

    path = write_kernel_profile_report(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
    )
    print(path.as_posix())
    return 0


def _iter_kernel_profile_paths(runs_dir: Path) -> Iterable[Path]:
    if not runs_dir.exists():
        return []
    return (
        run_dir / "kernel_profile.json"
        for run_dir in sorted(runs_dir.iterdir())
        if run_dir.is_dir() and (run_dir / "kernel_profile.json").is_file()
    )


def _kernel_cost_row(profile: dict[str, Any]) -> KernelCostRow:
    estimated_work = _mapping(profile.get("estimated_kernel_work"))
    cost_ratios = _mapping(profile.get("cost_ratios"))
    epoch_durations = _numbers(profile.get("epoch_durations_seconds"))
    return KernelCostRow(
        dataset=_string(profile.get("dataset")),
        model=_string(profile.get("model")),
        run_id=_string(profile.get("run_id")),
        epochs=_integer(profile.get("epochs")),
        latent_dim=_integer(profile.get("latent_dim")),
        train_rows=_integer(profile.get("train_rows")),
        fit_seconds_total=float(sum(epoch_durations)),
        fit_seconds_per_epoch_mean=_number(cost_ratios.get("fit_seconds_per_epoch_mean")),
        implicit_history_visits=_integer(estimated_work.get("implicit_history_visits")),
        explicit_history_visits=_integer(estimated_work.get("explicit_history_visits")),
        cluster_history_visits=_integer(estimated_work.get("cluster_history_visits")),
        estimated_factor_touches=_integer(estimated_work.get("estimated_factor_touches")),
        fit_seconds_per_million_estimated_factor_touches=_number(
            cost_ratios.get("fit_seconds_per_million_estimated_factor_touches")
        ),
    )


def _write_csv(rows: Sequence[KernelCostRow], *, columns: Sequence[str], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.table_payload())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _numbers(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    return [_number(item) for item in value]


def _number(value: Any) -> float:
    if value is None or value == "" or isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _integer(value: Any) -> int:
    return int(_number(value))


def _string(value: Any) -> str:
    return "" if value is None else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
