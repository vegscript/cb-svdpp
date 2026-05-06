from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STAGE_BREAKDOWN_COLUMNS = (
    "dataset",
    "model",
    "run_id",
    "stage_name",
    "wall_clock_seconds",
    "share_of_profiled_time",
    "rss_start_mb",
    "rss_end_mb",
    "rss_delta_mb",
    "status",
)

HOTSPOT_COLUMNS = (
    "dataset",
    "model",
    "run_id",
    "top_1_stage",
    "top_1_seconds",
    "top_2_stage",
    "top_2_seconds",
    "top_3_stage",
    "top_3_seconds",
    "total_profiled_wall_clock_seconds",
)


@dataclass(frozen=True, slots=True)
class StageBreakdownRow:
    dataset: str
    model: str
    run_id: str
    stage_name: str
    wall_clock_seconds: float
    share_of_profiled_time: float
    rss_start_mb: float
    rss_end_mb: float
    rss_delta_mb: float
    status: str

    def table_payload(self) -> dict[str, Any]:
        return {column: getattr(self, column) for column in STAGE_BREAKDOWN_COLUMNS}


@dataclass(frozen=True, slots=True)
class HotspotRow:
    dataset: str
    model: str
    run_id: str
    top_1_stage: str | None
    top_1_seconds: float | None
    top_2_stage: str | None
    top_2_seconds: float | None
    top_3_stage: str | None
    top_3_seconds: float | None
    total_profiled_wall_clock_seconds: float

    def table_payload(self) -> dict[str, Any]:
        return {column: getattr(self, column) for column in HOTSPOT_COLUMNS}


def collect_performance_profile_rows(
    runs_dir: Path,
) -> tuple[list[StageBreakdownRow], list[HotspotRow]]:
    stage_rows: list[StageBreakdownRow] = []
    hotspot_rows: list[HotspotRow] = []
    for profile_path in _iter_performance_profile_paths(runs_dir):
        profile = _load_json(profile_path)
        stage_rows.extend(_stage_breakdown_rows(profile))
        hotspot_rows.append(_hotspot_row(profile))
    return (
        sorted(stage_rows, key=lambda row: (row.dataset, row.model, row.run_id, row.stage_name)),
        sorted(hotspot_rows, key=lambda row: (row.dataset, row.model, row.run_id)),
    )


def write_performance_profile_reports(
    *,
    runs_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    stage_rows, hotspot_rows = collect_performance_profile_rows(runs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_path = output_dir / "performance_stage_breakdown.csv"
    hotspot_path = output_dir / "performance_hotspots.csv"
    _write_csv(stage_rows, columns=STAGE_BREAKDOWN_COLUMNS, output_path=stage_path)
    _write_csv(hotspot_rows, columns=HOTSPOT_COLUMNS, output_path=hotspot_path)
    return {
        "stage_breakdown": stage_path,
        "hotspots": hotspot_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect performance profile CSV reports from run artifacts.")
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

    paths = write_performance_profile_reports(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
    )
    for path in paths.values():
        print(path.as_posix())
    return 0


def _iter_performance_profile_paths(runs_dir: Path) -> Iterable[Path]:
    if not runs_dir.exists():
        return []
    return (
        run_dir / "performance_profile.json"
        for run_dir in sorted(runs_dir.iterdir())
        if run_dir.is_dir() and (run_dir / "performance_profile.json").is_file()
    )


def _stage_breakdown_rows(profile: dict[str, Any]) -> list[StageBreakdownRow]:
    dataset = _string(profile.get("dataset"))
    model = _string(profile.get("model"))
    run_id = _string(profile.get("run_id"))
    total_seconds = _number(profile.get("total_profiled_wall_clock_seconds")) or 0.0
    rows: list[StageBreakdownRow] = []
    for stage in profile.get("stages", []):
        if not isinstance(stage, dict):
            continue
        stage_seconds = _number(stage.get("wall_clock_seconds")) or 0.0
        rows.append(
            StageBreakdownRow(
                dataset=dataset,
                model=model,
                run_id=run_id,
                stage_name=_string(stage.get("name")),
                wall_clock_seconds=stage_seconds,
                share_of_profiled_time=stage_seconds / total_seconds if total_seconds > 0.0 else 0.0,
                rss_start_mb=_number(stage.get("rss_start_mb")) or 0.0,
                rss_end_mb=_number(stage.get("rss_end_mb")) or 0.0,
                rss_delta_mb=_number(stage.get("rss_delta_mb")) or 0.0,
                status=_string(stage.get("status")),
            )
        )
    return rows


def _hotspot_row(profile: dict[str, Any]) -> HotspotRow:
    hotspots = profile.get("hotspots", [])
    top_hotspots = [hotspot for hotspot in hotspots[:3] if isinstance(hotspot, dict)]
    while len(top_hotspots) < 3:
        top_hotspots.append({})
    return HotspotRow(
        dataset=_string(profile.get("dataset")),
        model=_string(profile.get("model")),
        run_id=_string(profile.get("run_id")),
        top_1_stage=_optional_string(top_hotspots[0].get("name")),
        top_1_seconds=_optional_number(top_hotspots[0].get("wall_clock_seconds")),
        top_2_stage=_optional_string(top_hotspots[1].get("name")),
        top_2_seconds=_optional_number(top_hotspots[1].get("wall_clock_seconds")),
        top_3_stage=_optional_string(top_hotspots[2].get("name")),
        top_3_seconds=_optional_number(top_hotspots[2].get("wall_clock_seconds")),
        total_profiled_wall_clock_seconds=_number(profile.get("total_profiled_wall_clock_seconds")) or 0.0,
    )


def _write_csv(rows: Sequence[Any], *, columns: Sequence[str], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(value) for key, value in row.table_payload().items()})


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_number(value: Any) -> float | None:
    return _number(value)


def _string(value: Any) -> str:
    return "" if value is None else str(value)


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _csv_cell(value: Any) -> Any:
    return "" if value is None else value


if __name__ == "__main__":
    raise SystemExit(main())
