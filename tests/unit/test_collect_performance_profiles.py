from __future__ import annotations

import csv
import json
from pathlib import Path

from recsys_lab.reporting.collect_performance_profiles import (
    collect_performance_profile_rows,
    write_performance_profile_reports,
)


def _write_profile(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "performance_profile.json").write_text(
        json.dumps(
            {
                "profile_version": "performance_forensics_v1",
                "run_id": "2026-05-06T000000Z_ml100k_cb_svdpp_local_s001",
                "dataset": "ml100k",
                "model": "cb_svdpp",
                "device_profile": "local_test",
                "split_family": "benchmark_random_v1",
                "split_seed": 1,
                "model_seed": 1,
                "total_profiled_wall_clock_seconds": 10.0,
                "stage_count": 3,
                "stages": [
                    {
                        "name": "fit_model",
                        "status": "completed",
                        "wall_clock_seconds": 6.0,
                        "rss_start_mb": 100.0,
                        "rss_end_mb": 120.0,
                        "rss_delta_mb": 20.0,
                        "metadata": {},
                    },
                    {
                        "name": "build_cluster_artifacts",
                        "status": "completed",
                        "wall_clock_seconds": 3.0,
                        "rss_start_mb": 90.0,
                        "rss_end_mb": 100.0,
                        "rss_delta_mb": 10.0,
                        "metadata": {},
                    },
                    {
                        "name": "write_performance_profile",
                        "status": "completed",
                        "wall_clock_seconds": 1.0,
                        "rss_start_mb": 120.0,
                        "rss_end_mb": 120.5,
                        "rss_delta_mb": 0.5,
                        "metadata": {},
                    },
                ],
                "hotspots": [
                    {
                        "name": "fit_model",
                        "wall_clock_seconds": 6.0,
                        "share_of_profiled_time": 0.6,
                    },
                    {
                        "name": "build_cluster_artifacts",
                        "wall_clock_seconds": 3.0,
                        "share_of_profiled_time": 0.3,
                    },
                    {
                        "name": "write_performance_profile",
                        "wall_clock_seconds": 1.0,
                        "share_of_profiled_time": 0.1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_collect_performance_profile_rows_flattens_stages_and_hotspots(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_profile(runs_dir / "run-a")

    stage_rows, hotspot_rows = collect_performance_profile_rows(runs_dir)

    assert len(stage_rows) == 3
    fit_row = next(row for row in stage_rows if row.stage_name == "fit_model")
    assert fit_row.dataset == "ml100k"
    assert fit_row.model == "cb_svdpp"
    assert fit_row.wall_clock_seconds == 6.0
    assert fit_row.share_of_profiled_time == 0.6
    assert fit_row.rss_delta_mb == 20.0
    assert fit_row.status == "completed"

    assert len(hotspot_rows) == 1
    hotspot_row = hotspot_rows[0]
    assert hotspot_row.top_1_stage == "fit_model"
    assert hotspot_row.top_1_seconds == 6.0
    assert hotspot_row.top_2_stage == "build_cluster_artifacts"
    assert hotspot_row.top_2_seconds == 3.0
    assert hotspot_row.top_3_stage == "write_performance_profile"
    assert hotspot_row.top_3_seconds == 1.0
    assert hotspot_row.total_profiled_wall_clock_seconds == 10.0


def test_write_performance_profile_reports_writes_expected_csvs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    output_dir = tmp_path / "reports"
    _write_profile(runs_dir / "run-a")

    paths = write_performance_profile_reports(runs_dir=runs_dir, output_dir=output_dir)

    assert paths["stage_breakdown"] == output_dir / "performance_stage_breakdown.csv"
    assert paths["hotspots"] == output_dir / "performance_hotspots.csv"
    assert paths["stage_breakdown"].is_file()
    assert paths["hotspots"].is_file()

    with paths["stage_breakdown"].open(encoding="utf-8", newline="") as input_file:
        stage_rows = list(csv.DictReader(input_file))
    with paths["hotspots"].open(encoding="utf-8", newline="") as input_file:
        hotspot_rows = list(csv.DictReader(input_file))

    assert stage_rows[0]["dataset"] == "ml100k"
    assert "stage_name" in stage_rows[0]
    assert hotspot_rows[0]["top_1_stage"] == "fit_model"
