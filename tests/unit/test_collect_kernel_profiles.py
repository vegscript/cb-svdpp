from __future__ import annotations

import csv
import json
from pathlib import Path

from recsys_lab.reporting.collect_kernel_profiles import collect_kernel_profile_rows, write_kernel_profile_report


def _write_profile(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "kernel_profile.json").write_text(
        json.dumps(
            {
                "profile_version": "kernel_cost_anatomy_v1",
                "run_id": "2026-05-06T000000Z_ml100k_cb_svdpp_local_s001",
                "dataset": "ml100k",
                "model": "cb_svdpp",
                "epochs": 2,
                "latent_dim": 8,
                "train_rows": 100,
                "epoch_durations_seconds": [1.0, 1.5],
                "ratings_per_second_by_epoch": [100.0, 66.6666666667],
                "history_structure": {
                    "implicit": {"total_edges": 10},
                    "explicit": {},
                    "cluster": {"total_edges": 5},
                },
                "estimated_kernel_work": {
                    "rating_updates": 200,
                    "implicit_history_visits": 1000,
                    "explicit_history_visits": 0,
                    "cluster_history_visits": 300,
                    "estimated_factor_touches": 10000,
                },
                "cost_ratios": {
                    "fit_seconds_per_epoch_mean": 1.25,
                    "fit_seconds_per_million_ratings": 12500.0,
                    "fit_seconds_per_million_estimated_factor_touches": 250.0,
                },
                "notes": [],
            }
        ),
        encoding="utf-8",
    )


def test_collect_kernel_profile_rows_flattens_kernel_costs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_profile(runs_dir / "run-a")

    rows = collect_kernel_profile_rows(runs_dir)

    assert len(rows) == 1
    row = rows[0]
    assert row.dataset == "ml100k"
    assert row.model == "cb_svdpp"
    assert row.epochs == 2
    assert row.latent_dim == 8
    assert row.train_rows == 100
    assert row.fit_seconds_total == 2.5
    assert row.fit_seconds_per_epoch_mean == 1.25
    assert row.implicit_history_visits == 1000
    assert row.explicit_history_visits == 0
    assert row.cluster_history_visits == 300
    assert row.estimated_factor_touches == 10000
    assert row.fit_seconds_per_million_estimated_factor_touches == 250.0


def test_write_kernel_profile_report_writes_expected_csv(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    output_dir = tmp_path / "reports"
    _write_profile(runs_dir / "run-a")

    output_path = write_kernel_profile_report(runs_dir=runs_dir, output_dir=output_dir)

    assert output_path == output_dir / "kernel_cost_anatomy.csv"
    assert output_path.is_file()
    with output_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    assert rows[0]["dataset"] == "ml100k"
    assert rows[0]["model"] == "cb_svdpp"
    assert rows[0]["fit_seconds_total"] == "2.5"
    assert rows[0]["estimated_factor_touches"] == "10000"
