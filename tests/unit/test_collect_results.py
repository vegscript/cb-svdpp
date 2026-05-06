import json
from pathlib import Path

from recsys_lab.reporting.collect_results import collect_result_rows, render_markdown_table


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run(
    runs_dir: Path,
    run_id: str,
    *,
    dataset: str,
    model: str,
    generated_at_utc: str,
    with_metrics: bool = True,
) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir()
    _write_json(
        run_dir / "run_manifest.json",
        {
            "dataset": {"short_name": dataset},
            "generated_at_utc": generated_at_utc,
            "model": {"name": model},
            "run_id": run_id,
            "status": "completed" if with_metrics else "started",
        },
    )
    if with_metrics:
        _write_json(
            run_dir / "metrics.json",
            {
                "cb_diagnostics": {
                    "alpha": 0.1,
                    "cb_claim_eligible": False,
                    "diagnostic_claim_ready": False,
                },
                "metrics": {
                    "test": {
                        "abs_error_p90": 1.2,
                        "mae": 0.6,
                        "prediction_out_of_range_rate": 0.0,
                        "rmse": 0.8,
                    },
                    "validation": {"mae": 0.5, "rmse": 0.7},
                },
                "system_metrics": {"peak_memory_mb": 123.4, "train_time_total": 45.6},
            },
        )
    return run_dir


def test_collect_result_rows_uses_latest_artifact_per_dataset_model(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _write_run(
        runs_dir,
        "older",
        dataset="ml100k",
        model="cb_svdpp",
        generated_at_utc="2026-05-01T000000Z",
    )
    _write_run(
        runs_dir,
        "newer",
        dataset="ml100k",
        model="cb_svdpp",
        generated_at_utc="2026-05-02T000000Z",
    )

    rows = collect_result_rows(runs_dir)

    assert len(rows) == 1
    row = rows[0]
    assert row.run_id == "newer"
    assert row.dataset == "ml100k"
    assert row.model == "cb_svdpp"
    assert row.validation_rmse == 0.7
    assert row.test_rmse == 0.8
    assert row.validation_mae == 0.5
    assert row.test_mae == 0.6
    assert row.abs_error_p90 == 1.2
    assert row.prediction_out_of_range_rate == 0.0
    assert row.train_time == 45.6
    assert row.peak_memory == 123.4
    assert row.alpha == 0.1
    assert row.cb_claim_eligible is False
    assert row.diagnostic_claim_ready is False
    assert row.status == "completed"


def test_collect_result_rows_marks_missing_metrics(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _write_run(
        runs_dir,
        "missing-metrics",
        dataset="ml20m",
        model="cb_svdpp",
        generated_at_utc="2026-05-03T000000Z",
        with_metrics=False,
    )

    rows = collect_result_rows(runs_dir)
    table = render_markdown_table(rows)

    assert rows[0].status == "started_missing_metrics"
    assert "| ml20m | cb_svdpp |" in table
    assert "started_missing_metrics" in table
