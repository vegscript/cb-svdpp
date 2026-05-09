from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from scripts import run_tuning_mini_study as mini_study_script
from scripts.plan_tuning_study import plan_tuning_study


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _planned_study(tmp_path: Path) -> tuple[Any, Path]:
    plan_tuning_study(
        search_space_path=_repo_root()
        / "configs"
        / "experiments"
        / "tuning"
        / "active"
        / "ml1m_cb_svdpp_small_study_v1.yaml",
        output_dir=tmp_path,
        study_id="selection_test",
        repo_root=_repo_root(),
        overwrite=True,
    )
    study_dir = tmp_path / "selection_test"
    return mini_study_script._load_plan(study_dir), study_dir


def _write_candidate_summary(
    *,
    plan: Any,
    study_dir: Path,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> None:
    fieldnames = [
        "candidate_id",
        "execution_status",
        "alpha",
        "learning_rate",
        "validation_rmse",
        "validation_mae",
        "test_rmse",
        "fit_model_seconds",
        "cluster_total_seconds",
        "cluster_cache_status",
        "user_cluster_history_cache_status",
        "cluster_reuse_group_id",
        "run_dir",
        "candidate_config_path",
    ]
    overrides = overrides or {}
    reports_dir = study_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    reuse_group_id = plan.artifact_reuse_groups[0].group_id
    rows = []
    for index, candidate in enumerate(plan.candidates):
        row = {
            "candidate_id": candidate.candidate_id,
            "execution_status": "succeeded",
            "alpha": str(candidate.parameter_values.get("alpha", "")),
            "learning_rate": str(candidate.parameter_values.get("learning_rate", "")),
            "validation_rmse": f"{1.0 + index * 0.01:.4f}",
            "validation_mae": f"{0.8 + index * 0.01:.4f}",
            "test_rmse": f"{0.7 + index * 0.01:.4f}",
            "fit_model_seconds": f"{100.0 + index:.4f}",
            "cluster_total_seconds": "0.1",
            "cluster_cache_status": "miss" if index == 0 else "hit",
            "user_cluster_history_cache_status": "miss" if index == 0 else "hit",
            "cluster_reuse_group_id": reuse_group_id,
            "run_dir": str(study_dir / "runs" / candidate.candidate_id),
            "candidate_config_path": str(study_dir / "candidates" / candidate.candidate_id / "candidate_config.yaml"),
        }
        row.update(overrides.get(candidate.candidate_id, {}))
        rows.append(row)

    with (reports_dir / "candidate_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _selection_payload(study_dir: Path) -> dict[str, Any]:
    return json.loads((study_dir / "selected" / "selected_candidate.json").read_text(encoding="utf-8"))


def _ranking_rows(study_dir: Path) -> list[dict[str, str]]:
    with (study_dir / "reports" / "candidate_ranking.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_candidate_ranking_orders_by_validation_rmse(tmp_path: Path) -> None:
    plan, study_dir = _planned_study(tmp_path)
    expected = plan.candidates[3]
    _write_candidate_summary(
        plan=plan,
        study_dir=study_dir,
        overrides={expected.candidate_id: {"validation_rmse": "0.5000"}},
    )

    mini_study_script._write_ranking_and_selection(plan, study_dir, repo_root=_repo_root())

    assert _selection_payload(study_dir)["selected_candidate_id"] == expected.candidate_id
    assert _ranking_rows(study_dir)[0]["candidate_id"] == expected.candidate_id


def test_candidate_ranking_uses_validation_mae_tiebreaker(tmp_path: Path) -> None:
    plan, study_dir = _planned_study(tmp_path)
    expected = plan.candidates[2]
    overrides = {
        candidate.candidate_id: {"validation_rmse": "0.9000", "validation_mae": "0.8000"}
        for candidate in plan.candidates
    }
    overrides[expected.candidate_id]["validation_mae"] = "0.7000"
    _write_candidate_summary(plan=plan, study_dir=study_dir, overrides=overrides)

    mini_study_script._write_ranking_and_selection(plan, study_dir, repo_root=_repo_root())

    assert _selection_payload(study_dir)["selected_candidate_id"] == expected.candidate_id


def test_candidate_ranking_uses_fit_time_second_tiebreaker(tmp_path: Path) -> None:
    plan, study_dir = _planned_study(tmp_path)
    expected = plan.candidates[4]
    overrides = {
        candidate.candidate_id: {
            "validation_rmse": "0.9000",
            "validation_mae": "0.7000",
            "fit_model_seconds": "50.0000",
        }
        for candidate in plan.candidates
    }
    overrides[expected.candidate_id]["fit_model_seconds"] = "10.0000"
    _write_candidate_summary(plan=plan, study_dir=study_dir, overrides=overrides)

    mini_study_script._write_ranking_and_selection(plan, study_dir, repo_root=_repo_root())

    assert _selection_payload(study_dir)["selected_candidate_id"] == expected.candidate_id


def test_selection_ignores_failed_candidates(tmp_path: Path) -> None:
    plan, study_dir = _planned_study(tmp_path)
    failed_best = plan.candidates[0]
    expected = plan.candidates[1]
    _write_candidate_summary(
        plan=plan,
        study_dir=study_dir,
        overrides={
            failed_best.candidate_id: {"execution_status": "failed", "validation_rmse": "0.1000"},
            expected.candidate_id: {"validation_rmse": "0.2000"},
        },
    )

    mini_study_script._write_ranking_and_selection(plan, study_dir, repo_root=_repo_root())

    rows = {row["candidate_id"]: row for row in _ranking_rows(study_dir)}
    assert rows[failed_best.candidate_id]["rank"] == ""
    assert rows[failed_best.candidate_id]["selected"] == "false"
    assert _selection_payload(study_dir)["selected_candidate_id"] == expected.candidate_id


def test_selection_rejects_test_metrics(tmp_path: Path) -> None:
    plan, study_dir = _planned_study(tmp_path)
    test_metric_favorite = plan.candidates[0]
    expected = plan.candidates[1]
    _write_candidate_summary(
        plan=plan,
        study_dir=study_dir,
        overrides={
            test_metric_favorite.candidate_id: {"validation_rmse": "0.9000", "test_rmse": "0.1000"},
            expected.candidate_id: {"validation_rmse": "0.8000", "test_rmse": "0.9999"},
        },
    )

    mini_study_script._write_ranking_and_selection(plan, study_dir, repo_root=_repo_root())

    assert _selection_payload(study_dir)["selected_candidate_id"] == expected.candidate_id


def test_selected_candidate_config_is_copied(tmp_path: Path) -> None:
    plan, study_dir = _planned_study(tmp_path)
    expected = plan.candidates[5]
    _write_candidate_summary(
        plan=plan,
        study_dir=study_dir,
        overrides={expected.candidate_id: {"validation_rmse": "0.4000"}},
    )

    mini_study_script._write_ranking_and_selection(plan, study_dir, repo_root=_repo_root())

    selected_config_path = study_dir / "selected" / "selected_candidate_config.yaml"
    expected_config_path = study_dir / "candidates" / expected.candidate_id / "candidate_config.yaml"
    assert selected_config_path.exists()
    assert selected_config_path.read_text(encoding="utf-8") == expected_config_path.read_text(encoding="utf-8")


def test_small_study_summary_requires_six_successes_for_selection(tmp_path: Path) -> None:
    plan, study_dir = _planned_study(tmp_path)
    overrides = {
        candidate.candidate_id: {"execution_status": "failed"}
        for candidate in plan.candidates[5:]
    }
    _write_candidate_summary(plan=plan, study_dir=study_dir, overrides=overrides)

    result = mini_study_script._write_ranking_and_selection(plan, study_dir, repo_root=_repo_root())
    payload = _selection_payload(study_dir)

    assert payload["decision"] == "EXECUTION_UNSTABLE_FIX_BEFORE_TUNING"
    assert payload["selected_candidate_id"] is None
    assert result["selected_candidate_config"] is None
    assert not (study_dir / "selected" / "selected_candidate_config.yaml").exists()
