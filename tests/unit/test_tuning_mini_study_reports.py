from __future__ import annotations

import csv
import json
from pathlib import Path

from recsys_lab.config.loader import load_yaml_file
from recsys_lab.tuning import SearchSpaceSpec, build_study_plan
from recsys_lab.tuning.execution import CandidateExecutionResult
from recsys_lab.tuning.planner import StudyPlan
from recsys_lab.tuning.writers import write_study_execution_artifacts
from scripts.plan_tuning_study import plan_tuning_study


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _planned_fixture_study(tmp_path: Path) -> tuple[StudyPlan, Path]:
    plan_tuning_study(
        search_space_path=_repo_root() / "tests" / "fixtures" / "tuning" / "cb_svdpp_tuning_mvp.yaml",
        output_dir=tmp_path,
        study_id="mini_study_report_test",
        repo_root=_repo_root(),
    )
    study_dir = tmp_path / "mini_study_report_test"
    search_space = SearchSpaceSpec.model_validate(load_yaml_file(study_dir / "search_space.yaml"))
    original_plan = build_study_plan(search_space)
    plan = StudyPlan(
        study_id="mini_study_report_test",
        search_space=original_plan.search_space,
        candidates=original_plan.candidates,
        artifact_reuse_groups=original_plan.artifact_reuse_groups,
    )
    return plan, study_dir


def _execution_result(
    *,
    tmp_path: Path,
    plan: StudyPlan,
    candidate_index: int,
    cluster_status: str,
    history_status: str,
) -> CandidateExecutionResult:
    candidate = plan.candidates[candidate_index]
    run_dir = tmp_path / "runs" / candidate.candidate_id
    run_dir.mkdir(parents=True)
    metrics_path = run_dir / "metrics.json"
    performance_path = run_dir / "performance_profile.json"
    kernel_path = run_dir / "kernel_profile.json"
    run_manifest_path = run_dir / "run_manifest.json"
    metrics_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "validation_rmse": 1.0 + candidate_index,
                    "validation_mae": 0.8 + candidate_index,
                },
                "caches": {
                    "cluster_artifacts": {"status": cluster_status},
                    "user_cluster_history": {"status": history_status},
                },
                "timing": {"cluster_induction_wall_clock_seconds": 0.2 + candidate_index},
            }
        ),
        encoding="utf-8",
    )
    performance_path.write_text(
        json.dumps(
            {
                "total_profiled_wall_clock_seconds": 1.0 + candidate_index,
                "stages": [
                    {"name": "fit_model", "wall_clock_seconds": 0.5 + candidate_index},
                    {"name": "build_cluster_artifacts", "wall_clock_seconds": 0.2 + candidate_index},
                ],
            }
        ),
        encoding="utf-8",
    )
    kernel_path.write_text("{}", encoding="utf-8")
    run_manifest_path.write_text("{}", encoding="utf-8")
    return CandidateExecutionResult(
        candidate_id=candidate.candidate_id,
        study_id=plan.study_id,
        execution_status="succeeded",
        run_id=f"run-{candidate.candidate_id}",
        run_dir=str(run_dir),
        metrics_path=str(metrics_path),
        performance_profile_path=str(performance_path),
        kernel_profile_path=str(kernel_path),
        run_manifest_path=str(run_manifest_path),
    )


def test_mini_study_summary_detects_cold_warm_cache_reuse(tmp_path: Path) -> None:
    plan, study_dir = _planned_fixture_study(tmp_path)
    results = [
        _execution_result(
            tmp_path=tmp_path,
            plan=plan,
            candidate_index=0,
            cluster_status="miss",
            history_status="miss",
        ),
        _execution_result(tmp_path=tmp_path, plan=plan, candidate_index=1, cluster_status="hit", history_status="hit"),
    ]

    paths = write_study_execution_artifacts(plan, study_dir, results)
    payload = json.loads(paths["mini_study_summary_json"].read_text(encoding="utf-8"))

    assert payload["cache_reuse_observed"] is True
    assert payload["cold_candidate_id"] == plan.candidates[0].candidate_id
    assert payload["warm_candidate_ids"] == [plan.candidates[1].candidate_id]


def test_mini_study_summary_marks_cache_reuse_false_without_hit_after_miss(tmp_path: Path) -> None:
    plan, study_dir = _planned_fixture_study(tmp_path)
    results = [
        _execution_result(
            tmp_path=tmp_path,
            plan=plan,
            candidate_index=0,
            cluster_status="miss",
            history_status="miss",
        ),
        _execution_result(
            tmp_path=tmp_path,
            plan=plan,
            candidate_index=1,
            cluster_status="miss",
            history_status="miss",
        ),
    ]

    paths = write_study_execution_artifacts(plan, study_dir, results)
    payload = json.loads(paths["mini_study_summary_json"].read_text(encoding="utf-8"))

    assert payload["cache_reuse_observed"] is False
    assert payload["cold_candidate_id"] == plan.candidates[0].candidate_id
    assert payload["warm_candidate_ids"] == []


def test_mini_study_writes_summary_csv_and_json(tmp_path: Path) -> None:
    plan, study_dir = _planned_fixture_study(tmp_path)
    results = [
        _execution_result(
            tmp_path=tmp_path,
            plan=plan,
            candidate_index=0,
            cluster_status="miss",
            history_status="miss",
        ),
        _execution_result(tmp_path=tmp_path, plan=plan, candidate_index=1, cluster_status="hit", history_status="hit"),
    ]

    paths = write_study_execution_artifacts(plan, study_dir, results)

    assert paths["mini_study_summary_csv"].exists()
    assert paths["mini_study_summary_json"].exists()
    with paths["mini_study_summary_csv"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    payload = json.loads(paths["mini_study_summary_json"].read_text(encoding="utf-8"))
    assert rows[0]["study_id"] == plan.study_id
    assert rows[0]["candidate_id"] == plan.candidates[0].candidate_id
    assert rows[0]["notes"] == "cold_cache_build"
    assert rows[1]["notes"] == "warm_cache_hit"
    assert payload["study_id"] == plan.study_id
    assert payload["candidate_count"] == len(plan.candidates)
    assert payload["executed_candidate_count"] == 2
    assert payload["cluster_reuse_group_count"] == 1
