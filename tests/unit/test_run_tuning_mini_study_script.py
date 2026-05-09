from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from recsys_lab.tuning.execution import CandidateExecutionResult
from recsys_lab.tuning.writers import write_study_execution_artifacts
from scripts import run_tuning_mini_study as mini_study_script


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _fixture_search_space() -> Path:
    return _repo_root() / "tests" / "fixtures" / "tuning" / "cb_svdpp_tuning_mvp.yaml"


def _runner_kwargs(tmp_path: Path) -> dict[str, Any]:
    return {
        "search_space": _fixture_search_space(),
        "output_dir": tmp_path / "tuning",
        "study_id": "mini_study_script_test",
        "processed_manifest": Path("data/processed/ml100k/manifest.json"),
        "runtime_config": Path("configs/runtime/base.yaml"),
        "device_config": Path("configs/runtime/devices/local_u300_24gb.yaml"),
        "cache_root": None,
        "repo_root": _repo_root(),
        "max_candidates": 2,
        "overwrite": False,
        "train_ratio": 0.8,
        "validation_ratio": 0.1,
        "split_seed": 1,
        "model_seed": 1,
        "evaluate_test": True,
        "use_split_cache": None,
        "use_training_index_cache": True,
        "use_cluster_artifact_cache": True,
        "require_cache_reuse_evidence": True,
        "benchmark_mode": False,
    }


def test_run_tuning_mini_study_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_tuning_mini_study.py", "--help"],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--search-space" in result.stdout
    assert "--max-candidates" in result.stdout


def test_mini_study_script_rejects_max_candidates_above_three(tmp_path: Path) -> None:
    kwargs = _runner_kwargs(tmp_path)
    kwargs["max_candidates"] = 4

    with pytest.raises(ValueError, match="--max-candidates must be <= 3"):
        mini_study_script.run_tuning_mini_study(**kwargs)


def test_mini_study_script_requires_ml1m_for_benchmark_mode(tmp_path: Path) -> None:
    kwargs = _runner_kwargs(tmp_path)
    kwargs["benchmark_mode"] = True

    with pytest.raises(ValueError, match="requires dataset ml1m"):
        mini_study_script.run_tuning_mini_study(**kwargs)


def test_mini_study_script_plans_and_runs_two_candidates_with_fake_runner(tmp_path: Path) -> None:
    write_counts: list[int] = []

    execution_index = 0

    def _fake_execute_candidate(candidate_manifest_path: Path, *, runner_kwargs: dict[str, Any], repo_root: Path):
        nonlocal execution_index
        execution_index += 1
        candidate_id = candidate_manifest_path.parent.name
        run_dir = tmp_path / "runs" / candidate_id
        run_dir.mkdir(parents=True)
        metrics_path = run_dir / "metrics.json"
        performance_path = run_dir / "performance_profile.json"
        kernel_path = run_dir / "kernel_profile.json"
        run_manifest_path = run_dir / "run_manifest.json"
        cluster_status = "miss" if execution_index == 1 else "hit"
        history_status = "miss" if execution_index == 1 else "hit"
        metrics_path.write_text(
            json.dumps(
                {
                    "metrics": {"validation_rmse": 1.0, "validation_mae": 0.8},
                    "caches": {
                        "cluster_artifacts": {"status": cluster_status},
                        "user_cluster_history": {"status": history_status},
                    },
                    "timing": {"cluster_induction_wall_clock_seconds": 0.2},
                }
            ),
            encoding="utf-8",
        )
        performance_path.write_text(
            json.dumps(
                {
                    "total_profiled_wall_clock_seconds": 0.5,
                    "stages": [
                        {"name": "fit_model", "wall_clock_seconds": 0.1},
                        {"name": "build_cluster_artifacts", "wall_clock_seconds": 0.2},
                    ],
                }
            ),
            encoding="utf-8",
        )
        kernel_path.write_text("{}", encoding="utf-8")
        run_manifest_path.write_text("{}", encoding="utf-8")
        assert runner_kwargs["use_cluster_artifact_cache"] is True
        assert repo_root == _repo_root()
        return CandidateExecutionResult(
            candidate_id=candidate_id,
            study_id="mini_study_script_test",
            execution_status="succeeded",
            run_id=f"run-{candidate_id}",
            run_dir=str(run_dir),
            metrics_path=str(metrics_path),
            performance_profile_path=str(performance_path),
            kernel_profile_path=str(kernel_path),
            run_manifest_path=str(run_manifest_path),
        )

    def _write_artifacts(plan: Any, study_dir: Path, results: list[CandidateExecutionResult]):
        write_counts.append(len(results))
        return write_study_execution_artifacts(plan, study_dir, results)

    result = mini_study_script.run_tuning_mini_study(
        **_runner_kwargs(tmp_path),
        execute_candidate_fn=_fake_execute_candidate,
        write_execution_artifacts_fn=_write_artifacts,
    )

    study_dir = Path(result["study_dir"])
    assert result["executed_candidate_count"] == 2
    assert write_counts == [1, 2]
    assert (study_dir / "reports" / "candidate_summary.csv").exists()
    assert (study_dir / "reports" / "execution_summary.csv").exists()
    with (study_dir / "reports" / "execution_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["execution_status"] for row in rows} == {"succeeded"}
    with (study_dir / "reports" / "candidate_summary.csv").open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert [row["cluster_cache_status"] for row in summary_rows[:2]] == ["miss", "hit"]
    assert [row["user_cluster_history_cache_status"] for row in summary_rows[:2]] == ["miss", "hit"]
    assert summary_rows[0]["cluster_reuse_group_id"] == summary_rows[1]["cluster_reuse_group_id"]
    assert result["cache_reuse_evidence"]["status"] == "validated"
    assert Path(result["mini_study_summary_csv"]).exists()
    assert Path(result["mini_study_summary_json"]).exists()
    with Path(result["mini_study_summary_csv"]).open(encoding="utf-8", newline="") as handle:
        mini_rows = list(csv.DictReader(handle))
    mini_payload = json.loads(Path(result["mini_study_summary_json"]).read_text(encoding="utf-8"))
    assert [row["notes"] for row in mini_rows[:2]] == ["cold_cache_build", "warm_cache_hit"]
    assert mini_payload["cache_reuse_observed"] is True
    assert mini_payload["cold_candidate_id"] == summary_rows[0]["candidate_id"]
    assert mini_payload["warm_candidate_ids"] == [summary_rows[1]["candidate_id"]]


def test_mini_study_stops_after_first_failed_candidate(tmp_path: Path) -> None:
    def _failed_execute_candidate(candidate_manifest_path: Path, *, runner_kwargs: dict[str, Any], repo_root: Path):
        return CandidateExecutionResult(
            candidate_id=candidate_manifest_path.parent.name,
            study_id="mini_study_script_test",
            execution_status="failed",
            error_message="controlled failure",
        )

    result = mini_study_script.run_tuning_mini_study(
        **_runner_kwargs(tmp_path),
        execute_candidate_fn=_failed_execute_candidate,
    )

    assert result["executed_candidate_count"] == 1
    assert list(result["execution_statuses"].values()) == ["failed"]


def test_mini_study_rejects_both_candidates_hit_for_cache_reuse_evidence(tmp_path: Path) -> None:
    def _hit_execute_candidate(candidate_manifest_path: Path, *, runner_kwargs: dict[str, Any], repo_root: Path):
        candidate_id = candidate_manifest_path.parent.name
        run_dir = tmp_path / "runs" / candidate_id
        run_dir.mkdir(parents=True)
        metrics_path = run_dir / "metrics.json"
        performance_path = run_dir / "performance_profile.json"
        kernel_path = run_dir / "kernel_profile.json"
        run_manifest_path = run_dir / "run_manifest.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "metrics": {"validation_rmse": 1.0, "validation_mae": 0.8},
                    "caches": {
                        "cluster_artifacts": {"status": "hit"},
                        "user_cluster_history": {"status": "hit"},
                    },
                }
            ),
            encoding="utf-8",
        )
        performance_path.write_text(
            json.dumps({"total_profiled_wall_clock_seconds": 0.5, "stages": []}),
            encoding="utf-8",
        )
        kernel_path.write_text("{}", encoding="utf-8")
        run_manifest_path.write_text("{}", encoding="utf-8")
        return CandidateExecutionResult(
            candidate_id=candidate_id,
            study_id="mini_study_script_test",
            execution_status="succeeded",
            run_id=f"run-{candidate_id}",
            run_dir=str(run_dir),
            metrics_path=str(metrics_path),
            performance_profile_path=str(performance_path),
            kernel_profile_path=str(kernel_path),
            run_manifest_path=str(run_manifest_path),
        )

    with pytest.raises(ValueError, match="first candidate must prove cold cluster cache status"):
        mini_study_script.run_tuning_mini_study(
            **_runner_kwargs(tmp_path),
            execute_candidate_fn=_hit_execute_candidate,
        )
