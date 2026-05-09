from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from recsys_lab.tuning import SearchSpaceSpec, build_study_plan, default_cluster_artifact_reuse_spec
from recsys_lab.tuning.execution import CandidateExecutionResult, execute_candidate
from recsys_lab.tuning.manifests import build_candidate_manifests
from recsys_lab.tuning.writers import (
    update_candidate_manifest_with_execution_result,
    write_execution_summary_csv,
    write_study_execution_artifacts,
    write_tuning_json,
)


def _payload() -> dict[str, object]:
    return {
        "search_space_version": "tuning_search_space_v1",
        "study": {
            "name": "execution_smoke_unit_study",
            "dataset": "ml100k",
            "split_family": "benchmark_random_v1",
            "model": "cb_svdpp",
            "seed": 1,
        },
        "base_model_config": "configs/models/cb_svdpp.yaml",
        "budget": {"max_candidates": 1},
        "generator": {"type": "grid", "deterministic_order": True},
        "search_space": {
            "alpha": {
                "type": "float",
                "values": [0.2],
                "target_path": "clustering.alpha",
            },
        },
        "artifact_reuse": {
            "cluster_artifacts": default_cluster_artifact_reuse_spec().model_dump(mode="json")
        },
        "objective": {"primary": {"metric": "validation_rmse"}},
    }


def _write_candidate_manifest(study_dir: Path):
    plan = build_study_plan(SearchSpaceSpec.model_validate(_payload()))
    manifest = build_candidate_manifests(plan, output_dir=str(study_dir))[0]
    candidate_dir = study_dir / "candidates" / manifest.candidate_id
    candidate_dir.mkdir(parents=True)
    candidate_manifest_path = candidate_dir / "candidate_manifest.json"
    write_tuning_json(manifest, candidate_manifest_path)
    return plan, manifest, candidate_manifest_path


def test_execution_result_schema_accepts_success(tmp_path: Path) -> None:
    result = CandidateExecutionResult(
        candidate_id="cand_0000_success",
        study_id="study",
        execution_status="succeeded",
        run_id="run-1",
        run_dir=str(tmp_path / "run-1"),
        metrics_path=str(tmp_path / "run-1" / "metrics.json"),
        performance_profile_path=str(tmp_path / "run-1" / "performance_profile.json"),
        kernel_profile_path=str(tmp_path / "run-1" / "kernel_profile.json"),
        run_manifest_path=str(tmp_path / "run-1" / "run_manifest.json"),
    )

    assert result.execution_status == "succeeded"
    assert result.run_id == "run-1"
    assert result.error_message is None


def test_execution_result_schema_accepts_failure() -> None:
    result = CandidateExecutionResult(
        candidate_id="cand_0000_failed",
        study_id="study",
        execution_status="failed",
        error_message="RuntimeError: synthetic failure",
    )

    assert result.execution_status == "failed"
    assert result.error_message == "RuntimeError: synthetic failure"


def test_execution_adapter_requires_existing_candidate_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        execute_candidate(
            tmp_path / "missing" / "candidate_manifest.json",
            runner_kwargs={},
            raise_on_error=True,
        )


def test_candidate_manifest_updates_execution_fields(tmp_path: Path) -> None:
    _plan, manifest, candidate_manifest_path = _write_candidate_manifest(tmp_path)
    result = CandidateExecutionResult(
        candidate_id=manifest.candidate_id,
        study_id=manifest.study_id,
        execution_status="succeeded",
        run_id="run-1",
        run_dir=str(tmp_path / "run-1"),
        metrics_path=str(tmp_path / "run-1" / "metrics.json"),
        performance_profile_path=str(tmp_path / "run-1" / "performance_profile.json"),
        kernel_profile_path=str(tmp_path / "run-1" / "kernel_profile.json"),
        run_manifest_path=str(tmp_path / "run-1" / "run_manifest.json"),
    )

    update_candidate_manifest_with_execution_result(candidate_manifest_path, result)

    updated = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    assert updated["execution_status"] == "succeeded"
    assert updated["run_id"] == "run-1"
    assert updated["metrics_path"].endswith("metrics.json")
    assert updated["error_message"] is None


def test_candidate_manifest_updates_after_execution_result(tmp_path: Path) -> None:
    test_candidate_manifest_updates_execution_fields(tmp_path)


def test_execution_summary_writer_writes_csv(tmp_path: Path) -> None:
    path = write_execution_summary_csv(
        [
            CandidateExecutionResult(
                candidate_id="cand_0000_summary",
                study_id="study",
                execution_status="failed",
                error_message="ValueError: synthetic",
            )
        ],
        tmp_path / "execution_summary.csv",
    )

    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert row["study_id"] == "study"
    assert row["candidate_id"] == "cand_0000_summary"
    assert row["execution_status"] == "failed"
    assert row["error_message"] == "ValueError: synthetic"


def test_study_execution_artifacts_write_summaries(tmp_path: Path) -> None:
    plan, manifest, _candidate_manifest_path = _write_candidate_manifest(tmp_path)
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    metrics_path = run_dir / "metrics.json"
    performance_profile_path = run_dir / "performance_profile.json"
    kernel_profile_path = run_dir / "kernel_profile.json"
    run_manifest_path = run_dir / "run_manifest.json"
    metrics_path.write_text(
        json.dumps(
            {
                "metrics": {"validation_rmse": 1.25, "validation_mae": 0.75},
                "caches": {
                    "cluster_artifacts": {"status": "hit"},
                    "user_cluster_history": {"status": "hit"},
                },
                "timing": {"cluster_induction_wall_clock_seconds": 0.4},
            }
        ),
        encoding="utf-8",
    )
    performance_profile_path.write_text(
        json.dumps(
            {
                "total_profiled_wall_clock_seconds": 4.0,
                "stages": [
                    {"name": "fit_model", "wall_clock_seconds": 2.5},
                    {"name": "build_cluster_artifacts", "wall_clock_seconds": 0.3},
                ]
            }
        ),
        encoding="utf-8",
    )
    kernel_profile_path.write_text("{}", encoding="utf-8")
    run_manifest_path.write_text("{}", encoding="utf-8")
    result = CandidateExecutionResult(
        candidate_id=manifest.candidate_id,
        study_id=manifest.study_id,
        execution_status="succeeded",
        run_id="run-1",
        run_dir=str(run_dir),
        metrics_path=str(metrics_path),
        performance_profile_path=str(performance_profile_path),
        kernel_profile_path=str(kernel_profile_path),
        run_manifest_path=str(run_manifest_path),
    )

    paths = write_study_execution_artifacts(plan, tmp_path, [result])

    with paths["candidate_summary"].open(encoding="utf-8", newline="") as handle:
        candidate_row = next(csv.DictReader(handle))
    with paths["execution_summary"].open(encoding="utf-8", newline="") as handle:
        execution_row = next(csv.DictReader(handle))

    assert candidate_row["execution_status"] == "succeeded"
    assert candidate_row["run_id"] == "run-1"
    assert candidate_row["validation_rmse"] == "1.25"
    assert candidate_row["validation_mae"] == "0.75"
    assert candidate_row["fit_model_seconds"] == "2.5"
    assert candidate_row["total_wall_seconds"] == "4.0"
    assert candidate_row["cluster_cache_status"] == "hit"
    assert candidate_row["user_cluster_history_cache_status"] == "hit"
    assert candidate_row["cluster_total_seconds"] == "0.3"
    assert candidate_row["run_dir"] == str(run_dir)
    assert candidate_row["metrics_path"] == str(metrics_path)
    assert candidate_row["performance_profile_path"] == str(performance_profile_path)
    assert candidate_row["kernel_profile_path"] == str(kernel_profile_path)
    assert candidate_row["run_manifest_path"] == str(run_manifest_path)
    assert execution_row["candidate_id"] == manifest.candidate_id
    assert execution_row["run_manifest_path"] == str(run_manifest_path)
    assert paths["mini_study_summary_csv"].exists()
    assert paths["mini_study_summary_json"].exists()
    with paths["mini_study_summary_csv"].open(encoding="utf-8", newline="") as handle:
        mini_row = next(csv.DictReader(handle))
    mini_payload = json.loads(paths["mini_study_summary_json"].read_text(encoding="utf-8"))
    assert mini_row["candidate_id"] == manifest.candidate_id
    assert mini_row["cluster_cache_status"] == "hit"
    assert mini_row["user_cluster_history_cache_status"] == "hit"
    assert mini_row["notes"] == "warm_cache_hit"
    assert mini_payload["study_id"] == manifest.study_id
    assert mini_payload["candidate_count"] == 1
    assert mini_payload["executed_candidate_count"] == 1
    assert mini_payload["cache_reuse_observed"] is False
