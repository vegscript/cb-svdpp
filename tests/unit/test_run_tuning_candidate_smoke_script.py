from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from recsys_lab.config.loader import dump_yaml_file
from recsys_lab.tuning.execution import CandidateExecutionResult
from scripts import run_tuning_candidate_smoke as smoke_script
from scripts.plan_tuning_study import plan_tuning_study


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _single_candidate_search_space(path: Path) -> Path:
    payload: dict[str, Any] = {
        "search_space_version": "tuning_search_space_v1",
        "study": {
            "name": "single_candidate_execution_smoke",
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
            "cluster_artifacts": {
                "reuse_across": [
                    "alpha",
                    "learning_rate",
                    "lambda_p",
                    "lambda_q",
                    "lambda_y",
                    "lambda_pC",
                    "lambda_qC",
                    "lambda_yC",
                    "epochs",
                ],
                "invalidate_on": [
                    "n_user_clusters",
                    "n_item_clusters",
                    "induction_config",
                    "kmeans_n_init",
                    "clustering_algorithm",
                    "dataset",
                    "split",
                    "train_fingerprint",
                ],
            },
        },
        "objective": {"primary": {"metric": "validation_rmse"}},
    }
    dump_yaml_file(path, payload)
    return path


def _planned_single_candidate_study(tmp_path: Path) -> Path:
    search_space_path = _single_candidate_search_space(tmp_path / "search_space.yaml")
    plan_tuning_study(
        search_space_path=search_space_path,
        output_dir=tmp_path,
        study_id="single_candidate_study",
        repo_root=_repo_root(),
    )
    return tmp_path / "single_candidate_study"


def test_run_tuning_candidate_smoke_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_tuning_candidate_smoke.py", "--help"],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--study-dir" in result.stdout
    assert "--candidate-id" in result.stdout
    assert "--max-candidates" not in result.stdout


def test_smoke_dry_run_selects_single_candidate_without_candidate_id(tmp_path: Path) -> None:
    study_dir = _planned_single_candidate_study(tmp_path)

    result = smoke_script.run_tuning_candidate_smoke(
        study_dir=study_dir,
        candidate_id=None,
        repo_root=_repo_root(),
        dry_run=True,
        processed_manifest=None,
        runtime_config=Path("configs/runtime/base.yaml"),
        device_config=Path("configs/runtime/devices/local_u300_24gb.yaml"),
        train_ratio=0.8,
        validation_ratio=0.1,
        split_seed=1,
        model_seed=1,
        evaluate_test=True,
        use_split_cache=None,
        use_training_index_cache=False,
        use_cluster_artifact_cache=False,
    )

    assert result["execution_status"] == "not_executed"
    assert result["dry_run"] is True
    assert str(result["candidate_id"]).startswith("cand_0000_")


def test_smoke_requires_candidate_id_for_multi_candidate_study(tmp_path: Path) -> None:
    plan_tuning_study(
        search_space_path=_repo_root() / "tests" / "fixtures" / "tuning" / "cb_svdpp_tuning_mvp.yaml",
        output_dir=tmp_path,
        study_id="multi_candidate_study",
        repo_root=_repo_root(),
    )

    try:
        smoke_script.run_tuning_candidate_smoke(
            study_dir=tmp_path / "multi_candidate_study",
            candidate_id=None,
            repo_root=_repo_root(),
            dry_run=True,
            processed_manifest=None,
            runtime_config=Path("configs/runtime/base.yaml"),
            device_config=Path("configs/runtime/devices/local_u300_24gb.yaml"),
            train_ratio=0.8,
            validation_ratio=0.1,
            split_seed=1,
            model_seed=1,
            evaluate_test=True,
            use_split_cache=None,
            use_training_index_cache=False,
            use_cluster_artifact_cache=False,
        )
    except ValueError as exc:
        assert "--candidate-id is required" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected multi-candidate smoke without candidate id to fail")


def test_run_candidate_script_refuses_multiple_candidates_without_candidate_id(tmp_path: Path) -> None:
    test_smoke_requires_candidate_id_for_multi_candidate_study(tmp_path)


def test_smoke_executes_one_candidate_and_updates_artifacts(tmp_path: Path) -> None:
    study_dir = _planned_single_candidate_study(tmp_path)
    candidate_manifest_path = next((study_dir / "candidates").glob("*/candidate_manifest.json"))
    candidate_id = candidate_manifest_path.parent.name

    def _fake_execute_candidate(candidate_manifest_path_arg, *, runner_kwargs, repo_root):
        assert candidate_manifest_path_arg == candidate_manifest_path
        assert runner_kwargs["model_name"] == "cb_svdpp"
        assert runner_kwargs["split_family"] == "benchmark_random_v1"
        run_dir = tmp_path / "runs" / "run-1"
        run_dir.mkdir(parents=True)
        metrics_path = run_dir / "metrics.json"
        performance_path = run_dir / "performance_profile.json"
        kernel_path = run_dir / "kernel_profile.json"
        run_manifest_path = run_dir / "run_manifest.json"
        metrics_path.write_text(
            json.dumps({"metrics": {"validation_rmse": 1.1, "validation_mae": 0.8}}),
            encoding="utf-8",
        )
        performance_path.write_text(json.dumps({"stages": []}), encoding="utf-8")
        kernel_path.write_text("{}", encoding="utf-8")
        run_manifest_path.write_text("{}", encoding="utf-8")
        return CandidateExecutionResult(
            candidate_id=candidate_id,
            study_id="single_candidate_study",
            execution_status="succeeded",
            run_id="run-1",
            run_dir=str(run_dir),
            metrics_path=str(metrics_path),
            performance_profile_path=str(performance_path),
            kernel_profile_path=str(kernel_path),
            run_manifest_path=str(run_manifest_path),
        )

    result = smoke_script.run_tuning_candidate_smoke(
        study_dir=study_dir,
        candidate_id=candidate_id,
        repo_root=_repo_root(),
        dry_run=False,
        processed_manifest=Path("data/processed/ml100k/manifest.json"),
        runtime_config=Path("configs/runtime/base.yaml"),
        device_config=Path("configs/runtime/devices/local_u300_24gb.yaml"),
        train_ratio=0.8,
        validation_ratio=0.1,
        split_seed=1,
        model_seed=1,
        evaluate_test=True,
        use_split_cache=None,
        use_training_index_cache=False,
        use_cluster_artifact_cache=False,
        execute_candidate_fn=_fake_execute_candidate,
    )

    assert result["execution_status"] == "succeeded"
    assert (study_dir / "reports" / "execution_summary.csv").exists()
    with (study_dir / "reports" / "candidate_summary.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["execution_status"] == "succeeded"
    assert row["run_id"] == "run-1"
    assert row["validation_rmse"] == "1.1"
