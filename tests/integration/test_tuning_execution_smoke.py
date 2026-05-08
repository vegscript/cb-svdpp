from __future__ import annotations

import csv
import json
from pathlib import Path

from recsys_lab.config.loader import dump_yaml_file
from recsys_lab.experiments.common import SplitConfig
from recsys_lab.experiments.unified_runner import build_experiment_services
from scripts.plan_tuning_study import plan_tuning_study
from scripts.run_tuning_candidate_smoke import run_tuning_candidate_smoke
from tests.integration.test_unified_pipeline_smoke_all_models import (
    _assert_train_coverage,
    _prepare_toy_repo,
)


def test_tuning_execution_smoke_runs_one_candidate_and_updates_manifest(tmp_path: Path) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    repo_root, processed_manifest_path, runtime_config_path, device_config_path, _model_config_paths = (
        _prepare_toy_repo(
            tmp_path,
            actual_repo_root,
        )
    )
    split_config = SplitConfig(train_ratio=0.5, validation_ratio=0.25, seed=3)
    _assert_train_coverage(processed_manifest_path, split_config)

    search_space_path = repo_root / "configs" / "experiments" / "tuning" / "cb_svdpp_execution_smoke.yaml"
    dump_yaml_file(
        search_space_path,
        {
            "search_space_version": "tuning_search_space_v1",
            "study": {
                "name": "cb_svdpp_execution_smoke_v1",
                "dataset": "ml_latest_small",
                "split_family": "benchmark_random_v1",
                "model": "cb_svdpp",
                "seed": 1,
            },
            "base_model_config": "configs/models/cb_svdpp.yaml",
            "budget": {"max_candidates": 1, "max_parallel": 1, "max_wall_seconds": None},
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
                }
            },
            "objective": {
                "primary": {"metric": "validation_rmse", "direction": "minimize", "aggregation": "mean"},
                "secondary": [{"metric": "validation_mae", "direction": "minimize", "aggregation": "mean"}],
                "required_guards": ["cluster_cache_status", "cluster_total_seconds"],
            },
        },
    )
    plan_tuning_study(
        search_space_path=search_space_path,
        output_dir=repo_root / "artifacts" / "tuning",
        study_id="cb_svdpp_execution_smoke_v1",
        repo_root=repo_root,
    )

    result = run_tuning_candidate_smoke(
        study_dir=repo_root / "artifacts" / "tuning" / "cb_svdpp_execution_smoke_v1",
        candidate_id=None,
        repo_root=repo_root,
        dry_run=False,
        processed_manifest=processed_manifest_path,
        runtime_config=runtime_config_path,
        device_config=device_config_path,
        train_ratio=split_config.train_ratio,
        validation_ratio=split_config.validation_ratio,
        split_seed=split_config.seed,
        model_seed=4,
        evaluate_test=False,
        use_split_cache=False,
        use_training_index_cache=False,
        use_cluster_artifact_cache=False,
        runner_kwargs={
            "command": "scripts/run_tuning_candidate_smoke.py --synthetic-cb-svdpp-execution-smoke",
            "services": build_experiment_services(
                git_snapshot_fn=lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": False},
            ),
        },
    )

    study_dir = repo_root / "artifacts" / "tuning" / "cb_svdpp_execution_smoke_v1"
    run_dir = Path(str(result["run_dir"]))
    candidate_manifests = list((study_dir / "candidates").glob("*/candidate_manifest.json"))

    assert result["execution_status"] == "succeeded"
    assert len(candidate_manifests) == 1
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "performance_profile.json").exists()
    assert (run_dir / "kernel_profile.json").exists()
    assert (run_dir / "run_manifest.json").exists()
    assert (study_dir / "reports" / "execution_summary.csv").exists()
    updated_manifest = json.loads(candidate_manifests[0].read_text(encoding="utf-8"))
    assert updated_manifest["execution_status"] == "succeeded"
    assert Path(updated_manifest["run_manifest_path"]).exists()
    assert Path(updated_manifest["metrics_path"]).exists()

    with (study_dir / "reports" / "candidate_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (study_dir / "reports" / "execution_summary.csv").open(encoding="utf-8", newline="") as handle:
        execution_rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert len(execution_rows) == 1
    row = rows[0]
    assert row["execution_status"] == "succeeded"
    assert row["run_id"]
    assert row["validation_rmse"]
    assert row["validation_mae"]
    assert row["fit_model_seconds"]
