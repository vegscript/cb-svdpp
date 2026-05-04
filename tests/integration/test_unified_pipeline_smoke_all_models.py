from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from recsys_lab.data.processed import load_ratings_data_from_manifest
from recsys_lab.data.splitters import random_split_with_train_coverage
from recsys_lab.experiments.common import SplitConfig
from recsys_lab.experiments.unified_runner import (
    build_experiment_services,
    run_unified_experiment,
)
from tests.support.model_configs import model_config_yaml

MODEL_NAMES = (
    "biased_mf",
    "svdpp",
    "asymmetric_svd",
    "asvdpp",
    "cb_svdpp",
    "cb_asvdpp",
)

EXPECTED_REQUIRED_ARTIFACTS = {
    "biased_mf": [],
    "svdpp": ["user_history_index"],
    "asymmetric_svd": ["user_history_index", "explicit_feedback_index"],
    "asvdpp": ["user_history_index", "explicit_feedback_index"],
    "cb_svdpp": ["user_history_index", "cluster_artifacts", "user_cluster_history_index"],
    "cb_asvdpp": [
        "user_history_index",
        "explicit_feedback_index",
        "cluster_artifacts",
        "user_cluster_history_index",
    ],
}

EXPECTED_AVAILABLE_ARTIFACTS = {
    "biased_mf": [],
    "svdpp": ["user_history_index"],
    "asymmetric_svd": ["user_history_index", "explicit_feedback_index"],
    "asvdpp": ["user_history_index", "explicit_feedback_index"],
    "cb_svdpp": ["user_history_index", "user_cluster_history_index", "cluster_artifacts"],
    "cb_asvdpp": [
        "user_history_index",
        "explicit_feedback_index",
        "user_cluster_history_index",
        "cluster_artifacts",
    ],
}

EXPECTED_RATING_METRIC_SUFFIXES = (
    "rmse",
    "mae",
    "residual_mean",
    "residual_std",
    "abs_error_p50",
    "abs_error_p90",
    "abs_error_p95",
    "abs_error_max",
    "prediction_mean",
    "prediction_std",
    "prediction_min",
    "prediction_max",
    "prediction_below_rating_min_rate",
    "prediction_above_rating_max_rate",
    "prediction_out_of_range_rate",
)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_runtime_configs(repo_root: Path) -> tuple[Path, Path]:
    runtime_config_path = repo_root / "configs" / "runtime" / "base.yaml"
    device_config_path = repo_root / "configs" / "runtime" / "devices" / "local.yaml"
    _write_text(
        runtime_config_path,
        "runtime:\n"
        "  default_precision_profile: performance_float32\n"
        "precision_profiles:\n"
        "  performance_float32:\n"
        "    dtype: float32\n",
    )
    _write_text(
        device_config_path,
        "device_profile:\n"
        "  name: local_test\n"
        "threading:\n"
        "  omp_num_threads: 1\n"
        "  blas_threads: 1\n"
        "precision:\n"
        "  default_dtype: float32\n",
    )
    return runtime_config_path, device_config_path


def _write_model_configs(repo_root: Path) -> dict[str, Path]:
    model_config_paths: dict[str, Path] = {}
    for model_name in MODEL_NAMES:
        training = {"latent_dim": 2, "epochs": 1, "dtype": "float32"}
        if model_name in {"biased_mf", "svdpp"}:
            training["training_backend"] = "python"
        clustering = None
        if model_name.startswith("cb_"):
            clustering = {
                "n_user_clusters": 2,
                "n_item_clusters": 2,
                "alpha": 0.2,
                "algorithm": "kmeans",
                "kmeans_n_init": 2,
            }
        model_config_path = repo_root / "configs" / "models" / f"{model_name}.yaml"
        _write_text(
            model_config_path,
            model_config_yaml(
                model_name,
                training=training,
                clustering=clustering,
                metadata={"purpose": "all-model unified pipeline smoke fixture"},
            ),
        )
        model_config_paths[model_name] = model_config_path
    return model_config_paths


def _write_toy_processed_dataset(repo_root: Path) -> Path:
    processed_dir = repo_root / "data" / "processed" / "ml_latest_small"
    interactions_path = processed_dir / "interactions.parquet"
    processed_manifest_path = processed_dir / "manifest.json"

    user_ids = np.repeat(np.arange(4, dtype=np.int32), 4)
    item_ids = np.tile(np.arange(4, dtype=np.int32), 4)
    ratings = np.asarray(
        [
            5.0,
            4.5,
            4.0,
            3.5,
            4.5,
            4.0,
            3.5,
            3.0,
            2.5,
            3.0,
            4.0,
            4.5,
            1.0,
            1.5,
            2.0,
            2.5,
        ],
        dtype=np.float32,
    )
    table = pa.table(
        {
            "user_idx": pa.array(user_ids, type=pa.int32()),
            "item_idx": pa.array(item_ids, type=pa.int32()),
            "rating": pa.array(ratings, type=pa.float32()),
        }
    )
    processed_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, interactions_path)

    processed_manifest = {
        "dataset_short_name": "ml_latest_small",
        "dataset_name": "Synthetic All-Model Smoke Dataset",
        "split_family": "benchmark_random_v1",
        "preprocessing_family": "explicit_v1",
        "counts": {
            "users": 4,
            "rated_items": 4,
        },
        "rating_range": {
            "min": 1.0,
            "max": 5.0,
        },
        "artifacts": {
            "interactions": str(interactions_path),
        },
    }
    _write_text(processed_manifest_path, json.dumps(processed_manifest, indent=2))
    return processed_manifest_path


def _prepare_toy_repo(tmp_path: Path, actual_repo_root: Path) -> tuple[Path, Path, Path, Path, dict[str, Path]]:
    repo_root = tmp_path / "repo"
    _write_text(repo_root / "AGENTS.md", "# Test Repo\n")
    _write_text(repo_root / "pyproject.toml", "[project]\nname = 'test-repo'\nversion = '0.0.0'\n")

    source_schema = actual_repo_root / "schema" / "reporting" / "run_manifest.schema.json"
    _write_text(
        repo_root / "schema" / "reporting" / "run_manifest.schema.json",
        source_schema.read_text(encoding="utf-8"),
    )
    (repo_root / "artifacts" / "runs").mkdir(parents=True, exist_ok=True)

    processed_manifest_path = _write_toy_processed_dataset(repo_root)
    runtime_config_path, device_config_path = _write_runtime_configs(repo_root)
    model_config_paths = _write_model_configs(repo_root)
    return repo_root, processed_manifest_path, runtime_config_path, device_config_path, model_config_paths


def _assert_train_coverage(processed_manifest_path: Path, split_config: SplitConfig) -> None:
    ratings_data = load_ratings_data_from_manifest(processed_manifest_path)
    split = random_split_with_train_coverage(
        ratings_data,
        train_ratio=split_config.train_ratio,
        validation_ratio=split_config.validation_ratio,
        seed=split_config.seed,
    )

    assert set(split.train.user_ids.tolist()) == {0, 1, 2, 3}
    assert set(split.train.item_ids.tolist()) == {0, 1, 2, 3}
    assert split.validation is not None
    assert len(split.validation) > 0
    assert len(split.test) > 0


def test_unified_pipeline_smoke_trains_all_models_without_model_mocks(tmp_path: Path) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    repo_root, processed_manifest_path, runtime_config_path, device_config_path, model_config_paths = _prepare_toy_repo(
        tmp_path,
        actual_repo_root,
    )
    split_config = SplitConfig(train_ratio=0.5, validation_ratio=0.25, seed=3)
    _assert_train_coverage(processed_manifest_path, split_config)

    services = build_experiment_services(
        git_snapshot_fn=lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": False},
    )

    for model_name in MODEL_NAMES:
        payload = run_unified_experiment(
            processed_manifest_path=processed_manifest_path,
            model_config_path=model_config_paths[model_name],
            runtime_config_path=runtime_config_path,
            device_config_path=device_config_path,
            split_config=split_config,
            model_seed=4,
            repo_root=repo_root,
            model_name=model_name,
            split_family="benchmark_random_v1",
            evaluate_test=True,
            use_split_cache=False,
            use_training_index_cache=False,
            use_cluster_artifact_cache=False,
            command=f"recsys-lab train --model {model_name} --synthetic-all-model-smoke",
            services=services,
        )

        run_dir = Path(payload["run_dir"])
        run_manifest_path = Path(payload["run_manifest"])
        metrics_path = run_dir / "metrics.json"
        config_snapshot_path = run_dir / "config_snapshot.yaml"

        assert run_manifest_path.exists()
        assert metrics_path.exists()
        assert config_snapshot_path.exists()

        manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        assert manifest["status"] == "completed"
        assert manifest["model"]["name"] == model_name
        assert manifest["profiling"] == metrics["profiling"]
        assert manifest["caches"] == metrics["caches"]
        assert metrics["metrics"]["train_rmse"] is not None
        assert metrics["metrics"]["validation_rmse"] is not None
        assert metrics["metrics"]["test_rmse"] is not None
        for split_name in ("train", "validation", "test"):
            split_metrics = metrics["metrics"][split_name]
            assert isinstance(split_metrics, dict)
            assert split_metrics["rmse"] == metrics["metrics"][f"{split_name}_rmse"]
            assert split_metrics["mae"] == metrics["metrics"][f"{split_name}_mae"]
            assert split_metrics["abs_error_p90"] == metrics["metrics"][f"{split_name}_abs_error_p90"]
            assert (
                split_metrics["prediction_out_of_range_rate"]
                == metrics["metrics"][f"{split_name}_prediction_out_of_range_rate"]
            )
            for metric_suffix in EXPECTED_RATING_METRIC_SUFFIXES:
                assert metrics["metrics"][f"{split_name}_{metric_suffix}"] is not None
                assert split_metrics[metric_suffix] is not None
        assert metrics["profiling"]["stage_count"] > 0
        assert "split" in metrics["caches"]
        assert metrics["model"]["requirements"]["required_artifacts"] == EXPECTED_REQUIRED_ARTIFACTS[model_name]
        assert metrics["model"]["available_fit_artifacts"] == EXPECTED_AVAILABLE_ARTIFACTS[model_name]

        if model_name.startswith("cb_"):
            cb_semantics = metrics["cb_semantics"]
            assert cb_semantics["cb_claim_eligible"] is False
            cb_diagnostics = metrics["cb_diagnostics"]
            assert cb_diagnostics["diagnostic_claim_ready"] is False
            assert cb_diagnostics["cluster_artifacts_present"] is True
            assert cb_diagnostics["missing_expected_artifacts"] == []
            assert cb_diagnostics["empty_user_clusters"] == 0
            assert cb_diagnostics["empty_item_clusters"] == 0
            assert cb_diagnostics["user_cluster_size_min"] > 0
            assert cb_diagnostics["item_cluster_size_min"] > 0

        if model_name == "cb_asvdpp":
            assert "explicit_feedback_index" in metrics["model"]["requirements"]["required_artifacts"]
            assert "explicit_feedback_index" in metrics["model"]["available_fit_artifacts"]
