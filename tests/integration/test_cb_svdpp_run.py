import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from recsys_lab.data.processed import load_ratings_data_from_manifest
from recsys_lab.data.splitters import RatingsSplit
from recsys_lab.experiments.cb_svdpp import run_cb_svdpp_experiment
from recsys_lab.experiments.common import SplitConfig


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _prepare_synthetic_repo(tmp_path: Path, actual_repo_root: Path) -> tuple[Path, Path, Path, Path]:
    repo_root = tmp_path / "repo"
    _write_text(repo_root / "AGENTS.md", "# Test Repo\n")
    _write_text(repo_root / "pyproject.toml", "[project]\nname = 'test-repo'\nversion = '0.0.0'\n")

    schema_dir = repo_root / "schema" / "reporting"
    schema_dir.mkdir(parents=True, exist_ok=True)
    source_schema = actual_repo_root / "schema" / "reporting" / "run_manifest.schema.json"
    _write_text(schema_dir / "run_manifest.schema.json", source_schema.read_text(encoding="utf-8"))

    (repo_root / "artifacts" / "runs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "models").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "runtime" / "devices").mkdir(parents=True, exist_ok=True)

    _write_text(
        repo_root / "configs" / "models" / "cb_svdpp.yaml",
        "model:\n  name: cb_svdpp\n  scope: paper_inspired\ntraining:\n"
        "  latent_dim: 8\n  epochs: 8\n  learning_rate: 0.02\n"
        "  lambda_b: 0.01\n  lambda_p: 0.01\n  lambda_q: 0.01\n"
        "  lambda_y: 0.01\n  lambda_pC: 0.01\n  lambda_qC: 0.01\n"
        "  lambda_yC: 0.01\n  init_std: 0.05\n  dtype: float32\n"
        "  implicit_policy: ratings_as_implicit\nclustering:\n"
        "  n_user_clusters: 2\n  n_item_clusters: 2\n  alpha: 0.2\n"
        "  algorithm: kmeans\n  kmeans_n_init: 5\n",
    )
    _write_text(
        repo_root / "configs" / "runtime" / "base.yaml",
        "runtime:\n  default_precision_profile: performance_float32\n"
        "precision_profiles:\n  performance_float32:\n    dtype: float32\n",
    )
    _write_text(
        repo_root / "configs" / "runtime" / "devices" / "local.yaml",
        "device_profile:\n  name: local_test\nthreading:\n  omp_num_threads: 1\n  blas_threads: 1\n"
        "precision:\n  default_dtype: float32\n",
    )

    processed_dir = repo_root / "data" / "processed" / "ml_latest_small"
    processed_dir.mkdir(parents=True, exist_ok=True)
    interactions_path = processed_dir / "toy_interactions.parquet"

    table = pa.table(
        {
            "user_idx": pa.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], type=pa.int32()),
            "item_idx": pa.array([0, 1, 2, 0, 2, 3, 1, 2, 3, 0, 1, 3], type=pa.int32()),
            "rating": pa.array([5.0, 4.5, 4.0, 4.5, 4.0, 3.5, 2.5, 3.0, 3.5, 1.0, 1.5, 2.0], type=pa.float32()),
        }
    )
    pq.write_table(table, interactions_path)

    processed_manifest_path = processed_dir / "toy_manifest.json"
    processed_manifest = {
        "dataset_short_name": "ml_latest_small",
        "dataset_name": "Synthetic MovieLens Small",
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

    return (
        repo_root,
        processed_manifest_path,
        repo_root / "configs" / "models" / "cb_svdpp.yaml",
        repo_root / "configs" / "runtime" / "devices" / "local.yaml",
    )


def test_run_cb_svdpp_experiment_writes_valid_run_artifacts(tmp_path: Path, monkeypatch) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    repo_root, processed_manifest_path, model_config_path, device_config_path = _prepare_synthetic_repo(
        tmp_path,
        actual_repo_root,
    )
    runtime_config_path = repo_root / "configs" / "runtime" / "base.yaml"

    monkeypatch.setattr(
        "recsys_lab.experiments.cb_svdpp.git_snapshot",
        lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": False},
    )

    payload = run_cb_svdpp_experiment(
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_config=SplitConfig(train_ratio=0.5, validation_ratio=0.25, seed=3),
        model_seed=4,
        repo_root=repo_root,
        command="recsys-lab train-cb-svdpp --synthetic",
    )

    run_manifest_path = Path(payload["run_manifest"])
    metrics_path = Path(payload["run_dir"]) / "metrics.json"
    stdout_log_path = Path(payload["run_dir"]) / "stdout.log"

    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "completed"
    assert manifest["model"]["name"] == "cb_svdpp"
    assert metrics["metrics"]["validation_rmse"] >= 0.0
    assert metrics["model"]["clustering"]["r_star_role"] == "diagnostic_only"
    assert metrics["model"]["clustering"]["train_only_assignments"] is True
    assert metrics["system_metrics"]["train_time_total"] >= metrics["timing"]["training_wall_clock_seconds"]
    assert metrics["system_metrics"]["train_time_per_epoch"] > 0.0
    assert metrics["system_metrics"]["ratings_per_second_train"] > 0.0
    assert metrics["system_metrics"]["ratings_per_second_inference"] > 0.0
    assert metrics["system_metrics"]["peak_memory_mb"] > 0.0
    assert metrics["system_metrics"]["peak_memory_delta_mb"] >= 0.0
    assert metrics["system_metrics"]["model_size_mb"] > 0.0
    assert metrics["system_metrics"]["cluster_induction_wall_clock_seconds"] > 0.0
    assert metrics["system_metrics"]["main_training_wall_clock_seconds"] > 0.0
    assert len(metrics["system_metrics"]["epoch_durations_seconds"]) == 8
    assert metrics["timing"]["inference_wall_clock_seconds"] >= 0.0
    assert stdout_log_path.exists()


def test_run_cb_svdpp_experiment_supports_official_split_family_without_test_eval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    repo_root, processed_manifest_path, model_config_path, device_config_path = _prepare_synthetic_repo(
        tmp_path,
        actual_repo_root,
    )
    runtime_config_path = repo_root / "configs" / "runtime" / "base.yaml"
    ratings_data = load_ratings_data_from_manifest(processed_manifest_path)

    monkeypatch.setattr(
        "recsys_lab.experiments.cb_svdpp.git_snapshot",
        lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": False},
    )

    def _fake_official_split(ratings_data, *, processed_manifest_path, fold_index):
        del processed_manifest_path, fold_index
        return RatingsSplit(
            train=ratings_data.subset([0, 1, 3, 4, 6, 7, 9, 10], name="train"),
            validation=None,
            test=ratings_data.subset([2, 5, 8, 11], name="test"),
        )

    monkeypatch.setattr(
        "recsys_lab.experiments.cb_svdpp.official_ml100k_paper_faithful_split",
        _fake_official_split,
    )

    payload = run_cb_svdpp_experiment(
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_config=SplitConfig(train_ratio=0.8, validation_ratio=0.1, seed=2),
        model_seed=5,
        repo_root=repo_root,
        command="recsys-lab train-cb-svdpp --synthetic-official",
        split_family="paper_faithful_ml100k_v1",
        evaluate_test=False,
    )

    metrics_path = Path(payload["run_dir"]) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert metrics["split"]["family"] == "paper_faithful_ml100k_v1"
    assert metrics["split"]["test_metrics_available"] is False
    assert metrics["metrics"]["validation_rmse"] is None
    assert metrics["metrics"]["test_rmse"] is None
