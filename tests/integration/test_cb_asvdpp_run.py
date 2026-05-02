import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from recsys_lab.data.splitters import RatingsSplit
from recsys_lab.experiments.cb_asvdpp import run_cb_asvdpp_experiment
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
        repo_root / "configs" / "models" / "cb_asvdpp.yaml",
        "model:\n  name: cb_asvdpp\n  scope: paper_inspired\ntraining:\n"
        "  latent_dim: 8\n  epochs: 8\n  learning_rate: 0.02\n"
        "  lambda_b: 0.01\n  lambda_p: 0.01\n  lambda_q: 0.01\n"
        "  lambda_x: 0.01\n  lambda_y: 0.01\n"
        "  lambda_pC: 0.01\n  lambda_qC: 0.01\n  lambda_xC: 0.01\n  lambda_yC: 0.01\n"
        "  init_std: 0.05\n  dtype: float32\n  implicit_policy: ratings_as_implicit\n"
        "  residual_weight_contract: detached\nclustering:\n"
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
        repo_root / "configs" / "models" / "cb_asvdpp.yaml",
        repo_root / "configs" / "runtime" / "devices" / "local.yaml",
    )


def test_run_cb_asvdpp_experiment_writes_valid_run_artifacts(tmp_path: Path, monkeypatch) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    repo_root, processed_manifest_path, model_config_path, device_config_path = _prepare_synthetic_repo(
        tmp_path,
        actual_repo_root,
    )
    runtime_config_path = repo_root / "configs" / "runtime" / "base.yaml"

    monkeypatch.setattr(
        "recsys_lab.experiments.cb_asvdpp.git_snapshot",
        lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": False},
    )

    payload = run_cb_asvdpp_experiment(
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_config=SplitConfig(train_ratio=0.5, validation_ratio=0.25, seed=3),
        model_seed=4,
        repo_root=repo_root,
    )

    run_manifest_path = Path(payload["run_manifest"])
    metrics_path = Path(payload["run_dir"]) / "metrics.json"
    stdout_log_path = Path(payload["run_dir"]) / "stdout.log"

    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "completed"
    assert manifest["model"]["name"] == "cb_asvdpp"
    assert "--processed-manifest" not in manifest["command"]
    assert "data/processed/ml_latest_small/toy_manifest.json" in manifest["command"]
    assert "--train-ratio 0.5" in manifest["command"]
    assert "--validation-ratio 0.25" in manifest["command"]
    assert "--disable-cluster-artifact-cache" in manifest["command"]
    assert manifest["profiling"] == metrics["profiling"]
    assert manifest["caches"] == metrics["caches"]
    assert metrics["caches"]["cluster_artifacts"]["status"] == "disabled"
    assert metrics["caches"]["user_cluster_history"]["status"] == "disabled"
    stage_names = [stage["name"] for stage in metrics["profiling"]["stages"]]
    assert metrics["profiling"]["profile_version"] == "stage_profile_v1"
    assert metrics["profiling"]["stage_count"] == len(stage_names)
    assert metrics["profiling"]["total_profiled_wall_clock_seconds"] > 0.0
    assert {
        "data_load",
        "split_resolution",
        "config_build",
        "cluster_induction",
        "explicit_feedback_index_build",
        "user_history_index_build",
        "user_cluster_history_build",
        "model_initialization",
        "main_training",
        "inference_train",
        "inference_validation",
        "inference_test",
    }.issubset(stage_names)
    assert metrics["metrics"]["validation_rmse"] >= 0.0
    assert metrics["model"]["clustering"]["r_star_role"] == "diagnostic_only"
    assert metrics["model"]["clustering"]["train_only_assignments"] is True
    assert metrics["model"]["explicit_summary"]["users_with_explicit_history"] == 4
    assert metrics["system_metrics"]["train_time_total"] >= metrics["timing"]["training_wall_clock_seconds"]
    assert metrics["system_metrics"]["cluster_induction_wall_clock_seconds"] > 0.0
    assert metrics["system_metrics"]["main_training_wall_clock_seconds"] > 0.0
    assert len(metrics["system_metrics"]["epoch_durations_seconds"]) == 8
    assert metrics["timing"]["inference_wall_clock_seconds"] >= 0.0
    assert "stage_profile stage_count=" in stdout_log_path.read_text(encoding="utf-8")


def test_run_cb_asvdpp_experiment_supports_official_split_family_without_test_eval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    repo_root, processed_manifest_path, model_config_path, device_config_path = _prepare_synthetic_repo(
        tmp_path,
        actual_repo_root,
    )
    runtime_config_path = repo_root / "configs" / "runtime" / "base.yaml"
    monkeypatch.setattr(
        "recsys_lab.experiments.cb_asvdpp.git_snapshot",
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
        "recsys_lab.experiments.cb_asvdpp.official_ml100k_paper_faithful_split",
        _fake_official_split,
    )

    payload = run_cb_asvdpp_experiment(
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_config=SplitConfig(train_ratio=0.8, validation_ratio=0.1, seed=2),
        model_seed=5,
        repo_root=repo_root,
        command="recsys-lab train-cb-asvdpp --synthetic-official",
        split_family="paper_faithful_ml100k_v1",
        evaluate_test=False,
    )

    metrics_path = Path(payload["run_dir"]) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert metrics["split"]["family"] == "paper_faithful_ml100k_v1"
    assert metrics["caches"]["cluster_artifacts"]["status"] == "disabled"
    assert metrics["caches"]["user_cluster_history"]["status"] == "disabled"
    assert metrics["split"]["test_metrics_available"] is False
    assert "inference_test" not in [stage["name"] for stage in metrics["profiling"]["stages"]]
    assert metrics["metrics"]["validation_rmse"] is None
    assert metrics["metrics"]["test_rmse"] is None
