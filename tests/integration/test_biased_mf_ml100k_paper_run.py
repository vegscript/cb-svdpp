import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from recsys_lab.experiments.biased_mf import run_biased_mf_experiment
from recsys_lab.experiments.common import SplitConfig


def _write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding, newline="\n")


def _prepare_synthetic_ml100k_repo(tmp_path: Path, actual_repo_root: Path) -> tuple[Path, Path, Path, Path]:
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
        repo_root / "configs" / "models" / "biased_mf.yaml",
        "model:\n  name: biased_mf\n  scope: paper_inspired\ntraining:\n"
        "  latent_dim: 8\n  epochs: 8\n  learning_rate: 0.02\n"
        "  lambda_b: 0.01\n  lambda_p: 0.01\n  lambda_q: 0.01\n"
        "  init_std: 0.05\n  dtype: float32\n",
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

    raw_dir = repo_root / "data" / "raw" / "ml100k" / "ml-100k"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_text(
        raw_dir / "u1.base",
        "1\t10\t4\t1000\n2\t10\t3\t1001\n",
        encoding="latin-1",
    )
    _write_text(
        raw_dir / "u1.test",
        "1\t20\t5\t1002\n2\t20\t2\t1003\n",
        encoding="latin-1",
    )

    processed_dir = repo_root / "data" / "processed" / "ml100k"
    processed_dir.mkdir(parents=True, exist_ok=True)
    interactions_path = processed_dir / "toy_interactions.parquet"
    table = pa.table(
        {
            "user_idx": pa.array([0, 0, 1, 1], type=pa.int32()),
            "item_idx": pa.array([0, 1, 0, 1], type=pa.int32()),
            "rating": pa.array([4.0, 5.0, 3.0, 2.0], type=pa.float32()),
            "timestamp": pa.array([1000, 1002, 1001, 1003], type=pa.int64()),
            "raw_user_id": pa.array([1, 1, 2, 2], type=pa.int32()),
            "raw_item_id": pa.array([10, 20, 10, 20], type=pa.int32()),
        }
    )
    pq.write_table(table, interactions_path)

    processed_manifest_path = processed_dir / "ml100k_manifest.json"
    processed_manifest = {
        "dataset_short_name": "ml100k",
        "dataset_name": "MovieLens 100K",
        "split_family": "benchmark_random_v1",
        "preprocessing_family": "explicit_v1",
        "source": {
            "raw_dir": str(raw_dir),
            "format_family": "legacy_100k",
        },
        "counts": {
            "users": 2,
            "rated_items": 2,
        },
        "rating_range": {
            "min": 2.0,
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
        repo_root / "configs" / "models" / "biased_mf.yaml",
        repo_root / "configs" / "runtime" / "devices" / "local.yaml",
    )


def test_run_biased_mf_experiment_supports_ml100k_paper_faithful_split(
    tmp_path: Path,
    monkeypatch,
) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    repo_root, processed_manifest_path, model_config_path, device_config_path = _prepare_synthetic_ml100k_repo(
        tmp_path,
        actual_repo_root,
    )
    runtime_config_path = repo_root / "configs" / "runtime" / "base.yaml"

    monkeypatch.setattr(
        "recsys_lab.experiments.biased_mf.git_snapshot",
        lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": False},
    )

    payload = run_biased_mf_experiment(
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_config=SplitConfig(train_ratio=0.8, validation_ratio=0.1, seed=1),
        model_seed=4,
        repo_root=repo_root,
        command="recsys-lab train-biased-mf --synthetic",
        split_family="paper_faithful_ml100k_v1",
    )

    run_manifest_path = Path(payload["run_manifest"])
    metrics_path = Path(payload["run_dir"]) / "metrics.json"

    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "completed"
    assert manifest["dataset"]["split_family"] == "paper_faithful_ml100k_v1"
    assert manifest["dataset"]["split_id"] == "paper_faithful_ml100k_v1_u1"
    assert metrics["model"]["config"]["training_backend"] == "auto"
    assert metrics["model"]["training_backend_effective"] in {"python", "numba"}
    assert metrics["split"]["has_validation"] is False
    assert metrics["metrics"]["validation_rmse"] is None
    assert metrics["metrics"]["test_rmse"] >= 0.0
