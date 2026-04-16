import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from recsys_lab.experiments.common import SplitConfig
from recsys_lab.experiments.precomputed_index_reuse_benchmark import (
    run_precomputed_index_reuse_benchmark,
)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _prepare_synthetic_repo(tmp_path: Path, actual_repo_root: Path) -> tuple[Path, Path, Path]:
    repo_root = tmp_path / "repo"
    _write_text(repo_root / "AGENTS.md", "# Test Repo\n")
    _write_text(repo_root / "pyproject.toml", "[project]\nname = 'test-repo'\nversion = '0.0.0'\n")

    schema_dir = repo_root / "schema" / "reporting"
    schema_dir.mkdir(parents=True, exist_ok=True)
    benchmark_schema = actual_repo_root / "schema" / "reporting" / "benchmark_manifest.schema.json"
    run_schema = actual_repo_root / "schema" / "reporting" / "run_manifest.schema.json"
    _write_text(schema_dir / "benchmark_manifest.schema.json", benchmark_schema.read_text(encoding="utf-8"))
    _write_text(schema_dir / "run_manifest.schema.json", run_schema.read_text(encoding="utf-8"))

    (repo_root / "artifacts" / "benchmarks").mkdir(parents=True, exist_ok=True)
    (repo_root / "artifacts" / "runs").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "models").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "runtime" / "devices").mkdir(parents=True, exist_ok=True)

    _write_text(
        repo_root / "configs" / "models" / "svdpp.yaml",
        "model:\n  name: svdpp\n  scope: paper_inspired\ntraining:\n"
        "  latent_dim: 8\n  epochs: 4\n  learning_rate: 0.02\n"
        "  lambda_b: 0.01\n  lambda_p: 0.01\n  lambda_q: 0.01\n"
        "  lambda_y: 0.01\n  init_std: 0.05\n  dtype: float32\n"
        "  implicit_policy: ratings_as_implicit\n  training_backend: python\n",
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
    pq.write_table(
        pa.table(
            {
                "user_idx": pa.array([0, 0, 0, 1, 1, 1], type=pa.int32()),
                "item_idx": pa.array([0, 1, 2, 0, 1, 2], type=pa.int32()),
                "rating": pa.array([5.0, 4.5, 4.0, 3.0, 3.5, 2.5], type=pa.float32()),
            }
        ),
        interactions_path,
    )

    processed_manifest_path = processed_dir / "toy_manifest.json"
    processed_manifest = {
        "dataset_short_name": "ml_latest_small",
        "dataset_name": "Synthetic MovieLens Small",
        "split_family": "benchmark_random_v1",
        "preprocessing_family": "explicit_v1",
        "dtype": "float32",
        "counts": {
            "interactions": 6,
            "users": 2,
            "rated_items": 3,
        },
        "rating_range": {
            "min": 2.5,
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
        repo_root / "configs" / "runtime" / "devices" / "local.yaml",
    )


def test_run_precomputed_index_reuse_benchmark_writes_valid_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    repo_root, processed_manifest_path, device_config_path = _prepare_synthetic_repo(
        tmp_path,
        actual_repo_root,
    )
    runtime_config_path = repo_root / "configs" / "runtime" / "base.yaml"

    git_payload = {"commit": "abcdef1234567", "branch": "main", "dirty": False}
    monkeypatch.setattr(
        "recsys_lab.experiments.precomputed_index_reuse_benchmark.git_snapshot",
        lambda _repo_root: git_payload,
    )
    monkeypatch.setattr(
        "recsys_lab.experiments.svdpp.git_snapshot",
        lambda _repo_root: git_payload,
    )

    payload = run_precomputed_index_reuse_benchmark(
        processed_manifest_path=processed_manifest_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        measured_repeats=1,
        override_epochs=1,
        model_names=("svdpp",),
        split_config=SplitConfig(train_ratio=0.5, validation_ratio=0.25, seed=3),
        repo_root=repo_root,
        command="development benchmark precomputed index reuse compare --synthetic",
    )

    benchmark_manifest_path = Path(payload["benchmark_manifest"])
    summary_path = Path(payload["summary_path"])

    manifest = json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "completed"
    assert summary["dataset"] == "ml_latest_small"
    assert summary["models"]["svdpp"]["variants"]["reference_rebuild_indices"]["warmup"][
        "precomputed_index_reuse"
    ] is False
    assert summary["models"]["svdpp"]["variants"]["optimized_reuse_precomputed_indices"]["warmup"][
        "precomputed_index_reuse"
    ] is True
    assert len(summary["models"]["svdpp"]["variants"]["reference_rebuild_indices"]["measured_runs"]) == 1
    assert len(summary["models"]["svdpp"]["variants"]["optimized_reuse_precomputed_indices"]["measured_runs"]) == 1
