import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from recsys_lab.experiments.processed_data_loading_benchmark import (
    run_processed_data_loading_benchmark,
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
    source_schema = actual_repo_root / "schema" / "reporting" / "benchmark_manifest.schema.json"
    _write_text(schema_dir / "benchmark_manifest.schema.json", source_schema.read_text(encoding="utf-8"))

    (repo_root / "artifacts" / "benchmarks").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "runtime" / "devices").mkdir(parents=True, exist_ok=True)

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
                "user_idx": pa.array([0, 0, 1], type=pa.int32()),
                "item_idx": pa.array([0, 1, 0], type=pa.int32()),
                "rating": pa.array([4.0, 3.5, 5.0], type=pa.float32()),
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
            "interactions": 3,
            "users": 2,
            "rated_items": 2,
        },
        "rating_range": {
            "min": 3.5,
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


def test_run_processed_data_loading_benchmark_writes_valid_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    repo_root, processed_manifest_path, device_config_path = _prepare_synthetic_repo(
        tmp_path,
        actual_repo_root,
    )
    runtime_config_path = repo_root / "configs" / "runtime" / "base.yaml"

    monkeypatch.setattr(
        "recsys_lab.experiments.processed_data_loading_benchmark.git_snapshot",
        lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": False},
    )

    payload = run_processed_data_loading_benchmark(
        processed_manifest_path=processed_manifest_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        measured_repeats=2,
        repo_root=repo_root,
        command="development benchmark processed data loading compare --synthetic",
    )

    benchmark_manifest_path = Path(payload["benchmark_manifest"])
    summary_path = Path(payload["summary_path"])

    manifest = json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "completed"
    assert summary["dataset"] == "ml_latest_small"
    assert summary["scenarios"]["construct_only"]["row_count"] == 3
    assert summary["scenarios"]["construct_only"]["readout"]["delta"] == 0.0
    assert summary["scenarios"]["full_scan_checksum"]["readout"]["delta"] == 0.0
