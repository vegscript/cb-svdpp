import json
from pathlib import Path

import pytest

from recsys_lab.experiments.ml100k_paper_multiseed_benchmark import run_ml100k_paper_multiseed_benchmark


def _write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding, newline="\n")


def _prepare_repo(tmp_path: Path, actual_repo_root: Path) -> tuple[Path, Path, Path, Path, Path]:
    repo_root = tmp_path / "repo"
    _write_text(repo_root / "AGENTS.md", "# Test Repo\n")
    _write_text(repo_root / "pyproject.toml", "[project]\nname = 'test-repo'\nversion = '0.0.0'\n")

    schema_dir = repo_root / "schema" / "reporting"
    schema_dir.mkdir(parents=True, exist_ok=True)
    source_schema = actual_repo_root / "schema" / "reporting" / "benchmark_manifest.schema.json"
    _write_text(schema_dir / "benchmark_manifest.schema.json", source_schema.read_text(encoding="utf-8"))

    (repo_root / "artifacts" / "benchmarks").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "models").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "runtime" / "devices").mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "processed" / "ml100k").mkdir(parents=True, exist_ok=True)

    processed_manifest_path = (
        repo_root / "data" / "processed" / "ml100k" / "ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json"
    )
    model_config_path = repo_root / "configs" / "models" / "biased_mf.yaml"
    runtime_config_path = repo_root / "configs" / "runtime" / "base.yaml"
    device_config_path = repo_root / "configs" / "runtime" / "devices" / "local.yaml"

    _write_text(
        processed_manifest_path,
        json.dumps(
            {
                "dataset_short_name": "ml100k",
                "artifacts": {"interactions": "data/processed/ml100k/interactions.parquet"},
            },
            indent=2,
        ),
    )
    _write_text(model_config_path, "model:\n  name: biased_mf\ntraining:\n  dtype: float32\n")
    _write_text(
        runtime_config_path,
        "runtime:\n  default_precision_profile: performance_float32\n"
        "precision_profiles:\n  performance_float32:\n    dtype: float32\n",
    )
    _write_text(
        device_config_path,
        "device_profile:\n  name: local_test\nthreading:\n  omp_num_threads: 1\n  blas_threads: 1\n"
        "precision:\n  default_dtype: float32\n",
    )
    return repo_root, processed_manifest_path, model_config_path, runtime_config_path, device_config_path


def _write_seed_benchmark(
    *,
    repo_root: Path,
    processed_manifest_path: Path,
    model_config_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    seed: int,
    test_rmse_mean: float,
    train_time_mean: float,
    benchmark_tag: str = "a",
    git_commit: str = "abcdef1234567",
    git_dirty: bool = False,
) -> None:
    benchmark_id = f"2026-04-13T12000{seed}Z_ml100k_paper_faithful_biased_mf_local_test_{benchmark_tag}"
    benchmark_dir = repo_root / "artifacts" / "benchmarks" / benchmark_id
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    benchmark_manifest_path = benchmark_dir / "benchmark_manifest.json"
    summary_path = benchmark_dir / "summary.json"
    config_snapshot_path = benchmark_dir / "config_snapshot.yaml"

    folds = []
    run_ids = []
    run_manifest_paths = []
    for fold_index in range(1, 6):
        run_id = f"2026-04-13T12{seed:02d}{fold_index:02d}Z_ml100k_biased_mf_local_test_s{seed:03d}"
        run_ids.append(run_id)
        run_manifest_paths.append(f"artifacts/runs/{run_id}/run_manifest.json")
        folds.append(
            {
                "fold": f"u{fold_index}",
                "run_id": run_id,
                "train_rmse": 0.55 + seed * 0.001 + fold_index * 0.0001,
                "test_rmse": test_rmse_mean + fold_index * 0.0001,
                "training_wall_clock_seconds": train_time_mean + fold_index,
            }
        )

    _write_text(
        config_snapshot_path,
        "benchmark_id: test\n"
        f"model_seed: {seed}\n"
        f"processed_manifest: {str(processed_manifest_path.relative_to(repo_root)).replace(chr(92), '/')}\n"
        f"model_config: {str(model_config_path.relative_to(repo_root)).replace(chr(92), '/')}\n"
        f"runtime_config: {str(runtime_config_path.relative_to(repo_root)).replace(chr(92), '/')}\n"
        f"device_config: {str(device_config_path.relative_to(repo_root)).replace(chr(92), '/')}\n",
    )
    _write_text(
        summary_path,
        json.dumps(
            {
                "benchmark_id": benchmark_id,
                "benchmark_scope": "paper_faithful_ml100k_v1_biased_mf_u1_u5",
                "dataset": "ml100k",
                "split_family": "paper_faithful_ml100k_v1",
                "model": "biased_mf",
                "folds": folds,
                "aggregate": {
                    "train_rmse": {
                        "mean": 0.55 + seed * 0.001,
                        "std": 0.001,
                        "min": 0.549,
                        "max": 0.551,
                    },
                    "test_rmse": {
                        "mean": test_rmse_mean,
                        "std": 0.002,
                        "min": test_rmse_mean - 0.001,
                        "max": test_rmse_mean + 0.001,
                    },
                    "training_wall_clock_seconds": {
                        "mean": train_time_mean,
                        "std": 2.0,
                        "min": train_time_mean - 1.0,
                        "max": train_time_mean + 1.0,
                    },
                },
            },
            indent=2,
        ),
    )
    _write_text(
        benchmark_manifest_path,
        json.dumps(
            {
                "manifest_version": "v1",
                "kind": "benchmark_manifest",
                "generated_at_utc": f"2026-04-13T12000{seed}Z",
                "benchmark_id": benchmark_id,
                "status": "completed",
                "benchmark_scope": "paper_faithful_ml100k_v1_biased_mf_u1_u5",
                "command": "recsys-lab benchmark-ml100k-paper --synthetic",
                "cwd": ".",
                "git": {
                    "commit": git_commit,
                    "branch": "main",
                    "dirty": git_dirty,
                },
                "runtime": {
                    "device_profile": "local_test",
                    "python_version": "3.11.9",
                    "dtype": "float32",
                    "threading": {"omp_num_threads": 1, "blas_threads": 1},
                },
                "inputs": {
                    "run_ids": run_ids,
                    "run_manifest_paths": run_manifest_paths,
                },
                "artifacts": {
                    "summary": str(summary_path.relative_to(repo_root)).replace("\\", "/"),
                    "tables": [],
                    "stdout_log": str((benchmark_dir / "stdout.log").relative_to(repo_root)).replace("\\", "/"),
                },
                "timing": {
                    "started_at_utc": f"2026-04-13T12000{seed}Z",
                    "finished_at_utc": f"2026-04-13T12010{seed}Z",
                },
            },
            indent=2,
        ),
    )
    _write_text(benchmark_dir / "stdout.log", "completed\n")


def test_run_ml100k_paper_multiseed_benchmark_aggregates_seed_benchmarks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    repo_root, processed_manifest_path, model_config_path, runtime_config_path, device_config_path = (
        _prepare_repo(tmp_path, actual_repo_root)
    )
    _write_seed_benchmark(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        seed=1,
        test_rmse_mean=0.93,
        train_time_mean=100.0,
    )
    _write_seed_benchmark(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        seed=2,
        test_rmse_mean=0.92,
        train_time_mean=110.0,
    )
    _write_seed_benchmark(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        seed=3,
        test_rmse_mean=0.91,
        train_time_mean=120.0,
    )

    monkeypatch.setattr(
        "recsys_lab.experiments.ml100k_paper_multiseed_benchmark.git_snapshot",
        lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": False},
    )

    payload = run_ml100k_paper_multiseed_benchmark(
        model_name="biased_mf",
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        model_seeds=[1, 2, 3],
        repo_root=repo_root,
        command="recsys-lab benchmark-ml100k-paper-multiseed --synthetic",
    )

    summary = json.loads((Path(payload["benchmark_dir"]) / "summary.json").read_text(encoding="utf-8"))

    assert summary["aggregate"]["seed_level"]["test_rmse"]["mean"] == 0.92
    assert summary["aggregate"]["fold_run_level"]["test_rmse"]["mean"] > 0.92
    assert len(summary["per_seed"]) == 3


def test_run_ml100k_paper_multiseed_benchmark_accepts_explicit_benchmark_manifest_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    repo_root, processed_manifest_path, model_config_path, runtime_config_path, device_config_path = (
        _prepare_repo(tmp_path, actual_repo_root)
    )
    _write_seed_benchmark(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        seed=1,
        test_rmse_mean=0.93,
        train_time_mean=100.0,
        benchmark_tag="a",
    )
    _write_seed_benchmark(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        seed=1,
        test_rmse_mean=0.931,
        train_time_mean=101.0,
        benchmark_tag="b",
    )
    _write_seed_benchmark(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        seed=2,
        test_rmse_mean=0.92,
        train_time_mean=110.0,
        benchmark_tag="a",
    )
    _write_seed_benchmark(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        seed=3,
        test_rmse_mean=0.91,
        train_time_mean=120.0,
        benchmark_tag="a",
    )

    monkeypatch.setattr(
        "recsys_lab.experiments.ml100k_paper_multiseed_benchmark.git_snapshot",
        lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": False},
    )

    with pytest.raises(ValueError, match="multiple matching seed benchmarks"):
        run_ml100k_paper_multiseed_benchmark(
            model_name="biased_mf",
            processed_manifest_path=processed_manifest_path,
            model_config_path=model_config_path,
            runtime_config_path=runtime_config_path,
            device_config_path=device_config_path,
            model_seeds=[1, 2, 3],
            repo_root=repo_root,
            command="recsys-lab benchmark-ml100k-paper-multiseed --synthetic",
        )

    payload = run_ml100k_paper_multiseed_benchmark(
        model_name="biased_mf",
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        model_seeds=[1, 2, 3],
        benchmark_manifest_paths=[
            repo_root
            / "artifacts"
            / "benchmarks"
            / "2026-04-13T120001Z_ml100k_paper_faithful_biased_mf_local_test_a"
            / "benchmark_manifest.json",
            repo_root
            / "artifacts"
            / "benchmarks"
            / "2026-04-13T120002Z_ml100k_paper_faithful_biased_mf_local_test_a"
            / "benchmark_manifest.json",
            repo_root
            / "artifacts"
            / "benchmarks"
            / "2026-04-13T120003Z_ml100k_paper_faithful_biased_mf_local_test_a"
            / "benchmark_manifest.json",
        ],
        repo_root=repo_root,
        command="recsys-lab benchmark-ml100k-paper-multiseed --synthetic --benchmark-manifest-paths explicit",
    )

    summary = json.loads((Path(payload["benchmark_dir"]) / "summary.json").read_text(encoding="utf-8"))
    assert summary["aggregate"]["seed_level"]["test_rmse"]["mean"] == 0.92


def test_run_ml100k_paper_multiseed_benchmark_rejects_mixed_git_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    repo_root, processed_manifest_path, model_config_path, runtime_config_path, device_config_path = (
        _prepare_repo(tmp_path, actual_repo_root)
    )
    _write_seed_benchmark(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        seed=1,
        test_rmse_mean=0.93,
        train_time_mean=100.0,
        benchmark_tag="a",
        git_commit="abcdef1234567",
        git_dirty=False,
    )
    _write_seed_benchmark(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        seed=2,
        test_rmse_mean=0.92,
        train_time_mean=110.0,
        benchmark_tag="a",
        git_commit="abcdef1234567",
        git_dirty=False,
    )
    _write_seed_benchmark(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        seed=3,
        test_rmse_mean=0.91,
        train_time_mean=120.0,
        benchmark_tag="a",
        git_commit="7654321fedcba",
        git_dirty=True,
    )

    monkeypatch.setattr(
        "recsys_lab.experiments.ml100k_paper_multiseed_benchmark.git_snapshot",
        lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": False},
    )

    with pytest.raises(ValueError, match="share identical git commit, branch, and dirty state"):
        run_ml100k_paper_multiseed_benchmark(
            model_name="biased_mf",
            processed_manifest_path=processed_manifest_path,
            model_config_path=model_config_path,
            runtime_config_path=runtime_config_path,
            device_config_path=device_config_path,
            model_seeds=[1, 2, 3],
            benchmark_manifest_paths=[
                repo_root
                / "artifacts"
                / "benchmarks"
                / "2026-04-13T120001Z_ml100k_paper_faithful_biased_mf_local_test_a"
                / "benchmark_manifest.json",
                repo_root
                / "artifacts"
                / "benchmarks"
                / "2026-04-13T120002Z_ml100k_paper_faithful_biased_mf_local_test_a"
                / "benchmark_manifest.json",
                repo_root
                / "artifacts"
                / "benchmarks"
                / "2026-04-13T120003Z_ml100k_paper_faithful_biased_mf_local_test_a"
                / "benchmark_manifest.json",
            ],
            repo_root=repo_root,
            command="recsys-lab benchmark-ml100k-paper-multiseed --synthetic --benchmark-manifest-paths explicit",
        )
