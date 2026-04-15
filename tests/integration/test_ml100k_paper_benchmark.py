import json
from pathlib import Path

from recsys_lab.experiments.ml100k_paper_benchmark import run_ml100k_paper_benchmark


def _write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding, newline="\n")


def _prepare_synthetic_benchmark_repo(tmp_path: Path, actual_repo_root: Path) -> tuple[Path, Path, Path, Path, Path]:
    repo_root = tmp_path / "repo"
    _write_text(repo_root / "AGENTS.md", "# Test Repo\n")
    _write_text(repo_root / "pyproject.toml", "[project]\nname = 'test-repo'\nversion = '0.0.0'\n")

    schema_dir = repo_root / "schema" / "reporting"
    schema_dir.mkdir(parents=True, exist_ok=True)
    for schema_name in ("run_manifest.schema.json", "benchmark_manifest.schema.json"):
        source_schema = actual_repo_root / "schema" / "reporting" / schema_name
        _write_text(schema_dir / schema_name, source_schema.read_text(encoding="utf-8"))

    (repo_root / "artifacts" / "runs").mkdir(parents=True, exist_ok=True)
    (repo_root / "artifacts" / "benchmarks").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "models").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "runtime" / "devices").mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "processed" / "ml100k").mkdir(parents=True, exist_ok=True)

    model_config_path = repo_root / "configs" / "models" / "biased_mf.yaml"
    runtime_config_path = repo_root / "configs" / "runtime" / "base.yaml"
    device_config_path = repo_root / "configs" / "runtime" / "devices" / "local.yaml"
    processed_manifest_path = (
        repo_root / "data" / "processed" / "ml100k" / "ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json"
    )

    _write_text(
        model_config_path,
        "model:\n  name: biased_mf\n  scope: paper_inspired\ntraining:\n  dtype: float32\n",
    )
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
    _write_text(
        processed_manifest_path,
        json.dumps(
            {
                "dataset_short_name": "ml100k",
                "dataset_name": "MovieLens 100K",
                "preprocessing_family": "explicit_v1",
                "split_family": "benchmark_random_v1",
                "artifacts": {
                    "interactions": "data/processed/ml100k/interactions.parquet",
                },
            },
            indent=2,
        ),
    )
    return (
        repo_root,
        processed_manifest_path,
        model_config_path,
        runtime_config_path,
        device_config_path,
    )


def _write_completed_run(
    *,
    repo_root: Path,
    processed_manifest_path: Path,
    model_config_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    fold_index: int,
    model_seed: int,
    train_rmse: float,
    test_rmse: float,
    training_seconds: float,
    model_name: str = "biased_mf",
    cluster_induction_seconds: float | None = None,
) -> Path:
    run_id = f"2026-04-13T00000{fold_index}Z_ml100k_{model_name}_local_test_s{model_seed:03d}"
    run_dir = repo_root / "artifacts" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.json"
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    stdout_log_path = run_dir / "stdout.log"
    run_manifest_path = run_dir / "run_manifest.json"

    _write_text(
        config_snapshot_path,
        json.dumps(
            {
                "inputs": {
                    "processed_manifest": str(processed_manifest_path.relative_to(repo_root)).replace("\\", "/"),
                    "model_config": str(model_config_path.relative_to(repo_root)).replace("\\", "/"),
                    "runtime_config": str(runtime_config_path.relative_to(repo_root)).replace("\\", "/"),
                    "device_config": str(device_config_path.relative_to(repo_root)).replace("\\", "/"),
                },
                "loaded_configs": {
                    "processed_manifest": json.loads(processed_manifest_path.read_text(encoding="utf-8")),
                    "model": {
                        "model": {"name": model_name, "scope": "paper_inspired"},
                        "training": {"dtype": "float32"},
                    },
                    "runtime": {
                        "runtime": {"default_precision_profile": "performance_float32"},
                        "precision_profiles": {"performance_float32": {"dtype": "float32"}},
                    },
                    "device": {
                        "device_profile": {"name": "local_test"},
                        "threading": {"omp_num_threads": 1, "blas_threads": 1},
                        "precision": {"default_dtype": "float32"},
                    },
                },
            },
            indent=2,
        ),
    )
    _write_text(stdout_log_path, f"run_id={run_id}\n")
    _write_text(
        metrics_path,
        json.dumps(
            {
                "metrics": {
                    "train_rmse": train_rmse,
                    "validation_rmse": None,
                    "test_rmse": test_rmse,
                },
                "timing": {
                    "cluster_induction_wall_clock_seconds": cluster_induction_seconds,
                    "training_wall_clock_seconds": training_seconds,
                },
            },
            indent=2,
        ),
    )
    _write_text(
        run_manifest_path,
        json.dumps(
            {
                "manifest_version": "v1",
                "kind": "run_manifest",
                "generated_at_utc": f"2026-04-13T00000{fold_index}Z",
                "run_id": run_id,
                "status": "completed",
                "command": "recsys-lab train-biased-mf --synthetic",
                "cwd": ".",
                "git": {
                    "commit": "abcdef1234567",
                    "branch": "main",
                    "dirty": False,
                },
                "dataset": {
                    "short_name": "ml100k",
                    "source": "MovieLens 100K",
                    "version": "explicit_v1",
                    "split_family": "paper_faithful_ml100k_v1",
                    "split_id": f"paper_faithful_ml100k_v1_u{fold_index}",
                    "manifest_ref": str(processed_manifest_path.relative_to(repo_root)).replace("\\", "/"),
                },
                "model": {
                    "name": model_name,
                    "scope": "paper_inspired",
                    "config_ref": str(model_config_path.relative_to(repo_root)).replace("\\", "/"),
                },
                "runtime": {
                    "device_profile": "local_test",
                    "python_version": "3.11.9",
                    "platform": "test-platform",
                    "hostname": "test-host",
                    "dtype": "float32",
                    "threading": {
                        "omp_num_threads": 1,
                        "blas_threads": 1,
                    },
                },
                "seeds": [model_seed],
                "artifacts": {
                    "config_snapshot": str(config_snapshot_path.relative_to(repo_root)).replace("\\", "/"),
                    "metrics": str(metrics_path.relative_to(repo_root)).replace("\\", "/"),
                    "stdout_log": str(stdout_log_path.relative_to(repo_root)).replace("\\", "/"),
                },
                "timing": {
                    "started_at_utc": f"2026-04-13T00000{fold_index}Z",
                    "finished_at_utc": f"2026-04-13T00001{fold_index}Z",
                },
            },
            indent=2,
        ),
    )
    return run_manifest_path


def test_run_ml100k_paper_benchmark_reuses_existing_runs_and_aggregates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    (
        repo_root,
        processed_manifest_path,
        model_config_path,
        runtime_config_path,
        device_config_path,
    ) = _prepare_synthetic_benchmark_repo(tmp_path, actual_repo_root)

    _write_completed_run(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        fold_index=1,
        model_seed=1,
        train_rmse=0.810,
        test_rmse=0.910,
        training_seconds=10.0,
    )

    monkeypatch.setattr(
        "recsys_lab.experiments.ml100k_paper_benchmark.git_snapshot",
        lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": False},
    )

    calls: list[int] = []

    def _fake_runner(**kwargs):
        fold_index = int(kwargs["split_config"].seed)
        calls.append(fold_index)
        run_manifest_path = _write_completed_run(
            repo_root=repo_root,
            processed_manifest_path=processed_manifest_path,
            model_config_path=model_config_path,
            runtime_config_path=runtime_config_path,
            device_config_path=device_config_path,
            fold_index=fold_index,
            model_seed=int(kwargs["model_seed"]),
            train_rmse=0.80 + fold_index * 0.01,
            test_rmse=0.90 + fold_index * 0.01,
            training_seconds=10.0 * fold_index,
        )
        return {
            "run_manifest": str(run_manifest_path),
        }

    monkeypatch.setattr(
        "recsys_lab.experiments.ml100k_paper_benchmark._runner_for_model",
        lambda _model_name: _fake_runner,
    )

    payload = run_ml100k_paper_benchmark(
        model_name="biased_mf",
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        model_seed=1,
        repo_root=repo_root,
        command="recsys-lab benchmark-ml100k-paper --synthetic",
    )

    benchmark_dir = Path(payload["benchmark_dir"])
    benchmark_manifest = json.loads((benchmark_dir / "benchmark_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((benchmark_dir / "summary.json").read_text(encoding="utf-8"))

    assert benchmark_manifest["status"] == "completed"
    assert benchmark_manifest["benchmark_scope"] == "paper_faithful_ml100k_v1_biased_mf_u1_u5"
    assert benchmark_manifest["runtime"]["cpu_logical_count"] >= 1
    assert benchmark_manifest["runtime"]["threading"]["omp_num_threads"] == 1
    assert benchmark_manifest["runtime"]["threading"]["blas_threads"] == 1
    assert benchmark_manifest["runtime"]["threading"]["env_omp_num_threads"] == "1"
    assert benchmark_manifest["runtime"]["threading"]["env_mkl_num_threads"] == "1"
    assert benchmark_manifest["runtime"]["threading"]["env_openblas_num_threads"] == "1"
    assert benchmark_manifest["runtime"]["threading"]["env_numexpr_num_threads"] == "1"
    assert benchmark_manifest["measurement"]["sample_unit"] == "official_fold_run"
    assert benchmark_manifest["measurement"]["measured_sample_count"] == 5
    assert len(benchmark_manifest["inputs"]["run_ids"]) == 5
    assert calls == [2, 3, 4, 5]
    assert summary["measurement"]["warmup_policy"] == "none"
    assert summary["aggregate"]["test_rmse"]["mean"] > 0.0
    assert summary["aggregate"]["test_rmse"]["std"] >= 0.0
    assert summary["aggregate"]["test_rmse"]["count"] == 5
    assert summary["aggregate"]["test_rmse"]["median"] > 0.0
    assert summary["aggregate"]["training_wall_clock_seconds"]["coefficient_of_variation"] >= 0.0
    assert [fold["fold"] for fold in summary["folds"]] == ["u1", "u2", "u3", "u4", "u5"]


def test_run_ml100k_paper_benchmark_disables_reuse_when_repo_is_dirty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    (
        repo_root,
        processed_manifest_path,
        model_config_path,
        runtime_config_path,
        device_config_path,
    ) = _prepare_synthetic_benchmark_repo(tmp_path, actual_repo_root)

    _write_completed_run(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        fold_index=1,
        model_seed=1,
        train_rmse=0.810,
        test_rmse=0.910,
        training_seconds=10.0,
    )

    monkeypatch.setattr(
        "recsys_lab.experiments.ml100k_paper_benchmark.git_snapshot",
        lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": True},
    )

    calls: list[int] = []

    def _fake_runner(**kwargs):
        fold_index = int(kwargs["split_config"].seed)
        calls.append(fold_index)
        run_manifest_path = _write_completed_run(
            repo_root=repo_root,
            processed_manifest_path=processed_manifest_path,
            model_config_path=model_config_path,
            runtime_config_path=runtime_config_path,
            device_config_path=device_config_path,
            fold_index=fold_index,
            model_seed=int(kwargs["model_seed"]),
            train_rmse=0.80 + fold_index * 0.01,
            test_rmse=0.90 + fold_index * 0.01,
            training_seconds=10.0 * fold_index,
        )
        return {
            "run_manifest": str(run_manifest_path),
        }

    monkeypatch.setattr(
        "recsys_lab.experiments.ml100k_paper_benchmark._runner_for_model",
        lambda _model_name: _fake_runner,
    )

    payload = run_ml100k_paper_benchmark(
        model_name="biased_mf",
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        model_seed=1,
        repo_root=repo_root,
        command="recsys-lab benchmark-ml100k-paper --synthetic-dirty",
    )

    benchmark_dir = Path(payload["benchmark_dir"])
    stdout_log = (benchmark_dir / "stdout.log").read_text(encoding="utf-8")

    assert calls == [1, 2, 3, 4, 5]
    assert "allow_existing_run_reuse=false" in stdout_log


def test_run_ml100k_paper_benchmark_supports_cb_svdpp_and_counts_cluster_time(
    tmp_path: Path,
    monkeypatch,
) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    (
        repo_root,
        processed_manifest_path,
        model_config_path,
        runtime_config_path,
        device_config_path,
    ) = _prepare_synthetic_benchmark_repo(tmp_path, actual_repo_root)

    _write_text(
        model_config_path,
        "model:\n  name: cb_svdpp\n  scope: paper_inspired\ntraining:\n  dtype: float32\n",
    )

    monkeypatch.setattr(
        "recsys_lab.experiments.ml100k_paper_benchmark.git_snapshot",
        lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": False},
    )

    calls: list[int] = []

    def _fake_runner(**kwargs):
        fold_index = int(kwargs["split_config"].seed)
        calls.append(fold_index)
        run_manifest_path = _write_completed_run(
            repo_root=repo_root,
            processed_manifest_path=processed_manifest_path,
            model_config_path=model_config_path,
            runtime_config_path=runtime_config_path,
            device_config_path=device_config_path,
            fold_index=fold_index,
            model_seed=int(kwargs["model_seed"]),
            train_rmse=0.78 + fold_index * 0.01,
            test_rmse=0.88 + fold_index * 0.01,
            training_seconds=20.0 * fold_index,
            model_name="cb_svdpp",
            cluster_induction_seconds=5.0 * fold_index,
        )
        return {
            "run_manifest": str(run_manifest_path),
        }

    monkeypatch.setattr(
        "recsys_lab.experiments.ml100k_paper_benchmark._runner_for_model",
        lambda _model_name: _fake_runner,
    )

    payload = run_ml100k_paper_benchmark(
        model_name="cb_svdpp",
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        model_seed=2,
        repo_root=repo_root,
        command="recsys-lab benchmark-ml100k-paper --synthetic-cb-svdpp",
    )

    benchmark_dir = Path(payload["benchmark_dir"])
    summary = json.loads((benchmark_dir / "summary.json").read_text(encoding="utf-8"))

    assert calls == [1, 2, 3, 4, 5]
    assert summary["model"] == "cb_svdpp"
    assert summary["measurement"]["time_metric"] == "training_wall_clock_seconds"
    assert summary["aggregate"]["training_wall_clock_seconds"]["mean"] == 75.0
    assert summary["aggregate"]["training_wall_clock_seconds"]["count"] == 5
    assert summary["folds"][0]["training_wall_clock_seconds"] == 25.0
