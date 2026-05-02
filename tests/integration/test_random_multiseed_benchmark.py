import json
from pathlib import Path

import pytest

from recsys_lab.config.loader import dump_yaml_file, load_yaml_file
from recsys_lab.experiments.random_multiseed_benchmark import run_random_multiseed_benchmark


def _write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding, newline="\n")


def _prepare_repo(
    tmp_path: Path,
    actual_repo_root: Path,
    *,
    model_name: str,
) -> tuple[Path, Path, Path, Path, Path]:
    repo_root = tmp_path / "repo"
    _write_text(repo_root / "AGENTS.md", "# Test Repo\n")
    _write_text(repo_root / "pyproject.toml", "[project]\nname = 'test-repo'\nversion = '0.0.0'\n")

    schema_dir = repo_root / "schema" / "reporting"
    schema_dir.mkdir(parents=True, exist_ok=True)
    source_schema = actual_repo_root / "schema" / "reporting" / "benchmark_manifest.schema.json"
    _write_text(schema_dir / "benchmark_manifest.schema.json", source_schema.read_text(encoding="utf-8"))

    (repo_root / "artifacts" / "runs").mkdir(parents=True, exist_ok=True)
    (repo_root / "artifacts" / "benchmarks").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "models").mkdir(parents=True, exist_ok=True)
    (repo_root / "configs" / "runtime" / "devices").mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "processed" / "ml1m").mkdir(parents=True, exist_ok=True)

    processed_manifest_path = (
        repo_root / "data" / "processed" / "ml1m" / "ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json"
    )
    model_config_path = repo_root / "configs" / "models" / f"{model_name}.yaml"
    runtime_config_path = repo_root / "configs" / "runtime" / "base.yaml"
    device_config_path = repo_root / "configs" / "runtime" / "devices" / "local.yaml"

    _write_text(
        processed_manifest_path,
        json.dumps(
            {
                "manifest_version": "v1",
                "kind": "processed_dataset_manifest",
                "dataset_name": "MovieLens 1M",
                "dataset_short_name": "ml1m",
                "split_family": "benchmark_random_v1",
                "preprocessing_family": "explicit_v1",
                "dtype": "float32",
                "source": {
                    "format_family": "legacy_1m",
                    "raw_dir": "C:/seed0/raw",
                    "raw_manifest_path": "C:/seed0/raw_manifest.json",
                },
                "validation": {
                    "format_family": "legacy_1m",
                    "counts": {
                        "ratings": 1000209,
                        "movies": 3883,
                        "users": 6040,
                        "links": 0,
                        "tags": 0,
                    },
                },
                "counts": {
                    "interactions": 1000209,
                    "users": 6040,
                    "rated_items": 3706,
                    "catalog_items": 3883,
                    "tags": 0,
                    "links": 0,
                },
                "rating_range": {"min": 1.0, "max": 5.0},
                "artifacts": {"interactions": "C:/seed0/interactions.parquet"},
            },
            indent=2,
        ),
    )
    if model_name == "cb_svdpp":
        _write_text(
            model_config_path,
            "metadata:\n  status: draft\n"
            "model:\n  name: cb_svdpp\n  scope: paper_inspired\n"
            "training:\n  dtype: float32\n"
            "clustering:\n  n_user_clusters: 80\n  n_item_clusters: 80\n  alpha: 0.10\n",
        )
    else:
        _write_text(
            model_config_path,
            "metadata:\n  status: draft\n"
            f"model:\n  name: {model_name}\n  scope: paper_inspired\n"
            "training:\n  dtype: float32\n",
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
    return (
        repo_root,
        processed_manifest_path,
        model_config_path,
        runtime_config_path,
        device_config_path,
    )


def _write_random_run(
    *,
    repo_root: Path,
    processed_manifest_path: Path,
    model_config_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    split_seed: int,
    model_seed: int,
    validation_rmse: float,
    test_rmse: float,
    training_seconds: float,
    peak_memory_mb: float,
    benchmark_tag: str = "a",
    git_commit: str = "abcdef1234567",
    git_dirty: bool = False,
    processed_path_tag: str | None = None,
    cluster_seconds: float | None = None,
) -> Path:
    processed_manifest_payload = json.loads(processed_manifest_path.read_text(encoding="utf-8"))
    model_config_payload = load_yaml_file(model_config_path)
    runtime_config_payload = load_yaml_file(runtime_config_path)
    device_config_payload = load_yaml_file(device_config_path)

    split_id_value = f"benchmark_random_v1_tr080_va010_s{split_seed:03d}"
    run_id = (
        f"2026-04-16T12{split_seed:02d}00Z_ml1m_"
        f"{model_config_path.stem}_local_test_{split_id_value}_s{model_seed:03d}_{benchmark_tag}"
    )
    run_dir = repo_root / "artifacts" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics.json"
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    stdout_log_path = run_dir / "stdout.log"

    loaded_processed_manifest = dict(processed_manifest_payload)
    fake_root = f"C:/synthetic/{processed_path_tag or f'seed{split_seed}'}"
    loaded_processed_manifest["source"] = {
        **dict(loaded_processed_manifest.get("source", {})),
        "raw_dir": f"{fake_root}/raw",
        "raw_manifest_path": f"{fake_root}/download_manifest.json",
    }
    loaded_processed_manifest["validation"] = {
        **dict(loaded_processed_manifest.get("validation", {})),
        "raw_dir": f"{fake_root}/raw",
    }
    loaded_processed_manifest["artifacts"] = {"interactions": f"{fake_root}/interactions.parquet"}

    metrics_payload = {
        "run_id": run_id,
        "metrics": {
            "train_rmse": validation_rmse - 0.1,
            "validation_rmse": validation_rmse,
            "test_rmse": test_rmse,
        },
        "split": {
            "family": "benchmark_random_v1",
            "seed": split_seed,
            "train_ratio": 0.8,
            "validation_ratio": 0.1,
            "train_rows": 800193,
            "validation_rows": 100020,
            "test_rows": 99996,
            "test_metrics_available": True,
        },
        "system_metrics": {"peak_memory_mb": peak_memory_mb},
        "timing": {"training_wall_clock_seconds": training_seconds},
    }
    if cluster_seconds is not None:
        metrics_payload["timing"]["cluster_induction_wall_clock_seconds"] = cluster_seconds

    _write_text(stdout_log_path, "completed\n")
    _write_text(metrics_path, json.dumps(metrics_payload, indent=2))
    _write_text(
        run_manifest_path,
        json.dumps(
            {
                "manifest_version": "v1",
                "kind": "run_manifest",
                "generated_at_utc": f"2026-04-16T12{split_seed:02d}30Z",
                "run_id": run_id,
                "status": "completed",
                "command": "recsys-lab train-model --synthetic",
                "cwd": ".",
                "git": {
                    "commit": git_commit,
                    "branch": "main",
                    "dirty": git_dirty,
                },
                "dataset": {
                    "short_name": "ml1m",
                    "source": "MovieLens 1M",
                    "version": "explicit_v1",
                    "split_family": "benchmark_random_v1",
                    "split_id": split_id_value,
                    "manifest_ref": str(processed_manifest_path.relative_to(repo_root)).replace("\\", "/"),
                },
                "model": {
                    "name": model_config_path.stem,
                    "scope": "paper_inspired",
                    "config_ref": str(model_config_path.relative_to(repo_root)).replace("\\", "/"),
                },
                "runtime": {
                    "device_profile": "local_test",
                    "python_version": "3.11.9",
                    "dtype": "float32",
                    "threading": {"omp_num_threads": 1, "blas_threads": 1},
                },
                "seeds": [model_seed],
                "artifacts": {
                    "config_snapshot": str(config_snapshot_path.relative_to(repo_root)).replace("\\", "/"),
                    "metrics": str(metrics_path.relative_to(repo_root)).replace("\\", "/"),
                    "stdout_log": str(stdout_log_path.relative_to(repo_root)).replace("\\", "/"),
                },
            },
            indent=2,
        ),
    )
    dump_yaml_file(
        config_snapshot_path,
        {
            "run_id": run_id,
            "command": "recsys-lab train-model --synthetic",
            "inputs": {
                "processed_manifest": str(processed_manifest_path.relative_to(repo_root)).replace("\\", "/"),
                "model_config": str(model_config_path.relative_to(repo_root)).replace("\\", "/"),
                "runtime_config": str(runtime_config_path.relative_to(repo_root)).replace("\\", "/"),
                "device_config": str(device_config_path.relative_to(repo_root)).replace("\\", "/"),
            },
            "split": {
                "train_ratio": 0.8,
                "validation_ratio": 0.1,
                "seed": split_seed,
            },
            "model_seed": model_seed,
            "loaded_configs": {
                "processed_manifest": loaded_processed_manifest,
                "model": model_config_payload,
                "runtime": runtime_config_payload,
                "device": device_config_payload,
            },
        },
    )
    return run_manifest_path


def test_run_random_multiseed_benchmark_aggregates_split_seed_runs(
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
    ) = _prepare_repo(tmp_path, actual_repo_root, model_name="cb_svdpp")
    _write_random_run(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_seed=1,
        model_seed=1,
        validation_rmse=0.858,
        test_rmse=0.860,
        training_seconds=300.0,
        cluster_seconds=40.0,
        peak_memory_mb=1400.0,
        processed_path_tag="clone_a_seed1",
    )
    _write_random_run(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_seed=2,
        model_seed=1,
        validation_rmse=0.857,
        test_rmse=0.859,
        training_seconds=310.0,
        cluster_seconds=50.0,
        peak_memory_mb=1410.0,
        processed_path_tag="clone_b_seed2",
    )
    _write_random_run(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_seed=3,
        model_seed=1,
        validation_rmse=0.856,
        test_rmse=0.858,
        training_seconds=320.0,
        cluster_seconds=60.0,
        peak_memory_mb=1420.0,
        processed_path_tag="clone_c_seed3",
    )

    monkeypatch.setattr(
        "recsys_lab.experiments.random_multiseed_benchmark.git_snapshot",
        lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": False},
    )

    payload = run_random_multiseed_benchmark(
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_seeds=[1, 2, 3],
        model_seed=1,
        repo_root=repo_root,
        command="recsys-lab benchmark-random-multiseed --synthetic",
    )

    summary = json.loads((Path(payload["benchmark_dir"]) / "summary.json").read_text(encoding="utf-8"))
    benchmark_manifest = json.loads(
        (Path(payload["benchmark_dir"]) / "benchmark_manifest.json").read_text(encoding="utf-8")
    )

    assert summary["measurement"]["sample_unit"] == "split_seed_run"
    assert summary["measurement"]["measured_sample_count"] == 3
    assert summary["aggregate"]["validation_rmse"]["mean"] == pytest.approx(0.857)
    assert summary["aggregate"]["test_rmse"]["mean"] == pytest.approx(0.859)
    assert summary["aggregate"]["training_wall_clock_seconds"]["mean"] == pytest.approx(360.0)
    assert summary["aggregate"]["peak_memory_mb"]["mean"] == pytest.approx(1410.0)
    assert len(summary["per_run"]) == 3
    assert benchmark_manifest["inputs"]["split_seeds"] == [1, 2, 3]
    assert benchmark_manifest["inputs"]["model_seeds"] == [1]


def test_run_random_multiseed_benchmark_accepts_explicit_run_manifest_paths(
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
    ) = _prepare_repo(tmp_path, actual_repo_root, model_name="biased_mf")
    _write_random_run(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_seed=1,
        model_seed=1,
        validation_rmse=0.867,
        test_rmse=0.869,
        training_seconds=55.0,
        peak_memory_mb=845.0,
        benchmark_tag="a",
    )
    _write_random_run(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_seed=1,
        model_seed=1,
        validation_rmse=0.868,
        test_rmse=0.870,
        training_seconds=56.0,
        peak_memory_mb=846.0,
        benchmark_tag="b",
    )
    run_manifest_path_seed_2 = _write_random_run(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_seed=2,
        model_seed=1,
        validation_rmse=0.866,
        test_rmse=0.868,
        training_seconds=57.0,
        peak_memory_mb=847.0,
        benchmark_tag="a",
    )
    run_manifest_path_seed_3 = _write_random_run(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_seed=3,
        model_seed=1,
        validation_rmse=0.865,
        test_rmse=0.867,
        training_seconds=58.0,
        peak_memory_mb=848.0,
        benchmark_tag="a",
    )

    monkeypatch.setattr(
        "recsys_lab.experiments.random_multiseed_benchmark.git_snapshot",
        lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": False},
    )

    with pytest.raises(ValueError, match="multiple matching run manifests"):
        run_random_multiseed_benchmark(
            processed_manifest_path=processed_manifest_path,
            model_config_path=model_config_path,
            runtime_config_path=runtime_config_path,
            device_config_path=device_config_path,
            split_seeds=[1, 2, 3],
            model_seed=1,
            repo_root=repo_root,
            command="recsys-lab benchmark-random-multiseed --synthetic",
        )

    explicit_seed_1 = (
        repo_root
        / "artifacts"
        / "runs"
        / "2026-04-16T120100Z_ml1m_biased_mf_local_test_benchmark_random_v1_tr080_va010_s001_s001_a"
        / "run_manifest.json"
    )
    payload = run_random_multiseed_benchmark(
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_seeds=[1, 2, 3],
        model_seed=1,
        run_manifest_paths=[explicit_seed_1, run_manifest_path_seed_2, run_manifest_path_seed_3],
        repo_root=repo_root,
        command="recsys-lab benchmark-random-multiseed --synthetic --run-manifest-paths explicit",
    )

    summary = json.loads((Path(payload["benchmark_dir"]) / "summary.json").read_text(encoding="utf-8"))
    assert summary["aggregate"]["test_rmse"]["mean"] == 0.868


def test_run_random_multiseed_benchmark_rejects_mixed_git_state(
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
    ) = _prepare_repo(tmp_path, actual_repo_root, model_name="cb_svdpp")
    run_manifest_path_seed_1 = _write_random_run(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_seed=1,
        model_seed=1,
        validation_rmse=0.858,
        test_rmse=0.860,
        training_seconds=300.0,
        cluster_seconds=40.0,
        peak_memory_mb=1400.0,
        git_commit="abcdef1234567",
        git_dirty=False,
    )
    run_manifest_path_seed_2 = _write_random_run(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_seed=2,
        model_seed=1,
        validation_rmse=0.857,
        test_rmse=0.859,
        training_seconds=310.0,
        cluster_seconds=50.0,
        peak_memory_mb=1410.0,
        git_commit="abcdef1234567",
        git_dirty=False,
    )
    run_manifest_path_seed_3 = _write_random_run(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_seed=3,
        model_seed=1,
        validation_rmse=0.856,
        test_rmse=0.858,
        training_seconds=320.0,
        cluster_seconds=60.0,
        peak_memory_mb=1420.0,
        git_commit="7654321fedcba",
        git_dirty=True,
    )

    monkeypatch.setattr(
        "recsys_lab.experiments.random_multiseed_benchmark.git_snapshot",
        lambda _repo_root: {
            "commit": "b91d606ae78b9170569ccf76683a92ad5d1d2900",
            "branch": "main",
            "dirty": False,
        },
    )

    with pytest.raises(ValueError, match="share identical git commit, branch, and dirty state"):
        run_random_multiseed_benchmark(
            processed_manifest_path=processed_manifest_path,
            model_config_path=model_config_path,
            runtime_config_path=runtime_config_path,
            device_config_path=device_config_path,
            split_seeds=[1, 2, 3],
            model_seed=1,
            run_manifest_paths=[
                run_manifest_path_seed_1,
                run_manifest_path_seed_2,
                run_manifest_path_seed_3,
            ],
            repo_root=repo_root,
            command="recsys-lab benchmark-random-multiseed --synthetic --run-manifest-paths explicit",
        )


def test_run_random_multiseed_benchmark_rejects_current_repo_git_mismatch(
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
    ) = _prepare_repo(tmp_path, actual_repo_root, model_name="biased_mf")
    run_manifest_path_seed_1 = _write_random_run(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_seed=1,
        model_seed=1,
        validation_rmse=0.867,
        test_rmse=0.869,
        training_seconds=55.0,
        peak_memory_mb=845.0,
    )
    run_manifest_path_seed_2 = _write_random_run(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_seed=2,
        model_seed=1,
        validation_rmse=0.866,
        test_rmse=0.868,
        training_seconds=57.0,
        peak_memory_mb=847.0,
    )
    run_manifest_path_seed_3 = _write_random_run(
        repo_root=repo_root,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_seed=3,
        model_seed=1,
        validation_rmse=0.865,
        test_rmse=0.867,
        training_seconds=58.0,
        peak_memory_mb=848.0,
    )

    monkeypatch.setattr(
        "recsys_lab.experiments.random_multiseed_benchmark.git_snapshot",
        lambda _repo_root: {"commit": "0000000", "branch": "main", "dirty": True},
    )

    with pytest.raises(ValueError, match="must match current repo git commit and dirty state"):
        run_random_multiseed_benchmark(
            processed_manifest_path=processed_manifest_path,
            model_config_path=model_config_path,
            runtime_config_path=runtime_config_path,
            device_config_path=device_config_path,
            split_seeds=[1, 2, 3],
            model_seed=1,
            run_manifest_paths=[
                run_manifest_path_seed_1,
                run_manifest_path_seed_2,
                run_manifest_path_seed_3,
            ],
            repo_root=repo_root,
            command="recsys-lab benchmark-random-multiseed --synthetic --run-manifest-paths explicit",
        )
