import json
from pathlib import Path

import pytest

from recsys_lab.experiments.ml100k_inner_tuning import (
    _candidate_config_filename,
    run_inner_tuning,
    run_ml100k_inner_tuning,
)


def _write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding, newline="\n")


def _prepare_synthetic_tuning_repo(
    tmp_path: Path,
    actual_repo_root: Path,
) -> tuple[Path, Path, Path, Path, Path]:
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
    (repo_root / "configs" / "experiments" / "tuning").mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "processed" / "ml100k").mkdir(parents=True, exist_ok=True)

    model_config_path = repo_root / "configs" / "models" / "biased_mf.yaml"
    runtime_config_path = repo_root / "configs" / "runtime" / "base.yaml"
    device_config_path = repo_root / "configs" / "runtime" / "devices" / "local.yaml"
    tuning_config_path = repo_root / "configs" / "experiments" / "tuning" / "ml100k_biased_mf_stage1.yaml"
    processed_manifest_path = (
        repo_root / "data" / "processed" / "ml100k" / "ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json"
    )

    _write_text(
        model_config_path,
        "metadata:\n  status: draft\nmodel:\n  name: biased_mf\n  scope: paper_inspired\ntraining:\n"
        "  latent_dim: 50\n  epochs: 20\n  learning_rate: 0.01\n  lambda_b: 0.02\n"
        "  lambda_p: 0.02\n  lambda_q: 0.02\n  dtype: float32\n",
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
        tuning_config_path,
        "tuning:\n  name: ml100k_biased_mf_stage1\n  dataset_short_name: ml100k\n"
        "  split_family: paper_faithful_ml100k_inner_v1\n  selection_stage: stage1\n"
        "  objective: validation_rmse_mean\n  folds:\n    - 1\n    - 2\n  validation_ratio: 0.1\n"
        "  inner_seed: 17\n  model_seed: 1\nbase_model_config: configs/models/biased_mf.yaml\n"
        "candidates:\n  - candidate_id: baseline\n    overrides:\n      training:\n        latent_dim: 50\n"
        "        epochs: 20\n        learning_rate: 0.01\n  - candidate_id: alt\n    overrides:\n"
        "      training:\n        latent_dim: 64\n        epochs: 25\n        learning_rate: 0.005\n",
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
        tuning_config_path,
        processed_manifest_path,
        runtime_config_path,
        device_config_path,
    )


def test_candidate_config_filename_shortens_long_ids() -> None:
    long_candidate_id = "rank032_uc064_ic064_a010_lr0100_reg0020_e002_extra_long_windows_path_guard"

    filename = _candidate_config_filename(long_candidate_id, candidate_index=7)

    assert filename.startswith("c007_")
    assert filename.endswith(".yaml")
    assert len(filename) < len(f"{long_candidate_id}.yaml")


def test_run_ml100k_inner_tuning_aggregates_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    (
        repo_root,
        tuning_config_path,
        processed_manifest_path,
        runtime_config_path,
        device_config_path,
    ) = _prepare_synthetic_tuning_repo(tmp_path, actual_repo_root)

    monkeypatch.setattr(
        "recsys_lab.experiments.ml100k_inner_tuning.git_snapshot",
        lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": False},
    )

    call_index = {"value": 0}
    observed_train_ratios: list[float] = []

    def _fake_runner(**kwargs):
        call_index["value"] += 1
        candidate_name = Path(kwargs["model_config_path"]).stem
        fold_index = int(kwargs["split_config"].seed)
        observed_train_ratios.append(float(kwargs["split_config"].train_ratio))
        run_id = f"2026-04-13T10000{call_index['value']}Z_ml100k_biased_mf_local_test_s001"
        run_dir = repo_root / "artifacts" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.json"
        config_snapshot_path = run_dir / "config_snapshot.yaml"
        stdout_log_path = run_dir / "stdout.log"
        run_manifest_path = run_dir / "run_manifest.json"

        validation_rmse = 0.91 if candidate_name == "baseline" else 0.89
        validation_rmse += fold_index * 0.001
        train_rmse = validation_rmse - 0.1
        training_seconds = 10.0 + fold_index

        _write_text(config_snapshot_path, "candidate: test\n")
        _write_text(stdout_log_path, f"run_id={run_id}\n")
        _write_text(
            metrics_path,
            json.dumps(
                {
                    "run_id": run_id,
                    "metrics": {
                        "train_rmse": train_rmse,
                        "validation_rmse": validation_rmse,
                        "test_rmse": None,
                    },
                    "timing": {
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
                    "generated_at_utc": "2026-04-13T100000Z",
                    "run_id": run_id,
                    "status": "completed",
                    "command": "recsys-lab tune-ml100k-inner --synthetic",
                    "cwd": ".",
                    "git": {
                        "commit": "abcdef1234567",
                        "branch": "main",
                        "dirty": False,
                    },
                    "dataset": {
                        "short_name": "ml100k",
                        "split_family": "paper_faithful_ml100k_inner_v1",
                    },
                    "model": {
                        "name": "biased_mf",
                        "scope": "paper_inspired",
                        "config_ref": str(Path(kwargs["model_config_path"]).relative_to(repo_root)).replace("\\", "/"),
                    },
                    "runtime": {
                        "device_profile": "local_test",
                        "python_version": "3.11.9",
                        "dtype": "float32",
                        "threading": {"omp_num_threads": 1, "blas_threads": 1},
                    },
                    "seeds": [1],
                    "artifacts": {
                        "config_snapshot": str(config_snapshot_path.relative_to(repo_root)).replace("\\", "/"),
                        "metrics": str(metrics_path.relative_to(repo_root)).replace("\\", "/"),
                        "stdout_log": str(stdout_log_path.relative_to(repo_root)).replace("\\", "/"),
                    },
                },
                indent=2,
            ),
        )
        return {
            "run_manifest": str(run_manifest_path),
        }

    monkeypatch.setattr(
        "recsys_lab.experiments.ml100k_inner_tuning._runner_for_model",
        lambda _model_name: _fake_runner,
    )

    payload = run_ml100k_inner_tuning(
        tuning_config_path=tuning_config_path,
        processed_manifest_path=processed_manifest_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        repo_root=repo_root,
        command="recsys-lab tune-ml100k-inner --synthetic",
    )

    summary = json.loads((Path(payload["benchmark_dir"]) / "summary.json").read_text(encoding="utf-8"))

    assert payload["best_candidate"] == "alt"
    assert summary["measurement"]["sample_unit"] == "inner_tuning_run"
    assert summary["measurement"]["measured_sample_count"] == 2
    assert summary["train_ratio"] == 0.9
    assert summary["best_candidate"]["candidate_id"] == "alt"
    assert summary["best_candidate"]["aggregate"]["validation_rmse"]["count"] == 2
    assert summary["best_candidate"]["aggregate"]["training_wall_clock_seconds"]["coefficient_of_variation"] >= 0.0
    assert len(summary["candidates"]) == 2
    assert [candidate["candidate_id"] for candidate in summary["candidates"]] == ["alt", "baseline"]
    assert observed_train_ratios == [0.9, 0.9, 0.9, 0.9]


def test_run_ml100k_inner_tuning_passes_split_cache_override_to_supported_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    (
        repo_root,
        tuning_config_path,
        processed_manifest_path,
        runtime_config_path,
        device_config_path,
    ) = _prepare_synthetic_tuning_repo(tmp_path, actual_repo_root)

    model_config_path = repo_root / "configs" / "models" / "svdpp.yaml"
    _write_text(
        model_config_path,
        "metadata:\n  status: draft\nmodel:\n  name: svdpp\n  scope: paper_inspired\ntraining:\n"
        "  latent_dim: 50\n  epochs: 20\n  learning_rate: 0.01\n  lambda_b: 0.02\n"
        "  lambda_p: 0.02\n  lambda_q: 0.02\n  lambda_y: 0.02\n  dtype: float32\n",
    )
    _write_text(
        tuning_config_path,
        "tuning:\n  name: ml100k_svdpp_stage1\n  dataset_short_name: ml100k\n"
        "  split_family: paper_faithful_ml100k_inner_v1\n  selection_stage: stage1\n"
        "  objective: validation_rmse_mean\n  folds:\n    - 1\n    - 2\n  validation_ratio: 0.1\n"
        "  inner_seed: 17\n  model_seed: 1\nbase_model_config: configs/models/svdpp.yaml\n"
        "candidates:\n  - candidate_id: baseline\n    overrides:\n      training:\n        learning_rate: 0.01\n",
    )

    monkeypatch.setattr(
        "recsys_lab.experiments.ml100k_inner_tuning.git_snapshot",
        lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": False},
    )

    observed_split_cache: list[bool | None] = []
    observed_training_index_cache: list[bool | None] = []

    def _fake_runner(**kwargs):
        fold_index = int(kwargs["split_config"].seed)
        observed_split_cache.append(kwargs.get("use_split_cache"))
        observed_training_index_cache.append(kwargs.get("use_training_index_cache"))
        run_id = f"2026-04-13T30000{fold_index}Z_ml100k_svdpp_local_test_s001"
        run_dir = repo_root / "artifacts" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.json"
        config_snapshot_path = run_dir / "config_snapshot.yaml"
        stdout_log_path = run_dir / "stdout.log"
        run_manifest_path = run_dir / "run_manifest.json"

        _write_text(config_snapshot_path, "candidate: test\n")
        _write_text(stdout_log_path, f"run_id={run_id}\n")
        _write_text(
            metrics_path,
            json.dumps(
                {
                    "run_id": run_id,
                    "metrics": {
                        "train_rmse": 0.80,
                        "validation_rmse": 0.90,
                        "test_rmse": None,
                    },
                    "timing": {
                        "training_wall_clock_seconds": 10.0 + fold_index,
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
                    "generated_at_utc": "2026-04-13T300000Z",
                    "run_id": run_id,
                    "status": "completed",
                    "command": "recsys-lab tune-ml100k-inner --synthetic-override",
                    "cwd": ".",
                    "git": {
                        "commit": "abcdef1234567",
                        "branch": "main",
                        "dirty": False,
                    },
                    "dataset": {
                        "short_name": "ml100k",
                        "split_family": "paper_faithful_ml100k_inner_v1",
                    },
                    "model": {
                        "name": "svdpp",
                        "scope": "paper_inspired",
                        "config_ref": str(Path(kwargs["model_config_path"]).relative_to(repo_root)).replace("\\", "/"),
                    },
                    "runtime": {
                        "device_profile": "local_test",
                        "python_version": "3.11.9",
                        "dtype": "float32",
                        "threading": {"omp_num_threads": 1, "blas_threads": 1},
                    },
                    "seeds": [1],
                    "artifacts": {
                        "config_snapshot": str(config_snapshot_path.relative_to(repo_root)).replace("\\", "/"),
                        "metrics": str(metrics_path.relative_to(repo_root)).replace("\\", "/"),
                        "stdout_log": str(stdout_log_path.relative_to(repo_root)).replace("\\", "/"),
                    },
                },
                indent=2,
            ),
        )
        return {
            "run_manifest": str(run_manifest_path),
        }

    monkeypatch.setattr(
        "recsys_lab.experiments.ml100k_inner_tuning._runner_for_model",
        lambda _model_name: _fake_runner,
    )

    run_ml100k_inner_tuning(
        tuning_config_path=tuning_config_path,
        processed_manifest_path=processed_manifest_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        use_split_cache=False,
        use_training_index_cache=True,
        repo_root=repo_root,
        command="recsys-lab tune-ml100k-inner --synthetic-override",
    )

    assert observed_split_cache == [False, False]
    assert observed_training_index_cache == [True, True]


def test_run_ml100k_inner_tuning_rejects_explicit_split_cache_override_for_unsupported_models(
    tmp_path: Path,
) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    (
        repo_root,
        tuning_config_path,
        processed_manifest_path,
        runtime_config_path,
        device_config_path,
    ) = _prepare_synthetic_tuning_repo(tmp_path, actual_repo_root)

    with pytest.raises(ValueError, match="explicit split-cache override"):
        run_ml100k_inner_tuning(
            tuning_config_path=tuning_config_path,
            processed_manifest_path=processed_manifest_path,
            runtime_config_path=runtime_config_path,
            device_config_path=device_config_path,
            use_split_cache=True,
            repo_root=repo_root,
        )


def test_run_ml100k_inner_tuning_rejects_unsupported_cache_controls(
    tmp_path: Path,
) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    (
        repo_root,
        tuning_config_path,
        processed_manifest_path,
        runtime_config_path,
        device_config_path,
    ) = _prepare_synthetic_tuning_repo(tmp_path, actual_repo_root)

    with pytest.raises(ValueError, match="training-index cache"):
        run_ml100k_inner_tuning(
            tuning_config_path=tuning_config_path,
            processed_manifest_path=processed_manifest_path,
            runtime_config_path=runtime_config_path,
            device_config_path=device_config_path,
            use_training_index_cache=True,
            repo_root=repo_root,
        )

    with pytest.raises(ValueError, match="cluster-artifact cache"):
        run_ml100k_inner_tuning(
            tuning_config_path=tuning_config_path,
            processed_manifest_path=processed_manifest_path,
            runtime_config_path=runtime_config_path,
            device_config_path=device_config_path,
            use_cluster_artifact_cache=True,
            repo_root=repo_root,
        )


def test_run_ml100k_inner_tuning_supports_cb_svdpp_and_counts_cluster_time(
    tmp_path: Path,
    monkeypatch,
) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    (
        repo_root,
        tuning_config_path,
        processed_manifest_path,
        runtime_config_path,
        device_config_path,
    ) = _prepare_synthetic_tuning_repo(tmp_path, actual_repo_root)

    model_config_path = repo_root / "configs" / "models" / "cb_svdpp.yaml"
    _write_text(
        model_config_path,
        "metadata:\n"
        "  status: draft\n"
        "model:\n"
        "  name: cb_svdpp\n"
        "  scope: paper_inspired\n"
        "training:\n"
        "  latent_dim: 64\n"
        "  epochs: 2\n"
        "  learning_rate: 0.0075\n"
        "  lambda_b: 0.025\n"
        "  lambda_p: 0.025\n"
        "  lambda_q: 0.025\n"
        "  lambda_y: 0.025\n"
        "  lambda_pC: 0.025\n"
        "  lambda_qC: 0.025\n"
        "  lambda_yC: 0.025\n"
        "  dtype: float32\n"
        "clustering:\n"
        "  n_user_clusters: 80\n"
        "  n_item_clusters: 80\n"
        "  alpha: 0.10\n",
    )
    _write_text(
        tuning_config_path,
        "tuning:\n"
        "  name: ml1m_cb_svdpp_stage0\n"
        "  dataset_short_name: ml1m\n"
        "  split_family: benchmark_random_v1\n"
        "  selection_stage: stage0\n"
        "  objective: validation_rmse_mean\n"
        "  split_seeds:\n"
        "    - 1\n"
        "    - 2\n"
        "  validation_ratio: 0.1\n"
        "  model_seed: 1\n"
        "base_model_config: configs/models/cb_svdpp.yaml\n"
        "candidates:\n"
        "  - candidate_id: base\n"
        "    overrides:\n"
        "      clustering:\n"
        "        alpha: 0.10\n"
        "  - candidate_id: alpha15\n"
        "    overrides:\n"
        "      clustering:\n"
        "        alpha: 0.15\n",
    )
    _write_text(
        processed_manifest_path,
        json.dumps(
            {
                "dataset_short_name": "ml1m",
                "dataset_name": "MovieLens 1M",
                "preprocessing_family": "explicit_v1",
                "split_family": "benchmark_random_v1",
                "artifacts": {
                    "interactions": "data/processed/ml1m/interactions.parquet",
                },
            },
            indent=2,
        ),
    )

    monkeypatch.setattr(
        "recsys_lab.experiments.ml100k_inner_tuning.git_snapshot",
        lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": False},
    )

    call_index = {"value": 0}
    observed_train_ratios: list[float] = []
    observed_split_cache: list[bool | None] = []
    observed_training_index_cache: list[bool | None] = []
    observed_cluster_artifact_cache: list[bool | None] = []

    def _fake_runner(**kwargs):
        call_index["value"] += 1
        candidate_name = Path(kwargs["model_config_path"]).stem
        split_seed = int(kwargs["split_config"].seed)
        observed_train_ratios.append(float(kwargs["split_config"].train_ratio))
        observed_split_cache.append(kwargs.get("use_split_cache"))
        observed_training_index_cache.append(kwargs.get("use_training_index_cache"))
        observed_cluster_artifact_cache.append(kwargs.get("use_cluster_artifact_cache"))
        run_id = f"2026-04-16T10000{call_index['value']}Z_ml1m_cb_svdpp_local_test_s001"
        run_dir = repo_root / "artifacts" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.json"
        config_snapshot_path = run_dir / "config_snapshot.yaml"
        stdout_log_path = run_dir / "stdout.log"
        run_manifest_path = run_dir / "run_manifest.json"

        validation_rmse = 0.905 if candidate_name == "base" else 0.895
        validation_rmse += split_seed * 0.001
        train_rmse = validation_rmse - 0.08
        cluster_seconds = 5.0 * split_seed
        training_seconds = 12.0 + split_seed

        _write_text(config_snapshot_path, "candidate: test\n")
        _write_text(stdout_log_path, f"run_id={run_id}\n")
        _write_text(
            metrics_path,
            json.dumps(
                {
                    "run_id": run_id,
                    "metrics": {
                        "train_rmse": train_rmse,
                        "validation_rmse": validation_rmse,
                        "test_rmse": None,
                    },
                    "timing": {
                        "cluster_induction_wall_clock_seconds": cluster_seconds,
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
                    "generated_at_utc": "2026-04-16T100000Z",
                    "run_id": run_id,
                    "status": "completed",
                    "command": "recsys-lab tune-inner --synthetic-random",
                    "cwd": ".",
                    "git": {
                        "commit": "abcdef1234567",
                        "branch": "main",
                        "dirty": False,
                    },
                    "dataset": {
                        "short_name": "ml1m",
                        "split_family": "benchmark_random_v1",
                    },
                    "model": {
                        "name": "cb_svdpp",
                        "scope": "paper_inspired",
                        "config_ref": str(Path(kwargs["model_config_path"]).relative_to(repo_root)).replace("\\", "/"),
                    },
                    "runtime": {
                        "device_profile": "local_test",
                        "python_version": "3.11.9",
                        "dtype": "float32",
                        "threading": {"omp_num_threads": 1, "blas_threads": 1},
                    },
                    "seeds": [1],
                    "artifacts": {
                        "config_snapshot": str(config_snapshot_path.relative_to(repo_root)).replace("\\", "/"),
                        "metrics": str(metrics_path.relative_to(repo_root)).replace("\\", "/"),
                        "stdout_log": str(stdout_log_path.relative_to(repo_root)).replace("\\", "/"),
                    },
                },
                indent=2,
            ),
        )
        return {
            "run_manifest": str(run_manifest_path),
        }

    monkeypatch.setattr(
        "recsys_lab.experiments.ml100k_inner_tuning._runner_for_model",
        lambda _model_name: _fake_runner,
    )

    payload = run_inner_tuning(
        tuning_config_path=tuning_config_path,
        processed_manifest_path=processed_manifest_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        use_split_cache=True,
        use_training_index_cache=True,
        use_cluster_artifact_cache=True,
        repo_root=repo_root,
        command="recsys-lab tune-inner --synthetic-random",
    )

    summary = json.loads((Path(payload["benchmark_dir"]) / "summary.json").read_text(encoding="utf-8"))

    assert payload["best_candidate"] == "alpha15"
    assert summary["dataset"] == "ml1m"
    assert summary["split_family"] == "benchmark_random_v1"
    assert summary["selection_unit_kind"] == "split_seed"
    assert summary["selection_units"] == ["s001", "s002"]
    assert summary["train_ratio"] == 0.8
    assert summary["cache_policy"]["split_cache"]["requested"] == "enable"
    assert summary["cache_policy"]["training_index_cache"]["enabled"] is True
    assert summary["cache_policy"]["cluster_artifact_cache"]["enabled"] is True
    assert summary["best_candidate"]["aggregate"]["training_wall_clock_seconds"]["mean"] == 21.0
    assert observed_train_ratios == [0.8, 0.8, 0.8, 0.8]
    assert observed_split_cache == [True, True, True, True]
    assert observed_training_index_cache == [True, True, True, True]
    assert observed_cluster_artifact_cache == [True, True, True, True]


def test_run_inner_tuning_supports_ml1m_cb_asvdpp_and_counts_cluster_time(
    tmp_path: Path,
    monkeypatch,
) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    (
        repo_root,
        tuning_config_path,
        processed_manifest_path,
        runtime_config_path,
        device_config_path,
    ) = _prepare_synthetic_tuning_repo(tmp_path, actual_repo_root)

    model_config_path = repo_root / "configs" / "models" / "cb_asvdpp.yaml"
    _write_text(
        model_config_path,
        "metadata:\n"
        "  status: draft\n"
        "model:\n"
        "  name: cb_asvdpp\n"
        "  scope: paper_inspired\n"
        "training:\n"
        "  latent_dim: 64\n"
        "  epochs: 2\n"
        "  learning_rate: 0.0075\n"
        "  lambda_b: 0.025\n"
        "  lambda_p: 0.025\n"
        "  lambda_q: 0.025\n"
        "  lambda_x: 0.025\n"
        "  lambda_y: 0.025\n"
        "  lambda_pC: 0.025\n"
        "  lambda_qC: 0.025\n"
        "  lambda_xC: 0.025\n"
        "  lambda_yC: 0.025\n"
        "  dtype: float32\n"
        "  residual_weight_contract: detached\n"
        "clustering:\n"
        "  n_user_clusters: 80\n"
        "  n_item_clusters: 80\n"
        "  alpha: 0.10\n",
    )
    _write_text(
        tuning_config_path,
        "tuning:\n"
        "  name: ml1m_cb_asvdpp_stage0\n"
        "  dataset_short_name: ml1m\n"
        "  split_family: benchmark_random_v1\n"
        "  selection_stage: stage0\n"
        "  objective: validation_rmse_mean\n"
        "  split_seeds:\n"
        "    - 1\n"
        "    - 2\n"
        "  validation_ratio: 0.1\n"
        "  model_seed: 1\n"
        "base_model_config: configs/models/cb_asvdpp.yaml\n"
        "candidates:\n"
        "  - candidate_id: base\n"
        "    overrides:\n"
        "      clustering:\n"
        "        alpha: 0.10\n"
        "  - candidate_id: alpha15\n"
        "    overrides:\n"
        "      clustering:\n"
        "        alpha: 0.15\n",
    )
    _write_text(
        processed_manifest_path,
        json.dumps(
            {
                "dataset_short_name": "ml1m",
                "dataset_name": "MovieLens 1M",
                "preprocessing_family": "explicit_v1",
                "split_family": "benchmark_random_v1",
                "artifacts": {
                    "interactions": "data/processed/ml1m/interactions.parquet",
                },
            },
            indent=2,
        ),
    )

    monkeypatch.setattr(
        "recsys_lab.experiments.ml100k_inner_tuning.git_snapshot",
        lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": False},
    )

    call_index = {"value": 0}
    observed_train_ratios: list[float] = []
    observed_cluster_artifact_cache: list[bool | None] = []

    def _fake_runner(**kwargs):
        call_index["value"] += 1
        candidate_name = Path(kwargs["model_config_path"]).stem
        split_seed = int(kwargs["split_config"].seed)
        observed_train_ratios.append(float(kwargs["split_config"].train_ratio))
        observed_cluster_artifact_cache.append(kwargs.get("use_cluster_artifact_cache"))
        run_id = f"2026-04-16T20000{call_index['value']}Z_ml1m_cb_asvdpp_local_test_s001"
        run_dir = repo_root / "artifacts" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.json"
        config_snapshot_path = run_dir / "config_snapshot.yaml"
        stdout_log_path = run_dir / "stdout.log"
        run_manifest_path = run_dir / "run_manifest.json"

        validation_rmse = 0.906 if candidate_name == "base" else 0.899
        validation_rmse += split_seed * 0.001
        train_rmse = validation_rmse - 0.07
        cluster_seconds = 7.0 * split_seed
        training_seconds = 14.0 + split_seed

        _write_text(config_snapshot_path, "candidate: test\n")
        _write_text(stdout_log_path, f"run_id={run_id}\n")
        _write_text(
            metrics_path,
            json.dumps(
                {
                    "run_id": run_id,
                    "metrics": {
                        "train_rmse": train_rmse,
                        "validation_rmse": validation_rmse,
                        "test_rmse": None,
                    },
                    "timing": {
                        "cluster_induction_wall_clock_seconds": cluster_seconds,
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
                    "generated_at_utc": "2026-04-16T200000Z",
                    "run_id": run_id,
                    "status": "completed",
                    "command": "recsys-lab tune-inner --synthetic-cb-asvdpp",
                    "cwd": ".",
                    "git": {
                        "commit": "abcdef1234567",
                        "branch": "main",
                        "dirty": False,
                    },
                    "dataset": {
                        "short_name": "ml1m",
                        "split_family": "benchmark_random_v1",
                    },
                    "model": {
                        "name": "cb_asvdpp",
                        "scope": "paper_inspired",
                        "config_ref": str(Path(kwargs["model_config_path"]).relative_to(repo_root)).replace("\\", "/"),
                    },
                    "runtime": {
                        "device_profile": "local_test",
                        "python_version": "3.11.9",
                        "dtype": "float32",
                        "threading": {"omp_num_threads": 1, "blas_threads": 1},
                    },
                    "seeds": [1],
                    "artifacts": {
                        "config_snapshot": str(config_snapshot_path.relative_to(repo_root)).replace("\\", "/"),
                        "metrics": str(metrics_path.relative_to(repo_root)).replace("\\", "/"),
                        "stdout_log": str(stdout_log_path.relative_to(repo_root)).replace("\\", "/"),
                    },
                },
                indent=2,
            ),
        )
        return {
            "run_manifest": str(run_manifest_path),
        }

    monkeypatch.setattr(
        "recsys_lab.experiments.ml100k_inner_tuning._runner_for_model",
        lambda _model_name: _fake_runner,
    )

    payload = run_inner_tuning(
        tuning_config_path=tuning_config_path,
        processed_manifest_path=processed_manifest_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        use_cluster_artifact_cache=True,
        repo_root=repo_root,
        command="recsys-lab tune-inner --synthetic-cb-asvdpp",
    )

    summary = json.loads((Path(payload["benchmark_dir"]) / "summary.json").read_text(encoding="utf-8"))

    assert payload["best_candidate"] == "alpha15"
    assert summary["dataset"] == "ml1m"
    assert summary["model"] == "cb_asvdpp"
    assert summary["split_family"] == "benchmark_random_v1"
    assert summary["selection_unit_kind"] == "split_seed"
    assert summary["selection_units"] == ["s001", "s002"]
    assert summary["train_ratio"] == 0.8
    assert summary["cache_policy"]["cluster_artifact_cache"]["enabled"] is True
    assert summary["best_candidate"]["aggregate"]["training_wall_clock_seconds"]["mean"] == 26.0
    assert observed_train_ratios == [0.8, 0.8, 0.8, 0.8]
    assert observed_cluster_artifact_cache == [True, True, True, True]
