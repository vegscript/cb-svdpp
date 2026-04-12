from __future__ import annotations

import traceback
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from threadpoolctl import threadpool_limits

from recsys_lab.config.loader import dump_yaml_file, load_yaml_file
from recsys_lab.data.histories import build_user_history_index
from recsys_lab.data.processed import load_processed_dataset_manifest, load_ratings_data_from_manifest
from recsys_lab.data.splitters import random_split_with_train_coverage
from recsys_lab.experiments.common import (
    SplitConfig,
    build_base_run_manifest,
    build_run_id,
    git_snapshot,
    ratings_summary,
    resolve_runtime_dtype,
    split_summary,
    utc_timestamp,
    write_json,
    write_log,
)
from recsys_lab.metrics import rmse
from recsys_lab.models.svdpp import SVDppConfig, SVDppRecommender
from recsys_lab.utils.manifests import validate_manifest_file
from recsys_lab.utils.paths import discover_repo_root, repo_path_string


def _build_svdpp_config(
    *,
    model_config_payload: dict[str, Any],
    model_seed: int,
    runtime_dtype: str,
) -> SVDppConfig:
    training = model_config_payload.get("training", {})
    return SVDppConfig(
        latent_dim=int(training.get("latent_dim", 50)),
        epochs=int(training.get("epochs", 20)),
        learning_rate=float(training.get("learning_rate", 0.01)),
        lambda_b=float(training.get("lambda_b", 0.02)),
        lambda_p=float(training.get("lambda_p", 0.02)),
        lambda_q=float(training.get("lambda_q", 0.02)),
        lambda_y=float(training.get("lambda_y", 0.02)),
        seed=model_seed,
        init_std=float(training.get("init_std", 0.1)),
        dtype=runtime_dtype,
        implicit_policy=str(training.get("implicit_policy", "ratings_as_implicit")),
    )


def run_svdpp_experiment(
    *,
    processed_manifest_path: Path,
    model_config_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    split_config: SplitConfig,
    model_seed: int,
    repo_root: Path | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    root = (repo_root or discover_repo_root()).resolve()

    processed_manifest_path = processed_manifest_path.resolve()
    model_config_path = model_config_path.resolve()
    runtime_config_path = runtime_config_path.resolve()
    device_config_path = device_config_path.resolve()

    processed_manifest = load_processed_dataset_manifest(processed_manifest_path)
    dataset_short_name = str(processed_manifest["dataset_short_name"])

    runtime_config_payload = load_yaml_file(runtime_config_path)
    device_config_payload = load_yaml_file(device_config_path)
    model_config_payload = load_yaml_file(model_config_path)
    runtime_dtype = resolve_runtime_dtype(
        runtime_config_payload=runtime_config_payload,
        device_config_payload=device_config_payload,
        model_config_payload=model_config_payload,
    )

    timestamp = utc_timestamp()
    device_profile_name = str(device_config_payload["device_profile"]["name"])
    run_id = build_run_id(
        timestamp=timestamp,
        dataset_short_name=dataset_short_name,
        model_name="svdpp",
        device_profile_name=device_profile_name,
        model_seed=model_seed,
    )

    run_dir = root / "artifacts" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    config_snapshot_path = run_dir / "config_snapshot.yaml"
    metrics_path = run_dir / "metrics.json"
    stdout_log_path = run_dir / "stdout.log"
    run_manifest_path = run_dir / "run_manifest.json"

    git = git_snapshot(root)
    command_string = command or (
        "recsys-lab train-svdpp "
        f"--processed-manifest {repo_path_string(processed_manifest_path, repo_root=root)} "
        f"--model-config {repo_path_string(model_config_path, repo_root=root)} "
        f"--runtime-config {repo_path_string(runtime_config_path, repo_root=root)} "
        f"--device-config {repo_path_string(device_config_path, repo_root=root)} "
        f"--split-seed {split_config.seed} --model-seed {model_seed}"
    )

    base_manifest = build_base_run_manifest(
        timestamp=timestamp,
        run_id=run_id,
        command=command_string,
        repo_root=root,
        git=git,
        processed_manifest=processed_manifest,
        processed_manifest_path=processed_manifest_path,
        model_name="svdpp",
        model_scope=str(model_config_payload.get("model", {}).get("scope", "paper_inspired")),
        model_config_path=model_config_path,
        device_profile_name=device_profile_name,
        runtime_dtype=runtime_dtype,
        device_config_payload=device_config_payload,
        model_seed=model_seed,
        split_config=split_config,
        config_snapshot_path=config_snapshot_path,
        metrics_path=metrics_path,
        stdout_log_path=stdout_log_path,
    )

    dump_yaml_file(
        config_snapshot_path,
        {
            "run_id": run_id,
            "command": command_string,
            "inputs": {
                "processed_manifest": repo_path_string(processed_manifest_path, repo_root=root),
                "model_config": repo_path_string(model_config_path, repo_root=root),
                "runtime_config": repo_path_string(runtime_config_path, repo_root=root),
                "device_config": repo_path_string(device_config_path, repo_root=root),
            },
            "split": asdict(split_config),
            "model_seed": model_seed,
            "loaded_configs": {
                "processed_manifest": processed_manifest,
                "model": model_config_payload,
                "runtime": runtime_config_payload,
                "device": device_config_payload,
            },
        },
    )

    write_log(
        stdout_log_path,
        [
            f"[{timestamp}] run_id={run_id}",
            f"command={command_string}",
            f"processed_manifest={repo_path_string(processed_manifest_path, repo_root=root)}",
        ],
    )
    write_json(run_manifest_path, base_manifest)

    try:
        ratings_data = load_ratings_data_from_manifest(processed_manifest_path)
        split = random_split_with_train_coverage(
            ratings_data,
            train_ratio=split_config.train_ratio,
            validation_ratio=split_config.validation_ratio,
            seed=split_config.seed,
        )
        history_index = build_user_history_index(split.train, dtype=runtime_dtype)

        model = SVDppRecommender(
            _build_svdpp_config(
                model_config_payload=model_config_payload,
                model_seed=model_seed,
                runtime_dtype=runtime_dtype,
            )
        )

        threading_config = base_manifest["runtime"]["threading"]
        training_started = perf_counter()
        with threadpool_limits(limits=int(threading_config["blas_threads"])):
            model.fit(split.train)
        training_seconds = perf_counter() - training_started

        train_predictions = model.predict_dataset(split.train)
        validation_predictions = model.predict_dataset(split.validation)
        test_predictions = model.predict_dataset(split.test)

        history_counts = history_index.counts.astype("int64", copy=False)
        metrics_payload = {
            "run_id": run_id,
            "dataset": ratings_summary(ratings_data),
            "split": {
                **asdict(split_config),
                **split_summary(split),
            },
            "model": {
                "name": "svdpp",
                "config": asdict(model.config),
                "implicit_summary": {
                    "users_with_history": int((history_counts > 0).sum()),
                    "mean_history_size": float(history_counts.mean()),
                    "max_history_size": int(history_counts.max()),
                },
            },
            "timing": {
                "training_wall_clock_seconds": training_seconds,
            },
            "metrics": {
                "train_rmse": rmse(split.train.ratings, train_predictions),
                "validation_rmse": rmse(split.validation.ratings, validation_predictions),
                "test_rmse": rmse(split.test.ratings, test_predictions),
            },
        }
        write_json(metrics_path, metrics_payload)

        finished_at = utc_timestamp()
        write_log(
            stdout_log_path,
            [
                f"[{timestamp}] run_id={run_id}",
                f"command={command_string}",
                f"processed_manifest={repo_path_string(processed_manifest_path, repo_root=root)}",
                f"train_rows={len(split.train)} validation_rows={len(split.validation)} test_rows={len(split.test)}",
                (
                    "rmse "
                    f"train={metrics_payload['metrics']['train_rmse']:.6f} "
                    f"validation={metrics_payload['metrics']['validation_rmse']:.6f} "
                    f"test={metrics_payload['metrics']['test_rmse']:.6f}"
                ),
                (
                    "implicit_summary "
                    f"users_with_history={metrics_payload['model']['implicit_summary']['users_with_history']} "
                    f"mean_history_size={metrics_payload['model']['implicit_summary']['mean_history_size']:.4f} "
                    f"max_history_size={metrics_payload['model']['implicit_summary']['max_history_size']}"
                ),
                f"training_wall_clock_seconds={training_seconds:.6f}",
                f"[{finished_at}] status=completed",
            ],
        )

        completed_manifest = {
            **base_manifest,
            "status": "completed",
            "generated_at_utc": finished_at,
            "timing": {
                **base_manifest["timing"],
                "finished_at_utc": finished_at,
            },
        }
        write_json(run_manifest_path, completed_manifest)
        validate_manifest_file(run_manifest_path, repo_root=root)
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "run_manifest": str(run_manifest_path),
            "metrics": metrics_payload["metrics"],
        }
    except Exception:
        finished_at = utc_timestamp()
        write_log(
            stdout_log_path,
            [
                f"[{timestamp}] run_id={run_id}",
                f"command={command_string}",
                f"[{finished_at}] status=failed",
                traceback.format_exc().strip(),
            ],
        )
        failed_manifest = {
            **base_manifest,
            "status": "failed",
            "generated_at_utc": finished_at,
            "timing": {
                **base_manifest["timing"],
                "finished_at_utc": finished_at,
            },
        }
        write_json(run_manifest_path, failed_manifest)
        validate_manifest_file(run_manifest_path, repo_root=root)
        raise
