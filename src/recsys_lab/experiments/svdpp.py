from __future__ import annotations

import traceback
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from recsys_lab.config.loader import dump_yaml_file, load_yaml_file
from recsys_lab.data.processed import load_processed_dataset_manifest, load_ratings_data_from_manifest
from recsys_lab.data.training_index_cache import load_or_build_user_history_index
from recsys_lab.data.splitters import (
    official_ml100k_inner_validation_split,
    official_ml100k_paper_faithful_split,
    random_split_with_train_coverage,
)
from recsys_lab.experiments.common import (
    SplitConfig,
    build_base_run_manifest,
    build_run_id,
    git_snapshot,
    paper_faithful_ml100k_inner_split_id,
    paper_faithful_ml100k_split_id,
    ratings_summary,
    resolve_runtime_dtype,
    split_summary,
    utc_timestamp,
    write_json,
    write_log,
)
from recsys_lab.experiments.performance import PeakMemoryMonitor, build_system_metrics
from recsys_lab.experiments.runtime import (
    resolve_runtime_threading_config,
    runtime_execution_context,
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
        training_backend=str(training.get("training_backend", "auto")),
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
    split_family: str | None = None,
    inner_validation_seed: int | None = None,
    evaluate_test: bool = True,
    reuse_precomputed_indices: bool = True,
    use_training_index_cache: bool = False,
) -> dict[str, Any]:
    root = (repo_root or discover_repo_root()).resolve()

    processed_manifest_path = processed_manifest_path.resolve()
    model_config_path = model_config_path.resolve()
    runtime_config_path = runtime_config_path.resolve()
    device_config_path = device_config_path.resolve()

    processed_manifest = load_processed_dataset_manifest(processed_manifest_path)
    dataset_short_name = str(processed_manifest["dataset_short_name"])
    requested_split_family = split_family or str(processed_manifest["split_family"])

    runtime_config_payload = load_yaml_file(runtime_config_path)
    device_config_payload = load_yaml_file(device_config_path)
    model_config_payload = load_yaml_file(model_config_path)
    runtime_dtype = resolve_runtime_dtype(
        runtime_config_payload=runtime_config_payload,
        device_config_payload=device_config_payload,
        model_config_payload=model_config_payload,
    )
    threading_config = resolve_runtime_threading_config(device_config_payload=device_config_payload)

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
        f"--split-family {requested_split_family} "
        f"--split-seed {split_config.seed} "
        f"{'' if inner_validation_seed is None else f'--inner-validation-seed {inner_validation_seed} '}"
        f"{'' if evaluate_test else '--skip-test-eval '}"
        f"{'' if use_training_index_cache else '--disable-training-index-cache '}"
        f"--model-seed {model_seed}"
    )

    if requested_split_family == "paper_faithful_ml100k_v1":
        split_id_value = paper_faithful_ml100k_split_id(split_config.seed)
    elif requested_split_family == "paper_faithful_ml100k_inner_v1":
        if inner_validation_seed is None:
            raise ValueError("inner_validation_seed is required for paper_faithful_ml100k_inner_v1")
        split_id_value = paper_faithful_ml100k_inner_split_id(
            fold_index=split_config.seed,
            validation_ratio=split_config.validation_ratio,
            inner_seed=inner_validation_seed,
        )
    else:
        split_id_value = None

    with runtime_execution_context(threading_config=threading_config):
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
            split_family_name=requested_split_family,
            split_id_value=split_id_value,
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
                "reuse_precomputed_indices": reuse_precomputed_indices,
                "use_training_index_cache": use_training_index_cache,
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
            if requested_split_family == "benchmark_random_v1":
                split = random_split_with_train_coverage(
                    ratings_data,
                    train_ratio=split_config.train_ratio,
                    validation_ratio=split_config.validation_ratio,
                    seed=split_config.seed,
                )
            elif requested_split_family == "paper_faithful_ml100k_v1":
                split = official_ml100k_paper_faithful_split(
                    ratings_data,
                    processed_manifest_path=processed_manifest_path,
                    fold_index=split_config.seed,
                )
            elif requested_split_family == "paper_faithful_ml100k_inner_v1":
                if inner_validation_seed is None:
                    raise ValueError("inner_validation_seed is required for paper_faithful_ml100k_inner_v1")
                split = official_ml100k_inner_validation_split(
                    ratings_data,
                    processed_manifest_path=processed_manifest_path,
                    fold_index=split_config.seed,
                    validation_ratio=split_config.validation_ratio,
                    inner_seed=inner_validation_seed,
                )
            else:
                raise ValueError(f"unsupported split family for svdpp: {requested_split_family}")
            split_id_for_cache = str(base_manifest["dataset"]["split_id"])
            history_index_result = load_or_build_user_history_index(
                data=split.train,
                dtype=runtime_dtype,
                dataset_short_name=dataset_short_name,
                split_family=requested_split_family,
                split_id=split_id_for_cache,
                processed_manifest_path=processed_manifest_path,
                repo_root=root,
                runtime_config_payload=runtime_config_payload,
                use_cache=use_training_index_cache,
            )
            history_index = history_index_result.index

            model = SVDppRecommender(
                _build_svdpp_config(
                    model_config_payload=model_config_payload,
                    model_seed=model_seed,
                    runtime_dtype=runtime_dtype,
                )
            )

            with PeakMemoryMonitor() as memory_monitor:
                training_started = perf_counter()
                model.fit(
                    split.train,
                    user_histories=history_index if reuse_precomputed_indices else None,
                )
                training_seconds = perf_counter() - training_started

                inference_started = perf_counter()
                train_predictions = model.predict_dataset(split.train)
                test_predictions = None if not evaluate_test else model.predict_dataset(split.test)
                validation_predictions = (
                    None if split.validation is None else model.predict_dataset(split.validation)
                )
                inference_seconds = perf_counter() - inference_started

            inference_rows = len(split.train)
            if split.validation is not None:
                inference_rows += len(split.validation)
            if evaluate_test:
                inference_rows += len(split.test)
            system_metrics = build_system_metrics(
                train_rows=len(split.train),
                epochs=model.config.epochs,
                training_wall_clock_seconds=training_seconds,
                inference_rows=inference_rows,
                inference_wall_clock_seconds=inference_seconds,
                peak_memory_bytes=memory_monitor.peak_rss_bytes,
                baseline_memory_bytes=memory_monitor.start_rss_bytes,
                model=model,
                epoch_durations_seconds=model.epoch_durations_seconds,
            )

            history_counts = history_index.counts.astype("int64", copy=False)
            metrics_payload = {
                "run_id": run_id,
                "dataset": ratings_summary(ratings_data),
                "split": {
                    "family": requested_split_family,
                    **asdict(split_config),
                    "inner_validation_seed": inner_validation_seed,
                    "test_metrics_available": evaluate_test,
                    **split_summary(split),
                },
                "model": {
                    "name": "svdpp",
                    "config": asdict(model.config),
                    "training_backend_effective": model.training_backend_effective,
                    "precomputed_index_reuse": reuse_precomputed_indices,
                    "training_index_cache": {
                        "enabled": use_training_index_cache,
                        "split_id": split_id_for_cache,
                        "cache_root": repo_path_string(
                            history_index_result.metadata.cache_root,
                            repo_root=root,
                        ),
                        "train_fingerprint_sha256": history_index_result.metadata.train_fingerprint.sha256,
                        "user_history": {
                            "status": history_index_result.metadata.cache_status,
                            "manifest": repo_path_string(
                                history_index_result.metadata.cache_manifest_path,
                                repo_root=root,
                            ),
                        },
                    },
                    "implicit_summary": {
                        "users_with_history": int((history_counts > 0).sum()),
                        "mean_history_size": float(history_counts.mean()),
                        "max_history_size": int(history_counts.max()),
                    },
                },
                "timing": {
                    "training_wall_clock_seconds": training_seconds,
                    "inference_wall_clock_seconds": inference_seconds,
                },
                "system_metrics": system_metrics,
                "metrics": {
                    "train_rmse": rmse(split.train.ratings, train_predictions),
                    "validation_rmse": (
                        None
                        if split.validation is None or validation_predictions is None
                        else rmse(split.validation.ratings, validation_predictions)
                    ),
                    "test_rmse": (
                        None
                        if not evaluate_test or test_predictions is None
                        else rmse(split.test.ratings, test_predictions)
                    ),
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
                    (
                        f"train_rows={len(split.train)} "
                        f"validation_rows={0 if split.validation is None else len(split.validation)} "
                        f"test_rows={len(split.test)}"
                    ),
                    (
                        "rmse "
                        f"train={metrics_payload['metrics']['train_rmse']:.6f} "
                        f"validation={'NA' if metrics_payload['metrics']['validation_rmse'] is None else format(metrics_payload['metrics']['validation_rmse'], '.6f')} "
                        f"test={'NA' if metrics_payload['metrics']['test_rmse'] is None else format(metrics_payload['metrics']['test_rmse'], '.6f')}"
                    ),
                    (
                        "implicit_summary "
                        f"users_with_history={metrics_payload['model']['implicit_summary']['users_with_history']} "
                        f"mean_history_size={metrics_payload['model']['implicit_summary']['mean_history_size']:.4f} "
                        f"max_history_size={metrics_payload['model']['implicit_summary']['max_history_size']}"
                    ),
                    f"training_wall_clock_seconds={training_seconds:.6f}",
                    (
                        "system_metrics "
                        f"ratings_per_second_train={system_metrics['ratings_per_second_train']:.6f} "
                        f"ratings_per_second_inference={system_metrics['ratings_per_second_inference']:.6f} "
                        f"peak_memory_mb={system_metrics['peak_memory_mb']:.6f} "
                        f"model_size_mb={system_metrics['model_size_mb']:.6f}"
                    ),
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
