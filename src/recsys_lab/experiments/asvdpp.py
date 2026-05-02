from __future__ import annotations

import traceback
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from recsys_lab.config.loader import dump_yaml_file, load_yaml_file
from recsys_lab.data.histories import build_user_explicit_feedback_index, build_user_history_index
from recsys_lab.data.processed import (
    load_processed_dataset_manifest,
    load_ratings_data_from_manifest,
)
from recsys_lab.data.splitters import random_split_with_train_coverage
from recsys_lab.data.training_index_cache import (
    load_or_build_user_explicit_feedback_index,
    load_or_build_user_history_index,
)
from recsys_lab.experiments.common import (
    SplitConfig,
    build_base_run_manifest,
    build_run_id,
    git_snapshot,
    ratings_summary,
    reserve_timestamped_artifact_dir,
    resolve_runtime_dtype,
    split_id,
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
from recsys_lab.experiments.split_cache import load_or_build_split_cache, resolve_split_cache_policy
from recsys_lab.metrics import rmse
from recsys_lab.models.asvdpp import ASVDppConfig, ASVDppRecommender
from recsys_lab.utils.manifests import validate_manifest_file
from recsys_lab.utils.paths import discover_repo_root, repo_path_string


def _build_asvdpp_config(
    *,
    model_config_payload: dict[str, Any],
    model_seed: int,
    runtime_dtype: str,
) -> ASVDppConfig:
    training = model_config_payload.get("training", {})
    return ASVDppConfig(
        latent_dim=int(training.get("latent_dim", 50)),
        epochs=int(training.get("epochs", 20)),
        learning_rate=float(training.get("learning_rate", 0.01)),
        lambda_b=float(training.get("lambda_b", 0.02)),
        lambda_p=float(training.get("lambda_p", 0.02)),
        lambda_q=float(training.get("lambda_q", 0.02)),
        lambda_x=float(training.get("lambda_x", 0.02)),
        lambda_y=float(training.get("lambda_y", 0.02)),
        seed=model_seed,
        init_std=float(training.get("init_std", 0.1)),
        dtype=runtime_dtype,
        implicit_policy=str(training.get("implicit_policy", "ratings_as_implicit")),
        residual_weight_contract=str(training.get("residual_weight_contract", "detached")),
    )


def run_asvdpp_experiment(
    *,
    processed_manifest_path: Path,
    model_config_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    split_config: SplitConfig,
    model_seed: int,
    repo_root: Path | None = None,
    command: str | None = None,
    use_split_cache: bool | None = None,
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
    split_cache_policy = resolve_split_cache_policy(
        split_family="benchmark_random_v1",
        use_split_cache=use_split_cache,
    )

    runtime_config_payload = load_yaml_file(runtime_config_path)
    device_config_payload = load_yaml_file(device_config_path)
    model_config_payload = load_yaml_file(model_config_path)
    runtime_dtype = resolve_runtime_dtype(
        runtime_config_payload=runtime_config_payload,
        device_config_payload=device_config_payload,
        model_config_payload=model_config_payload,
    )
    threading_config = resolve_runtime_threading_config(device_config_payload=device_config_payload)

    run_context_slug = split_id("benchmark_random_v1", split_config)

    device_profile_name = str(device_config_payload["device_profile"]["name"])
    timestamp, run_id, run_dir = reserve_timestamped_artifact_dir(
        artifacts_root=root / "artifacts" / "runs",
        id_from_timestamp=lambda reserved_timestamp: build_run_id(
            timestamp=reserved_timestamp,
            dataset_short_name=dataset_short_name,
            model_name="asvdpp",
            device_profile_name=device_profile_name,
            model_seed=model_seed,
            split_id_value=run_context_slug,
        ),
    )

    config_snapshot_path = run_dir / "config_snapshot.yaml"
    metrics_path = run_dir / "metrics.json"
    stdout_log_path = run_dir / "stdout.log"
    run_manifest_path = run_dir / "run_manifest.json"

    git = git_snapshot(root)
    split_cache_command_fragment = (
        "" if use_split_cache is None else f"--split-cache {'enable' if use_split_cache else 'disable'} "
    )
    command_string = command or (
        "recsys-lab train-asvdpp "
        f"--processed-manifest {repo_path_string(processed_manifest_path, repo_root=root)} "
        f"--model-config {repo_path_string(model_config_path, repo_root=root)} "
        f"--runtime-config {repo_path_string(runtime_config_path, repo_root=root)} "
        f"--device-config {repo_path_string(device_config_path, repo_root=root)} "
        f"{split_cache_command_fragment}"
        f"{'' if use_training_index_cache else '--disable-training-index-cache '}"
        f"--split-seed {split_config.seed} --model-seed {model_seed}"
    )

    with runtime_execution_context(threading_config=threading_config):
        base_manifest = build_base_run_manifest(
            timestamp=timestamp,
            run_id=run_id,
            command=command_string,
            repo_root=root,
            git=git,
            processed_manifest=processed_manifest,
            processed_manifest_path=processed_manifest_path,
            model_name="asvdpp",
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
                "use_split_cache": split_cache_policy.effective_use_cache,
                "use_split_cache_policy_requested": split_cache_policy.requested_policy,
                "use_split_cache_decision_reason": split_cache_policy.decision_reason,
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
            split_id_for_cache = (
                f"benchmark_random_v1_tr{int(round(split_config.train_ratio * 100)):03d}"
                f"_va{int(round(split_config.validation_ratio * 100)):03d}_s{split_config.seed:03d}"
            )
            split_result = load_or_build_split_cache(
                data=ratings_data,
                dataset_short_name=dataset_short_name,
                split_family="benchmark_random_v1",
                split_id=split_id_for_cache,
                processed_manifest_path=processed_manifest_path,
                repo_root=root,
                runtime_config_payload=runtime_config_payload,
                build_split=lambda: random_split_with_train_coverage(
                    ratings_data,
                    train_ratio=split_config.train_ratio,
                    validation_ratio=split_config.validation_ratio,
                    seed=split_config.seed,
                ),
                use_cache=split_cache_policy.effective_use_cache,
            )
            split = split_result.split
            validation = split.validation
            if validation is None:
                raise ValueError("asvdpp experiment requires validation_ratio > 0")
            explicit_index_result = load_or_build_user_explicit_feedback_index(
                data=split.train,
                dtype=runtime_dtype,
                dataset_short_name=dataset_short_name,
                split_family="benchmark_random_v1",
                split_id=split_id_for_cache,
                processed_manifest_path=processed_manifest_path,
                repo_root=root,
                runtime_config_payload=runtime_config_payload,
                use_cache=use_training_index_cache,
            )
            implicit_index_result = load_or_build_user_history_index(
                data=split.train,
                dtype=runtime_dtype,
                dataset_short_name=dataset_short_name,
                split_family="benchmark_random_v1",
                split_id=split_id_for_cache,
                processed_manifest_path=processed_manifest_path,
                repo_root=root,
                runtime_config_payload=runtime_config_payload,
                use_cache=use_training_index_cache,
            )
            if reuse_precomputed_indices:
                explicit_index = explicit_index_result.index
                implicit_index = implicit_index_result.index
            else:
                explicit_index = build_user_explicit_feedback_index(split.train, dtype=runtime_dtype)
                implicit_index = build_user_history_index(split.train, dtype=runtime_dtype)

            model = ASVDppRecommender(
                _build_asvdpp_config(
                    model_config_payload=model_config_payload,
                    model_seed=model_seed,
                    runtime_dtype=runtime_dtype,
                )
            )

            with PeakMemoryMonitor() as memory_monitor:
                training_started = perf_counter()
                model.fit(
                    split.train,
                    explicit_feedback=explicit_index if reuse_precomputed_indices else None,
                    implicit_history=implicit_index if reuse_precomputed_indices else None,
                )
                training_seconds = perf_counter() - training_started

                inference_started = perf_counter()
                train_predictions = model.predict_dataset(split.train)
                validation_predictions = model.predict_dataset(validation)
                test_predictions = model.predict_dataset(split.test)
                inference_seconds = perf_counter() - inference_started

            system_metrics = build_system_metrics(
                train_rows=len(split.train),
                epochs=model.config.epochs,
                training_wall_clock_seconds=training_seconds,
                inference_rows=len(split.train) + len(validation) + len(split.test),
                inference_wall_clock_seconds=inference_seconds,
                peak_memory_bytes=memory_monitor.peak_rss_bytes,
                baseline_memory_bytes=memory_monitor.start_rss_bytes,
                model=model,
                epoch_durations_seconds=model.epoch_durations_seconds,
            )

            explicit_counts = explicit_index.counts.astype("int64", copy=False)
            implicit_counts = implicit_index.counts.astype("int64", copy=False)
            metrics_payload: dict[str, Any] = {
                "run_id": run_id,
                "dataset": ratings_summary(ratings_data),
                "split": {
                    **asdict(split_config),
                    **split_summary(split),
                },
                "model": {
                    "name": "asvdpp",
                    "config": asdict(model.config),
                    "split_cache": {
                        "enabled": split_cache_policy.effective_use_cache,
                        "requested_policy": split_cache_policy.requested_policy,
                        "decision_reason": split_cache_policy.decision_reason,
                        "split_id": split_id_for_cache,
                        "status": split_result.metadata.cache_status,
                        "manifest": repo_path_string(split_result.metadata.cache_manifest_path, repo_root=root),
                    },
                    "precomputed_index_reuse": reuse_precomputed_indices,
                    "training_index_cache": {
                        "enabled": use_training_index_cache,
                        "split_id": split_id_for_cache,
                        "cache_root": repo_path_string(
                            explicit_index_result.metadata.cache_root,
                            repo_root=root,
                        ),
                        "train_fingerprint_sha256": explicit_index_result.metadata.train_fingerprint.sha256,
                        "user_history": {
                            "status": implicit_index_result.metadata.cache_status,
                            "manifest": repo_path_string(
                                implicit_index_result.metadata.cache_manifest_path,
                                repo_root=root,
                            ),
                        },
                        "explicit_feedback": {
                            "status": explicit_index_result.metadata.cache_status,
                            "manifest": repo_path_string(
                                explicit_index_result.metadata.cache_manifest_path,
                                repo_root=root,
                            ),
                        },
                    },
                    "explicit_summary": {
                        "users_with_explicit_history": int((explicit_counts > 0).sum()),
                        "mean_explicit_history_size": float(explicit_counts.mean()),
                        "max_explicit_history_size": int(explicit_counts.max()),
                    },
                    "implicit_summary": {
                        "users_with_history": int((implicit_counts > 0).sum()),
                        "mean_history_size": float(implicit_counts.mean()),
                        "max_history_size": int(implicit_counts.max()),
                    },
                },
                "timing": {
                    "training_wall_clock_seconds": training_seconds,
                    "inference_wall_clock_seconds": inference_seconds,
                },
                "system_metrics": system_metrics,
                "metrics": {
                    "train_rmse": rmse(split.train.ratings, train_predictions),
                    "validation_rmse": rmse(validation.ratings, validation_predictions),
                    "test_rmse": rmse(split.test.ratings, test_predictions),
                },
            }
            write_json(metrics_path, metrics_payload)

            finished_at = utc_timestamp()
            explicit_summary = metrics_payload["model"]["explicit_summary"]
            write_log(
                stdout_log_path,
                [
                    f"[{timestamp}] run_id={run_id}",
                    f"command={command_string}",
                    f"processed_manifest={repo_path_string(processed_manifest_path, repo_root=root)}",
                    (f"train_rows={len(split.train)} validation_rows={len(validation)} test_rows={len(split.test)}"),
                    (
                        "rmse "
                        f"train={metrics_payload['metrics']['train_rmse']:.6f} "
                        f"validation={metrics_payload['metrics']['validation_rmse']:.6f} "
                        f"test={metrics_payload['metrics']['test_rmse']:.6f}"
                    ),
                    (
                        "explicit_summary "
                        f"users_with_explicit_history={explicit_summary['users_with_explicit_history']} "
                        f"mean_explicit_history_size={explicit_summary['mean_explicit_history_size']:.4f} "
                        f"max_explicit_history_size={explicit_summary['max_explicit_history_size']}"
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
