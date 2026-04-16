from __future__ import annotations

import traceback
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from recsys_lab.clustering import induce_train_only_clusters
from recsys_lab.config.loader import dump_yaml_file, load_yaml_file
from recsys_lab.data.histories import build_user_cluster_count_index
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
from recsys_lab.models.biased_mf import BiasedMFConfig
from recsys_lab.models.cb_svdpp import CBSVDppConfig, CBSVDppRecommender
from recsys_lab.utils.manifests import validate_manifest_file
from recsys_lab.utils.paths import discover_repo_root, repo_path_string


def _build_cb_svdpp_config(
    *,
    model_config_payload: dict[str, Any],
    model_seed: int,
    runtime_dtype: str,
) -> CBSVDppConfig:
    training = model_config_payload.get("training", {})
    clustering = model_config_payload.get("clustering", {})
    return CBSVDppConfig(
        latent_dim=int(training.get("latent_dim", 50)),
        epochs=int(training.get("epochs", 20)),
        learning_rate=float(training.get("learning_rate", 0.01)),
        lambda_b=float(training.get("lambda_b", 0.02)),
        lambda_p=float(training.get("lambda_p", 0.02)),
        lambda_q=float(training.get("lambda_q", 0.02)),
        lambda_y=float(training.get("lambda_y", 0.02)),
        lambda_pC=float(training.get("lambda_pC", 0.02)),
        lambda_qC=float(training.get("lambda_qC", 0.02)),
        lambda_yC=float(training.get("lambda_yC", 0.02)),
        alpha=float(clustering.get("alpha", 0.10)),
        seed=model_seed,
        init_std=float(training.get("init_std", 0.1)),
        dtype=runtime_dtype,
        implicit_policy=str(training.get("implicit_policy", "ratings_as_implicit")),
    )


def _build_induction_config(
    *,
    cb_config: CBSVDppConfig,
    model_seed: int,
) -> BiasedMFConfig:
    return BiasedMFConfig(
        latent_dim=cb_config.latent_dim,
        epochs=cb_config.epochs,
        learning_rate=cb_config.learning_rate,
        lambda_b=cb_config.lambda_b,
        lambda_p=cb_config.lambda_p,
        lambda_q=cb_config.lambda_q,
        seed=model_seed,
        init_std=cb_config.init_std,
        dtype=cb_config.dtype,
    )


def run_cb_svdpp_experiment(
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
        model_name="cb_svdpp",
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
        "recsys-lab train-cb-svdpp "
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
            model_name="cb_svdpp",
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
                raise ValueError(f"unsupported split family for cb_svdpp: {requested_split_family}")

            cb_config = _build_cb_svdpp_config(
                model_config_payload=model_config_payload,
                model_seed=model_seed,
                runtime_dtype=runtime_dtype,
            )
            induction_config = _build_induction_config(
                cb_config=cb_config,
                model_seed=model_seed,
            )
            clustering_config = model_config_payload.get("clustering", {})

            with PeakMemoryMonitor() as memory_monitor:
                clustering_started = perf_counter()
                cluster_artifacts = induce_train_only_clusters(
                    split.train,
                    induction_config=induction_config,
                    n_user_clusters=int(clustering_config.get("n_user_clusters", 100)),
                    n_item_clusters=int(clustering_config.get("n_item_clusters", 100)),
                    algorithm=str(clustering_config.get("algorithm", "kmeans")),
                    kmeans_n_init=int(clustering_config.get("kmeans_n_init", 10)),
                )
                clustering_seconds = perf_counter() - clustering_started

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
                cluster_history_index = build_user_cluster_count_index(
                    history_index,
                    cluster_artifacts.item_clusters,
                    n_clusters=cluster_artifacts.r_star_counts.shape[1],
                )

                model = CBSVDppRecommender(
                    cb_config,
                    user_clusters=cluster_artifacts.user_clusters,
                    item_clusters=cluster_artifacts.item_clusters,
                    n_user_clusters=cluster_artifacts.r_star_counts.shape[0],
                    n_item_clusters=cluster_artifacts.r_star_counts.shape[1],
                )

                training_started = perf_counter()
                model.fit(
                    split.train,
                    user_histories=history_index if reuse_precomputed_indices else None,
                    user_cluster_histories=cluster_history_index if reuse_precomputed_indices else None,
                )
                training_seconds = perf_counter() - training_started

                inference_started = perf_counter()
                train_predictions = model.predict_dataset(split.train)
                validation_predictions = (
                    None if split.validation is None else model.predict_dataset(split.validation)
                )
                test_predictions = None if not evaluate_test else model.predict_dataset(split.test)
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
                train_time_total_seconds=clustering_seconds + training_seconds,
                extra_fields={
                    "cluster_induction_wall_clock_seconds": float(clustering_seconds),
                    "main_training_wall_clock_seconds": float(training_seconds),
                },
            )

            history_counts = history_index.counts.astype("int64", copy=False)
            per_user_active_cluster_counts = np.diff(cluster_history_index.indptr).astype("int64", copy=False)
            r_star_counts = cluster_artifacts.r_star_counts.astype("int64", copy=False)
            nonempty_cluster_pairs = int((r_star_counts > 0).sum())
            total_cluster_pairs = int(r_star_counts.size)

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
                    "name": "cb_svdpp",
                    "config": asdict(model.config),
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
                        "user_cluster_history": {
                            "status": "not_persisted",
                            "reason": "depends_on_train_only_cluster_induction",
                        },
                    },
                    "clustering": {
                        "induction_model": "biased_mf",
                        "induction_config": asdict(induction_config),
                        "algorithm": str(clustering_config.get("algorithm", "kmeans")),
                        "kmeans_n_init": int(clustering_config.get("kmeans_n_init", 10)),
                        "train_only_assignments": True,
                        "fixed_assignments_during_cb_training": True,
                        "r_star_role": "diagnostic_only",
                        "n_user_clusters": int(cluster_artifacts.r_star_counts.shape[0]),
                        "n_item_clusters": int(cluster_artifacts.r_star_counts.shape[1]),
                        "user_cluster_summary": {
                            "min_size": int(cluster_artifacts.user_cluster_sizes.min()),
                            "max_size": int(cluster_artifacts.user_cluster_sizes.max()),
                            "mean_size": float(cluster_artifacts.user_cluster_sizes.mean()),
                        },
                        "item_cluster_summary": {
                            "min_size": int(cluster_artifacts.item_cluster_sizes.min()),
                            "max_size": int(cluster_artifacts.item_cluster_sizes.max()),
                            "mean_size": float(cluster_artifacts.item_cluster_sizes.mean()),
                        },
                        "history_cluster_summary": {
                            "mean_active_item_clusters_per_user": float(per_user_active_cluster_counts.mean()),
                            "max_active_item_clusters_per_user": int(per_user_active_cluster_counts.max()),
                            "users_with_history": int((history_counts > 0).sum()),
                        },
                        "r_star_summary": {
                            "nonempty_pairs": nonempty_cluster_pairs,
                            "total_pairs": total_cluster_pairs,
                            "density": float(nonempty_cluster_pairs / total_cluster_pairs),
                            "observed_rating_min": float(
                                cluster_artifacts.r_star_means[r_star_counts > 0].min()
                            ),
                            "observed_rating_max": float(
                                cluster_artifacts.r_star_means[r_star_counts > 0].max()
                            ),
                        },
                        "induction_diagnostics": {
                            "train_rmse": cluster_artifacts.induction_train_rmse,
                            "user_kmeans_inertia": cluster_artifacts.user_kmeans_inertia,
                            "item_kmeans_inertia": cluster_artifacts.item_kmeans_inertia,
                        },
                    },
                    "implicit_summary": {
                        "users_with_history": int((history_counts > 0).sum()),
                        "mean_history_size": float(history_counts.mean()),
                        "max_history_size": int(history_counts.max()),
                    },
                },
                "timing": {
                    "cluster_induction_wall_clock_seconds": clustering_seconds,
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
            validation_rmse_value = metrics_payload["metrics"]["validation_rmse"]
            test_rmse_value = metrics_payload["metrics"]["test_rmse"]
            validation_rmse_display = (
                "NA" if validation_rmse_value is None else f"{validation_rmse_value:.6f}"
            )
            test_rmse_display = "NA" if test_rmse_value is None else f"{test_rmse_value:.6f}"
            write_log(
                stdout_log_path,
                [
                    f"[{timestamp}] run_id={run_id}",
                    f"command={command_string}",
                    f"processed_manifest={repo_path_string(processed_manifest_path, repo_root=root)}",
                    f"train_rows={len(split.train)} validation_rows={0 if split.validation is None else len(split.validation)} test_rows={len(split.test)}",
                    (
                        "clustering "
                        f"user_clusters={metrics_payload['model']['clustering']['n_user_clusters']} "
                        f"item_clusters={metrics_payload['model']['clustering']['n_item_clusters']} "
                        f"r_star_density={metrics_payload['model']['clustering']['r_star_summary']['density']:.6f}"
                    ),
                    (
                        "rmse "
                        f"train={metrics_payload['metrics']['train_rmse']:.6f} "
                        f"validation={validation_rmse_display} "
                        f"test={test_rmse_display}"
                    ),
                    f"cluster_induction_wall_clock_seconds={clustering_seconds:.6f}",
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
