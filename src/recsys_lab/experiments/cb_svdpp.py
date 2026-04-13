from __future__ import annotations

import traceback
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from recsys_lab.clustering import induce_train_only_clusters
from recsys_lab.config.loader import dump_yaml_file, load_yaml_file
from recsys_lab.data.histories import build_user_cluster_count_index, build_user_history_index
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

        threading_config = base_manifest["runtime"]["threading"]
        clustering_started = perf_counter()
        with threadpool_limits(limits=int(threading_config["blas_threads"])):
            cluster_artifacts = induce_train_only_clusters(
                split.train,
                induction_config=induction_config,
                n_user_clusters=int(clustering_config.get("n_user_clusters", 100)),
                n_item_clusters=int(clustering_config.get("n_item_clusters", 100)),
                algorithm=str(clustering_config.get("algorithm", "kmeans")),
                kmeans_n_init=int(clustering_config.get("kmeans_n_init", 10)),
            )
        clustering_seconds = perf_counter() - clustering_started

        history_index = build_user_history_index(split.train, dtype=runtime_dtype)
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
        with threadpool_limits(limits=int(threading_config["blas_threads"])):
            model.fit(split.train)
        training_seconds = perf_counter() - training_started

        train_predictions = model.predict_dataset(split.train)
        validation_predictions = model.predict_dataset(split.validation)
        test_predictions = model.predict_dataset(split.test)

        history_counts = history_index.counts.astype("int64", copy=False)
        per_user_active_cluster_counts = np.diff(cluster_history_index.indptr).astype("int64", copy=False)
        r_star_counts = cluster_artifacts.r_star_counts.astype("int64", copy=False)
        nonempty_cluster_pairs = int((r_star_counts > 0).sum())
        total_cluster_pairs = int(r_star_counts.size)

        metrics_payload = {
            "run_id": run_id,
            "dataset": ratings_summary(ratings_data),
            "split": {
                **asdict(split_config),
                **split_summary(split),
            },
            "model": {
                "name": "cb_svdpp",
                "config": asdict(model.config),
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
                    "clustering "
                    f"user_clusters={metrics_payload['model']['clustering']['n_user_clusters']} "
                    f"item_clusters={metrics_payload['model']['clustering']['n_item_clusters']} "
                    f"r_star_density={metrics_payload['model']['clustering']['r_star_summary']['density']:.6f}"
                ),
                (
                    "rmse "
                    f"train={metrics_payload['metrics']['train_rmse']:.6f} "
                    f"validation={metrics_payload['metrics']['validation_rmse']:.6f} "
                    f"test={metrics_payload['metrics']['test_rmse']:.6f}"
                ),
                f"cluster_induction_wall_clock_seconds={clustering_seconds:.6f}",
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
