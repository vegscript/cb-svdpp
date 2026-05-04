from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import numpy as np

from recsys_lab.clustering import induce_train_only_clusters
from recsys_lab.config.loader import dump_yaml_file, load_yaml_file
from recsys_lab.data.processed import (
    load_processed_dataset_manifest,
    load_ratings_data_from_manifest,
)
from recsys_lab.data.splitters import random_split_with_train_coverage
from recsys_lab.experiments.benchmarking import (
    build_benchmark_measurement,
    summarize_scalar_samples,
)
from recsys_lab.experiments.cb_svdpp import (
    _build_cb_svdpp_config,
    _build_induction_config,
    run_cb_svdpp_experiment,
)
from recsys_lab.experiments.common import (
    SplitConfig,
    build_runtime_metadata,
    git_snapshot,
    resolve_runtime_dtype,
    utc_timestamp,
    write_json,
    write_log,
)
from recsys_lab.experiments.inference_benchmarking import (
    build_repeated_sorted_prefix_query,
    summarize_inference_variant,
    time_inference_variant,
)
from recsys_lab.experiments.runtime import (
    resolve_runtime_threading_config,
    runtime_execution_context,
)
from recsys_lab.models.cb_svdpp import CBSVDppRecommender
from recsys_lab.models.config_schemas import CBSVDppModelProfile
from recsys_lab.models.inference import build_unique_user_context_batch
from recsys_lab.models.registry import (
    validate_model_config_payload,
    validated_model_config_payload_with_training_overrides,
)
from recsys_lab.utils.manifests import load_json_file, validate_manifest_file
from recsys_lab.utils.paths import discover_repo_root, repo_path_string


def _predict_many_unique_context_reference(
    model: CBSVDppRecommender,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    *,
    clip: bool = True,
) -> np.ndarray:
    if (
        model.user_bias is None
        or model.item_bias is None
        or model.item_factors is None
        or model.item_cluster_factors is None
        or model.user_factors is None
    ):
        raise RuntimeError("model parameters are not initialized")

    users, inverse, contexts = build_unique_user_context_batch(
        user_ids=user_ids,
        context_dim=model.user_factors.shape[1],
        build_user_context=model._compute_user_context,
    )
    items = np.asarray(item_ids, dtype=np.int64)
    alpha = float(model.config.alpha)
    one_minus_alpha = 1.0 - alpha
    mixed_item_factors = one_minus_alpha * model.item_factors[items].astype(
        np.float64, copy=False
    ) + alpha * model.item_cluster_factors[model.item_clusters[items]].astype(np.float64, copy=False)
    predictions = (
        model.global_mean
        + model.user_bias[users].astype(np.float64, copy=False)
        + model.item_bias[items].astype(np.float64, copy=False)
        + np.sum(contexts[inverse] * mixed_item_factors, axis=1)
    )
    if clip:
        predictions = np.clip(predictions, model.rating_min, model.rating_max)
    return np.asarray(predictions, dtype=np.float64)


def run_cb_svdpp_inference_cache_benchmark(
    *,
    processed_manifest_path: Path,
    model_config_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    model_seed: int,
    split_seed: int = 1,
    reduced_epochs: int = 1,
    measured_repeats: int = 3,
    repeated_prefix_rows: int = 512,
    repeated_prefix_repeat_factor: int = 32,
    repo_root: Path | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    if measured_repeats <= 0:
        raise ValueError("measured_repeats must be positive")
    if reduced_epochs <= 0:
        raise ValueError("reduced_epochs must be positive")

    root = (repo_root or discover_repo_root()).resolve()
    processed_manifest_path = processed_manifest_path.resolve()
    model_config_path = model_config_path.resolve()
    runtime_config_path = runtime_config_path.resolve()
    device_config_path = device_config_path.resolve()

    processed_manifest = load_processed_dataset_manifest(processed_manifest_path)
    if str(processed_manifest["dataset_short_name"]) != "ml100k":
        raise ValueError("cb_svdpp inference cache benchmark requires dataset_short_name='ml100k'")

    runtime_config_payload = load_yaml_file(runtime_config_path)
    device_config_payload = load_yaml_file(device_config_path)
    model_config_payload = load_yaml_file(model_config_path)
    runtime_dtype = resolve_runtime_dtype(
        runtime_config_payload=runtime_config_payload,
        device_config_payload=device_config_payload,
        model_config_payload=model_config_payload,
    )
    threading_config = resolve_runtime_threading_config(device_config_payload=device_config_payload)
    device_profile_name = str(device_config_payload["device_profile"]["name"])
    git = git_snapshot(root)

    timestamp = utc_timestamp()
    benchmark_id = "_".join(
        [
            timestamp,
            "ml100k",
            "cb_svdpp",
            "inference_cache_compare",
            f"s{split_seed:03d}",
            f"epochs{reduced_epochs}",
            device_profile_name,
        ]
    )
    benchmark_scope = f"development_ml100k_cb_svdpp_inference_cache_compare_s{split_seed:03d}_epochs{reduced_epochs}"
    benchmark_dir = root / "artifacts" / "benchmarks" / benchmark_id
    benchmark_dir.mkdir(parents=True, exist_ok=False)
    config_dir = benchmark_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)

    summary_path = benchmark_dir / "summary.json"
    summary_md_path = benchmark_dir / "summary.md"
    stdout_log_path = benchmark_dir / "stdout.log"
    benchmark_manifest_path = benchmark_dir / "benchmark_manifest.json"
    config_snapshot_path = benchmark_dir / "config_snapshot.yaml"

    command_string = command or (
        "development benchmark cb_svdpp inference cache compare "
        f"--processed-manifest {repo_path_string(processed_manifest_path, repo_root=root)} "
        f"--model-config {repo_path_string(model_config_path, repo_root=root)} "
        f"--runtime-config {repo_path_string(runtime_config_path, repo_root=root)} "
        f"--device-config {repo_path_string(device_config_path, repo_root=root)} "
        f"--model-seed {model_seed} --split-seed {split_seed} --epochs {reduced_epochs}"
    )
    measurement = build_benchmark_measurement(
        time_metric="inference_wall_clock_seconds",
        time_metric_semantics=(
            "Per-variant inference time on a single trained cb_svdpp model. "
            "Each scenario executes separate unmeasured warmups for both variants and then measured repeats."
        ),
        sample_unit="scenario_variant_inference_run",
        measured_sample_count=measured_repeats,
        warmup_policy="separate_unmeasured",
        warmup_sample_count=1,
        notes=[
            "This is a development inference benchmark with reduced training epochs, not a benchmark-final claim.",
            (
                "Reference recomputes unique-user contexts and per-query mixed item factors on each call; "
                "optimized uses lazy persistent caches."
            ),
        ],
    )

    benchmark_manifest: dict[str, Any] = {
        "manifest_version": "v1",
        "kind": "benchmark_manifest",
        "generated_at_utc": timestamp,
        "benchmark_id": benchmark_id,
        "status": "started",
        "benchmark_scope": benchmark_scope,
        "command": command_string,
        "cwd": repo_path_string(root, repo_root=root),
        "git": git,
        "runtime": build_runtime_metadata(
            device_profile_name=device_profile_name,
            runtime_dtype=runtime_dtype,
            device_config_payload=device_config_payload,
        ),
        "measurement": measurement,
        "inputs": {
            "run_ids": [],
            "run_manifest_paths": [],
        },
        "artifacts": {
            "summary": repo_path_string(summary_path, repo_root=root),
            "tables": [repo_path_string(summary_md_path, repo_root=root)],
            "stdout_log": repo_path_string(stdout_log_path, repo_root=root),
        },
        "timing": {
            "started_at_utc": timestamp,
        },
    }

    benchmark_model_config = validated_model_config_payload_with_training_overrides(
        model_config_payload,
        expected_model_name="cb_svdpp",
        training_overrides={"epochs": reduced_epochs},
    )
    benchmark_model_config_path = config_dir / "cb_svdpp_inference_benchmark.yaml"
    dump_yaml_file(benchmark_model_config_path, benchmark_model_config)

    dump_yaml_file(
        config_snapshot_path,
        {
            "benchmark_id": benchmark_id,
            "benchmark_scope": benchmark_scope,
            "processed_manifest": repo_path_string(processed_manifest_path, repo_root=root),
            "model_config": repo_path_string(model_config_path, repo_root=root),
            "runtime_config": repo_path_string(runtime_config_path, repo_root=root),
            "device_config": repo_path_string(device_config_path, repo_root=root),
            "benchmark_model_config": repo_path_string(benchmark_model_config_path, repo_root=root),
            "comparison_model_seed": model_seed,
            "split_seed": split_seed,
            "reduced_epochs": reduced_epochs,
            "measured_repeats": measured_repeats,
            "repeated_prefix_rows": repeated_prefix_rows,
            "repeated_prefix_repeat_factor": repeated_prefix_repeat_factor,
            "loaded_configs": {
                "processed_manifest": processed_manifest,
                "model": model_config_payload,
                "runtime": runtime_config_payload,
                "device": device_config_payload,
                "benchmark_model": benchmark_model_config,
            },
        },
    )
    write_log(
        stdout_log_path,
        [
            f"[{timestamp}] benchmark_id={benchmark_id}",
            f"command={command_string}",
        ],
    )
    write_json(benchmark_manifest_path, benchmark_manifest)

    split_config = SplitConfig(train_ratio=0.8, validation_ratio=0.1, seed=split_seed)
    run_manifest_path: Path | None = None
    try:
        training_payload = run_cb_svdpp_experiment(
            processed_manifest_path=processed_manifest_path,
            model_config_path=benchmark_model_config_path,
            runtime_config_path=runtime_config_path,
            device_config_path=device_config_path,
            split_config=split_config,
            model_seed=model_seed,
            repo_root=root,
            command=(
                "development benchmark cb_svdpp inference cache training provenance "
                f"--split-seed {split_seed} --model-seed {model_seed} --epochs {reduced_epochs}"
            ),
        )
        run_manifest_path = Path(str(training_payload["run_manifest"])).resolve()

        with runtime_execution_context(threading_config=threading_config):
            ratings_data = load_ratings_data_from_manifest(processed_manifest_path)
            split = random_split_with_train_coverage(
                ratings_data,
                train_ratio=split_config.train_ratio,
                validation_ratio=split_config.validation_ratio,
                seed=split_config.seed,
            )
            cb_config = _build_cb_svdpp_config(
                model_config_payload=benchmark_model_config,
                model_seed=model_seed,
                runtime_dtype=runtime_dtype,
            )
            induction_config = _build_induction_config(
                cb_config=cb_config,
                model_seed=model_seed,
            )
            _, validated_model_profile = validate_model_config_payload(
                benchmark_model_config,
                expected_model_name="cb_svdpp",
            )
            if not isinstance(validated_model_profile, CBSVDppModelProfile):
                raise TypeError("cb_svdpp inference benchmark requires a cb_svdpp model profile")
            clustering_config = validated_model_profile.clustering
            cluster_artifacts = induce_train_only_clusters(
                split.train,
                induction_config=induction_config,
                n_user_clusters=clustering_config.n_user_clusters,
                n_item_clusters=clustering_config.n_item_clusters,
                algorithm=clustering_config.algorithm,
                kmeans_n_init=clustering_config.kmeans_n_init,
            )
            model = CBSVDppRecommender(
                cb_config,
                user_clusters=cluster_artifacts.user_clusters,
                item_clusters=cluster_artifacts.item_clusters,
                n_user_clusters=cluster_artifacts.r_star_counts.shape[0],
                n_item_clusters=cluster_artifacts.r_star_counts.shape[1],
            )
            model.fit(split.train)

            test_users = split.test.user_ids.astype(np.int64, copy=False)
            test_items = split.test.item_ids.astype(np.int64, copy=False)
            repeated_users, repeated_items = build_repeated_sorted_prefix_query(
                user_ids=split.test.user_ids,
                item_ids=split.test.item_ids,
                prefix_rows=repeated_prefix_rows,
                repeat_factor=repeated_prefix_repeat_factor,
            )
            scenarios = [
                ("official_test", test_users, test_items),
                ("repeated_sorted_prefix", repeated_users, repeated_items),
            ]

            summary_payload: dict[str, Any] = {
                "benchmark_id": benchmark_id,
                "benchmark_scope": benchmark_scope,
                "dataset": "ml100k",
                "split_family": "benchmark_random_v1",
                "split_seed": split_seed,
                "model": "cb_svdpp",
                "measurement": measurement,
                "reduced_epochs": reduced_epochs,
                "comparison_model_seed": model_seed,
                "scenarios": {},
            }

            def reference_predict_many(
                user_ids: np.ndarray,
                item_ids: np.ndarray,
                clip: bool = False,
            ) -> np.ndarray:
                return _predict_many_unique_context_reference(
                    model,
                    user_ids,
                    item_ids,
                    clip=clip,
                )

            for scenario_name, query_users, query_items in scenarios:
                optimized_user_context_cache = model._ensure_user_context_cache()
                optimized_mixed_item_factors_cache = model._ensure_mixed_item_factors_cache()
                reference_timings, reference_predictions = time_inference_variant(
                    predict_many_fn=reference_predict_many,
                    user_ids=query_users,
                    item_ids=query_items,
                    repeats=measured_repeats,
                )
                optimized_timings, optimized_predictions = time_inference_variant(
                    predict_many_fn=model.predict_many,
                    user_ids=query_users,
                    item_ids=query_items,
                    repeats=measured_repeats,
                )
                query_rows = int(query_users.shape[0])
                unique_users = int(np.unique(query_users).shape[0])
                max_abs_delta = float(np.max(np.abs(reference_predictions - optimized_predictions)))
                summary_payload["scenarios"][scenario_name] = {
                    "query_rows": query_rows,
                    "unique_users": unique_users,
                    "materialization": {
                        "reference_context_rows_materialized_per_call": unique_users,
                        "reference_mixed_item_rows_materialized_per_call": query_rows,
                        "optimized_persistent_user_context_rows": int(optimized_user_context_cache.shape[0]),
                        "optimized_persistent_mixed_item_rows": int(optimized_mixed_item_factors_cache.shape[0]),
                        "optimized_context_rows_materialized_per_call": 0,
                        "optimized_mixed_item_rows_materialized_per_call": 0,
                    },
                    "variants": {
                        "reference_unique_context": summarize_inference_variant(query_rows, reference_timings),
                        "optimized_cached_context": summarize_inference_variant(query_rows, optimized_timings),
                    },
                    "prediction_delta": {
                        "max_abs_reference_vs_optimized": max_abs_delta,
                    },
                    "comparison": {
                        "inference_wall_clock_seconds_speedup_reference_over_optimized": float(
                            summarize_scalar_samples(reference_timings)["mean"]
                            / summarize_scalar_samples(optimized_timings)["mean"]
                        ),
                        "ratings_per_second_inference_speedup_optimized_over_reference": float(
                            summarize_scalar_samples([query_rows / value for value in optimized_timings])["mean"]
                            / summarize_scalar_samples([query_rows / value for value in reference_timings])["mean"]
                        ),
                    },
                }

        write_json(summary_path, summary_payload)
        markdown_lines = [
            "# CB-SVDpp Inference Cache Compare",
            "",
            f"- benchmark_id: `{benchmark_id}`",
            f"- benchmark_scope: `{benchmark_scope}`",
            "- dataset: `ml100k`",
            f"- split_seed: `{split_seed}`",
            f"- reduced_epochs: `{reduced_epochs}`",
            f"- measured_repeats: `{measured_repeats}`",
            f"- warmup_policy: `{measurement['warmup_policy']}`",
            "",
        ]
        for scenario_name, scenario_payload in summary_payload["scenarios"].items():
            materialization = scenario_payload["materialization"]
            prediction_delta = scenario_payload["prediction_delta"]
            comparison = scenario_payload["comparison"]
            reference = scenario_payload["variants"]["reference_unique_context"]["aggregate"]
            optimized = scenario_payload["variants"]["optimized_cached_context"]["aggregate"]
            reference_time = reference["inference_wall_clock_seconds"]
            reference_throughput = reference["ratings_per_second_inference"]
            optimized_time = optimized["inference_wall_clock_seconds"]
            optimized_throughput = optimized["ratings_per_second_inference"]
            markdown_lines.extend(
                [
                    f"## {scenario_name}",
                    "",
                    f"- query_rows: `{scenario_payload['query_rows']}`",
                    f"- unique_users: `{scenario_payload['unique_users']}`",
                    (
                        "- reference_context_rows_materialized_per_call: "
                        f"`{materialization['reference_context_rows_materialized_per_call']}`"
                    ),
                    (
                        "- reference_mixed_item_rows_materialized_per_call: "
                        f"`{materialization['reference_mixed_item_rows_materialized_per_call']}`"
                    ),
                    (
                        "- optimized_persistent_user_context_rows: "
                        f"`{materialization['optimized_persistent_user_context_rows']}`"
                    ),
                    (
                        "- optimized_persistent_mixed_item_rows: "
                        f"`{materialization['optimized_persistent_mixed_item_rows']}`"
                    ),
                    (
                        "- optimized_context_rows_materialized_per_call: "
                        f"`{materialization['optimized_context_rows_materialized_per_call']}`"
                    ),
                    (
                        "- optimized_mixed_item_rows_materialized_per_call: "
                        f"`{materialization['optimized_mixed_item_rows_materialized_per_call']}`"
                    ),
                    (f"- max_abs_prediction_delta: `{prediction_delta['max_abs_reference_vs_optimized']:.12f}`"),
                    "",
                    "| Variant | Time Mean (s) | Time CV | Ratings/s Mean |",
                    "| --- | ---: | ---: | ---: |",
                    (
                        f"| `reference_unique_context` | {reference_time['mean']:.6f} | "
                        f"{reference_time['coefficient_of_variation']:.6f} | "
                        f"{reference_throughput['mean']:.6f} |"
                    ),
                    (
                        f"| `optimized_cached_context` | {optimized_time['mean']:.6f} | "
                        f"{optimized_time['coefficient_of_variation']:.6f} | "
                        f"{optimized_throughput['mean']:.6f} |"
                    ),
                    "",
                    (
                        "- time_speedup_reference_over_optimized: "
                        f"`{comparison['inference_wall_clock_seconds_speedup_reference_over_optimized']:.6f}`"
                    ),
                    (
                        "- throughput_speedup_optimized_over_reference: "
                        f"`{comparison['ratings_per_second_inference_speedup_optimized_over_reference']:.6f}`"
                    ),
                    "",
                ]
            )
        summary_md_path.write_text("\n".join(markdown_lines).strip() + "\n", encoding="utf-8", newline="\n")

        finished_at = utc_timestamp()
        write_log(
            stdout_log_path,
            [
                f"[{timestamp}] benchmark_id={benchmark_id}",
                f"command={command_string}",
                f"[{finished_at}] status=completed",
            ],
        )
        completed_manifest = {
            **benchmark_manifest,
            "status": "completed",
            "generated_at_utc": finished_at,
            "inputs": {
                "run_ids": [str(load_json_file(run_manifest_path)["run_id"])] if run_manifest_path is not None else [],
                "run_manifest_paths": (
                    [repo_path_string(run_manifest_path, repo_root=root)] if run_manifest_path is not None else []
                ),
            },
            "timing": {
                **benchmark_manifest["timing"],
                "finished_at_utc": finished_at,
            },
        }
        write_json(benchmark_manifest_path, completed_manifest)
        validate_manifest_file(benchmark_manifest_path, repo_root=root)
        return {
            "benchmark_id": benchmark_id,
            "benchmark_dir": str(benchmark_dir),
            "benchmark_manifest": str(benchmark_manifest_path),
            "summary_path": str(summary_path),
        }
    except Exception:
        finished_at = utc_timestamp()
        write_log(
            stdout_log_path,
            [
                f"[{timestamp}] benchmark_id={benchmark_id}",
                f"command={command_string}",
                f"[{finished_at}] status=failed",
                traceback.format_exc().strip(),
            ],
        )
        failed_manifest = {
            **benchmark_manifest,
            "status": "failed",
            "generated_at_utc": finished_at,
            "inputs": {
                "run_ids": [str(load_json_file(run_manifest_path)["run_id"])] if run_manifest_path is not None else [],
                "run_manifest_paths": (
                    [repo_path_string(run_manifest_path, repo_root=root)] if run_manifest_path is not None else []
                ),
            },
            "timing": {
                **benchmark_manifest["timing"],
                "finished_at_utc": finished_at,
            },
        }
        write_json(benchmark_manifest_path, failed_manifest)
        validate_manifest_file(benchmark_manifest_path, repo_root=root)
        raise
