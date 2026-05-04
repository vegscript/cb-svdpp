from __future__ import annotations

import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, cast

import numpy as np

from recsys_lab.clustering import load_or_build_cluster_artifacts, load_or_build_user_cluster_history_index
from recsys_lab.config.loader import dump_yaml_file, load_yaml_file
from recsys_lab.data.processed import load_processed_dataset_manifest, load_ratings_data_from_manifest
from recsys_lab.data.splitters import (
    RatingsSplit,
    official_ml100k_inner_validation_split,
    official_ml100k_paper_faithful_split,
    random_split_with_train_coverage,
)
from recsys_lab.data.training_index_cache import (
    UserExplicitFeedbackIndexCacheResult,
    UserHistoryIndexCacheResult,
    load_or_build_user_explicit_feedback_index,
    load_or_build_user_history_index,
)
from recsys_lab.experiments.common import (
    SplitConfig,
    build_base_run_manifest,
    build_run_id,
    git_snapshot,
    paper_faithful_ml100k_inner_split_id,
    paper_faithful_ml100k_split_id,
    ratings_summary,
    reserve_timestamped_artifact_dir,
    split_id,
    split_summary,
    utc_timestamp,
    write_json,
    write_log,
)
from recsys_lab.experiments.performance import PeakMemoryMonitor, StageProfiler, build_system_metrics
from recsys_lab.experiments.runtime import resolve_runtime_threading_config, runtime_execution_context
from recsys_lab.experiments.split_cache import SplitCacheResult, load_or_build_split_cache, resolve_split_cache_policy
from recsys_lab.metrics import rating_error_metrics
from recsys_lab.models.cb_asvdpp import CBASVDppRecommender
from recsys_lab.models.cb_svdpp import CBSVDppRecommender
from recsys_lab.models.config_schemas import ClusteringSchema, ModelProfileSchema
from recsys_lab.models.registry import (
    FitArtifacts,
    ModelAdapter,
    build_cb_semantics,
    model_requirements_payload,
    pydantic_profile_payload,
    validate_model_config_payload,
)
from recsys_lab.utils.manifests import validate_manifest_file
from recsys_lab.utils.paths import discover_repo_root, repo_path_string


@dataclass(frozen=True, slots=True)
class ExperimentServices:
    git_snapshot_fn: Callable[[Path], dict[str, Any]]
    paper_faithful_split_fn: Callable[..., RatingsSplit]
    inner_validation_split_fn: Callable[..., RatingsSplit]


DEFAULT_EXPERIMENT_SERVICES = ExperimentServices(
    git_snapshot_fn=git_snapshot,
    paper_faithful_split_fn=official_ml100k_paper_faithful_split,
    inner_validation_split_fn=official_ml100k_inner_validation_split,
)


def build_experiment_services(
    *,
    git_snapshot_fn: Callable[[Path], dict[str, Any]] | None = None,
    paper_faithful_split_fn: Callable[..., RatingsSplit] | None = None,
    inner_validation_split_fn: Callable[..., RatingsSplit] | None = None,
) -> ExperimentServices:
    return ExperimentServices(
        git_snapshot_fn=git_snapshot_fn or DEFAULT_EXPERIMENT_SERVICES.git_snapshot_fn,
        paper_faithful_split_fn=paper_faithful_split_fn or DEFAULT_EXPERIMENT_SERVICES.paper_faithful_split_fn,
        inner_validation_split_fn=inner_validation_split_fn or DEFAULT_EXPERIMENT_SERVICES.inner_validation_split_fn,
    )


def run_unified_experiment(
    *,
    processed_manifest_path: Path,
    model_config_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    split_config: SplitConfig,
    model_seed: int,
    repo_root: Path | None = None,
    command: str | None = None,
    model_name: str | None = None,
    split_family: str | None = None,
    inner_validation_seed: int | None = None,
    evaluate_test: bool = True,
    use_split_cache: bool | None = None,
    reuse_precomputed_indices: bool = True,
    use_training_index_cache: bool = False,
    use_cluster_artifact_cache: bool = False,
    services: ExperimentServices | None = None,
) -> dict[str, Any]:
    root = (repo_root or discover_repo_root()).resolve()
    experiment_services = services or DEFAULT_EXPERIMENT_SERVICES

    processed_manifest_path = processed_manifest_path.resolve()
    model_config_path = model_config_path.resolve()
    runtime_config_path = runtime_config_path.resolve()
    device_config_path = device_config_path.resolve()

    processed_manifest = load_processed_dataset_manifest(processed_manifest_path)
    dataset_short_name = str(processed_manifest["dataset_short_name"])
    requested_split_family = split_family or str(processed_manifest["split_family"])

    runtime_config_payload = load_yaml_file(runtime_config_path)
    device_config_payload = load_yaml_file(device_config_path)
    raw_model_config_payload = load_yaml_file(model_config_path)
    adapter, model_profile = validate_model_config_payload(
        raw_model_config_payload,
        expected_model_name=model_name,
    )
    runtime_dtype = adapter.runtime_dtype(model_profile)
    threading_config = resolve_runtime_threading_config(device_config_payload=device_config_payload)
    split_cache_policy = resolve_split_cache_policy(
        split_family=requested_split_family,
        use_split_cache=use_split_cache,
    )
    _validate_cache_options(
        adapter=adapter,
        use_training_index_cache=use_training_index_cache,
        use_cluster_artifact_cache=use_cluster_artifact_cache,
    )

    run_context_slug = _run_context_slug(
        requested_split_family=requested_split_family,
        split_config=split_config,
        inner_validation_seed=inner_validation_seed,
    )
    device_profile_name = str(device_config_payload["device_profile"]["name"])
    timestamp, run_id, run_dir = reserve_timestamped_artifact_dir(
        artifacts_root=root / "artifacts" / "runs",
        id_from_timestamp=lambda reserved_timestamp: build_run_id(
            timestamp=reserved_timestamp,
            dataset_short_name=dataset_short_name,
            model_name=adapter.name,
            device_profile_name=device_profile_name,
            model_seed=model_seed,
            split_id_value=run_context_slug,
        ),
    )

    config_snapshot_path = run_dir / "config_snapshot.yaml"
    metrics_path = run_dir / "metrics.json"
    stdout_log_path = run_dir / "stdout.log"
    run_manifest_path = run_dir / "run_manifest.json"
    git = experiment_services.git_snapshot_fn(root)
    command_string = command or _default_command_string(
        adapter=adapter,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        root=root,
        requested_split_family=requested_split_family,
        split_config=split_config,
        inner_validation_seed=inner_validation_seed,
        evaluate_test=evaluate_test,
        use_split_cache=use_split_cache,
        use_training_index_cache=use_training_index_cache,
        use_cluster_artifact_cache=use_cluster_artifact_cache,
        model_seed=model_seed,
    )
    cb_semantics = _cb_semantics_for_profile(adapter=adapter, model_profile=model_profile)

    with runtime_execution_context(threading_config=threading_config):
        stage_profiler = StageProfiler()
        base_manifest = build_base_run_manifest(
            timestamp=timestamp,
            run_id=run_id,
            command=command_string,
            repo_root=root,
            git=git,
            processed_manifest=processed_manifest,
            processed_manifest_path=processed_manifest_path,
            model_name=adapter.name,
            model_scope=str(model_profile.model.scope),
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
            split_id_value=run_context_slug,
        )
        base_manifest = {
            **base_manifest,
            "profiling": stage_profiler.to_payload(),
        }
        if cb_semantics is not None:
            base_manifest["cb_semantics"] = cb_semantics

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
                "split_family": requested_split_family,
                "inner_validation_seed": inner_validation_seed,
                "model_seed": model_seed,
                "model_requirements": model_requirements_payload(adapter),
                "use_split_cache": split_cache_policy.effective_use_cache,
                "use_split_cache_policy_requested": split_cache_policy.requested_policy,
                "use_split_cache_decision_reason": split_cache_policy.decision_reason,
                "reuse_precomputed_indices": reuse_precomputed_indices,
                "use_training_index_cache": use_training_index_cache,
                "use_cluster_artifact_cache": use_cluster_artifact_cache,
                "loaded_configs": {
                    "processed_manifest": processed_manifest,
                    "model": raw_model_config_payload,
                    "validated_model": pydantic_profile_payload(model_profile),
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
            with stage_profiler.stage("data_load", metadata={"processed_manifest": str(processed_manifest_path)}):
                ratings_data = load_ratings_data_from_manifest(processed_manifest_path)

            split_id_for_cache = str(base_manifest["dataset"]["split_id"])
            with stage_profiler.stage(
                "split_resolution",
                metadata={
                    "split_family": requested_split_family,
                    "split_id": split_id_for_cache,
                    "cache_enabled": split_cache_policy.effective_use_cache,
                },
            ) as split_stage:
                split_result = load_or_build_split_cache(
                    data=ratings_data,
                    dataset_short_name=dataset_short_name,
                    split_family=requested_split_family,
                    split_id=split_id_for_cache,
                    processed_manifest_path=processed_manifest_path,
                    repo_root=root,
                    runtime_config_payload=runtime_config_payload,
                    build_split=lambda: _build_split(
                        ratings_data=ratings_data,
                        processed_manifest_path=processed_manifest_path,
                        requested_split_family=requested_split_family,
                        split_config=split_config,
                        inner_validation_seed=inner_validation_seed,
                        services=experiment_services,
                    ),
                    use_cache=split_cache_policy.effective_use_cache,
                )
                split_stage["cache_status"] = split_result.metadata.cache_status
            split = split_result.split

            with stage_profiler.stage("config_build"):
                model_config = adapter.build_model_config(
                    model_profile,
                    model_seed=model_seed,
                    runtime_dtype=runtime_dtype,
                )
                induction_config = adapter.build_induction_config(model_config, model_seed=model_seed)

            clustering_seconds = 0.0
            with PeakMemoryMonitor() as memory_monitor:
                artifact_resolution = _resolve_fit_artifacts(
                    adapter=adapter,
                    split=split,
                    model_profile=model_profile,
                    model_config=model_config,
                    induction_config=induction_config,
                    dataset_short_name=dataset_short_name,
                    requested_split_family=requested_split_family,
                    split_id_for_cache=split_id_for_cache,
                    processed_manifest_path=processed_manifest_path,
                    root=root,
                    runtime_config_payload=runtime_config_payload,
                    runtime_dtype=runtime_dtype,
                    use_training_index_cache=use_training_index_cache,
                    use_cluster_artifact_cache=use_cluster_artifact_cache,
                    stage_profiler=stage_profiler,
                )
                fit_artifacts = artifact_resolution["artifacts"]
                clustering_seconds = float(artifact_resolution["cluster_induction_wall_clock_seconds"])

                with stage_profiler.stage("model_initialization"):
                    model = adapter.instantiate(model_config, artifacts=fit_artifacts)

                training_started = perf_counter()
                with stage_profiler.stage(
                    "main_training",
                    metadata={"epochs": _model_epochs(model_config), "train_rows": len(split.train)},
                ):
                    adapter.fit(
                        model,
                        split.train,
                        artifacts=fit_artifacts,
                        reuse_precomputed_indices=reuse_precomputed_indices,
                    )
                training_seconds = perf_counter() - training_started

                inference_started = perf_counter()
                with stage_profiler.stage("inference_train", metadata={"rows": len(split.train)}):
                    train_predictions = model.predict_dataset(split.train)
                validation_predictions = None
                if split.validation is not None:
                    with stage_profiler.stage("inference_validation", metadata={"rows": len(split.validation)}):
                        validation_predictions = model.predict_dataset(split.validation)
                test_predictions = None
                if evaluate_test:
                    with stage_profiler.stage("inference_test", metadata={"rows": len(split.test)}):
                        test_predictions = model.predict_dataset(split.test)
                inference_seconds = perf_counter() - inference_started

            inference_rows = len(split.train)
            if split.validation is not None:
                inference_rows += len(split.validation)
            if evaluate_test:
                inference_rows += len(split.test)

            system_metrics = build_system_metrics(
                train_rows=len(split.train),
                epochs=_model_epochs(model_config),
                training_wall_clock_seconds=training_seconds,
                inference_rows=inference_rows,
                inference_wall_clock_seconds=inference_seconds,
                peak_memory_bytes=memory_monitor.peak_rss_bytes,
                baseline_memory_bytes=memory_monitor.start_rss_bytes,
                model=model,
                epoch_durations_seconds=getattr(model, "epoch_durations_seconds", []),
                train_time_total_seconds=(
                    clustering_seconds + training_seconds
                    if adapter.requirements.needs_cluster_artifacts
                    else training_seconds
                ),
                extra_fields=(
                    {
                        "cluster_induction_wall_clock_seconds": float(clustering_seconds),
                        "main_training_wall_clock_seconds": float(training_seconds),
                    }
                    if adapter.requirements.needs_cluster_artifacts
                    else None
                ),
            )

            profiling_payload = stage_profiler.to_payload()
            caches_payload = _build_caches_payload(
                split_result=split_result,
                artifact_resolution=artifact_resolution,
                repo_root=root,
            )
            model_payload = _build_model_payload(
                adapter=adapter,
                model=model,
                model_config=model_config,
                model_profile=model_profile,
                fit_artifacts=fit_artifacts,
                split_result=split_result,
                artifact_resolution=artifact_resolution,
                split_id_for_cache=split_id_for_cache,
                split_cache_policy=split_cache_policy,
                use_training_index_cache=use_training_index_cache,
                use_cluster_artifact_cache=use_cluster_artifact_cache,
                reuse_precomputed_indices=reuse_precomputed_indices,
                repo_root=root,
                cb_semantics=cb_semantics,
            )

            metrics_payload: dict[str, Any] = {
                "run_id": run_id,
                "profiling": profiling_payload,
                "caches": caches_payload,
                "dataset": ratings_summary(ratings_data),
                "split": {
                    "family": requested_split_family,
                    **asdict(split_config),
                    "inner_validation_seed": inner_validation_seed,
                    "test_metrics_available": evaluate_test,
                    **split_summary(split),
                },
                "model": model_payload,
                "timing": {
                    "training_wall_clock_seconds": training_seconds,
                    "inference_wall_clock_seconds": inference_seconds,
                    **(
                        {"cluster_induction_wall_clock_seconds": clustering_seconds}
                        if adapter.requirements.needs_cluster_artifacts
                        else {}
                    ),
                },
                "system_metrics": system_metrics,
                "metrics": _build_rating_metrics_payload(
                    train_ratings=split.train.ratings,
                    train_predictions=train_predictions,
                    validation_ratings=None if split.validation is None else split.validation.ratings,
                    validation_predictions=validation_predictions,
                    test_ratings=split.test.ratings if evaluate_test else None,
                    test_predictions=test_predictions,
                    rating_min=ratings_data.rating_min,
                    rating_max=ratings_data.rating_max,
                ),
            }
            if cb_semantics is not None:
                metrics_payload["cb_semantics"] = cb_semantics
                metrics_payload["cb_diagnostics"] = _build_cb_diagnostics(
                    model=model,
                    train_data=split.train,
                    fit_artifacts=fit_artifacts,
                    cb_semantics=cb_semantics,
                )
            write_json(metrics_path, metrics_payload)

            finished_at = utc_timestamp()
            validation_rmse = metrics_payload["metrics"]["validation_rmse"]
            test_rmse = metrics_payload["metrics"]["test_rmse"]
            validation_rmse_display = "NA" if validation_rmse is None else format(validation_rmse, ".6f")
            test_rmse_display = "NA" if test_rmse is None else format(test_rmse, ".6f")
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
                        f"validation={validation_rmse_display} "
                        f"test={test_rmse_display}"
                    ),
                    f"training_wall_clock_seconds={training_seconds:.6f}",
                    (
                        "system_metrics "
                        f"ratings_per_second_train={system_metrics['ratings_per_second_train']:.6f} "
                        f"ratings_per_second_inference={system_metrics['ratings_per_second_inference']:.6f} "
                        f"peak_memory_mb={system_metrics['peak_memory_mb']:.6f} "
                        f"model_size_mb={system_metrics['model_size_mb']:.6f}"
                    ),
                    (
                        "stage_profile "
                        f"stage_count={profiling_payload['stage_count']} "
                        "total_profiled_wall_clock_seconds="
                        f"{profiling_payload['total_profiled_wall_clock_seconds']:.6f}"
                    ),
                    f"[{finished_at}] status=completed",
                ],
            )

            completed_manifest = {
                **base_manifest,
                "status": "completed",
                "generated_at_utc": finished_at,
                "profiling": profiling_payload,
                "caches": caches_payload,
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
            profiling_payload = stage_profiler.to_payload()
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
                "profiling": profiling_payload,
                "timing": {
                    **base_manifest["timing"],
                    "finished_at_utc": finished_at,
                },
            }
            write_json(run_manifest_path, failed_manifest)
            validate_manifest_file(run_manifest_path, repo_root=root)
            raise


def _validate_cache_options(
    *,
    adapter: type[ModelAdapter],
    use_training_index_cache: bool,
    use_cluster_artifact_cache: bool,
) -> None:
    requirements = adapter.requirements
    if use_training_index_cache and not (
        requirements.needs_implicit_history or requirements.needs_explicit_feedback
    ):
        raise ValueError(f"training-index cache is not applicable to model '{adapter.name}'")
    if use_cluster_artifact_cache and not requirements.needs_cluster_artifacts:
        raise ValueError(f"cluster-artifact cache is not applicable to model '{adapter.name}'")


def _run_context_slug(
    *,
    requested_split_family: str,
    split_config: SplitConfig,
    inner_validation_seed: int | None,
) -> str:
    if requested_split_family == "paper_faithful_ml100k_v1":
        return paper_faithful_ml100k_split_id(split_config.seed)
    if requested_split_family == "paper_faithful_ml100k_inner_v1":
        if inner_validation_seed is None:
            raise ValueError("inner_validation_seed is required for paper_faithful_ml100k_inner_v1")
        return paper_faithful_ml100k_inner_split_id(
            fold_index=split_config.seed,
            validation_ratio=split_config.validation_ratio,
            inner_seed=inner_validation_seed,
        )
    return split_id(requested_split_family, split_config)


def _build_split(
    *,
    ratings_data,
    processed_manifest_path: Path,
    requested_split_family: str,
    split_config: SplitConfig,
    inner_validation_seed: int | None,
    services: ExperimentServices,
) -> RatingsSplit:
    if requested_split_family == "benchmark_random_v1":
        return random_split_with_train_coverage(
            ratings_data,
            train_ratio=split_config.train_ratio,
            validation_ratio=split_config.validation_ratio,
            seed=split_config.seed,
        )
    if requested_split_family == "paper_faithful_ml100k_v1":
        return services.paper_faithful_split_fn(
            ratings_data,
            processed_manifest_path=processed_manifest_path,
            fold_index=split_config.seed,
        )
    if requested_split_family == "paper_faithful_ml100k_inner_v1":
        if inner_validation_seed is None:
            raise ValueError("inner_validation_seed is required for paper_faithful_ml100k_inner_v1")
        return services.inner_validation_split_fn(
            ratings_data,
            processed_manifest_path=processed_manifest_path,
            fold_index=split_config.seed,
            validation_ratio=split_config.validation_ratio,
            inner_seed=inner_validation_seed,
        )
    raise ValueError(f"unsupported split family: {requested_split_family}")


def _prefixed_rating_metrics(
    prefix: str,
    ratings: np.ndarray,
    predictions: np.ndarray,
    *,
    rating_min: float,
    rating_max: float,
) -> dict[str, float]:
    return {
        f"{prefix}_{metric_name}": metric_value
        for metric_name, metric_value in rating_error_metrics(
            ratings,
            predictions,
            rating_min=rating_min,
            rating_max=rating_max,
        ).items()
    }


def _build_rating_metrics_payload(
    *,
    train_ratings: np.ndarray,
    train_predictions: np.ndarray,
    validation_ratings: np.ndarray | None,
    validation_predictions: np.ndarray | None,
    test_ratings: np.ndarray | None,
    test_predictions: np.ndarray | None,
    rating_min: float,
    rating_max: float,
) -> dict[str, Any]:
    train_metrics = rating_error_metrics(
        train_ratings,
        train_predictions,
        rating_min=rating_min,
        rating_max=rating_max,
    )
    payload: dict[str, Any] = {
        "train": train_metrics,
        **_prefixed_rating_metrics(
            "train",
            train_ratings,
            train_predictions,
            rating_min=rating_min,
            rating_max=rating_max,
        ),
    }

    if validation_ratings is None or validation_predictions is None:
        payload["validation"] = None
        payload["validation_rmse"] = None
    else:
        validation_metrics = rating_error_metrics(
            validation_ratings,
            validation_predictions,
            rating_min=rating_min,
            rating_max=rating_max,
        )
        payload["validation"] = validation_metrics
        payload.update(
            _prefixed_rating_metrics(
                "validation",
                validation_ratings,
                validation_predictions,
                rating_min=rating_min,
                rating_max=rating_max,
            )
        )

    if test_ratings is None or test_predictions is None:
        payload["test"] = None
        payload["test_rmse"] = None
    else:
        test_metrics = rating_error_metrics(
            test_ratings,
            test_predictions,
            rating_min=rating_min,
            rating_max=rating_max,
        )
        payload["test"] = test_metrics
        payload.update(
            _prefixed_rating_metrics(
                "test",
                test_ratings,
                test_predictions,
                rating_min=rating_min,
                rating_max=rating_max,
            )
        )
    return payload


def _default_command_string(
    *,
    adapter: type[ModelAdapter],
    processed_manifest_path: Path,
    model_config_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    root: Path,
    requested_split_family: str,
    split_config: SplitConfig,
    inner_validation_seed: int | None,
    evaluate_test: bool,
    use_split_cache: bool | None,
    use_training_index_cache: bool,
    use_cluster_artifact_cache: bool,
    model_seed: int,
) -> str:
    split_cache_fragment = (
        "" if use_split_cache is None else f"--split-cache {'enable' if use_split_cache else 'disable'} "
    )
    return (
        "recsys-lab train "
        f"--model {adapter.name} "
        f"--processed-manifest {repo_path_string(processed_manifest_path, repo_root=root)} "
        f"--model-config {repo_path_string(model_config_path, repo_root=root)} "
        f"--runtime-config {repo_path_string(runtime_config_path, repo_root=root)} "
        f"--device-config {repo_path_string(device_config_path, repo_root=root)} "
        f"--split-family {requested_split_family} "
        f"--train-ratio {split_config.train_ratio} "
        f"--validation-ratio {split_config.validation_ratio} "
        f"--split-seed {split_config.seed} "
        f"{'' if inner_validation_seed is None else f'--inner-validation-seed {inner_validation_seed} '}"
        f"{'' if evaluate_test else '--skip-test-eval '}"
        f"{split_cache_fragment}"
        f"{'--training-index-cache ' if use_training_index_cache else '--disable-training-index-cache '}"
        f"{'--cluster-artifact-cache ' if use_cluster_artifact_cache else '--disable-cluster-artifact-cache '}"
        f"--model-seed {model_seed}"
    )


def _cb_semantics_for_profile(
    *,
    adapter: type[ModelAdapter],
    model_profile: ModelProfileSchema,
) -> dict[str, Any] | None:
    alpha = adapter.cb_alpha(model_profile)
    if alpha is None:
        return None
    return build_cb_semantics(alpha)


def _clustering_config(model_profile: ModelProfileSchema) -> ClusteringSchema:
    clustering = getattr(model_profile, "clustering", None)
    if not isinstance(clustering, ClusteringSchema):
        raise ValueError("CB model profile requires a validated clustering block")
    return clustering


def _resolve_fit_artifacts(
    *,
    adapter: type[ModelAdapter],
    split: RatingsSplit,
    model_profile: ModelProfileSchema,
    model_config: object,
    induction_config: object | None,
    dataset_short_name: str,
    requested_split_family: str,
    split_id_for_cache: str,
    processed_manifest_path: Path,
    root: Path,
    runtime_config_payload: dict[str, Any],
    runtime_dtype: str,
    use_training_index_cache: bool,
    use_cluster_artifact_cache: bool,
    stage_profiler: StageProfiler,
) -> dict[str, Any]:
    requirements = adapter.requirements
    history_result: UserHistoryIndexCacheResult | None = None
    explicit_result: UserExplicitFeedbackIndexCacheResult | None = None
    cluster_artifact_result = None
    cluster_history_result = None
    clustering_seconds = 0.0

    if requirements.needs_cluster_artifacts:
        if induction_config is None:
            raise ValueError(f"{adapter.name} requires an induction config for cluster artifacts")
        clustering_config = _clustering_config(model_profile)
        clustering_started = perf_counter()
        with stage_profiler.stage(
            "cluster_induction",
            metadata={
                "algorithm": clustering_config.algorithm,
                "n_user_clusters": clustering_config.n_user_clusters,
                "n_item_clusters": clustering_config.n_item_clusters,
                "kmeans_n_init": clustering_config.kmeans_n_init,
                "cache_enabled": use_cluster_artifact_cache,
            },
        ) as cluster_stage:
            cluster_artifact_result = load_or_build_cluster_artifacts(
                data=split.train,
                induction_config=induction_config,  # type: ignore[arg-type]
                n_user_clusters=clustering_config.n_user_clusters,
                n_item_clusters=clustering_config.n_item_clusters,
                algorithm=clustering_config.algorithm,
                kmeans_n_init=clustering_config.kmeans_n_init,
                dataset_short_name=dataset_short_name,
                split_family=requested_split_family,
                split_id=split_id_for_cache,
                processed_manifest_path=processed_manifest_path,
                repo_root=root,
                runtime_config_payload=runtime_config_payload,
                use_cache=use_cluster_artifact_cache,
            )
            cluster_stage["cache_status"] = cluster_artifact_result.metadata.cache_status
            cluster_stage["cache_key"] = cluster_artifact_result.metadata.cache_key
        clustering_seconds = perf_counter() - clustering_started

    if requirements.needs_implicit_history:
        with stage_profiler.stage(
            "user_history_index_resolution",
            metadata={"cache_enabled": use_training_index_cache, "split_id": split_id_for_cache},
        ) as index_stage:
            history_result = load_or_build_user_history_index(
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
            index_stage["cache_status"] = history_result.metadata.cache_status

    if requirements.needs_explicit_feedback:
        with stage_profiler.stage(
            "explicit_feedback_index_resolution",
            metadata={"cache_enabled": use_training_index_cache, "split_id": split_id_for_cache},
        ) as index_stage:
            explicit_result = load_or_build_user_explicit_feedback_index(
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
            index_stage["cache_status"] = explicit_result.metadata.cache_status

    if requirements.needs_user_cluster_history:
        if history_result is None or cluster_artifact_result is None:
            raise ValueError("user-cluster history requires user history and cluster artifacts")
        n_item_clusters = int(cluster_artifact_result.artifacts.r_star_counts.shape[1])
        with stage_profiler.stage(
            "user_cluster_history_build",
            metadata={"n_item_clusters": n_item_clusters, "cache_enabled": use_cluster_artifact_cache},
        ) as cluster_history_stage:
            cluster_history_result = load_or_build_user_cluster_history_index(
                history_index=history_result.index,
                item_clusters=cluster_artifact_result.artifacts.item_clusters,
                n_clusters=n_item_clusters,
                dataset_short_name=dataset_short_name,
                split_family=requested_split_family,
                split_id=split_id_for_cache,
                processed_manifest_path=processed_manifest_path,
                repo_root=root,
                runtime_config_payload=runtime_config_payload,
                train_fingerprint=history_result.metadata.train_fingerprint,
                cluster_cache_key=cluster_artifact_result.metadata.cache_key,
                cluster_cache_fingerprint_sha256=cluster_artifact_result.metadata.cache_fingerprint_sha256,
                use_cache=use_cluster_artifact_cache,
            )
            cluster_history_stage["cache_status"] = cluster_history_result.metadata.cache_status
            cluster_history_stage["cache_key"] = cluster_history_result.metadata.cache_key

    return {
        "artifacts": FitArtifacts(
            user_history_index=None if history_result is None else history_result.index,
            explicit_feedback_index=None if explicit_result is None else explicit_result.index,
            user_cluster_history_index=None if cluster_history_result is None else cluster_history_result.index,
            cluster_artifacts=None if cluster_artifact_result is None else cluster_artifact_result.artifacts,
        ),
        "user_history_result": history_result,
        "explicit_feedback_result": explicit_result,
        "cluster_artifact_result": cluster_artifact_result,
        "cluster_history_result": cluster_history_result,
        "induction_config": induction_config,
        "cluster_induction_wall_clock_seconds": clustering_seconds,
    }


def _build_caches_payload(
    *,
    split_result: SplitCacheResult,
    artifact_resolution: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "split": {
            "status": split_result.metadata.cache_status,
            "manifest": repo_path_string(split_result.metadata.cache_manifest_path, repo_root=repo_root),
            "cache_root": repo_path_string(split_result.metadata.cache_root, repo_root=repo_root),
            "source_fingerprint_sha256": split_result.metadata.source_fingerprint.sha256,
        }
    }
    history_result = artifact_resolution["user_history_result"]
    if history_result is not None:
        payload["training_user_history"] = {
            "status": history_result.metadata.cache_status,
            "manifest": repo_path_string(history_result.metadata.cache_manifest_path, repo_root=repo_root),
            "cache_root": repo_path_string(history_result.metadata.cache_root, repo_root=repo_root),
            "train_fingerprint_sha256": history_result.metadata.train_fingerprint.sha256,
        }
    explicit_result = artifact_resolution["explicit_feedback_result"]
    if explicit_result is not None:
        payload["training_explicit_feedback"] = {
            "status": explicit_result.metadata.cache_status,
            "manifest": repo_path_string(explicit_result.metadata.cache_manifest_path, repo_root=repo_root),
            "cache_root": repo_path_string(explicit_result.metadata.cache_root, repo_root=repo_root),
            "train_fingerprint_sha256": explicit_result.metadata.train_fingerprint.sha256,
        }
    cluster_result = artifact_resolution["cluster_artifact_result"]
    if cluster_result is not None:
        payload["cluster_artifacts"] = {
            "status": cluster_result.metadata.cache_status,
            "manifest": repo_path_string(cluster_result.metadata.cache_manifest_path, repo_root=repo_root),
            "cache_root": repo_path_string(cluster_result.metadata.cache_root, repo_root=repo_root),
            "cache_key": cluster_result.metadata.cache_key,
            "cache_fingerprint_sha256": cluster_result.metadata.cache_fingerprint_sha256,
            "train_fingerprint_sha256": cluster_result.metadata.train_fingerprint.sha256,
        }
    cluster_history_result = artifact_resolution["cluster_history_result"]
    if cluster_history_result is not None:
        payload["user_cluster_history"] = {
            "status": cluster_history_result.metadata.cache_status,
            "manifest": repo_path_string(cluster_history_result.metadata.cache_manifest_path, repo_root=repo_root),
            "cache_root": repo_path_string(cluster_history_result.metadata.cache_root, repo_root=repo_root),
            "cache_key": cluster_history_result.metadata.cache_key,
            "cache_fingerprint_sha256": cluster_history_result.metadata.cache_fingerprint_sha256,
            "train_fingerprint_sha256": cluster_history_result.metadata.train_fingerprint.sha256,
        }
    return payload


def _build_model_payload(
    *,
    adapter: type[ModelAdapter],
    model: object,
    model_config: object,
    model_profile: ModelProfileSchema,
    fit_artifacts: FitArtifacts,
    split_result: SplitCacheResult,
    artifact_resolution: dict[str, Any],
    split_id_for_cache: str,
    split_cache_policy,
    use_training_index_cache: bool,
    use_cluster_artifact_cache: bool,
    reuse_precomputed_indices: bool,
    repo_root: Path,
    cb_semantics: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": adapter.name,
        "family": adapter.family,
        "scope": str(model_profile.model.scope),
        "config": adapter.config_payload(model_config),
        "requirements": model_requirements_payload(adapter),
        "available_fit_artifacts": fit_artifacts.available_artifact_names(),
        "precomputed_index_reuse": reuse_precomputed_indices,
        "split_cache": {
            "enabled": split_cache_policy.effective_use_cache,
            "requested_policy": split_cache_policy.requested_policy,
            "decision_reason": split_cache_policy.decision_reason,
            "split_id": split_id_for_cache,
            "status": split_result.metadata.cache_status,
            "manifest": repo_path_string(split_result.metadata.cache_manifest_path, repo_root=repo_root),
        },
    }
    training_backend = getattr(model, "training_backend_effective", None)
    if training_backend is not None:
        payload["training_backend_effective"] = training_backend
    if cb_semantics is not None:
        payload["cb_semantics"] = cb_semantics

    history_result = artifact_resolution["user_history_result"]
    explicit_result = artifact_resolution["explicit_feedback_result"]
    cluster_history_result = artifact_resolution["cluster_history_result"]
    cluster_result = artifact_resolution["cluster_artifact_result"]

    if history_result is not None or explicit_result is not None or cluster_history_result is not None:
        training_index_cache: dict[str, Any] = {
            "enabled": use_training_index_cache,
            "split_id": split_id_for_cache,
        }
        if history_result is not None:
            training_index_cache["cache_root"] = repo_path_string(
                history_result.metadata.cache_root,
                repo_root=repo_root,
            )
            training_index_cache["train_fingerprint_sha256"] = history_result.metadata.train_fingerprint.sha256
            training_index_cache["user_history"] = {
                "status": history_result.metadata.cache_status,
                "manifest": repo_path_string(history_result.metadata.cache_manifest_path, repo_root=repo_root),
            }
            payload["implicit_summary"] = _implicit_summary(history_result.index)
        if explicit_result is not None:
            training_index_cache["cache_root"] = repo_path_string(
                explicit_result.metadata.cache_root,
                repo_root=repo_root,
            )
            training_index_cache["train_fingerprint_sha256"] = explicit_result.metadata.train_fingerprint.sha256
            training_index_cache["explicit_feedback"] = {
                "status": explicit_result.metadata.cache_status,
                "manifest": repo_path_string(explicit_result.metadata.cache_manifest_path, repo_root=repo_root),
            }
            payload["explicit_summary"] = _explicit_summary(explicit_result.index)
        if cluster_history_result is not None:
            training_index_cache["user_cluster_history"] = {
                "enabled": use_cluster_artifact_cache,
                "status": cluster_history_result.metadata.cache_status,
                "manifest": repo_path_string(cluster_history_result.metadata.cache_manifest_path, repo_root=repo_root),
                "cache_root": repo_path_string(cluster_history_result.metadata.cache_root, repo_root=repo_root),
                "cache_key": cluster_history_result.metadata.cache_key,
                "cache_fingerprint_sha256": cluster_history_result.metadata.cache_fingerprint_sha256,
                "train_fingerprint_sha256": cluster_history_result.metadata.train_fingerprint.sha256,
            }
        payload["training_index_cache"] = training_index_cache

    if cluster_result is not None:
        cluster_cache_payload = {
            "enabled": use_cluster_artifact_cache,
            "status": cluster_result.metadata.cache_status,
            "manifest": repo_path_string(cluster_result.metadata.cache_manifest_path, repo_root=repo_root),
            "cache_root": repo_path_string(cluster_result.metadata.cache_root, repo_root=repo_root),
            "cache_key": cluster_result.metadata.cache_key,
            "cache_fingerprint_sha256": cluster_result.metadata.cache_fingerprint_sha256,
            "train_fingerprint_sha256": cluster_result.metadata.train_fingerprint.sha256,
        }
        payload["cluster_artifact_cache"] = cluster_cache_payload
        payload["cluster_history_cache"] = (
            None
            if cluster_history_result is None
            else payload["training_index_cache"]["user_cluster_history"]
        )
        payload["clustering"] = _clustering_payload(
            model_profile=model_profile,
            model_config=model_config,
            fit_artifacts=fit_artifacts,
            artifact_resolution=artifact_resolution,
        )
    return payload


def _implicit_summary(index) -> dict[str, Any]:
    counts = index.counts.astype("int64", copy=False)
    return {
        "users_with_history": int((counts > 0).sum()),
        "mean_history_size": float(counts.mean()),
        "max_history_size": int(counts.max()),
    }


def _explicit_summary(index) -> dict[str, Any]:
    counts = index.counts.astype("int64", copy=False)
    return {
        "users_with_explicit_history": int((counts > 0).sum()),
        "mean_explicit_history_size": float(counts.mean()),
        "max_explicit_history_size": int(counts.max()),
    }


def _clustering_payload(
    *,
    model_profile: ModelProfileSchema,
    model_config: object,
    fit_artifacts: FitArtifacts,
    artifact_resolution: dict[str, Any],
) -> dict[str, Any]:
    cluster_artifacts = fit_artifacts.cluster_artifacts
    history_index = fit_artifacts.user_history_index
    cluster_history_index = fit_artifacts.user_cluster_history_index
    if cluster_artifacts is None or history_index is None or cluster_history_index is None:
        raise ValueError("clustering payload requires cluster, user-history, and user-cluster-history artifacts")

    clustering_config = _clustering_config(model_profile)
    history_counts = history_index.counts.astype("int64", copy=False)
    per_user_active_cluster_counts = np.diff(cluster_history_index.indptr).astype("int64", copy=False)
    r_star_counts = cluster_artifacts.r_star_counts.astype("int64", copy=False)
    nonempty_cluster_pairs = int((r_star_counts > 0).sum())
    total_cluster_pairs = int(r_star_counts.size)
    observed = cluster_artifacts.r_star_means[r_star_counts > 0]
    return {
        "induction_model": "biased_mf",
        "induction_config": asdict(artifact_resolution["induction_config"])
        if artifact_resolution.get("induction_config") is not None
        else _induction_payload_from_model_config(model_config),
        "algorithm": clustering_config.algorithm,
        "kmeans_n_init": clustering_config.kmeans_n_init,
        "train_only_assignments": True,
        "fixed_assignments_during_cb_training": True,
        "r_star_role": "diagnostic_only",
        "n_user_clusters": int(cluster_artifacts.r_star_counts.shape[0]),
        "n_item_clusters": int(cluster_artifacts.r_star_counts.shape[1]),
        "user_cluster_summary": _cluster_size_summary(cluster_artifacts.user_cluster_sizes),
        "item_cluster_summary": _cluster_size_summary(cluster_artifacts.item_cluster_sizes),
        "history_cluster_summary": {
            "mean_active_item_clusters_per_user": float(per_user_active_cluster_counts.mean()),
            "max_active_item_clusters_per_user": int(per_user_active_cluster_counts.max()),
            "users_with_history": int((history_counts > 0).sum()),
        },
        "r_star_summary": {
            "nonempty_pairs": nonempty_cluster_pairs,
            "total_pairs": total_cluster_pairs,
            "density": float(nonempty_cluster_pairs / total_cluster_pairs) if total_cluster_pairs else 0.0,
            "observed_rating_min": None if observed.size == 0 else float(observed.min()),
            "observed_rating_max": None if observed.size == 0 else float(observed.max()),
        },
        "induction_diagnostics": {
            "train_rmse": cluster_artifacts.induction_train_rmse,
            "user_kmeans_inertia": cluster_artifacts.user_kmeans_inertia,
            "item_kmeans_inertia": cluster_artifacts.item_kmeans_inertia,
        },
    }


def _induction_payload_from_model_config(model_config: object) -> dict[str, Any]:
    config = cast(Any, model_config)
    return {
        "latent_dim": int(config.latent_dim),
        "epochs": int(config.epochs),
        "learning_rate": float(config.learning_rate),
        "lambda_b": float(config.lambda_b),
        "lambda_p": float(config.lambda_p),
        "lambda_q": float(config.lambda_q),
        "seed": int(config.seed),
        "init_std": float(config.init_std),
        "dtype": str(config.dtype),
        "training_backend": "auto",
    }


def _cluster_size_summary(sizes: np.ndarray) -> dict[str, Any]:
    size_values = np.asarray(sizes, dtype=np.int64)
    if size_values.size == 0:
        return {
            "min_size": None,
            "max_size": None,
            "mean_size": None,
            "empty_cluster_count": 0,
            "near_empty_cluster_count_le1": 0,
            "nonempty_cluster_count": 0,
        }
    return {
        "min_size": int(size_values.min()),
        "max_size": int(size_values.max()),
        "mean_size": float(size_values.mean()),
        "empty_cluster_count": int((size_values == 0).sum()),
        "near_empty_cluster_count_le1": int((size_values <= 1).sum()),
        "nonempty_cluster_count": int((size_values > 0).sum()),
    }


def _model_epochs(model_config: object) -> int:
    return int(cast(Any, model_config).epochs)


def _build_cb_diagnostics(
    *,
    model: object,
    train_data,
    fit_artifacts: FitArtifacts,
    cb_semantics: dict[str, Any],
) -> dict[str, Any]:
    cluster_artifacts = fit_artifacts.cluster_artifacts
    missing_expected_artifacts = _missing_expected_cb_artifacts(fit_artifacts)
    missing_expected_model_fields = _missing_expected_cb_model_fields(model)
    diagnostic_warnings: list[str] = []

    user_cluster_sizes = (
        np.asarray(cluster_artifacts.user_cluster_sizes, dtype=np.int64)
        if cluster_artifacts is not None
        else np.asarray([], dtype=np.int64)
    )
    item_cluster_sizes = (
        np.asarray(cluster_artifacts.item_cluster_sizes, dtype=np.int64)
        if cluster_artifacts is not None
        else np.asarray([], dtype=np.int64)
    )

    contribution: np.ndarray | None = None
    try:
        sample_indices = _diagnostic_sample_indices(train_data, max_rows=10_000)
        users = train_data.user_ids[sample_indices]
        items = train_data.item_ids[sample_indices]
        actual = model.predict_many(users, items, clip=False)
        alpha_zero = _predict_alpha_zero(model=model, user_ids=users, item_ids=items)
        contribution = np.asarray(actual - alpha_zero, dtype=np.float64)
        diagnostic_sample_rows = int(sample_indices.shape[0])
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        diagnostic_sample_rows = 0
        diagnostic_warnings.append(f"cluster contribution diagnostic failed: {type(exc).__name__}: {exc}")

    individual_factor_norm_mean = _factor_norm_mean(
        model,
        ("user_factors", "item_factors"),
    )
    cluster_factor_norm_mean = _factor_norm_mean(
        model,
        ("user_cluster_factors", "item_cluster_factors"),
    )
    cluster_to_individual_norm_ratio = _safe_ratio(
        numerator=cluster_factor_norm_mean,
        denominator=individual_factor_norm_mean,
    )
    cluster_contribution_measured = (
        _cluster_contribution_measured(contribution) if contribution is not None else None
    )
    return {
        "alpha": float(cb_semantics["alpha"]),
        "effective_alpha": float(cb_semantics["alpha"]),
        "cluster_artifacts_present": cluster_artifacts is not None,
        "user_cluster_count": int(user_cluster_sizes.size) if cluster_artifacts is not None else None,
        "item_cluster_count": int(item_cluster_sizes.size) if cluster_artifacts is not None else None,
        "empty_user_clusters": _empty_cluster_count(user_cluster_sizes),
        "empty_item_clusters": _empty_cluster_count(item_cluster_sizes),
        "user_cluster_size_min": _cluster_size_min(user_cluster_sizes),
        "user_cluster_size_max": _cluster_size_max(user_cluster_sizes),
        "item_cluster_size_min": _cluster_size_min(item_cluster_sizes),
        "item_cluster_size_max": _cluster_size_max(item_cluster_sizes),
        "individual_factor_norm_mean": individual_factor_norm_mean,
        "cluster_factor_norm_mean": cluster_factor_norm_mean,
        "cluster_to_individual_norm_ratio": cluster_to_individual_norm_ratio,
        "implicit_factor_norm_mean": _factor_norm_mean(model, ("implicit_factors",)),
        "implicit_cluster_factor_norm_mean": _factor_norm_mean(model, ("implicit_cluster_factors",)),
        "explicit_factor_norm_mean": _factor_norm_mean(model, ("explicit_factors",)),
        "explicit_cluster_factor_norm_mean": _factor_norm_mean(model, ("explicit_cluster_factors",)),
        "missing_expected_artifacts": missing_expected_artifacts,
        "missing_expected_model_fields": missing_expected_model_fields,
        "diagnostic_claim_ready": False,
        "diagnostic_warnings": diagnostic_warnings,
        "cluster_contribution_config_enabled": bool(cb_semantics["cluster_contribution_config_enabled"]),
        "cluster_contribution_measured": cluster_contribution_measured,
        "cb_claim_eligible": bool(cb_semantics["cb_claim_eligible"]),
        "claim_gate_reason": str(cb_semantics["claim_gate_reason"]),
        "cluster_factor_norm_ratio": cluster_to_individual_norm_ratio,
        "average_absolute_cluster_prediction_contribution": (
            float(np.mean(np.abs(contribution))) if contribution is not None else None
        ),
        "diagnostic_sample_rows": diagnostic_sample_rows,
        "r_star_density": (
            _r_star_density(cluster_artifacts.r_star_counts) if cluster_artifacts is not None else None
        ),
        "cluster_size_distribution": {
            "users": _cluster_size_summary(user_cluster_sizes),
            "items": _cluster_size_summary(item_cluster_sizes),
        },
        "cluster_coverage": {
            "user_cluster_coverage": _cluster_coverage(user_cluster_sizes),
            "item_cluster_coverage": _cluster_coverage(item_cluster_sizes),
        },
        "alpha_zero_ablation_reference": {
            "available": False,
            "reason": "no same-split alpha=0 reference run was supplied to the unified runner",
        },
    }


def _missing_expected_cb_artifacts(fit_artifacts: FitArtifacts) -> list[str]:
    missing: list[str] = []
    if fit_artifacts.cluster_artifacts is None:
        missing.append("cluster_artifacts")
    if fit_artifacts.user_cluster_history_index is None:
        missing.append("user_cluster_history_index")
    return missing


def _missing_expected_cb_model_fields(model: object) -> list[str]:
    if isinstance(model, CBSVDppRecommender):
        expected = (
            "user_factors",
            "item_factors",
            "implicit_factors",
            "user_cluster_factors",
            "item_cluster_factors",
            "implicit_cluster_factors",
            "user_histories",
            "user_cluster_histories",
        )
    elif isinstance(model, CBASVDppRecommender):
        expected = (
            "user_factors",
            "item_factors",
            "explicit_factors",
            "implicit_factors",
            "user_cluster_factors",
            "item_cluster_factors",
            "explicit_cluster_factors",
            "implicit_cluster_factors",
            "explicit_feedback",
            "implicit_history",
            "implicit_cluster_history",
        )
    else:
        return ["unsupported_cb_model_type"]
    return [name for name in expected if getattr(model, name, None) is None]


def _cluster_contribution_measured(contribution: np.ndarray | None) -> bool:
    if contribution is None:
        return False
    values = np.asarray(contribution, dtype=np.float64)
    return bool(values.size and np.max(np.abs(values)) > 1.0e-12)


def _empty_cluster_count(sizes: np.ndarray) -> int | None:
    values = np.asarray(sizes, dtype=np.int64)
    return int((values == 0).sum()) if values.size else None


def _cluster_size_min(sizes: np.ndarray) -> int | None:
    values = np.asarray(sizes, dtype=np.int64)
    return int(values.min()) if values.size else None


def _cluster_size_max(sizes: np.ndarray) -> int | None:
    values = np.asarray(sizes, dtype=np.int64)
    return int(values.max()) if values.size else None


def _diagnostic_sample_indices(data, *, max_rows: int) -> np.ndarray:
    row_count = len(data)
    if row_count <= max_rows:
        return np.arange(row_count, dtype=np.int64)
    return np.unique(np.linspace(0, row_count - 1, num=max_rows, dtype=np.int64))


def _r_star_density(r_star_counts: np.ndarray) -> float:
    counts = np.asarray(r_star_counts)
    return float((counts > 0).sum() / counts.size) if counts.size else 0.0


def _cluster_coverage(sizes: np.ndarray) -> float:
    size_values = np.asarray(sizes)
    return float((size_values > 0).sum() / size_values.size) if size_values.size else 0.0


def _factor_norm_mean(model: object, factor_names: tuple[str, ...]) -> float | None:
    row_norms: list[np.ndarray] = []
    for factor_name in factor_names:
        factor = getattr(model, factor_name, None)
        if factor is None:
            continue
        values = np.asarray(factor, dtype=np.float64)
        if values.size == 0:
            continue
        if values.ndim == 1:
            row_norms.append(np.abs(values))
        else:
            rows = values.reshape(values.shape[0], -1)
            row_norms.append(np.linalg.norm(rows, axis=1))
    if not row_norms:
        return None
    return float(np.mean(np.concatenate(row_norms)))


def _safe_ratio(*, numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return float(numerator / denominator)


def _predict_alpha_zero(*, model: object, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
    if isinstance(model, CBSVDppRecommender):
        return _predict_cb_svdpp_alpha_zero(model=model, user_ids=user_ids, item_ids=item_ids)
    if isinstance(model, CBASVDppRecommender):
        return _predict_cb_asvdpp_alpha_zero(model=model, user_ids=user_ids, item_ids=item_ids)
    raise TypeError("alpha-zero CB diagnostic supports cb_svdpp and cb_asvdpp models")


def _predict_cb_svdpp_alpha_zero(
    *,
    model: CBSVDppRecommender,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
) -> np.ndarray:
    if (
        model.user_bias is None
        or model.item_bias is None
        or model.user_factors is None
        or model.item_factors is None
        or model.implicit_factors is None
        or model.user_histories is None
    ):
        raise RuntimeError("model parameters are not initialized")
    users = np.asarray(user_ids, dtype=np.int64)
    items = np.asarray(item_ids, dtype=np.int64)
    contexts = model.user_factors.astype(np.float64, copy=True)
    for user_id in np.unique(users):
        history = model.user_histories.items_for_user(int(user_id))
        if history.size > 0:
            norm = float(model.user_histories.norms[int(user_id)])
            contexts[int(user_id)] += norm * np.sum(
                model.implicit_factors[history].astype(np.float64, copy=False),
                axis=0,
            )
    return np.asarray(
        model.global_mean
        + model.user_bias[users].astype(np.float64, copy=False)
        + model.item_bias[items].astype(np.float64, copy=False)
        + np.sum(contexts[users] * model.item_factors[items].astype(np.float64, copy=False), axis=1),
        dtype=np.float64,
    )


def _predict_cb_asvdpp_alpha_zero(
    *,
    model: CBASVDppRecommender,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
) -> np.ndarray:
    if (
        model.user_bias is None
        or model.item_bias is None
        or model.user_factors is None
        or model.item_factors is None
        or model.explicit_factors is None
        or model.implicit_factors is None
        or model.explicit_feedback is None
        or model.implicit_history is None
    ):
        raise RuntimeError("model parameters are not initialized")
    users = np.asarray(user_ids, dtype=np.int64)
    items = np.asarray(item_ids, dtype=np.int64)
    contexts = model.user_factors.astype(np.float64, copy=True)
    for user_id in np.unique(users):
        user_index = int(user_id)
        explicit_items = model.explicit_feedback.items_for_user(user_index)
        if explicit_items.size > 0:
            explicit_ratings = model.explicit_feedback.ratings_for_user(user_index).astype(np.float64, copy=False)
            residual_weights = explicit_ratings - (
                model.global_mean
                + float(model.user_bias[user_index])
                + model.item_bias[explicit_items].astype(np.float64, copy=False)
            )
            contexts[user_index] += float(model.explicit_feedback.norms[user_index]) * np.sum(
                residual_weights[:, None] * model.explicit_factors[explicit_items].astype(np.float64, copy=False),
                axis=0,
            )
        implicit_items = model.implicit_history.items_for_user(user_index)
        if implicit_items.size > 0:
            contexts[user_index] += float(model.implicit_history.norms[user_index]) * np.sum(
                model.implicit_factors[implicit_items].astype(np.float64, copy=False),
                axis=0,
            )
    return np.asarray(
        model.global_mean
        + model.user_bias[users].astype(np.float64, copy=False)
        + model.item_bias[items].astype(np.float64, copy=False)
        + np.sum(contexts[users] * model.item_factors[items].astype(np.float64, copy=False), axis=1),
        dtype=np.float64,
    )
