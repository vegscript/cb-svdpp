from __future__ import annotations

import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, cast

import numpy as np

from recsys_lab.config.loader import dump_yaml_file
from recsys_lab.data.splitters import (
    RatingsSplit,
    official_ml100k_inner_validation_split,
    official_ml100k_paper_faithful_split,
)
from recsys_lab.experiments.common import (
    SplitConfig,
    build_base_run_manifest,
    build_run_id,
    git_snapshot,
    ratings_summary,
    reserve_timestamped_artifact_dir,
    split_summary,
    utc_timestamp,
    write_json,
    write_log,
)
from recsys_lab.experiments.kernel_profile import build_kernel_profile_payload
from recsys_lab.experiments.performance import (
    PeakMemoryMonitor,
    StageProfiler,
    build_performance_profile_payload,
    build_system_metrics,
)
from recsys_lab.experiments.runtime import runtime_execution_context
from recsys_lab.experiments.split_cache import SplitCacheResult
from recsys_lab.experiments.unified.artifact_resolution import resolve_fit_artifacts
from recsys_lab.experiments.unified.config_resolution import resolve_unified_experiment_config
from recsys_lab.experiments.unified.context import FitArtifactResolution
from recsys_lab.experiments.unified.data_resolution import ratings_stage_metadata, resolve_unified_data_split
from recsys_lab.metrics import rating_error_metrics
from recsys_lab.models.cb_asvdpp import CBASVDppRecommender
from recsys_lab.models.cb_svdpp import CBSVDppRecommender
from recsys_lab.models.config_schemas import ModelProfileSchema
from recsys_lab.models.registry import (
    FitArtifacts,
    ModelAdapter,
    model_requirements_payload,
    pydantic_profile_payload,
)
from recsys_lab.utils.manifests import validate_manifest_file
from recsys_lab.utils.paths import repo_path_string


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
    stage_profiler = StageProfiler()
    experiment_services = services or DEFAULT_EXPERIMENT_SERVICES
    config_resolution = resolve_unified_experiment_config(
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_config=split_config,
        model_seed=model_seed,
        repo_root=repo_root,
        command=command,
        model_name=model_name,
        split_family=split_family,
        inner_validation_seed=inner_validation_seed,
        evaluate_test=evaluate_test,
        use_split_cache=use_split_cache,
        use_training_index_cache=use_training_index_cache,
        use_cluster_artifact_cache=use_cluster_artifact_cache,
        stage_profiler=stage_profiler,
    )
    experiment_config = config_resolution.experiment_config
    resolved_model_profile = config_resolution.model_profile
    root = experiment_config.repo_root
    processed_manifest_path = experiment_config.processed_manifest_path
    model_config_path = experiment_config.model_config_path
    runtime_config_path = experiment_config.runtime_config_path
    device_config_path = experiment_config.device_config_path
    runtime_config_payload = experiment_config.runtime_config_payload
    device_config_payload = experiment_config.device_config_payload
    threading_config = config_resolution.threading_config
    processed_manifest = experiment_config.processed_manifest
    dataset_short_name = experiment_config.dataset_short_name
    requested_split_family = experiment_config.requested_split_family
    raw_model_config_payload = experiment_config.raw_model_config_payload
    adapter = resolved_model_profile.adapter
    model_profile = resolved_model_profile.model_profile
    runtime_dtype = resolved_model_profile.runtime_dtype
    split_cache_policy = config_resolution.split_cache_policy
    run_context_slug = config_resolution.run_context_slug
    device_profile_name = config_resolution.device_profile_name
    command_string = config_resolution.command_string
    cb_semantics = resolved_model_profile.cb_semantics

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
    performance_profile_path = run_dir / "performance_profile.json"
    kernel_profile_path = run_dir / "kernel_profile.json"
    stdout_log_path = run_dir / "stdout.log"
    run_manifest_path = run_dir / "run_manifest.json"
    git = experiment_services.git_snapshot_fn(root)

    with runtime_execution_context(threading_config=threading_config):
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
        base_manifest["artifacts"]["performance_profile"] = repo_path_string(performance_profile_path, repo_root=root)
        base_manifest["artifacts"]["kernel_profile"] = repo_path_string(kernel_profile_path, repo_root=root)

        with stage_profiler.stage("resolve_model_requirements", metadata={"model_name": adapter.name}) as req_stage:
            model_requirements = model_requirements_payload(adapter)
            req_stage["required_artifacts"] = list(model_requirements["required_artifacts"])
            req_stage["required_artifact_count"] = len(model_requirements["required_artifacts"])

        with stage_profiler.stage(
            "write_config_snapshot",
            metadata={"path": repo_path_string(config_snapshot_path, repo_root=root)},
        ):
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
                "model_requirements": model_requirements,
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
        with stage_profiler.stage(
            "write_run_manifest",
            metadata={"path": repo_path_string(run_manifest_path, repo_root=root), "status": "started"},
        ):
            write_json(run_manifest_path, base_manifest)

        try:
            split_id_for_cache = str(base_manifest["dataset"]["split_id"])
            split_bundle = resolve_unified_data_split(
                processed_manifest_path=processed_manifest_path,
                dataset_short_name=dataset_short_name,
                requested_split_family=requested_split_family,
                split_config=split_config,
                inner_validation_seed=inner_validation_seed,
                split_id_for_cache=split_id_for_cache,
                repo_root=root,
                runtime_config_payload=runtime_config_payload,
                split_cache_policy=split_cache_policy,
                services=experiment_services,
                stage_profiler=stage_profiler,
            )
            ratings_data = split_bundle.ratings_data
            split_result = split_bundle.split_result
            split = split_bundle.split
            train_data = split_bundle.train_data
            validation_data = split_bundle.validation_data
            test_data = split_bundle.test_data

            with stage_profiler.stage("build_model_config", metadata={"model_name": adapter.name}) as config_stage:
                model_config = adapter.build_model_config(
                    model_profile,
                    model_seed=model_seed,
                    runtime_dtype=runtime_dtype,
                )
                induction_config = adapter.build_induction_config(
                    model_config,
                    model_seed=model_seed,
                    model_profile=model_profile,
                )
                config_stage.update(_fit_stage_model_config_metadata(model_config))

            clustering_seconds = 0.0
            with PeakMemoryMonitor() as memory_monitor:
                with stage_profiler.stage(
                    "build_fit_artifacts",
                    metadata={
                        "model_name": adapter.name,
                        "required_artifacts": list(model_requirements["required_artifacts"]),
                    },
                ):
                    artifact_resolution = resolve_fit_artifacts(
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
                fit_artifacts = artifact_resolution.artifacts
                clustering_seconds = float(artifact_resolution.cluster_induction_wall_clock_seconds)

                with stage_profiler.stage(
                    "initialize_model",
                    metadata={"model_name": adapter.name},
                ):
                    model = adapter.instantiate(model_config, artifacts=fit_artifacts)

                training_started = perf_counter()
                with stage_profiler.stage(
                    "fit_model",
                    metadata={
                        "train_rows": len(train_data),
                        **_fit_stage_model_config_metadata(model_config),
                    },
                ):
                    adapter.fit(
                        model,
                        train_data,
                        artifacts=fit_artifacts,
                        reuse_precomputed_indices=reuse_precomputed_indices,
                    )
                training_seconds = perf_counter() - training_started
                kernel_profile_payload = build_kernel_profile_payload(
                    run_id=run_id,
                    dataset=dataset_short_name,
                    model=adapter.name,
                    epochs=_model_epochs(model_config),
                    latent_dim=_model_latent_dim(model_config),
                    train_rows=len(train_data),
                    train_user_ids=train_data.user_ids,
                    epoch_durations_seconds=getattr(model, "epoch_durations_seconds", []),
                    fit_artifacts=fit_artifacts,
                )
                with stage_profiler.stage(
                    "write_kernel_profile",
                    metadata={"path": repo_path_string(kernel_profile_path, repo_root=root)},
                ):
                    write_json(kernel_profile_path, kernel_profile_payload)

                inference_started = perf_counter()
                with stage_profiler.stage("predict_train", metadata=ratings_stage_metadata(train_data)):
                    train_predictions = model.predict_dataset(train_data)
                validation_predictions = None
                if validation_data is not None:
                    with stage_profiler.stage("predict_validation", metadata=ratings_stage_metadata(validation_data)):
                        validation_predictions = model.predict_dataset(validation_data)
                test_predictions = None
                if evaluate_test:
                    with stage_profiler.stage("predict_test", metadata=ratings_stage_metadata(test_data)):
                        test_predictions = model.predict_dataset(test_data)
                inference_seconds = perf_counter() - inference_started

            inference_rows = len(train_data)
            if validation_data is not None:
                inference_rows += len(validation_data)
            if evaluate_test:
                inference_rows += len(test_data)

            system_metrics = build_system_metrics(
                train_rows=len(train_data),
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

            with stage_profiler.stage(
                "build_rating_metrics",
                metadata={
                    "train_rows": len(train_data),
                    "validation_rows": 0 if validation_data is None else len(validation_data),
                    "test_rows": len(test_data) if evaluate_test else 0,
                    "test_metrics_available": evaluate_test,
                },
            ):
                rating_metrics_payload = _build_rating_metrics_payload(
                    train_ratings=train_data.ratings,
                    train_predictions=train_predictions,
                    validation_ratings=None if validation_data is None else validation_data.ratings,
                    validation_predictions=validation_predictions,
                    test_ratings=test_data.ratings if evaluate_test else None,
                    test_predictions=test_predictions,
                    rating_min=ratings_data.rating_min,
                    rating_max=ratings_data.rating_max,
                )
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
            profiling_payload = stage_profiler.to_payload()
            performance_profile_summary_payload = _build_performance_profile(
                stage_profiler=stage_profiler,
                run_id=run_id,
                dataset_short_name=dataset_short_name,
                model_name=adapter.name,
                device_profile_name=device_profile_name,
                requested_split_family=requested_split_family,
                split_config=split_config,
                model_seed=model_seed,
            )

            metrics_payload: dict[str, Any] = {
                "run_id": run_id,
                "profiling": profiling_payload,
                "performance_profile": _performance_profile_summary(
                    performance_profile_payload=performance_profile_summary_payload,
                    performance_profile_path=performance_profile_path,
                ),
                "kernel_profile": _kernel_profile_summary(
                    kernel_profile_payload=kernel_profile_payload,
                    kernel_profile_path=kernel_profile_path,
                ),
                "artifacts": {
                    "performance_profile": repo_path_string(performance_profile_path, repo_root=root),
                    "kernel_profile": repo_path_string(kernel_profile_path, repo_root=root),
                },
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
                "metrics": rating_metrics_payload,
            }
            if cb_semantics is not None:
                metrics_payload["cb_semantics"] = cb_semantics
                with stage_profiler.stage(
                    "build_cb_diagnostics",
                    metadata={
                        "alpha": float(cb_semantics["alpha"]),
                        "train_rows": len(train_data),
                    },
                ):
                    metrics_payload["cb_diagnostics"] = _build_cb_diagnostics(
                        model=model,
                        train_data=train_data,
                        fit_artifacts=fit_artifacts,
                        cb_semantics=cb_semantics,
                    )
            with stage_profiler.stage(
                "write_metrics_json",
                metadata={"path": repo_path_string(metrics_path, repo_root=root)},
            ):
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
                        f"train_rows={len(train_data)} "
                        f"validation_rows={0 if validation_data is None else len(validation_data)} "
                        f"test_rows={len(test_data)}"
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
            with stage_profiler.stage(
                "write_run_manifest",
                metadata={"path": repo_path_string(run_manifest_path, repo_root=root), "status": "completed"},
            ):
                write_json(run_manifest_path, completed_manifest)
            validate_manifest_file(run_manifest_path, repo_root=root)
            _write_performance_profile(
                performance_profile_path=performance_profile_path,
                stage_profiler=stage_profiler,
                run_id=run_id,
                dataset_short_name=dataset_short_name,
                model_name=adapter.name,
                device_profile_name=device_profile_name,
                requested_split_family=requested_split_family,
                split_config=split_config,
                model_seed=model_seed,
                repo_root=root,
            )
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
            with stage_profiler.stage(
                "write_run_manifest",
                metadata={"path": repo_path_string(run_manifest_path, repo_root=root), "status": "failed"},
            ):
                write_json(run_manifest_path, failed_manifest)
            validate_manifest_file(run_manifest_path, repo_root=root)
            _write_performance_profile(
                performance_profile_path=performance_profile_path,
                stage_profiler=stage_profiler,
                run_id=run_id,
                dataset_short_name=dataset_short_name,
                model_name=adapter.name,
                device_profile_name=device_profile_name,
                requested_split_family=requested_split_family,
                split_config=split_config,
                model_seed=model_seed,
                repo_root=root,
            )
            raise


def _write_performance_profile(
    *,
    performance_profile_path: Path,
    stage_profiler: StageProfiler,
    run_id: str,
    dataset_short_name: str,
    model_name: str,
    device_profile_name: str,
    requested_split_family: str,
    split_config: SplitConfig,
    model_seed: int,
    repo_root: Path,
) -> dict[str, Any]:
    initial_payload = _build_performance_profile(
        stage_profiler=stage_profiler,
        run_id=run_id,
        dataset_short_name=dataset_short_name,
        model_name=model_name,
        device_profile_name=device_profile_name,
        requested_split_family=requested_split_family,
        split_config=split_config,
        model_seed=model_seed,
    )
    with stage_profiler.stage(
        "write_performance_profile",
        metadata={"path": repo_path_string(performance_profile_path, repo_root=repo_root)},
    ):
        write_json(performance_profile_path, initial_payload)
    final_payload = _build_performance_profile(
        stage_profiler=stage_profiler,
        run_id=run_id,
        dataset_short_name=dataset_short_name,
        model_name=model_name,
        device_profile_name=device_profile_name,
        requested_split_family=requested_split_family,
        split_config=split_config,
        model_seed=model_seed,
    )
    write_json(performance_profile_path, final_payload)
    return final_payload


def _build_performance_profile(
    *,
    stage_profiler: StageProfiler,
    run_id: str,
    dataset_short_name: str,
    model_name: str,
    device_profile_name: str,
    requested_split_family: str,
    split_config: SplitConfig,
    model_seed: int,
) -> dict[str, Any]:
    return build_performance_profile_payload(
        stage_profile=stage_profiler.to_payload(),
        run_id=run_id,
        dataset=dataset_short_name,
        model=model_name,
        device_profile=device_profile_name,
        split_family=requested_split_family,
        split_seed=split_config.seed,
        model_seed=model_seed,
    )


def _performance_profile_summary(
    *,
    performance_profile_payload: dict[str, Any],
    performance_profile_path: Path,
) -> dict[str, Any]:
    return {
        "path": performance_profile_path.name,
        "profile_version": performance_profile_payload["profile_version"],
        "stage_count": performance_profile_payload["stage_count"],
        "total_profiled_wall_clock_seconds": performance_profile_payload["total_profiled_wall_clock_seconds"],
        "top_hotspots": performance_profile_payload["hotspots"][:5],
    }


def _kernel_profile_summary(
    *,
    kernel_profile_payload: dict[str, Any],
    kernel_profile_path: Path,
) -> dict[str, Any]:
    return {
        "path": kernel_profile_path.name,
        "profile_version": kernel_profile_payload["profile_version"],
        "epoch_count": len(kernel_profile_payload["epoch_durations_seconds"]),
        "train_rows": kernel_profile_payload["train_rows"],
        "estimated_factor_touches": kernel_profile_payload["estimated_kernel_work"]["estimated_factor_touches"],
        "fit_seconds_per_million_estimated_factor_touches": kernel_profile_payload["cost_ratios"][
            "fit_seconds_per_million_estimated_factor_touches"
        ],
    }


def _cache_hit(cache_status: str) -> bool:
    return cache_status == "hit"


def _cache_metadata_payload(
    *,
    cache_status: str,
    cache_manifest_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "cache_status": cache_status,
        "cache_hit": _cache_hit(cache_status),
        "cache_path": repo_path_string(cache_manifest_path, repo_root=repo_root),
    }


def _fit_stage_model_config_metadata(model_config: object) -> dict[str, Any]:
    config = cast(Any, model_config)
    payload = {
        "epochs": int(config.epochs),
        "latent_dim": int(config.latent_dim),
    }
    training_backend = getattr(config, "training_backend", None)
    if training_backend is not None:
        payload["training_backend"] = str(training_backend)
    return payload


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


def _build_caches_payload(
    *,
    split_result: SplitCacheResult,
    artifact_resolution: FitArtifactResolution,
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
    history_result = artifact_resolution.user_history_result
    if history_result is not None:
        payload["training_user_history"] = {
            "status": history_result.metadata.cache_status,
            "manifest": repo_path_string(history_result.metadata.cache_manifest_path, repo_root=repo_root),
            "cache_root": repo_path_string(history_result.metadata.cache_root, repo_root=repo_root),
            "train_fingerprint_sha256": history_result.metadata.train_fingerprint.sha256,
        }
    explicit_result = artifact_resolution.explicit_feedback_result
    if explicit_result is not None:
        payload["training_explicit_feedback"] = {
            "status": explicit_result.metadata.cache_status,
            "manifest": repo_path_string(explicit_result.metadata.cache_manifest_path, repo_root=repo_root),
            "cache_root": repo_path_string(explicit_result.metadata.cache_root, repo_root=repo_root),
            "train_fingerprint_sha256": explicit_result.metadata.train_fingerprint.sha256,
        }
    cluster_result = artifact_resolution.cluster_artifact_result
    if cluster_result is not None:
        payload["cluster_artifacts"] = {
            "status": cluster_result.metadata.cache_status,
            "manifest": repo_path_string(cluster_result.metadata.cache_manifest_path, repo_root=repo_root),
            "cache_root": repo_path_string(cluster_result.metadata.cache_root, repo_root=repo_root),
            "cache_key": cluster_result.metadata.cache_key,
            "cache_fingerprint_sha256": cluster_result.metadata.cache_fingerprint_sha256,
            "train_fingerprint_sha256": cluster_result.metadata.train_fingerprint.sha256,
        }
    cluster_history_result = artifact_resolution.cluster_history_result
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
    artifact_resolution: FitArtifactResolution,
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

    history_result = artifact_resolution.user_history_result
    explicit_result = artifact_resolution.explicit_feedback_result
    cluster_history_result = artifact_resolution.cluster_history_result
    cluster_result = artifact_resolution.cluster_artifact_result

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
    artifact_resolution: FitArtifactResolution,
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
        "induction_config": asdict(artifact_resolution.induction_config)
        if artifact_resolution.induction_config is not None
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


def _model_latent_dim(model_config: object) -> int:
    return int(cast(Any, model_config).latent_dim)


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
