from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recsys_lab.config.loader import load_yaml_file
from recsys_lab.data.processed import load_processed_dataset_manifest
from recsys_lab.experiments.common import (
    SplitConfig,
    paper_faithful_ml100k_inner_split_id,
    paper_faithful_ml100k_split_id,
    split_id,
)
from recsys_lab.experiments.performance import StageProfiler
from recsys_lab.experiments.runtime import RuntimeThreadingConfig, resolve_runtime_threading_config
from recsys_lab.experiments.split_cache import SplitCachePolicy, resolve_split_cache_policy
from recsys_lab.experiments.unified.context import ResolvedExperimentConfig, ResolvedModelProfile
from recsys_lab.models.registry import (
    ModelAdapter,
    build_cb_semantics,
    model_requirements_payload,
    validate_model_config_payload,
)
from recsys_lab.utils.paths import discover_repo_root, repo_path_string


@dataclass(frozen=True, slots=True)
class UnifiedConfigResolution:
    experiment_config: ResolvedExperimentConfig
    model_profile: ResolvedModelProfile
    split_cache_policy: SplitCachePolicy
    run_context_slug: str
    device_profile_name: str
    command_string: str
    threading_config: RuntimeThreadingConfig


def resolve_unified_experiment_config(
    *,
    processed_manifest_path: Path,
    model_config_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    split_config: SplitConfig,
    model_seed: int,
    repo_root: Path | None,
    command: str | None,
    model_name: str | None,
    split_family: str | None,
    inner_validation_seed: int | None,
    evaluate_test: bool,
    use_split_cache: bool | None,
    use_training_index_cache: bool,
    use_cluster_artifact_cache: bool,
    stage_profiler: StageProfiler,
) -> UnifiedConfigResolution:
    with stage_profiler.stage("resolve_experiment_config"):
        root = (repo_root or discover_repo_root()).resolve()

        processed_manifest_path = processed_manifest_path.resolve()
        model_config_path = model_config_path.resolve()
        runtime_config_path = runtime_config_path.resolve()
        device_config_path = device_config_path.resolve()

        runtime_config_payload = load_yaml_file(runtime_config_path)
        device_config_payload = load_yaml_file(device_config_path)
        threading_config = resolve_runtime_threading_config(device_config_payload=device_config_payload)

    with stage_profiler.stage(
        "resolve_dataset_manifest",
        metadata={"processed_manifest": str(processed_manifest_path)},
    ) as dataset_stage:
        processed_manifest = load_processed_dataset_manifest(processed_manifest_path)
        dataset_short_name = str(processed_manifest["dataset_short_name"])
        requested_split_family = split_family or str(processed_manifest["split_family"])
        dataset_stage["dataset"] = dataset_short_name
        dataset_stage["split_family"] = requested_split_family

    with stage_profiler.stage("resolve_model_profile", metadata={"model_config": str(model_config_path)}):
        raw_model_config_payload = load_yaml_file(model_config_path)

    with stage_profiler.stage(
        "validate_model_config",
        metadata={"expected_model_name": model_name},
    ):
        adapter, model_profile = validate_model_config_payload(
            raw_model_config_payload,
            expected_model_name=model_name,
        )
        runtime_dtype = adapter.runtime_dtype(model_profile)

    with stage_profiler.stage(
        "resolve_split_cache_policy",
        metadata={"split_family": requested_split_family, "requested_policy": use_split_cache},
    ):
        split_cache_policy = resolve_split_cache_policy(
            split_family=requested_split_family,
            use_split_cache=use_split_cache,
        )
        validate_cache_options(
            adapter=adapter,
            use_training_index_cache=use_training_index_cache,
            use_cluster_artifact_cache=use_cluster_artifact_cache,
        )

    run_context_slug = run_context_slug_for_split(
        requested_split_family=requested_split_family,
        split_config=split_config,
        inner_validation_seed=inner_validation_seed,
    )
    device_profile_name = str(device_config_payload["device_profile"]["name"])
    command_string = command or default_command_string(
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
    cb_semantics = cb_semantics_for_profile(adapter=adapter, model_profile=model_profile)

    return UnifiedConfigResolution(
        experiment_config=ResolvedExperimentConfig(
            repo_root=root,
            processed_manifest_path=processed_manifest_path,
            model_config_path=model_config_path,
            runtime_config_path=runtime_config_path,
            device_config_path=device_config_path,
            runtime_config_payload=runtime_config_payload,
            device_config_payload=device_config_payload,
            threading_config=threading_config,
            processed_manifest=processed_manifest,
            dataset_short_name=dataset_short_name,
            requested_split_family=requested_split_family,
            raw_model_config_payload=raw_model_config_payload,
        ),
        model_profile=ResolvedModelProfile(
            adapter=adapter,
            model_profile=model_profile,
            runtime_dtype=runtime_dtype,
            model_config=None,
            induction_config=None,
            cb_semantics=cb_semantics,
            model_requirements=model_requirements_payload(adapter),
        ),
        split_cache_policy=split_cache_policy,
        run_context_slug=run_context_slug,
        device_profile_name=device_profile_name,
        command_string=command_string,
        threading_config=threading_config,
    )


def validate_cache_options(
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


def run_context_slug_for_split(
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


def default_command_string(
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


def cb_semantics_for_profile(
    *,
    adapter: type[ModelAdapter],
    model_profile: Any,
) -> dict[str, Any] | None:
    alpha = adapter.cb_alpha(model_profile)
    if alpha is None:
        return None
    return build_cb_semantics(alpha)
