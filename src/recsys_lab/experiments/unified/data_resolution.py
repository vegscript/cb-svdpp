from __future__ import annotations

from pathlib import Path
from typing import Any

from recsys_lab.data.processed import load_ratings_data_from_manifest
from recsys_lab.data.splitters import RatingsSplit, random_split_with_train_coverage
from recsys_lab.experiments.common import SplitConfig
from recsys_lab.experiments.performance import StageProfiler
from recsys_lab.experiments.split_cache import SplitCachePolicy, load_or_build_split_cache
from recsys_lab.experiments.unified.context import ResolvedSplitBundle
from recsys_lab.utils.paths import repo_path_string


def resolve_unified_data_split(
    *,
    processed_manifest_path: Path,
    dataset_short_name: str,
    requested_split_family: str,
    split_config: SplitConfig,
    inner_validation_seed: int | None,
    split_id_for_cache: str,
    repo_root: Path,
    runtime_config_payload: dict[str, Any],
    split_cache_policy: SplitCachePolicy,
    services: Any,
    stage_profiler: StageProfiler,
) -> ResolvedSplitBundle:
    with stage_profiler.stage(
        "load_ratings_data",
        metadata={"processed_manifest": str(processed_manifest_path)},
    ) as ratings_stage:
        ratings_data = load_ratings_data_from_manifest(processed_manifest_path)
        ratings_stage["rows"] = len(ratings_data)
        ratings_stage["n_users"] = ratings_data.n_users
        ratings_stage["n_items"] = ratings_data.n_items

    with stage_profiler.stage(
        "resolve_split_cache",
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
            repo_root=repo_root,
            runtime_config_payload=runtime_config_payload,
            build_split=lambda: build_unified_split(
                ratings_data=ratings_data,
                processed_manifest_path=processed_manifest_path,
                requested_split_family=requested_split_family,
                split_config=split_config,
                inner_validation_seed=inner_validation_seed,
                services=services,
            ),
            use_cache=split_cache_policy.effective_use_cache,
        )
        split_stage["cache_status"] = split_result.metadata.cache_status
        split_stage["cache_hit"] = split_result.metadata.cache_status == "hit"
        split_stage["cache_path"] = repo_path_string(
            split_result.metadata.cache_manifest_path,
            repo_root=repo_root,
        )

    split = split_result.split
    with stage_profiler.stage(
        "load_train_ratings",
        metadata=ratings_stage_metadata(split.train),
    ):
        train_data = split.train

    validation_data = None
    if split.validation is not None:
        with stage_profiler.stage(
            "load_validation_ratings",
            metadata=ratings_stage_metadata(split.validation),
        ):
            validation_data = split.validation

    with stage_profiler.stage(
        "load_test_ratings",
        metadata=ratings_stage_metadata(split.test),
    ):
        test_data = split.test

    return ResolvedSplitBundle(
        ratings_data=ratings_data,
        split_result=split_result,
        split=split,
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        split_id_for_cache=split_id_for_cache,
    )


def build_unified_split(
    *,
    ratings_data,
    processed_manifest_path: Path,
    requested_split_family: str,
    split_config: SplitConfig,
    inner_validation_seed: int | None,
    services: Any,
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


def ratings_stage_metadata(data) -> dict[str, Any]:
    return {
        "rows": len(data),
        "n_users": int(data.n_users),
        "n_items": int(data.n_items),
    }
