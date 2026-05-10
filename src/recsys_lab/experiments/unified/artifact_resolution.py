from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from recsys_lab.clustering import load_or_build_cluster_artifacts, load_or_build_user_cluster_history_index
from recsys_lab.data.splitters import RatingsSplit
from recsys_lab.data.training_index_cache import (
    UserExplicitFeedbackIndexCacheResult,
    UserHistoryIndexCacheResult,
    load_or_build_user_explicit_feedback_index,
    load_or_build_user_history_index,
)
from recsys_lab.experiments.performance import StageProfiler
from recsys_lab.experiments.unified.context import FitArtifactResolution
from recsys_lab.models.config_schemas import ClusteringSchema, ModelProfileSchema
from recsys_lab.models.registry import FitArtifacts, ModelAdapter
from recsys_lab.utils.paths import repo_path_string


def resolve_fit_artifacts(
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
) -> FitArtifactResolution:
    requirements = adapter.requirements
    history_result: UserHistoryIndexCacheResult | None = None
    explicit_result: UserExplicitFeedbackIndexCacheResult | None = None
    cluster_artifact_result = None
    cluster_history_result = None
    clustering_seconds = 0.0

    if requirements.needs_cluster_artifacts:
        if induction_config is None:
            raise ValueError(f"{adapter.name} requires an induction config for cluster artifacts")
        clustering_config = clustering_config_from_profile(model_profile)
        clustering_started = perf_counter()
        with stage_profiler.stage(
            "build_cluster_artifacts",
            metadata={
                "required_by_model": True,
                "artifact_name": "cluster_artifacts",
                "algorithm": clustering_config.algorithm,
                "n_user_clusters": clustering_config.n_user_clusters,
                "n_item_clusters": clustering_config.n_item_clusters,
                "alpha": float(clustering_config.alpha),
                "kmeans_n_init": clustering_config.kmeans_n_init,
                "cluster_artifact_cache_enabled": use_cluster_artifact_cache,
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
            cluster_stage.update(
                cache_metadata_payload(
                    cache_status=cluster_artifact_result.metadata.cache_status,
                    cache_manifest_path=cluster_artifact_result.metadata.cache_manifest_path,
                    repo_root=root,
                )
            )
            cluster_stage["cache_key"] = cluster_artifact_result.metadata.cache_key
        clustering_seconds = perf_counter() - clustering_started

    if requirements.needs_implicit_history or requirements.needs_explicit_feedback:
        with stage_profiler.stage(
            "build_training_indices",
            metadata={
                "needs_user_history_index": requirements.needs_implicit_history,
                "needs_explicit_feedback_index": requirements.needs_explicit_feedback,
                "cache_enabled": use_training_index_cache,
            },
        ):
            if requirements.needs_implicit_history:
                with stage_profiler.stage(
                    "build_user_history_index",
                    metadata={
                        "required_by_model": True,
                        "artifact_name": "user_history_index",
                        "cache_enabled": use_training_index_cache,
                        "split_id": split_id_for_cache,
                    },
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
                    index_stage.update(
                        cache_metadata_payload(
                            cache_status=history_result.metadata.cache_status,
                            cache_manifest_path=history_result.metadata.cache_manifest_path,
                            repo_root=root,
                        )
                    )
                    index_stage["rows"] = len(split.train)
                    index_stage["n_users"] = split.train.n_users
                    index_stage["n_items"] = split.train.n_items

            if requirements.needs_explicit_feedback:
                with stage_profiler.stage(
                    "build_explicit_feedback_index",
                    metadata={
                        "required_by_model": True,
                        "artifact_name": "explicit_feedback_index",
                        "cache_enabled": use_training_index_cache,
                        "split_id": split_id_for_cache,
                    },
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
                    index_stage.update(
                        cache_metadata_payload(
                            cache_status=explicit_result.metadata.cache_status,
                            cache_manifest_path=explicit_result.metadata.cache_manifest_path,
                            repo_root=root,
                        )
                    )
                    index_stage["rows"] = len(split.train)
                    index_stage["n_users"] = split.train.n_users
                    index_stage["n_items"] = split.train.n_items

    if requirements.needs_user_cluster_history:
        if history_result is None or cluster_artifact_result is None:
            raise ValueError("user-cluster history requires user history and cluster artifacts")
        n_item_clusters = int(cluster_artifact_result.artifacts.r_star_counts.shape[1])
        with stage_profiler.stage(
            "build_user_cluster_history_index",
            metadata={
                "required_by_model": True,
                "artifact_name": "user_cluster_history_index",
                "n_item_clusters": n_item_clusters,
                "cluster_artifact_cache_enabled": use_cluster_artifact_cache,
            },
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
            cluster_history_stage.update(
                cache_metadata_payload(
                    cache_status=cluster_history_result.metadata.cache_status,
                    cache_manifest_path=cluster_history_result.metadata.cache_manifest_path,
                    repo_root=root,
                )
            )
            cluster_history_stage["cache_key"] = cluster_history_result.metadata.cache_key

    return FitArtifactResolution(
        artifacts=FitArtifacts(
            user_history_index=None if history_result is None else history_result.index,
            explicit_feedback_index=None if explicit_result is None else explicit_result.index,
            user_cluster_history_index=None if cluster_history_result is None else cluster_history_result.index,
            cluster_artifacts=None if cluster_artifact_result is None else cluster_artifact_result.artifacts,
        ),
        user_history_result=history_result,
        explicit_feedback_result=explicit_result,
        cluster_artifact_result=cluster_artifact_result,
        cluster_history_result=cluster_history_result,
        induction_config=induction_config,
        cluster_induction_wall_clock_seconds=clustering_seconds,
    )


def cache_hit(cache_status: str) -> bool:
    return cache_status == "hit"


def cache_metadata_payload(
    *,
    cache_status: str,
    cache_manifest_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "cache_status": cache_status,
        "cache_hit": cache_hit(cache_status),
        "cache_path": repo_path_string(cache_manifest_path, repo_root=repo_root),
    }


def clustering_config_from_profile(model_profile: ModelProfileSchema) -> ClusteringSchema:
    clustering = getattr(model_profile, "clustering", None)
    if not isinstance(clustering, ClusteringSchema):
        raise ValueError("CB model profile requires a validated clustering block")
    return clustering
