from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np

from recsys_lab.clustering.latent_kmeans import ClusterArtifacts, ClusterInductionProfile, induce_train_only_clusters
from recsys_lab.data.histories import (
    UserClusterCountIndex,
    UserHistoryIndex,
    build_user_cluster_count_index,
    validate_user_cluster_count_index,
)
from recsys_lab.data.processed import RatingsData
from recsys_lab.data.training_index_cache import (
    HISTORY_INDEX_DTYPE,
    HISTORY_LAYOUT_VERSION,
    RatingsDataFingerprint,
    fingerprint_ratings_data,
    resolve_training_index_cache_root,
)
from recsys_lab.models.biased_mf import BiasedMFConfig
from recsys_lab.utils.atomic_io import atomic_save_array, atomic_write_json
from recsys_lab.utils.paths import repo_path_string

MMapMode = Literal["r+", "r", "w+", "c"]


@dataclass(frozen=True, slots=True)
class ClusterCacheMetadata:
    cache_status: str
    cache_manifest_path: Path
    cache_root: Path
    train_fingerprint: RatingsDataFingerprint
    cache_key: str
    cache_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class ClusterStageTimings:
    cluster_total_seconds: float = 0.0
    cluster_cache_read_seconds: float = 0.0
    cluster_cache_write_seconds: float = 0.0
    induction_fit_seconds: float = 0.0
    induction_predict_seconds: float = 0.0
    induction_train_rmse_seconds: float = 0.0
    user_kmeans_seconds: float = 0.0
    item_kmeans_seconds: float = 0.0
    r_star_seconds: float = 0.0
    cluster_artifact_validation_seconds: float = 0.0
    user_cluster_history_total_seconds: float = 0.0
    user_cluster_history_cache_read_seconds: float = 0.0
    user_cluster_history_cache_write_seconds: float = 0.0
    user_cluster_history_build_seconds: float = 0.0
    user_cluster_history_validation_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class ClusterArtifactProfile:
    timings: ClusterStageTimings
    cluster_cache_status: str | None = None
    user_cluster_history_cache_status: str | None = None
    dataset_short_name: str | None = None
    split_family: str | None = None
    split_id: str | None = None
    model: str | None = None
    n_users: int | None = None
    n_items: int | None = None
    train_rows: int | None = None
    n_user_clusters: int | None = None
    n_item_clusters: int | None = None
    algorithm: str | None = None
    kmeans_n_init: int | None = None
    induction_seed: int | None = None
    induction_latent_dim: int | None = None
    induction_epochs: int | None = None
    induction_dtype: str | None = None
    cluster_cache_key: str | None = None
    cluster_cache_fingerprint_sha256: str | None = None
    user_cluster_history_cache_key: str | None = None
    user_cluster_history_cache_fingerprint_sha256: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            **asdict(self.timings),
            "cluster_cache_status": self.cluster_cache_status,
            "user_cluster_history_cache_status": self.user_cluster_history_cache_status,
            "dataset_short_name": self.dataset_short_name,
            "split_family": self.split_family,
            "split_id": self.split_id,
            "model": self.model,
            "n_users": self.n_users,
            "n_items": self.n_items,
            "train_rows": self.train_rows,
            "n_user_clusters": self.n_user_clusters,
            "n_item_clusters": self.n_item_clusters,
            "algorithm": self.algorithm,
            "kmeans_n_init": self.kmeans_n_init,
            "induction_seed": self.induction_seed,
            "induction_latent_dim": self.induction_latent_dim,
            "induction_epochs": self.induction_epochs,
            "induction_dtype": self.induction_dtype,
            "cluster_cache_key": self.cluster_cache_key,
            "cluster_cache_fingerprint_sha256": self.cluster_cache_fingerprint_sha256,
            "user_cluster_history_cache_key": self.user_cluster_history_cache_key,
            "user_cluster_history_cache_fingerprint_sha256": self.user_cluster_history_cache_fingerprint_sha256,
        }


@dataclass(frozen=True, slots=True)
class ClusterArtifactsCacheResult:
    artifacts: ClusterArtifacts
    metadata: ClusterCacheMetadata
    profile: ClusterArtifactProfile | None = None


@dataclass(frozen=True, slots=True)
class UserClusterHistoryCacheResult:
    index: UserClusterCountIndex
    metadata: ClusterCacheMetadata
    profile: ClusterArtifactProfile | None = None


def load_or_build_cluster_artifacts(
    *,
    data: RatingsData,
    induction_config: BiasedMFConfig,
    n_user_clusters: int,
    n_item_clusters: int,
    algorithm: str,
    kmeans_n_init: int,
    dataset_short_name: str,
    split_family: str,
    split_id: str,
    processed_manifest_path: Path,
    repo_root: Path,
    runtime_config_payload: dict[str, Any],
    use_cache: bool = True,
    mmap_mode: MMapMode | None = "r",
    model: str | None = None,
) -> ClusterArtifactsCacheResult:
    total_started = perf_counter()
    fingerprint = fingerprint_ratings_data(data)
    cache_root = resolve_training_index_cache_root(
        repo_root=repo_root,
        runtime_config_payload=runtime_config_payload,
    )
    identity_payload = _cluster_identity_payload(
        dataset_short_name=dataset_short_name,
        split_family=split_family,
        split_id=split_id,
        processed_manifest_path=processed_manifest_path,
        repo_root=repo_root,
        fingerprint=fingerprint,
        induction_config=induction_config,
        n_users=data.n_users,
        n_items=data.n_items,
        n_user_clusters=n_user_clusters,
        n_item_clusters=n_item_clusters,
        algorithm=algorithm,
        kmeans_n_init=kmeans_n_init,
    )
    cache_fingerprint_sha256 = _stable_sha256(identity_payload)
    cache_key = cache_fingerprint_sha256[:16]
    cache_dir = _cache_dir(cache_root=cache_root) / cache_key
    manifest_path = cache_dir / "manifest.json"
    read_seconds = 0.0
    write_seconds = 0.0
    validation_seconds = 0.0

    if use_cache:
        read_started = perf_counter()
        cached_result = _try_load_cluster_artifacts(
            manifest_path=manifest_path,
            expected_identity=identity_payload,
            expected_cache_fingerprint_sha256=cache_fingerprint_sha256,
            mmap_mode=mmap_mode,
        )
        if cached_result is not None:
            cached_artifacts, validation_seconds = cached_result
            read_seconds = max(float(perf_counter() - read_started) - float(validation_seconds), 0.0)
            metadata = ClusterCacheMetadata(
                cache_status="hit",
                cache_manifest_path=manifest_path,
                cache_root=cache_root,
                train_fingerprint=fingerprint,
                cache_key=cache_key,
                cache_fingerprint_sha256=cache_fingerprint_sha256,
            )
            return ClusterArtifactsCacheResult(
                artifacts=cached_artifacts,
                metadata=metadata,
                profile=_cluster_profile(
                    timings=ClusterStageTimings(
                        cluster_total_seconds=float(perf_counter() - total_started),
                        cluster_cache_read_seconds=read_seconds,
                        cluster_artifact_validation_seconds=float(validation_seconds),
                    ),
                    metadata=metadata,
                    data=data,
                    dataset_short_name=dataset_short_name,
                    split_family=split_family,
                    split_id=split_id,
                    induction_config=induction_config,
                    n_user_clusters=n_user_clusters,
                    n_item_clusters=n_item_clusters,
                    algorithm=algorithm,
                    kmeans_n_init=kmeans_n_init,
                    model=model,
                ),
            )
        read_seconds = perf_counter() - read_started

    artifacts = induce_train_only_clusters(
        data,
        induction_config=induction_config,
        n_user_clusters=n_user_clusters,
        n_item_clusters=n_item_clusters,
        algorithm=algorithm,
        kmeans_n_init=kmeans_n_init,
    )
    validation_started = perf_counter()
    _validate_cluster_artifacts(artifacts, identity=identity_payload)
    validation_seconds = perf_counter() - validation_started
    if use_cache:
        write_started = perf_counter()
        _write_cluster_artifacts_cache(
            artifacts=artifacts,
            manifest_path=manifest_path,
            identity_payload=identity_payload,
            cache_fingerprint_sha256=cache_fingerprint_sha256,
        )
        write_seconds = perf_counter() - write_started
    metadata = ClusterCacheMetadata(
        cache_status="miss" if use_cache else "disabled",
        cache_manifest_path=manifest_path,
        cache_root=cache_root,
        train_fingerprint=fingerprint,
        cache_key=cache_key,
        cache_fingerprint_sha256=cache_fingerprint_sha256,
    )
    induction_profile = artifacts.induction_profile or ClusterInductionProfile()
    return ClusterArtifactsCacheResult(
        artifacts=artifacts,
        metadata=metadata,
        profile=_cluster_profile(
            timings=ClusterStageTimings(
                cluster_total_seconds=float(perf_counter() - total_started),
                cluster_cache_read_seconds=float(read_seconds),
                cluster_cache_write_seconds=float(write_seconds),
                induction_fit_seconds=induction_profile.induction_fit_seconds,
                induction_predict_seconds=induction_profile.induction_predict_seconds,
                induction_train_rmse_seconds=induction_profile.induction_train_rmse_seconds,
                user_kmeans_seconds=induction_profile.user_kmeans_seconds,
                item_kmeans_seconds=induction_profile.item_kmeans_seconds,
                r_star_seconds=induction_profile.r_star_seconds,
                cluster_artifact_validation_seconds=float(validation_seconds),
            ),
            metadata=metadata,
            data=data,
            dataset_short_name=dataset_short_name,
            split_family=split_family,
            split_id=split_id,
            induction_config=induction_config,
            n_user_clusters=n_user_clusters,
            n_item_clusters=n_item_clusters,
            algorithm=algorithm,
            kmeans_n_init=kmeans_n_init,
            model=model,
        ),
    )


def load_or_build_user_cluster_history_index(
    *,
    history_index: UserHistoryIndex,
    item_clusters: np.ndarray,
    n_clusters: int,
    dataset_short_name: str,
    split_family: str,
    split_id: str,
    processed_manifest_path: Path,
    repo_root: Path,
    runtime_config_payload: dict[str, Any],
    train_fingerprint: RatingsDataFingerprint,
    cluster_cache_key: str,
    cluster_cache_fingerprint_sha256: str,
    use_cache: bool = True,
    mmap_mode: MMapMode | None = "r",
    model: str | None = None,
) -> UserClusterHistoryCacheResult:
    total_started = perf_counter()
    cache_root = resolve_training_index_cache_root(
        repo_root=repo_root,
        runtime_config_payload=runtime_config_payload,
    )
    item_cluster_fingerprint_sha256 = _array_sha256(np.asarray(item_clusters))
    identity_payload = _cluster_history_identity_payload(
        dataset_short_name=dataset_short_name,
        split_family=split_family,
        split_id=split_id,
        processed_manifest_path=processed_manifest_path,
        repo_root=repo_root,
        train_fingerprint=train_fingerprint,
        n_users=history_index.counts.shape[0],
        n_clusters=n_clusters,
        cluster_cache_key=cluster_cache_key,
        cluster_cache_fingerprint_sha256=cluster_cache_fingerprint_sha256,
        item_cluster_fingerprint_sha256=item_cluster_fingerprint_sha256,
    )
    cache_fingerprint_sha256 = _stable_sha256(identity_payload)
    cache_key = cache_fingerprint_sha256[:16]
    cache_dir = _cache_dir(cache_root=cache_root) / cluster_cache_key / "uch" / HISTORY_LAYOUT_VERSION / cache_key
    manifest_path = cache_dir / "manifest.json"
    read_seconds = 0.0
    write_seconds = 0.0
    build_seconds = 0.0
    validation_seconds = 0.0

    if use_cache:
        read_started = perf_counter()
        cached_result = _try_load_user_cluster_history_index(
            manifest_path=manifest_path,
            expected_identity=identity_payload,
            expected_cache_fingerprint_sha256=cache_fingerprint_sha256,
            mmap_mode=mmap_mode,
            n_users=history_index.counts.shape[0],
            n_clusters=n_clusters,
        )
        if cached_result is not None:
            cached_index, validation_seconds = cached_result
            read_seconds = max(float(perf_counter() - read_started) - float(validation_seconds), 0.0)
            metadata = ClusterCacheMetadata(
                cache_status="hit",
                cache_manifest_path=manifest_path,
                cache_root=cache_root,
                train_fingerprint=train_fingerprint,
                cache_key=cache_key,
                cache_fingerprint_sha256=cache_fingerprint_sha256,
            )
            return UserClusterHistoryCacheResult(
                index=cached_index,
                metadata=metadata,
                profile=_cluster_history_profile(
                    timings=ClusterStageTimings(
                        user_cluster_history_total_seconds=float(perf_counter() - total_started),
                        user_cluster_history_cache_read_seconds=read_seconds,
                        user_cluster_history_validation_seconds=float(validation_seconds),
                    ),
                    metadata=metadata,
                    dataset_short_name=dataset_short_name,
                    split_family=split_family,
                    split_id=split_id,
                    train_fingerprint=train_fingerprint,
                    n_users=history_index.counts.shape[0],
                    n_clusters=n_clusters,
                    cluster_cache_key=cluster_cache_key,
                    cluster_cache_fingerprint_sha256=cluster_cache_fingerprint_sha256,
                    model=model,
                ),
            )
        read_seconds = perf_counter() - read_started

    build_started = perf_counter()
    index = build_user_cluster_count_index(history_index, item_clusters, n_clusters=n_clusters)
    build_seconds = perf_counter() - build_started
    validation_started = perf_counter()
    validate_user_cluster_count_index(index, n_users=history_index.counts.shape[0], n_clusters=n_clusters)
    validation_seconds = perf_counter() - validation_started
    if use_cache:
        write_started = perf_counter()
        _write_user_cluster_history_cache(
            index=index,
            manifest_path=manifest_path,
            identity_payload=identity_payload,
            cache_fingerprint_sha256=cache_fingerprint_sha256,
        )
        write_seconds = perf_counter() - write_started
    metadata = ClusterCacheMetadata(
        cache_status="miss" if use_cache else "disabled",
        cache_manifest_path=manifest_path,
        cache_root=cache_root,
        train_fingerprint=train_fingerprint,
        cache_key=cache_key,
        cache_fingerprint_sha256=cache_fingerprint_sha256,
    )
    return UserClusterHistoryCacheResult(
        index=index,
        metadata=metadata,
        profile=_cluster_history_profile(
            timings=ClusterStageTimings(
                user_cluster_history_total_seconds=float(perf_counter() - total_started),
                user_cluster_history_cache_read_seconds=float(read_seconds),
                user_cluster_history_cache_write_seconds=float(write_seconds),
                user_cluster_history_build_seconds=float(build_seconds),
                user_cluster_history_validation_seconds=float(validation_seconds),
            ),
            metadata=metadata,
            dataset_short_name=dataset_short_name,
            split_family=split_family,
            split_id=split_id,
            train_fingerprint=train_fingerprint,
            n_users=history_index.counts.shape[0],
            n_clusters=n_clusters,
            cluster_cache_key=cluster_cache_key,
            cluster_cache_fingerprint_sha256=cluster_cache_fingerprint_sha256,
            model=model,
        ),
    )


def _cache_dir(*, cache_root: Path) -> Path:
    return cache_root / "cb_clusters"


def _cluster_profile(
    *,
    timings: ClusterStageTimings,
    metadata: ClusterCacheMetadata,
    data: RatingsData,
    dataset_short_name: str,
    split_family: str,
    split_id: str,
    induction_config: BiasedMFConfig,
    n_user_clusters: int,
    n_item_clusters: int,
    algorithm: str,
    kmeans_n_init: int,
    model: str | None,
) -> ClusterArtifactProfile:
    return ClusterArtifactProfile(
        timings=timings,
        cluster_cache_status=metadata.cache_status,
        dataset_short_name=dataset_short_name,
        split_family=split_family,
        split_id=split_id,
        model=model,
        n_users=int(data.n_users),
        n_items=int(data.n_items),
        train_rows=int(metadata.train_fingerprint.row_count),
        n_user_clusters=int(n_user_clusters),
        n_item_clusters=int(n_item_clusters),
        algorithm=algorithm,
        kmeans_n_init=int(kmeans_n_init),
        induction_seed=int(induction_config.seed),
        induction_latent_dim=int(induction_config.latent_dim),
        induction_epochs=int(induction_config.epochs),
        induction_dtype=str(induction_config.dtype),
        cluster_cache_key=metadata.cache_key,
        cluster_cache_fingerprint_sha256=metadata.cache_fingerprint_sha256,
    )


def _cluster_history_profile(
    *,
    timings: ClusterStageTimings,
    metadata: ClusterCacheMetadata,
    dataset_short_name: str,
    split_family: str,
    split_id: str,
    train_fingerprint: RatingsDataFingerprint,
    n_users: int,
    n_clusters: int,
    cluster_cache_key: str,
    cluster_cache_fingerprint_sha256: str,
    model: str | None,
) -> ClusterArtifactProfile:
    return ClusterArtifactProfile(
        timings=timings,
        user_cluster_history_cache_status=metadata.cache_status,
        dataset_short_name=dataset_short_name,
        split_family=split_family,
        split_id=split_id,
        model=model,
        n_users=int(n_users),
        train_rows=int(train_fingerprint.row_count),
        n_item_clusters=int(n_clusters),
        cluster_cache_key=cluster_cache_key,
        cluster_cache_fingerprint_sha256=cluster_cache_fingerprint_sha256,
        user_cluster_history_cache_key=metadata.cache_key,
        user_cluster_history_cache_fingerprint_sha256=metadata.cache_fingerprint_sha256,
    )


def _cluster_identity_payload(
    *,
    dataset_short_name: str,
    split_family: str,
    split_id: str,
    processed_manifest_path: Path,
    repo_root: Path,
    fingerprint: RatingsDataFingerprint,
    induction_config: BiasedMFConfig,
    n_users: int,
    n_items: int,
    n_user_clusters: int,
    n_item_clusters: int,
    algorithm: str,
    kmeans_n_init: int,
) -> dict[str, Any]:
    return {
        "manifest_version": "v1",
        "kind": "cluster_artifact_cache",
        "dataset": {
            "short_name": dataset_short_name,
            "split_family": split_family,
            "split_id": split_id,
            "processed_manifest_ref": repo_path_string(processed_manifest_path, repo_root=repo_root),
        },
        "shape": {
            "n_users": int(n_users),
            "n_items": int(n_items),
            "train_rows": int(fingerprint.row_count),
        },
        "fingerprint": {
            "train_sha256": fingerprint.sha256,
        },
        "induction": {
            "model": "biased_mf",
            "config": asdict(induction_config),
        },
        "clustering": {
            "algorithm": algorithm,
            "kmeans_n_init": int(kmeans_n_init),
            "n_user_clusters": int(n_user_clusters),
            "n_item_clusters": int(n_item_clusters),
            "r_star_role": "diagnostic_only",
            "train_only_assignments": True,
        },
    }


def _cluster_history_identity_payload(
    *,
    dataset_short_name: str,
    split_family: str,
    split_id: str,
    processed_manifest_path: Path,
    repo_root: Path,
    train_fingerprint: RatingsDataFingerprint,
    n_users: int,
    n_clusters: int,
    cluster_cache_key: str,
    cluster_cache_fingerprint_sha256: str,
    item_cluster_fingerprint_sha256: str,
) -> dict[str, Any]:
    return {
        "manifest_version": "v1",
        "kind": "user_cluster_history_cache",
        "layout": {
            "layout_version": HISTORY_LAYOUT_VERSION,
            "index_dtype": HISTORY_INDEX_DTYPE,
            "count_dtype": HISTORY_INDEX_DTYPE,
        },
        "dataset": {
            "short_name": dataset_short_name,
            "split_family": split_family,
            "split_id": split_id,
            "processed_manifest_ref": repo_path_string(processed_manifest_path, repo_root=repo_root),
        },
        "shape": {
            "n_users": int(n_users),
            "n_clusters": int(n_clusters),
            "train_rows": int(train_fingerprint.row_count),
        },
        "fingerprint": {
            "train_sha256": train_fingerprint.sha256,
            "cluster_cache_key": cluster_cache_key,
            "cluster_cache_sha256": cluster_cache_fingerprint_sha256,
            "item_cluster_sha256": item_cluster_fingerprint_sha256,
        },
    }


def _try_load_cluster_artifacts(
    *,
    manifest_path: Path,
    expected_identity: dict[str, Any],
    expected_cache_fingerprint_sha256: str,
    mmap_mode: MMapMode | None,
) -> tuple[ClusterArtifacts, float] | None:
    try:
        payload = _load_and_validate_manifest(
            manifest_path=manifest_path,
            expected_identity=expected_identity,
            expected_cache_fingerprint_sha256=expected_cache_fingerprint_sha256,
        )
        artifacts = payload["artifacts"]
        cluster_artifacts = ClusterArtifacts(
            user_clusters=np.load(_artifact_path(manifest_path, artifacts["user_clusters_npy"]), mmap_mode=mmap_mode),
            item_clusters=np.load(_artifact_path(manifest_path, artifacts["item_clusters_npy"]), mmap_mode=mmap_mode),
            user_cluster_sizes=np.load(
                _artifact_path(manifest_path, artifacts["user_cluster_sizes_npy"]),
                mmap_mode=mmap_mode,
            ),
            item_cluster_sizes=np.load(
                _artifact_path(manifest_path, artifacts["item_cluster_sizes_npy"]),
                mmap_mode=mmap_mode,
            ),
            r_star_means=np.load(_artifact_path(manifest_path, artifacts["r_star_means_npy"]), mmap_mode=mmap_mode),
            r_star_counts=np.load(_artifact_path(manifest_path, artifacts["r_star_counts_npy"]), mmap_mode=mmap_mode),
            induction_train_rmse=float(payload["diagnostics"]["induction_train_rmse"]),
            user_kmeans_inertia=float(payload["diagnostics"]["user_kmeans_inertia"]),
            item_kmeans_inertia=float(payload["diagnostics"]["item_kmeans_inertia"]),
        )
        validation_started = perf_counter()
        _validate_cluster_artifacts(cluster_artifacts, identity=expected_identity)
        validation_seconds = perf_counter() - validation_started
        return cluster_artifacts, float(validation_seconds)
    except Exception:
        return None


def _try_load_user_cluster_history_index(
    *,
    manifest_path: Path,
    expected_identity: dict[str, Any],
    expected_cache_fingerprint_sha256: str,
    mmap_mode: MMapMode | None,
    n_users: int,
    n_clusters: int,
) -> tuple[UserClusterCountIndex, float] | None:
    try:
        payload = _load_and_validate_manifest(
            manifest_path=manifest_path,
            expected_identity=expected_identity,
            expected_cache_fingerprint_sha256=expected_cache_fingerprint_sha256,
        )
        artifacts = payload["artifacts"]
        index = UserClusterCountIndex(
            indptr=np.load(_artifact_path(manifest_path, artifacts["indptr_npy"]), mmap_mode=mmap_mode),
            cluster_ids=np.load(_artifact_path(manifest_path, artifacts["cluster_ids_npy"]), mmap_mode=mmap_mode),
            counts=np.load(_artifact_path(manifest_path, artifacts["counts_npy"]), mmap_mode=mmap_mode),
        )
        validation_started = perf_counter()
        validate_user_cluster_count_index(index, n_users=n_users, n_clusters=n_clusters)
        validation_seconds = perf_counter() - validation_started
        return index, float(validation_seconds)
    except Exception:
        return None


def _load_and_validate_manifest(
    *,
    manifest_path: Path,
    expected_identity: dict[str, Any],
    expected_cache_fingerprint_sha256: str,
) -> dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("identity") != expected_identity:
        raise ValueError("cluster cache identity mismatch")
    if payload.get("cache_fingerprint_sha256") != expected_cache_fingerprint_sha256:
        raise ValueError("cluster cache fingerprint mismatch")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError("cluster cache artifacts must be an object")
    return payload


def _write_cluster_artifacts_cache(
    *,
    artifacts: ClusterArtifacts,
    manifest_path: Path,
    identity_payload: dict[str, Any],
    cache_fingerprint_sha256: str,
) -> None:
    cache_dir = manifest_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "user_clusters_npy": "user_clusters.npy",
        "item_clusters_npy": "item_clusters.npy",
        "user_cluster_sizes_npy": "user_cluster_sizes.npy",
        "item_cluster_sizes_npy": "item_cluster_sizes.npy",
        "r_star_means_npy": "r_star_means.npy",
        "r_star_counts_npy": "r_star_counts.npy",
    }
    atomic_save_array(cache_dir / artifact_paths["user_clusters_npy"], artifacts.user_clusters)
    atomic_save_array(cache_dir / artifact_paths["item_clusters_npy"], artifacts.item_clusters)
    atomic_save_array(cache_dir / artifact_paths["user_cluster_sizes_npy"], artifacts.user_cluster_sizes)
    atomic_save_array(cache_dir / artifact_paths["item_cluster_sizes_npy"], artifacts.item_cluster_sizes)
    atomic_save_array(cache_dir / artifact_paths["r_star_means_npy"], artifacts.r_star_means)
    atomic_save_array(cache_dir / artifact_paths["r_star_counts_npy"], artifacts.r_star_counts)
    atomic_write_json(
        manifest_path,
        {
            "identity": identity_payload,
            "cache_fingerprint_sha256": cache_fingerprint_sha256,
            "diagnostics": {
                "induction_train_rmse": float(artifacts.induction_train_rmse),
                "user_kmeans_inertia": float(artifacts.user_kmeans_inertia),
                "item_kmeans_inertia": float(artifacts.item_kmeans_inertia),
            },
            "artifacts": artifact_paths,
        },
    )


def _write_user_cluster_history_cache(
    *,
    index: UserClusterCountIndex,
    manifest_path: Path,
    identity_payload: dict[str, Any],
    cache_fingerprint_sha256: str,
) -> None:
    cache_dir = manifest_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "indptr_npy": "indptr.npy",
        "cluster_ids_npy": "cluster_ids.npy",
        "counts_npy": "counts.npy",
    }
    atomic_save_array(cache_dir / artifact_paths["indptr_npy"], index.indptr)
    atomic_save_array(cache_dir / artifact_paths["cluster_ids_npy"], index.cluster_ids)
    atomic_save_array(cache_dir / artifact_paths["counts_npy"], index.counts)
    atomic_write_json(
        manifest_path,
        {
            "identity": identity_payload,
            "cache_fingerprint_sha256": cache_fingerprint_sha256,
            "artifacts": artifact_paths,
        },
    )


def _validate_cluster_artifacts(artifacts: ClusterArtifacts, *, identity: dict[str, Any]) -> None:
    shape = identity["shape"]
    clustering = identity["clustering"]
    n_users = int(shape["n_users"])
    n_items = int(shape["n_items"])
    n_user_clusters = int(clustering["n_user_clusters"])
    n_item_clusters = int(clustering["n_item_clusters"])
    if artifacts.user_clusters.shape != (n_users,):
        raise ValueError("cached user_clusters shape mismatch")
    if artifacts.item_clusters.shape != (n_items,):
        raise ValueError("cached item_clusters shape mismatch")
    if artifacts.user_cluster_sizes.shape != (n_user_clusters,):
        raise ValueError("cached user_cluster_sizes shape mismatch")
    if artifacts.item_cluster_sizes.shape != (n_item_clusters,):
        raise ValueError("cached item_cluster_sizes shape mismatch")
    if artifacts.r_star_means.shape != (n_user_clusters, n_item_clusters):
        raise ValueError("cached r_star_means shape mismatch")
    if artifacts.r_star_counts.shape != (n_user_clusters, n_item_clusters):
        raise ValueError("cached r_star_counts shape mismatch")


def _artifact_path(manifest_path: Path, artifact_ref: str) -> Path:
    path = Path(artifact_ref)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _stable_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(contiguous.dtype.str.encode("ascii"))
    hasher.update(str(contiguous.shape).encode("ascii"))
    hasher.update(memoryview(cast(Any, contiguous)).cast("B"))
    return hasher.hexdigest()
