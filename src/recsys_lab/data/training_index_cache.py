from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from recsys_lab.data.histories import (
    UserExplicitFeedbackIndex,
    UserHistoryIndex,
    build_user_explicit_feedback_index,
    build_user_history_index,
    validate_user_explicit_feedback_index,
    validate_user_history_index,
)
from recsys_lab.data.processed import RatingsData
from recsys_lab.utils.atomic_io import atomic_save_array, atomic_write_json
from recsys_lab.utils.paths import repo_path_string

MMapMode = Literal["r+", "r", "w+", "c"]


@dataclass(frozen=True, slots=True)
class RatingsDataFingerprint:
    row_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class TrainingIndexCacheMetadata:
    cache_status: str
    cache_manifest_path: Path
    cache_root: Path
    train_fingerprint: RatingsDataFingerprint


@dataclass(frozen=True, slots=True)
class UserHistoryIndexCacheResult:
    index: UserHistoryIndex
    metadata: TrainingIndexCacheMetadata


@dataclass(frozen=True, slots=True)
class UserExplicitFeedbackIndexCacheResult:
    index: UserExplicitFeedbackIndex
    metadata: TrainingIndexCacheMetadata


def resolve_training_index_cache_root(
    *,
    repo_root: Path,
    runtime_config_payload: dict[str, Any],
) -> Path:
    env_override = os.environ.get("RECSYS_CACHE_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()

    runtime = runtime_config_payload.get("runtime", {})
    configured_root = Path(str(runtime.get("cache_root", "artifacts/local")))
    if not configured_root.is_absolute():
        configured_root = (repo_root / configured_root).resolve()
    return configured_root


def fingerprint_ratings_data(data: RatingsData) -> RatingsDataFingerprint:
    hasher = hashlib.sha256()
    if data.row_indices is None:
        arrays = (data.base_user_ids, data.base_item_ids, data.base_ratings)
    else:
        arrays = (
            data.base_user_ids[data.row_indices],
            data.base_item_ids[data.row_indices],
            data.base_ratings[data.row_indices],
        )
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        hasher.update(contiguous.dtype.str.encode("ascii"))
        hasher.update(str(contiguous.shape).encode("ascii"))
        hasher.update(memoryview(cast(Any, contiguous)).cast("B"))
    return RatingsDataFingerprint(
        row_count=len(data),
        sha256=hasher.hexdigest(),
    )


def load_or_build_user_history_index(
    *,
    data: RatingsData,
    dtype: str,
    dataset_short_name: str,
    split_family: str,
    split_id: str,
    processed_manifest_path: Path,
    repo_root: Path,
    runtime_config_payload: dict[str, Any],
    use_cache: bool = True,
    mmap_mode: MMapMode | None = "r",
) -> UserHistoryIndexCacheResult:
    return _load_or_build_user_history_index(
        data=data,
        dtype=dtype,
        dataset_short_name=dataset_short_name,
        split_family=split_family,
        split_id=split_id,
        processed_manifest_path=processed_manifest_path,
        repo_root=repo_root,
        runtime_config_payload=runtime_config_payload,
        use_cache=use_cache,
        mmap_mode=mmap_mode,
    )


def load_or_build_user_explicit_feedback_index(
    *,
    data: RatingsData,
    dtype: str,
    dataset_short_name: str,
    split_family: str,
    split_id: str,
    processed_manifest_path: Path,
    repo_root: Path,
    runtime_config_payload: dict[str, Any],
    use_cache: bool = True,
    mmap_mode: MMapMode | None = "r",
) -> UserExplicitFeedbackIndexCacheResult:
    return _load_or_build_user_explicit_feedback_index(
        data=data,
        dtype=dtype,
        dataset_short_name=dataset_short_name,
        split_family=split_family,
        split_id=split_id,
        processed_manifest_path=processed_manifest_path,
        repo_root=repo_root,
        runtime_config_payload=runtime_config_payload,
        use_cache=use_cache,
        mmap_mode=mmap_mode,
    )


def _load_or_build_user_history_index(
    *,
    data: RatingsData,
    dtype: str,
    dataset_short_name: str,
    split_family: str,
    split_id: str,
    processed_manifest_path: Path,
    repo_root: Path,
    runtime_config_payload: dict[str, Any],
    use_cache: bool,
    mmap_mode: MMapMode | None,
) -> UserHistoryIndexCacheResult:
    fingerprint = fingerprint_ratings_data(data)
    cache_root = resolve_training_index_cache_root(
        repo_root=repo_root,
        runtime_config_payload=runtime_config_payload,
    )
    cache_dir = _cache_dir(
        cache_root=cache_root,
        dataset_short_name=dataset_short_name,
        split_id=split_id,
        dtype=dtype,
    )
    manifest_path = cache_dir / "user_history_manifest.json"
    if use_cache:
        cached_index = _try_load_user_history_index(
            manifest_path=manifest_path,
            expected_metadata=_expected_manifest_payload(
                index_kind="user_history",
                dataset_short_name=dataset_short_name,
                split_family=split_family,
                split_id=split_id,
                dtype=dtype,
                processed_manifest_path=processed_manifest_path,
                repo_root=repo_root,
                fingerprint=fingerprint,
                n_users=data.n_users,
                n_items=data.n_items,
            ),
            mmap_mode=mmap_mode,
            n_users=data.n_users,
        )
        if cached_index is not None:
            return UserHistoryIndexCacheResult(
                index=cached_index,
                metadata=TrainingIndexCacheMetadata(
                    cache_status="hit",
                    cache_manifest_path=manifest_path,
                    cache_root=cache_root,
                    train_fingerprint=fingerprint,
                ),
            )

    index = build_user_history_index(data, dtype=dtype)
    if use_cache:
        _write_user_history_cache(
            index=index,
            manifest_path=manifest_path,
            expected_metadata=_expected_manifest_payload(
                index_kind="user_history",
                dataset_short_name=dataset_short_name,
                split_family=split_family,
                split_id=split_id,
                dtype=dtype,
                processed_manifest_path=processed_manifest_path,
                repo_root=repo_root,
                fingerprint=fingerprint,
                n_users=data.n_users,
                n_items=data.n_items,
            ),
        )
    return UserHistoryIndexCacheResult(
        index=index,
        metadata=TrainingIndexCacheMetadata(
            cache_status="miss" if use_cache else "disabled",
            cache_manifest_path=manifest_path,
            cache_root=cache_root,
            train_fingerprint=fingerprint,
        ),
    )


def _load_or_build_user_explicit_feedback_index(
    *,
    data: RatingsData,
    dtype: str,
    dataset_short_name: str,
    split_family: str,
    split_id: str,
    processed_manifest_path: Path,
    repo_root: Path,
    runtime_config_payload: dict[str, Any],
    use_cache: bool,
    mmap_mode: MMapMode | None,
) -> UserExplicitFeedbackIndexCacheResult:
    fingerprint = fingerprint_ratings_data(data)
    cache_root = resolve_training_index_cache_root(
        repo_root=repo_root,
        runtime_config_payload=runtime_config_payload,
    )
    cache_dir = _cache_dir(
        cache_root=cache_root,
        dataset_short_name=dataset_short_name,
        split_id=split_id,
        dtype=dtype,
    )
    manifest_path = cache_dir / "user_explicit_feedback_manifest.json"
    if use_cache:
        cached_index = _try_load_user_explicit_feedback_index(
            manifest_path=manifest_path,
            expected_metadata=_expected_manifest_payload(
                index_kind="user_explicit_feedback",
                dataset_short_name=dataset_short_name,
                split_family=split_family,
                split_id=split_id,
                dtype=dtype,
                processed_manifest_path=processed_manifest_path,
                repo_root=repo_root,
                fingerprint=fingerprint,
                n_users=data.n_users,
                n_items=data.n_items,
            ),
            mmap_mode=mmap_mode,
            n_users=data.n_users,
        )
        if cached_index is not None:
            return UserExplicitFeedbackIndexCacheResult(
                index=cached_index,
                metadata=TrainingIndexCacheMetadata(
                    cache_status="hit",
                    cache_manifest_path=manifest_path,
                    cache_root=cache_root,
                    train_fingerprint=fingerprint,
                ),
            )

    index = build_user_explicit_feedback_index(data, dtype=dtype)
    if use_cache:
        _write_user_explicit_feedback_cache(
            index=index,
            manifest_path=manifest_path,
            expected_metadata=_expected_manifest_payload(
                index_kind="user_explicit_feedback",
                dataset_short_name=dataset_short_name,
                split_family=split_family,
                split_id=split_id,
                dtype=dtype,
                processed_manifest_path=processed_manifest_path,
                repo_root=repo_root,
                fingerprint=fingerprint,
                n_users=data.n_users,
                n_items=data.n_items,
            ),
        )
    return UserExplicitFeedbackIndexCacheResult(
        index=index,
        metadata=TrainingIndexCacheMetadata(
            cache_status="miss" if use_cache else "disabled",
            cache_manifest_path=manifest_path,
            cache_root=cache_root,
            train_fingerprint=fingerprint,
        ),
    )


def _cache_dir(
    *,
    cache_root: Path,
    dataset_short_name: str,
    split_id: str,
    dtype: str,
) -> Path:
    return cache_root / "ti" / dataset_short_name / split_id / dtype


def _expected_manifest_payload(
    *,
    index_kind: str,
    dataset_short_name: str,
    split_family: str,
    split_id: str,
    dtype: str,
    processed_manifest_path: Path,
    repo_root: Path,
    fingerprint: RatingsDataFingerprint,
    n_users: int,
    n_items: int,
) -> dict[str, Any]:
    return {
        "manifest_version": "v1",
        "kind": "training_index_cache",
        "index_kind": index_kind,
        "dataset": {
            "short_name": dataset_short_name,
            "split_family": split_family,
            "split_id": split_id,
            "processed_manifest_ref": repo_path_string(processed_manifest_path, repo_root=repo_root),
        },
        "dtype": dtype,
        "shape": {
            "n_users": int(n_users),
            "n_items": int(n_items),
            "train_rows": int(fingerprint.row_count),
        },
        "fingerprint": {
            "sha256": fingerprint.sha256,
        },
    }


def _try_load_user_history_index(
    *,
    manifest_path: Path,
    expected_metadata: dict[str, Any],
    mmap_mode: MMapMode | None,
    n_users: int,
) -> UserHistoryIndex | None:
    try:
        payload = _load_and_validate_manifest(
            manifest_path=manifest_path,
            expected_metadata=expected_metadata,
        )
        artifacts = payload["artifacts"]
        index = UserHistoryIndex(
            indptr=np.load(_artifact_path(manifest_path, artifacts["indptr_npy"]), mmap_mode=mmap_mode),
            item_indices=np.load(_artifact_path(manifest_path, artifacts["item_indices_npy"]), mmap_mode=mmap_mode),
            counts=np.load(_artifact_path(manifest_path, artifacts["counts_npy"]), mmap_mode=mmap_mode),
            norms=np.load(_artifact_path(manifest_path, artifacts["norms_npy"]), mmap_mode=mmap_mode),
        )
        validate_user_history_index(index, n_users=n_users)
        return index
    except Exception:
        return None


def _try_load_user_explicit_feedback_index(
    *,
    manifest_path: Path,
    expected_metadata: dict[str, Any],
    mmap_mode: MMapMode | None,
    n_users: int,
) -> UserExplicitFeedbackIndex | None:
    try:
        payload = _load_and_validate_manifest(
            manifest_path=manifest_path,
            expected_metadata=expected_metadata,
        )
        artifacts = payload["artifacts"]
        index = UserExplicitFeedbackIndex(
            indptr=np.load(_artifact_path(manifest_path, artifacts["indptr_npy"]), mmap_mode=mmap_mode),
            item_indices=np.load(_artifact_path(manifest_path, artifacts["item_indices_npy"]), mmap_mode=mmap_mode),
            ratings=np.load(_artifact_path(manifest_path, artifacts["ratings_npy"]), mmap_mode=mmap_mode),
            counts=np.load(_artifact_path(manifest_path, artifacts["counts_npy"]), mmap_mode=mmap_mode),
            norms=np.load(_artifact_path(manifest_path, artifacts["norms_npy"]), mmap_mode=mmap_mode),
        )
        validate_user_explicit_feedback_index(index, n_users=n_users)
        return index
    except Exception:
        return None


def _load_and_validate_manifest(
    *,
    manifest_path: Path,
    expected_metadata: dict[str, Any],
) -> dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("manifest_version", "kind", "index_kind", "dtype"):
        if payload.get(key) != expected_metadata[key]:
            raise ValueError(f"cache manifest mismatch at key '{key}'")
    for key in ("dataset", "shape", "fingerprint"):
        if payload.get(key) != expected_metadata[key]:
            raise ValueError(f"cache manifest mismatch at key '{key}'")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError("cache manifest artifacts must be an object")
    return payload


def _write_user_history_cache(
    *,
    index: UserHistoryIndex,
    manifest_path: Path,
    expected_metadata: dict[str, Any],
) -> None:
    cache_dir = manifest_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "indptr_npy": "uh_i.npy",
        "item_indices_npy": "uh_items.npy",
        "counts_npy": "uh_c.npy",
        "norms_npy": "uh_n.npy",
    }
    atomic_save_array(cache_dir / artifacts["indptr_npy"], index.indptr)
    atomic_save_array(cache_dir / artifacts["item_indices_npy"], index.item_indices)
    atomic_save_array(cache_dir / artifacts["counts_npy"], index.counts)
    atomic_save_array(cache_dir / artifacts["norms_npy"], index.norms)
    atomic_write_json(
        manifest_path,
        {
            **expected_metadata,
            "artifacts": artifacts,
        },
    )


def _write_user_explicit_feedback_cache(
    *,
    index: UserExplicitFeedbackIndex,
    manifest_path: Path,
    expected_metadata: dict[str, Any],
) -> None:
    cache_dir = manifest_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "indptr_npy": "uef_i.npy",
        "item_indices_npy": "uef_items.npy",
        "ratings_npy": "uef_r.npy",
        "counts_npy": "uef_c.npy",
        "norms_npy": "uef_n.npy",
    }
    atomic_save_array(cache_dir / artifacts["indptr_npy"], index.indptr)
    atomic_save_array(cache_dir / artifacts["item_indices_npy"], index.item_indices)
    atomic_save_array(cache_dir / artifacts["ratings_npy"], index.ratings)
    atomic_save_array(cache_dir / artifacts["counts_npy"], index.counts)
    atomic_save_array(cache_dir / artifacts["norms_npy"], index.norms)
    atomic_write_json(
        manifest_path,
        {
            **expected_metadata,
            "artifacts": artifacts,
        },
    )


def _artifact_path(manifest_path: Path, artifact_ref: str) -> Path:
    path = Path(artifact_ref)
    if path.is_absolute():
        return path
    return manifest_path.parent / path
