from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from recsys_lab.data.processed import RatingsData
from recsys_lab.data.splitters import RatingsSplit
from recsys_lab.data.training_index_cache import RatingsDataFingerprint, fingerprint_ratings_data
from recsys_lab.utils.paths import repo_path_string


@dataclass(frozen=True, slots=True)
class SplitCachePolicy:
    requested_policy: str
    effective_use_cache: bool
    decision_reason: str


@dataclass(frozen=True, slots=True)
class SplitCacheMetadata:
    cache_status: str
    cache_manifest_path: Path
    cache_root: Path
    source_fingerprint: RatingsDataFingerprint


@dataclass(frozen=True, slots=True)
class SplitCacheResult:
    split: RatingsSplit
    metadata: SplitCacheMetadata


def resolve_split_cache_policy(
    *,
    split_family: str,
    use_split_cache: bool | None,
) -> SplitCachePolicy:
    if use_split_cache is True:
        return SplitCachePolicy(
            requested_policy="force_on",
            effective_use_cache=True,
            decision_reason="explicit_enable",
        )
    if use_split_cache is False:
        return SplitCachePolicy(
            requested_policy="force_off",
            effective_use_cache=False,
            decision_reason="explicit_disable",
        )
    return SplitCachePolicy(
        requested_policy="auto",
        effective_use_cache=False,
        decision_reason="auto_disable_no_positive_evidence_split_family",
    )


def resolve_split_cache_root(
    *,
    repo_root: Path,
    runtime_config_payload: dict[str, Any],
) -> Path:
    env_override = os.environ.get("RECSYS_CACHE_ROOT")
    if env_override:
        return (Path(env_override).expanduser().resolve() / "split_cache").resolve()

    runtime = runtime_config_payload.get("runtime", {})
    configured_root = Path(str(runtime.get("cache_root", "artifacts/local")))
    if not configured_root.is_absolute():
        configured_root = (repo_root / configured_root).resolve()
    return (configured_root / "split_cache").resolve()


def load_or_build_split_cache(
    *,
    data: RatingsData,
    dataset_short_name: str,
    split_family: str,
    split_id: str,
    processed_manifest_path: Path,
    repo_root: Path,
    runtime_config_payload: dict[str, Any],
    build_split: Callable[[], RatingsSplit],
    use_cache: bool,
) -> SplitCacheResult:
    fingerprint = fingerprint_ratings_data(data)
    cache_root = resolve_split_cache_root(
        repo_root=repo_root,
        runtime_config_payload=runtime_config_payload,
    )
    cache_dir = cache_root / dataset_short_name / split_id
    manifest_path = cache_dir / "split_manifest.json"

    if use_cache:
        cached_split = _try_load_split_cache(
            data=data,
            manifest_path=manifest_path,
            expected_manifest=_expected_manifest_payload(
                dataset_short_name=dataset_short_name,
                split_family=split_family,
                split_id=split_id,
                processed_manifest_path=processed_manifest_path,
                repo_root=repo_root,
                fingerprint=fingerprint,
            ),
        )
        if cached_split is not None:
            return SplitCacheResult(
                split=cached_split,
                metadata=SplitCacheMetadata(
                    cache_status="hit",
                    cache_manifest_path=manifest_path,
                    cache_root=cache_root,
                    source_fingerprint=fingerprint,
                ),
            )

    split = build_split()
    if use_cache:
        _write_split_cache(
            split=split,
            manifest_path=manifest_path,
            expected_manifest=_expected_manifest_payload(
                dataset_short_name=dataset_short_name,
                split_family=split_family,
                split_id=split_id,
                processed_manifest_path=processed_manifest_path,
                repo_root=repo_root,
                fingerprint=fingerprint,
            ),
        )

    return SplitCacheResult(
        split=split,
        metadata=SplitCacheMetadata(
            cache_status="miss" if use_cache else "disabled",
            cache_manifest_path=manifest_path,
            cache_root=cache_root,
            source_fingerprint=fingerprint,
        ),
    )


def _expected_manifest_payload(
    *,
    dataset_short_name: str,
    split_family: str,
    split_id: str,
    processed_manifest_path: Path,
    repo_root: Path,
    fingerprint: RatingsDataFingerprint,
) -> dict[str, Any]:
    return {
        "kind": "split_cache_manifest",
        "dataset_short_name": dataset_short_name,
        "split_family": split_family,
        "split_id": split_id,
        "processed_manifest_path": repo_path_string(processed_manifest_path, repo_root=repo_root),
        "source_manifest_path": repo_path_string(
            data_source_manifest_path(processed_manifest_path), repo_root=repo_root
        ),
        "source_fingerprint": {
            "row_count": fingerprint.row_count,
            "sha256": fingerprint.sha256,
        },
    }


def data_source_manifest_path(processed_manifest_path: Path) -> Path:
    return processed_manifest_path.resolve()


def _write_split_cache(
    *,
    split: RatingsSplit,
    manifest_path: Path,
    expected_manifest: dict[str, Any],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    train_path = manifest_path.parent / "train_row_indices.npy"
    validation_path = manifest_path.parent / "validation_row_indices.npy"
    test_path = manifest_path.parent / "test_row_indices.npy"

    np.save(train_path, split.train.training_row_indices(), allow_pickle=False)
    np.save(test_path, split.test.training_row_indices(), allow_pickle=False)

    validation_payload: str | None = None
    if split.validation is not None:
        np.save(validation_path, split.validation.training_row_indices(), allow_pickle=False)
        validation_payload = str(validation_path)

    payload = {
        **expected_manifest,
        "artifacts": {
            "train_row_indices_npy": str(train_path),
            "validation_row_indices_npy": validation_payload,
            "test_row_indices_npy": str(test_path),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")


def _try_load_split_cache(
    *,
    data: RatingsData,
    manifest_path: Path,
    expected_manifest: dict[str, Any],
) -> RatingsSplit | None:
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    for key in (
        "kind",
        "dataset_short_name",
        "split_family",
        "split_id",
        "processed_manifest_path",
        "source_manifest_path",
    ):
        if payload.get(key) != expected_manifest.get(key):
            return None
    if payload.get("source_fingerprint") != expected_manifest.get("source_fingerprint"):
        return None

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return None

    train_path = artifacts.get("train_row_indices_npy")
    test_path = artifacts.get("test_row_indices_npy")
    if not isinstance(train_path, str) or not isinstance(test_path, str):
        return None

    train_idx = _load_row_indices(Path(train_path))
    test_idx = _load_row_indices(Path(test_path))

    validation_idx = None
    validation_path = artifacts.get("validation_row_indices_npy")
    if validation_path is not None:
        if not isinstance(validation_path, str):
            return None
        validation_idx = _load_row_indices(Path(validation_path))

    validation = None
    if validation_idx is not None:
        validation = data.subset(validation_idx, name=f"{data.name}:validation")

    return RatingsSplit(
        train=data.subset(train_idx, name=f"{data.name}:train"),
        validation=validation,
        test=data.subset(test_idx, name=f"{data.name}:test"),
    )


def _load_row_indices(path: Path) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    if array.ndim != 1:
        raise ValueError(f"split cache row index array must be 1D: {path}")
    return np.asarray(array, dtype=np.int64)
