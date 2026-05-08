from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from recsys_lab.clustering.cache import (
    _array_sha256,
    _cluster_history_identity_payload,
    _cluster_identity_payload,
    _stable_sha256,
)
from recsys_lab.data.processed import RatingsData
from recsys_lab.data.training_index_cache import HISTORY_LAYOUT_VERSION, fingerprint_ratings_data
from recsys_lab.models.biased_mf import BiasedMFConfig


def _toy_data(tmp_path: Path, *, split_rows: np.ndarray | None = None) -> tuple[RatingsData, Path]:
    manifest_path = tmp_path / "data" / "processed" / "toy_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8", newline="\n")
    data = RatingsData(
        user_ids=np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32),
        item_ids=np.asarray([0, 1, 1, 2, 2, 3, 0, 3], dtype=np.int32),
        ratings=np.asarray([5.0, 4.5, 4.0, 3.5, 2.5, 2.0, 1.5, 1.0], dtype=np.float32),
        n_users=4,
        n_items=4,
        name="toy",
        rating_min=1.0,
        rating_max=5.0,
        source_manifest_path=manifest_path,
    )
    if split_rows is not None:
        data = data.subset(split_rows, name="train")
    return data, manifest_path


def _induction_config() -> BiasedMFConfig:
    return BiasedMFConfig(
        latent_dim=4,
        epochs=6,
        learning_rate=0.02,
        lambda_b=0.01,
        lambda_p=0.01,
        lambda_q=0.01,
        seed=7,
        init_std=0.05,
        dtype="float32",
    )


def _target_model_metadata(
    *,
    alpha: float = 0.2,
    lambda_pC: float = 0.01,
    lambda_qC: float = 0.01,
    lambda_yC: float = 0.01,
    epochs: int = 8,
) -> dict[str, Any]:
    return {
        "target_alpha": float(alpha),
        "target_lambda_pC": float(lambda_pC),
        "target_lambda_qC": float(lambda_qC),
        "target_lambda_yC": float(lambda_yC),
        "target_model_epochs": int(epochs),
    }


def _cluster_cache_identity(
    tmp_path: Path,
    *,
    data: RatingsData | None = None,
    manifest_path: Path | None = None,
    induction_config: BiasedMFConfig | None = None,
    split_family: str = "benchmark_random_v1",
    split_id: str = "benchmark_random_v1_tr080_va010_s001",
    n_user_clusters: int = 2,
    n_item_clusters: int = 2,
    target_model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_data, resolved_manifest_path = _toy_data(tmp_path) if data is None else (data, manifest_path)
    if resolved_manifest_path is None:
        raise ValueError("manifest_path is required when data is provided")
    # Target-model metadata is intentionally not passed into the production
    # cluster identity. It belongs to run/report metadata, not cache identity.
    assert target_model_metadata is not None
    return _cluster_identity_payload(
        dataset_short_name="ml_latest_small",
        split_family=split_family,
        split_id=split_id,
        processed_manifest_path=resolved_manifest_path,
        repo_root=tmp_path,
        fingerprint=fingerprint_ratings_data(resolved_data),
        induction_config=induction_config or _induction_config(),
        n_users=resolved_data.n_users,
        n_items=resolved_data.n_items,
        n_user_clusters=n_user_clusters,
        n_item_clusters=n_item_clusters,
        algorithm="kmeans",
        kmeans_n_init=2,
    )


def _cluster_cache_key(tmp_path: Path, **kwargs: Any) -> str:
    return _stable_sha256(_cluster_cache_identity(tmp_path, **kwargs))[:16]


def test_cluster_artifact_cache_key_stable_when_alpha_changes(tmp_path: Path) -> None:
    baseline = _cluster_cache_key(tmp_path, target_model_metadata=_target_model_metadata(alpha=0.1))
    changed = _cluster_cache_key(tmp_path, target_model_metadata=_target_model_metadata(alpha=0.8))

    assert changed == baseline


def test_cluster_artifact_cache_key_stable_when_cb_lambda_changes(tmp_path: Path) -> None:
    baseline = _cluster_cache_key(
        tmp_path,
        target_model_metadata=_target_model_metadata(lambda_pC=0.01, lambda_qC=0.01, lambda_yC=0.01),
    )
    changed = _cluster_cache_key(
        tmp_path,
        target_model_metadata=_target_model_metadata(lambda_pC=0.05, lambda_qC=0.07, lambda_yC=0.09),
    )

    assert changed == baseline


def test_cluster_artifact_cache_key_stable_when_cb_epochs_change(tmp_path: Path) -> None:
    baseline = _cluster_cache_key(tmp_path, target_model_metadata=_target_model_metadata(epochs=8))
    changed = _cluster_cache_key(tmp_path, target_model_metadata=_target_model_metadata(epochs=30))

    assert changed == baseline


def test_cluster_artifact_cache_key_changes_when_user_cluster_count_changes(tmp_path: Path) -> None:
    baseline = _cluster_cache_key(
        tmp_path,
        n_user_clusters=2,
        target_model_metadata=_target_model_metadata(),
    )
    changed = _cluster_cache_key(
        tmp_path,
        n_user_clusters=3,
        target_model_metadata=_target_model_metadata(),
    )

    assert changed != baseline


def test_cluster_artifact_cache_key_changes_when_item_cluster_count_changes(tmp_path: Path) -> None:
    baseline = _cluster_cache_key(
        tmp_path,
        n_item_clusters=2,
        target_model_metadata=_target_model_metadata(),
    )
    changed = _cluster_cache_key(
        tmp_path,
        n_item_clusters=3,
        target_model_metadata=_target_model_metadata(),
    )

    assert changed != baseline


def test_cluster_artifact_cache_key_changes_when_induction_config_changes(tmp_path: Path) -> None:
    induction_config = _induction_config()
    baseline = _cluster_cache_key(
        tmp_path,
        induction_config=induction_config,
        target_model_metadata=_target_model_metadata(),
    )
    changed = _cluster_cache_key(
        tmp_path,
        induction_config=replace(induction_config, seed=induction_config.seed + 1),
        target_model_metadata=_target_model_metadata(),
    )

    assert changed != baseline


def test_cluster_artifact_cache_key_changes_when_split_or_train_fingerprint_changes(tmp_path: Path) -> None:
    data, manifest_path = _toy_data(tmp_path)
    split_changed = _cluster_cache_key(
        tmp_path,
        data=data,
        manifest_path=manifest_path,
        split_id="benchmark_random_v1_tr080_va010_s002",
        target_model_metadata=_target_model_metadata(),
    )
    baseline = _cluster_cache_key(
        tmp_path,
        data=data,
        manifest_path=manifest_path,
        target_model_metadata=_target_model_metadata(),
    )
    train_rows = np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    changed_data, changed_manifest_path = _toy_data(tmp_path, split_rows=train_rows)
    fingerprint_changed = _cluster_cache_key(
        tmp_path,
        data=changed_data,
        manifest_path=changed_manifest_path,
        target_model_metadata=_target_model_metadata(),
    )

    assert split_changed != baseline
    assert fingerprint_changed != baseline


def _cluster_history_cache_identity(tmp_path: Path, *, item_clusters: np.ndarray) -> dict[str, Any]:
    data, manifest_path = _toy_data(tmp_path)
    train_fingerprint = fingerprint_ratings_data(data)
    return _cluster_history_identity_payload(
        dataset_short_name="ml_latest_small",
        split_family="benchmark_random_v1",
        split_id="benchmark_random_v1_tr080_va010_s001",
        processed_manifest_path=manifest_path,
        repo_root=tmp_path,
        train_fingerprint=train_fingerprint,
        n_users=data.n_users,
        n_clusters=2,
        cluster_cache_key="cluster-artifact-key",
        cluster_cache_fingerprint_sha256="cluster-artifact-fingerprint",
        item_cluster_fingerprint_sha256=_array_sha256(item_clusters),
    )


def test_user_cluster_history_cache_key_depends_on_item_cluster_assignments(tmp_path: Path) -> None:
    baseline = _stable_sha256(
        _cluster_history_cache_identity(
            tmp_path,
            item_clusters=np.asarray([0, 0, 1, 1], dtype=np.int32),
        )
    )[:16]
    changed = _stable_sha256(
        _cluster_history_cache_identity(
            tmp_path,
            item_clusters=np.asarray([0, 1, 1, 0], dtype=np.int32),
        )
    )[:16]

    assert changed != baseline


def test_user_cluster_history_cache_key_uses_history_layout_version(tmp_path: Path) -> None:
    identity = _cluster_history_cache_identity(
        tmp_path,
        item_clusters=np.asarray([0, 0, 1, 1], dtype=np.int32),
    )

    assert identity["layout"]["layout_version"] == HISTORY_LAYOUT_VERSION
    assert identity["layout"]["index_dtype"] == "int32"
    assert identity["layout"]["count_dtype"] == "int32"
