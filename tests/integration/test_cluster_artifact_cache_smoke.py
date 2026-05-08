from __future__ import annotations

from pathlib import Path

import numpy as np

from recsys_lab.clustering import load_or_build_cluster_artifacts, load_or_build_user_cluster_history_index
from recsys_lab.data.histories import build_user_history_index
from recsys_lab.data.processed import RatingsData
from recsys_lab.models.biased_mf import BiasedMFConfig


def _toy_data(tmp_path: Path) -> tuple[RatingsData, Path]:
    manifest_path = tmp_path / "data" / "processed" / "toy_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8", newline="\n")
    return (
        RatingsData(
            user_ids=np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32),
            item_ids=np.asarray([0, 1, 1, 2, 2, 3, 0, 3], dtype=np.int32),
            ratings=np.asarray([5.0, 4.5, 4.0, 3.5, 2.5, 2.0, 1.5, 1.0], dtype=np.float32),
            n_users=4,
            n_items=4,
            name="toy",
            rating_min=1.0,
            rating_max=5.0,
            source_manifest_path=manifest_path,
        ),
        manifest_path,
    )


def _induction_config() -> BiasedMFConfig:
    return BiasedMFConfig(
        latent_dim=2,
        epochs=1,
        learning_rate=0.02,
        lambda_b=0.01,
        lambda_p=0.01,
        lambda_q=0.01,
        seed=7,
        init_std=0.05,
        dtype="float32",
    )


def _cluster_kwargs(data: RatingsData, manifest_path: Path, repo_root: Path) -> dict[str, object]:
    return {
        "data": data,
        "induction_config": _induction_config(),
        "n_user_clusters": 2,
        "n_item_clusters": 2,
        "algorithm": "kmeans",
        "kmeans_n_init": 1,
        "dataset_short_name": "synthetic_tiny",
        "split_family": "benchmark_random_v1",
        "split_id": "benchmark_random_v1_tiny",
        "processed_manifest_path": manifest_path,
        "repo_root": repo_root,
        "runtime_config_payload": {"runtime": {"cache_root": "artifacts/local"}},
        "use_cache": True,
        "mmap_mode": None,
        "model": "cb_svdpp",
    }


def _assert_same_array(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected)


def test_cluster_artifact_cache_smoke_cold_then_warm(tmp_path: Path) -> None:
    data, manifest_path = _toy_data(tmp_path)
    cluster_kwargs = _cluster_kwargs(data, manifest_path, tmp_path)

    cold_cluster = load_or_build_cluster_artifacts(**cluster_kwargs)
    warm_cluster = load_or_build_cluster_artifacts(**cluster_kwargs)

    assert cold_cluster.metadata.cache_status == "miss"
    assert warm_cluster.metadata.cache_status == "hit"
    assert cold_cluster.metadata.cache_key == warm_cluster.metadata.cache_key
    assert cold_cluster.profile is not None
    assert warm_cluster.profile is not None
    assert cold_cluster.profile.cluster_cache_status == "miss"
    assert warm_cluster.profile.cluster_cache_status == "hit"
    assert cold_cluster.profile.timings.cluster_cache_write_seconds >= 0.0
    assert cold_cluster.profile.timings.induction_fit_seconds >= 0.0
    assert warm_cluster.profile.timings.cluster_cache_read_seconds >= 0.0
    assert warm_cluster.profile.timings.induction_fit_seconds == 0.0

    _assert_same_array(warm_cluster.artifacts.user_clusters, cold_cluster.artifacts.user_clusters)
    _assert_same_array(warm_cluster.artifacts.item_clusters, cold_cluster.artifacts.item_clusters)
    _assert_same_array(warm_cluster.artifacts.user_cluster_sizes, cold_cluster.artifacts.user_cluster_sizes)
    _assert_same_array(warm_cluster.artifacts.item_cluster_sizes, cold_cluster.artifacts.item_cluster_sizes)
    _assert_same_array(warm_cluster.artifacts.r_star_means, cold_cluster.artifacts.r_star_means)
    _assert_same_array(warm_cluster.artifacts.r_star_counts, cold_cluster.artifacts.r_star_counts)

    history_index = build_user_history_index(data, dtype="float32")
    history_kwargs = {
        "history_index": history_index,
        "item_clusters": cold_cluster.artifacts.item_clusters,
        "n_clusters": int(cold_cluster.artifacts.r_star_counts.shape[1]),
        "dataset_short_name": "synthetic_tiny",
        "split_family": "benchmark_random_v1",
        "split_id": "benchmark_random_v1_tiny",
        "processed_manifest_path": manifest_path,
        "repo_root": tmp_path,
        "runtime_config_payload": {"runtime": {"cache_root": "artifacts/local"}},
        "train_fingerprint": cold_cluster.metadata.train_fingerprint,
        "cluster_cache_key": cold_cluster.metadata.cache_key,
        "cluster_cache_fingerprint_sha256": cold_cluster.metadata.cache_fingerprint_sha256,
        "use_cache": True,
        "mmap_mode": None,
        "model": "cb_svdpp",
    }

    cold_history = load_or_build_user_cluster_history_index(**history_kwargs)
    warm_history = load_or_build_user_cluster_history_index(**history_kwargs)

    assert cold_history.metadata.cache_status == "miss"
    assert warm_history.metadata.cache_status == "hit"
    assert cold_history.metadata.cache_key == warm_history.metadata.cache_key
    assert cold_history.profile is not None
    assert warm_history.profile is not None
    assert cold_history.profile.user_cluster_history_cache_status == "miss"
    assert warm_history.profile.user_cluster_history_cache_status == "hit"
    assert cold_history.profile.timings.user_cluster_history_build_seconds >= 0.0
    assert cold_history.profile.timings.user_cluster_history_cache_write_seconds >= 0.0
    assert warm_history.profile.timings.user_cluster_history_cache_read_seconds >= 0.0
    assert warm_history.profile.timings.user_cluster_history_build_seconds == 0.0

    _assert_same_array(warm_history.index.indptr, cold_history.index.indptr)
    _assert_same_array(warm_history.index.cluster_ids, cold_history.index.cluster_ids)
    _assert_same_array(warm_history.index.counts, cold_history.index.counts)
