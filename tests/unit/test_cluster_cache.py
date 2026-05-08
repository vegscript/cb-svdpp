import json
from pathlib import Path

import numpy as np

from recsys_lab.clustering import (
    load_or_build_cluster_artifacts,
    load_or_build_user_cluster_history_index,
)
from recsys_lab.data.histories import build_user_history_index
from recsys_lab.data.processed import RatingsData
from recsys_lab.models.biased_mf import BiasedMFConfig


def _toy_data(tmp_path: Path, *, ratings: np.ndarray | None = None) -> tuple[RatingsData, Path]:
    manifest_path = tmp_path / "data" / "processed" / "toy_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8", newline="\n")
    return (
        RatingsData(
            user_ids=np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32),
            item_ids=np.asarray([0, 1, 1, 2, 2, 3, 0, 3], dtype=np.int32),
            ratings=(
                np.asarray([5.0, 4.5, 4.0, 3.5, 2.5, 2.0, 1.5, 1.0], dtype=np.float32)
                if ratings is None
                else ratings
            ),
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


def _cluster_artifact_kwargs(data: RatingsData, manifest_path: Path, repo_root: Path) -> dict[str, object]:
    return {
        "data": data,
        "induction_config": _induction_config(),
        "n_user_clusters": 2,
        "n_item_clusters": 2,
        "algorithm": "kmeans",
        "kmeans_n_init": 2,
        "dataset_short_name": "ml_latest_small",
        "split_family": "benchmark_random_v1",
        "split_id": "benchmark_random_v1_tr080_va010_s001",
        "processed_manifest_path": manifest_path,
        "repo_root": repo_root,
        "runtime_config_payload": {"runtime": {"cache_root": "artifacts/local"}},
        "use_cache": True,
        "mmap_mode": None,
    }


def test_cluster_artifact_cache_hits_on_second_load(tmp_path: Path) -> None:
    data, manifest_path = _toy_data(tmp_path)

    miss_result = load_or_build_cluster_artifacts(**_cluster_artifact_kwargs(data, manifest_path, tmp_path))
    hit_result = load_or_build_cluster_artifacts(**_cluster_artifact_kwargs(data, manifest_path, tmp_path))

    assert miss_result.metadata.cache_status == "miss"
    assert hit_result.metadata.cache_status == "hit"
    assert hit_result.metadata.cache_key == miss_result.metadata.cache_key
    assert hit_result.metadata.cache_manifest_path.exists()
    assert np.array_equal(miss_result.artifacts.user_clusters, hit_result.artifacts.user_clusters)
    assert np.array_equal(miss_result.artifacts.item_clusters, hit_result.artifacts.item_clusters)
    assert np.allclose(miss_result.artifacts.r_star_means, hit_result.artifacts.r_star_means)

    manifest = json.loads(hit_result.metadata.cache_manifest_path.read_text(encoding="utf-8"))
    assert manifest["identity"]["fingerprint"]["train_sha256"] == hit_result.metadata.train_fingerprint.sha256
    assert all(not Path(path_ref).is_absolute() for path_ref in manifest["artifacts"].values())


def test_cluster_artifact_cache_invalidates_on_train_fingerprint_change(tmp_path: Path) -> None:
    data, manifest_path = _toy_data(tmp_path)
    changed_ratings = np.asarray([5.0, 4.5, 4.0, 3.5, 2.5, 2.0, 1.5, 1.25], dtype=np.float32)
    changed_data, _ = _toy_data(tmp_path, ratings=changed_ratings)

    first_result = load_or_build_cluster_artifacts(**_cluster_artifact_kwargs(data, manifest_path, tmp_path))
    changed_result = load_or_build_cluster_artifacts(**_cluster_artifact_kwargs(changed_data, manifest_path, tmp_path))

    assert first_result.metadata.cache_status == "miss"
    assert changed_result.metadata.cache_status == "miss"
    assert changed_result.metadata.train_fingerprint.sha256 != first_result.metadata.train_fingerprint.sha256
    assert changed_result.metadata.cache_key != first_result.metadata.cache_key


def test_cluster_artifact_cache_ignores_non_train_rating_changes(tmp_path: Path) -> None:
    data, manifest_path = _toy_data(tmp_path)
    train_rows = np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    train_data = data.subset(train_rows, name="train")
    changed_ratings = np.asarray([5.0, 4.5, 4.0, 3.5, 2.5, 2.0, 1.5, 3.75], dtype=np.float32)
    changed_data, _ = _toy_data(tmp_path, ratings=changed_ratings)
    changed_train_data = changed_data.subset(train_rows, name="train")

    first_result = load_or_build_cluster_artifacts(**_cluster_artifact_kwargs(train_data, manifest_path, tmp_path))
    changed_result = load_or_build_cluster_artifacts(
        **_cluster_artifact_kwargs(changed_train_data, manifest_path, tmp_path)
    )

    assert first_result.metadata.cache_status == "miss"
    assert changed_result.metadata.cache_status == "hit"
    assert changed_result.metadata.train_fingerprint.sha256 == first_result.metadata.train_fingerprint.sha256
    assert changed_result.metadata.cache_key == first_result.metadata.cache_key


def test_user_cluster_history_cache_hits_on_second_load(tmp_path: Path) -> None:
    data, manifest_path = _toy_data(tmp_path)
    cluster_result = load_or_build_cluster_artifacts(**_cluster_artifact_kwargs(data, manifest_path, tmp_path))
    history_index = build_user_history_index(data, dtype="float32")
    kwargs = {
        "history_index": history_index,
        "item_clusters": cluster_result.artifacts.item_clusters,
        "n_clusters": cluster_result.artifacts.r_star_counts.shape[1],
        "dataset_short_name": "ml_latest_small",
        "split_family": "benchmark_random_v1",
        "split_id": "benchmark_random_v1_tr080_va010_s001",
        "processed_manifest_path": manifest_path,
        "repo_root": tmp_path,
        "runtime_config_payload": {"runtime": {"cache_root": "artifacts/local"}},
        "train_fingerprint": cluster_result.metadata.train_fingerprint,
        "cluster_cache_key": cluster_result.metadata.cache_key,
        "cluster_cache_fingerprint_sha256": cluster_result.metadata.cache_fingerprint_sha256,
        "use_cache": True,
        "mmap_mode": None,
    }

    miss_result = load_or_build_user_cluster_history_index(**kwargs)
    hit_result = load_or_build_user_cluster_history_index(**kwargs)

    assert miss_result.metadata.cache_status == "miss"
    assert hit_result.metadata.cache_status == "hit"
    assert hit_result.metadata.cache_key == miss_result.metadata.cache_key
    assert np.array_equal(miss_result.index.indptr, hit_result.index.indptr)
    assert np.array_equal(miss_result.index.cluster_ids, hit_result.index.cluster_ids)
    assert np.array_equal(miss_result.index.counts, hit_result.index.counts)

    manifest = json.loads(hit_result.metadata.cache_manifest_path.read_text(encoding="utf-8"))
    assert manifest["identity"]["fingerprint"]["cluster_cache_key"] == cluster_result.metadata.cache_key
    assert manifest["identity"]["layout"] == {
        "layout_version": "history_data_layout_v1",
        "index_dtype": "int32",
        "count_dtype": "int32",
    }
    assert all(not Path(path_ref).is_absolute() for path_ref in manifest["artifacts"].values())


def test_user_cluster_history_cache_rebuilds_legacy_layout_manifest(tmp_path: Path) -> None:
    data, manifest_path = _toy_data(tmp_path)
    cluster_result = load_or_build_cluster_artifacts(**_cluster_artifact_kwargs(data, manifest_path, tmp_path))
    history_index = build_user_history_index(data, dtype="float32")
    kwargs = {
        "history_index": history_index,
        "item_clusters": cluster_result.artifacts.item_clusters,
        "n_clusters": cluster_result.artifacts.r_star_counts.shape[1],
        "dataset_short_name": "ml_latest_small",
        "split_family": "benchmark_random_v1",
        "split_id": "benchmark_random_v1_tr080_va010_s001",
        "processed_manifest_path": manifest_path,
        "repo_root": tmp_path,
        "runtime_config_payload": {"runtime": {"cache_root": "artifacts/local"}},
        "train_fingerprint": cluster_result.metadata.train_fingerprint,
        "cluster_cache_key": cluster_result.metadata.cache_key,
        "cluster_cache_fingerprint_sha256": cluster_result.metadata.cache_fingerprint_sha256,
        "use_cache": True,
        "mmap_mode": None,
    }

    initial_result = load_or_build_user_cluster_history_index(**kwargs)
    assert initial_result.metadata.cache_status == "miss"

    cache_manifest = json.loads(initial_result.metadata.cache_manifest_path.read_text(encoding="utf-8"))
    cache_manifest["identity"].pop("layout")
    initial_result.metadata.cache_manifest_path.write_text(json.dumps(cache_manifest), encoding="utf-8")

    rebuilt_result = load_or_build_user_cluster_history_index(**kwargs)

    assert rebuilt_result.metadata.cache_status == "miss"
    rebuilt_manifest = json.loads(rebuilt_result.metadata.cache_manifest_path.read_text(encoding="utf-8"))
    assert rebuilt_manifest["identity"]["layout"]["layout_version"] == "history_data_layout_v1"
