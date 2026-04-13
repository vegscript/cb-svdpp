from pathlib import Path

import numpy as np

from recsys_lab.clustering import induce_train_only_clusters
from recsys_lab.data.histories import build_user_cluster_count_index, build_user_history_index
from recsys_lab.data.processed import RatingsData
from recsys_lab.models.biased_mf import BiasedMFConfig


def _toy_data() -> RatingsData:
    return RatingsData(
        user_ids=np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32),
        item_ids=np.asarray([0, 1, 1, 2, 2, 3, 0, 3], dtype=np.int32),
        ratings=np.asarray([5.0, 4.5, 4.0, 3.5, 2.5, 2.0, 1.5, 1.0], dtype=np.float32),
        n_users=4,
        n_items=4,
        name="toy",
        rating_min=1.0,
        rating_max=5.0,
        source_manifest_path=Path("toy_manifest.json"),
    )


def test_build_user_cluster_count_index_aggregates_duplicate_clusters() -> None:
    data = _toy_data()
    history_index = build_user_history_index(data, dtype="float32")
    item_clusters = np.asarray([0, 0, 1, 1], dtype=np.int32)

    cluster_history = build_user_cluster_count_index(
        history_index,
        item_clusters,
        n_clusters=2,
    )

    assert cluster_history.clusters_for_user(0).tolist() == [0]
    assert cluster_history.counts_for_user(0).tolist() == [2]
    assert cluster_history.clusters_for_user(1).tolist() == [0, 1]
    assert cluster_history.counts_for_user(1).tolist() == [1, 1]


def test_induce_train_only_clusters_builds_dense_r_star() -> None:
    data = _toy_data()
    artifacts = induce_train_only_clusters(
        data,
        induction_config=BiasedMFConfig(
            latent_dim=4,
            epochs=8,
            learning_rate=0.02,
            lambda_b=0.01,
            lambda_p=0.01,
            lambda_q=0.01,
            seed=7,
            init_std=0.05,
            dtype="float32",
        ),
        n_user_clusters=2,
        n_item_clusters=2,
        algorithm="kmeans",
        kmeans_n_init=5,
    )

    assert artifacts.user_clusters.shape == (data.n_users,)
    assert artifacts.item_clusters.shape == (data.n_items,)
    assert artifacts.r_star_means.shape == (2, 2)
    assert artifacts.r_star_counts.shape == (2, 2)
    assert int(artifacts.r_star_counts.sum()) == len(data)
