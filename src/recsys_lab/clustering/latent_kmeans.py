from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.cluster import KMeans

from recsys_lab.data.processed import RatingsData
from recsys_lab.metrics import rmse
from recsys_lab.models.biased_mf import BiasedMFConfig, BiasedMFRecommender


@dataclass(frozen=True, slots=True)
class ClusterInductionProfile:
    induction_fit_seconds: float = 0.0
    induction_predict_seconds: float = 0.0
    induction_train_rmse_seconds: float = 0.0
    user_kmeans_seconds: float = 0.0
    item_kmeans_seconds: float = 0.0
    r_star_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class ClusterArtifacts:
    user_clusters: np.ndarray
    item_clusters: np.ndarray
    user_cluster_sizes: np.ndarray
    item_cluster_sizes: np.ndarray
    r_star_means: np.ndarray
    r_star_counts: np.ndarray
    induction_train_rmse: float
    user_kmeans_inertia: float
    item_kmeans_inertia: float
    induction_profile: ClusterInductionProfile | None = None


def _validate_cluster_request(*, n_clusters: int, n_samples: int, axis_name: str) -> None:
    if n_clusters <= 0:
        raise ValueError(f"{axis_name} cluster count must be positive")
    if n_clusters > n_samples:
        raise ValueError(f"{axis_name} cluster count {n_clusters} exceeds available samples {n_samples}")


def _compute_r_star(
    data: RatingsData,
    *,
    user_clusters: np.ndarray,
    item_clusters: np.ndarray,
    n_user_clusters: int,
    n_item_clusters: int,
    dtype: str,
) -> tuple[np.ndarray, np.ndarray]:
    flat_cluster_ids = user_clusters[data.user_ids].astype(np.int64, copy=False) * int(n_item_clusters) + item_clusters[
        data.item_ids
    ].astype(np.int64, copy=False)
    cluster_pair_count = int(n_user_clusters) * int(n_item_clusters)

    counts = np.bincount(flat_cluster_ids, minlength=cluster_pair_count).reshape(
        n_user_clusters,
        n_item_clusters,
    )
    sums = np.bincount(
        flat_cluster_ids,
        weights=data.ratings.astype(np.float64, copy=False),
        minlength=cluster_pair_count,
    ).reshape(n_user_clusters, n_item_clusters)

    means = np.zeros((n_user_clusters, n_item_clusters), dtype=np.float64)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero].astype(np.float64, copy=False)
    return means.astype(np.dtype(dtype), copy=False), counts.astype(np.int32, copy=False)


def induce_train_only_clusters(
    data: RatingsData,
    *,
    induction_config: BiasedMFConfig,
    n_user_clusters: int,
    n_item_clusters: int,
    algorithm: str,
    kmeans_n_init: int,
) -> ClusterArtifacts:
    if algorithm != "kmeans":
        raise ValueError("only clustering.algorithm='kmeans' is currently supported")
    if kmeans_n_init <= 0:
        raise ValueError("kmeans_n_init must be positive")

    _validate_cluster_request(
        n_clusters=n_user_clusters,
        n_samples=data.n_users,
        axis_name="user",
    )
    _validate_cluster_request(
        n_clusters=n_item_clusters,
        n_samples=data.n_items,
        axis_name="item",
    )

    started = perf_counter()
    induction_model = BiasedMFRecommender(induction_config).fit(data)
    induction_fit_seconds = perf_counter() - started
    if induction_model.user_factors is None or induction_model.item_factors is None:
        raise RuntimeError("biased_mf induction model did not initialize latent factors")

    started = perf_counter()
    induction_predictions = induction_model.predict_dataset(data)
    induction_predict_seconds = perf_counter() - started
    started = perf_counter()
    induction_train_rmse = rmse(data.ratings, induction_predictions)
    induction_train_rmse_seconds = perf_counter() - started

    user_kmeans = KMeans(
        n_clusters=n_user_clusters,
        n_init=kmeans_n_init,
        random_state=induction_config.seed,
    )
    item_kmeans = KMeans(
        n_clusters=n_item_clusters,
        n_init=kmeans_n_init,
        random_state=induction_config.seed,
    )

    started = perf_counter()
    user_clusters = user_kmeans.fit_predict(induction_model.user_factors.astype(np.float64, copy=False)).astype(
        np.int32, copy=False
    )
    user_kmeans_seconds = perf_counter() - started
    started = perf_counter()
    item_clusters = item_kmeans.fit_predict(induction_model.item_factors.astype(np.float64, copy=False)).astype(
        np.int32, copy=False
    )
    item_kmeans_seconds = perf_counter() - started

    user_cluster_sizes = np.bincount(user_clusters, minlength=n_user_clusters).astype(
        np.int32,
        copy=False,
    )
    item_cluster_sizes = np.bincount(item_clusters, minlength=n_item_clusters).astype(
        np.int32,
        copy=False,
    )
    started = perf_counter()
    r_star_means, r_star_counts = _compute_r_star(
        data,
        user_clusters=user_clusters,
        item_clusters=item_clusters,
        n_user_clusters=n_user_clusters,
        n_item_clusters=n_item_clusters,
        dtype=induction_config.dtype,
    )
    r_star_seconds = perf_counter() - started

    return ClusterArtifacts(
        user_clusters=user_clusters,
        item_clusters=item_clusters,
        user_cluster_sizes=user_cluster_sizes,
        item_cluster_sizes=item_cluster_sizes,
        r_star_means=r_star_means,
        r_star_counts=r_star_counts,
        induction_train_rmse=induction_train_rmse,
        user_kmeans_inertia=float(user_kmeans.inertia_),
        item_kmeans_inertia=float(item_kmeans.inertia_),
        induction_profile=ClusterInductionProfile(
            induction_fit_seconds=float(induction_fit_seconds),
            induction_predict_seconds=float(induction_predict_seconds),
            induction_train_rmse_seconds=float(induction_train_rmse_seconds),
            user_kmeans_seconds=float(user_kmeans_seconds),
            item_kmeans_seconds=float(item_kmeans_seconds),
            r_star_seconds=float(r_star_seconds),
        ),
    )
