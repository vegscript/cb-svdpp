from pathlib import Path

import numpy as np

from recsys_lab.data.histories import build_user_cluster_count_index, build_user_history_index
from recsys_lab.data.processed import RatingsData
from recsys_lab.metrics import rmse
from recsys_lab.models.cb_svdpp import CBSVDppConfig, CBSVDppRecommender


def test_cb_svdpp_improves_over_global_mean_baseline() -> None:
    data = RatingsData(
        user_ids=np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32),
        item_ids=np.asarray([0, 1, 2, 0, 1, 3, 1, 2, 3], dtype=np.int32),
        ratings=np.asarray([5.0, 4.5, 4.0, 4.5, 4.0, 3.5, 2.0, 2.5, 3.0], dtype=np.float32),
        n_users=3,
        n_items=4,
        name="toy",
        rating_min=2.0,
        rating_max=5.0,
        source_manifest_path=Path("toy_manifest.json"),
    )

    baseline = np.full_like(data.ratings, fill_value=float(np.mean(data.ratings)), dtype=np.float64)
    baseline_rmse = rmse(data.ratings, baseline)

    model = CBSVDppRecommender(
        CBSVDppConfig(
            latent_dim=6,
            epochs=20,
            learning_rate=0.02,
            lambda_b=0.01,
            lambda_p=0.01,
            lambda_q=0.01,
            lambda_y=0.01,
            lambda_pC=0.01,
            lambda_qC=0.01,
            lambda_yC=0.01,
            alpha=0.2,
            seed=11,
            init_std=0.05,
            dtype="float32",
            implicit_policy="ratings_as_implicit",
        ),
        user_clusters=np.asarray([0, 0, 1], dtype=np.int32),
        item_clusters=np.asarray([0, 0, 1, 1], dtype=np.int32),
        n_user_clusters=2,
        n_item_clusters=2,
    )
    model.fit(data)
    trained_rmse = rmse(data.ratings, model.predict_dataset(data))

    assert trained_rmse < baseline_rmse


def test_cb_svdpp_predict_many_matches_scalar_predict_on_repeated_subset() -> None:
    data = RatingsData(
        user_ids=np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32),
        item_ids=np.asarray([0, 1, 2, 0, 1, 3, 1, 2, 3], dtype=np.int32),
        ratings=np.asarray([5.0, 4.5, 4.0, 4.5, 4.0, 3.5, 2.0, 2.5, 3.0], dtype=np.float32),
        n_users=3,
        n_items=4,
        name="toy",
        rating_min=2.0,
        rating_max=5.0,
        source_manifest_path=Path("toy_manifest.json"),
    )

    model = CBSVDppRecommender(
        CBSVDppConfig(
            latent_dim=5,
            epochs=6,
            learning_rate=0.02,
            lambda_b=0.01,
            lambda_p=0.01,
            lambda_q=0.01,
            lambda_y=0.01,
            lambda_pC=0.01,
            lambda_qC=0.01,
            lambda_yC=0.01,
            alpha=0.2,
            seed=23,
            init_std=0.04,
            dtype="float32",
            implicit_policy="ratings_as_implicit",
        ),
        user_clusters=np.asarray([0, 0, 1], dtype=np.int32),
        item_clusters=np.asarray([0, 0, 1, 1], dtype=np.int32),
        n_user_clusters=2,
        n_item_clusters=2,
    )
    model.fit(data)
    assert model._user_context_cache is None
    assert model._mixed_item_factors_cache is None

    user_ids = np.asarray([2, 0, 2, 1, 0], dtype=np.int32)
    item_ids = np.asarray([3, 1, 1, 0, 2], dtype=np.int32)
    batch_predictions = model.predict_many(user_ids, item_ids, clip=False)
    assert model._user_context_cache is not None
    assert model._mixed_item_factors_cache is not None
    cached_contexts = model._user_context_cache.copy()
    cached_item_factors = model._mixed_item_factors_cache.copy()
    scalar_predictions = np.asarray(
        [model.predict(int(user_id), int(item_id), clip=False) for user_id, item_id in zip(user_ids, item_ids)],
        dtype=np.float64,
    )

    np.testing.assert_allclose(batch_predictions, scalar_predictions, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(model._user_context_cache, cached_contexts, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(model._mixed_item_factors_cache, cached_item_factors, rtol=0.0, atol=0.0)


def test_cb_svdpp_fit_accepts_precomputed_indices_without_changing_result() -> None:
    data = RatingsData(
        user_ids=np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32),
        item_ids=np.asarray([0, 1, 2, 0, 1, 3, 1, 2, 3], dtype=np.int32),
        ratings=np.asarray([5.0, 4.5, 4.0, 4.5, 4.0, 3.5, 2.0, 2.5, 3.0], dtype=np.float32),
        n_users=3,
        n_items=4,
        name="toy",
        rating_min=2.0,
        rating_max=5.0,
        source_manifest_path=Path("toy_manifest.json"),
    )
    config = CBSVDppConfig(
        latent_dim=4,
        epochs=4,
        learning_rate=0.02,
        lambda_b=0.01,
        lambda_p=0.01,
        lambda_q=0.01,
        lambda_y=0.01,
        lambda_pC=0.01,
        lambda_qC=0.01,
        lambda_yC=0.01,
        alpha=0.2,
        seed=47,
        init_std=0.02,
        dtype="float64",
        implicit_policy="ratings_as_implicit",
    )
    user_clusters = np.asarray([0, 0, 1], dtype=np.int32)
    item_clusters = np.asarray([0, 0, 1, 1], dtype=np.int32)

    baseline_model = CBSVDppRecommender(
        config,
        user_clusters=user_clusters,
        item_clusters=item_clusters,
        n_user_clusters=2,
        n_item_clusters=2,
    )
    baseline_model.fit(data)

    user_histories = build_user_history_index(data, dtype=config.dtype)
    cluster_histories = build_user_cluster_count_index(
        user_histories,
        item_clusters,
        n_clusters=2,
    )
    optimized_model = CBSVDppRecommender(
        config,
        user_clusters=user_clusters,
        item_clusters=item_clusters,
        n_user_clusters=2,
        n_item_clusters=2,
    )
    optimized_model.fit(
        data,
        user_histories=user_histories,
        user_cluster_histories=cluster_histories,
    )

    np.testing.assert_allclose(optimized_model.user_bias, baseline_model.user_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(optimized_model.item_bias, baseline_model.item_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(optimized_model.user_factors, baseline_model.user_factors, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(optimized_model.item_factors, baseline_model.item_factors, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        optimized_model.user_cluster_factors,
        baseline_model.user_cluster_factors,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        optimized_model.item_cluster_factors,
        baseline_model.item_cluster_factors,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        optimized_model.implicit_factors,
        baseline_model.implicit_factors,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        optimized_model.implicit_cluster_factors,
        baseline_model.implicit_cluster_factors,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        optimized_model.predict_dataset(data, clip=False),
        baseline_model.predict_dataset(data, clip=False),
        rtol=1e-12,
        atol=1e-12,
    )
