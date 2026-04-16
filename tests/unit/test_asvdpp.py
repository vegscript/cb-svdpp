from pathlib import Path

import numpy as np

from recsys_lab.data.histories import (
    build_user_explicit_feedback_index,
    build_user_history_index,
)
from recsys_lab.data.processed import RatingsData
from recsys_lab.metrics import rmse
from recsys_lab.models.asvdpp import ASVDppConfig, ASVDppRecommender


def test_asvdpp_improves_over_global_mean_baseline() -> None:
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

    model = ASVDppRecommender(
        ASVDppConfig(
            latent_dim=6,
            epochs=25,
            learning_rate=0.01,
            lambda_b=0.02,
            lambda_p=0.02,
            lambda_q=0.02,
            lambda_x=0.02,
            lambda_y=0.02,
            seed=13,
            init_std=0.05,
            dtype="float32",
            implicit_policy="ratings_as_implicit",
            residual_weight_contract="detached",
        )
    )
    model.fit(data)
    trained_rmse = rmse(data.ratings, model.predict_dataset(data))

    assert trained_rmse < baseline_rmse


def test_asvdpp_predict_many_matches_scalar_predict_on_repeated_subset() -> None:
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

    model = ASVDppRecommender(
        ASVDppConfig(
            latent_dim=5,
            epochs=6,
            learning_rate=0.01,
            lambda_b=0.02,
            lambda_p=0.02,
            lambda_q=0.02,
            lambda_x=0.02,
            lambda_y=0.02,
            seed=17,
            init_std=0.04,
            dtype="float32",
            implicit_policy="ratings_as_implicit",
            residual_weight_contract="detached",
        )
    )
    model.fit(data)
    assert model._user_context_cache is None

    user_ids = np.asarray([2, 0, 2, 1, 0], dtype=np.int32)
    item_ids = np.asarray([3, 1, 1, 0, 2], dtype=np.int32)
    batch_predictions = model.predict_many(user_ids, item_ids, clip=False)
    assert model._user_context_cache is not None
    cached_contexts = model._user_context_cache.copy()
    scalar_predictions = np.asarray(
        [model.predict(int(user_id), int(item_id), clip=False) for user_id, item_id in zip(user_ids, item_ids)],
        dtype=np.float64,
    )

    np.testing.assert_allclose(batch_predictions, scalar_predictions, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(model._user_context_cache, cached_contexts, rtol=0.0, atol=0.0)


def test_asvdpp_fit_accepts_precomputed_indices_without_changing_result() -> None:
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
    config = ASVDppConfig(
        latent_dim=4,
        epochs=4,
        learning_rate=0.01,
        lambda_b=0.02,
        lambda_p=0.02,
        lambda_q=0.02,
        lambda_x=0.02,
        lambda_y=0.02,
        seed=43,
        init_std=0.02,
        dtype="float64",
        implicit_policy="ratings_as_implicit",
        residual_weight_contract="detached",
    )

    baseline_model = ASVDppRecommender(config)
    baseline_model.fit(data)

    explicit_feedback = build_user_explicit_feedback_index(data, dtype=config.dtype)
    implicit_history = build_user_history_index(data, dtype=config.dtype)
    optimized_model = ASVDppRecommender(config)
    optimized_model.fit(
        data,
        explicit_feedback=explicit_feedback,
        implicit_history=implicit_history,
    )

    np.testing.assert_allclose(optimized_model.user_bias, baseline_model.user_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(optimized_model.item_bias, baseline_model.item_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(optimized_model.user_factors, baseline_model.user_factors, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(optimized_model.item_factors, baseline_model.item_factors, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        optimized_model.explicit_factors,
        baseline_model.explicit_factors,
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
        optimized_model.predict_dataset(data, clip=False),
        baseline_model.predict_dataset(data, clip=False),
        rtol=1e-12,
        atol=1e-12,
    )
