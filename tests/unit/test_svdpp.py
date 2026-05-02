from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import recsys_lab.models.svdpp as svdpp_module
from recsys_lab.data.histories import build_user_history_index
from recsys_lab.data.processed import RatingsData
from recsys_lab.metrics import rmse
from recsys_lab.models.svdpp import SVDppConfig, SVDppRecommender


def _toy_ratings_data() -> RatingsData:
    return RatingsData(
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


def test_svdpp_improves_over_global_mean_baseline() -> None:
    data = _toy_ratings_data()

    baseline = np.full_like(data.ratings, fill_value=float(np.mean(data.ratings)), dtype=np.float64)
    baseline_rmse = rmse(data.ratings, baseline)

    model = SVDppRecommender(
        SVDppConfig(
            latent_dim=6,
            epochs=25,
            learning_rate=0.02,
            lambda_b=0.01,
            lambda_p=0.01,
            lambda_q=0.01,
            lambda_y=0.01,
            seed=9,
            init_std=0.05,
            dtype="float32",
        )
    )
    model.fit(data)
    trained_rmse = rmse(data.ratings, model.predict_dataset(data))

    assert trained_rmse < baseline_rmse
    assert model.training_backend_effective in {"python", "numba"}


def test_svdpp_numba_epoch_matches_python_fallback() -> None:
    pytest.importorskip("numba")
    data = _toy_ratings_data()
    config = SVDppConfig(
        latent_dim=4,
        epochs=4,
        learning_rate=0.015,
        lambda_b=0.02,
        lambda_p=0.03,
        lambda_q=0.04,
        lambda_y=0.05,
        seed=13,
        init_std=0.02,
        dtype="float64",
        training_backend="numba",
    )

    python_model = SVDppRecommender(replace(config, training_backend="python"))
    python_model.fit(data)

    numba_model = SVDppRecommender(config)
    numba_model.fit(data)

    np.testing.assert_allclose(numba_model.user_bias, python_model.user_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_model.item_bias, python_model.item_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_model.user_factors, python_model.user_factors, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_model.item_factors, python_model.item_factors, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        numba_model.implicit_factors,
        python_model.implicit_factors,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        numba_model.predict_dataset(data, clip=False),
        python_model.predict_dataset(data, clip=False),
        rtol=1e-12,
        atol=1e-12,
    )
    assert python_model.training_backend_effective == "python"
    assert numba_model.training_backend_effective == "numba"
    assert len(numba_model.epoch_durations_seconds) == config.epochs
    assert len(python_model.epoch_durations_seconds) == config.epochs


def test_svdpp_auto_backend_falls_back_to_python(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _toy_ratings_data()

    def _force_kernel_failure(*args: object, **kwargs: object) -> None:
        raise RuntimeError("forced failure")

    monkeypatch.setattr(svdpp_module, "train_svdpp_epoch_numba", _force_kernel_failure)
    model = SVDppRecommender(SVDppConfig(epochs=2, seed=3, training_backend="auto"))

    model.fit(data)

    assert model.training_backend_effective == "python"


def test_svdpp_explicit_numba_backend_propagates_kernel_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _toy_ratings_data()

    def _force_kernel_failure(*args: object, **kwargs: object) -> None:
        raise RuntimeError("forced failure")

    monkeypatch.setattr(svdpp_module, "train_svdpp_epoch_numba", _force_kernel_failure)
    model = SVDppRecommender(SVDppConfig(epochs=2, seed=3, training_backend="numba"))

    with pytest.raises(RuntimeError, match="forced failure"):
        model.fit(data)


def test_svdpp_rejects_unknown_training_backend() -> None:
    data = _toy_ratings_data()
    model = SVDppRecommender(SVDppConfig(training_backend="gpu_magic"))

    with pytest.raises(ValueError, match="training_backend"):
        model.fit(data)


def test_svdpp_predict_many_matches_scalar_predict_on_repeated_subset() -> None:
    data = _toy_ratings_data()
    model = SVDppRecommender(
        SVDppConfig(
            latent_dim=5,
            epochs=6,
            learning_rate=0.02,
            lambda_b=0.01,
            lambda_p=0.01,
            lambda_q=0.01,
            lambda_y=0.01,
            seed=19,
            init_std=0.04,
            dtype="float32",
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
        [
            model.predict(int(user_id), int(item_id), clip=False)
            for user_id, item_id in zip(user_ids, item_ids, strict=True)
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(batch_predictions, scalar_predictions, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(model._user_context_cache, cached_contexts, rtol=0.0, atol=0.0)


def test_svdpp_fit_accepts_precomputed_user_histories_without_changing_result() -> None:
    data = _toy_ratings_data()
    config = SVDppConfig(
        latent_dim=4,
        epochs=4,
        learning_rate=0.015,
        lambda_b=0.02,
        lambda_p=0.03,
        lambda_q=0.04,
        lambda_y=0.05,
        seed=41,
        init_std=0.02,
        dtype="float64",
        training_backend="python",
    )

    baseline_model = SVDppRecommender(config)
    baseline_model.fit(data)

    precomputed_histories = build_user_history_index(data, dtype=config.dtype)
    optimized_model = SVDppRecommender(config)
    optimized_model.fit(data, user_histories=precomputed_histories)

    np.testing.assert_allclose(optimized_model.user_bias, baseline_model.user_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(optimized_model.item_bias, baseline_model.item_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(optimized_model.user_factors, baseline_model.user_factors, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(optimized_model.item_factors, baseline_model.item_factors, rtol=1e-12, atol=1e-12)
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


def test_svdpp_subset_view_matches_materialized_subset_training_result() -> None:
    data = _toy_ratings_data()
    subset_view = data.subset(np.asarray([0, 1, 3, 4, 6, 7], dtype=np.int64), name="toy:train_view")
    subset_materialized = subset_view.materialize(force_copy=True)
    config = SVDppConfig(
        latent_dim=4,
        epochs=4,
        learning_rate=0.015,
        lambda_b=0.02,
        lambda_p=0.03,
        lambda_q=0.04,
        lambda_y=0.05,
        seed=23,
        init_std=0.02,
        dtype="float64",
        training_backend="python",
    )

    view_model = SVDppRecommender(config)
    materialized_model = SVDppRecommender(config)
    view_model.fit(subset_view)
    materialized_model.fit(subset_materialized)

    np.testing.assert_allclose(view_model.user_bias, materialized_model.user_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(view_model.item_bias, materialized_model.item_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(view_model.user_factors, materialized_model.user_factors, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(view_model.item_factors, materialized_model.item_factors, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        view_model.implicit_factors,
        materialized_model.implicit_factors,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        view_model.predict_dataset(subset_materialized, clip=False),
        materialized_model.predict_dataset(subset_materialized, clip=False),
        rtol=1e-12,
        atol=1e-12,
    )
