from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from recsys_lab.data.processed import RatingsData
from recsys_lab.metrics import rmse
import recsys_lab.models.svdpp as svdpp_module
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
