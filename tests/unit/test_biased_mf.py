from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from recsys_lab.data.processed import RatingsData
from recsys_lab.metrics import rmse
import recsys_lab.models.biased_mf as biased_mf_module
from recsys_lab.models.biased_mf import BiasedMFConfig, BiasedMFRecommender


def _toy_ratings_data() -> RatingsData:
    return RatingsData(
        user_ids=np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32),
        item_ids=np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32),
        ratings=np.asarray([5.0, 4.0, 4.5, 3.5, 2.0, 1.0, 1.5, 2.5], dtype=np.float32),
        n_users=4,
        n_items=2,
        name="toy",
        rating_min=1.0,
        rating_max=5.0,
        source_manifest_path=Path("toy_manifest.json"),
    )


def test_biased_mf_improves_over_global_mean_baseline() -> None:
    data = _toy_ratings_data()

    baseline = np.full_like(data.ratings, fill_value=float(np.mean(data.ratings)), dtype=np.float64)
    baseline_rmse = rmse(data.ratings, baseline)

    model = BiasedMFRecommender(
        BiasedMFConfig(
            latent_dim=4,
            epochs=40,
            learning_rate=0.02,
            lambda_b=0.01,
            lambda_p=0.01,
            lambda_q=0.01,
            seed=5,
            init_std=0.05,
            dtype="float32",
        )
    )
    model.fit(data)
    trained_rmse = rmse(data.ratings, model.predict_dataset(data))

    assert trained_rmse < baseline_rmse
    assert model.training_backend_effective in {"python", "numba"}


def test_biased_mf_numba_epoch_matches_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("numba")
    data = _toy_ratings_data()
    config = BiasedMFConfig(
        latent_dim=3,
        epochs=4,
        learning_rate=0.015,
        lambda_b=0.02,
        lambda_p=0.03,
        lambda_q=0.04,
        seed=11,
        init_std=0.02,
        dtype="float64",
        training_backend="numba",
    )
    python_model = BiasedMFRecommender(replace(config, training_backend="python"))
    python_model.fit(data)

    numba_model = BiasedMFRecommender(config)
    numba_model.fit(data)

    np.testing.assert_allclose(numba_model.user_bias, python_model.user_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_model.item_bias, python_model.item_bias, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_model.user_factors, python_model.user_factors, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numba_model.item_factors, python_model.item_factors, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        numba_model.predict_dataset(data, clip=False),
        python_model.predict_dataset(data, clip=False),
        rtol=1e-12,
        atol=1e-12,
    )
    assert len(numba_model.epoch_durations_seconds) == config.epochs
    assert len(python_model.epoch_durations_seconds) == config.epochs
    assert python_model.training_backend_effective == "python"
    assert numba_model.training_backend_effective == "numba"


def test_biased_mf_auto_backend_falls_back_to_python(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _toy_ratings_data()

    def _force_kernel_failure(*args: object, **kwargs: object) -> None:
        raise RuntimeError("forced failure")

    monkeypatch.setattr(biased_mf_module, "train_biased_mf_epoch_numba", _force_kernel_failure)
    model = BiasedMFRecommender(BiasedMFConfig(epochs=2, seed=3, training_backend="auto"))

    model.fit(data)

    assert model.training_backend_effective == "python"


def test_biased_mf_explicit_numba_backend_propagates_kernel_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _toy_ratings_data()

    def _force_kernel_failure(*args: object, **kwargs: object) -> None:
        raise RuntimeError("forced failure")

    monkeypatch.setattr(biased_mf_module, "train_biased_mf_epoch_numba", _force_kernel_failure)
    model = BiasedMFRecommender(BiasedMFConfig(epochs=2, seed=3, training_backend="numba"))

    with pytest.raises(RuntimeError, match="forced failure"):
        model.fit(data)


def test_biased_mf_rejects_unknown_training_backend() -> None:
    data = _toy_ratings_data()
    model = BiasedMFRecommender(BiasedMFConfig(training_backend="gpu_magic"))

    with pytest.raises(ValueError, match="training_backend"):
        model.fit(data)
