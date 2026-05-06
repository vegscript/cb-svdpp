from pathlib import Path

import numpy as np
from pytest import MonkeyPatch

import recsys_lab.models.asymmetric_svd as asymmetric_svd_module
from recsys_lab.data.processed import RatingsData
from recsys_lab.metrics import rmse
from recsys_lab.models.asymmetric_svd import AsymmetricSVDConfig, AsymmetricSVDRecommender


def test_asymmetric_svd_improves_over_global_mean_baseline() -> None:
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

    model = AsymmetricSVDRecommender(
        AsymmetricSVDConfig(
            latent_dim=6,
            epochs=25,
            learning_rate=0.01,
            lambda_b=0.02,
            lambda_q=0.02,
            lambda_x=0.02,
            lambda_y=0.02,
            seed=11,
            init_std=0.05,
            dtype="float32",
            implicit_policy="ratings_as_implicit",
            residual_weight_contract="detached",
        )
    )
    model.fit(data)
    trained_rmse = rmse(data.ratings, model.predict_dataset(data))

    assert trained_rmse < baseline_rmse


def test_asymmetric_svd_numba_kernel_matches_python_reference(monkeypatch: MonkeyPatch) -> None:
    data = RatingsData(
        user_ids=np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 0, 2, 1, 3], dtype=np.int32),
        item_ids=np.asarray([0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 4, 1], dtype=np.int32),
        ratings=np.asarray([4.5, 4.0, 3.5, 3.0, 5.0, 4.0, 2.0, 2.5, 4.2, 4.8, 3.2, 2.8]),
        n_users=4,
        n_items=5,
        name="toy_equivalence",
        rating_min=1.0,
        rating_max=5.0,
        source_manifest_path=Path("toy_equivalence_manifest.json"),
    )
    config = AsymmetricSVDConfig(
        latent_dim=5,
        epochs=3,
        learning_rate=0.01,
        lambda_b=0.01,
        lambda_q=0.03,
        lambda_x=0.04,
        lambda_y=0.05,
        seed=123,
        init_std=0.02,
        dtype="float64",
        implicit_policy="ratings_as_implicit",
        residual_weight_contract="detached",
    )

    numba_model = AsymmetricSVDRecommender(config).fit(data)

    def fail_kernel(*args: object, **kwargs: object) -> None:
        raise RuntimeError("force python reference")

    monkeypatch.setattr(asymmetric_svd_module, "train_asymmetric_svd_epoch_numba", fail_kernel)
    python_model = AsymmetricSVDRecommender(config).fit(data)

    for field_name in (
        "user_bias",
        "item_bias",
        "item_factors",
        "explicit_factors",
        "implicit_factors",
    ):
        numba_values = getattr(numba_model, field_name)
        python_values = getattr(python_model, field_name)
        assert numba_values.shape == python_values.shape
        assert np.all(np.isfinite(numba_values))
        np.testing.assert_allclose(numba_values, python_values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        numba_model.predict_dataset(data, clip=False),
        python_model.predict_dataset(data, clip=False),
        rtol=1e-12,
        atol=1e-12,
    )
    assert len(numba_model.epoch_durations_seconds) == config.epochs
    assert len(python_model.epoch_durations_seconds) == config.epochs
