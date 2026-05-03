from pathlib import Path

import numpy as np
from pytest import MonkeyPatch

import recsys_lab.models.cb_asvdpp as cb_asvdpp_module
from recsys_lab.data.processed import RatingsData
from recsys_lab.metrics import rmse
from recsys_lab.models.cb_asvdpp import CBASVDppConfig, CBASVDppRecommender


def test_cb_asvdpp_improves_over_global_mean_baseline() -> None:
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

    model = CBASVDppRecommender(
        CBASVDppConfig(
            latent_dim=6,
            epochs=20,
            learning_rate=0.02,
            lambda_b=0.01,
            lambda_p=0.01,
            lambda_q=0.01,
            lambda_x=0.01,
            lambda_y=0.01,
            lambda_pC=0.01,
            lambda_qC=0.01,
            lambda_xC=0.01,
            lambda_yC=0.01,
            alpha=0.2,
            seed=17,
            init_std=0.05,
            dtype="float32",
            implicit_policy="ratings_as_implicit",
            residual_weight_contract="detached",
        ),
        user_clusters=np.asarray([0, 0, 1], dtype=np.int32),
        item_clusters=np.asarray([0, 0, 1, 1], dtype=np.int32),
        n_user_clusters=2,
        n_item_clusters=2,
    )
    model.fit(data)
    trained_rmse = rmse(data.ratings, model.predict_dataset(data))

    assert trained_rmse < baseline_rmse


def test_cb_asvdpp_numba_kernel_matches_python_reference(monkeypatch: MonkeyPatch) -> None:
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
    config = CBASVDppConfig(
        latent_dim=5,
        epochs=3,
        learning_rate=0.01,
        lambda_b=0.01,
        lambda_p=0.02,
        lambda_q=0.03,
        lambda_x=0.04,
        lambda_y=0.05,
        lambda_pC=0.06,
        lambda_qC=0.07,
        lambda_xC=0.08,
        lambda_yC=0.09,
        alpha=0.2,
        seed=123,
        init_std=0.02,
        dtype="float64",
        implicit_policy="ratings_as_implicit",
        residual_weight_contract="detached",
    )
    user_clusters = np.asarray([0, 0, 1, 1], dtype=np.int32)
    item_clusters = np.asarray([0, 1, 1, 2, 2], dtype=np.int32)

    def make_model() -> CBASVDppRecommender:
        return CBASVDppRecommender(
            config,
            user_clusters=user_clusters,
            item_clusters=item_clusters,
            n_user_clusters=2,
            n_item_clusters=3,
        )

    numba_model = make_model().fit(data)

    def fail_kernel(*args: object, **kwargs: object) -> None:
        raise RuntimeError("force python reference")

    monkeypatch.setattr(cb_asvdpp_module, "train_cb_asvdpp_epoch_numba", fail_kernel)
    python_model = make_model().fit(data)

    for field_name in (
        "user_bias",
        "item_bias",
        "user_factors",
        "item_factors",
        "explicit_factors",
        "implicit_factors",
        "user_cluster_factors",
        "item_cluster_factors",
        "explicit_cluster_factors",
        "implicit_cluster_factors",
    ):
        np.testing.assert_allclose(
            getattr(numba_model, field_name),
            getattr(python_model, field_name),
            rtol=1e-6,
            atol=1e-6,
        )
    np.testing.assert_allclose(
        numba_model.predict_dataset(data, clip=False),
        python_model.predict_dataset(data, clip=False),
        rtol=1e-6,
        atol=1e-6,
    )
