from pathlib import Path

import numpy as np

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
