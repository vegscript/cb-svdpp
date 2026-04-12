from pathlib import Path

import numpy as np

from recsys_lab.data.processed import RatingsData
from recsys_lab.metrics import rmse
from recsys_lab.models.biased_mf import BiasedMFConfig, BiasedMFRecommender


def test_biased_mf_improves_over_global_mean_baseline() -> None:
    data = RatingsData(
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
