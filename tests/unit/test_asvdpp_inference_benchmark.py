from pathlib import Path

import numpy as np

from recsys_lab.data.processed import RatingsData
from recsys_lab.experiments.asvdpp_inference_benchmark import _predict_many_unique_context_reference
from recsys_lab.models.asvdpp import ASVDppConfig, ASVDppRecommender


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


def test_predict_many_unique_context_reference_matches_current_predict_many() -> None:
    data = _toy_ratings_data()
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
            seed=31,
            init_std=0.04,
            dtype="float32",
            implicit_policy="ratings_as_implicit",
            residual_weight_contract="detached",
        )
    )
    model.fit(data)

    user_ids = np.asarray([2, 0, 2, 1, 0], dtype=np.int32)
    item_ids = np.asarray([3, 1, 1, 0, 2], dtype=np.int32)

    reference_predictions = _predict_many_unique_context_reference(
        model,
        user_ids,
        item_ids,
        clip=False,
    )
    optimized_predictions = model.predict_many(user_ids, item_ids, clip=False)

    np.testing.assert_allclose(reference_predictions, optimized_predictions, rtol=1e-6, atol=1e-6)
