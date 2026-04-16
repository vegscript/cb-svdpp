from pathlib import Path

import numpy as np

from recsys_lab.data.processed import RatingsData
from recsys_lab.experiments.inference_benchmarking import (
    build_repeated_sorted_prefix_query,
    time_inference_variant,
)
from recsys_lab.experiments.svdpp_inference_benchmark import (
    _predict_many_full_context_reference,
)
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


def test_predict_many_full_context_reference_matches_current_predict_many() -> None:
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
            seed=23,
            init_std=0.04,
            dtype="float32",
        )
    )
    model.fit(data)

    user_ids = np.asarray([2, 0, 2, 1, 0], dtype=np.int32)
    item_ids = np.asarray([3, 1, 1, 0, 2], dtype=np.int32)

    reference_predictions = _predict_many_full_context_reference(
        model,
        user_ids,
        item_ids,
        clip=False,
    )
    optimized_predictions = model.predict_many(user_ids, item_ids, clip=False)

    np.testing.assert_allclose(reference_predictions, optimized_predictions, rtol=1e-6, atol=1e-6)


def test_build_repeated_sorted_prefix_query_sorts_and_repeats_prefix() -> None:
    user_ids = np.asarray([2, 0, 1, 0], dtype=np.int64)
    item_ids = np.asarray([7, 8, 9, 10], dtype=np.int64)

    repeated_users, repeated_items = build_repeated_sorted_prefix_query(
        user_ids=user_ids,
        item_ids=item_ids,
        prefix_rows=3,
        repeat_factor=2,
    )

    np.testing.assert_array_equal(repeated_users, np.asarray([0, 0, 1, 0, 0, 1], dtype=np.int64))
    np.testing.assert_array_equal(repeated_items, np.asarray([8, 10, 9, 8, 10, 9], dtype=np.int64))


def test_time_inference_variant_matches_reference_and_optimized_predictions() -> None:
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
            seed=29,
            init_std=0.04,
            dtype="float32",
        )
    )
    model.fit(data)

    user_ids = np.asarray([2, 0, 2, 1, 0], dtype=np.int32)
    item_ids = np.asarray([3, 1, 1, 0, 2], dtype=np.int32)

    _, reference_predictions = time_inference_variant(
        predict_many_fn=lambda users, items, clip=False: _predict_many_full_context_reference(
            model,
            users,
            items,
            clip=clip,
        ),
        user_ids=user_ids,
        item_ids=item_ids,
        repeats=2,
    )
    _, optimized_predictions = time_inference_variant(
        predict_many_fn=model.predict_many,
        user_ids=user_ids,
        item_ids=item_ids,
        repeats=2,
    )

    np.testing.assert_allclose(reference_predictions, optimized_predictions, rtol=1e-6, atol=1e-6)
