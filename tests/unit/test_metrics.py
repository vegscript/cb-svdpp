import numpy as np
import pytest

from recsys_lab.metrics import mae, rating_error_metrics, rmse


def test_rmse_matches_known_value() -> None:
    truth = np.asarray([3.0, 4.0, 5.0], dtype=np.float32)
    pred = np.asarray([2.0, 4.0, 4.0], dtype=np.float32)
    assert rmse(truth, pred) == pytest.approx((2.0 / 3.0) ** 0.5)


def test_mae_matches_known_value() -> None:
    truth = np.asarray([3.0, 4.0, 5.0], dtype=np.float32)
    pred = np.asarray([2.0, 4.0, 4.0], dtype=np.float32)
    assert mae(truth, pred) == pytest.approx(2.0 / 3.0)


def test_rating_error_metrics_cover_error_and_prediction_distribution() -> None:
    truth = np.asarray([1.0, 3.0, 5.0, 5.0], dtype=np.float32)
    pred = np.asarray([0.0, 2.0, 6.0, 5.5], dtype=np.float32)

    metrics = rating_error_metrics(truth, pred, rating_min=1.0, rating_max=5.0)

    assert metrics == pytest.approx(
        {
            "rmse": np.sqrt(3.25 / 4.0),
            "mae": 0.875,
            "residual_mean": -0.125,
            "residual_std": np.std(np.asarray([-1.0, -1.0, 1.0, 0.5], dtype=np.float64)),
            "abs_error_p50": 1.0,
            "abs_error_p90": 1.0,
            "abs_error_p95": 1.0,
            "abs_error_max": 1.0,
            "prediction_mean": 3.375,
            "prediction_std": np.std(np.asarray([0.0, 2.0, 6.0, 5.5], dtype=np.float64)),
            "prediction_min": 0.0,
            "prediction_max": 6.0,
            "prediction_below_rating_min_rate": 0.25,
            "prediction_above_rating_max_rate": 0.5,
            "prediction_out_of_range_rate": 0.75,
        }
    )


def test_rmse_rejects_empty_arrays() -> None:
    with pytest.raises(ValueError):
        rmse(np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32))


def test_rating_error_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        rating_error_metrics(
            np.asarray([1.0, 2.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            rating_min=1.0,
            rating_max=5.0,
        )


def test_rating_error_metrics_reject_invalid_rating_range() -> None:
    with pytest.raises(ValueError, match="rating_min"):
        rating_error_metrics(
            np.asarray([1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            rating_min=5.0,
            rating_max=1.0,
        )
