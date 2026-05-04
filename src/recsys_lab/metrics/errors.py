from __future__ import annotations

import numpy as np


def _validated_rating_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)

    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if truth.size == 0:
        raise ValueError("rating error metrics are undefined for empty arrays")
    return truth, pred


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, pred = _validated_rating_arrays(y_true, y_pred)

    squared_error = np.square(truth - pred)
    return float(np.sqrt(np.mean(squared_error)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, pred = _validated_rating_arrays(y_true, y_pred)

    return float(np.mean(np.abs(truth - pred)))


def rating_error_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    rating_min: float,
    rating_max: float,
) -> dict[str, float]:
    if rating_min > rating_max:
        raise ValueError("rating_min must be less than or equal to rating_max")

    truth, pred = _validated_rating_arrays(y_true, y_pred)
    residuals = pred - truth
    absolute_errors = np.abs(residuals)
    below_rating_min = pred < float(rating_min)
    above_rating_max = pred > float(rating_max)
    out_of_range = below_rating_min | above_rating_max

    return {
        "rmse": rmse(truth, pred),
        "mae": float(np.mean(absolute_errors)),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "abs_error_p50": float(np.percentile(absolute_errors, 50)),
        "abs_error_p90": float(np.percentile(absolute_errors, 90)),
        "abs_error_p95": float(np.percentile(absolute_errors, 95)),
        "abs_error_max": float(np.max(absolute_errors)),
        "prediction_mean": float(np.mean(pred)),
        "prediction_std": float(np.std(pred)),
        "prediction_min": float(np.min(pred)),
        "prediction_max": float(np.max(pred)),
        "prediction_below_rating_min_rate": float(np.mean(below_rating_min)),
        "prediction_above_rating_max_rate": float(np.mean(above_rating_max)),
        "prediction_out_of_range_rate": float(np.mean(out_of_range)),
    }
