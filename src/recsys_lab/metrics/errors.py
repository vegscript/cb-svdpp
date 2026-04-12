from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)

    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if truth.size == 0:
        raise ValueError("rmse is undefined for empty arrays")

    squared_error = np.square(truth - pred)
    return float(np.sqrt(np.mean(squared_error)))
