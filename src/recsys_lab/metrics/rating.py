from __future__ import annotations

import math
from typing import Iterable

from recsys_lab.core.types import Prediction


def rmse(predictions: Iterable[Prediction]) -> float:
    rows = tuple(predictions)
    if not rows:
        raise ValueError("rmse requires at least one prediction")
    mse = sum((row.y_true - row.y_pred) ** 2 for row in rows) / len(rows)
    return math.sqrt(mse)


def mae(predictions: Iterable[Prediction]) -> float:
    rows = tuple(predictions)
    if not rows:
        raise ValueError("mae requires at least one prediction")
    return sum(abs(row.y_true - row.y_pred) for row in rows) / len(rows)
