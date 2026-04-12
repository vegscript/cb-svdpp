from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Interaction:
    user_id: int
    item_id: int
    rating: float


@dataclass(frozen=True, slots=True)
class Prediction:
    user_id: int
    item_id: int
    y_true: float
    y_pred: float
