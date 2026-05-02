from __future__ import annotations

from typing import Protocol

from recsys_lab.data.schemas import InteractionDataset


class Recommender(Protocol):
    name: str

    def fit(self, dataset: InteractionDataset) -> "Recommender":
        """Train the model on a canonical interaction dataset."""

    def predict(self, user_id: int, item_id: int) -> float:
        """Predict the rating for a given user-item pair."""
