from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from recsys_lab.data.schemas import InteractionDataset


@dataclass
class BaseRecommender:
    name: str
    is_fitted: bool = field(default=False, init=False)

    def fit(self, dataset: InteractionDataset) -> "BaseRecommender":
        raise NotImplementedError

    def predict(self, user_id: int, item_id: int) -> float:
        raise NotImplementedError

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted")

    @staticmethod
    def _rng(seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)
