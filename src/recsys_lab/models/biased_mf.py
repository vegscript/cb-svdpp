from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recsys_lab.data.processed import RatingsData


@dataclass(frozen=True, slots=True)
class BiasedMFConfig:
    latent_dim: int = 50
    epochs: int = 20
    learning_rate: float = 0.01
    lambda_b: float = 0.02
    lambda_p: float = 0.02
    lambda_q: float = 0.02
    seed: int = 42
    init_std: float = 0.1
    dtype: str = "float32"


class BiasedMFRecommender:
    def __init__(self, config: BiasedMFConfig) -> None:
        self.config = config
        self.is_fitted = False
        self.global_mean = 0.0
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.rating_min = 0.0
        self.rating_max = 0.0

    def _parameter_dtype(self) -> np.dtype:
        if self.config.dtype not in {"float32", "float64"}:
            raise ValueError("biased_mf dtype must be 'float32' or 'float64'")
        return np.dtype(self.config.dtype)

    def fit(self, data: RatingsData) -> "BiasedMFRecommender":
        rng = np.random.default_rng(self.config.seed)
        parameter_dtype = self._parameter_dtype()
        self.global_mean = float(np.mean(data.ratings))
        self.rating_min = data.rating_min
        self.rating_max = data.rating_max
        self.user_bias = np.zeros(data.n_users, dtype=parameter_dtype)
        self.item_bias = np.zeros(data.n_items, dtype=parameter_dtype)
        self.user_factors = rng.normal(
            0.0, self.config.init_std, size=(data.n_users, self.config.latent_dim)
        ).astype(parameter_dtype, copy=False)
        self.item_factors = rng.normal(
            0.0, self.config.init_std, size=(data.n_items, self.config.latent_dim)
        ).astype(parameter_dtype, copy=False)

        order = np.arange(len(data), dtype=np.int64)
        user_ids = data.user_ids
        item_ids = data.item_ids
        ratings = data.ratings.astype(parameter_dtype, copy=False)

        for _ in range(self.config.epochs):
            rng.shuffle(order)
            for idx in order:
                user_id = int(user_ids[idx])
                item_id = int(item_ids[idx])
                rating = float(ratings[idx])

                prediction = self._predict_unchecked(user_id, item_id)
                error = rating - prediction

                user_bias_old = self.user_bias[user_id]
                item_bias_old = self.item_bias[item_id]
                user_vector_old = self.user_factors[user_id].copy()
                item_vector_old = self.item_factors[item_id].copy()

                self.user_bias[user_id] = user_bias_old + self.config.learning_rate * (
                    error - self.config.lambda_b * user_bias_old
                )
                self.item_bias[item_id] = item_bias_old + self.config.learning_rate * (
                    error - self.config.lambda_b * item_bias_old
                )
                self.user_factors[user_id] = user_vector_old + self.config.learning_rate * (
                    error * item_vector_old - self.config.lambda_p * user_vector_old
                )
                self.item_factors[item_id] = item_vector_old + self.config.learning_rate * (
                    error * user_vector_old - self.config.lambda_q * item_vector_old
                )

        self.is_fitted = True
        return self

    def _predict_unchecked(self, user_id: int, item_id: int) -> float:
        if (
            self.user_bias is None
            or self.item_bias is None
            or self.user_factors is None
            or self.item_factors is None
        ):
            raise RuntimeError("model parameters are not initialized")

        return float(
            self.global_mean
            + self.user_bias[user_id]
            + self.item_bias[item_id]
            + np.dot(self.user_factors[user_id], self.item_factors[item_id])
        )

    def predict(self, user_id: int, item_id: int, *, clip: bool = True) -> float:
        if not self.is_fitted:
            raise RuntimeError("biased_mf is not fitted")
        prediction = self._predict_unchecked(user_id, item_id)
        if clip:
            prediction = float(np.clip(prediction, self.rating_min, self.rating_max))
        return prediction

    def predict_many(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        *,
        clip: bool = True,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("biased_mf is not fitted")
        if (
            self.user_bias is None
            or self.item_bias is None
            or self.user_factors is None
            or self.item_factors is None
        ):
            raise RuntimeError("model parameters are not initialized")

        users = np.asarray(user_ids, dtype=np.int64)
        items = np.asarray(item_ids, dtype=np.int64)
        predictions = (
            self.global_mean
            + self.user_bias[users]
            + self.item_bias[items]
            + np.sum(self.user_factors[users] * self.item_factors[items], axis=1)
        )
        if clip:
            predictions = np.clip(predictions, self.rating_min, self.rating_max)
        return np.asarray(predictions, dtype=np.float64)

    def predict_dataset(self, data: RatingsData, *, clip: bool = True) -> np.ndarray:
        return self.predict_many(data.user_ids, data.item_ids, clip=clip)
