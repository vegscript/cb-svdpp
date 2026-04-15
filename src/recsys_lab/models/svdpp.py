from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from recsys_lab.data.histories import UserHistoryIndex, build_user_history_index
from recsys_lab.data.processed import RatingsData


@dataclass(frozen=True, slots=True)
class SVDppConfig:
    latent_dim: int = 50
    epochs: int = 20
    learning_rate: float = 0.01
    lambda_b: float = 0.02
    lambda_p: float = 0.02
    lambda_q: float = 0.02
    lambda_y: float = 0.02
    seed: int = 42
    init_std: float = 0.1
    dtype: str = "float32"
    implicit_policy: str = "ratings_as_implicit"


class SVDppRecommender:
    def __init__(self, config: SVDppConfig) -> None:
        self.config = config
        self.is_fitted = False
        self.global_mean = 0.0
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.implicit_factors: np.ndarray | None = None
        self.user_histories: UserHistoryIndex | None = None
        self.rating_min = 0.0
        self.rating_max = 0.0
        self.epoch_durations_seconds: list[float] = []

    def _parameter_dtype(self) -> np.dtype:
        if self.config.dtype not in {"float32", "float64"}:
            raise ValueError("svdpp dtype must be 'float32' or 'float64'")
        return np.dtype(self.config.dtype)

    def _validate_policy(self) -> None:
        if self.config.implicit_policy != "ratings_as_implicit":
            raise ValueError("only implicit_policy='ratings_as_implicit' is currently supported")

    def fit(self, data: RatingsData) -> "SVDppRecommender":
        self._validate_policy()
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
        self.implicit_factors = rng.normal(
            0.0, self.config.init_std, size=(data.n_items, self.config.latent_dim)
        ).astype(parameter_dtype, copy=False)
        self.user_histories = build_user_history_index(data, dtype=self.config.dtype)

        order = np.arange(len(data), dtype=np.int64)
        user_ids = data.user_ids
        item_ids = data.item_ids
        ratings = data.ratings.astype(parameter_dtype, copy=False)
        zero_vector = np.zeros(self.config.latent_dim, dtype=parameter_dtype)
        self.epoch_durations_seconds = []

        for _ in range(self.config.epochs):
            epoch_started = perf_counter()
            rng.shuffle(order)
            for idx in order:
                user_id = int(user_ids[idx])
                item_id = int(item_ids[idx])
                rating = float(ratings[idx])

                history = self.user_histories.items_for_user(user_id)
                norm = float(self.user_histories.norms[user_id])
                implicit_sum = zero_vector
                if history.size > 0:
                    implicit_sum = norm * np.sum(self.implicit_factors[history], axis=0)

                user_vector_old = self.user_factors[user_id].copy()
                item_vector_old = self.item_factors[item_id].copy()
                z_user = user_vector_old + implicit_sum
                prediction = float(
                    self.global_mean
                    + self.user_bias[user_id]
                    + self.item_bias[item_id]
                    + np.dot(item_vector_old, z_user)
                )
                error = rating - prediction

                user_bias_old = self.user_bias[user_id]
                item_bias_old = self.item_bias[item_id]
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
                    error * z_user - self.config.lambda_q * item_vector_old
                )

                if history.size > 0:
                    implicit_update = self.config.learning_rate * error * norm * item_vector_old
                    implicit_old = self.implicit_factors[history].copy()
                    self.implicit_factors[history] = implicit_old + implicit_update - (
                        self.config.learning_rate * self.config.lambda_y * implicit_old
                    )
            self.epoch_durations_seconds.append(perf_counter() - epoch_started)

        self.is_fitted = True
        return self

    def _user_context(self, user_id: int) -> np.ndarray:
        if self.user_factors is None or self.implicit_factors is None or self.user_histories is None:
            raise RuntimeError("model parameters are not initialized")

        history = self.user_histories.items_for_user(user_id)
        context = self.user_factors[user_id].astype(np.float64, copy=True)
        if history.size == 0:
            return context

        norm = float(self.user_histories.norms[user_id])
        implicit_sum = norm * np.sum(self.implicit_factors[history], axis=0)
        return context + implicit_sum.astype(np.float64, copy=False)

    def predict(self, user_id: int, item_id: int, *, clip: bool = True) -> float:
        if not self.is_fitted:
            raise RuntimeError("svdpp is not fitted")
        if self.user_bias is None or self.item_bias is None or self.item_factors is None:
            raise RuntimeError("model parameters are not initialized")

        context = self._user_context(user_id)
        prediction = float(
            self.global_mean
            + self.user_bias[user_id]
            + self.item_bias[item_id]
            + np.dot(context, self.item_factors[item_id].astype(np.float64, copy=False))
        )
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
            raise RuntimeError("svdpp is not fitted")
        if (
            self.user_bias is None
            or self.item_bias is None
            or self.user_factors is None
            or self.item_factors is None
            or self.implicit_factors is None
            or self.user_histories is None
        ):
            raise RuntimeError("model parameters are not initialized")

        users = np.asarray(user_ids, dtype=np.int64)
        items = np.asarray(item_ids, dtype=np.int64)
        contexts = np.zeros((self.user_factors.shape[0], self.user_factors.shape[1]), dtype=np.float64)
        contexts[:] = self.user_factors.astype(np.float64, copy=False)

        for user_id in np.unique(users):
            history = self.user_histories.items_for_user(int(user_id))
            if history.size == 0:
                continue
            norm = float(self.user_histories.norms[int(user_id)])
            contexts[int(user_id)] += (
                norm * np.sum(self.implicit_factors[history], axis=0).astype(np.float64, copy=False)
            )

        predictions = (
            self.global_mean
            + self.user_bias[users].astype(np.float64, copy=False)
            + self.item_bias[items].astype(np.float64, copy=False)
            + np.sum(contexts[users] * self.item_factors[items].astype(np.float64, copy=False), axis=1)
        )
        if clip:
            predictions = np.clip(predictions, self.rating_min, self.rating_max)
        return np.asarray(predictions, dtype=np.float64)

    def predict_dataset(self, data: RatingsData, *, clip: bool = True) -> np.ndarray:
        return self.predict_many(data.user_ids, data.item_ids, clip=clip)
