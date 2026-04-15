from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from recsys_lab.data.histories import (
    UserExplicitFeedbackIndex,
    UserHistoryIndex,
    build_user_explicit_feedback_index,
    build_user_history_index,
)
from recsys_lab.data.processed import RatingsData
from recsys_lab.models.kernels import train_asvdpp_epoch_numba


@dataclass(frozen=True, slots=True)
class ASVDppConfig:
    latent_dim: int = 50
    epochs: int = 20
    learning_rate: float = 0.01
    lambda_b: float = 0.02
    lambda_p: float = 0.02
    lambda_q: float = 0.02
    lambda_x: float = 0.02
    lambda_y: float = 0.02
    seed: int = 42
    init_std: float = 0.1
    dtype: str = "float32"
    implicit_policy: str = "ratings_as_implicit"
    residual_weight_contract: str = "detached"


class ASVDppRecommender:
    def __init__(self, config: ASVDppConfig) -> None:
        self.config = config
        self.is_fitted = False
        self.global_mean = 0.0
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.explicit_factors: np.ndarray | None = None
        self.implicit_factors: np.ndarray | None = None
        self.explicit_feedback: UserExplicitFeedbackIndex | None = None
        self.implicit_history: UserHistoryIndex | None = None
        self.rating_min = 0.0
        self.rating_max = 0.0
        self.epoch_durations_seconds: list[float] = []

    def _parameter_dtype(self) -> np.dtype:
        if self.config.dtype not in {"float32", "float64"}:
            raise ValueError("asvdpp dtype must be 'float32' or 'float64'")
        return np.dtype(self.config.dtype)

    def _validate_contracts(self) -> None:
        if self.config.implicit_policy != "ratings_as_implicit":
            raise ValueError("only implicit_policy='ratings_as_implicit' is currently supported")
        if self.config.residual_weight_contract != "detached":
            raise ValueError("only residual_weight_contract='detached' is currently supported")

    def _train_epoch_python(
        self,
        order: np.ndarray,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        *,
        parameter_dtype: np.dtype,
    ) -> None:
        if (
            self.user_bias is None
            or self.item_bias is None
            or self.user_factors is None
            or self.item_factors is None
            or self.explicit_factors is None
            or self.implicit_factors is None
            or self.explicit_feedback is None
            or self.implicit_history is None
        ):
            raise RuntimeError("model parameters are not initialized")

        zero_vector = np.zeros(self.config.latent_dim, dtype=parameter_dtype)
        for idx in order:
            user_id = int(user_ids[idx])
            item_id = int(item_ids[idx])
            rating = float(ratings[idx])

            explicit_items = self.explicit_feedback.items_for_user(user_id)
            explicit_ratings = self.explicit_feedback.ratings_for_user(user_id)
            implicit_items = self.implicit_history.items_for_user(user_id)

            explicit_norm = float(self.explicit_feedback.norms[user_id])
            implicit_norm = float(self.implicit_history.norms[user_id])

            user_bias_old = self.user_bias[user_id]
            item_bias_old = self.item_bias[item_id]
            user_vector_old = self.user_factors[user_id].copy()
            item_vector_old = self.item_factors[item_id].copy()

            explicit_sum = zero_vector
            residual_weights = np.empty(0, dtype=parameter_dtype)
            if explicit_items.size > 0:
                residual_weights = explicit_ratings - (
                    self.global_mean
                    + user_bias_old
                    + self.item_bias[explicit_items]
                )
                explicit_sum = explicit_norm * np.sum(
                    residual_weights[:, None] * self.explicit_factors[explicit_items],
                    axis=0,
                )

            implicit_sum = zero_vector
            if implicit_items.size > 0:
                implicit_sum = implicit_norm * np.sum(self.implicit_factors[implicit_items], axis=0)

            user_context = user_vector_old + explicit_sum + implicit_sum
            prediction = float(
                self.global_mean
                + user_bias_old
                + item_bias_old
                + np.dot(item_vector_old, user_context)
            )
            error = rating - prediction

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
                error * user_context - self.config.lambda_q * item_vector_old
            )

            if explicit_items.size > 0:
                explicit_old = self.explicit_factors[explicit_items].copy()
                explicit_update = (
                    self.config.learning_rate
                    * error
                    * explicit_norm
                    * residual_weights[:, None]
                    * item_vector_old
                )
                self.explicit_factors[explicit_items] = explicit_old + explicit_update - (
                    self.config.learning_rate * self.config.lambda_x * explicit_old
                )

            if implicit_items.size > 0:
                implicit_old = self.implicit_factors[implicit_items].copy()
                implicit_update = (
                    self.config.learning_rate
                    * error
                    * implicit_norm
                    * item_vector_old
                )
                self.implicit_factors[implicit_items] = implicit_old + implicit_update - (
                    self.config.learning_rate * self.config.lambda_y * implicit_old
                )

    def fit(self, data: RatingsData) -> "ASVDppRecommender":
        self._validate_contracts()
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
        self.explicit_factors = rng.normal(
            0.0, self.config.init_std, size=(data.n_items, self.config.latent_dim)
        ).astype(parameter_dtype, copy=False)
        self.implicit_factors = rng.normal(
            0.0, self.config.init_std, size=(data.n_items, self.config.latent_dim)
        ).astype(parameter_dtype, copy=False)
        self.explicit_feedback = build_user_explicit_feedback_index(data, dtype=self.config.dtype)
        self.implicit_history = build_user_history_index(data, dtype=self.config.dtype)

        order = np.arange(len(data), dtype=np.int64)
        user_ids = data.user_ids
        item_ids = data.item_ids
        ratings = data.ratings.astype(parameter_dtype, copy=False)
        self.epoch_durations_seconds = []

        for _ in range(self.config.epochs):
            epoch_started = perf_counter()
            rng.shuffle(order)
            try:
                train_asvdpp_epoch_numba(
                    order,
                    user_ids,
                    item_ids,
                    ratings,
                    self.explicit_feedback.indptr,
                    self.explicit_feedback.item_indices,
                    self.explicit_feedback.ratings,
                    self.explicit_feedback.norms,
                    self.implicit_history.indptr,
                    self.implicit_history.item_indices,
                    self.implicit_history.norms,
                    self.global_mean,
                    self.config.learning_rate,
                    self.config.lambda_b,
                    self.config.lambda_p,
                    self.config.lambda_q,
                    self.config.lambda_x,
                    self.config.lambda_y,
                    self.user_bias,
                    self.item_bias,
                    self.user_factors,
                    self.item_factors,
                    self.explicit_factors,
                    self.implicit_factors,
                )
            except RuntimeError:
                self._train_epoch_python(
                    order,
                    user_ids,
                    item_ids,
                    ratings,
                    parameter_dtype=parameter_dtype,
                )
            self.epoch_durations_seconds.append(perf_counter() - epoch_started)

        self.is_fitted = True
        return self

    def _user_context(self, user_id: int) -> np.ndarray:
        if (
            self.user_bias is None
            or self.item_bias is None
            or self.user_factors is None
            or self.explicit_factors is None
            or self.implicit_factors is None
            or self.explicit_feedback is None
            or self.implicit_history is None
        ):
            raise RuntimeError("model parameters are not initialized")

        context = self.user_factors[user_id].astype(np.float64, copy=True)

        explicit_items = self.explicit_feedback.items_for_user(user_id)
        if explicit_items.size > 0:
            explicit_ratings = self.explicit_feedback.ratings_for_user(user_id).astype(np.float64, copy=False)
            explicit_norm = float(self.explicit_feedback.norms[user_id])
            residual_weights = explicit_ratings - (
                self.global_mean
                + float(self.user_bias[user_id])
                + self.item_bias[explicit_items].astype(np.float64, copy=False)
            )
            context += explicit_norm * np.sum(
                residual_weights[:, None] * self.explicit_factors[explicit_items].astype(np.float64, copy=False),
                axis=0,
            )

        implicit_items = self.implicit_history.items_for_user(user_id)
        if implicit_items.size > 0:
            implicit_norm = float(self.implicit_history.norms[user_id])
            context += implicit_norm * np.sum(
                self.implicit_factors[implicit_items].astype(np.float64, copy=False),
                axis=0,
            )

        return context

    def predict(self, user_id: int, item_id: int, *, clip: bool = True) -> float:
        if not self.is_fitted:
            raise RuntimeError("asvdpp is not fitted")
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
            raise RuntimeError("asvdpp is not fitted")
        if (
            self.item_factors is None
            or self.user_bias is None
            or self.item_bias is None
            or self.user_factors is None
        ):
            raise RuntimeError("model parameters are not initialized")

        users = np.asarray(user_ids, dtype=np.int64)
        items = np.asarray(item_ids, dtype=np.int64)
        contexts = np.zeros((self.user_factors.shape[0], self.item_factors.shape[1]), dtype=np.float64)
        for user_id in np.unique(users):
            contexts[int(user_id)] = self._user_context(int(user_id))

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
