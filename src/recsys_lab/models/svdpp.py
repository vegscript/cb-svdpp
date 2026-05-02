from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from recsys_lab.data.histories import (
    UserHistoryIndex,
    build_user_history_index,
    validate_user_history_index,
)
from recsys_lab.data.processed import RatingsData
from recsys_lab.models.inference import build_user_context_cache
from recsys_lab.models.kernels import train_svdpp_epoch_numba


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
    training_backend: str = "auto"


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
        self.training_backend_effective: str | None = None
        self._user_context_cache: np.ndarray | None = None

    def _parameter_dtype(self) -> np.dtype:
        if self.config.dtype not in {"float32", "float64"}:
            raise ValueError("svdpp dtype must be 'float32' or 'float64'")
        return np.dtype(self.config.dtype)

    def _validate_policy(self) -> None:
        if self.config.implicit_policy != "ratings_as_implicit":
            raise ValueError("only implicit_policy='ratings_as_implicit' is currently supported")

    def _validate_training_backend(self) -> str:
        backend = str(self.config.training_backend)
        if backend not in {"auto", "python", "numba"}:
            raise ValueError("svdpp training_backend must be 'auto', 'python', or 'numba'")
        return backend

    def _train_epoch_python(
        self,
        order: np.ndarray,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        *,
        zero_vector: np.ndarray,
    ) -> None:
        if (
            self.user_bias is None
            or self.item_bias is None
            or self.user_factors is None
            or self.item_factors is None
            or self.implicit_factors is None
            or self.user_histories is None
        ):
            raise RuntimeError("model parameters are not initialized")

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
                self.global_mean + self.user_bias[user_id] + self.item_bias[item_id] + np.dot(item_vector_old, z_user)
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
                self.implicit_factors[history] = (
                    implicit_old + implicit_update - (self.config.learning_rate * self.config.lambda_y * implicit_old)
                )

    def fit(
        self,
        data: RatingsData,
        *,
        user_histories: UserHistoryIndex | None = None,
    ) -> "SVDppRecommender":
        self._validate_policy()
        requested_backend = self._validate_training_backend()
        rng = np.random.default_rng(self.config.seed)
        parameter_dtype = self._parameter_dtype()

        self.global_mean = data.effective_ratings_mean()
        self.rating_min = data.rating_min
        self.rating_max = data.rating_max
        self.user_bias = np.zeros(data.n_users, dtype=parameter_dtype)
        self.item_bias = np.zeros(data.n_items, dtype=parameter_dtype)
        self.user_factors = rng.normal(0.0, self.config.init_std, size=(data.n_users, self.config.latent_dim)).astype(
            parameter_dtype, copy=False
        )
        self.item_factors = rng.normal(0.0, self.config.init_std, size=(data.n_items, self.config.latent_dim)).astype(
            parameter_dtype, copy=False
        )
        self.implicit_factors = rng.normal(
            0.0, self.config.init_std, size=(data.n_items, self.config.latent_dim)
        ).astype(parameter_dtype, copy=False)
        if user_histories is None:
            self.user_histories = build_user_history_index(data, dtype=self.config.dtype)
        else:
            validate_user_history_index(user_histories, n_users=data.n_users)
            self.user_histories = user_histories

        order = data.training_row_indices()
        user_ids = data.base_user_ids
        item_ids = data.base_item_ids
        ratings = data.base_ratings.astype(parameter_dtype, copy=False)
        zero_vector = np.zeros(self.config.latent_dim, dtype=parameter_dtype)
        self.epoch_durations_seconds = []
        self.training_backend_effective = None
        self._user_context_cache = None

        for _ in range(self.config.epochs):
            epoch_started = perf_counter()
            rng.shuffle(order)
            if requested_backend == "python":
                self._train_epoch_python(
                    order,
                    user_ids,
                    item_ids,
                    ratings,
                    zero_vector=zero_vector,
                )
                self.training_backend_effective = "python"
            else:
                try:
                    train_svdpp_epoch_numba(
                        order,
                        user_ids,
                        item_ids,
                        ratings,
                        self.user_histories.indptr,
                        self.user_histories.item_indices,
                        self.user_histories.norms,
                        self.global_mean,
                        self.config.learning_rate,
                        self.config.lambda_b,
                        self.config.lambda_p,
                        self.config.lambda_q,
                        self.config.lambda_y,
                        self.user_bias,
                        self.item_bias,
                        self.user_factors,
                        self.item_factors,
                        self.implicit_factors,
                    )
                    self.training_backend_effective = "numba"
                except RuntimeError:
                    if requested_backend == "numba":
                        raise
                    self._train_epoch_python(
                        order,
                        user_ids,
                        item_ids,
                        ratings,
                        zero_vector=zero_vector,
                    )
                    self.training_backend_effective = "python"
            self.epoch_durations_seconds.append(perf_counter() - epoch_started)

        self.is_fitted = True
        return self

    def _compute_user_context(self, user_id: int) -> np.ndarray:
        if self.user_factors is None or self.implicit_factors is None or self.user_histories is None:
            raise RuntimeError("model parameters are not initialized")

        history = self.user_histories.items_for_user(user_id)
        context = self.user_factors[user_id].astype(np.float64, copy=True)
        if history.size == 0:
            return context

        norm = float(self.user_histories.norms[user_id])
        implicit_sum = norm * np.sum(self.implicit_factors[history], axis=0)
        return context + implicit_sum.astype(np.float64, copy=False)

    def _ensure_user_context_cache(self) -> np.ndarray:
        if self.user_factors is None:
            raise RuntimeError("model parameters are not initialized")
        if self._user_context_cache is None:
            self._user_context_cache = build_user_context_cache(
                n_users=self.user_factors.shape[0],
                context_dim=self.user_factors.shape[1],
                build_user_context=self._compute_user_context,
            )
        return self._user_context_cache

    def _user_context(self, user_id: int) -> np.ndarray:
        return self._ensure_user_context_cache()[user_id]

    def predict(self, user_id: int, item_id: int, *, clip: bool = True) -> float:
        if not self.is_fitted:
            raise RuntimeError("svdpp is not fitted")
        if self.user_bias is None or self.item_bias is None or self.item_factors is None:
            raise RuntimeError("model parameters are not initialized")

        context = self._ensure_user_context_cache()[user_id]
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
        contexts = self._ensure_user_context_cache()[users]
        items = np.asarray(item_ids, dtype=np.int64)
        predictions = (
            self.global_mean
            + self.user_bias[users].astype(np.float64, copy=False)
            + self.item_bias[items].astype(np.float64, copy=False)
            + np.sum(contexts * self.item_factors[items].astype(np.float64, copy=False), axis=1)
        )
        if clip:
            predictions = np.clip(predictions, self.rating_min, self.rating_max)
        return np.asarray(predictions, dtype=np.float64)

    def predict_dataset(self, data: RatingsData, *, clip: bool = True) -> np.ndarray:
        return self.predict_many(data.user_ids, data.item_ids, clip=clip)
