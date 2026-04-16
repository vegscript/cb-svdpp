from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from recsys_lab.data.histories import (
    UserClusterCountIndex,
    UserExplicitFeedbackIndex,
    UserHistoryIndex,
    build_user_cluster_count_index,
    build_user_explicit_feedback_index,
    build_user_history_index,
)
from recsys_lab.data.processed import RatingsData
from recsys_lab.models.kernels import train_cb_asvdpp_epoch_numba


@dataclass(frozen=True, slots=True)
class CBASVDppConfig:
    latent_dim: int = 50
    epochs: int = 20
    learning_rate: float = 0.01
    lambda_b: float = 0.02
    lambda_p: float = 0.02
    lambda_q: float = 0.02
    lambda_x: float = 0.02
    lambda_y: float = 0.02
    lambda_pC: float = 0.02
    lambda_qC: float = 0.02
    lambda_xC: float = 0.02
    lambda_yC: float = 0.02
    alpha: float = 0.10
    seed: int = 42
    init_std: float = 0.1
    dtype: str = "float32"
    implicit_policy: str = "ratings_as_implicit"
    residual_weight_contract: str = "detached"


class CBASVDppRecommender:
    def __init__(
        self,
        config: CBASVDppConfig,
        *,
        user_clusters: np.ndarray,
        item_clusters: np.ndarray,
        n_user_clusters: int,
        n_item_clusters: int,
    ) -> None:
        self.config = config
        self.is_fitted = False
        self.user_clusters = np.asarray(user_clusters, dtype=np.int32)
        self.item_clusters = np.asarray(item_clusters, dtype=np.int32)
        self.n_user_clusters = int(n_user_clusters)
        self.n_item_clusters = int(n_item_clusters)
        self.global_mean = 0.0
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.explicit_factors: np.ndarray | None = None
        self.implicit_factors: np.ndarray | None = None
        self.user_cluster_factors: np.ndarray | None = None
        self.item_cluster_factors: np.ndarray | None = None
        self.explicit_cluster_factors: np.ndarray | None = None
        self.implicit_cluster_factors: np.ndarray | None = None
        self.explicit_feedback: UserExplicitFeedbackIndex | None = None
        self.implicit_history: UserHistoryIndex | None = None
        self.implicit_cluster_history: UserClusterCountIndex | None = None
        self.rating_min = 0.0
        self.rating_max = 0.0
        self.epoch_durations_seconds: list[float] = []

    def _parameter_dtype(self) -> np.dtype:
        if self.config.dtype not in {"float32", "float64"}:
            raise ValueError("cb_asvdpp dtype must be 'float32' or 'float64'")
        return np.dtype(self.config.dtype)

    def _validate_contracts(self, data: RatingsData) -> None:
        if self.config.implicit_policy != "ratings_as_implicit":
            raise ValueError("only implicit_policy='ratings_as_implicit' is currently supported")
        if self.config.residual_weight_contract != "detached":
            raise ValueError("only residual_weight_contract='detached' is currently supported")
        if not 0.0 <= self.config.alpha <= 1.0:
            raise ValueError("alpha must be in the closed interval [0, 1]")
        if self.user_clusters.shape[0] != data.n_users:
            raise ValueError("user cluster assignments do not match dataset n_users")
        if self.item_clusters.shape[0] != data.n_items:
            raise ValueError("item cluster assignments do not match dataset n_items")
        if self.n_user_clusters <= 0 or self.n_item_clusters <= 0:
            raise ValueError("cluster counts must be positive")

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
            or self.user_cluster_factors is None
            or self.item_cluster_factors is None
            or self.explicit_cluster_factors is None
            or self.implicit_cluster_factors is None
            or self.explicit_feedback is None
            or self.implicit_history is None
            or self.implicit_cluster_history is None
        ):
            raise RuntimeError("model parameters are not initialized")

        alpha = float(self.config.alpha)
        one_minus_alpha = 1.0 - alpha

        for idx in order:
            user_id = int(user_ids[idx])
            item_id = int(item_ids[idx])
            rating = float(ratings[idx])

            user_cluster_id = int(self.user_clusters[user_id])
            item_cluster_id = int(self.item_clusters[item_id])
            explicit_items = self.explicit_feedback.items_for_user(user_id)
            explicit_ratings = self.explicit_feedback.ratings_for_user(user_id)
            implicit_items = self.implicit_history.items_for_user(user_id)
            implicit_clusters = self.implicit_cluster_history.clusters_for_user(user_id)
            implicit_cluster_counts = self.implicit_cluster_history.counts_for_user(user_id)
            explicit_norm = float(self.explicit_feedback.norms[user_id])
            implicit_norm = float(self.implicit_history.norms[user_id])

            user_bias_old = self.user_bias[user_id]
            item_bias_old = self.item_bias[item_id]
            user_vector_old = self.user_factors[user_id].copy()
            user_cluster_vector_old = self.user_cluster_factors[user_cluster_id].copy()
            item_vector_old = self.item_factors[item_id].copy()
            item_cluster_vector_old = self.item_cluster_factors[item_cluster_id].copy()

            q_mix_old = one_minus_alpha * item_vector_old + alpha * item_cluster_vector_old
            context = one_minus_alpha * user_vector_old + alpha * user_cluster_vector_old

            residual_weights = np.empty(0, dtype=parameter_dtype)
            explicit_clusters = np.empty(0, dtype=np.int32)
            if explicit_items.size > 0:
                explicit_item_biases = self.item_bias[explicit_items].copy()
                explicit_item_biases[explicit_items == item_id] = item_bias_old
                residual_weights = explicit_ratings - (
                    self.global_mean + user_bias_old + explicit_item_biases
                )
                explicit_clusters = self.item_clusters[explicit_items]
                context += explicit_norm * one_minus_alpha * np.sum(
                    residual_weights[:, None] * self.explicit_factors[explicit_items],
                    axis=0,
                )
                context += explicit_norm * alpha * np.sum(
                    residual_weights[:, None]
                    * self.explicit_cluster_factors[explicit_clusters],
                    axis=0,
                )

            if implicit_items.size > 0:
                context += implicit_norm * one_minus_alpha * np.sum(
                    self.implicit_factors[implicit_items],
                    axis=0,
                )
            if implicit_clusters.size > 0:
                context += implicit_norm * alpha * np.sum(
                    implicit_cluster_counts[:, None].astype(np.float64, copy=False)
                    * self.implicit_cluster_factors[implicit_clusters].astype(np.float64, copy=False),
                    axis=0,
                ).astype(parameter_dtype, copy=False)

            prediction = float(
                self.global_mean
                + user_bias_old
                + item_bias_old
                + np.dot(q_mix_old, context)
            )
            error = rating - prediction

            self.user_bias[user_id] = user_bias_old + self.config.learning_rate * (
                error - self.config.lambda_b * user_bias_old
            )
            self.item_bias[item_id] = item_bias_old + self.config.learning_rate * (
                error - self.config.lambda_b * item_bias_old
            )
            self.user_factors[user_id] = user_vector_old + self.config.learning_rate * (
                error * one_minus_alpha * q_mix_old - self.config.lambda_p * user_vector_old
            )
            self.user_cluster_factors[user_cluster_id] = user_cluster_vector_old + (
                self.config.learning_rate
                * (
                    error * alpha * q_mix_old
                    - self.config.lambda_pC * user_cluster_vector_old
                )
            )
            self.item_factors[item_id] = item_vector_old + self.config.learning_rate * (
                error * one_minus_alpha * context - self.config.lambda_q * item_vector_old
            )
            self.item_cluster_factors[item_cluster_id] = item_cluster_vector_old + (
                self.config.learning_rate
                * (
                    error * alpha * context
                    - self.config.lambda_qC * item_cluster_vector_old
                )
            )

            if explicit_items.size > 0:
                explicit_old = self.explicit_factors[explicit_items].copy()
                self.explicit_factors[explicit_items] = explicit_old + (
                    self.config.learning_rate
                    * error
                    * explicit_norm
                    * one_minus_alpha
                    * residual_weights[:, None]
                    * q_mix_old
                ) - (self.config.learning_rate * self.config.lambda_x * explicit_old)

                for history_pos, history_cluster in enumerate(explicit_clusters):
                    cluster_id = int(history_cluster)
                    cluster_old = self.explicit_cluster_factors[cluster_id].copy()
                    cluster_update = (
                        self.config.learning_rate
                        * error
                        * explicit_norm
                        * alpha
                        * float(residual_weights[history_pos])
                        * q_mix_old
                    )
                    self.explicit_cluster_factors[cluster_id] = cluster_old + cluster_update - (
                        self.config.learning_rate * self.config.lambda_xC * cluster_old
                    )

            if implicit_items.size > 0:
                implicit_old = self.implicit_factors[implicit_items].copy()
                implicit_update = (
                    self.config.learning_rate
                    * error
                    * implicit_norm
                    * one_minus_alpha
                    * q_mix_old
                )
                self.implicit_factors[implicit_items] = implicit_old + implicit_update - (
                    self.config.learning_rate * self.config.lambda_y * implicit_old
                )

            if implicit_clusters.size > 0:
                cluster_old = self.implicit_cluster_factors[implicit_clusters].copy()
                cluster_update = (
                    self.config.learning_rate
                    * error
                    * implicit_norm
                    * alpha
                    * implicit_cluster_counts[:, None].astype(np.float64, copy=False)
                    * q_mix_old.astype(np.float64, copy=False)
                ).astype(parameter_dtype, copy=False)
                self.implicit_cluster_factors[implicit_clusters] = cluster_old + cluster_update - (
                    self.config.learning_rate * self.config.lambda_yC * cluster_old
                )

    def fit(self, data: RatingsData) -> "CBASVDppRecommender":
        self._validate_contracts(data)
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
        self.user_cluster_factors = rng.normal(
            0.0, self.config.init_std, size=(self.n_user_clusters, self.config.latent_dim)
        ).astype(parameter_dtype, copy=False)
        self.item_cluster_factors = rng.normal(
            0.0, self.config.init_std, size=(self.n_item_clusters, self.config.latent_dim)
        ).astype(parameter_dtype, copy=False)
        self.explicit_cluster_factors = rng.normal(
            0.0, self.config.init_std, size=(self.n_item_clusters, self.config.latent_dim)
        ).astype(parameter_dtype, copy=False)
        self.implicit_cluster_factors = rng.normal(
            0.0, self.config.init_std, size=(self.n_item_clusters, self.config.latent_dim)
        ).astype(parameter_dtype, copy=False)
        self.explicit_feedback = build_user_explicit_feedback_index(data, dtype=self.config.dtype)
        self.implicit_history = build_user_history_index(data, dtype=self.config.dtype)
        self.implicit_cluster_history = build_user_cluster_count_index(
            self.implicit_history,
            self.item_clusters,
            n_clusters=self.n_item_clusters,
        )

        order = np.arange(len(data), dtype=np.int64)
        user_ids = data.user_ids
        item_ids = data.item_ids
        ratings = data.ratings.astype(parameter_dtype, copy=False)
        self.epoch_durations_seconds = []

        for _ in range(self.config.epochs):
            epoch_started = perf_counter()
            rng.shuffle(order)
            try:
                train_cb_asvdpp_epoch_numba(
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
                    self.implicit_cluster_history.indptr,
                    self.implicit_cluster_history.cluster_ids,
                    self.implicit_cluster_history.counts,
                    self.user_clusters,
                    self.item_clusters,
                    float(self.config.alpha),
                    self.global_mean,
                    self.config.learning_rate,
                    self.config.lambda_b,
                    self.config.lambda_p,
                    self.config.lambda_q,
                    self.config.lambda_x,
                    self.config.lambda_y,
                    self.config.lambda_pC,
                    self.config.lambda_qC,
                    self.config.lambda_xC,
                    self.config.lambda_yC,
                    self.user_bias,
                    self.item_bias,
                    self.user_factors,
                    self.item_factors,
                    self.explicit_factors,
                    self.implicit_factors,
                    self.user_cluster_factors,
                    self.item_cluster_factors,
                    self.explicit_cluster_factors,
                    self.implicit_cluster_factors,
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
            or self.user_cluster_factors is None
            or self.explicit_cluster_factors is None
            or self.implicit_cluster_factors is None
            or self.explicit_feedback is None
            or self.implicit_history is None
            or self.implicit_cluster_history is None
        ):
            raise RuntimeError("model parameters are not initialized")

        alpha = float(self.config.alpha)
        one_minus_alpha = 1.0 - alpha
        user_cluster_id = int(self.user_clusters[user_id])
        context = (
            one_minus_alpha * self.user_factors[user_id].astype(np.float64, copy=False)
            + alpha * self.user_cluster_factors[user_cluster_id].astype(np.float64, copy=False)
        ).copy()

        explicit_items = self.explicit_feedback.items_for_user(user_id)
        if explicit_items.size > 0:
            explicit_ratings = self.explicit_feedback.ratings_for_user(user_id).astype(np.float64, copy=False)
            explicit_norm = float(self.explicit_feedback.norms[user_id])
            residual_weights = explicit_ratings - (
                self.global_mean
                + float(self.user_bias[user_id])
                + self.item_bias[explicit_items].astype(np.float64, copy=False)
            )
            explicit_clusters = self.item_clusters[explicit_items]
            context += (
                explicit_norm
                * one_minus_alpha
                * np.sum(
                    residual_weights[:, None]
                    * self.explicit_factors[explicit_items].astype(np.float64, copy=False),
                    axis=0,
                )
            )
            context += (
                explicit_norm
                * alpha
                * np.sum(
                    residual_weights[:, None]
                    * self.explicit_cluster_factors[explicit_clusters].astype(np.float64, copy=False),
                    axis=0,
                )
            )

        implicit_items = self.implicit_history.items_for_user(user_id)
        implicit_norm = float(self.implicit_history.norms[user_id])
        if implicit_items.size > 0:
            context += (
                implicit_norm
                * one_minus_alpha
                * np.sum(self.implicit_factors[implicit_items].astype(np.float64, copy=False), axis=0)
            )

        implicit_clusters = self.implicit_cluster_history.clusters_for_user(user_id)
        if implicit_clusters.size > 0:
            implicit_cluster_counts = self.implicit_cluster_history.counts_for_user(user_id).astype(
                np.float64,
                copy=False,
            )
            context += (
                implicit_norm
                * alpha
                * np.sum(
                    implicit_cluster_counts[:, None]
                    * self.implicit_cluster_factors[implicit_clusters].astype(np.float64, copy=False),
                    axis=0,
                )
            )

        return context

    def predict(self, user_id: int, item_id: int, *, clip: bool = True) -> float:
        if not self.is_fitted:
            raise RuntimeError("cb_asvdpp is not fitted")
        if (
            self.user_bias is None
            or self.item_bias is None
            or self.item_factors is None
            or self.item_cluster_factors is None
        ):
            raise RuntimeError("model parameters are not initialized")

        alpha = float(self.config.alpha)
        one_minus_alpha = 1.0 - alpha
        item_cluster_id = int(self.item_clusters[item_id])
        item_vector = (
            one_minus_alpha * self.item_factors[item_id].astype(np.float64, copy=False)
            + alpha * self.item_cluster_factors[item_cluster_id].astype(np.float64, copy=False)
        )
        context = self._user_context(user_id)
        prediction = float(
            self.global_mean
            + self.user_bias[user_id]
            + self.item_bias[item_id]
            + np.dot(context, item_vector)
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
            raise RuntimeError("cb_asvdpp is not fitted")
        if (
            self.user_bias is None
            or self.item_bias is None
            or self.item_factors is None
            or self.item_cluster_factors is None
            or self.user_factors is None
        ):
            raise RuntimeError("model parameters are not initialized")

        users = np.asarray(user_ids, dtype=np.int64)
        items = np.asarray(item_ids, dtype=np.int64)
        contexts = np.zeros((self.user_factors.shape[0], self.user_factors.shape[1]), dtype=np.float64)
        for user_id in np.unique(users):
            contexts[int(user_id)] = self._user_context(int(user_id))

        alpha = float(self.config.alpha)
        one_minus_alpha = 1.0 - alpha
        mixed_item_factors = (
            one_minus_alpha * self.item_factors.astype(np.float64, copy=False)
            + alpha * self.item_cluster_factors[self.item_clusters].astype(np.float64, copy=False)
        )

        predictions = (
            self.global_mean
            + self.user_bias[users].astype(np.float64, copy=False)
            + self.item_bias[items].astype(np.float64, copy=False)
            + np.sum(contexts[users] * mixed_item_factors[items], axis=1)
        )
        if clip:
            predictions = np.clip(predictions, self.rating_min, self.rating_max)
        return np.asarray(predictions, dtype=np.float64)

    def predict_dataset(self, data: RatingsData, *, clip: bool = True) -> np.ndarray:
        return self.predict_many(data.user_ids, data.item_ids, clip=clip)
