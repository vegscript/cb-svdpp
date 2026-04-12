from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - fallback only used when numba is unavailable
    njit = None


if njit is not None:

    @njit(cache=True)
    def train_asymmetric_svd_epoch_numba(
        order: np.ndarray,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        explicit_indptr: np.ndarray,
        explicit_items: np.ndarray,
        explicit_ratings: np.ndarray,
        explicit_norms: np.ndarray,
        implicit_indptr: np.ndarray,
        implicit_items: np.ndarray,
        implicit_norms: np.ndarray,
        global_mean: float,
        learning_rate: float,
        lambda_b: float,
        lambda_q: float,
        lambda_x: float,
        lambda_y: float,
        user_bias: np.ndarray,
        item_bias: np.ndarray,
        item_factors: np.ndarray,
        explicit_factors: np.ndarray,
        implicit_factors: np.ndarray,
    ) -> None:
        latent_dim = item_factors.shape[1]

        for position in range(order.shape[0]):
            idx = order[position]
            user_id = int(user_ids[idx])
            item_id = int(item_ids[idx])
            rating = ratings[idx]

            user_bias_old = user_bias[user_id]
            item_bias_old = item_bias[item_id]

            q_old = np.empty(latent_dim, dtype=item_factors.dtype)
            context = np.zeros(latent_dim, dtype=item_factors.dtype)
            for factor_idx in range(latent_dim):
                q_old[factor_idx] = item_factors[item_id, factor_idx]

            explicit_start = explicit_indptr[user_id]
            explicit_end = explicit_indptr[user_id + 1]
            explicit_norm = explicit_norms[user_id]
            for history_pos in range(explicit_start, explicit_end):
                history_item = explicit_items[history_pos]
                history_item_bias = item_bias_old if history_item == item_id else item_bias[history_item]
                residual_weight = explicit_ratings[history_pos] - (
                    global_mean + user_bias_old + history_item_bias
                )
                for factor_idx in range(latent_dim):
                    context[factor_idx] += (
                        explicit_norm
                        * residual_weight
                        * explicit_factors[history_item, factor_idx]
                    )

            implicit_start = implicit_indptr[user_id]
            implicit_end = implicit_indptr[user_id + 1]
            implicit_norm = implicit_norms[user_id]
            for history_pos in range(implicit_start, implicit_end):
                history_item = implicit_items[history_pos]
                for factor_idx in range(latent_dim):
                    context[factor_idx] += implicit_norm * implicit_factors[history_item, factor_idx]

            prediction = global_mean + user_bias_old + item_bias_old
            for factor_idx in range(latent_dim):
                prediction += q_old[factor_idx] * context[factor_idx]
            error = rating - prediction

            user_bias[user_id] = user_bias_old + learning_rate * (
                error - lambda_b * user_bias_old
            )
            item_bias[item_id] = item_bias_old + learning_rate * (
                error - lambda_b * item_bias_old
            )
            for factor_idx in range(latent_dim):
                item_factors[item_id, factor_idx] = q_old[factor_idx] + learning_rate * (
                    error * context[factor_idx] - lambda_q * q_old[factor_idx]
                )

            for history_pos in range(explicit_start, explicit_end):
                history_item = explicit_items[history_pos]
                history_item_bias = item_bias_old if history_item == item_id else item_bias[history_item]
                residual_weight = explicit_ratings[history_pos] - (
                    global_mean + user_bias_old + history_item_bias
                )
                for factor_idx in range(latent_dim):
                    x_old = explicit_factors[history_item, factor_idx]
                    explicit_factors[history_item, factor_idx] = x_old + learning_rate * (
                        error * explicit_norm * residual_weight * q_old[factor_idx]
                        - lambda_x * x_old
                    )

            for history_pos in range(implicit_start, implicit_end):
                history_item = implicit_items[history_pos]
                for factor_idx in range(latent_dim):
                    y_old = implicit_factors[history_item, factor_idx]
                    implicit_factors[history_item, factor_idx] = y_old + learning_rate * (
                        error * implicit_norm * q_old[factor_idx] - lambda_y * y_old
                    )

    @njit(cache=True)
    def train_asvdpp_epoch_numba(
        order: np.ndarray,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        explicit_indptr: np.ndarray,
        explicit_items: np.ndarray,
        explicit_ratings: np.ndarray,
        explicit_norms: np.ndarray,
        implicit_indptr: np.ndarray,
        implicit_items: np.ndarray,
        implicit_norms: np.ndarray,
        global_mean: float,
        learning_rate: float,
        lambda_b: float,
        lambda_p: float,
        lambda_q: float,
        lambda_x: float,
        lambda_y: float,
        user_bias: np.ndarray,
        item_bias: np.ndarray,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        explicit_factors: np.ndarray,
        implicit_factors: np.ndarray,
    ) -> None:
        latent_dim = item_factors.shape[1]

        for position in range(order.shape[0]):
            idx = order[position]
            user_id = int(user_ids[idx])
            item_id = int(item_ids[idx])
            rating = ratings[idx]

            user_bias_old = user_bias[user_id]
            item_bias_old = item_bias[item_id]

            p_old = np.empty(latent_dim, dtype=user_factors.dtype)
            q_old = np.empty(latent_dim, dtype=item_factors.dtype)
            context = np.zeros(latent_dim, dtype=item_factors.dtype)
            for factor_idx in range(latent_dim):
                p_old[factor_idx] = user_factors[user_id, factor_idx]
                q_old[factor_idx] = item_factors[item_id, factor_idx]
                context[factor_idx] = p_old[factor_idx]

            explicit_start = explicit_indptr[user_id]
            explicit_end = explicit_indptr[user_id + 1]
            explicit_norm = explicit_norms[user_id]
            for history_pos in range(explicit_start, explicit_end):
                history_item = explicit_items[history_pos]
                history_item_bias = item_bias_old if history_item == item_id else item_bias[history_item]
                residual_weight = explicit_ratings[history_pos] - (
                    global_mean + user_bias_old + history_item_bias
                )
                for factor_idx in range(latent_dim):
                    context[factor_idx] += (
                        explicit_norm
                        * residual_weight
                        * explicit_factors[history_item, factor_idx]
                    )

            implicit_start = implicit_indptr[user_id]
            implicit_end = implicit_indptr[user_id + 1]
            implicit_norm = implicit_norms[user_id]
            for history_pos in range(implicit_start, implicit_end):
                history_item = implicit_items[history_pos]
                for factor_idx in range(latent_dim):
                    context[factor_idx] += implicit_norm * implicit_factors[history_item, factor_idx]

            prediction = global_mean + user_bias_old + item_bias_old
            for factor_idx in range(latent_dim):
                prediction += q_old[factor_idx] * context[factor_idx]
            error = rating - prediction

            user_bias[user_id] = user_bias_old + learning_rate * (
                error - lambda_b * user_bias_old
            )
            item_bias[item_id] = item_bias_old + learning_rate * (
                error - lambda_b * item_bias_old
            )
            for factor_idx in range(latent_dim):
                user_factors[user_id, factor_idx] = p_old[factor_idx] + learning_rate * (
                    error * q_old[factor_idx] - lambda_p * p_old[factor_idx]
                )
                item_factors[item_id, factor_idx] = q_old[factor_idx] + learning_rate * (
                    error * context[factor_idx] - lambda_q * q_old[factor_idx]
                )

            for history_pos in range(explicit_start, explicit_end):
                history_item = explicit_items[history_pos]
                history_item_bias = item_bias_old if history_item == item_id else item_bias[history_item]
                residual_weight = explicit_ratings[history_pos] - (
                    global_mean + user_bias_old + history_item_bias
                )
                for factor_idx in range(latent_dim):
                    x_old = explicit_factors[history_item, factor_idx]
                    explicit_factors[history_item, factor_idx] = x_old + learning_rate * (
                        error * explicit_norm * residual_weight * q_old[factor_idx]
                        - lambda_x * x_old
                    )

            for history_pos in range(implicit_start, implicit_end):
                history_item = implicit_items[history_pos]
                for factor_idx in range(latent_dim):
                    y_old = implicit_factors[history_item, factor_idx]
                    implicit_factors[history_item, factor_idx] = y_old + learning_rate * (
                        error * implicit_norm * q_old[factor_idx] - lambda_y * y_old
                    )

else:

    def train_asymmetric_svd_epoch_numba(*args: object, **kwargs: object) -> None:
        raise RuntimeError("numba is not available")

    def train_asvdpp_epoch_numba(*args: object, **kwargs: object) -> None:
        raise RuntimeError("numba is not available")
