from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - fallback only used when numba is unavailable
    njit = None


if njit is not None:

    @njit(cache=True)
    def train_biased_mf_epoch_numba(
        order: np.ndarray,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        global_mean: float,
        learning_rate: float,
        lambda_b: float,
        lambda_p: float,
        lambda_q: float,
        user_bias: np.ndarray,
        item_bias: np.ndarray,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
    ) -> None:
        latent_dim = user_factors.shape[1]

        for position in range(order.shape[0]):
            idx = order[position]
            user_id = int(user_ids[idx])
            item_id = int(item_ids[idx])
            rating = float(ratings[idx])

            user_bias_old = user_bias[user_id]
            item_bias_old = item_bias[item_id]
            user_vector_old = np.empty(latent_dim, dtype=user_factors.dtype)
            item_vector_old = np.empty(latent_dim, dtype=item_factors.dtype)

            prediction = global_mean + user_bias_old + item_bias_old
            for factor_idx in range(latent_dim):
                user_value = user_factors[user_id, factor_idx]
                item_value = item_factors[item_id, factor_idx]
                user_vector_old[factor_idx] = user_value
                item_vector_old[factor_idx] = item_value
                prediction += user_value * item_value
            error = rating - prediction

            user_bias[user_id] = user_bias_old + learning_rate * (error - lambda_b * user_bias_old)
            item_bias[item_id] = item_bias_old + learning_rate * (error - lambda_b * item_bias_old)
            for factor_idx in range(latent_dim):
                user_factors[user_id, factor_idx] = user_vector_old[factor_idx] + learning_rate * (
                    error * item_vector_old[factor_idx] - lambda_p * user_vector_old[factor_idx]
                )
                item_factors[item_id, factor_idx] = item_vector_old[factor_idx] + learning_rate * (
                    error * user_vector_old[factor_idx] - lambda_q * item_vector_old[factor_idx]
                )

    @njit(cache=True)
    def train_svdpp_epoch_numba(
        order: np.ndarray,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        history_indptr: np.ndarray,
        history_items: np.ndarray,
        history_norms: np.ndarray,
        global_mean: float,
        learning_rate: float,
        lambda_b: float,
        lambda_p: float,
        lambda_q: float,
        lambda_y: float,
        user_bias: np.ndarray,
        item_bias: np.ndarray,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        implicit_factors: np.ndarray,
    ) -> None:
        latent_dim = user_factors.shape[1]

        for position in range(order.shape[0]):
            idx = order[position]
            user_id = int(user_ids[idx])
            item_id = int(item_ids[idx])
            rating = float(ratings[idx])

            history_start = history_indptr[user_id]
            history_end = history_indptr[user_id + 1]
            norm = float(history_norms[user_id])

            user_bias_old = user_bias[user_id]
            item_bias_old = item_bias[item_id]
            user_vector_old = np.empty(latent_dim, dtype=user_factors.dtype)
            item_vector_old = np.empty(latent_dim, dtype=item_factors.dtype)
            z_user = np.empty(latent_dim, dtype=user_factors.dtype)

            for factor_idx in range(latent_dim):
                user_value = user_factors[user_id, factor_idx]
                item_value = item_factors[item_id, factor_idx]
                user_vector_old[factor_idx] = user_value
                item_vector_old[factor_idx] = item_value
                z_user[factor_idx] = user_value

            for history_pos in range(history_start, history_end):
                history_item = int(history_items[history_pos])
                for factor_idx in range(latent_dim):
                    z_user[factor_idx] += norm * implicit_factors[history_item, factor_idx]

            prediction = global_mean + user_bias_old + item_bias_old
            for factor_idx in range(latent_dim):
                prediction += item_vector_old[factor_idx] * z_user[factor_idx]
            error = rating - prediction

            user_bias[user_id] = user_bias_old + learning_rate * (error - lambda_b * user_bias_old)
            item_bias[item_id] = item_bias_old + learning_rate * (error - lambda_b * item_bias_old)
            for factor_idx in range(latent_dim):
                user_factors[user_id, factor_idx] = user_vector_old[factor_idx] + learning_rate * (
                    error * item_vector_old[factor_idx] - lambda_p * user_vector_old[factor_idx]
                )
                item_factors[item_id, factor_idx] = item_vector_old[factor_idx] + learning_rate * (
                    error * z_user[factor_idx] - lambda_q * item_vector_old[factor_idx]
                )

            for history_pos in range(history_start, history_end):
                history_item = int(history_items[history_pos])
                for factor_idx in range(latent_dim):
                    y_old = implicit_factors[history_item, factor_idx]
                    implicit_factors[history_item, factor_idx] = y_old + learning_rate * (
                        error * norm * item_vector_old[factor_idx] - lambda_y * y_old
                    )

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
                residual_weight = explicit_ratings[history_pos] - (global_mean + user_bias_old + history_item_bias)
                for factor_idx in range(latent_dim):
                    context[factor_idx] += explicit_norm * residual_weight * explicit_factors[history_item, factor_idx]

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

            user_bias[user_id] = user_bias_old + learning_rate * (error - lambda_b * user_bias_old)
            item_bias[item_id] = item_bias_old + learning_rate * (error - lambda_b * item_bias_old)
            for factor_idx in range(latent_dim):
                item_factors[item_id, factor_idx] = q_old[factor_idx] + learning_rate * (
                    error * context[factor_idx] - lambda_q * q_old[factor_idx]
                )

            for history_pos in range(explicit_start, explicit_end):
                history_item = explicit_items[history_pos]
                history_item_bias = item_bias_old if history_item == item_id else item_bias[history_item]
                residual_weight = explicit_ratings[history_pos] - (global_mean + user_bias_old + history_item_bias)
                for factor_idx in range(latent_dim):
                    x_old = explicit_factors[history_item, factor_idx]
                    explicit_factors[history_item, factor_idx] = x_old + learning_rate * (
                        error * explicit_norm * residual_weight * q_old[factor_idx] - lambda_x * x_old
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
                residual_weight = explicit_ratings[history_pos] - (global_mean + user_bias_old + history_item_bias)
                for factor_idx in range(latent_dim):
                    context[factor_idx] += explicit_norm * residual_weight * explicit_factors[history_item, factor_idx]

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

            user_bias[user_id] = user_bias_old + learning_rate * (error - lambda_b * user_bias_old)
            item_bias[item_id] = item_bias_old + learning_rate * (error - lambda_b * item_bias_old)
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
                residual_weight = explicit_ratings[history_pos] - (global_mean + user_bias_old + history_item_bias)
                for factor_idx in range(latent_dim):
                    x_old = explicit_factors[history_item, factor_idx]
                    explicit_factors[history_item, factor_idx] = x_old + learning_rate * (
                        error * explicit_norm * residual_weight * q_old[factor_idx] - lambda_x * x_old
                    )

            for history_pos in range(implicit_start, implicit_end):
                history_item = implicit_items[history_pos]
                for factor_idx in range(latent_dim):
                    y_old = implicit_factors[history_item, factor_idx]
                    implicit_factors[history_item, factor_idx] = y_old + learning_rate * (
                        error * implicit_norm * q_old[factor_idx] - lambda_y * y_old
                    )

    @njit(cache=True)
    def train_cb_svdpp_epoch_numba(
        order: np.ndarray,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        implicit_indptr: np.ndarray,
        implicit_items: np.ndarray,
        implicit_norms: np.ndarray,
        cluster_indptr: np.ndarray,
        cluster_ids: np.ndarray,
        cluster_counts: np.ndarray,
        user_clusters: np.ndarray,
        item_clusters: np.ndarray,
        alpha: float,
        global_mean: float,
        learning_rate: float,
        lambda_b: float,
        lambda_p: float,
        lambda_q: float,
        lambda_y: float,
        lambda_pC: float,
        lambda_qC: float,
        lambda_yC: float,
        user_bias: np.ndarray,
        item_bias: np.ndarray,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        implicit_factors: np.ndarray,
        user_cluster_factors: np.ndarray,
        item_cluster_factors: np.ndarray,
        implicit_cluster_factors: np.ndarray,
    ) -> None:
        latent_dim = item_factors.shape[1]
        one_minus_alpha = 1.0 - alpha
        p_old = np.empty(latent_dim, dtype=user_factors.dtype)
        p_cluster_old = np.empty(latent_dim, dtype=user_cluster_factors.dtype)
        q_old = np.empty(latent_dim, dtype=item_factors.dtype)
        q_cluster_old = np.empty(latent_dim, dtype=item_cluster_factors.dtype)
        q_mix_old = np.empty(latent_dim, dtype=item_factors.dtype)
        context = np.empty(latent_dim, dtype=item_factors.dtype)

        for position in range(order.shape[0]):
            idx = order[position]
            user_id = int(user_ids[idx])
            item_id = int(item_ids[idx])
            rating = ratings[idx]

            user_cluster_id = int(user_clusters[user_id])
            item_cluster_id = int(item_clusters[item_id])
            user_bias_old = user_bias[user_id]
            item_bias_old = item_bias[item_id]

            for factor_idx in range(latent_dim):
                p_old[factor_idx] = user_factors[user_id, factor_idx]
                p_cluster_old[factor_idx] = user_cluster_factors[user_cluster_id, factor_idx]
                q_old[factor_idx] = item_factors[item_id, factor_idx]
                q_cluster_old[factor_idx] = item_cluster_factors[item_cluster_id, factor_idx]
                q_mix_old[factor_idx] = one_minus_alpha * q_old[factor_idx] + alpha * q_cluster_old[factor_idx]
                context[factor_idx] = one_minus_alpha * p_old[factor_idx] + alpha * p_cluster_old[factor_idx]

            implicit_start = implicit_indptr[user_id]
            implicit_end = implicit_indptr[user_id + 1]
            implicit_norm = implicit_norms[user_id]
            for history_pos in range(implicit_start, implicit_end):
                history_item = implicit_items[history_pos]
                for factor_idx in range(latent_dim):
                    context[factor_idx] += implicit_norm * one_minus_alpha * implicit_factors[history_item, factor_idx]

            cluster_start = cluster_indptr[user_id]
            cluster_end = cluster_indptr[user_id + 1]
            for cluster_pos in range(cluster_start, cluster_end):
                history_cluster = int(cluster_ids[cluster_pos])
                history_cluster_count = float(cluster_counts[cluster_pos])
                for factor_idx in range(latent_dim):
                    context[factor_idx] += (
                        implicit_norm
                        * alpha
                        * history_cluster_count
                        * implicit_cluster_factors[history_cluster, factor_idx]
                    )

            prediction = global_mean + user_bias_old + item_bias_old
            for factor_idx in range(latent_dim):
                prediction += q_mix_old[factor_idx] * context[factor_idx]
            error = rating - prediction

            user_bias[user_id] = user_bias_old + learning_rate * (error - lambda_b * user_bias_old)
            item_bias[item_id] = item_bias_old + learning_rate * (error - lambda_b * item_bias_old)
            for factor_idx in range(latent_dim):
                user_factors[user_id, factor_idx] = p_old[factor_idx] + learning_rate * (
                    error * one_minus_alpha * q_mix_old[factor_idx] - lambda_p * p_old[factor_idx]
                )
                user_cluster_factors[user_cluster_id, factor_idx] = p_cluster_old[factor_idx] + learning_rate * (
                    error * alpha * q_mix_old[factor_idx] - lambda_pC * p_cluster_old[factor_idx]
                )
                item_factors[item_id, factor_idx] = q_old[factor_idx] + learning_rate * (
                    error * one_minus_alpha * context[factor_idx] - lambda_q * q_old[factor_idx]
                )
                item_cluster_factors[item_cluster_id, factor_idx] = q_cluster_old[factor_idx] + learning_rate * (
                    error * alpha * context[factor_idx] - lambda_qC * q_cluster_old[factor_idx]
                )

            for history_pos in range(implicit_start, implicit_end):
                history_item = implicit_items[history_pos]
                for factor_idx in range(latent_dim):
                    y_old = implicit_factors[history_item, factor_idx]
                    implicit_factors[history_item, factor_idx] = y_old + learning_rate * (
                        error * implicit_norm * one_minus_alpha * q_mix_old[factor_idx] - lambda_y * y_old
                    )

            for cluster_pos in range(cluster_start, cluster_end):
                history_cluster = int(cluster_ids[cluster_pos])
                history_cluster_count = float(cluster_counts[cluster_pos])
                for factor_idx in range(latent_dim):
                    y_cluster_old = implicit_cluster_factors[history_cluster, factor_idx]
                    implicit_cluster_factors[history_cluster, factor_idx] = y_cluster_old + learning_rate * (
                        error * implicit_norm * alpha * history_cluster_count * q_mix_old[factor_idx]
                        - lambda_yC * y_cluster_old
                    )

    @njit(cache=True)
    def train_cb_asvdpp_epoch_numba(
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
        cluster_indptr: np.ndarray,
        cluster_ids: np.ndarray,
        cluster_counts: np.ndarray,
        user_clusters: np.ndarray,
        item_clusters: np.ndarray,
        alpha: float,
        global_mean: float,
        learning_rate: float,
        lambda_b: float,
        lambda_p: float,
        lambda_q: float,
        lambda_x: float,
        lambda_y: float,
        lambda_pC: float,
        lambda_qC: float,
        lambda_xC: float,
        lambda_yC: float,
        user_bias: np.ndarray,
        item_bias: np.ndarray,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        explicit_factors: np.ndarray,
        implicit_factors: np.ndarray,
        user_cluster_factors: np.ndarray,
        item_cluster_factors: np.ndarray,
        explicit_cluster_factors: np.ndarray,
        implicit_cluster_factors: np.ndarray,
    ) -> None:
        latent_dim = item_factors.shape[1]
        one_minus_alpha = 1.0 - alpha
        p_old = np.empty(latent_dim, dtype=user_factors.dtype)
        p_cluster_old = np.empty(latent_dim, dtype=user_cluster_factors.dtype)
        q_old = np.empty(latent_dim, dtype=item_factors.dtype)
        q_cluster_old = np.empty(latent_dim, dtype=item_cluster_factors.dtype)
        q_mix_old = np.empty(latent_dim, dtype=item_factors.dtype)
        context = np.empty(latent_dim, dtype=item_factors.dtype)

        for position in range(order.shape[0]):
            idx = order[position]
            user_id = int(user_ids[idx])
            item_id = int(item_ids[idx])
            rating = ratings[idx]

            user_cluster_id = int(user_clusters[user_id])
            item_cluster_id = int(item_clusters[item_id])
            user_bias_old = user_bias[user_id]
            item_bias_old = item_bias[item_id]

            for factor_idx in range(latent_dim):
                p_old[factor_idx] = user_factors[user_id, factor_idx]
                p_cluster_old[factor_idx] = user_cluster_factors[user_cluster_id, factor_idx]
                q_old[factor_idx] = item_factors[item_id, factor_idx]
                q_cluster_old[factor_idx] = item_cluster_factors[item_cluster_id, factor_idx]
                q_mix_old[factor_idx] = one_minus_alpha * q_old[factor_idx] + alpha * q_cluster_old[factor_idx]
                context[factor_idx] = one_minus_alpha * p_old[factor_idx] + alpha * p_cluster_old[factor_idx]

            explicit_start = explicit_indptr[user_id]
            explicit_end = explicit_indptr[user_id + 1]
            explicit_norm = explicit_norms[user_id]
            for history_pos in range(explicit_start, explicit_end):
                history_item = int(explicit_items[history_pos])
                history_cluster = int(item_clusters[history_item])
                history_item_bias = item_bias_old if history_item == item_id else item_bias[history_item]
                residual_weight = explicit_ratings[history_pos] - (global_mean + user_bias_old + history_item_bias)
                for factor_idx in range(latent_dim):
                    context[factor_idx] += (
                        explicit_norm * one_minus_alpha * residual_weight * explicit_factors[history_item, factor_idx]
                    )
                    context[factor_idx] += (
                        explicit_norm * alpha * residual_weight * explicit_cluster_factors[history_cluster, factor_idx]
                    )

            implicit_start = implicit_indptr[user_id]
            implicit_end = implicit_indptr[user_id + 1]
            implicit_norm = implicit_norms[user_id]
            for history_pos in range(implicit_start, implicit_end):
                history_item = implicit_items[history_pos]
                for factor_idx in range(latent_dim):
                    context[factor_idx] += implicit_norm * one_minus_alpha * implicit_factors[history_item, factor_idx]

            cluster_start = cluster_indptr[user_id]
            cluster_end = cluster_indptr[user_id + 1]
            for cluster_pos in range(cluster_start, cluster_end):
                history_cluster = int(cluster_ids[cluster_pos])
                history_cluster_count = float(cluster_counts[cluster_pos])
                for factor_idx in range(latent_dim):
                    context[factor_idx] += (
                        implicit_norm
                        * alpha
                        * history_cluster_count
                        * implicit_cluster_factors[history_cluster, factor_idx]
                    )

            prediction = global_mean + user_bias_old + item_bias_old
            for factor_idx in range(latent_dim):
                prediction += q_mix_old[factor_idx] * context[factor_idx]
            error = rating - prediction

            user_bias[user_id] = user_bias_old + learning_rate * (error - lambda_b * user_bias_old)
            item_bias[item_id] = item_bias_old + learning_rate * (error - lambda_b * item_bias_old)
            for factor_idx in range(latent_dim):
                user_factors[user_id, factor_idx] = p_old[factor_idx] + learning_rate * (
                    error * one_minus_alpha * q_mix_old[factor_idx] - lambda_p * p_old[factor_idx]
                )
                user_cluster_factors[user_cluster_id, factor_idx] = p_cluster_old[factor_idx] + learning_rate * (
                    error * alpha * q_mix_old[factor_idx] - lambda_pC * p_cluster_old[factor_idx]
                )
                item_factors[item_id, factor_idx] = q_old[factor_idx] + learning_rate * (
                    error * one_minus_alpha * context[factor_idx] - lambda_q * q_old[factor_idx]
                )
                item_cluster_factors[item_cluster_id, factor_idx] = q_cluster_old[factor_idx] + learning_rate * (
                    error * alpha * context[factor_idx] - lambda_qC * q_cluster_old[factor_idx]
                )

            for history_pos in range(explicit_start, explicit_end):
                history_item = int(explicit_items[history_pos])
                history_cluster = int(item_clusters[history_item])
                history_item_bias = item_bias_old if history_item == item_id else item_bias[history_item]
                residual_weight = explicit_ratings[history_pos] - (global_mean + user_bias_old + history_item_bias)
                for factor_idx in range(latent_dim):
                    x_old = explicit_factors[history_item, factor_idx]
                    explicit_factors[history_item, factor_idx] = x_old + learning_rate * (
                        error * explicit_norm * one_minus_alpha * residual_weight * q_mix_old[factor_idx]
                        - lambda_x * x_old
                    )
                    x_cluster_old = explicit_cluster_factors[history_cluster, factor_idx]
                    explicit_cluster_factors[history_cluster, factor_idx] = x_cluster_old + learning_rate * (
                        error * explicit_norm * alpha * residual_weight * q_mix_old[factor_idx]
                        - lambda_xC * x_cluster_old
                    )

            for history_pos in range(implicit_start, implicit_end):
                history_item = implicit_items[history_pos]
                for factor_idx in range(latent_dim):
                    y_old = implicit_factors[history_item, factor_idx]
                    implicit_factors[history_item, factor_idx] = y_old + learning_rate * (
                        error * implicit_norm * one_minus_alpha * q_mix_old[factor_idx] - lambda_y * y_old
                    )

            for cluster_pos in range(cluster_start, cluster_end):
                history_cluster = int(cluster_ids[cluster_pos])
                history_cluster_count = float(cluster_counts[cluster_pos])
                for factor_idx in range(latent_dim):
                    y_cluster_old = implicit_cluster_factors[history_cluster, factor_idx]
                    implicit_cluster_factors[history_cluster, factor_idx] = y_cluster_old + learning_rate * (
                        error * implicit_norm * alpha * history_cluster_count * q_mix_old[factor_idx]
                        - lambda_yC * y_cluster_old
                    )

else:

    def train_biased_mf_epoch_numba(*args: object, **kwargs: object) -> None:
        raise RuntimeError("numba is not available")

    def train_svdpp_epoch_numba(*args: object, **kwargs: object) -> None:
        raise RuntimeError("numba is not available")

    def train_asymmetric_svd_epoch_numba(*args: object, **kwargs: object) -> None:
        raise RuntimeError("numba is not available")

    def train_asvdpp_epoch_numba(*args: object, **kwargs: object) -> None:
        raise RuntimeError("numba is not available")

    def train_cb_svdpp_epoch_numba(*args: object, **kwargs: object) -> None:
        raise RuntimeError("numba is not available")

    def train_cb_asvdpp_epoch_numba(*args: object, **kwargs: object) -> None:
        raise RuntimeError("numba is not available")
