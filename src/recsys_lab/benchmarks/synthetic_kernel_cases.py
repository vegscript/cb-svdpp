from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ArrayMap = dict[str, np.ndarray]
ScalarMap = dict[str, float | int]

KERNEL_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "biased_mf": (
        "order",
        "user_ids",
        "item_ids",
        "ratings",
        "global_mean",
        "learning_rate",
        "lambda_b",
        "lambda_p",
        "lambda_q",
        "user_bias",
        "item_bias",
        "user_factors",
        "item_factors",
    ),
    "svdpp": (
        "order",
        "user_ids",
        "item_ids",
        "ratings",
        "implicit_indptr",
        "implicit_items",
        "implicit_norms",
        "global_mean",
        "learning_rate",
        "lambda_b",
        "lambda_p",
        "lambda_q",
        "lambda_y",
        "user_bias",
        "item_bias",
        "user_factors",
        "item_factors",
        "implicit_factors",
    ),
    "asymmetric_svd": (
        "order",
        "user_ids",
        "item_ids",
        "ratings",
        "explicit_indptr",
        "explicit_items",
        "explicit_ratings",
        "explicit_norms",
        "implicit_indptr",
        "implicit_items",
        "implicit_norms",
        "global_mean",
        "learning_rate",
        "lambda_b",
        "lambda_q",
        "lambda_x",
        "lambda_y",
        "user_bias",
        "item_bias",
        "item_factors",
        "explicit_factors",
        "implicit_factors",
    ),
    "asvdpp": (
        "order",
        "user_ids",
        "item_ids",
        "ratings",
        "explicit_indptr",
        "explicit_items",
        "explicit_ratings",
        "explicit_norms",
        "implicit_indptr",
        "implicit_items",
        "implicit_norms",
        "global_mean",
        "learning_rate",
        "lambda_b",
        "lambda_p",
        "lambda_q",
        "lambda_x",
        "lambda_y",
        "user_bias",
        "item_bias",
        "user_factors",
        "item_factors",
        "explicit_factors",
        "implicit_factors",
    ),
    "cb_svdpp": (
        "order",
        "user_ids",
        "item_ids",
        "ratings",
        "implicit_indptr",
        "implicit_items",
        "implicit_norms",
        "cluster_indptr",
        "cluster_ids",
        "cluster_counts",
        "user_clusters",
        "item_clusters",
        "alpha",
        "global_mean",
        "learning_rate",
        "lambda_b",
        "lambda_p",
        "lambda_q",
        "lambda_y",
        "lambda_pC",
        "lambda_qC",
        "lambda_yC",
        "user_bias",
        "item_bias",
        "user_factors",
        "item_factors",
        "implicit_factors",
        "user_cluster_factors",
        "item_cluster_factors",
        "implicit_cluster_factors",
    ),
    "cb_asvdpp": (
        "order",
        "user_ids",
        "item_ids",
        "ratings",
        "explicit_indptr",
        "explicit_items",
        "explicit_ratings",
        "explicit_norms",
        "implicit_indptr",
        "implicit_items",
        "implicit_norms",
        "cluster_indptr",
        "cluster_ids",
        "cluster_counts",
        "user_clusters",
        "item_clusters",
        "alpha",
        "global_mean",
        "learning_rate",
        "lambda_b",
        "lambda_p",
        "lambda_q",
        "lambda_x",
        "lambda_y",
        "lambda_pC",
        "lambda_qC",
        "lambda_xC",
        "lambda_yC",
        "user_bias",
        "item_bias",
        "user_factors",
        "item_factors",
        "explicit_factors",
        "implicit_factors",
        "user_cluster_factors",
        "item_cluster_factors",
        "explicit_cluster_factors",
        "implicit_cluster_factors",
    ),
}

KERNEL_NAMES: dict[str, str] = {
    "biased_mf": "train_biased_mf_epoch_numba",
    "svdpp": "train_svdpp_epoch_numba",
    "asymmetric_svd": "train_asymmetric_svd_epoch_numba",
    "asvdpp": "train_asvdpp_epoch_numba",
    "cb_svdpp": "train_cb_svdpp_epoch_numba",
    "cb_asvdpp": "train_cb_asvdpp_epoch_numba",
}


@dataclass(frozen=True, slots=True)
class KernelBenchmarkCase:
    name: str
    model: str
    kernel_name: str
    dtype: str
    latent_dim: int
    train_rows: int
    arrays: ArrayMap
    scalars: ScalarMap
    metadata: dict[str, Any]

    def clone_arrays(self) -> ArrayMap:
        return {name: np.ascontiguousarray(array.copy()) for name, array in self.arrays.items()}


def build_synthetic_kernel_cases() -> tuple[KernelBenchmarkCase, ...]:
    base_arrays = _build_base_arrays()
    scalars: ScalarMap = {
        "alpha": 0.25,
        "global_mean": 3.5,
        "learning_rate": 0.005,
        "lambda_b": 0.02,
        "lambda_p": 0.02,
        "lambda_q": 0.02,
        "lambda_x": 0.02,
        "lambda_y": 0.02,
        "lambda_pC": 0.02,
        "lambda_qC": 0.02,
        "lambda_xC": 0.02,
        "lambda_yC": 0.02,
    }
    metadata: dict[str, Any] = {
        "case_family": "synthetic_kernel_case_v1",
        "n_users": 4,
        "n_items": 5,
        "n_ratings": 16,
        "n_user_clusters": 2,
        "n_item_clusters": 3,
        "rating_min": 1.0,
        "rating_max": 5.0,
        "order_policy": "identity_int32",
        "notes": "Synthetic inputs only; no model formula or update rule is implemented here.",
    }

    cases = tuple(
        KernelBenchmarkCase(
            name=f"tiny_{model}_float32",
            model=model,
            kernel_name=KERNEL_NAMES[model],
            dtype="float32",
            latent_dim=3,
            train_rows=int(base_arrays["ratings"].shape[0]),
            arrays=_select_arrays_for_model(model, base_arrays),
            scalars=_select_scalars_for_model(model, scalars),
            metadata={**metadata, "kernel_arguments": KERNEL_ARGUMENTS[model]},
        )
        for model in KERNEL_ARGUMENTS
    )
    for case in cases:
        validate_kernel_benchmark_case(case)
    return cases


def get_synthetic_kernel_case(model: str) -> KernelBenchmarkCase:
    for case in build_synthetic_kernel_cases():
        if case.model == model:
            return case
    raise KeyError(f"unknown synthetic kernel case model: {model}")


def validate_kernel_benchmark_case(case: KernelBenchmarkCase) -> None:
    required_names = set(KERNEL_ARGUMENTS[case.model])
    available_names = set(case.arrays) | set(case.scalars)
    missing = sorted(required_names - available_names)
    if missing:
        raise ValueError(f"{case.name} missing kernel inputs: {missing}")

    if case.dtype != "float32":
        raise ValueError(f"{case.name} must use float32 dtype")
    if case.latent_dim < 1:
        raise ValueError(f"{case.name} latent_dim must be positive")
    if case.train_rows != int(case.arrays["ratings"].shape[0]):
        raise ValueError(f"{case.name} train_rows must match ratings length")

    _validate_arrays(case)
    if "implicit_indptr" in case.arrays:
        _validate_history_index(case, "implicit")
    if "explicit_indptr" in case.arrays:
        _validate_history_index(case, "explicit", has_ratings=True)
    if "cluster_indptr" in case.arrays:
        _validate_cluster_index(case)


def _build_base_arrays() -> ArrayMap:
    dtype = np.float32
    n_users = 4
    n_items = 5
    latent_dim = 3

    arrays: ArrayMap = {
        "order": _as_int_array(range(16)),
        "user_ids": _as_int_array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]),
        "item_ids": _as_int_array([0, 1, 2, 3, 4, 2, 3, 4, 1, 0, 4, 2, 3, 1, 0, 4]),
        "ratings": _as_float_array(
            [4.0, 5.0, 3.0, 2.0, 4.5, 3.5, 1.5, 5.0, 2.5, 4.0, 3.0, 2.0, 5.0, 3.5, 1.0, 4.5]
        ),
        "implicit_indptr": _as_int_array([0, 3, 5, 8, 10]),
        "implicit_items": _as_int_array([0, 1, 3, 1, 2, 2, 4, 0, 3, 4]),
        "explicit_indptr": _as_int_array([0, 2, 4, 7, 9]),
        "explicit_items": _as_int_array([0, 3, 1, 2, 2, 4, 0, 3, 4]),
        "explicit_ratings": _as_float_array([4.0, 2.5, 5.0, 3.5, 3.0, 4.5, 1.5, 2.0, 5.0]),
        "cluster_indptr": _as_int_array([0, 2, 4, 7, 9]),
        "cluster_ids": _as_int_array([0, 1, 1, 2, 0, 1, 2, 0, 1]),
        "cluster_counts": _as_int_array([2, 1, 1, 1, 1, 1, 1, 1, 1]),
        "user_clusters": _as_int_array([0, 1, 0, 1]),
        "item_clusters": _as_int_array([0, 1, 2, 0, 1]),
        "user_bias": _as_float_array([0.02, -0.03, 0.01, -0.01]),
        "item_bias": _as_float_array([0.01, -0.02, 0.03, 0.0, -0.01]),
    }
    arrays["implicit_norms"] = _history_norms(arrays["implicit_indptr"])
    arrays["explicit_norms"] = _history_norms(arrays["explicit_indptr"])

    arrays["user_factors"] = _factor_matrix(n_users, latent_dim, start=0.01, dtype=dtype)
    arrays["item_factors"] = _factor_matrix(n_items, latent_dim, start=0.04, dtype=dtype)
    arrays["explicit_factors"] = _factor_matrix(n_items, latent_dim, start=0.07, dtype=dtype)
    arrays["implicit_factors"] = _factor_matrix(n_items, latent_dim, start=0.10, dtype=dtype)
    arrays["user_cluster_factors"] = _factor_matrix(2, latent_dim, start=0.13, dtype=dtype)
    arrays["item_cluster_factors"] = _factor_matrix(3, latent_dim, start=0.16, dtype=dtype)
    arrays["explicit_cluster_factors"] = _factor_matrix(3, latent_dim, start=0.19, dtype=dtype)
    arrays["implicit_cluster_factors"] = _factor_matrix(3, latent_dim, start=0.22, dtype=dtype)

    return {name: np.ascontiguousarray(array.astype(array.dtype, copy=False)) for name, array in arrays.items()}


def _select_arrays_for_model(model: str, arrays: ArrayMap) -> ArrayMap:
    selected = {name: arrays[name] for name in KERNEL_ARGUMENTS[model] if name in arrays}
    return {name: np.ascontiguousarray(array.copy()) for name, array in selected.items()}


def _select_scalars_for_model(model: str, scalars: ScalarMap) -> ScalarMap:
    return {name: scalars[name] for name in KERNEL_ARGUMENTS[model] if name in scalars}


def _as_int_array(values: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.int32))


def _as_float_array(values: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float32))


def _history_norms(indptr: np.ndarray) -> np.ndarray:
    lengths = np.diff(indptr).astype(np.float32, copy=False)
    if np.any(lengths <= 0):
        raise ValueError("synthetic histories must not contain empty users")
    return np.ascontiguousarray(1.0 / np.sqrt(lengths, dtype=np.float32))


def _factor_matrix(rows: int, cols: int, *, start: float, dtype: type[np.float32]) -> np.ndarray:
    values = np.arange(rows * cols, dtype=dtype).reshape(rows, cols)
    return np.ascontiguousarray(start + values * dtype(0.01))


def _validate_arrays(case: KernelBenchmarkCase) -> None:
    for name, array in case.arrays.items():
        if not array.flags.c_contiguous:
            raise ValueError(f"{case.name}.{name} must be contiguous")
        if array.ndim == 0:
            raise ValueError(f"{case.name}.{name} must not be scalar array")
        if array.dtype.kind in {"f", "c"} and not np.all(np.isfinite(array)):
            raise ValueError(f"{case.name}.{name} must contain only finite values")
        if array.dtype.kind in {"i", "u"} and not np.all(np.isfinite(array.astype(np.float64))):
            raise ValueError(f"{case.name}.{name} must contain only finite values")

    if case.arrays["order"].dtype != np.int32:
        raise ValueError(f"{case.name}.order must be int32")
    if case.arrays["user_ids"].dtype != np.int32:
        raise ValueError(f"{case.name}.user_ids must be int32")
    if case.arrays["item_ids"].dtype != np.int32:
        raise ValueError(f"{case.name}.item_ids must be int32")
    if case.arrays["ratings"].dtype != np.float32:
        raise ValueError(f"{case.name}.ratings must be float32")
    if np.min(case.arrays["ratings"]) < 1.0 or np.max(case.arrays["ratings"]) > 5.0:
        raise ValueError(f"{case.name}.ratings must be in [1, 5]")
    if not np.array_equal(np.sort(case.arrays["order"]), np.arange(case.train_rows, dtype=np.int32)):
        raise ValueError(f"{case.name}.order must be a permutation of train row indices")


def _validate_history_index(case: KernelBenchmarkCase, prefix: str, *, has_ratings: bool = False) -> None:
    indptr = case.arrays[f"{prefix}_indptr"]
    items = case.arrays[f"{prefix}_items"]
    norms = case.arrays[f"{prefix}_norms"]

    if indptr.dtype != np.int32 or items.dtype != np.int32:
        raise ValueError(f"{case.name}.{prefix} history ids must be int32")
    if norms.dtype != np.float32:
        raise ValueError(f"{case.name}.{prefix}_norms must be float32")
    if indptr.shape[0] != int(case.metadata["n_users"]) + 1:
        raise ValueError(f"{case.name}.{prefix}_indptr has invalid length")
    if int(indptr[0]) != 0 or int(indptr[-1]) != int(items.shape[0]):
        raise ValueError(f"{case.name}.{prefix}_indptr endpoints are invalid")
    if np.any(np.diff(indptr) <= 0):
        raise ValueError(f"{case.name}.{prefix} histories must be non-empty for each user")
    if np.min(items) < 0 or np.max(items) >= int(case.metadata["n_items"]):
        raise ValueError(f"{case.name}.{prefix}_items contains invalid item ids")
    if has_ratings:
        explicit_ratings = case.arrays[f"{prefix}_ratings"]
        if explicit_ratings.dtype != np.float32:
            raise ValueError(f"{case.name}.{prefix}_ratings must be float32")
        if explicit_ratings.shape != items.shape:
            raise ValueError(f"{case.name}.{prefix}_ratings must align with items")
        if np.min(explicit_ratings) < 1.0 or np.max(explicit_ratings) > 5.0:
            raise ValueError(f"{case.name}.{prefix}_ratings must be in [1, 5]")


def _validate_cluster_index(case: KernelBenchmarkCase) -> None:
    indptr = case.arrays["cluster_indptr"]
    cluster_ids = case.arrays["cluster_ids"]
    cluster_counts = case.arrays["cluster_counts"]
    user_clusters = case.arrays["user_clusters"]
    item_clusters = case.arrays["item_clusters"]

    if any(array.dtype != np.int32 for array in (indptr, cluster_ids, cluster_counts, user_clusters, item_clusters)):
        raise ValueError(f"{case.name} cluster id/count arrays must be int32")
    if indptr.shape[0] != int(case.metadata["n_users"]) + 1:
        raise ValueError(f"{case.name}.cluster_indptr has invalid length")
    if int(indptr[0]) != 0 or int(indptr[-1]) != int(cluster_ids.shape[0]):
        raise ValueError(f"{case.name}.cluster_indptr endpoints are invalid")
    if cluster_ids.shape != cluster_counts.shape:
        raise ValueError(f"{case.name}.cluster_ids and cluster_counts must align")
    if np.any(np.diff(indptr) <= 0):
        raise ValueError(f"{case.name}.cluster histories must be non-empty for each user")
    if np.min(cluster_counts) < 1:
        raise ValueError(f"{case.name}.cluster_counts must be positive")
    if np.min(cluster_ids) < 0 or np.max(cluster_ids) >= int(case.metadata["n_item_clusters"]):
        raise ValueError(f"{case.name}.cluster_ids contains invalid item cluster ids")
    if np.min(user_clusters) < 0 or np.max(user_clusters) >= int(case.metadata["n_user_clusters"]):
        raise ValueError(f"{case.name}.user_clusters contains invalid user cluster ids")
    if np.min(item_clusters) < 0 or np.max(item_clusters) >= int(case.metadata["n_item_clusters"]):
        raise ValueError(f"{case.name}.item_clusters contains invalid item cluster ids")
