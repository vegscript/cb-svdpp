from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean, median, pstdev
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np

from recsys_lab.benchmarks.synthetic_kernel_cases import (
    ArrayMap,
    KernelBenchmarkCase,
    ScalarMap,
    validate_kernel_benchmark_case,
)
from recsys_lab.experiments.kernel_profile import (
    estimate_kernel_work,
    summarize_history_index,
)
from recsys_lab.models.kernels import (
    train_asvdpp_epoch_numba,
    train_asymmetric_svd_epoch_numba,
    train_biased_mf_epoch_numba,
    train_cb_asvdpp_epoch_numba,
    train_cb_svdpp_alpha0_epoch_numba,
    train_cb_svdpp_epoch_numba,
    train_svdpp_epoch_numba,
)

BENCHMARK_VERSION = "kernel_benchmark_harness_v1"
DATASET_PROFILE = "synthetic_tiny"
CLAIM_BOUNDARY = "Diagnostic kernel benchmark only; no broad performance claim."

# Training kernels mutate model parameters in place: biases, latent factors,
# explicit/implicit feedback factors, and cluster-level factors. Input arrays
# such as ids, ratings, indptr, items, norms, and cluster assignments are reused.
MUTATED_ARRAYS_BY_MODEL: dict[str, tuple[str, ...]] = {
    "biased_mf": (
        "user_bias",
        "item_bias",
        "user_factors",
        "item_factors",
    ),
    "svdpp": (
        "user_bias",
        "item_bias",
        "user_factors",
        "item_factors",
        "implicit_factors",
    ),
    "asymmetric_svd": (
        "user_bias",
        "item_bias",
        "item_factors",
        "explicit_factors",
        "implicit_factors",
    ),
    "asvdpp": (
        "user_bias",
        "item_bias",
        "user_factors",
        "item_factors",
        "explicit_factors",
        "implicit_factors",
    ),
    "cb_svdpp": (
        "user_bias",
        "item_bias",
        "user_factors",
        "item_factors",
        "implicit_factors",
        "user_cluster_factors",
        "item_cluster_factors",
        "implicit_cluster_factors",
    ),
    "cb_svdpp_alpha0": (
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


@dataclass(frozen=True, slots=True)
class KernelState:
    case: KernelBenchmarkCase
    arrays: ArrayMap
    scalars: ScalarMap
    mutated_arrays: tuple[str, ...]


KernelRunner = Callable[[KernelState], None]


def clone_case_state(case: KernelBenchmarkCase) -> KernelState:
    validate_kernel_benchmark_case(case)
    mutated_arrays = mutated_arrays_for_model(case.model)
    missing = sorted(name for name in mutated_arrays if name not in case.arrays)
    if missing:
        raise ValueError(f"{case.name} missing mutated arrays: {missing}")

    mutable_names = set(mutated_arrays)
    arrays = {
        name: np.ascontiguousarray(array.copy()) if name in mutable_names else array
        for name, array in case.arrays.items()
    }
    state = KernelState(
        case=case,
        arrays=arrays,
        scalars=dict(case.scalars),
        mutated_arrays=mutated_arrays,
    )
    assert_mutated_state_finite(state)
    return state


def mutated_arrays_for_model(model: str) -> tuple[str, ...]:
    try:
        return MUTATED_ARRAYS_BY_MODEL[model]
    except KeyError as exc:
        raise KeyError(f"unknown kernel benchmark model: {model}") from exc


def assert_mutated_state_finite(state: KernelState) -> None:
    for name in state.mutated_arrays:
        array = state.arrays[name]
        if not array.flags.c_contiguous:
            raise ValueError(f"{state.case.name}.{name} must remain contiguous")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{state.case.name}.{name} contains non-finite values after run")


def run_biased_mf_kernel_once(state: KernelState) -> None:
    arrays = state.arrays
    scalars = state.scalars
    train_biased_mf_epoch_numba(
        arrays["order"],
        arrays["user_ids"],
        arrays["item_ids"],
        arrays["ratings"],
        scalars["global_mean"],
        scalars["learning_rate"],
        scalars["lambda_b"],
        scalars["lambda_p"],
        scalars["lambda_q"],
        arrays["user_bias"],
        arrays["item_bias"],
        arrays["user_factors"],
        arrays["item_factors"],
    )


def run_svdpp_kernel_once(state: KernelState) -> None:
    arrays = state.arrays
    scalars = state.scalars
    train_svdpp_epoch_numba(
        arrays["order"],
        arrays["user_ids"],
        arrays["item_ids"],
        arrays["ratings"],
        arrays["implicit_indptr"],
        arrays["implicit_items"],
        arrays["implicit_norms"],
        scalars["global_mean"],
        scalars["learning_rate"],
        scalars["lambda_b"],
        scalars["lambda_p"],
        scalars["lambda_q"],
        scalars["lambda_y"],
        arrays["user_bias"],
        arrays["item_bias"],
        arrays["user_factors"],
        arrays["item_factors"],
        arrays["implicit_factors"],
    )


def run_asymmetric_svd_kernel_once(state: KernelState) -> None:
    arrays = state.arrays
    scalars = state.scalars
    train_asymmetric_svd_epoch_numba(
        arrays["order"],
        arrays["user_ids"],
        arrays["item_ids"],
        arrays["ratings"],
        arrays["explicit_indptr"],
        arrays["explicit_items"],
        arrays["explicit_ratings"],
        arrays["explicit_norms"],
        arrays["implicit_indptr"],
        arrays["implicit_items"],
        arrays["implicit_norms"],
        scalars["global_mean"],
        scalars["learning_rate"],
        scalars["lambda_b"],
        scalars["lambda_q"],
        scalars["lambda_x"],
        scalars["lambda_y"],
        arrays["user_bias"],
        arrays["item_bias"],
        arrays["item_factors"],
        arrays["explicit_factors"],
        arrays["implicit_factors"],
    )


def run_asvdpp_kernel_once(state: KernelState) -> None:
    arrays = state.arrays
    scalars = state.scalars
    train_asvdpp_epoch_numba(
        arrays["order"],
        arrays["user_ids"],
        arrays["item_ids"],
        arrays["ratings"],
        arrays["explicit_indptr"],
        arrays["explicit_items"],
        arrays["explicit_ratings"],
        arrays["explicit_norms"],
        arrays["implicit_indptr"],
        arrays["implicit_items"],
        arrays["implicit_norms"],
        scalars["global_mean"],
        scalars["learning_rate"],
        scalars["lambda_b"],
        scalars["lambda_p"],
        scalars["lambda_q"],
        scalars["lambda_x"],
        scalars["lambda_y"],
        arrays["user_bias"],
        arrays["item_bias"],
        arrays["user_factors"],
        arrays["item_factors"],
        arrays["explicit_factors"],
        arrays["implicit_factors"],
    )


def run_cb_svdpp_kernel_once(state: KernelState) -> None:
    arrays = state.arrays
    scalars = state.scalars
    train_cb_svdpp_epoch_numba(
        arrays["order"],
        arrays["user_ids"],
        arrays["item_ids"],
        arrays["ratings"],
        arrays["implicit_indptr"],
        arrays["implicit_items"],
        arrays["implicit_norms"],
        arrays["cluster_indptr"],
        arrays["cluster_ids"],
        arrays["cluster_counts"],
        arrays["user_clusters"],
        arrays["item_clusters"],
        scalars["alpha"],
        scalars["global_mean"],
        scalars["learning_rate"],
        scalars["lambda_b"],
        scalars["lambda_p"],
        scalars["lambda_q"],
        scalars["lambda_y"],
        scalars["lambda_pC"],
        scalars["lambda_qC"],
        scalars["lambda_yC"],
        arrays["user_bias"],
        arrays["item_bias"],
        arrays["user_factors"],
        arrays["item_factors"],
        arrays["implicit_factors"],
        arrays["user_cluster_factors"],
        arrays["item_cluster_factors"],
        arrays["implicit_cluster_factors"],
    )


def run_cb_svdpp_alpha0_kernel_once(state: KernelState) -> None:
    arrays = state.arrays
    scalars = state.scalars
    train_cb_svdpp_alpha0_epoch_numba(
        arrays["order"],
        arrays["user_ids"],
        arrays["item_ids"],
        arrays["ratings"],
        arrays["implicit_indptr"],
        arrays["implicit_items"],
        arrays["implicit_norms"],
        arrays["cluster_indptr"],
        arrays["cluster_ids"],
        arrays["cluster_counts"],
        arrays["user_clusters"],
        arrays["item_clusters"],
        scalars["global_mean"],
        scalars["learning_rate"],
        scalars["lambda_b"],
        scalars["lambda_p"],
        scalars["lambda_q"],
        scalars["lambda_y"],
        scalars["lambda_pC"],
        scalars["lambda_qC"],
        scalars["lambda_yC"],
        arrays["user_bias"],
        arrays["item_bias"],
        arrays["user_factors"],
        arrays["item_factors"],
        arrays["implicit_factors"],
        arrays["user_cluster_factors"],
        arrays["item_cluster_factors"],
        arrays["implicit_cluster_factors"],
    )


def run_cb_asvdpp_kernel_once(state: KernelState) -> None:
    arrays = state.arrays
    scalars = state.scalars
    train_cb_asvdpp_epoch_numba(
        arrays["order"],
        arrays["user_ids"],
        arrays["item_ids"],
        arrays["ratings"],
        arrays["explicit_indptr"],
        arrays["explicit_items"],
        arrays["explicit_ratings"],
        arrays["explicit_norms"],
        arrays["implicit_indptr"],
        arrays["implicit_items"],
        arrays["implicit_norms"],
        arrays["cluster_indptr"],
        arrays["cluster_ids"],
        arrays["cluster_counts"],
        arrays["user_clusters"],
        arrays["item_clusters"],
        scalars["alpha"],
        scalars["global_mean"],
        scalars["learning_rate"],
        scalars["lambda_b"],
        scalars["lambda_p"],
        scalars["lambda_q"],
        scalars["lambda_x"],
        scalars["lambda_y"],
        scalars["lambda_pC"],
        scalars["lambda_qC"],
        scalars["lambda_xC"],
        scalars["lambda_yC"],
        arrays["user_bias"],
        arrays["item_bias"],
        arrays["user_factors"],
        arrays["item_factors"],
        arrays["explicit_factors"],
        arrays["implicit_factors"],
        arrays["user_cluster_factors"],
        arrays["item_cluster_factors"],
        arrays["explicit_cluster_factors"],
        arrays["implicit_cluster_factors"],
    )


KERNEL_DISPATCH: dict[str, KernelRunner] = {
    "biased_mf": run_biased_mf_kernel_once,
    "svdpp": run_svdpp_kernel_once,
    "asymmetric_svd": run_asymmetric_svd_kernel_once,
    "asvdpp": run_asvdpp_kernel_once,
    "cb_svdpp": run_cb_svdpp_kernel_once,
    "cb_svdpp_alpha0": run_cb_svdpp_alpha0_kernel_once,
    "cb_asvdpp": run_cb_asvdpp_kernel_once,
}


def run_kernel_benchmark(
    case: KernelBenchmarkCase,
    warmup_repeats: int = 1,
    timed_repeats: int = 5,
    epochs_per_repeat: int = 1,
) -> dict[str, Any]:
    if warmup_repeats < 1:
        raise ValueError("warmup_repeats must be at least 1 to exclude Numba compile time from timed repeats")
    if timed_repeats < 1:
        raise ValueError("timed_repeats must be positive")
    if epochs_per_repeat < 1:
        raise ValueError("epochs_per_repeat must be positive")

    validate_kernel_benchmark_case(case)
    kernel = kernel_runner_for_model(case.model)

    warmup_seconds = []
    for _ in range(warmup_repeats):
        state = clone_case_state(case)
        kernel_once = _bind_kernel_runner(kernel, state)
        start = perf_counter()
        _run_epochs(kernel_once, epochs_per_repeat)
        elapsed = perf_counter() - start
        assert_mutated_state_finite(state)
        warmup_seconds.append(elapsed)

    timed_seconds = []
    for _ in range(timed_repeats):
        state = clone_case_state(case)
        kernel_once = _bind_kernel_runner(kernel, state)
        start = perf_counter()
        _run_epochs(kernel_once, epochs_per_repeat)
        elapsed = perf_counter() - start
        assert_mutated_state_finite(state)
        timed_seconds.append(elapsed)

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_id": _benchmark_id(case, warmup_repeats, timed_repeats, epochs_per_repeat),
        "case_name": case.name,
        "kernel_name": case.kernel_name,
        "model": case.model,
        "dataset_profile": DATASET_PROFILE,
        "dtype": case.dtype,
        "latent_dim": case.latent_dim,
        "train_rows": case.train_rows,
        "epochs_per_repeat": epochs_per_repeat,
        "warmup_repeats": warmup_repeats,
        "timed_repeats": timed_repeats,
        "compile_excluded": True,
        "compile_exclusion_method": "warmup_repeats_before_timed_repeats",
        "state_copy_excluded": True,
        "warmup_wall_seconds": warmup_seconds,
        "warmup_excluded_from_timed": True,
        "repeat_wall_seconds": timed_seconds,
        **_timing_summary(timed_seconds=timed_seconds, train_rows=case.train_rows, epochs_per_repeat=epochs_per_repeat),
        **_kernel_work_summary(
            case=case,
            epochs_per_repeat=epochs_per_repeat,
            mean_wall_seconds=float(mean(timed_seconds)),
        ),
        "state_checks": {
            "finite_parameters_after": True,
            "mutated_array_count": len(mutated_arrays_for_model(case.model)),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def kernel_runner_for_model(model: str) -> KernelRunner:
    try:
        return KERNEL_DISPATCH[model]
    except KeyError as exc:
        raise KeyError(f"unknown kernel benchmark model: {model}") from exc


def _bind_kernel_runner(kernel: KernelRunner, state: KernelState) -> Callable[[], None]:
    def kernel_once() -> None:
        kernel(state)

    return kernel_once


def _run_epochs(kernel_once: Callable[[], None], epochs_per_repeat: int) -> None:
    for _ in range(epochs_per_repeat):
        kernel_once()


def _benchmark_id(
    case: KernelBenchmarkCase,
    warmup_repeats: int,
    timed_repeats: int,
    epochs_per_repeat: int,
) -> str:
    return f"{case.name}_wr{warmup_repeats}_tr{timed_repeats}_epr{epochs_per_repeat}"


def _timing_summary(
    *,
    timed_seconds: list[float],
    train_rows: int,
    epochs_per_repeat: int,
) -> dict[str, float]:
    mean_wall_seconds = float(mean(timed_seconds))
    estimated_rating_updates = float(train_rows) * float(epochs_per_repeat)
    return {
        "mean_wall_seconds": mean_wall_seconds,
        "median_wall_seconds": float(median(timed_seconds)),
        "std_wall_seconds": float(pstdev(timed_seconds)) if len(timed_seconds) > 1 else 0.0,
        "min_wall_seconds": float(min(timed_seconds)),
        "max_wall_seconds": float(max(timed_seconds)),
        "ratings_per_second_mean": _safe_divide(estimated_rating_updates, mean_wall_seconds),
    }


def _kernel_work_summary(case: KernelBenchmarkCase, epochs_per_repeat: int, mean_wall_seconds: float) -> dict[str, Any]:
    history_structure = _history_structure(case)
    kernel_work = estimate_kernel_work(
        train_user_ids=case.arrays["user_ids"],
        train_rows=case.train_rows,
        epochs=epochs_per_repeat,
        latent_dim=case.latent_dim,
        implicit_history_index=_history_index(case, "implicit"),
        explicit_feedback_index=_history_index(case, "explicit"),
        cluster_history_index=_history_index(case, "cluster"),
    )
    estimated_factor_touches = int(kernel_work["estimated_factor_touches"])
    return {
        "history_structure": history_structure,
        "estimated_factor_touches": estimated_factor_touches,
        "seconds_per_million_estimated_factor_touches": _safe_divide(
            mean_wall_seconds,
            float(estimated_factor_touches) / 1_000_000.0,
        ),
        "estimated_kernel_work": kernel_work,
        "touches_note": (
            "Synthetic estimated touches reuse Kernel Cost Anatomy V1 and are diagnostic, not CPU instructions."
        ),
    }


def _history_structure(case: KernelBenchmarkCase) -> dict[str, Any]:
    return {
        "implicit": summarize_history_index(_history_index(case, "implicit")),
        "explicit": summarize_history_index(_history_index(case, "explicit")),
        "cluster": summarize_history_index(_history_index(case, "cluster")),
    }


def _history_index(case: KernelBenchmarkCase, prefix: str) -> object | None:
    indptr_name = f"{prefix}_indptr"
    if indptr_name not in case.arrays:
        return None
    return SimpleNamespace(indptr=case.arrays[indptr_name])


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)
