from __future__ import annotations

import numpy as np
import pytest

from recsys_lab.benchmarks.kernel_harness import (
    BENCHMARK_VERSION,
    MUTATED_ARRAYS_BY_MODEL,
    assert_mutated_state_finite,
    clone_case_state,
    kernel_runner_for_model,
    run_kernel_benchmark,
)
from recsys_lab.benchmarks.synthetic_kernel_cases import get_synthetic_kernel_case

TARGET_MODELS = ("asymmetric_svd", "asvdpp", "cb_asvdpp")
NON_TARGET_MODELS = ("biased_mf", "svdpp", "cb_svdpp")


@pytest.mark.parametrize("model", TARGET_MODELS)
def test_exact_residual_reuse_target_cases_run(model: str) -> None:
    case = get_synthetic_kernel_case(model)
    state = clone_case_state(case)

    kernel_runner_for_model(model)(state)

    assert_mutated_state_finite(state)


@pytest.mark.parametrize("model", TARGET_MODELS)
def test_exact_residual_reuse_target_cases_keep_finite_state(model: str) -> None:
    case = get_synthetic_kernel_case(model)
    state = clone_case_state(case)
    runner = kernel_runner_for_model(model)

    runner(state)
    runner(state)

    for name in MUTATED_ARRAYS_BY_MODEL[model]:
        actual = state.arrays[name]
        original = case.arrays[name]
        assert actual.shape == original.shape, name
        assert actual.dtype == original.dtype, name
        assert actual.flags.c_contiguous, name
        assert np.all(np.isfinite(actual)), name


@pytest.mark.parametrize("model", NON_TARGET_MODELS)
def test_exact_residual_reuse_does_not_modify_non_target_cases(model: str) -> None:
    case = get_synthetic_kernel_case(model)
    state = clone_case_state(case)

    kernel_runner_for_model(model)(state)

    assert state.mutated_arrays == MUTATED_ARRAYS_BY_MODEL[model]
    assert_mutated_state_finite(state)


@pytest.mark.parametrize("model", TARGET_MODELS)
def test_kernel_benchmark_payload_still_valid_for_target_cases(model: str) -> None:
    case = get_synthetic_kernel_case(model)

    payload = run_kernel_benchmark(
        case,
        warmup_repeats=1,
        timed_repeats=1,
        epochs_per_repeat=1,
    )

    assert payload["benchmark_version"] == BENCHMARK_VERSION
    assert payload["model"] == model
    assert payload["kernel_name"] == case.kernel_name
    assert payload["dataset_profile"] == "synthetic_tiny"
    assert payload["compile_excluded"] is True
    assert payload["state_copy_excluded"] is True
    assert payload["warmup_repeats"] == 1
    assert payload["timed_repeats"] == 1
    assert payload["epochs_per_repeat"] == 1
    assert len(payload["repeat_wall_seconds"]) == 1
    assert payload["repeat_wall_seconds"][0] > 0.0
    assert payload["mean_wall_seconds"] > 0.0
    assert payload["ratings_per_second_mean"] > 0.0
    assert payload["state_checks"] == {
        "finite_parameters_after": True,
        "mutated_array_count": len(MUTATED_ARRAYS_BY_MODEL[model]),
    }
    assert payload["estimated_factor_touches"] > 0
