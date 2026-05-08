import numpy as np

from recsys_lab.benchmarks.kernel_harness import (
    MUTATED_ARRAYS_BY_MODEL,
    assert_mutated_state_finite,
    clone_case_state,
    run_kernel_benchmark,
)
from recsys_lab.benchmarks.synthetic_kernel_cases import get_synthetic_kernel_case
from recsys_lab.models import kernels

CB_SVDPP_MUTATED_ARRAYS = (
    "user_bias",
    "item_bias",
    "user_factors",
    "item_factors",
    "implicit_factors",
    "user_cluster_factors",
    "item_cluster_factors",
    "implicit_cluster_factors",
)


def test_cb_svdpp_alpha0_specialized_matches_generic_all_mutated_arrays() -> None:
    generic_state, specialized_state = _run_generic_and_specialized_alpha0_states()

    assert generic_state.mutated_arrays == specialized_state.mutated_arrays == CB_SVDPP_MUTATED_ARRAYS
    assert_mutated_state_finite(generic_state)
    assert_mutated_state_finite(specialized_state)

    for name in CB_SVDPP_MUTATED_ARRAYS:
        np.testing.assert_allclose(
            specialized_state.arrays[name],
            generic_state.arrays[name],
            rtol=0.0,
            atol=0.0,
            err_msg=name,
        )


def test_cb_svdpp_alpha0_specialized_preserves_cluster_regularization() -> None:
    case = get_synthetic_kernel_case("cb_svdpp_alpha0")
    initial_arrays = case.clone_arrays()
    generic_state, specialized_state = _run_generic_and_specialized_alpha0_states()

    for name in ("user_cluster_factors", "item_cluster_factors", "implicit_cluster_factors"):
        assert not np.array_equal(specialized_state.arrays[name], initial_arrays[name]), name
        np.testing.assert_allclose(
            specialized_state.arrays[name],
            generic_state.arrays[name],
            rtol=0.0,
            atol=0.0,
            err_msg=name,
        )


def test_cb_svdpp_alpha0_specialized_does_not_change_shapes_or_dtypes() -> None:
    generic_state, specialized_state = _run_generic_and_specialized_alpha0_states()

    assert generic_state.mutated_arrays == specialized_state.mutated_arrays == CB_SVDPP_MUTATED_ARRAYS
    for name in CB_SVDPP_MUTATED_ARRAYS:
        assert specialized_state.arrays[name].shape == generic_state.arrays[name].shape, name
        assert specialized_state.arrays[name].dtype == generic_state.arrays[name].dtype, name
        assert np.all(np.isfinite(specialized_state.arrays[name])), name


def test_cb_svdpp_alpha0_specialized_case_has_nonzero_cluster_factors() -> None:
    case = get_synthetic_kernel_case("cb_svdpp_alpha0")
    state = clone_case_state(case)

    assert case.model == "cb_svdpp_alpha0"
    assert case.scalars["alpha"] == 0.0
    assert state.mutated_arrays == CB_SVDPP_MUTATED_ARRAYS
    assert MUTATED_ARRAYS_BY_MODEL["cb_svdpp"] == CB_SVDPP_MUTATED_ARRAYS
    assert MUTATED_ARRAYS_BY_MODEL["cb_svdpp_alpha0"] == CB_SVDPP_MUTATED_ARRAYS
    for name in ("user_cluster_factors", "item_cluster_factors", "implicit_cluster_factors"):
        assert case.arrays[name].dtype == np.float32
        assert np.any(case.arrays[name] != np.float32(0.0)), name


def test_cb_svdpp_alpha0_specialized_benchmark_payload_contract() -> None:
    case = get_synthetic_kernel_case("cb_svdpp_alpha0")
    payload = run_kernel_benchmark(case, warmup_repeats=1, timed_repeats=2)

    assert payload["benchmark_version"] == "kernel_benchmark_harness_v1"
    assert payload["case_name"] == "tiny_cb_svdpp_alpha0_float32"
    assert payload["model"] == "cb_svdpp_alpha0"
    assert payload["kernel_name"] == "train_cb_svdpp_alpha0_epoch_numba"
    assert payload["compile_excluded"] is True
    assert payload["state_copy_excluded"] is True
    assert len(payload["repeat_wall_seconds"]) == 2
    assert all(seconds > 0.0 for seconds in payload["repeat_wall_seconds"])
    assert payload["state_checks"]["finite_parameters_after"] is True
    assert payload["state_checks"]["mutated_array_count"] == len(CB_SVDPP_MUTATED_ARRAYS)


def _run_generic_and_specialized_alpha0_states():
    specialized_kernel = kernels.train_cb_svdpp_alpha0_epoch_numba
    assert callable(specialized_kernel)

    case = get_synthetic_kernel_case("cb_svdpp_alpha0")
    generic_state = clone_case_state(case)
    specialized_state = clone_case_state(case)

    _run_generic_cb_svdpp_alpha0_once(generic_state.arrays, generic_state.scalars)
    _run_specialized_cb_svdpp_alpha0_once(specialized_state.arrays, specialized_state.scalars, specialized_kernel)
    return generic_state, specialized_state


def _run_generic_cb_svdpp_alpha0_once(arrays: dict[str, np.ndarray], scalars: dict[str, float | int]) -> None:
    kernels.train_cb_svdpp_epoch_numba(
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


def _run_specialized_cb_svdpp_alpha0_once(
    arrays: dict[str, np.ndarray],
    scalars: dict[str, float | int],
    specialized_kernel: object,
) -> None:
    specialized_kernel(
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
