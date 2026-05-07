import csv
import json
import sys

import numpy as np
import pytest

from recsys_lab.benchmarks.kernel_harness import (
    KERNEL_DISPATCH,
    MUTATED_ARRAYS_BY_MODEL,
    assert_mutated_state_finite,
    clone_case_state,
    kernel_runner_for_model,
    run_kernel_benchmark,
)
from recsys_lab.benchmarks.synthetic_kernel_cases import (
    KERNEL_ARGUMENTS,
    build_synthetic_kernel_cases,
    get_synthetic_kernel_case,
)
from recsys_lab.benchmarks.writers import (
    SUMMARY_FIELDS,
    write_kernel_benchmark_json,
    write_kernel_benchmark_summary_csv,
)
from scripts.run_kernel_benchmarks import DEFAULT_OUTPUT_DIR, _parse_args, _select_cases

EXPECTED_MODELS = (
    "biased_mf",
    "svdpp",
    "asymmetric_svd",
    "asvdpp",
    "cb_svdpp",
    "cb_asvdpp",
)


def test_synthetic_cases_exist_for_all_six_models() -> None:
    cases = build_synthetic_kernel_cases()

    assert tuple(case.model for case in cases) == EXPECTED_MODELS
    for case in cases:
        assert case.name == f"tiny_{case.model}_float32"
        assert case.kernel_name.startswith("train_")
        assert case.dtype == "float32"
        assert case.latent_dim == 3
        assert case.train_rows == 16
        assert set(KERNEL_ARGUMENTS[case.model]) == set(case.arrays) | set(case.scalars)


def test_synthetic_cases_have_contiguous_arrays() -> None:
    for case in build_synthetic_kernel_cases():
        for name, array in case.arrays.items():
            assert array.flags.c_contiguous, name
            if name.endswith(("ids", "indptr", "items", "counts", "clusters")) or name == "order":
                assert array.dtype == np.int32, name
            else:
                assert array.dtype == np.float32, name


def test_synthetic_cases_have_finite_values() -> None:
    for case in build_synthetic_kernel_cases():
        for name, array in case.arrays.items():
            assert np.all(np.isfinite(array)), name

        assert np.array_equal(np.sort(case.arrays["order"]), np.arange(case.train_rows, dtype=np.int32))
        assert np.min(case.arrays["ratings"]) >= 1.0
        assert np.max(case.arrays["ratings"]) <= 5.0

        if "implicit_indptr" in case.arrays:
            assert np.all(np.diff(case.arrays["implicit_indptr"]) > 0)
        if "explicit_indptr" in case.arrays:
            assert np.all(np.diff(case.arrays["explicit_indptr"]) > 0)
        if "cluster_indptr" in case.arrays:
            assert np.all(np.diff(case.arrays["cluster_indptr"]) > 0)
            assert np.max(case.arrays["cluster_ids"]) < case.metadata["n_item_clusters"]
            assert np.max(case.arrays["user_clusters"]) < case.metadata["n_user_clusters"]
            assert np.max(case.arrays["item_clusters"]) < case.metadata["n_item_clusters"]


def test_synthetic_kernel_case_clone_arrays_returns_independent_state() -> None:
    case = get_synthetic_kernel_case("cb_asvdpp")

    cloned = case.clone_arrays()
    cloned["ratings"][0] = np.float32(1.25)

    assert case.arrays["ratings"][0] != cloned["ratings"][0]
    assert not np.shares_memory(case.arrays["ratings"], cloned["ratings"])


def test_clone_state_copies_mutated_arrays() -> None:
    case = get_synthetic_kernel_case("cb_asvdpp")

    state_a = clone_case_state(case)
    state_b = clone_case_state(case)

    assert state_a.mutated_arrays == MUTATED_ARRAYS_BY_MODEL[case.model]
    assert state_a.scalars == case.scalars
    assert state_a.scalars is not case.scalars

    for name, array in case.arrays.items():
        if name in state_a.mutated_arrays:
            assert np.array_equal(state_a.arrays[name], array), name
            assert np.array_equal(state_b.arrays[name], array), name
            assert not np.shares_memory(state_a.arrays[name], array), name
            assert not np.shares_memory(state_a.arrays[name], state_b.arrays[name]), name
        else:
            assert state_a.arrays[name] is array, name
            assert state_b.arrays[name] is array, name

    state_a.arrays["user_factors"][0, 0] = np.float32(42.0)

    assert case.arrays["user_factors"][0, 0] != state_a.arrays["user_factors"][0, 0]
    assert state_b.arrays["user_factors"][0, 0] != state_a.arrays["user_factors"][0, 0]


def test_assert_mutated_state_finite_rejects_non_finite_values() -> None:
    state = clone_case_state(get_synthetic_kernel_case("biased_mf"))
    state.arrays["user_bias"][0] = np.float32(np.nan)

    with pytest.raises(ValueError, match="non-finite"):
        assert_mutated_state_finite(state)


def test_kernel_dispatch_covers_all_synthetic_models() -> None:
    assert tuple(KERNEL_DISPATCH) == EXPECTED_MODELS
    for model, runner in KERNEL_DISPATCH.items():
        assert runner.__name__ == f"run_{model}_kernel_once"
        assert kernel_runner_for_model(model) is runner


def test_run_kernel_benchmark_payload_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    case = get_synthetic_kernel_case("biased_mf")
    user_bias_refs: list[np.ndarray] = []

    def fake_kernel(state: object) -> None:
        assert hasattr(state, "arrays")
        user_bias = state.arrays["user_bias"]
        user_bias_refs.append(user_bias)
        user_bias[0] += np.float32(0.5)

    monkeypatch.setitem(KERNEL_DISPATCH, "biased_mf", fake_kernel)

    result = run_kernel_benchmark(case, warmup_repeats=1, timed_repeats=2, epochs_per_repeat=3)

    assert result["case_name"] == case.name
    assert result["benchmark_version"] == "kernel_benchmark_harness_v1"
    assert result["benchmark_id"] == "tiny_biased_mf_float32_wr1_tr2_epr3"
    assert result["kernel_name"] == case.kernel_name
    assert result["model"] == "biased_mf"
    assert result["dataset_profile"] == "synthetic_tiny"
    assert result["compile_excluded"] is True
    assert result["state_copy_excluded"] is True
    assert result["warmup_excluded_from_timed"] is True
    assert len(result["warmup_wall_seconds"]) == 1
    assert len(result["repeat_wall_seconds"]) == 2
    assert all(seconds >= 0.0 for seconds in result["warmup_wall_seconds"])
    assert all(seconds >= 0.0 for seconds in result["repeat_wall_seconds"])
    assert result["mean_wall_seconds"] >= 0.0
    assert result["median_wall_seconds"] >= 0.0
    assert result["std_wall_seconds"] >= 0.0
    assert result["min_wall_seconds"] >= 0.0
    assert result["max_wall_seconds"] >= 0.0
    assert result["ratings_per_second_mean"] >= 0.0
    assert result["history_structure"] == {"implicit": {}, "explicit": {}, "cluster": {}}
    assert result["estimated_factor_touches"] == 576
    assert result["seconds_per_million_estimated_factor_touches"] >= 0.0
    assert result["state_checks"] == {
        "finite_parameters_after": True,
        "mutated_array_count": len(MUTATED_ARRAYS_BY_MODEL["biased_mf"]),
    }
    assert result["claim_boundary"] == "Diagnostic kernel benchmark only; no broad performance claim."
    assert len(user_bias_refs) == 9
    assert len({id(user_bias) for user_bias in user_bias_refs}) == 3
    assert case.arrays["user_bias"][0] == np.float32(0.02)


def test_run_kernel_benchmark_payload_includes_history_and_work(monkeypatch: pytest.MonkeyPatch) -> None:
    case = get_synthetic_kernel_case("cb_asvdpp")

    def fake_kernel(state: object) -> None:
        assert hasattr(state, "arrays")
        state.arrays["user_bias"][0] += np.float32(0.25)

    monkeypatch.setitem(KERNEL_DISPATCH, "cb_asvdpp", fake_kernel)

    result = run_kernel_benchmark(case, warmup_repeats=0, timed_repeats=1, epochs_per_repeat=2)

    assert result["history_structure"]["implicit"]["total_edges"] == 10
    assert result["history_structure"]["explicit"]["total_edges"] == 9
    assert result["history_structure"]["cluster"]["total_edges"] == 9
    assert result["estimated_kernel_work"]["implicit_history_visits_per_epoch"] == 40
    assert result["estimated_kernel_work"]["explicit_history_visits_per_epoch"] == 36
    assert result["estimated_kernel_work"]["cluster_history_visits_per_epoch"] == 36
    assert result["estimated_factor_touches"] == result["estimated_kernel_work"]["estimated_factor_touches"]
    assert result["estimated_factor_touches"] == 1728
    assert result["touches_note"]


def test_run_kernel_benchmark_rejects_invalid_repeat_counts() -> None:
    case = get_synthetic_kernel_case("biased_mf")

    with pytest.raises(ValueError, match="warmup_repeats"):
        run_kernel_benchmark(case, warmup_repeats=-1)
    with pytest.raises(ValueError, match="timed_repeats"):
        run_kernel_benchmark(case, timed_repeats=0)
    with pytest.raises(ValueError, match="epochs_per_repeat"):
        run_kernel_benchmark(case, epochs_per_repeat=0)


def test_run_kernel_benchmark_rejects_unknown_case() -> None:
    case = get_synthetic_kernel_case("biased_mf")
    unknown_case = type(case)(
        name="tiny_unknown_float32",
        model="unknown",
        kernel_name="train_unknown_epoch_numba",
        dtype=case.dtype,
        latent_dim=case.latent_dim,
        train_rows=case.train_rows,
        arrays=case.arrays,
        scalars=case.scalars,
        metadata=case.metadata,
    )

    with pytest.raises(KeyError):
        run_kernel_benchmark(unknown_case)


def test_write_kernel_benchmark_json_uses_benchmark_layout(tmp_path) -> None:
    payload = {
        "benchmark_id": "tiny_biased_mf_float32_wr1_tr1_epr1",
        "model": "biased_mf",
        "state_checks": {"finite_parameters_after": np.bool_(True), "mutated_array_count": np.int64(4)},
    }

    output_path = write_kernel_benchmark_json(payload, tmp_path / "artifacts" / "benchmarks" / "kernel")

    assert output_path == (
        tmp_path
        / "artifacts"
        / "benchmarks"
        / "kernel"
        / "tiny_biased_mf_float32_wr1_tr1_epr1"
        / "kernel_benchmark.json"
    )
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == {
        "benchmark_id": "tiny_biased_mf_float32_wr1_tr1_epr1",
        "model": "biased_mf",
        "state_checks": {"finite_parameters_after": True, "mutated_array_count": 4},
    }


def test_write_kernel_benchmark_summary_csv_uses_flat_aggregates(tmp_path) -> None:
    payloads = [
        {
            "benchmark_id": "bench-1",
            "kernel_name": "train_biased_mf_epoch_numba",
            "model": "biased_mf",
            "dataset_profile": "synthetic_tiny",
            "dtype": "float32",
            "latent_dim": 3,
            "train_rows": 16,
            "epochs_per_repeat": 1,
            "warmup_repeats": 1,
            "timed_repeats": 5,
            "mean_wall_seconds": 0.01,
            "median_wall_seconds": 0.01,
            "std_wall_seconds": 0.0,
            "min_wall_seconds": 0.01,
            "max_wall_seconds": 0.01,
            "ratings_per_second_mean": 1600.0,
            "estimated_factor_touches": 192,
            "seconds_per_million_estimated_factor_touches": 52.0,
            "state_checks": {"finite_parameters_after": True, "mutated_array_count": 4},
            "claim_boundary": "Diagnostic kernel benchmark only; no broad performance claim.",
            "repeat_wall_seconds": [0.01],
            "history_structure": {"implicit": {}},
        }
    ]

    output_path = write_kernel_benchmark_summary_csv(
        payloads,
        tmp_path / "artifacts" / "benchmarks" / "kernel" / "kernel_benchmark_summary.csv",
    )

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert tuple(rows[0]) == SUMMARY_FIELDS
    assert rows[0]["benchmark_id"] == "bench-1"
    assert rows[0]["finite_parameters_after"] == "True"
    assert rows[0]["mutated_array_count"] == "4"
    assert "repeat_wall_seconds" not in rows[0]
    assert "history_structure" not in rows[0]


def test_kernel_benchmark_writers_reject_numpy_arrays(tmp_path) -> None:
    payload = {"benchmark_id": "bench-with-array", "repeat_wall_seconds": np.asarray([0.1], dtype=np.float32)}

    with pytest.raises(TypeError, match="numpy arrays"):
        write_kernel_benchmark_json(payload, tmp_path)
    with pytest.raises(TypeError, match="numpy arrays"):
        write_kernel_benchmark_summary_csv([payload], tmp_path / "summary.csv")


def test_writer_outputs_json_and_csv(tmp_path) -> None:
    payload = {
        "benchmark_id": "bench-combined",
        "kernel_name": "train_biased_mf_epoch_numba",
        "model": "biased_mf",
        "dataset_profile": "synthetic_tiny",
        "dtype": "float32",
        "latent_dim": 3,
        "train_rows": 16,
        "epochs_per_repeat": 1,
        "warmup_repeats": 1,
        "timed_repeats": 2,
        "mean_wall_seconds": 0.01,
        "median_wall_seconds": 0.01,
        "std_wall_seconds": 0.0,
        "min_wall_seconds": 0.01,
        "max_wall_seconds": 0.01,
        "ratings_per_second_mean": 1600.0,
        "estimated_factor_touches": 192,
        "seconds_per_million_estimated_factor_touches": 52.0,
        "state_checks": {"finite_parameters_after": True, "mutated_array_count": 4},
        "claim_boundary": "Diagnostic kernel benchmark only; no broad performance claim.",
    }

    json_path = write_kernel_benchmark_json(payload, tmp_path)
    csv_path = write_kernel_benchmark_summary_csv([payload], tmp_path / "kernel_benchmark_summary.csv")

    assert json_path.exists()
    assert csv_path.exists()


def test_run_kernel_benchmarks_script_selects_cases() -> None:
    assert tuple(case.model for case in _select_cases("all")) == EXPECTED_MODELS
    assert tuple(case.model for case in _select_cases("svdpp")) == ("svdpp",)


def test_run_kernel_benchmarks_script_parser_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_kernel_benchmarks.py"])

    args = _parse_args()

    assert args.case == "all"
    assert args.timed_repeats == 5
    assert args.warmup_repeats == 1
    assert args.epochs_per_repeat == 1
    assert args.output_dir == DEFAULT_OUTPUT_DIR
