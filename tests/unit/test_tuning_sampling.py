from __future__ import annotations

import random

import pytest

from recsys_lab.tuning import SearchSpaceSpec, generate_candidates
from recsys_lab.tuning.sampling import generate_latin_hypercube_candidates, generate_random_candidates


def _sampling_payload(*, generator_type: str = "random", seed: int = 7, n_candidates: int = 6) -> dict:
    return {
        "search_space_version": "tuning_search_space_v1",
        "study": {
            "name": "ml1m_cb_svdpp_sota_sampling_v1",
            "dataset": "ml1m",
            "split_family": "benchmark_random_v1",
            "model": "cb_svdpp",
            "seed": 1,
        },
        "base_model_config": "configs/models/selected/ml1m/ml1m_cb_svdpp_fidelity_promotion_selected.yaml",
        "budget": {"max_candidates": n_candidates},
        "generator": {
            "type": generator_type,
            "deterministic_order": True,
            "seed": seed,
            "n_candidates": n_candidates,
        },
        "search_space": {
            "alpha": {
                "type": "float",
                "distribution": "uniform",
                "low": 0.05,
                "high": 0.95,
                "target_path": "clustering.alpha",
            },
            "learning_rate": {
                "type": "float",
                "distribution": "loguniform",
                "low": 0.001,
                "high": 0.05,
                "target_path": "training.learning_rate",
            },
            "epochs": {
                "type": "int",
                "distribution": "uniform",
                "low": 3,
                "high": 20,
                "target_path": "training.epochs",
            },
            "lambda_q": {
                "type": "categorical",
                "values": [0.015, 0.025, 0.04],
                "target_path": "training.lambda_q",
            },
        },
        "objective": {"primary": {"metric": "validation_rmse"}},
    }


def test_random_sampler_is_seed_deterministic() -> None:
    spec = SearchSpaceSpec.model_validate(_sampling_payload(generator_type="random", seed=17))

    first = generate_random_candidates(spec, study_id="study", stage_name="stage1")
    second = generate_random_candidates(spec, study_id="study", stage_name="stage1")

    assert [candidate.parameter_values for candidate in first] == [
        candidate.parameter_values for candidate in second
    ]
    assert [candidate.candidate_id for candidate in first] == [candidate.candidate_id for candidate in second]


def test_random_sampler_seed_changes_candidate_values() -> None:
    first = generate_random_candidates(
        SearchSpaceSpec.model_validate(_sampling_payload(generator_type="random", seed=17)),
        study_id="study",
    )
    second = generate_random_candidates(
        SearchSpaceSpec.model_validate(_sampling_payload(generator_type="random", seed=18)),
        study_id="study",
    )

    assert [candidate.parameter_values for candidate in first] != [
        candidate.parameter_values for candidate in second
    ]


def test_random_sampler_respects_dimension_bounds_and_types() -> None:
    spec = SearchSpaceSpec.model_validate(_sampling_payload(generator_type="random", seed=17))

    candidates = generate_candidates(spec, study_id="study")

    assert len(candidates) == 6
    for candidate in candidates:
        values = candidate.parameter_values
        assert 0.05 < values["alpha"] < 0.95
        assert 0.001 <= values["learning_rate"] <= 0.05
        assert isinstance(values["epochs"], int)
        assert 3 <= values["epochs"] <= 20
        assert values["lambda_q"] in {0.015, 0.025, 0.04}


def test_loguniform_samples_are_positive_and_in_bounds() -> None:
    spec = SearchSpaceSpec.model_validate(_sampling_payload(generator_type="random", seed=17))

    candidates = generate_candidates(spec, study_id="study")

    for candidate in candidates:
        learning_rate = candidate.parameter_values["learning_rate"]
        assert learning_rate > 0.0
        assert 0.001 <= learning_rate <= 0.05


def test_uniform_samples_are_in_bounds() -> None:
    spec = SearchSpaceSpec.model_validate(_sampling_payload(generator_type="random", seed=17))

    candidates = generate_candidates(spec, study_id="study")

    for candidate in candidates:
        assert 0.05 < candidate.parameter_values["alpha"] < 0.95


def test_latin_hypercube_sampler_is_seed_deterministic() -> None:
    spec = SearchSpaceSpec.model_validate(_sampling_payload(generator_type="latin_hypercube", seed=23))

    first = generate_latin_hypercube_candidates(spec, study_id="study", stage_name="stage1")
    second = generate_latin_hypercube_candidates(spec, study_id="study", stage_name="stage1")

    assert [candidate.parameter_values for candidate in first] == [
        candidate.parameter_values for candidate in second
    ]
    assert [candidate.candidate_id for candidate in first] == [candidate.candidate_id for candidate in second]


def test_latin_hypercube_numeric_values_cover_strata() -> None:
    spec = SearchSpaceSpec.model_validate(_sampling_payload(generator_type="latin_hypercube", seed=23, n_candidates=6))

    candidates = generate_latin_hypercube_candidates(spec, study_id="study")
    alpha_values = sorted(candidate.parameter_values["alpha"] for candidate in candidates)

    assert len(alpha_values) == 6
    for index, value in enumerate(alpha_values):
        assert 0.05 + index * 0.15 <= value <= 0.05 + (index + 1) * 0.15


def test_sampling_does_not_mutate_global_random_state() -> None:
    random.seed(123)
    before = random.random()
    spec = SearchSpaceSpec.model_validate(_sampling_payload(generator_type="random", seed=17))

    generate_random_candidates(spec, study_id="study")
    after = random.random()

    random.seed(123)
    assert before == random.random()
    assert after == random.random()


def test_candidate_ids_include_stage_name_when_provided() -> None:
    spec = SearchSpaceSpec.model_validate(_sampling_payload(generator_type="random", seed=17))

    stage1 = generate_random_candidates(spec, study_id="study", stage_name="stage1")[0]
    stage2 = generate_random_candidates(spec, study_id="study", stage_name="stage2")[0]

    assert stage1.parameter_values == stage2.parameter_values
    assert stage1.candidate_id != stage2.candidate_id


def test_categorical_dimension_sampling_is_deterministic() -> None:
    spec = SearchSpaceSpec.model_validate(_sampling_payload(generator_type="random", seed=17))

    first = generate_candidates(spec, study_id="study")
    second = generate_candidates(spec, study_id="study")

    assert [candidate.parameter_values["lambda_q"] for candidate in first] == [
        candidate.parameter_values["lambda_q"] for candidate in second
    ]


def test_candidate_ids_are_stable() -> None:
    spec = SearchSpaceSpec.model_validate(_sampling_payload(generator_type="latin_hypercube", seed=17))

    first = generate_candidates(spec, study_id="study")
    second = generate_candidates(spec, study_id="study")

    assert [candidate.candidate_id for candidate in first] == [candidate.candidate_id for candidate in second]


def test_productive_cb_alpha_endpoints_are_rejected_before_sampling() -> None:
    payload = _sampling_payload(generator_type="random", seed=17)
    payload["search_space"]["alpha"]["low"] = 0.0

    with pytest.raises(ValueError, match="alpha"):
        SearchSpaceSpec.model_validate(payload)


def test_alpha_samples_exclude_endpoints_for_cb_models() -> None:
    spec = SearchSpaceSpec.model_validate(_sampling_payload(generator_type="latin_hypercube", seed=23))

    candidates = generate_candidates(spec, study_id="study")

    assert all(0.0 < candidate.parameter_values["alpha"] < 1.0 for candidate in candidates)
