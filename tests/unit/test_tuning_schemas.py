from __future__ import annotations

import pytest
from pydantic import ValidationError

from recsys_lab.tuning.schemas import (
    DEFAULT_CLUSTER_ARTIFACT_REUSE_ACROSS,
    SearchSpaceSpec,
    default_cluster_artifact_reuse_spec,
)


def _valid_search_space_payload() -> dict[str, object]:
    return {
        "search_space_version": "tuning_search_space_v1",
        "study": {
            "name": "ml1m_cb_svdpp_alpha_regularization_v1",
            "dataset": "ml1m",
            "split_family": "benchmark_random_v1",
            "model": "cb_svdpp",
            "seed": 1,
        },
        "base_model_config": "configs/models/selected/ml1m/ml1m_cb_svdpp_stage0_transfer.yaml",
        "budget": {
            "max_candidates": 24,
            "max_parallel": 1,
            "max_wall_seconds": None,
        },
        "generator": {
            "type": "grid",
            "deterministic_order": True,
        },
        "search_space": {
            "alpha": {
                "type": "float",
                "distribution": "uniform",
                "low": 0.05,
                "high": 0.95,
            },
            "learning_rate": {
                "type": "float",
                "distribution": "loguniform",
                "low": 0.001,
                "high": 0.05,
            },
        },
        "artifact_reuse": {
            "cluster_artifacts": {
                "reuse_across": list(DEFAULT_CLUSTER_ARTIFACT_REUSE_ACROSS),
                "invalidate_on": [
                    "n_user_clusters",
                    "n_item_clusters",
                    "induction_config",
                    "dataset",
                    "split",
                    "train_fingerprint",
                ],
            }
        },
        "objective": {
            "primary": {
                "metric": "validation_rmse",
                "direction": "minimize",
                "aggregation": "mean",
            },
            "secondary": [
                {"metric": "validation_mae", "direction": "minimize", "aggregation": "mean"},
                {"metric": "fit_model_seconds", "direction": "minimize", "aggregation": "mean"},
            ],
            "required_guards": ["cluster_cache_status", "cluster_total_seconds"],
        },
    }


def test_tuning_study_schema_rejects_unknown_fields() -> None:
    payload = _valid_search_space_payload()
    payload["unknown_field"] = "not part of the contract"

    with pytest.raises(ValidationError):
        SearchSpaceSpec.model_validate(payload)


def test_cb_search_space_rejects_alpha_zero_for_productive_cb_model() -> None:
    payload = _valid_search_space_payload()
    payload["search_space"]["alpha"]["low"] = 0.0  # type: ignore[index]

    with pytest.raises(ValidationError, match="alpha"):
        SearchSpaceSpec.model_validate(payload)


def test_cb_search_space_rejects_alpha_one_for_productive_cb_model() -> None:
    payload = _valid_search_space_payload()
    payload["search_space"]["alpha"]["high"] = 1.0  # type: ignore[index]

    with pytest.raises(ValidationError, match="alpha"):
        SearchSpaceSpec.model_validate(payload)


def test_cb_search_space_accepts_strictly_between_zero_and_one() -> None:
    spec = SearchSpaceSpec.model_validate(_valid_search_space_payload())

    assert spec.study.model == "cb_svdpp"
    assert spec.search_space["alpha"].low == 0.05
    assert spec.search_space["alpha"].high == 0.95


def test_artifact_reuse_contract_documents_cluster_reuse_across_alpha() -> None:
    spec = default_cluster_artifact_reuse_spec()

    assert "alpha" in DEFAULT_CLUSTER_ARTIFACT_REUSE_ACROSS
    assert "alpha" in spec.reuse_across
    assert "n_user_clusters" in spec.invalidate_on
    assert "induction_config" in spec.invalidate_on


def test_objective_rejects_test_rmse_as_primary_tuning_metric() -> None:
    payload = _valid_search_space_payload()
    payload["objective"]["primary"]["metric"] = "test_rmse"  # type: ignore[index]

    with pytest.raises(ValidationError, match="test metrics"):
        SearchSpaceSpec.model_validate(payload)


def _manual_alpha_payload(
    *, model: str = "cb_svdpp", target_path: str = "alpha", value: float = 0.2
) -> dict[str, object]:
    payload = _valid_search_space_payload()
    payload["study"]["model"] = model  # type: ignore[index]
    payload["generator"] = {"type": "manual", "deterministic_order": True}
    payload["search_space"] = {
        "cb_alpha": {
            "type": "categorical",
            "values": [0.2, 0.5],
            "target_path": target_path,
        }
    }
    payload["manual_candidates"] = [{"cb_alpha": value}]
    return payload


def test_cb_manual_candidate_rejects_alpha_zero_via_target_path_alpha() -> None:
    with pytest.raises(ValidationError, match="alpha"):
        SearchSpaceSpec.model_validate(_manual_alpha_payload(target_path="alpha", value=0.0))


def test_cb_manual_candidate_rejects_alpha_one_via_target_path_clustering_alpha() -> None:
    with pytest.raises(ValidationError, match="alpha"):
        SearchSpaceSpec.model_validate(_manual_alpha_payload(target_path="clustering.alpha", value=1.0))


def test_cb_manual_candidate_accepts_alpha_between_zero_and_one_via_target_path() -> None:
    spec = SearchSpaceSpec.model_validate(_manual_alpha_payload(target_path="clustering.alpha", value=0.2))

    assert spec.manual_candidates == [{"cb_alpha": 0.2}]


def test_cb_manual_candidate_alpha_policy_applies_to_cb_asvdpp() -> None:
    with pytest.raises(ValidationError, match="alpha"):
        SearchSpaceSpec.model_validate(_manual_alpha_payload(model="cb_asvdpp", target_path="alpha", value=0.0))


def test_non_cb_manual_candidate_not_subject_to_cb_alpha_policy() -> None:
    spec = SearchSpaceSpec.model_validate(_manual_alpha_payload(model="svdpp", target_path="alpha", value=0.0))

    assert spec.study.model == "svdpp"
    assert spec.manual_candidates == [{"cb_alpha": 0.0}]


def test_random_generator_requires_candidate_count() -> None:
    payload = _valid_search_space_payload()
    payload["generator"] = {"type": "random", "deterministic_order": True, "seed": 7}

    with pytest.raises(ValidationError, match="n_candidates"):
        SearchSpaceSpec.model_validate(payload)


def test_random_generator_accepts_seed_and_candidate_count() -> None:
    payload = _valid_search_space_payload()
    payload["generator"] = {
        "type": "random",
        "deterministic_order": True,
        "seed": 7,
        "n_candidates": 8,
    }

    spec = SearchSpaceSpec.model_validate(payload)

    assert spec.generator.type == "random"
    assert spec.generator.seed == 7
    assert spec.generator.n_candidates == 8


def test_latin_hypercube_generator_accepts_seed_and_candidate_count() -> None:
    payload = _valid_search_space_payload()
    payload["generator"] = {
        "type": "latin_hypercube",
        "deterministic_order": True,
        "seed": 11,
        "n_candidates": 8,
    }

    spec = SearchSpaceSpec.model_validate(payload)

    assert spec.generator.type == "latin_hypercube"
    assert spec.generator.seed == 11
    assert spec.generator.n_candidates == 8


def test_sampling_generator_candidate_count_must_not_exceed_budget() -> None:
    payload = _valid_search_space_payload()
    payload["budget"]["max_candidates"] = 4  # type: ignore[index]
    payload["generator"] = {
        "type": "latin_hypercube",
        "deterministic_order": True,
        "seed": 11,
        "n_candidates": 8,
    }

    with pytest.raises(ValidationError, match="budget.max_candidates"):
        SearchSpaceSpec.model_validate(payload)


def test_grid_generator_rejects_candidate_count() -> None:
    payload = _valid_search_space_payload()
    payload["generator"] = {
        "type": "grid",
        "deterministic_order": True,
        "n_candidates": 8,
    }

    with pytest.raises(ValidationError, match="n_candidates"):
        SearchSpaceSpec.model_validate(payload)


def test_schedule_accepts_fidelity_stages() -> None:
    payload = _valid_search_space_payload()
    payload["schedule"] = {
        "stages": [
            {
                "name": "stage1_low_fidelity",
                "max_candidates": 32,
                "promote_top_k": 8,
                "overrides": {"training.epochs": 3},
            },
            {
                "name": "stage2_mid_fidelity",
                "max_candidates": 8,
                "promote_top_k": 3,
                "overrides": {"training.epochs": 10},
            },
            {
                "name": "stage3_full_fidelity",
                "max_candidates": 3,
                "promote_top_k": 1,
                "overrides": {"training.epochs": 20},
            },
        ]
    }

    spec = SearchSpaceSpec.model_validate(payload)

    assert [stage.name for stage in spec.schedule.stages] == [  # type: ignore[union-attr]
        "stage1_low_fidelity",
        "stage2_mid_fidelity",
        "stage3_full_fidelity",
    ]
    assert spec.schedule.stages[0].objective_metric == "validation_rmse"  # type: ignore[union-attr]
    assert spec.schedule.stages[0].tie_breakers == ["validation_mae", "fit_model_seconds"]  # type: ignore[union-attr]


def test_schedule_rejects_duplicate_stage_names() -> None:
    payload = _valid_search_space_payload()
    payload["schedule"] = {
        "stages": [
            {"name": "stage1", "max_candidates": 4, "overrides": {"training.epochs": 3}},
            {"name": "stage1", "max_candidates": 2, "overrides": {"training.epochs": 10}},
        ]
    }

    with pytest.raises(ValidationError, match="unique"):
        SearchSpaceSpec.model_validate(payload)


def test_schedule_rejects_test_metrics() -> None:
    payload = _valid_search_space_payload()
    payload["schedule"] = {
        "stages": [
            {
                "name": "stage1",
                "max_candidates": 4,
                "overrides": {"training.epochs": 3},
                "objective_metric": "test_rmse",
            }
        ]
    }

    with pytest.raises(ValidationError, match="test metrics"):
        SearchSpaceSpec.model_validate(payload)


def test_schedule_allows_induction_stage_override_as_outer_fidelity_coordinate() -> None:
    payload = _valid_search_space_payload()
    payload["schedule"] = {
        "stages": [
            {
                "name": "stage1_induction_low_fidelity",
                "max_candidates": 4,
                "promote_top_k": 2,
                "overrides": {
                    "training.epochs": 3,
                    "clustering.induction.epochs": 3,
                },
            }
        ]
    }

    spec = SearchSpaceSpec.model_validate(payload)

    assert spec.schedule.stages[0].overrides["training.epochs"] == 3  # type: ignore[union-attr]
    assert spec.schedule.stages[0].overrides["clustering.induction.epochs"] == 3  # type: ignore[union-attr]
