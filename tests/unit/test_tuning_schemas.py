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
