from __future__ import annotations

from recsys_lab.tuning import SearchRole, classify_search_coordinate
from recsys_lab.tuning.planner import build_study_plan
from recsys_lab.tuning.schemas import SearchSpaceSpec, default_cluster_artifact_reuse_spec


def _payload_for_dimension(name: str, target_path: str, values: list[float | int | str]) -> dict:
    return {
        "search_space_version": "tuning_search_space_v1",
        "study": {
            "name": "ml1m_cb_svdpp_role_contract_v1",
            "dataset": "ml1m",
            "split_family": "benchmark_random_v1",
            "model": "cb_svdpp",
            "seed": 1,
        },
        "base_model_config": "configs/models/selected/ml1m/ml1m_cb_svdpp_fidelity_promotion_selected.yaml",
        "budget": {"max_candidates": len(values)},
        "generator": {"type": "grid", "deterministic_order": True},
        "search_space": {
            name: {
                "type": "categorical" if isinstance(values[0], str) else "float",
                "values": values,
                "target_path": target_path,
            }
        },
        "artifact_reuse": {
            "cluster_artifacts": default_cluster_artifact_reuse_spec().model_dump(mode="json")
        },
        "objective": {"primary": {"metric": "validation_rmse"}},
    }


def test_search_role_classification_contract() -> None:
    assert classify_search_coordinate("training.learning_rate") == SearchRole.INNER_TARGET_PARAM
    assert classify_search_coordinate("training.lambda_q") == SearchRole.INNER_TARGET_PARAM
    assert classify_search_coordinate("training.epochs") == SearchRole.INNER_TARGET_PARAM
    assert classify_search_coordinate("clustering.alpha") == SearchRole.INNER_TARGET_PARAM
    assert classify_search_coordinate("clustering.n_user_clusters") == SearchRole.OUTER_CLUSTER_PARAM
    assert classify_search_coordinate("clustering.induction.learning_rate") == SearchRole.OUTER_CLUSTER_PARAM
    assert classify_search_coordinate("clustering.induction.epochs") == SearchRole.OUTER_CLUSTER_PARAM


def test_training_learning_rate_is_inner_param() -> None:
    assert classify_search_coordinate("training.learning_rate") == SearchRole.INNER_TARGET_PARAM


def test_training_epochs_is_inner_param() -> None:
    assert classify_search_coordinate("training.epochs") == SearchRole.INNER_TARGET_PARAM


def test_clustering_induction_learning_rate_is_outer_param() -> None:
    assert classify_search_coordinate("clustering.induction.learning_rate") == SearchRole.OUTER_CLUSTER_PARAM


def test_clustering_induction_epochs_is_outer_param() -> None:
    assert classify_search_coordinate("clustering.induction.epochs") == SearchRole.OUTER_CLUSTER_PARAM


def test_training_learning_rate_does_not_change_reuse_group() -> None:
    plan = build_study_plan(
        SearchSpaceSpec.model_validate(
            _payload_for_dimension("learning_rate", "training.learning_rate", [0.0075, 0.01])
        )
    )

    assert len(plan.artifact_reuse_groups) == 1


def test_inner_param_changes_do_not_change_cluster_reuse_group() -> None:
    plan = build_study_plan(
        SearchSpaceSpec.model_validate(
            _payload_for_dimension("learning_rate", "training.learning_rate", [0.0075, 0.01])
        )
    )

    assert len(plan.artifact_reuse_groups) == 1


def test_clustering_induction_learning_rate_changes_reuse_group() -> None:
    plan = build_study_plan(
        SearchSpaceSpec.model_validate(
            _payload_for_dimension(
                "induction_learning_rate",
                "clustering.induction.learning_rate",
                [0.0075, 0.01],
            )
        )
    )

    assert len(plan.artifact_reuse_groups) == 2


def test_outer_param_changes_change_cluster_reuse_group() -> None:
    plan = build_study_plan(
        SearchSpaceSpec.model_validate(
            _payload_for_dimension(
                "induction_learning_rate",
                "clustering.induction.learning_rate",
                [0.0075, 0.01],
            )
        )
    )

    assert len(plan.artifact_reuse_groups) == 2


def test_training_epochs_does_not_change_reuse_group() -> None:
    plan = build_study_plan(
        SearchSpaceSpec.model_validate(_payload_for_dimension("epochs", "training.epochs", [3, 20]))
    )

    assert len(plan.artifact_reuse_groups) == 1


def test_clustering_induction_epochs_changes_reuse_group() -> None:
    plan = build_study_plan(
        SearchSpaceSpec.model_validate(
            _payload_for_dimension("induction_epochs", "clustering.induction.epochs", [3, 20])
        )
    )

    assert len(plan.artifact_reuse_groups) == 2


def test_stage_training_epochs_override_does_not_change_reuse_key() -> None:
    payload = _payload_for_dimension("alpha", "clustering.alpha", [0.2, 0.3])
    payload["schedule"] = {
        "stages": [
            {
                "name": "stage1_low_fidelity",
                "max_candidates": 2,
                "overrides": {"training.epochs": 3},
            }
        ]
    }
    plan = build_study_plan(SearchSpaceSpec.model_validate(payload))

    assert plan.artifact_reuse_groups[0].reuse_key["retained_stage_overrides"] == {}


def test_stage_induction_epochs_override_changes_reuse_key() -> None:
    payload = _payload_for_dimension("alpha", "clustering.alpha", [0.2, 0.3])
    payload["schedule"] = {
        "stages": [
            {
                "name": "stage1_low_fidelity",
                "max_candidates": 2,
                "overrides": {"clustering.induction.epochs": 3},
            }
        ]
    }
    plan = build_study_plan(SearchSpaceSpec.model_validate(payload))

    assert plan.artifact_reuse_groups[0].reuse_key["retained_stage_overrides"] == {
        "stage1_low_fidelity": {"clustering.induction.epochs": 3}
    }
