from __future__ import annotations

import csv
from pathlib import Path

import pytest

from recsys_lab.config.loader import load_yaml_file
from recsys_lab.tuning import (
    SearchSpaceSpec,
    build_study_plan,
    default_cluster_artifact_reuse_spec,
    generate_candidates,
)
from recsys_lab.tuning.manifests import build_candidate_manifests
from recsys_lab.tuning.writers import materialize_candidate_config


def _base_payload() -> dict[str, object]:
    return {
        "search_space_version": "tuning_search_space_v1",
        "study": {
            "name": "ml100k_cb_svdpp_cache_aware_mvp_v1",
            "dataset": "ml100k",
            "split_family": "benchmark_random_v1",
            "model": "cb_svdpp",
            "seed": 1,
        },
        "base_model_config": "configs/models/cb_svdpp.yaml",
        "budget": {"max_candidates": 3},
        "generator": {"type": "grid", "deterministic_order": True},
        "search_space": {
            "lambda_q": {
                "type": "float",
                "values": [0.01, 0.02],
                "target_path": "training.lambda_q",
            },
            "alpha": {
                "type": "float",
                "values": [0.2, 0.8],
                "target_path": "clustering.alpha",
            },
        },
        "artifact_reuse": {
            "cluster_artifacts": default_cluster_artifact_reuse_spec().model_dump(mode="json")
        },
        "objective": {"primary": {"metric": "validation_rmse"}},
    }


def test_grid_candidates_are_deterministic_and_budget_limited() -> None:
    spec = SearchSpaceSpec.model_validate(_base_payload())

    first = generate_candidates(spec, study_id="study")
    second = generate_candidates(spec, study_id="study")

    assert [candidate.candidate_id for candidate in first] == [candidate.candidate_id for candidate in second]
    assert [candidate.index for candidate in first] == [0, 1, 2]
    assert all(candidate.candidate_id.startswith(f"cand_{candidate.index:04d}_") for candidate in first)
    assert [candidate.parameter_values for candidate in first] == [
        {"alpha": 0.2, "lambda_q": 0.01},
        {"alpha": 0.2, "lambda_q": 0.02},
        {"alpha": 0.8, "lambda_q": 0.01},
    ]


def test_grid_candidate_generation_is_deterministic() -> None:
    spec = SearchSpaceSpec.model_validate(_base_payload())

    first = generate_candidates(spec, study_id="study")
    second = generate_candidates(spec, study_id="study")

    assert [candidate.parameter_values for candidate in first] == [
        candidate.parameter_values for candidate in second
    ]
    assert [candidate.candidate_id for candidate in first] == [candidate.candidate_id for candidate in second]


def test_grid_candidate_generation_respects_max_candidates() -> None:
    payload = _base_payload()
    payload["budget"] = {"max_candidates": 2}
    spec = SearchSpaceSpec.model_validate(payload)

    candidates = generate_candidates(spec, study_id="study")

    assert len(candidates) == 2
    assert [candidate.index for candidate in candidates] == [0, 1]


def test_candidate_ids_are_stable() -> None:
    spec = SearchSpaceSpec.model_validate(_base_payload())

    first = generate_candidates(spec, study_id="study")[0]
    second = generate_candidates(spec, study_id="study")[0]

    assert first.candidate_id == second.candidate_id
    assert first.candidate_id.startswith("cand_0000_")


def test_grid_candidate_contains_materialized_config_payload() -> None:
    spec = SearchSpaceSpec.model_validate(_base_payload())

    candidate = generate_candidates(spec, study_id="study")[0]

    assert candidate.base_model_config == "configs/models/cb_svdpp.yaml"
    assert candidate.objective_status == "planned"
    assert candidate.materialized_config_payload == {
        "base_model_config": "configs/models/cb_svdpp.yaml",
        "overrides": {
            "clustering": {"alpha": 0.2},
            "training": {"lambda_q": 0.01},
        },
    }


def test_manual_candidates_preserve_declared_order() -> None:
    payload = _base_payload()
    payload["generator"] = {"type": "manual", "deterministic_order": True}
    payload["manual_candidates"] = [
        {"alpha": 0.8, "lambda_q": 0.02},
        {"alpha": 0.2, "lambda_q": 0.01},
    ]
    spec = SearchSpaceSpec.model_validate(payload)

    candidates = generate_candidates(spec, study_id="study")

    assert [candidate.parameter_values for candidate in candidates] == [
        {"alpha": 0.8, "lambda_q": 0.02},
        {"alpha": 0.2, "lambda_q": 0.01},
    ]


def test_numeric_grid_without_values_is_deferred() -> None:
    payload = _base_payload()
    payload["search_space"]["alpha"] = {  # type: ignore[index]
        "type": "float",
        "distribution": "uniform",
        "low": 0.2,
        "high": 0.8,
        "target_path": "clustering.alpha",
    }
    spec = SearchSpaceSpec.model_validate(payload)

    with pytest.raises(ValueError, match="continuous linspace/logspace sampling is deferred"):
        generate_candidates(spec, study_id="study")


def test_cluster_reuse_group_ignores_reuse_safe_coordinates() -> None:
    spec = SearchSpaceSpec.model_validate(_base_payload())

    plan = build_study_plan(spec)

    assert len(plan.candidates) == 3
    assert len(plan.artifact_reuse_groups) == 1
    assert plan.artifact_reuse_groups[0].group_id.startswith("cluster_rg_")
    assert plan.artifact_reuse_groups[0].candidate_ids == [
        candidate.candidate_id for candidate in plan.candidates
    ]
    assert "alpha" in plan.artifact_reuse_groups[0].reuse_across
    assert "n_user_clusters" in plan.artifact_reuse_groups[0].invalidate_on


def test_cluster_reuse_group_changes_when_cluster_count_changes() -> None:
    payload = _base_payload()
    payload["budget"] = {"max_candidates": 4}
    payload["search_space"]["n_user_clusters"] = {  # type: ignore[index]
        "type": "int",
        "values": [8, 16],
        "target_path": "clustering.n_user_clusters",
    }
    spec = SearchSpaceSpec.model_validate(payload)

    plan = build_study_plan(spec)

    assert len(plan.artifact_reuse_groups) == 2


def test_cluster_reuse_group_id_is_stable_when_alpha_changes_only() -> None:
    alpha_low_payload = _base_payload()
    alpha_low_payload["search_space"] = {  # type: ignore[assignment]
        "alpha": {
            "type": "float",
            "values": [0.2],
            "target_path": "clustering.alpha",
        }
    }
    alpha_high_payload = _base_payload()
    alpha_high_payload["search_space"] = {  # type: ignore[assignment]
        "alpha": {
            "type": "float",
            "values": [0.8],
            "target_path": "clustering.alpha",
        }
    }

    low_group = build_study_plan(SearchSpaceSpec.model_validate(alpha_low_payload)).artifact_reuse_groups[0]
    high_group = build_study_plan(SearchSpaceSpec.model_validate(alpha_high_payload)).artifact_reuse_groups[0]

    assert low_group.group_id == high_group.group_id


def test_cb_candidates_share_cluster_reuse_group_when_only_alpha_changes() -> None:
    payload = _base_payload()
    payload["budget"] = {"max_candidates": 2}
    payload["search_space"] = {
        "alpha": {
            "type": "float",
            "values": [0.2, 0.8],
            "target_path": "clustering.alpha",
        }
    }
    plan = build_study_plan(SearchSpaceSpec.model_validate(payload))

    assert len(plan.artifact_reuse_groups) == 1
    assert plan.artifact_reuse_groups[0].candidate_ids == [
        candidate.candidate_id for candidate in plan.candidates
    ]


def test_cb_candidates_share_cluster_reuse_group_when_only_learning_rate_changes() -> None:
    payload = _base_payload()
    payload["budget"] = {"max_candidates": 2}
    payload["search_space"] = {
        "learning_rate": {
            "type": "float",
            "values": [0.005, 0.01],
            "target_path": "training.learning_rate",
        }
    }
    plan = build_study_plan(SearchSpaceSpec.model_validate(payload))

    assert len(plan.artifact_reuse_groups) == 1
    assert plan.artifact_reuse_groups[0].candidate_ids == [
        candidate.candidate_id for candidate in plan.candidates
    ]


def test_cb_candidates_share_cluster_reuse_group_when_only_target_lambda_q_changes() -> None:
    payload = _base_payload()
    payload["budget"] = {"max_candidates": 2}
    payload["search_space"] = {
        "target_lambda_q": {
            "type": "float",
            "values": [0.015, 0.04],
            "target_path": "training.lambda_q",
        }
    }
    plan = build_study_plan(SearchSpaceSpec.model_validate(payload))

    assert len(plan.artifact_reuse_groups) == 1
    assert plan.artifact_reuse_groups[0].candidate_ids == [
        candidate.candidate_id for candidate in plan.candidates
    ]


def test_cb_candidates_have_different_cluster_reuse_group_when_cluster_count_changes() -> None:
    payload = _base_payload()
    payload["budget"] = {"max_candidates": 2}
    payload["search_space"] = {
        "n_user_clusters": {
            "type": "int",
            "values": [8, 16],
            "target_path": "clustering.n_user_clusters",
        }
    }
    plan = build_study_plan(SearchSpaceSpec.model_validate(payload))

    assert len(plan.artifact_reuse_groups) == 2


def test_cb_candidates_have_different_cluster_reuse_group_when_induction_learning_rate_changes() -> None:
    payload = _base_payload()
    payload["budget"] = {"max_candidates": 2}
    payload["search_space"] = {
        "induction_learning_rate": {
            "type": "float",
            "values": [0.005, 0.01],
            "target_path": "clustering.induction.learning_rate",
        }
    }
    plan = build_study_plan(SearchSpaceSpec.model_validate(payload))

    assert len(plan.artifact_reuse_groups) == 2


def test_cb_candidates_have_different_cluster_reuse_group_when_induction_lambda_q_changes() -> None:
    payload = _base_payload()
    payload["budget"] = {"max_candidates": 2}
    payload["search_space"] = {
        "induction_lambda_q": {
            "type": "float",
            "values": [0.015, 0.04],
            "target_path": "clustering.induction.lambda_q",
        }
    }
    plan = build_study_plan(SearchSpaceSpec.model_validate(payload))

    assert len(plan.artifact_reuse_groups) == 2


def test_ml1m_small_study_target_only_variation_uses_one_cluster_reuse_group() -> None:
    payload = load_yaml_file(
        Path("configs/experiments/tuning/active/ml1m_cb_svdpp_small_study_v1.yaml")
    )
    plan = build_study_plan(SearchSpaceSpec.model_validate(payload))

    assert len(plan.candidates) == 12
    assert len(plan.artifact_reuse_groups) == 1


def test_materialize_candidate_config_applies_overrides_without_mutating_base() -> None:
    base_config = {
        "metadata": {"status": "draft", "owner": "tests", "purpose": "unit"},
        "model": {"name": "cb_svdpp", "family": "matrix_factorization", "scope": "extended"},
        "notes": [],
        "training": {
            "latent_dim": 4,
            "epochs": 1,
            "learning_rate": 0.01,
            "lambda_b": 0.01,
            "lambda_p": 0.01,
            "lambda_q": 0.01,
            "lambda_y": 0.01,
            "lambda_pC": 0.01,
            "lambda_qC": 0.01,
            "lambda_yC": 0.01,
            "init_std": 0.01,
            "dtype": "float32",
            "implicit_policy": "ratings_as_implicit",
        },
        "clustering": {
            "n_user_clusters": 2,
            "n_item_clusters": 2,
            "alpha": 0.5,
            "algorithm": "kmeans",
            "kmeans_n_init": 1,
        },
    }
    spec = SearchSpaceSpec.model_validate(_base_payload())
    manifest = build_candidate_manifests(build_study_plan(spec))[0]

    materialized = materialize_candidate_config(
        base_model_config_payload=base_config,
        candidate_manifest=manifest,
    )

    assert materialized["clustering"]["alpha"] == 0.2
    assert materialized["training"]["lambda_q"] == 0.01
    assert base_config["clustering"]["alpha"] == 0.5


def test_materialize_candidate_config_rejects_unknown_override_field() -> None:
    spec = SearchSpaceSpec.model_validate(_base_payload())
    manifest = build_candidate_manifests(build_study_plan(spec))[0].model_copy(
        update={"overrides": {"training": {"lambda_typo": 0.1}}}
    )

    with pytest.raises(ValueError, match="training.lambda_typo"):
        materialize_candidate_config(
            base_model_config_payload={"training": {"lambda_q": 0.01}},
            candidate_manifest=manifest,
        )


def test_study_manifest_contains_required_phase5_fields() -> None:
    from recsys_lab.tuning.manifests import build_study_manifest

    spec = SearchSpaceSpec.model_validate(_base_payload())
    manifest = build_study_manifest(build_study_plan(spec), created_at_utc="2026-05-08T000000Z")

    assert manifest.study_id
    assert manifest.study_name == "ml100k_cb_svdpp_cache_aware_mvp_v1"
    assert manifest.dataset == "ml100k"
    assert manifest.split_family == "benchmark_random_v1"
    assert manifest.model == "cb_svdpp"
    assert manifest.seed == 1
    assert manifest.base_model_config == "configs/models/cb_svdpp.yaml"
    assert manifest.budget.max_candidates == 3
    assert manifest.generator.type == "grid"
    assert manifest.schedule is None
    assert manifest.current_stage is None
    assert manifest.objective.primary.metric == "validation_rmse"
    assert manifest.candidate_count == 3
    assert manifest.artifact_reuse_contract is not None
    assert manifest.created_at_utc == "2026-05-08T000000Z"
    assert manifest.schema_version == "study_manifest_v1"
    assert "no performance or quality claim" in manifest.claim_boundary


def test_candidate_manifest_contains_required_phase5_fields() -> None:
    spec = SearchSpaceSpec.model_validate(_base_payload())
    manifest = build_candidate_manifests(build_study_plan(spec), output_dir="artifacts/tuning/study")[0]

    assert manifest.candidate_id.startswith("cand_0000_")
    assert manifest.candidate_index == 0
    assert manifest.study_id
    assert manifest.stage_name is None
    assert manifest.parameter_values == {"alpha": 0.2, "lambda_q": 0.01}
    assert manifest.base_model_config == "configs/models/cb_svdpp.yaml"
    assert manifest.candidate_config_path.endswith("/candidate_config.yaml")
    assert manifest.artifact_reuse_group_ids["cluster_artifacts"].startswith("cluster_rg_")
    assert manifest.objective_status == "planned"
    assert manifest.execution_status == "not_executed"
    assert "no performance or quality claim" in manifest.claim_boundary


def test_candidate_summary_csv_contains_required_phase5_columns(tmp_path) -> None:
    from recsys_lab.tuning.writers import write_candidate_summary_csv

    spec = SearchSpaceSpec.model_validate(_base_payload())
    plan = build_study_plan(spec)
    path = write_candidate_summary_csv(plan, tmp_path / "candidate_summary.csv")

    rows = list(csv.DictReader(path.open(encoding="utf-8")))

    assert rows
    assert set(rows[0]) == {
        "candidate_id",
        "candidate_index",
        "study_id",
        "model",
        "dataset",
        "split_family",
        "alpha",
        "learning_rate",
        "latent_dim",
        "epochs",
        "cluster_reuse_group_id",
        "candidate_config_path",
        "candidate_manifest_path",
        "status",
        "execution_status",
        "run_id",
        "run_dir",
        "run_manifest_path",
        "metrics_path",
        "performance_profile_path",
        "kernel_profile_path",
        "validation_rmse",
        "validation_mae",
        "fit_model_seconds",
        "total_wall_seconds",
        "cluster_cache_status",
        "user_cluster_history_cache_status",
        "cluster_total_seconds",
    }
    assert rows[0]["alpha"] == "0.2"
    assert rows[0]["learning_rate"] == ""
    assert rows[0]["cluster_reuse_group_id"].startswith("cluster_rg_")
    assert rows[0]["status"] == "planned"
    assert rows[0]["execution_status"] == "not_executed"
    assert rows[0]["validation_rmse"] == ""
