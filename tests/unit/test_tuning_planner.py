from __future__ import annotations

from recsys_lab.tuning import SearchSpaceSpec, build_study_plan, default_cluster_artifact_reuse_spec


def _payload() -> dict[str, object]:
    return {
        "search_space_version": "tuning_search_space_v1",
        "study": {
            "name": "planner_unit_study",
            "dataset": "ml100k",
            "split_family": "benchmark_random_v1",
            "model": "cb_svdpp",
            "seed": 1,
        },
        "base_model_config": "configs/models/cb_svdpp.yaml",
        "budget": {"max_candidates": 8},
        "generator": {"type": "grid", "deterministic_order": True},
        "search_space": {
            "alpha": {
                "type": "float",
                "values": [0.2, 0.8],
                "target_path": "clustering.alpha",
            },
            "learning_rate": {
                "type": "float",
                "values": [0.005, 0.01],
                "target_path": "training.learning_rate",
            },
        },
        "artifact_reuse": {
            "cluster_artifacts": default_cluster_artifact_reuse_spec().model_dump(mode="json")
        },
        "objective": {"primary": {"metric": "validation_rmse"}},
    }


def test_study_plan_has_deterministic_study_id() -> None:
    spec = SearchSpaceSpec.model_validate(_payload())

    first = build_study_plan(spec)
    second = build_study_plan(spec)

    assert first.study_id == second.study_id
    assert first.study_id.startswith("planner_unit_study_")


def test_study_plan_groups_reuse_safe_cb_candidates_together() -> None:
    plan = build_study_plan(SearchSpaceSpec.model_validate(_payload()))

    assert len(plan.candidates) == 4
    assert len(plan.artifact_reuse_groups) == 1
    assert plan.artifact_reuse_groups[0].candidate_ids == [
        candidate.candidate_id for candidate in plan.candidates
    ]


def test_study_plan_splits_cluster_reuse_groups_on_cluster_count() -> None:
    payload = _payload()
    payload["search_space"]["n_user_clusters"] = {  # type: ignore[index]
        "type": "int",
        "values": [8, 16],
        "target_path": "clustering.n_user_clusters",
    }

    plan = build_study_plan(SearchSpaceSpec.model_validate(payload))

    assert len(plan.artifact_reuse_groups) == 2
