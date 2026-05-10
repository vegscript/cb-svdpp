from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from recsys_lab.config.loader import dump_yaml_file, load_yaml_file
from recsys_lab.tuning import (
    FidelityStageSpec,
    SearchSpaceSpec,
    build_promotion_plan,
    default_cluster_artifact_reuse_spec,
    materialize_promoted_candidates,
    materialize_stage_candidates,
    plan_stage_1_candidates,
)


def _base_model_config() -> dict[str, object]:
    return {
        "metadata": {"status": "unit", "owner": "tests", "purpose": "model_profile"},
        "model": {"name": "cb_svdpp", "family": "matrix_factorization", "scope": "extended"},
        "training": {
            "latent_dim": 64,
            "epochs": 3,
            "learning_rate": 0.0075,
            "lambda_b": 0.01,
            "lambda_p": 0.01,
            "lambda_q": 0.025,
            "lambda_y": 0.01,
            "lambda_pC": 0.01,
            "lambda_qC": 0.01,
            "lambda_yC": 0.01,
            "init_std": 0.1,
            "dtype": "float32",
            "implicit_policy": "ratings_as_implicit",
        },
        "clustering": {
            "alpha": 0.2,
            "n_user_clusters": 80,
            "n_item_clusters": 80,
            "algorithm": "kmeans",
            "kmeans_n_init": 10,
            "induction": {
                "latent_dim": 64,
                "epochs": 20,
                "learning_rate": 0.0075,
                "lambda_b": 0.025,
                "lambda_p": 0.025,
                "lambda_q": 0.025,
                "init_std": 0.1,
                "dtype": "float32",
                "seed": 1,
            },
        },
        "notes": [],
    }


def _search_space_payload(base_config: str) -> dict[str, object]:
    return {
        "search_space_version": "tuning_search_space_v1",
        "study": {
            "name": "staged_planner_unit",
            "dataset": "ml1m",
            "split_family": "benchmark_random_v1",
            "model": "cb_svdpp",
            "seed": 1,
        },
        "base_model_config": base_config,
        "budget": {"max_candidates": 4},
        "generator": {"type": "grid", "deterministic_order": True},
        "search_space": {
            "alpha": {
                "type": "float",
                "values": [0.1, 0.2, 0.3],
                "target_path": "clustering.alpha",
            },
            "learning_rate": {
                "type": "float",
                "values": [0.0075, 0.01],
                "target_path": "training.learning_rate",
            },
        },
        "artifact_reuse": {
            "cluster_artifacts": default_cluster_artifact_reuse_spec().model_dump(mode="json"),
        },
        "objective": {"primary": {"metric": "validation_rmse"}},
        "schedule": {
            "stages": [
                {
                    "name": "stage1_low_fidelity",
                    "max_candidates": 2,
                    "promote_top_k": 1,
                    "overrides": {"training.epochs": 3},
                },
                {
                    "name": "stage2_mid_fidelity",
                    "max_candidates": 1,
                    "promote_top_k": None,
                    "overrides": {"training.epochs": 10},
                },
            ],
        },
    }


def _write_base_config(tmp_path: Path) -> Path:
    path = tmp_path / "base_cb_svdpp.yaml"
    dump_yaml_file(path, _base_model_config())
    return path


def test_plan_stage_1_candidates_uses_first_stage_budget_and_stage_ids(tmp_path: Path) -> None:
    base_config_path = _write_base_config(tmp_path)
    spec = SearchSpaceSpec.model_validate(_search_space_payload(str(base_config_path)))

    plan = plan_stage_1_candidates(spec)

    assert len(plan.candidates) == 2
    assert {candidate.stage_name for candidate in plan.candidates} == {"stage1_low_fidelity"}
    assert all(candidate.candidate_id.startswith(f"cand_{candidate.index:04d}_") for candidate in plan.candidates)


def test_stage1_materializes_requested_candidate_count(tmp_path: Path) -> None:
    base_config_path = _write_base_config(tmp_path)
    spec = SearchSpaceSpec.model_validate(_search_space_payload(str(base_config_path)))

    plan = plan_stage_1_candidates(spec)

    assert len(plan.candidates) == spec.schedule.stages[0].max_candidates


def test_materialize_stage_candidates_applies_stage_overrides_without_changing_induction(
    tmp_path: Path,
) -> None:
    base_config_path = _write_base_config(tmp_path)
    spec = SearchSpaceSpec.model_validate(_search_space_payload(str(base_config_path)))
    stage = spec.schedule.stages[0] if spec.schedule is not None else None
    plan = plan_stage_1_candidates(spec)

    paths = materialize_stage_candidates(plan, tmp_path / "study", stage=stage)
    candidate_config_paths = [
        path for key, path in paths.items() if key.startswith("candidate_config:")
    ]
    materialized = load_yaml_file(candidate_config_paths[0])

    assert materialized["training"]["epochs"] == 3
    assert materialized["clustering"]["induction"]["epochs"] == 20
    assert paths["candidate_summary"].exists()


def test_stage_overrides_are_applied_to_candidate_configs(tmp_path: Path) -> None:
    base_config_path = _write_base_config(tmp_path)
    spec = SearchSpaceSpec.model_validate(_search_space_payload(str(base_config_path)))
    stage = spec.schedule.stages[1] if spec.schedule is not None else None
    plan = plan_stage_1_candidates(spec)

    paths = materialize_stage_candidates(plan, tmp_path / "study", stage=stage)
    materialized = load_yaml_file(next(path for key, path in paths.items() if key.startswith("candidate_config:")))

    assert materialized["training"]["epochs"] == 10


def test_materialize_stage_candidates_rejects_unknown_stage_override(tmp_path: Path) -> None:
    base_config_path = _write_base_config(tmp_path)
    payload = _search_space_payload(str(base_config_path))
    payload["schedule"]["stages"][0]["overrides"] = {"training.epoch_typo": 3}  # type: ignore[index]
    spec = SearchSpaceSpec.model_validate(payload)
    stage = spec.schedule.stages[0] if spec.schedule is not None else None

    with pytest.raises(ValueError, match="training.epoch_typo"):
        materialize_stage_candidates(
            plan_stage_1_candidates(spec),
            tmp_path / "study",
            stage=stage,
        )


def test_build_promotion_plan_selects_succeeded_by_rmse_mae_fit_time(tmp_path: Path) -> None:
    result_rows = [
        {
            "candidate_id": "failed_candidate",
            "execution_status": "failed",
            "validation_rmse": "0.800",
            "validation_mae": "0.600",
            "fit_model_seconds": "10.0",
            "candidate_config_path": str(tmp_path / "failed.yaml"),
        },
        {
            "candidate_id": "best_by_mae",
            "execution_status": "succeeded",
            "validation_rmse": "0.900",
            "validation_mae": "0.650",
            "fit_model_seconds": "20.0",
            "candidate_config_path": str(tmp_path / "best_by_mae.yaml"),
        },
        {
            "candidate_id": "best_by_fit_time",
            "execution_status": "succeeded",
            "validation_rmse": "0.900",
            "validation_mae": "0.650",
            "fit_model_seconds": "12.0",
            "candidate_config_path": str(tmp_path / "best_by_fit_time.yaml"),
        },
        {
            "candidate_id": "worse_rmse",
            "execution_status": "succeeded",
            "validation_rmse": "0.910",
            "validation_mae": "0.640",
            "fit_model_seconds": "5.0",
            "candidate_config_path": str(tmp_path / "worse_rmse.yaml"),
        },
    ]
    next_stage = FidelityStageSpec(
        name="stage2_mid_fidelity",
        max_candidates=2,
        overrides={"training.epochs": 10},
    )

    plan = build_promotion_plan(result_rows, next_stage)

    assert [candidate.source_candidate_id for candidate in plan.promoted_candidates] == [
        "best_by_fit_time",
        "best_by_mae",
    ]
    assert all(candidate.promoted_candidate_id.startswith("prom_") for candidate in plan.promoted_candidates)


def test_promotion_selects_top_k_by_validation_rmse(tmp_path: Path) -> None:
    rows = [
        {
            "candidate_id": "candidate_a",
            "execution_status": "succeeded",
            "validation_rmse": "0.910",
            "validation_mae": "0.700",
            "fit_model_seconds": "10.0",
            "candidate_config_path": str(tmp_path / "candidate_a.yaml"),
        },
        {
            "candidate_id": "candidate_b",
            "execution_status": "succeeded",
            "validation_rmse": "0.900",
            "validation_mae": "0.710",
            "fit_model_seconds": "20.0",
            "candidate_config_path": str(tmp_path / "candidate_b.yaml"),
        },
    ]

    plan = build_promotion_plan(rows, FidelityStageSpec(name="stage2_mid_fidelity", max_candidates=1))

    assert [candidate.source_candidate_id for candidate in plan.promoted_candidates] == ["candidate_b"]


def test_promotion_uses_validation_mae_tiebreaker(tmp_path: Path) -> None:
    rows = [
        {
            "candidate_id": "higher_mae",
            "execution_status": "succeeded",
            "validation_rmse": "0.900",
            "validation_mae": "0.720",
            "fit_model_seconds": "5.0",
            "candidate_config_path": str(tmp_path / "higher_mae.yaml"),
        },
        {
            "candidate_id": "lower_mae",
            "execution_status": "succeeded",
            "validation_rmse": "0.900",
            "validation_mae": "0.710",
            "fit_model_seconds": "20.0",
            "candidate_config_path": str(tmp_path / "lower_mae.yaml"),
        },
    ]

    plan = build_promotion_plan(rows, FidelityStageSpec(name="stage2_mid_fidelity", max_candidates=1))

    assert [candidate.source_candidate_id for candidate in plan.promoted_candidates] == ["lower_mae"]


def test_promotion_uses_fit_time_second_tiebreaker(tmp_path: Path) -> None:
    rows = [
        {
            "candidate_id": "slow",
            "execution_status": "succeeded",
            "validation_rmse": "0.900",
            "validation_mae": "0.710",
            "fit_model_seconds": "20.0",
            "candidate_config_path": str(tmp_path / "slow.yaml"),
        },
        {
            "candidate_id": "fast",
            "execution_status": "succeeded",
            "validation_rmse": "0.900",
            "validation_mae": "0.710",
            "fit_model_seconds": "5.0",
            "candidate_config_path": str(tmp_path / "fast.yaml"),
        },
    ]

    plan = build_promotion_plan(rows, FidelityStageSpec(name="stage2_mid_fidelity", max_candidates=1))

    assert [candidate.source_candidate_id for candidate in plan.promoted_candidates] == ["fast"]


def test_promotion_ignores_failed_candidates(tmp_path: Path) -> None:
    rows = [
        {
            "candidate_id": "failed_best_metric",
            "execution_status": "failed",
            "validation_rmse": "0.100",
            "validation_mae": "0.100",
            "fit_model_seconds": "1.0",
            "candidate_config_path": str(tmp_path / "failed.yaml"),
        },
        {
            "candidate_id": "succeeded",
            "execution_status": "succeeded",
            "validation_rmse": "0.900",
            "validation_mae": "0.710",
            "fit_model_seconds": "5.0",
            "candidate_config_path": str(tmp_path / "succeeded.yaml"),
        },
    ]

    plan = build_promotion_plan(rows, FidelityStageSpec(name="stage2_mid_fidelity", max_candidates=1))

    assert [candidate.source_candidate_id for candidate in plan.promoted_candidates] == ["succeeded"]


def test_build_promotion_plan_rejects_test_metrics(tmp_path: Path) -> None:
    rows = [
        {
            "candidate_id": "candidate",
            "execution_status": "succeeded",
            "validation_rmse": "0.900",
            "validation_mae": "0.650",
            "fit_model_seconds": "12.0",
            "candidate_config_path": str(tmp_path / "candidate.yaml"),
            "test_rmse": "0.880",
        }
    ]
    next_stage = FidelityStageSpec(name="stage2_mid_fidelity", max_candidates=1)

    with pytest.raises(ValueError, match="test metrics"):
        build_promotion_plan(rows, next_stage)


def test_promotion_rejects_test_metric_objective() -> None:
    with pytest.raises(ValueError, match="test metrics"):
        FidelityStageSpec(name="stage2_mid_fidelity", max_candidates=1, objective_metric="test_rmse")


def test_build_promotion_plan_reads_csv_results(tmp_path: Path) -> None:
    config_path = tmp_path / "candidate.yaml"
    result_path = tmp_path / "stage1_results.csv"
    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "execution_status",
                "validation_rmse",
                "validation_mae",
                "fit_model_seconds",
                "candidate_config_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "candidate_id": "candidate",
                "execution_status": "succeeded",
                "validation_rmse": "0.900",
                "validation_mae": "0.650",
                "fit_model_seconds": "12.0",
                "candidate_config_path": str(config_path),
            }
        )

    plan = build_promotion_plan(
        result_path,
        FidelityStageSpec(name="stage2_mid_fidelity", max_candidates=1),
    )

    assert plan.promoted_candidates[0].source_candidate_id == "candidate"


def test_materialize_promoted_candidates_applies_stage_overrides_and_writes_plan(
    tmp_path: Path,
) -> None:
    source_config_path = tmp_path / "source_candidate.yaml"
    source_payload = _base_model_config()
    source_payload["training"]["epochs"] = 3  # type: ignore[index]
    dump_yaml_file(source_config_path, source_payload)
    promotion_plan = build_promotion_plan(
        [
            {
                "candidate_id": "candidate",
                "execution_status": "succeeded",
                "validation_rmse": "0.900",
                "validation_mae": "0.650",
                "fit_model_seconds": "12.0",
                "candidate_config_path": str(source_config_path),
            }
        ],
        FidelityStageSpec(
            name="stage2_mid_fidelity",
            max_candidates=1,
            overrides={"training.epochs": 10},
        ),
    )

    paths = materialize_promoted_candidates(promotion_plan, tmp_path / "study")
    promoted_config = load_yaml_file(next(path for key, path in paths.items() if key.startswith("promoted_config:")))
    promotion_payload = json.loads(paths["promotion_plan"].read_text(encoding="utf-8"))

    assert promoted_config["training"]["epochs"] == 10
    assert promoted_config["clustering"]["induction"]["epochs"] == 20
    assert promotion_payload["to_stage"] == "stage2_mid_fidelity"
    assert promotion_payload["promoted_candidates"][0]["promoted_candidate_config_path"].endswith(
        "candidate_config.yaml"
    )


def test_promoted_configs_apply_next_stage_overrides(tmp_path: Path) -> None:
    test_materialize_promoted_candidates_applies_stage_overrides_and_writes_plan(tmp_path)


def test_materialize_promoted_candidates_can_apply_induction_stage_override(tmp_path: Path) -> None:
    source_config_path = tmp_path / "source_candidate.yaml"
    dump_yaml_file(source_config_path, _base_model_config())
    promotion_plan = build_promotion_plan(
        [
            {
                "candidate_id": "candidate",
                "execution_status": "succeeded",
                "validation_rmse": "0.900",
                "validation_mae": "0.650",
                "fit_model_seconds": "12.0",
                "candidate_config_path": str(source_config_path),
            }
        ],
        FidelityStageSpec(
            name="stage_outer_fidelity",
            max_candidates=1,
            overrides={"clustering.induction.epochs": 10},
        ),
    )

    paths = materialize_promoted_candidates(promotion_plan, tmp_path / "study")
    promoted_config = load_yaml_file(next(path for key, path in paths.items() if key.startswith("promoted_config:")))

    assert promoted_config["training"]["epochs"] == 3
    assert promoted_config["clustering"]["induction"]["epochs"] == 10
