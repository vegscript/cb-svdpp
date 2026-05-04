from __future__ import annotations

from pathlib import Path
from typing import Any

from recsys_lab.config.loader import load_yaml_file

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "tuning"
    / "active"
    / "ml100k_cb_svdpp_g6_validation_grid.yaml"
)
EVIDENCE_PATH = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "reproduction"
    / "current"
    / "2026-05-03_cb_svdpp_g6_validation_grid_contract.md"
)
RUN_EVIDENCE_PATH = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "reproduction"
    / "current"
    / "2026-05-03_cb_svdpp_g6_validation_grid_run.md"
)
SELECTED_CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "models"
    / "selected"
    / "ml100k"
    / "ml100k_cb_svdpp_g6_validation_selected.yaml"
)
ROADMAP_PATH = REPO_ROOT / "docs" / "roadmaps" / "2026-05-02_claim_unlock_and_scalability_plan.md"


def _walk_mapping_values(value: Any) -> list[Any]:
    values = [value]
    if isinstance(value, dict):
        for child in value.values():
            values.extend(_walk_mapping_values(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(_walk_mapping_values(child))
    return values


def test_g6_validation_grid_contract_is_validation_only_and_method_bounded() -> None:
    payload = load_yaml_file(CONFIG_PATH)
    evidence = EVIDENCE_PATH.read_text(encoding="utf-8")
    run_evidence = RUN_EVIDENCE_PATH.read_text(encoding="utf-8")
    selected_payload = load_yaml_file(SELECTED_CONFIG_PATH)
    roadmap = ROADMAP_PATH.read_text(encoding="utf-8")

    assert payload["metadata"]["status"] == "validation_grid_contract"
    assert payload["base_model_config"] == "configs/models/cb_svdpp.yaml"

    tuning = payload["tuning"]
    assert tuning["name"] == "ml100k_cb_svdpp_g6_validation_grid"
    assert tuning["dataset_short_name"] == "ml100k"
    assert tuning["split_family"] == "benchmark_random_v1"
    assert tuning["selection_stage"] == "g6_validation_grid"
    assert tuning["objective"] == "validation_rmse_mean"
    assert tuning["split_seeds"] == [1, 2, 3]
    assert tuning["model_seed"] == 1
    assert tuning["promoted_from"]["candidate_id"] == "rank032_uc064_ic064_a000_lr0100_reg0020_e002"

    candidates = payload["candidates"]
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    assert len(candidates) == 12
    assert len(set(candidate_ids)) == len(candidate_ids)
    assert "rank032_uc064_ic064_a0000_lr0100_reg0020_e002" in candidate_ids

    cluster_counts: set[int] = set()
    alphas: set[float] = set()
    planned_runs = len(candidates) * len(tuning["split_seeds"])
    for candidate in candidates:
        overrides = candidate["overrides"]
        training = overrides["training"]
        clustering = overrides["clustering"]

        assert training["latent_dim"] == 32
        assert training["epochs"] == 2
        assert training["learning_rate"] == 0.01
        assert training["lambda_b"] == 0.02
        assert clustering["n_user_clusters"] == clustering["n_item_clusters"]
        assert "algorithm" not in clustering

        cluster_counts.add(int(clustering["n_user_clusters"]))
        alphas.add(float(clustering["alpha"]))

    assert cluster_counts == {32, 64, 80, 100}
    assert alphas == {0.0, 0.025, 0.05}
    assert planned_runs == 36

    serialized_values = {str(value).lower() for value in _walk_mapping_values(payload)}
    assert "test_rmse" not in serialized_values
    assert "evaluate_test" not in serialized_values
    assert "r_star" not in serialized_values
    assert "minibatchkmeans" not in serialized_values

    notes = "\n".join(str(note) for note in payload["notes"])
    assert "no test-set evaluation during selection" in notes
    assert "R_star remains diagnostic-only" in notes
    assert "no MiniBatchKMeans substitution" in notes
    assert "selection evidence only" in notes

    assert "contract_ready_g5_to_g6_validation_grid" in evidence
    assert "planned validation-only runs: `36`" in evidence
    assert "no test-set evaluation during selection" in evidence
    assert "no G6 tuning result yet" in evidence

    selected_metadata = selected_payload["metadata"]
    selected_training = selected_payload["training"]
    selected_clustering = selected_payload["clustering"]
    assert selected_metadata["status"] == "validation_selected"
    assert selected_metadata["provenance"]["selected_candidate_id"] == "rank032_uc100_ic100_a0000_lr0100_reg0020_e002"
    assert selected_metadata["provenance"]["selection_objective"] == "validation_rmse_mean"
    assert selected_metadata["provenance"]["selection_split_seeds"] == [1, 2, 3]
    assert selected_training["latent_dim"] == 32
    assert selected_training["epochs"] == 2
    assert selected_training["learning_rate"] == 0.01
    assert selected_clustering["n_user_clusters"] == 100
    assert selected_clustering["n_item_clusters"] == 100
    assert selected_clustering["alpha"] == 0.0
    assert selected_clustering["algorithm"] == "kmeans"

    assert "status: `pass_for_validation_only_selection`" in run_evidence
    assert "candidate run count: `36`" in run_evidence
    assert "non-null `test_rmse` count across candidate metrics: `0`" in run_evidence
    assert "rank032_uc100_ic100_a0000_lr0100_reg0020_e002" in run_evidence
    assert "validation RMSE mean: `0.9566122815305916`" in run_evidence
    assert "configs/models/selected/ml100k/ml100k_cb_svdpp_g6_validation_selected.yaml" in run_evidence
    assert "no final `ml100k cb_svdpp` quality claim" in run_evidence

    assert "status: `completed_g6_validation_only_selection`" in roadmap
    assert "docs/evidence/reproduction/current/2026-05-03_cb_svdpp_g6_validation_grid_run.md" in roadmap
