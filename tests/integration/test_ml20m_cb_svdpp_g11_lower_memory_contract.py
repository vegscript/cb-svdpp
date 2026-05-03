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
    / "ml20m_cb_svdpp_g11_lower_memory_validation_grid.yaml"
)
EVIDENCE_PATH = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "reproduction"
    / "2026-05-03_ml20m_cb_svdpp_g11_lower_memory_validation_contract.md"
)
ROADMAP_PATH = REPO_ROOT / "docs" / "roadmaps" / "2026-05-02_claim_unlock_and_scalability_plan.md"
READINESS_MATRIX_PATH = REPO_ROOT / "docs" / "publish_readiness_matrix.md"
README_PATH = REPO_ROOT / "README.md"


def _walk_mapping_values(value: Any) -> list[Any]:
    values = [value]
    if isinstance(value, dict):
        for child in value.values():
            values.extend(_walk_mapping_values(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(_walk_mapping_values(child))
    return values


def test_ml20m_g11_lower_memory_contract_is_validation_only_and_resource_gated() -> None:
    payload = load_yaml_file(CONFIG_PATH)
    evidence = EVIDENCE_PATH.read_text(encoding="utf-8")
    roadmap = ROADMAP_PATH.read_text(encoding="utf-8")
    matrix = READINESS_MATRIX_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")

    assert payload["metadata"]["status"] == "validation_grid_contract"
    assert payload["metadata"]["purpose"] == "g11_ml20m_lower_memory_reassessment"
    assert payload["base_model_config"] == "configs/models/cb_svdpp.yaml"

    tuning = payload["tuning"]
    assert tuning["name"] == "ml20m_cb_svdpp_g11_lower_memory_validation_grid"
    assert tuning["dataset_short_name"] == "ml20m"
    assert tuning["split_family"] == "benchmark_random_v1"
    assert tuning["selection_stage"] == "g11_lower_memory_validation_grid"
    assert tuning["objective"] == "validation_rmse_mean"
    assert tuning["split_seeds"] == [1, 2]
    assert tuning["model_seed"] == 1

    resource_gate = payload["resource_gate"]
    assert resource_gate["device_profile"] == "local_i5_2500k_24gb"
    assert resource_gate["ram_guardrail_fraction"] == 0.8
    assert resource_gate["max_peak_memory_mb"] == 19660.8
    assert resource_gate["reject_candidate_on_any_guardrail_breach"] is True

    candidates = payload["candidates"]
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    assert len(candidates) == 8
    assert len(set(candidate_ids)) == len(candidate_ids)
    assert len(candidates) * len(tuning["split_seeds"]) == 16

    latent_dims: set[int] = set()
    cluster_counts: set[int] = set()
    alphas: set[float] = set()
    for candidate in candidates:
        overrides = candidate["overrides"]
        training = overrides["training"]
        clustering = overrides["clustering"]

        assert training["epochs"] == 1
        assert training["learning_rate"] == 0.01
        assert training["lambda_b"] == 0.02
        assert clustering["n_user_clusters"] == clustering["n_item_clusters"]
        assert clustering["algorithm"] == "kmeans"
        assert clustering["kmeans_n_init"] == 10

        latent_dims.add(int(training["latent_dim"]))
        cluster_counts.add(int(clustering["n_user_clusters"]))
        alphas.add(float(clustering["alpha"]))

    assert latent_dims == {16, 32}
    assert cluster_counts == {32, 64}
    assert alphas == {0.0, 0.025}

    serialized_values = {str(value).lower() for value in _walk_mapping_values(payload)}
    assert "test_rmse" not in serialized_values
    assert "evaluate_test" not in serialized_values
    assert "minibatchkmeans" not in serialized_values

    notes = "\n".join(str(note) for note in payload["notes"])
    assert "no test-set evaluation during selection" in notes
    assert "every candidate must stay below the local 80 percent RAM guardrail" in notes
    assert "no MiniBatchKMeans substitution" in notes
    assert "R_star remains diagnostic-only" in notes
    assert "selection evidence only" in notes

    assert "contract_ready_g11_ml20m_lower_memory_validation_reassessment" in evidence
    assert "planned validation-only runs: `16`" in evidence
    assert "test-set evaluation during selection: not allowed" in evidence
    assert "no final `ml20m cb_svdpp` benchmark claim" in evidence
    assert "no scalability claim from this contract" in evidence

    assert "status: `contract_ready_g11_ml20m_lower_memory_validation_reassessment`" in roadmap
    assert "docs/evidence/reproduction/2026-05-03_ml20m_cb_svdpp_g11_lower_memory_validation_contract.md" in roadmap
    assert "`95%`" in roadmap

    assert "G11 lower-memory validation contract exists" in matrix
    assert "no final `ml20m cb_svdpp` benchmark claim" in matrix

    assert "ML20M CB-SVD++ G11 lower-memory validation contract evidence" in readme
