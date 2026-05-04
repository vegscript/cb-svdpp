from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
READINESS_MATRIX = REPO_ROOT / "docs" / "publish_readiness_matrix.md"
PROJECT_REPORT = REPO_ROOT / "docs" / "report" / "project_report.md"
CAMPAIGN_CONTRACT = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "benchmarking"
    / "2026-04-30_large_cb_svdpp_matched_campaign_contract.md"
)
ML10M_CB_SVDPP_MATCHED_PROFILE = (
    REPO_ROOT / "configs" / "models" / "selected" / "ml10m" / "ml10m_cb_svdpp_stage0_transfer.yaml"
)
ML20M_CB_SVDPP_MATCHED_PROFILE = (
    REPO_ROOT / "configs" / "models" / "archive" / "tuned" / "ml20m_cb_svdpp_stage0_transfer.yaml"
)
ML20M_CB_SVDPP_BUDGET_GATE = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "benchmarking"
    / "2026-05-01_ml20m_cb_svdpp_matched_campaign_budget_gate.md"
)
ML20M_CB_SVDPP_SEED1_READOUT = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "models"
    / "cb_svdpp"
    / "2026-05-01_ml20m_cb_svdpp_stage0_transfer_seed1_readout.md"
)
ML20M_CB_SVDPP_SEED2_GATE = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "benchmarking"
    / "2026-05-01_ml20m_cb_svdpp_matched_campaign_seed2_gate.md"
)
ML20M_CB_SVDPP_SEED2_READOUT = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "models"
    / "cb_svdpp"
    / "2026-05-01_ml20m_cb_svdpp_stage0_transfer_seed2_readout.md"
)
ML20M_CB_SVDPP_SEED3_GATE = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "benchmarking"
    / "2026-05-01_ml20m_cb_svdpp_matched_campaign_seed3_gate.md"
)
ML20M_CB_SVDPP_SEED3_BREACH_READOUT = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "models"
    / "cb_svdpp"
    / "2026-05-02_ml20m_cb_svdpp_stage0_transfer_seed3_guardrail_breach.md"
)
RELEASE_MARKER = "submission-2026-05-02-r10"


def _section_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def _single_line(text: str) -> str:
    return " ".join(text.split())


def test_large_dataset_cb_svdpp_claim_boundaries_stay_explicit() -> None:
    matrix = READINESS_MATRIX.read_text(encoding="utf-8")
    report = PROJECT_REPORT.read_text(encoding="utf-8")

    final_claim_matrix = _section_between(matrix, "## Final Claim Matrix", "## Feasibility And Selection Evidence")
    feasibility_matrix = _section_between(matrix, "## Feasibility And Selection Evidence", "## Global Claim Locks")
    report_anchor_section = _section_between(
        report,
        "### 7.1 Clean Benchmark Anchors",
        "### 7.2 Feasibility, Selection, And Deferral Evidence",
    )
    report_feasibility_section = _section_between(
        report,
        "### 7.2 Feasibility, Selection, And Deferral Evidence",
        "### 7.3 Current Non-Claims",
    )

    assert "| `ml10m` | `biased_mf" in final_claim_matrix
    assert "| `ml10m` | `cb_svdpp stage0_transfer`" in final_claim_matrix
    assert "| `ml10m` | `biased_mf`" in report_anchor_section
    assert "| `ml10m` | `cb_svdpp`" in report_anchor_section
    assert "higher validation RMSE, higher test RMSE" in final_claim_matrix
    assert "higher validation RMSE, higher test RMSE" in report_anchor_section

    assert "| `ml10m` | `cb_svdpp stage0_probe_e001`" in feasibility_matrix
    assert "| `ml10m` | `cb_svdpp stage0_probe_e001`" in report_feasibility_section

    assert "| `ml20m` | `biased_mf" in final_claim_matrix
    assert "| `ml20m` | `biased_mf" in report_anchor_section
    assert "| `ml20m` | `cb_svdpp" not in final_claim_matrix
    assert "| `ml20m` | `cb_svdpp" not in report_anchor_section
    assert "| `ml20m` | `cb_svdpp" in feasibility_matrix
    assert "| `ml20m` | `cb_svdpp" in report_feasibility_section

    assert "No final `ml20m` model-comparison claim is allowed" in matrix
    assert "still does not support final `ml20m` model rankings" in _single_line(report)


def test_large_dataset_campaign_contract_is_not_reported_as_completed_evidence() -> None:
    contract = CAMPAIGN_CONTRACT.read_text(encoding="utf-8")
    matrix = READINESS_MATRIX.read_text(encoding="utf-8")
    report = PROJECT_REPORT.read_text(encoding="utf-8")

    assert "This is a run contract, not a benchmark result" in contract
    assert "Do not start a local `ml20m cb_svdpp` matched campaign" in contract
    assert "has since been satisfied for `ml10m`" in contract
    assert "Keep final `ml20m` model-comparison claims blocked" in contract
    assert "current release marker" in contract
    assert "2026-05-01_ml20m_cb_svdpp_matched_campaign_budget_gate.md" in contract
    assert "2026-05-01_ml20m_cb_svdpp_matched_campaign_seed2_gate.md" in contract
    assert "2026-05-01_ml20m_cb_svdpp_matched_campaign_seed3_gate.md" in contract
    assert "2026-05-02_ml20m_cb_svdpp_stage0_transfer_seed3_guardrail_breach.md" in contract
    assert "negative resource evidence" in contract

    final_claim_matrix = _section_between(matrix, "## Final Claim Matrix", "## Feasibility And Selection Evidence")
    assert "2026-04-30_large_cb_svdpp_matched_campaign_contract.md" not in final_claim_matrix
    assert "2026-05-02_ml20m_cb_svdpp_stage0_transfer_seed3_guardrail_breach.md" not in final_claim_matrix
    assert "matched-campaign contract" in matrix
    assert "matched-campaign contract estimates" in _single_line(report)


def test_ml10m_cb_svdpp_matched_profile_obeys_campaign_contract() -> None:
    profile = yaml.safe_load(ML10M_CB_SVDPP_MATCHED_PROFILE.read_text(encoding="utf-8"))
    contract = CAMPAIGN_CONTRACT.read_text(encoding="utf-8")

    assert profile["metadata"]["status"] == "stage0_transfer"
    assert profile["metadata"]["provenance"]["transfer_stage"] == "ml10m_matched_campaign"
    assert profile["model"]["name"] == "cb_svdpp"
    assert profile["training"]["epochs"] == 20
    assert profile["training"]["dtype"] == "float32"
    assert profile["training"]["implicit_policy"] == "ratings_as_implicit"
    assert profile["clustering"]["n_user_clusters"] == 80
    assert profile["clustering"]["n_item_clusters"] == 80
    assert profile["clustering"]["alpha"] == 0.10
    assert "split seeds:" in contract
    assert "`1,2,3`" in contract
    assert "model seed:" in contract
    assert "`1`" in contract


def test_ml20m_cb_svdpp_matched_profile_and_budget_gate_are_split_seed_1_only() -> None:
    profile = yaml.safe_load(ML20M_CB_SVDPP_MATCHED_PROFILE.read_text(encoding="utf-8"))
    budget_gate = ML20M_CB_SVDPP_BUDGET_GATE.read_text(encoding="utf-8")

    assert profile["metadata"]["status"] == "stage0_transfer"
    assert profile["metadata"]["provenance"]["transfer_stage"] == "ml20m_matched_campaign"
    assert profile["model"]["name"] == "cb_svdpp"
    assert profile["training"]["epochs"] == 20
    assert profile["training"]["dtype"] == "float32"
    assert profile["training"]["implicit_policy"] == "ratings_as_implicit"
    assert profile["clustering"]["n_user_clusters"] == 80
    assert profile["clustering"]["n_item_clusters"] == 80
    assert profile["clustering"]["alpha"] == 0.10
    assert "status: `approved_for_split_seed_1_only`" in budget_gate
    assert "split seed:" in budget_gate
    assert "`1`" in budget_gate
    assert "Split seeds `2` and `3` are not automatically authorized" in budget_gate
    assert "does not unlock any final `ml20m` model-comparison claim" in budget_gate


def test_ml20m_cb_svdpp_seed1_readout_only_authorizes_seed2() -> None:
    seed1_readout = ML20M_CB_SVDPP_SEED1_READOUT.read_text(encoding="utf-8")
    seed2_gate = ML20M_CB_SVDPP_SEED2_GATE.read_text(encoding="utf-8")

    assert "status: `completed_single_split_seed`" in seed1_readout
    assert "git_commit: `aa167dbe8df0a4dee0933612a3ea7c0c0dec7ffd`" in seed1_readout
    assert "validation RMSE:" in seed1_readout
    assert "`0.780558`" in seed1_readout
    assert "test RMSE:" in seed1_readout
    assert "`0.781255`" in seed1_readout
    assert "peak memory MB:" in seed1_readout
    assert "`18586.136719`" in seed1_readout
    assert "does not unlock a final `ml20m` model-comparison claim" in seed1_readout

    assert "status: `approved_for_split_seed_2_only`" in seed2_gate
    assert "split seed:" in seed2_gate
    assert "`2`" in seed2_gate
    assert "Split seed `3` is not automatically authorized" in seed2_gate
    assert "does not unlock any final `ml20m` model-comparison claim" in seed2_gate


def test_ml20m_cb_svdpp_seed2_readout_only_authorizes_seed3() -> None:
    seed2_readout = ML20M_CB_SVDPP_SEED2_READOUT.read_text(encoding="utf-8")
    seed3_gate = ML20M_CB_SVDPP_SEED3_GATE.read_text(encoding="utf-8")

    assert "status: `completed_single_split_seed`" in seed2_readout
    assert "git_commit: `001f86f70298ed1848fbb4c8f5daeca24f62bb96`" in seed2_readout
    assert "validation RMSE:" in seed2_readout
    assert "`0.781702`" in seed2_readout
    assert "test RMSE:" in seed2_readout
    assert "`0.781773`" in seed2_readout
    assert "peak memory MB:" in seed2_readout
    assert "`19128.371094`" in seed2_readout
    assert "does not unlock a final `ml20m` model-comparison claim" in seed2_readout

    assert "status: `approved_for_split_seed_3_only`" in seed3_gate
    assert "split seed:" in seed3_gate
    assert "`3`" in seed3_gate
    assert "final single-split gate" in seed3_gate
    assert "does not unlock any final `ml20m` model-comparison claim" in seed3_gate


def test_ml20m_cb_svdpp_seed3_guardrail_breach_blocks_final_claim() -> None:
    seed3_readout = ML20M_CB_SVDPP_SEED3_BREACH_READOUT.read_text(encoding="utf-8")
    matrix = READINESS_MATRIX.read_text(encoding="utf-8")

    assert "status: `completed_guardrail_breach_negative_evidence`" in seed3_readout
    assert "git_commit: `1cb39de0fbd8ca05644f62d03903eeb172fa9fee`" in seed3_readout
    assert "validation RMSE:" in seed3_readout
    assert "`0.781010`" in seed3_readout
    assert "test RMSE:" in seed3_readout
    assert "`0.781511`" in seed3_readout
    assert "peak memory MB:" in seed3_readout
    assert "`19898.871094`" in seed3_readout
    assert "exceeding the guardrail by approximately `251.58 MB`" in seed3_readout
    assert "negative resource evidence" in seed3_readout
    assert "must not be folded into a final benchmark matrix" in _single_line(seed3_readout)
    assert "does not unlock a final `ml20m` model-comparison claim" in seed3_readout

    final_claim_matrix = _section_between(matrix, "## Final Claim Matrix", "## Feasibility And Selection Evidence")
    feasibility_matrix = _section_between(matrix, "## Feasibility And Selection Evidence", "## Global Claim Locks")
    assert "2026-05-02_ml20m_cb_svdpp_stage0_transfer_seed3_guardrail_breach.md" not in final_claim_matrix
    assert "2026-05-02_ml20m_cb_svdpp_stage0_transfer_seed3_guardrail_breach.md" in feasibility_matrix
    assert "guardrail_breached_local_campaign" in matrix


def test_release_marker_is_consistent_across_release_facing_documents() -> None:
    release_files = (
        README,
        READINESS_MATRIX,
        PROJECT_REPORT,
        REPO_ROOT / "docs" / "evidence" / "release" / "2026-05-02_release_hygiene.md",
        REPO_ROOT / "docs" / "evidence" / "reproduction" / "current" / "2026-05-02_public_clean_import.md",
    )

    for path in release_files:
        text = path.read_text(encoding="utf-8")
        assert RELEASE_MARKER in text, path
        assert "submission-2026-04-30-r8" not in text, path
