from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "reproduction"
    / "2026-05-03_cb_svdpp_g6_outer_benchmark_contract.md"
)
RUN_EVIDENCE_PATH = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "reproduction"
    / "2026-05-03_cb_svdpp_g6_outer_benchmark_run.md"
)
ROADMAP_PATH = REPO_ROOT / "docs" / "roadmaps" / "2026-05-02_claim_unlock_and_scalability_plan.md"
READINESS_MATRIX_PATH = REPO_ROOT / "docs" / "publish_readiness_matrix.md"
README_PATH = REPO_ROOT / "README.md"
SELECTED_CONFIG = "configs/models/tuned/ml100k_cb_svdpp_g6_validation_selected.yaml"
SELECTION_EVIDENCE = "docs/evidence/reproduction/2026-05-03_cb_svdpp_g6_validation_grid_run.md"
PROCESSED_MANIFEST = (
    "data/processed/ml100k/ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json"
)


def test_g6_outer_benchmark_contract_freezes_inputs_and_claim_boundary() -> None:
    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    roadmap = ROADMAP_PATH.read_text(encoding="utf-8")
    matrix = READINESS_MATRIX_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")

    assert "status: `approved_for_clean_outer_benchmark_contract`" in contract
    assert "run contract, not a benchmark result" in contract
    assert SELECTED_CONFIG in contract
    assert SELECTION_EVIDENCE in contract
    assert PROCESSED_MANIFEST in contract
    assert "selected by `validation_rmse_mean`" in contract
    assert "`0` non-null `test_rmse` values" in contract

    assert "train-cb-svdpp" in contract
    assert "benchmark-random-multiseed" in contract
    assert "--split-family benchmark_random_v1" in contract
    assert "--train-ratio 0.8" in contract
    assert "--validation-ratio 0.1" in contract
    assert "--model-seed 1" in contract
    assert "--split-cache enable" in contract
    assert "--training-index-cache" in contract
    assert "--cluster-artifact-cache" in contract
    assert "--split-seeds 1,2,3" in contract

    for split_seed in ("--split-seed 1", "--split-seed 2", "--split-seed 3"):
        assert split_seed in contract

    assert "worktree must be clean before the first outer run starts" in contract
    assert "contract must already be committed before the first outer run starts" in contract
    assert "same git commit as the aggregation process" in contract
    assert "without editing model config" in contract
    assert "seeing any outer test metric" in contract
    assert "campaign is invalid for final G6 promotion" in contract

    assert "final `ml100k cb_svdpp` quality claim" in contract
    assert "speed claim" in contract
    assert "scalability claim" in contract
    assert "production-readiness claim" in contract
    assert "SOTA claim" in contract
    assert "paper-faithfulness claim" in contract
    assert "`R_star` optimization claim" in contract
    assert "the outer run has not been executed yet" in contract

    assert "approved_for_clean_outer_benchmark_contract" in roadmap
    assert "commit this contract first" in roadmap
    assert "completed_g6_clean_outer_benchmark" in roadmap
    assert "`91%`" in roadmap

    assert "completed clean outer benchmark readout under `benchmark_random_v1`" in matrix
    assert "2026-05-03_cb_svdpp_g6_outer_benchmark_run.md" in matrix
    assert "benchmark_evidence_ready_g6_outer_anchor_documented" in matrix
    assert "the G6 `cb_svdpp` outer benchmark may be used only as its own" in matrix

    assert "G6 outer benchmark contract evidence" in readme
    assert "2026-05-03_cb_svdpp_g6_outer_benchmark_contract.md" in readme


def test_g6_outer_benchmark_run_records_clean_aggregate_without_broad_claims() -> None:
    run_evidence = RUN_EVIDENCE_PATH.read_text(encoding="utf-8")
    matrix = READINESS_MATRIX_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")

    assert "status: `pass_for_clean_outer_benchmark`" in run_evidence
    assert "git commit: `67570ed8fde4c158848ab80494524ab203b40df5`" in run_evidence
    assert "git dirty: `false`" in run_evidence
    assert "measured sample count: `3`" in run_evidence
    assert "split seeds: `1,2,3`" in run_evidence
    assert "model seed: `1`" in run_evidence
    assert "device profile claim eligible: `true`" in run_evidence
    assert "validation RMSE | `0.9566122815305916`" in run_evidence
    assert "test RMSE | `0.9595668222022953`" in run_evidence
    assert "training wall-clock seconds | `4.767408333330725`" in run_evidence
    assert "peak memory MB | `237.70703125`" in run_evidence
    assert "SOTA, scalability, speed" in run_evidence
    assert "paper-faithfulness claim" in run_evidence
    assert "comparison against the older `paper_faithful_ml100k_v1`" in run_evidence
    assert "Pytest: `136 passed`" in run_evidence

    assert "pass_for_current_anchor_set_plus_g6_outer_benchmark" in matrix
    assert "test RMSE mean `0.9595668222022953`" in matrix
    assert "paper_faithful_ml100k_v1` anchor rows" in matrix
    assert "G6 outer benchmark run evidence" in readme
