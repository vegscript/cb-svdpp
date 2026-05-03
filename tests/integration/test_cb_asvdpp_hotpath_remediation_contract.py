from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "reproduction"
    / "2026-05-03_cb_asvdpp_hotpath_remediation_contract_g8.md"
)
ROADMAP_PATH = REPO_ROOT / "docs" / "roadmaps" / "2026-05-02_claim_unlock_and_scalability_plan.md"
READINESS_MATRIX_PATH = REPO_ROOT / "docs" / "publish_readiness_matrix.md"
README_PATH = REPO_ROOT / "README.md"


def test_cb_asvdpp_hotpath_remediation_contract_has_exact_gates() -> None:
    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    roadmap = ROADMAP_PATH.read_text(encoding="utf-8")
    matrix = READINESS_MATRIX_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")

    assert "status: `approved_for_exact_remediation_contract`" in contract
    assert "train_cb_asvdpp_epoch_numba" in contract
    assert "fixed-size work-buffer reuse" in contract
    assert "current Python fallback semantics" in contract
    assert "absolute tolerance: `1e-6`" in contract
    assert "relative tolerance: `1e-6`" in contract
    assert "main_training_wall_clock_seconds` must decrease by at least `1.0%`" in contract
    assert "absolute drift must be at most `1e-6`" in contract
    assert "existing G7 profiling run is prioritization evidence only" in contract
    assert "fresh pre-change baseline" in contract
    assert "no reordered SGD updates" in contract
    assert "no MiniBatchKMeans substitution" in contract
    assert "no `R_star` objective integration" in contract
    assert "Pytest: `138 passed`" in contract

    for array_name in (
        "user bias",
        "item bias",
        "user factors",
        "item factors",
        "explicit item factors",
        "implicit item factors",
        "user-cluster factors",
        "item-cluster factors",
        "explicit-cluster factors",
        "implicit-cluster factors",
    ):
        assert array_name in contract

    assert "approved_for_exact_remediation_contract" in roadmap
    assert "pass_for_clean_prechange_baseline" in roadmap
    assert "`94%`" in roadmap

    assert "clean post-change work-buffer benchmark" in matrix
    assert "unlocks no speed claim" in matrix

    assert "CB-ASVD++ hotpath remediation contract evidence" in readme
