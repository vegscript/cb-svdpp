from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "reproduction"
    / "2026-05-03_cb_asvdpp_hotpath_decision_g7.md"
)
ROADMAP_PATH = REPO_ROOT / "docs" / "roadmaps" / "2026-05-02_claim_unlock_and_scalability_plan.md"
READINESS_MATRIX_PATH = REPO_ROOT / "docs" / "publish_readiness_matrix.md"
README_PATH = REPO_ROOT / "README.md"


def test_cb_asvdpp_hotpath_decision_is_profile_bound_and_non_claiming() -> None:
    evidence = EVIDENCE_PATH.read_text(encoding="utf-8")
    roadmap = ROADMAP_PATH.read_text(encoding="utf-8")
    matrix = READINESS_MATRIX_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")

    assert "status: `pass_for_hotpath_prioritization_not_remediation`" in evidence
    assert "git dirty: `false`" in evidence
    assert "stage count: `12`" in evidence
    assert "cluster artifacts: `miss`" in evidence
    assert "user-cluster history: `miss`" in evidence
    assert "main_training` | `115.10358980001183` | `92.44%`" in evidence
    assert "Total profiled wall-clock seconds: `124.51294220011914`" in evidence
    assert "train RMSE: `0.6848969434499206`" in evidence
    assert "validation RMSE: `0.9134162708331054`" in evidence
    assert "test RMSE: `0.9102128098774724`" in evidence
    assert "single-run profiling decision, not a benchmark anchor" in evidence
    assert "no speed claim" in evidence
    assert "no scalability claim" in evidence
    assert "no claim that a remediation has already improved runtime" in evidence
    assert "Pytest: `137 passed`" in evidence

    assert "pass_for_hotpath_prioritization_not_remediation" in roadmap
    assert "main_training` share: about `92.44%`" in roadmap
    assert "`94%`" in roadmap
    assert "approved_for_exact_remediation_contract" in roadmap

    assert "bounded hot-path profiling decision" in matrix
    assert "clean post-change work-buffer benchmark" in matrix
    assert "hot-path decision may support only prioritization" in matrix

    assert "CB-ASVD++ hotpath decision evidence" in readme
