from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "reproduction"
    / "2026-05-03_cb_asvdpp_hotpath_prechange_baseline_g9.md"
)
ROADMAP_PATH = REPO_ROOT / "docs" / "roadmaps" / "2026-05-02_claim_unlock_and_scalability_plan.md"
READINESS_MATRIX_PATH = REPO_ROOT / "docs" / "publish_readiness_matrix.md"
README_PATH = REPO_ROOT / "README.md"


def test_cb_asvdpp_prechange_baseline_is_clean_and_non_claiming() -> None:
    evidence = EVIDENCE_PATH.read_text(encoding="utf-8")
    roadmap = ROADMAP_PATH.read_text(encoding="utf-8")
    matrix = READINESS_MATRIX_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")

    assert "status: `pass_for_clean_prechange_baseline`" in evidence
    assert "git commit: `bc966e42f4fc2cf3d09c7f7194e17a81c93617cc`" in evidence
    assert "git dirty: `false`" in evidence
    assert "stage count: `12`" in evidence
    assert "cluster artifact cache status: `hit`" in evidence
    assert "user-cluster history cache status: `hit`" in evidence
    assert "train RMSE: `0.6848969434499206`" in evidence
    assert "validation RMSE: `0.9134162708331054`" in evidence
    assert "test RMSE: `0.9102128098774724`" in evidence
    assert "main training wall-clock seconds: `122.91284980002092`" in evidence
    assert "total profiled wall-clock seconds: `125.95959799998673`" in evidence
    assert "no claim that a remediation has already improved runtime" in evidence
    assert "Pytest: `139 passed`" in evidence

    assert "pass_for_clean_prechange_baseline" in roadmap
    assert "main_training_wall_clock_seconds`: `122.91284980002092`" in roadmap
    assert "`94%`" in roadmap
    assert "pass_for_exact_workbuffer_remediation_context" in roadmap

    assert "clean post-change work-buffer benchmark" in matrix
    assert "pre-change baseline is comparison evidence only" in matrix

    assert "CB-ASVD++ hotpath pre-change baseline evidence" in readme
