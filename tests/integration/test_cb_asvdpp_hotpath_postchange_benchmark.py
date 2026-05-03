from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "reproduction"
    / "2026-05-03_cb_asvdpp_hotpath_postchange_benchmark_g10.md"
)
ROADMAP_PATH = REPO_ROOT / "docs" / "roadmaps" / "2026-05-02_claim_unlock_and_scalability_plan.md"
READINESS_MATRIX_PATH = REPO_ROOT / "docs" / "publish_readiness_matrix.md"
README_PATH = REPO_ROOT / "README.md"


def test_cb_asvdpp_postchange_benchmark_is_clean_exact_and_claim_limited() -> None:
    evidence = EVIDENCE_PATH.read_text(encoding="utf-8")
    roadmap = ROADMAP_PATH.read_text(encoding="utf-8")
    matrix = READINESS_MATRIX_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")

    assert "status: `pass_for_exact_workbuffer_remediation_context`" in evidence
    assert "git commit: `e6b77c7f9bc5a87259a5e18e618dc18941a3a9e3`" in evidence
    assert "git dirty: `false`" in evidence
    assert "stage count: `12`" in evidence
    assert "cluster artifact cache status: `hit`" in evidence
    assert "user-cluster history cache status: `hit`" in evidence
    assert "train RMSE: `0.6848969434499206`" in evidence
    assert "validation RMSE: `0.9134162708331054`" in evidence
    assert "test RMSE: `0.9102128098774724`" in evidence
    assert "main training wall-clock seconds: `113.81238390004728`" in evidence
    assert "required post-change main-training wall-clock threshold:" in evidence
    assert "`121.68372130202071` seconds" in evidence
    assert "`-7.403998780257792%`" in evidence
    assert "| train RMSE | `0.6848969434499206` | `0.6848969434499206` | `0.0` |" in evidence
    assert "no broad or unqualified speed claim" in evidence
    assert "no scalability claim" in evidence
    assert "Pytest after this evidence note and guardrail test: `141 passed`" in evidence

    assert "pass_for_exact_workbuffer_remediation_context" in roadmap
    assert "docs/evidence/reproduction/2026-05-03_cb_asvdpp_hotpath_postchange_benchmark_g10.md" in roadmap
    assert "observed main-training wall-clock change: `-7.403998780257792%`" in roadmap
    assert "`94%`" in roadmap

    assert "clean post-change work-buffer benchmark" in matrix
    assert "G10 `cb_asvdpp` post-change benchmark unlocks only a narrow speed claim" in matrix

    assert "CB-ASVD++ hotpath post-change benchmark evidence" in readme
