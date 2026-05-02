from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ML100K_CLEAN_ANCHORS = {
    "biased_mf": {
        "config": "configs/models/tuned/ml100k_biased_mf_stage1.yaml",
        "evidence": [
            "docs/evidence/models/biased_mf/2026-04-13_ml100k_biased_mf_inner_tuning_stage1.md",
            "docs/evidence/models/biased_mf/2026-04-13_ml100k_biased_mf_stage1_tuned_benchmark.md",
            "docs/evidence/models/biased_mf/2026-04-15_ml100k_biased_mf_stage1_tuned_clean_multiseed.md",
        ],
    },
    "svdpp": {
        "config": "configs/models/tuned/ml100k_svdpp_stage1.yaml",
        "evidence": [
            "docs/evidence/models/svdpp/2026-04-13_ml100k_svdpp_inner_tuning_stage1.md",
            "docs/evidence/models/svdpp/2026-04-13_ml100k_svdpp_stage1_tuned_benchmark.md",
            "docs/evidence/models/svdpp/2026-04-15_ml100k_svdpp_stage1_tuned_clean_multiseed.md",
        ],
    },
    "cb_svdpp": {
        "config": "configs/models/tuned/ml100k_cb_svdpp_stage1.yaml",
        "evidence": [
            "docs/evidence/models/cb_svdpp/2026-04-15_ml100k_cb_svdpp_inner_tuning_stage1.md",
            "docs/evidence/models/cb_svdpp/2026-04-15_ml100k_cb_svdpp_stage1_tuned_benchmark.md",
            "docs/evidence/models/cb_svdpp/2026-04-15_ml100k_cb_svdpp_stage1_tuned_clean_multiseed.md",
        ],
    },
    "cb_asvdpp": {
        "config": "configs/models/tuned/ml100k_cb_asvdpp_stage1.yaml",
        "evidence": [
            "docs/evidence/models/cb_asvdpp/2026-04-15_ml100k_cb_asvdpp_inner_tuning_stage1.md",
            "docs/evidence/models/cb_asvdpp/2026-04-15_ml100k_cb_asvdpp_stage1_tuned_benchmark.md",
            "docs/evidence/models/cb_asvdpp/2026-04-15_ml100k_cb_asvdpp_stage1_tuned_clean_multiseed.md",
        ],
    },
}


def _read_repo_text(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_ml100k_clean_anchor_inventory_is_backed_by_configs_and_evidence() -> None:
    readme_text = _read_repo_text("README.md")
    master_plan_text = _read_repo_text("docs/project_master_plan.md")
    report_text = _read_repo_text("docs/report/project_report.md")

    assert "Current clean benchmark anchors:" in readme_text
    assert "`ml100k` | Clean multi-seed anchors" in readme_text
    assert "clean `ml100k` Mehrfach-Seed-Anker" in master_plan_text
    assert "### 7.1 Clean Benchmark Anchors" in report_text
    assert "## 12. Appendix: Evidence Map" in report_text

    for model_name, anchor in ML100K_CLEAN_ANCHORS.items():
        config_path = REPO_ROOT / anchor["config"]
        assert config_path.exists()
        assert f"| `ml100k` | `{model_name}` |" in report_text

        for evidence_path_string in anchor["evidence"]:
            evidence_path = REPO_ROOT / evidence_path_string
            assert evidence_path.exists()
            assert evidence_path_string in report_text
