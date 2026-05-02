from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ML1M_EXPLORATORY_READOUTS = {
    "biased_mf": {
        "config": "configs/models/tuned/ml1m_biased_mf_stage0_transfer.yaml",
        "evidence": ("docs/evidence/models/biased_mf/2026-04-16_ml1m_biased_mf_stage0_transfer_clean_seed_sweep.md"),
    },
    "cb_svdpp": {
        "config": "configs/models/tuned/ml1m_cb_svdpp_stage0_transfer.yaml",
        "evidence": ("docs/evidence/models/cb_svdpp/2026-04-16_ml1m_cb_svdpp_stage0_transfer_clean_seed_readout.md"),
    },
}


def _read_repo_text(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_ml1m_exploratory_inventory_is_backed_by_configs_evidence_and_report() -> None:
    report_text = _read_repo_text("docs/report/project_report.md")

    assert "Historical but superseded `ml1m` exploratory evidence" in report_text

    for item in ML1M_EXPLORATORY_READOUTS.values():
        config_path = REPO_ROOT / item["config"]
        evidence_path = REPO_ROOT / item["evidence"]

        assert config_path.exists()
        assert evidence_path.exists()
        assert item["evidence"] in report_text
