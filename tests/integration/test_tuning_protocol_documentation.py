from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_tuning_protocol_documents_active_configs_and_selection_rule() -> None:
    protocol = (REPO_ROOT / "docs" / "evaluation_protocol.md").read_text(encoding="utf-8")

    for config_name in (
        "ml100k_cb_svdpp_g6_validation_grid.yaml",
        "ml1m_cb_svdpp_stage0.yaml",
        "ml1m_cb_asvdpp_stage0.yaml",
        "ml20m_cb_svdpp_g11_lower_memory_validation_grid.yaml",
    ):
        assert config_name in protocol

    for required_text in (
        "1. niedrigste mittlere `validation_rmse`",
        "2. niedrigere `validation_rmse`-Streuung",
        "3. niedrigere Trainingszeit",
        "4. besserer Memory-/Resource-Status",
    ):
        assert required_text in protocol


def test_tuning_protocol_documents_alpha_and_claim_boundary() -> None:
    protocol = (REPO_ROOT / "docs" / "evaluation_protocol.md").read_text(encoding="utf-8")

    for required_text in (
        "`alpha=0` ist ein expliziter Ablationskandidat",
        "`alpha>0` aktiviert nur den Cluster-Kanal",
        "weder `alpha=0` noch `alpha>0` machen einen Run automatisch",
        "nachgelagerte Outer-Benchmarks mit reserviertem Test-Split",
        "validation-only Selection Runs",
        "blocked/negative-resource Evidence",
    ):
        assert required_text in protocol
