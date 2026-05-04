import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

TUNING_CONFIG = "configs/experiments/tuning/ml1m_cb_asvdpp_stage0.yaml"
EVIDENCE_NOTE = "docs/evidence/models/cb_asvdpp/2026-04-21_ml1m_cb_asvdpp_inner_tuning_stage0.md"
BENCHMARK_DIR = "artifacts/benchmarks/2026-04-21T204336Z_ml1m_inner_tuning_cb_asvdpp_stage0_local_i5_2500k_24gb"


def _read_repo_text(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_ml1m_cb_asvdpp_clean_tuning_inventory_is_backed_by_evidence_and_report() -> None:
    readme_text = _read_repo_text("README.md")
    master_plan_text = _read_repo_text("docs/project_master_plan.md")
    report_text = _read_repo_text("docs/report/project_report.md")

    assert "no final `ml1m cb_asvdpp` benchmark-anchor claim" in readme_text
    assert "clean reduced-budget" in master_plan_text
    assert "### 7.2 Feasibility, Selection, And Deferral Evidence" in report_text
    assert "clean reduced-budget selection" in report_text
    assert EVIDENCE_NOTE in report_text

    tuning_config_path = REPO_ROOT / TUNING_CONFIG
    evidence_note_path = REPO_ROOT / EVIDENCE_NOTE
    benchmark_manifest_path = REPO_ROOT / BENCHMARK_DIR / "benchmark_manifest.json"
    summary_path = REPO_ROOT / BENCHMARK_DIR / "summary.json"

    assert tuning_config_path.exists()
    assert evidence_note_path.exists()
    missing_artifacts = [
        str(path.relative_to(REPO_ROOT))
        for path in (benchmark_manifest_path, summary_path)
        if not path.exists()
    ]
    if missing_artifacts:
        pytest.skip(
            "ignored ml1m cb_asvdpp tuning artifacts are not present in this workspace: "
            + ", ".join(missing_artifacts)
        )

    benchmark_manifest = json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert benchmark_manifest["status"] == "completed"
    assert benchmark_manifest["git"]["dirty"] is False
    assert benchmark_manifest["git"]["commit"] == "e515d20f6f78d1bdc88d89a13876e0ea6272bd0e"
    assert summary["model"] == "cb_asvdpp"
    assert summary["selection_units"] == ["s001", "s002"]
    assert len(summary["candidates"]) == 3
    assert summary["best_candidate"]["candidate_id"] == "rank064_uc064_ic064_a010_lr0075_reg0025_e002"
