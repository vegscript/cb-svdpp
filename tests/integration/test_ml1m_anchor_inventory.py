import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ML1M_CLEAN_ANCHORS = {
    "biased_mf": {
        "config": "configs/models/tuned/ml1m_biased_mf_stage0_transfer.yaml",
        "evidence": (
            "docs/evidence/models/biased_mf/2026-04-21_ml1m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md"
        ),
        "benchmark_manifest": (
            "artifacts/benchmarks/"
            "2026-04-21T193857Z_ml1m_benchmark_random_v1_biased_mf_multiseed_"
            "s001_s002_s003_modelseed_s001_local_i5_2500k_24gb/benchmark_manifest.json"
        ),
    },
    "cb_svdpp": {
        "config": "configs/models/tuned/ml1m_cb_svdpp_stage0_transfer.yaml",
        "evidence": (
            "docs/evidence/models/cb_svdpp/2026-04-21_ml1m_cb_svdpp_stage0_transfer_clean_multiseed_benchmark.md"
        ),
        "benchmark_manifest": (
            "artifacts/benchmarks/"
            "2026-04-21T193859Z_ml1m_benchmark_random_v1_cb_svdpp_multiseed_"
            "s001_s002_s003_modelseed_s001_local_i5_2500k_24gb/benchmark_manifest.json"
        ),
    },
}


def _read_repo_text(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_ml1m_clean_anchor_inventory_is_backed_by_benchmark_evidence_and_docs() -> None:
    readme_text = _read_repo_text("README.md")
    master_plan_text = _read_repo_text("docs/project_master_plan.md")
    report_text = _read_repo_text("docs/report/project_report.md")

    assert "Current clean benchmark anchors:" in readme_text
    assert "`ml1m` | Clean matched multi-seed comparison for `biased_mf` vs `cb_svdpp`." in readme_text
    assert "ml1m` existieren clean Mehrfach-Seed-Benchmark-Anker fuer" in master_plan_text
    assert "### 7.1 Clean Benchmark Anchors" in report_text
    assert "The clean `ml1m` anchor table is narrower" in report_text

    for model_name, anchor in ML1M_CLEAN_ANCHORS.items():
        assert f"| `ml1m` | `{model_name}` | `stage0_transfer`, seeds `1,2,3` |" in report_text
        assert anchor["evidence"] in report_text

        config_path = REPO_ROOT / anchor["config"]
        evidence_path = REPO_ROOT / anchor["evidence"]
        benchmark_manifest_path = REPO_ROOT / anchor["benchmark_manifest"]

        assert config_path.exists()
        assert evidence_path.exists()
        assert benchmark_manifest_path.exists()

        benchmark_manifest = json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
        assert benchmark_manifest["status"] == "completed"
        assert benchmark_manifest["git"]["dirty"] is False
        assert benchmark_manifest["inputs"]["split_seeds"] == [1, 2, 3]
        assert benchmark_manifest["inputs"]["model_seeds"] == [1]
