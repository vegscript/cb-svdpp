from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
READINESS_MATRIX = REPO_ROOT / "docs" / "publish_readiness_matrix.md"
ML10M_RAW_EVIDENCE = REPO_ROOT / "docs" / "evidence" / "data" / "2026-04-24_ml10m_raw_acquisition.md"
ML10M_PROCESSED_EVIDENCE = REPO_ROOT / "docs" / "evidence" / "data" / "2026-04-24_ml10m_processed_ingestion.md"
ML10M_BIASED_MF_EVIDENCE = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "models"
    / "biased_mf"
    / "2026-04-24_ml10m_biased_mf_stage0_transfer_feasibility.md"
)
ML10M_BIASED_MF_MULTISEED_EVIDENCE = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "models"
    / "biased_mf"
    / "2026-04-30_ml10m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md"
)
ML10M_CB_SVDPP_EVIDENCE = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "models"
    / "cb_svdpp"
    / "2026-04-24_ml10m_cb_svdpp_stage0_probe_e001_feasibility.md"
)
ML10M_CB_SVDPP_MULTISEED_EVIDENCE = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "models"
    / "cb_svdpp"
    / "2026-05-01_ml10m_cb_svdpp_stage0_transfer_clean_multiseed_benchmark.md"
)
ML20M_INGESTION_EVIDENCE = REPO_ROOT / "docs" / "evidence" / "data" / "2026-04-24_ml20m_official_ingestion.md"
ML20M_BIASED_MF_EVIDENCE = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "models"
    / "biased_mf"
    / "2026-04-24_ml20m_biased_mf_stage0_transfer_feasibility.md"
)
ML20M_BIASED_MF_MULTISEED_EVIDENCE = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "models"
    / "biased_mf"
    / "2026-04-30_ml20m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md"
)
ML20M_CB_SVDPP_EVIDENCE = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "models"
    / "cb_svdpp"
    / "2026-04-24_ml20m_cb_svdpp_stage0_probe_e001_feasibility.md"
)
LARGE_CB_DEFERRAL_EVIDENCE = (
    REPO_ROOT / "docs" / "evidence" / "benchmarking" / "2026-04-24_large_cb_svdpp_deeper_run_deferral.md"
)
LARGE_CB_CAMPAIGN_CONTRACT = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "benchmarking"
    / "2026-04-30_large_cb_svdpp_matched_campaign_contract.md"
)
ML20M_CB_SVDPP_BUDGET_GATE = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "benchmarking"
    / "2026-05-01_ml20m_cb_svdpp_matched_campaign_budget_gate.md"
)
ML20M_CB_SVDPP_SEED1_READOUT = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "models"
    / "cb_svdpp"
    / "2026-05-01_ml20m_cb_svdpp_stage0_transfer_seed1_readout.md"
)
ML20M_CB_SVDPP_SEED2_GATE = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "benchmarking"
    / "2026-05-01_ml20m_cb_svdpp_matched_campaign_seed2_gate.md"
)
ML20M_CB_SVDPP_SEED2_READOUT = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "models"
    / "cb_svdpp"
    / "2026-05-01_ml20m_cb_svdpp_stage0_transfer_seed2_readout.md"
)
ML20M_CB_SVDPP_SEED3_GATE = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "benchmarking"
    / "2026-05-01_ml20m_cb_svdpp_matched_campaign_seed3_gate.md"
)
ML20M_CB_SVDPP_SEED3_BREACH_READOUT = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "models"
    / "cb_svdpp"
    / "2026-05-02_ml20m_cb_svdpp_stage0_transfer_seed3_guardrail_breach.md"
)
REPRODUCTION_SMOKE_EVIDENCE = (
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-01_quality_gate_reproduction.md"
)
RELEASE_HYGIENE_EVIDENCE = REPO_ROOT / "docs" / "evidence" / "release" / "2026-05-01_release_hygiene.md"
CLAIM_UNLOCK_ROADMAP = REPO_ROOT / "docs" / "roadmaps" / "2026-05-02_claim_unlock_and_scalability_plan.md"
G1_RUNTIME_PROFILE_EVIDENCE = (
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-02_runtime_profile_contract_g1.md"
)
G2_STAGE_PROFILE_EVIDENCE = REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-02_cb_stage_profile_g2.md"
G2_LARGE_STAGE_PROFILE_EVIDENCE = (
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-02_ml10m_cb_svdpp_large_stage_profile_g2.md"
)
G3_CB_SVDPP_HOTPATH_EVIDENCE = (
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-02_cb_svdpp_hotpath_g3.md"
)
G4_CLUSTER_CACHE_EVIDENCE = (
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-02_cluster_artifact_cache_g4.md"
)
G5_TUNE_INNER_CACHE_EVIDENCE = (
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-02_tune_inner_cache_controls_g5.md"
)
G6_VALIDATION_GRID_EVIDENCE = (
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-03_cb_svdpp_g6_validation_grid_run.md"
)
G6_OUTER_BENCHMARK_CONTRACT = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "reproduction"
    / "2026-05-03_cb_svdpp_g6_outer_benchmark_contract.md"
)
G6_OUTER_BENCHMARK_RUN = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "reproduction"
    / "2026-05-03_cb_svdpp_g6_outer_benchmark_run.md"
)
CB_ASVDPP_HOTPATH_DECISION = (
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-03_cb_asvdpp_hotpath_decision_g7.md"
)
CB_ASVDPP_HOTPATH_REMEDIATION_CONTRACT = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "reproduction"
    / "2026-05-03_cb_asvdpp_hotpath_remediation_contract_g8.md"
)
CB_ASVDPP_HOTPATH_PRECHANGE_BASELINE = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "reproduction"
    / "2026-05-03_cb_asvdpp_hotpath_prechange_baseline_g9.md"
)
CB_ASVDPP_HOTPATH_POSTCHANGE_BENCHMARK = (
    REPO_ROOT
    / "docs"
    / "evidence"
    / "reproduction"
    / "2026-05-03_cb_asvdpp_hotpath_postchange_benchmark_g10.md"
)


def _single_line(text: str) -> str:
    return " ".join(text.split())


def test_master_plan_contains_publish_readiness_gates_and_full_dataset_scope() -> None:
    master_plan = (REPO_ROOT / "docs" / "project_master_plan.md").read_text(encoding="utf-8")

    assert "## 14.1 Publish Readiness Gates" in master_plan
    assert "MovieLens 10M" in master_plan
    assert "MovieLens 20M" in master_plan
    assert "`ml10m` und `ml20m`" in master_plan
    assert "duerfen nicht stillschweigend aus dem finalen Scope entfernt werden" in master_plan
    assert "G6" in master_plan
    assert "validation-only Selection" in master_plan
    assert "cleanen Outer-Benchmark" in master_plan
    assert "RAM-Guardrail-Grenze" in master_plan

    for gate in (
        "Gate 1: Scope Freeze",
        "Gate 2: Dataset Evidence",
        "Gate 3: Benchmark Evidence",
        "Gate 4: Claim Freeze",
        "Gate 5: Report Ready",
        "Gate 6: Reproduction Ready",
        "Gate 7: Release Hygiene",
    ):
        assert gate in master_plan

    assert "docs/publish_readiness_matrix.md" in master_plan


def test_master_plan_tracks_claim_unlock_scalability_backlog() -> None:
    master_plan = (REPO_ROOT / "docs" / "project_master_plan.md").read_text(encoding="utf-8")
    roadmap = CLAIM_UNLOCK_ROADMAP.read_text(encoding="utf-8")

    assert "## 18. Claim-Unlock- und Skalierbarkeits-Backlog" in master_plan
    assert "docs/roadmaps/2026-05-02_claim_unlock_and_scalability_plan.md" in master_plan
    assert "Stage-Level-Profiling" in master_plan
    assert "leakage-sichere Cluster-Artefakt-Caches" in master_plan
    assert "validation-only Tuning" in master_plan
    assert "`R_star`" in master_plan
    assert "unqualifizierte Claims wie `faster`, `scalable`, `production-ready`" in master_plan

    assert "status: `active_work_queue`" in roadmap
    assert "It does not unlock any claim by itself" in roadmap
    assert "Keep" in roadmap
    assert "Change" in roadmap
    assert "Implement" in roadmap
    assert "Concrete HPC And Runtime Contract" in roadmap
    assert "evidence: `docs/evidence/reproduction/2026-05-02_runtime_profile_contract_g1.md`" in roadmap
    assert "Stage-Level CB Profiler" in roadmap
    assert "evidence: `docs/evidence/reproduction/2026-05-02_cb_stage_profile_g2.md`" in roadmap
    assert "implemented_g2_instrumentation_ml100k_and_ml10m_profile" in roadmap
    assert "docs/evidence/reproduction/2026-05-02_ml10m_cb_svdpp_large_stage_profile_g2.md" in roadmap
    assert "CB Training Hot-Path Remediation" in roadmap
    assert "implemented_g3_cb_svdpp_workbuffer_ml100k" in roadmap
    assert "docs/evidence/reproduction/2026-05-02_cb_svdpp_hotpath_g3.md" in roadmap
    assert "Algorithmic Acceleration Track" in roadmap
    assert "Leakage-Safe Cluster Artifact Cache" in roadmap
    assert "status: `implemented_g4_cluster_artifact_cache`" in roadmap
    assert "docs/evidence/reproduction/2026-05-02_cluster_artifact_cache_g4.md" in roadmap
    assert "Methodical Hyperparameter Tuning" in roadmap
    assert "implemented_g5_bounded_validation_only_selection_probe" in roadmap
    assert "docs/evidence/reproduction/2026-05-02_tune_inner_cache_controls_g5.md" in roadmap
    assert "`R_star` Decision Track" in roadmap
    assert "Do Not Do" in roadmap
    assert "Do not tune on test data" in roadmap
    assert "Do not call current CB large-dataset behavior `scalable`" in roadmap
    assert "Do not replace KMeans with MiniBatchKMeans without a new config" in roadmap
    assert "Do not claim `R_star` is optimized unless a new objective" in roadmap
    assert "`94%`" in roadmap
    assert "completed_g6_validation_only_selection" in roadmap
    assert "docs/evidence/reproduction/2026-05-03_cb_svdpp_g6_validation_grid_run.md" in roadmap
    assert "configs/models/tuned/ml100k_cb_svdpp_g6_validation_selected.yaml" in roadmap
    assert "approved_for_clean_outer_benchmark_contract" in roadmap
    assert "docs/evidence/reproduction/2026-05-03_cb_svdpp_g6_outer_benchmark_contract.md" in roadmap
    assert "completed_g6_clean_outer_benchmark" in roadmap
    assert "docs/evidence/reproduction/2026-05-03_cb_svdpp_g6_outer_benchmark_run.md" in roadmap
    assert "test RMSE mean: `0.9595668222022953`" in roadmap
    assert "pass_for_hotpath_prioritization_not_remediation" in roadmap
    assert "docs/evidence/reproduction/2026-05-03_cb_asvdpp_hotpath_decision_g7.md" in roadmap
    assert "main_training` share: about `92.44%`" in roadmap
    assert "approved_for_exact_remediation_contract" in roadmap
    assert "docs/evidence/reproduction/2026-05-03_cb_asvdpp_hotpath_remediation_contract_g8.md" in roadmap
    assert "pass_for_clean_prechange_baseline" in roadmap
    assert "docs/evidence/reproduction/2026-05-03_cb_asvdpp_hotpath_prechange_baseline_g9.md" in roadmap
    assert "main_training_wall_clock_seconds`: `122.91284980002092`" in roadmap
    assert "pass_for_exact_workbuffer_remediation_context" in roadmap
    assert "docs/evidence/reproduction/2026-05-03_cb_asvdpp_hotpath_postchange_benchmark_g10.md" in roadmap
    assert "observed main-training wall-clock change: `-7.403998780257792%`" in roadmap

    g1_evidence = G1_RUNTIME_PROFILE_EVIDENCE.read_text(encoding="utf-8")
    assert "status: `pass`" in g1_evidence
    assert "validate-runtime-profile" in g1_evidence
    assert "claim eligible: `true`" in g1_evidence
    assert "status: expected failure" in g1_evidence
    assert "Pytest: `120 passed`" in g1_evidence
    assert "no `scalable`, `production-ready`, or `publish-ready` claim" in g1_evidence

    g2_evidence = G2_STAGE_PROFILE_EVIDENCE.read_text(encoding="utf-8")
    assert "status: `pass_for_instrumentation_and_ml100k_smoke`" in g2_evidence
    assert "StageProfiler" in g2_evidence
    assert "profile_version: stage_profile_v1" in g2_evidence
    assert "git dirty: `false`" in g2_evidence
    assert "stage count: `11`" in g2_evidence
    assert "Pytest: `121 passed`" in g2_evidence
    assert "no speed claim" in g2_evidence
    assert "no large-dataset profiling claim yet" in g2_evidence

    g2_large_evidence = G2_LARGE_STAGE_PROFILE_EVIDENCE.read_text(encoding="utf-8")
    assert "status: `pass_for_bounded_large_dataset_stage_profile`" in g2_large_evidence
    assert "git dirty: `false`" in g2_large_evidence
    assert "stage count: `11`" in g2_large_evidence
    assert "`main_training`: `498.060299` seconds" in g2_large_evidence
    assert "about `73.87%`" in g2_large_evidence
    assert "Pytest: `127 passed`" in g2_large_evidence
    assert "no speed claim" in g2_large_evidence
    assert "no scalability claim" in g2_large_evidence

    g3_evidence = G3_CB_SVDPP_HOTPATH_EVIDENCE.read_text(encoding="utf-8")
    assert "status: `pass_for_cb_svdpp_workbuffer_candidate`" in g3_evidence
    assert "scalar re-read candidate" in g3_evidence
    assert "accepted workbuffer candidate" in g3_evidence
    assert "`52.348266` seconds to" in g3_evidence
    assert "`51.257735` seconds" in g3_evidence
    assert "`-2.083223%`" in g3_evidence
    assert "Pytest: `4 passed`" in g3_evidence
    assert "Pytest: `128 passed`" in g3_evidence
    assert "no broad speed claim outside this benchmark context" in g3_evidence
    assert "no `ml10m` or `ml20m` speed claim" in g3_evidence
    assert "no `cb_asvdpp` hot-path claim" in g3_evidence

    g4_evidence = G4_CLUSTER_CACHE_EVIDENCE.read_text(encoding="utf-8")
    assert "status: `pass_for_synthetic_cache_gate`" in g4_evidence
    assert "artifacts/local/cb_clusters" in g4_evidence
    assert "cache `miss -> hit`" in g4_evidence
    assert "train-fingerprint invalidation" in g4_evidence
    assert "non-train rating isolation" in g4_evidence
    assert "Pytest: `127 passed`" in g4_evidence
    assert "no speed claim" in g4_evidence
    assert "no scalability claim" in g4_evidence

    g5_evidence = G5_TUNE_INNER_CACHE_EVIDENCE.read_text(encoding="utf-8")
    assert "status: `pass_for_bounded_validation_only_selection_probe`" in g5_evidence
    assert "git dirty: `false`" in g5_evidence
    assert "test_metrics_available`: `false`" in g5_evidence
    assert "test_rmse`: `None`" in g5_evidence
    assert "rank032_uc064_ic064_a000_lr0100_reg0020_e002" in g5_evidence
    assert "validation RMSE mean: `0.960717726`" in g5_evidence
    assert "Cache hits and invalidations are manifest-visible" in g5_evidence
    assert "Pytest: `13 passed`" in g5_evidence
    assert "Pytest: `131 passed`" in g5_evidence
    assert "no final `ml100k` quality claim" in g5_evidence
    assert "no `ml10m` or `ml20m` tuning claim" in g5_evidence

    g6_evidence = G6_VALIDATION_GRID_EVIDENCE.read_text(encoding="utf-8")
    assert "status: `pass_for_validation_only_selection`" in g6_evidence
    assert "git dirty: `false`" in g6_evidence
    assert "candidate run count: `36`" in g6_evidence
    assert "non-null `test_rmse` count across candidate metrics: `0`" in g6_evidence
    assert "rank032_uc100_ic100_a0000_lr0100_reg0020_e002" in g6_evidence
    assert "validation RMSE mean: `0.9566122815305916`" in g6_evidence
    assert "configs/models/tuned/ml100k_cb_svdpp_g6_validation_selected.yaml" in g6_evidence
    assert "no final `ml100k cb_svdpp` quality claim" in g6_evidence
    assert "no test-set result or test-set comparison" in g6_evidence

    g6_outer_contract = G6_OUTER_BENCHMARK_CONTRACT.read_text(encoding="utf-8")
    assert "status: `approved_for_clean_outer_benchmark_contract`" in g6_outer_contract
    assert "run contract, not a benchmark result" in g6_outer_contract
    assert "selected by `validation_rmse_mean`" in g6_outer_contract
    assert "`0` non-null `test_rmse` values" in g6_outer_contract
    assert "contract must already be committed before the first outer run starts" in g6_outer_contract
    assert "same git commit as the aggregation process" in g6_outer_contract
    assert "the outer run has not been executed yet" in g6_outer_contract

    g6_outer_run = G6_OUTER_BENCHMARK_RUN.read_text(encoding="utf-8")
    assert "status: `pass_for_clean_outer_benchmark`" in g6_outer_run
    assert "git dirty: `false`" in g6_outer_run
    assert "test RMSE | `0.9595668222022953`" in g6_outer_run
    assert "validation RMSE | `0.9566122815305916`" in g6_outer_run
    assert "SOTA, scalability, speed" in g6_outer_run

    cb_asvdpp_hotpath_decision = CB_ASVDPP_HOTPATH_DECISION.read_text(encoding="utf-8")
    assert "status: `pass_for_hotpath_prioritization_not_remediation`" in cb_asvdpp_hotpath_decision
    assert "main_training` | `115.10358980001183` | `92.44%`" in cb_asvdpp_hotpath_decision
    assert "no claim that a remediation has already improved runtime" in cb_asvdpp_hotpath_decision

    cb_asvdpp_remediation_contract = CB_ASVDPP_HOTPATH_REMEDIATION_CONTRACT.read_text(encoding="utf-8")
    assert "status: `approved_for_exact_remediation_contract`" in cb_asvdpp_remediation_contract
    assert "fresh pre-change baseline" in cb_asvdpp_remediation_contract
    assert "main_training_wall_clock_seconds` must decrease by at least `1.0%`" in cb_asvdpp_remediation_contract

    cb_asvdpp_prechange_baseline = CB_ASVDPP_HOTPATH_PRECHANGE_BASELINE.read_text(encoding="utf-8")
    assert "status: `pass_for_clean_prechange_baseline`" in cb_asvdpp_prechange_baseline
    assert "main training wall-clock seconds: `122.91284980002092`" in cb_asvdpp_prechange_baseline
    assert "no claim that a remediation has already improved runtime" in cb_asvdpp_prechange_baseline

    cb_asvdpp_postchange_benchmark = CB_ASVDPP_HOTPATH_POSTCHANGE_BENCHMARK.read_text(encoding="utf-8")
    assert "status: `pass_for_exact_workbuffer_remediation_context`" in cb_asvdpp_postchange_benchmark
    assert "main training wall-clock seconds: `113.81238390004728`" in cb_asvdpp_postchange_benchmark
    assert "`-7.403998780257792%`" in cb_asvdpp_postchange_benchmark
    assert "no broad or unqualified speed claim" in cb_asvdpp_postchange_benchmark


def test_publish_readiness_matrix_tracks_gates_scope_and_current_blockers() -> None:
    matrix = READINESS_MATRIX.read_text(encoding="utf-8")
    report = (REPO_ROOT / "docs" / "report" / "project_report.md").read_text(encoding="utf-8")

    assert "status: `release_candidate_claim_limited`" in matrix
    assert "Gate 1: Scope Freeze" in matrix
    assert "Gate 7: Release Hygiene" in matrix
    assert "| Gate 4: Claim Freeze | `pass` |" in matrix
    assert "| Gate 5: Report Ready | `pass` |" in matrix
    assert "| Gate 6: Reproduction Ready | `pass` |" in matrix
    assert "| Gate 7: Release Hygiene | `pass` |" in matrix
    assert "## Final Claim Matrix" in matrix
    assert "## Feasibility And Selection Evidence" in matrix
    assert (
        "| `ml100k` | `in_scope` | `pass` | `pass` | "
        "`pass_for_current_anchor_set_plus_g6_outer_benchmark` |"
    ) in matrix
    assert "| `ml10m` | `in_scope` | `pass` | `pass` | `matched_biased_mf_cb_svdpp_anchor` |" in matrix
    assert (
        "| `ml20m` | `in_scope` | `pass` | `pass` | "
        "`partial_baseline_anchor_plus_cb_negative_resource_evidence` |"
    ) in matrix
    assert "`ml100k` | `cb_asvdpp stage1_tuned`" in matrix
    assert "test RMSE mean `0.916839`" in matrix
    assert "clean_multiseed_anchor_plus_g10_context_speed_readout" in matrix
    assert "clean post-change work-buffer benchmark" in matrix
    assert "`ml100k` | `cb_svdpp g6_validation_selected`" in matrix
    assert "selected validation RMSE mean `0.9566122815305916`" in matrix
    assert "test RMSE mean `0.9595668222022953`" in matrix
    assert "non-null `test_rmse` count `0`" in matrix
    assert "Not itself a benchmark anchor" in matrix
    assert "`ml1m` | `cb_svdpp stage0_transfer`" in matrix
    assert "test RMSE mean `0.857365`" in matrix
    assert "`ml10m` | `biased_mf stage0_transfer`" in matrix
    assert "test RMSE mean `0.787738`" in matrix
    assert "clean_multiseed_baseline_anchor" in matrix
    assert "`ml20m` | `biased_mf stage0_transfer`" in matrix
    assert "test RMSE mean `0.775803`" in matrix
    assert "`ml20m` | `cb_svdpp stage0_probe_e001`" in matrix
    assert "validation RMSE `0.863001`" in matrix
    assert "matched clean `biased_mf` and `cb_svdpp` multi-split-seed anchors" in matrix
    assert "docs/evidence/benchmarking/2026-04-24_large_cb_svdpp_deeper_run_deferral.md" in matrix
    assert "docs/evidence/benchmarking/2026-04-30_large_cb_svdpp_matched_campaign_contract.md" in matrix
    assert "Further local `ml20m cb_svdpp` promotion attempts require" in matrix
    assert "`uv sync --extra dev --locked` completed" in matrix
    assert "Ruff, Mypy, focused regression tests, and the full test suite pass" in matrix
    assert "full test suite pass from the `uv` environment" in matrix
    assert "Keep `uv.lock` versioned" in matrix
    assert "release marker `submission-2026-05-02-r10`" in matrix
    assert "Post-Release Work Queue" in matrix
    assert "frozen G6-selected two-epoch profile" in matrix
    assert "completed clean outer benchmark readout under `benchmark_random_v1`" in matrix
    assert "do not reintroduce chronological work-log sections" in matrix
    assert "`ml1m cb_asvdpp` is not a benchmark anchor" in matrix
    assert "G6 `ml100k cb_svdpp` is validation-only selection evidence" in matrix
    assert "No final `ml20m` model-comparison claim is allowed" in matrix
    assert "### 7.1 Clean Benchmark Anchors" in report
    assert "### 7.2 Feasibility, Selection, And Deferral Evidence" in report
    assert "large-dataset `cb_svdpp` run boundary" in report
    assert "`16.54h` before overhead" in _single_line(report)
    assert "### 7.0" not in report
    assert "This section should" not in report
    assert "docs/evidence/data/2026-04-24_ml10m_processed_ingestion.md" in matrix
    assert (
        "docs/evidence/models/biased_mf/2026-04-30_ml10m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md"
        in matrix
    )
    assert "docs/evidence/models/cb_svdpp/2026-04-24_ml10m_cb_svdpp_stage0_probe_e001_feasibility.md" in matrix
    assert (
        "docs/evidence/models/cb_svdpp/2026-05-01_ml10m_cb_svdpp_stage0_transfer_clean_multiseed_benchmark.md"
        in matrix
    )
    assert "docs/evidence/data/2026-04-24_ml20m_official_ingestion.md" in matrix
    assert (
        "docs/evidence/models/biased_mf/2026-04-30_ml20m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md"
        in matrix
    )
    assert "docs/evidence/models/cb_svdpp/2026-04-24_ml20m_cb_svdpp_stage0_probe_e001_feasibility.md" in matrix
    assert "docs/evidence/models/cb_svdpp/2026-05-02_ml20m_cb_svdpp_stage0_transfer_seed3_guardrail_breach.md" in matrix

    raw_evidence = ML10M_RAW_EVIDENCE.read_text(encoding="utf-8")
    assert "ratings rows:" in raw_evidence
    assert "`10000054`" in raw_evidence

    processed_evidence = ML10M_PROCESSED_EVIDENCE.read_text(encoding="utf-8")
    assert "processed dataset manifest:" in processed_evidence
    assert "interactions:" in processed_evidence
    assert "`10000054`" in processed_evidence
    assert "Do not make model or benchmark claims on `ml10m`" in processed_evidence

    biased_mf_evidence = ML10M_BIASED_MF_EVIDENCE.read_text(encoding="utf-8")
    assert "single-seed feasibility baseline" in biased_mf_evidence
    assert "git_dirty: `false`" in biased_mf_evidence
    assert "validation RMSE:" in biased_mf_evidence
    assert "`0.786906`" in biased_mf_evidence
    assert "not a final benchmark anchor" in biased_mf_evidence

    ml10m_biased_mf_multiseed = ML10M_BIASED_MF_MULTISEED_EVIDENCE.read_text(encoding="utf-8")
    assert "clean three-split-seed `ml10m biased_mf` baseline anchor" in ml10m_biased_mf_multiseed
    assert "git_commit: `bbe5f816a3fdab6757fb5ba8a457ad7389b32cde`" in ml10m_biased_mf_multiseed
    assert "validation RMSE mean:" in ml10m_biased_mf_multiseed
    assert "`0.787190`" in ml10m_biased_mf_multiseed
    assert "test RMSE mean:" in ml10m_biased_mf_multiseed
    assert "`0.787738`" in ml10m_biased_mf_multiseed
    assert "not a final `ml10m` model-comparison claim" in ml10m_biased_mf_multiseed

    cb_svdpp_evidence = ML10M_CB_SVDPP_EVIDENCE.read_text(encoding="utf-8")
    assert "single-epoch clustering-model feasibility probe" in cb_svdpp_evidence
    assert "git_dirty: `false`" in cb_svdpp_evidence
    assert "validation RMSE:" in cb_svdpp_evidence
    assert "`0.872094`" in cb_svdpp_evidence
    assert "not a final benchmark anchor" in cb_svdpp_evidence

    ml10m_cb_svdpp_multiseed = ML10M_CB_SVDPP_MULTISEED_EVIDENCE.read_text(encoding="utf-8")
    assert "clean three-split-seed `ml10m cb_svdpp` benchmark anchor" in ml10m_cb_svdpp_multiseed
    assert "git_commit: `b70904985d1b84bad5c3ef6d0b69592a0b4fa8b0`" in ml10m_cb_svdpp_multiseed
    assert "validation RMSE mean:" in ml10m_cb_svdpp_multiseed
    assert "`0.790782`" in ml10m_cb_svdpp_multiseed
    assert "test RMSE mean:" in ml10m_cb_svdpp_multiseed
    assert "`0.791315`" in ml10m_cb_svdpp_multiseed
    assert "higher validation RMSE, higher test RMSE" in ml10m_cb_svdpp_multiseed
    assert "not evidence for `ml20m`" in ml10m_cb_svdpp_multiseed

    ml20m_evidence = ML20M_INGESTION_EVIDENCE.read_text(encoding="utf-8")
    assert "Official ingestion of `MovieLens 20M`" in ml20m_evidence
    assert "MD5 verification:" in ml20m_evidence
    assert "`pass`" in ml20m_evidence
    assert "processed interactions:" in ml20m_evidence
    assert "`20000263`" in ml20m_evidence
    assert "Do not make model or benchmark claims on `ml20m`" in ml20m_evidence

    ml20m_biased_mf_evidence = ML20M_BIASED_MF_EVIDENCE.read_text(encoding="utf-8")
    assert "single-seed feasibility baseline" in ml20m_biased_mf_evidence
    assert "git_dirty: `false`" in ml20m_biased_mf_evidence
    assert "validation RMSE:" in ml20m_biased_mf_evidence
    assert "`0.774734`" in ml20m_biased_mf_evidence
    assert "not a final benchmark anchor" in ml20m_biased_mf_evidence

    ml20m_biased_mf_multiseed = ML20M_BIASED_MF_MULTISEED_EVIDENCE.read_text(encoding="utf-8")
    assert "clean three-split-seed `ml20m biased_mf` baseline anchor" in ml20m_biased_mf_multiseed
    assert "git_commit: `e9ce60ed0ff895f14a1f899e966a5b724c8f54c1`" in ml20m_biased_mf_multiseed
    assert "validation RMSE mean:" in ml20m_biased_mf_multiseed
    assert "`0.775339`" in ml20m_biased_mf_multiseed
    assert "test RMSE mean:" in ml20m_biased_mf_multiseed
    assert "`0.775803`" in ml20m_biased_mf_multiseed
    assert "not a final `ml20m` model-comparison claim" in ml20m_biased_mf_multiseed

    ml20m_cb_svdpp_evidence = ML20M_CB_SVDPP_EVIDENCE.read_text(encoding="utf-8")
    assert "single-epoch clustering-model feasibility probe" in ml20m_cb_svdpp_evidence
    assert "git_dirty: `false`" in ml20m_cb_svdpp_evidence
    assert "validation RMSE:" in ml20m_cb_svdpp_evidence
    assert "`0.863001`" in ml20m_cb_svdpp_evidence
    assert "not a final benchmark anchor" in ml20m_cb_svdpp_evidence

    deferral_evidence = LARGE_CB_DEFERRAL_EVIDENCE.read_text(encoding="utf-8")
    assert "defer deeper local `ml10m` and `ml20m` `cb_svdpp` runs" in deferral_evidence
    assert "`ml10m`: `78.425572 + 20 * 422.423619 = 8526.898` seconds" in deferral_evidence
    assert "`ml20m`: `195.149008 + 20 * 983.076083 = 19856.671` seconds" in deferral_evidence
    assert "This extrapolation is a planning estimate, not a benchmark result" in deferral_evidence
    assert "Keep all final `ml10m` and `ml20m` model-comparison claims blocked" in deferral_evidence

    campaign_contract = LARGE_CB_CAMPAIGN_CONTRACT.read_text(encoding="utf-8")
    assert "status: `accepted`" in campaign_contract
    assert "This is a run contract, not a benchmark result" in campaign_contract
    assert "split seeds:" in campaign_contract
    assert "`1,2,3`" in campaign_contract
    assert "model seed:" in campaign_contract
    assert "`1`" in campaign_contract
    assert "`7.11h` before overhead" in campaign_contract
    assert "`16.54h` before overhead" in campaign_contract
    assert "Do not start a local `ml20m cb_svdpp` matched campaign" in campaign_contract
    assert "Keep final `ml20m` model-comparison claims blocked" in campaign_contract
    assert "2026-05-01_ml20m_cb_svdpp_matched_campaign_budget_gate.md" in campaign_contract
    assert "2026-05-01_ml20m_cb_svdpp_matched_campaign_seed2_gate.md" in campaign_contract
    assert "2026-05-01_ml20m_cb_svdpp_matched_campaign_seed3_gate.md" in campaign_contract
    assert "2026-05-02_ml20m_cb_svdpp_stage0_transfer_seed3_guardrail_breach.md" in campaign_contract
    assert "negative resource evidence" in campaign_contract

    budget_gate = ML20M_CB_SVDPP_BUDGET_GATE.read_text(encoding="utf-8")
    assert "status: `approved_for_split_seed_1_only`" in budget_gate
    assert "This is a budget gate and run authorization note, not a benchmark result" in budget_gate
    assert "Split seeds `2` and `3` are not automatically authorized" in budget_gate
    assert "does not unlock any final `ml20m` model-comparison claim" in budget_gate

    seed1_readout = ML20M_CB_SVDPP_SEED1_READOUT.read_text(encoding="utf-8")
    assert "status: `completed_single_split_seed`" in seed1_readout
    assert "`0.780558`" in seed1_readout
    assert "`0.781255`" in seed1_readout
    assert "`18586.136719`" in seed1_readout
    assert "does not unlock a final `ml20m` model-comparison claim" in seed1_readout

    seed2_gate = ML20M_CB_SVDPP_SEED2_GATE.read_text(encoding="utf-8")
    assert "status: `approved_for_split_seed_2_only`" in seed2_gate
    assert "Split seed `3` is not automatically authorized" in seed2_gate

    seed2_readout = ML20M_CB_SVDPP_SEED2_READOUT.read_text(encoding="utf-8")
    assert "status: `completed_single_split_seed`" in seed2_readout
    assert "`0.781702`" in seed2_readout
    assert "`0.781773`" in seed2_readout
    assert "`19128.371094`" in seed2_readout
    assert "does not unlock a final `ml20m` model-comparison claim" in seed2_readout

    seed3_gate = ML20M_CB_SVDPP_SEED3_GATE.read_text(encoding="utf-8")
    assert "status: `approved_for_split_seed_3_only`" in seed3_gate
    assert "final single-split gate" in seed3_gate

    seed3_readout = ML20M_CB_SVDPP_SEED3_BREACH_READOUT.read_text(encoding="utf-8")
    assert "status: `completed_guardrail_breach_negative_evidence`" in seed3_readout
    assert "`0.781010`" in seed3_readout
    assert "`0.781511`" in seed3_readout
    assert "`19898.871094`" in seed3_readout
    assert "exceeding the guardrail by approximately `251.58 MB`" in seed3_readout
    assert "does not unlock a final `ml20m` model-comparison claim" in seed3_readout

    reproduction_smoke_evidence = REPRODUCTION_SMOKE_EVIDENCE.read_text(encoding="utf-8")
    assert "status: `pass`" in reproduction_smoke_evidence
    assert "release_marker: `submission-2026-05-01-r9`" in reproduction_smoke_evidence
    assert "`uv 0.11.7`" in reproduction_smoke_evidence
    assert "`All checks passed!`" in reproduction_smoke_evidence
    assert "`Success: no issues found in 60 source files`" in reproduction_smoke_evidence
    assert "`9 passed" in reproduction_smoke_evidence
    assert "`111 passed" in reproduction_smoke_evidence
    assert "does not unlock a final" in reproduction_smoke_evidence

    release_hygiene_evidence = RELEASE_HYGIENE_EVIDENCE.read_text(encoding="utf-8")
    assert "status: `pass`" in release_hygiene_evidence
    assert "release_marker: `submission-2026-05-01-r9`" in release_hygiene_evidence
    assert "Evidence links" in release_hygiene_evidence
    assert "README states the claim-limited release status" in release_hygiene_evidence
    assert "matched-campaign contract is linked from" in release_hygiene_evidence
    assert "Mark Gate 7 as `pass`" in release_hygiene_evidence
