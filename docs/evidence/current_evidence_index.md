# Current Evidence Index

This index is the current navigation point for evidence. The reproduction files
were reorganized on 2026-05-04 into `current/` and `archive/` directories. Historical
commands and original path strings inside evidence notes remain historical
execution provenance; external references should use the paths listed here.

## Current Reproduction Evidence

- Controlled run indexes:
  `docs/evidence/runs/ml100k_run_index.md`,
  `docs/evidence/runs/ml1m_run_index.md`,
  `docs/evidence/runs/ml10m_run_index.md`,
  `docs/evidence/runs/ml20m_run_index.md`
- Public clean import:
  `docs/evidence/reproduction/current/2026-05-02_public_clean_import.md`
- Public path hygiene:
  `docs/evidence/reproduction/current/2026-05-03_public_path_hygiene.md`
- Runtime profile contract:
  `docs/evidence/reproduction/current/2026-05-02_runtime_profile_contract_g1.md`
- CB stage profiling:
  `docs/evidence/reproduction/current/2026-05-02_cb_stage_profile_g2.md`
- ML10M CB-SVD++ large stage profile:
  `docs/evidence/reproduction/current/2026-05-02_ml10m_cb_svdpp_large_stage_profile_g2.md`
- CB-SVD++ hotpath:
  `docs/evidence/reproduction/current/2026-05-02_cb_svdpp_hotpath_g3.md`
- Cluster artifact cache:
  `docs/evidence/reproduction/current/2026-05-02_cluster_artifact_cache_g4.md`
- Inner tuning cache controls:
  `docs/evidence/reproduction/current/2026-05-02_tune_inner_cache_controls_g5.md`
- Historical release evidence: CB-SVD++ G6 validation-grid contract:
  `docs/evidence/reproduction/current/2026-05-03_cb_svdpp_g6_validation_grid_contract.md`
- Historical release evidence: CB-SVD++ G6 validation-grid run:
  `docs/evidence/reproduction/current/2026-05-03_cb_svdpp_g6_validation_grid_run.md`
- Historical release evidence: CB-SVD++ G6 outer benchmark contract:
  `docs/evidence/reproduction/current/2026-05-03_cb_svdpp_g6_outer_benchmark_contract.md`
- Historical release evidence: CB-SVD++ G6 outer benchmark run:
  `docs/evidence/reproduction/current/2026-05-03_cb_svdpp_g6_outer_benchmark_run.md`
- CB-ASVD++ hotpath decision:
  `docs/evidence/reproduction/current/2026-05-03_cb_asvdpp_hotpath_decision_g7.md`
- CB-ASVD++ hotpath remediation contract:
  `docs/evidence/reproduction/current/2026-05-03_cb_asvdpp_hotpath_remediation_contract_g8.md`
- CB-ASVD++ hotpath pre-change baseline:
  `docs/evidence/reproduction/current/2026-05-03_cb_asvdpp_hotpath_prechange_baseline_g9.md`
- CB-ASVD++ hotpath post-change benchmark:
  `docs/evidence/reproduction/current/2026-05-03_cb_asvdpp_hotpath_postchange_benchmark_g10.md`

## Archived Or Negative Evidence

- Archived/resource evidence: ML20M CB-SVD++ G11 lower-memory validation contract:
  `docs/evidence/reproduction/current/2026-05-03_ml20m_cb_svdpp_g11_lower_memory_validation_contract.md`

  This is archived/blocked negative resource evidence for `ml20m cb_svdpp`.
  It is not an active selected profile and does not establish a final ML20M
  model-comparison claim.

- Earlier reproduction quality gates:
  `docs/evidence/reproduction/archive/2026-04-24_current_main_reproduction_smoke.md`,
  `docs/evidence/reproduction/archive/2026-04-30_quality_gate_reproduction.md`,
  `docs/evidence/reproduction/archive/2026-05-01_quality_gate_reproduction.md`

## Config Locations

- Base model profiles:
  `configs/models/biased_mf.yaml`,
  `configs/models/svdpp.yaml`,
  `configs/models/asymmetric_svd.yaml`,
  `configs/models/asvdpp.yaml`,
  `configs/models/cb_svdpp.yaml`,
  `configs/models/cb_asvdpp.yaml`
- Selected profiles:
  `configs/models/selected/`
- Archived tuned, probe, blocked, and negative-resource profiles:
  `configs/models/archive/tuned/`
- Archived development-only smoke profiles:
  `configs/models/archive/development/`
- Active tuning configs:
  `configs/experiments/tuning/active/`
- Tuning templates:
  `configs/experiments/tuning/templates/`
- Archived tuning configs:
  `configs/experiments/tuning/archive/`

## Migration Notes

- Archived original path `configs/models/tuned/ml20m_cb_svdpp_stage0_transfer.yaml` moved to
  `configs/models/archive/tuned/ml20m_cb_svdpp_stage0_transfer.yaml` as
  blocked negative-resource evidence, not as an active selected profile.
- `configs/models/development/ml100k_cb_svdpp_stage_profile_smoke.yaml` moved
  to `configs/models/archive/development/ml100k_cb_svdpp_stage_profile_smoke.yaml`
  as development-only stage-profiling evidence, not as an active selected
  profile.
- Current references to moved selected profiles should use
  `configs/models/selected/<dataset>/`.
- Historical evidence notes may still contain original command strings with
  pre-migration paths. Those strings describe how the historical run was
  recorded and should not be rewritten as if the run happened under the new
  directory structure.
