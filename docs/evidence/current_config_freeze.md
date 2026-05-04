# Current Config Freeze

- date: `2026-05-05`
- status: `config_freeze_documented_not_a_run`
- git_commit: `cc45604da5d95e86a0b01b96ee49cfb6a13c3678`
- git_dirty_at_documentation: `true`

This file freezes the active configuration basis after the Unified Framework
hygiene and tuning-protocol documentation work. It is a repository navigation
and claim-boundary document. It does not introduce new benchmark results,
quality claims, speed claims, scalability claims, or paper-faithfulness claims.

The dirty status records that this freeze is being written in the same working
tree as the documentation and guardrail changes. It is not a benchmark-reuse
approval.

## Active Model Profiles

Canonical base profiles:

- `configs/models/biased_mf.yaml`
- `configs/models/svdpp.yaml`
- `configs/models/asymmetric_svd.yaml`
- `configs/models/asvdpp.yaml`
- `configs/models/cb_svdpp.yaml`
- `configs/models/cb_asvdpp.yaml`

Selected visible profiles:

- `configs/models/selected/ml100k/ml100k_biased_mf_stage1.yaml`
- `configs/models/selected/ml100k/ml100k_svdpp_stage1.yaml`
- `configs/models/selected/ml100k/ml100k_cb_svdpp_stage1.yaml`
- `configs/models/selected/ml100k/ml100k_cb_asvdpp_stage1.yaml`
- `configs/models/selected/ml100k/ml100k_cb_svdpp_g6_validation_selected.yaml`
- `configs/models/selected/ml1m/ml1m_biased_mf_stage0_transfer.yaml`
- `configs/models/selected/ml1m/ml1m_cb_svdpp_stage0_transfer.yaml`
- `configs/models/selected/ml1m/ml1m_cb_asvdpp_stage0_transfer.yaml`
- `configs/models/selected/ml10m/ml10m_biased_mf_stage0_transfer.yaml`
- `configs/models/selected/ml10m/ml10m_cb_svdpp_stage0_transfer.yaml`
- `configs/models/selected/ml20m/ml20m_biased_mf_stage0_transfer.yaml`

There is no active selected `ml20m cb_svdpp` profile.
`ml20m cb_svdpp` remains archived/blocked negative-resource evidence until a
separate lower-memory validation reassessment and clean outer benchmark contract
exist.

## Active Tuning Configs

Active tuning configs live only under `configs/experiments/tuning/active/`:

- `configs/experiments/tuning/active/ml100k_cb_svdpp_g6_validation_grid.yaml`
- `configs/experiments/tuning/active/ml1m_cb_svdpp_stage0.yaml`
- `configs/experiments/tuning/active/ml1m_cb_asvdpp_stage0.yaml`
- `configs/experiments/tuning/active/ml20m_cb_svdpp_g11_lower_memory_validation_grid.yaml`

These configs must load through the generic active tuning config validation
test. Their `base_model_config` fields must point to canonical base profiles or
selected profiles, not archived profiles.

The `ml100k` G6 tuning config was motivated by the archived G5 selection probe:

- archived config:
  `configs/experiments/tuning/archive/ml100k_cb_svdpp_g5_bounded_alpha_cluster.yaml`
- current evidence:
  `docs/evidence/reproduction/current/2026-05-02_tune_inner_cache_controls_g5.md`
- promoted candidate:
  `rank032_uc064_ic064_a000_lr0100_reg0020_e002`

This provenance is intentionally documented here instead of being a load path in
the active YAML.

## Archived Profiles

Archived model profiles:

- `configs/models/archive/tuned/ml10m_cb_svdpp_stage0_probe_e001.yaml`
- `configs/models/archive/tuned/ml1m_cb_asvdpp_stage0_probe_e003.yaml`
- `configs/models/archive/tuned/ml1m_cb_svdpp_stage0_probe_e003.yaml`
- `configs/models/archive/tuned/ml20m_cb_svdpp_stage0_probe_e001.yaml`
- `configs/models/archive/tuned/ml20m_cb_svdpp_stage0_transfer.yaml`
- `configs/models/archive/development/ml100k_cb_svdpp_stage_profile_smoke.yaml`

Archived tuning configs live under:

- `configs/experiments/tuning/archive/`

Archived configs may be referenced by historical evidence as original
provenance. They are not active defaults and must not be used as selected
profiles without a new documented promotion decision.

## Claim Boundary

Claim-relevant config use is limited to:

- active base profiles
- selected profiles
- active tuning configs for validation-only selection
- clean outer benchmark configs after a separate benchmark contract

Not claim-relevant by itself:

- archived model profiles
- archived tuning configs
- single tuning candidates
- validation-only selection summaries
- `alpha=0` ablation candidates
- resource-gate failures
- development-only smoke profiles

`alpha > 0` only activates the cluster channel. It does not make a run
`cb_claim_eligible` without the manifest and evidence requirements defined by
the Unified Framework.

## Resource Constraints

Default local resource profile:

- device profile: `local_i5_2500k_24gb`
- RAM guardrail for constrained large-dataset reassessment: `80%`
- ML20M CB-SVD++ lower-memory reassessment max peak memory:
  `19660.8 MB`
- GPU is not part of the default training path
- cache controls must remain explicit in execution commands and summaries

Resource-gated tuning candidates must pass the configured memory guardrail
before validation RMSE can be used for selection. A resource failure is
negative/resource evidence, not a model-quality comparison.
