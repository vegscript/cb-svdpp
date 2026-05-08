# Exact Residual Reuse Validation V1

## Branch

`exact-residual-reuse-validation-v1`

## Validation Goal

Validate the Step 16b exact residual reuse change on the local laptop ML1M
benchmark context.

The validation compares local before artifacts from the last pre-16b laptop
baseline against after runs produced from the validation branch. The goal is to
check contract consistency, quality stability, runtime readouts, and control
model noise before making an acceptance decision for the narrow exact
implementation slice.

No new kernel optimization is part of this validation step.

## Before Artifact Inventory

Source evidence:

- `docs/evidence/performance/2026-05-08_ml1m_laptop_u300_benchmark_readout.md`

Local report directory:

- `artifacts/reports/ml1m_laptop_u300_2026-05-08/`

Run index:

- `artifacts/reports/ml1m_laptop_u300_2026-05-08/ml1m_laptop_u300_six_model_run_index.csv`
- `artifacts/reports/ml1m_laptop_u300_2026-05-08/ml1m_laptop_u300_six_model_run_index.json`

Common before-run provenance:

- branch: `residual-history-duplication-audit-v1`
- commit: `9bc7d1b9d1b66275e8e64396697178e1e1039c36`
- dirty: `false`
- dataset: `ml1m`
- split family: `benchmark_random_v1`
- split seed: `1`
- model seed: `1`
- dtype: `float32`
- device profile: `local_u300_24gb`
- processed manifest:
  `data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json`

| Model | Role | Before run id | Epochs | Latent dim | Cache policy/status |
| --- | --- | --- | ---: | ---: | --- |
| `asymmetric_svd` | target | `2026-05-08T005445Z_ml1m_asymmetric_svd_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | 20 | 50 | split=`hit`; training_user_history=`hit`; training_explicit_feedback=`miss`; cluster cache disabled |
| `asvdpp` | target | `2026-05-08T010531Z_ml1m_asvdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | 20 | 50 | split=`hit`; training_user_history=`hit`; training_explicit_feedback=`hit`; cluster cache disabled |
| `cb_asvdpp` | target | `2026-05-08T011957Z_ml1m_cb_asvdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | 20 | 64 | split=`hit`; training_user_history=`hit`; training_explicit_feedback=`hit`; cluster_artifacts=`hit` |
| `biased_mf` | control | `2026-05-08T004817Z_ml1m_biased_mf_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | 25 | 64 | split=`miss`; training index disabled; cluster cache disabled |
| `svdpp` | control | `2026-05-08T004906Z_ml1m_svdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | 20 | 50 | split=`hit`; training_user_history=`miss`; cluster cache disabled |
| `cb_svdpp` | control | `2026-05-08T011336Z_ml1m_cb_svdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | 20 | 64 | split=`hit`; training_user_history=`hit`; cluster_artifacts=`miss` |

Each before run has these artifacts:

- `artifacts/runs/<run_id>/metrics.json`
- `artifacts/runs/<run_id>/performance_profile.json`
- `artifacts/runs/<run_id>/kernel_profile.json`
- `artifacts/runs/<run_id>/run_manifest.json`

No required before run is missing.

## After Run Plan

Run after benchmarks on the current validation branch for the same six models:

- target models:
  - `asymmetric_svd`
  - `asvdpp`
  - `cb_asvdpp`
- control models:
  - `biased_mf`
  - `svdpp`
  - `cb_svdpp`

Use the same command shape as the before runs, preserving:

- processed manifest
- model config path
- runtime config path
- device config path
- split family and ratios
- split seed
- model seed
- split cache setting
- training index cache setting
- cluster artifact cache setting

After artifacts should be collected into a new local report directory, for
example:

- `artifacts/reports/exact_residual_reuse_validation_v1/`

The report directory should contain at least:

- run index CSV
- run index JSON
- performance stage breakdown CSV
- performance hotspots CSV
- kernel cost anatomy CSV

The raw run directories remain under:

- `artifacts/runs/<after_run_id>/`

## Same-Contract Checks

The acceptance comparison requires these fields to match before and after:

| Field | Required value/check | Handling if unavailable |
| --- | --- | --- |
| dataset | `ml1m` | If missing from a manifest, infer only from run id and processed manifest path, and mark as `not available in manifest`. |
| model | identical before/after | Required. A mismatch invalidates that pair. |
| split_family | identical before/after | Required. A mismatch invalidates that pair. |
| split_seed | identical before/after | Required. A mismatch invalidates that pair. |
| model_seed | identical before/after | Required. A mismatch invalidates that pair. |
| dtype | identical before/after | Required when readable from `run_manifest.json` or `config_snapshot.yaml`; otherwise mark as unavailable and inspect config snapshot. |
| epochs | identical before/after | Required. A mismatch invalidates runtime comparison for that pair. |
| latent_dim | identical before/after | Required. A mismatch invalidates runtime comparison for that pair. |
| effective model config | identical before/after | Compare config path and, where possible, selected effective fields from `config_snapshot.yaml`. If a full structural diff is unavailable, document the available evidence and residual risk. |
| cache policy | identical, or explicitly documented | Cache hits/misses can differ due local state. Command-level cache policy must match. Hit/miss differences must be documented and kept out of broad runtime claims. |
| device profile | local laptop profile `local_u300_24gb` | Required. A different device profile invalidates the laptop before/after comparison. |

Additional consistency checks:

- processed manifest path should match
- train rows should match
- validation and test rows should match when available
- `kernel_profile.profile_version` should match
- `estimated_factor_touches` should match for each model
- `implicit_history_visits`, `explicit_history_visits`, and
  `cluster_history_visits` should match for each model where present
- `git.dirty` should be recorded for both before and after

## Target Model Results

Pending after-run generation.

Expected target models:

| Model | Before run id | After run id | Same-contract status | Quality status | Runtime readout |
| --- | --- | --- | --- | --- | --- |
| `asymmetric_svd` | `2026-05-08T005445Z_ml1m_asymmetric_svd_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | pending | pending | pending | pending |
| `asvdpp` | `2026-05-08T010531Z_ml1m_asvdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | pending | pending | pending | pending |
| `cb_asvdpp` | `2026-05-08T011957Z_ml1m_cb_asvdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | pending | pending | pending | pending |

## Control Model Results

Pending after-run generation.

Expected control models:

| Model | Before run id | After run id | Same-contract status | Quality status | Runtime readout |
| --- | --- | --- | --- | --- | --- |
| `biased_mf` | `2026-05-08T004817Z_ml1m_biased_mf_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | pending | pending | pending | pending |
| `svdpp` | `2026-05-08T004906Z_ml1m_svdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | pending | pending | pending | pending |
| `cb_svdpp` | `2026-05-08T011336Z_ml1m_cb_svdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | pending | pending | pending | pending |

## Quality Deltas

Pending after-run generation.

Quality comparison should include at least:

- train RMSE and MAE
- validation RMSE and MAE
- test RMSE and MAE
- prediction clipping/out-of-range rates where present

The exact residual reuse change is intended to preserve model semantics.
Numerical deltas should therefore be evaluated conservatively and tied to the
recorded same-contract status.

## Runtime Deltas

Pending after-run generation.

Runtime comparison should include at least:

- `fit_seconds_total`
- `fit_seconds_per_epoch_mean`
- `ratings_per_second_epoch_mean`
- `total_profiled_wall_clock_seconds`
- top performance hotspot and seconds
- `fit_seconds_per_million_estimated_factor_touches`

Runtime deltas must be reported as local laptop readouts only. Cache hit/miss
differences and control-model movement must be considered before interpreting
target-model movement.

## Noise/Control Interpretation

Control models are required because only three kernels changed in Step 16b.

Expected interpretation:

- movement in `biased_mf`, `svdpp`, or `cb_svdpp` cannot be attributed to the
  exact residual reuse kernel changes
- target-model movement should be read against same-device control movement
- cache hit/miss differences affect total wall-clock interpretation
- `fit_model` and kernel-profile counters are more relevant than full pipeline
  wall-clock when cache status differs
- one local laptop run per model is useful evidence, but not a portable
  performance claim

## Acceptance Decision

Pending after-run generation and comparison.

Possible outcomes:

- `ACCEPT`: same-contract checks pass, quality remains stable, target runtime
  readouts are compatible with accepting the exact reuse slice, and no control
  signal contradicts the interpretation.
- `ACCEPT_WITH_LIMITATIONS`: exactness and quality pass, but runtime readout is
  noisy or mixed; keep the slice because it removes exact duplicated work and
  does not regress the validation evidence enough to require revert.
- `DEFER`: evidence is incomplete, same-contract checks are inconclusive, or
  after runs are missing.
- `REVERT`: a target model shows unacceptable quality or runtime evidence under
  a valid same-contract comparison and the issue is attributable to the 16b
  change.

## Claim Boundary

This validation may support an acceptance decision for the narrow Step 16b
implementation slice on the local laptop profile. It does not establish a
portable runtime claim, a dataset-general claim, or a cross-device claim.

Synthetic tiny benchmark values from Step 16b are not used as speed evidence.

## Next Step

Generate after runs for the six ML1M laptop benchmark commands, build the
after-run index, compare against the before inventory, and update this report
with results and an acceptance decision.
