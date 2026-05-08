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

Comparison CSV:

- `artifacts/reports/exact_residual_reuse_validation_v1.csv`

| Model | Before run id | After run id | Same-contract status | Quality status | Runtime readout |
| --- | --- | --- | --- | --- | --- |
| `asymmetric_svd` | `2026-05-08T005445Z_ml1m_asymmetric_svd_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `2026-05-08T124759Z_ml1m_asymmetric_svd_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `pass_with_notes` | stable: test RMSE delta `4.1722181265413383e-13`, test MAE delta `9.01708707701232e-11` | local fit-model ratio `0.6827821541007663`; cache status changed from explicit-feedback miss to hit |
| `asvdpp` | `2026-05-08T010531Z_ml1m_asvdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `2026-05-08T125523Z_ml1m_asvdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `pass` | stable: test RMSE delta `4.313439605496683e-10`, test MAE delta `-2.9677804658234663e-10` | local fit-model ratio `0.9916830347792213` |
| `cb_asvdpp` | `2026-05-08T011957Z_ml1m_cb_asvdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `2026-05-08T130305Z_ml1m_cb_asvdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `pass` | stable: test RMSE delta `1.178921404942912e-10`, test MAE delta `2.710514035442202e-10` | local fit-model ratio `0.9371622788151166` |

## Control Model Results

| Model | Before run id | After run id | Same-contract status | Quality status | Runtime readout |
| --- | --- | --- | --- | --- | --- |
| `biased_mf` | `2026-05-08T004817Z_ml1m_biased_mf_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `2026-05-08T131541Z_ml1m_biased_mf_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `pass_with_notes` | stable: test RMSE delta `0.0`, test MAE delta `0.0` | local fit-model ratio `0.5193254460492281`; split cache status changed from miss to hit |
| `svdpp` | `2026-05-08T004906Z_ml1m_svdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `2026-05-08T131609Z_ml1m_svdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `pass_with_notes` | stable: test RMSE delta `0.0`, test MAE delta `0.0` | local fit-model ratio `0.6097998909318263`; user-history cache status changed from miss to hit |
| `cb_svdpp` | `2026-05-08T011336Z_ml1m_cb_svdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `2026-05-08T131921Z_ml1m_cb_svdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `pass_with_notes` | stable: test RMSE delta `0.0`, test MAE delta `0.0` | local fit-model ratio `0.926872109237755`; cluster cache status changed from miss to hit |

## Quality Deltas

Quality is stable for all six compared models.

Target model test deltas:

| Model | Test RMSE delta | Test MAE delta |
| --- | ---: | ---: |
| `asymmetric_svd` | `4.1722181265413383e-13` | `9.01708707701232e-11` |
| `asvdpp` | `4.313439605496683e-10` | `-2.9677804658234663e-10` |
| `cb_asvdpp` | `1.178921404942912e-10` | `2.710514035442202e-10` |

Control model test deltas:

| Model | Test RMSE delta | Test MAE delta |
| --- | ---: | ---: |
| `biased_mf` | `0.0` | `0.0` |
| `svdpp` | `0.0` | `0.0` |
| `cb_svdpp` | `0.0` | `0.0` |

The target deltas are at floating-point noise scale for this validation
context. No quality-change evidence requires a revert.

## Runtime Deltas

Runtime deltas are local laptop readouts only.

| Model | Role | Same-contract status | Fit-model seconds before | Fit-model seconds after | After/before fit ratio | After/before train-time ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `asymmetric_svd` | target | `pass_with_notes` | `608.6185130999656` | `415.55385939999996` | `0.6827821541007663` | `0.6827898313449888` |
| `asvdpp` | target | `pass` | `438.50667919998523` | `434.8596343999998` | `0.9916830347792213` | `0.9916899964904987` |
| `cb_asvdpp` | target | `pass` | `777.1392291000229` | `728.3055709000005` | `0.9371622788151166` | `0.9371539215239106` |
| `biased_mf` | control | `pass_with_notes` | `14.73108559998218` | `7.6502276000001075` | `0.5193254460492281` | `0.5165728345242372` |
| `svdpp` | control | `pass_with_notes` | `278.94070240005385` | `170.09800989999985` | `0.6097998909318263` | `0.6098104865453873` |
| `cb_svdpp` | control | `pass_with_notes` | `291.6227717999718` | `270.29701360000035` | `0.926872109237755` | `0.926918850960623` |

Target readouts do not show a target runtime regression. However, non-target
controls also moved substantially, especially `biased_mf` and `svdpp`.
Therefore this evidence must not attribute the target runtime movement solely
to the Step 16b kernel change.

## Noise/Control Interpretation

Control models are required because only three kernels changed in Step 16b.

Observed interpretation:

- movement in `biased_mf`, `svdpp`, and `cb_svdpp` cannot be attributed to the
  exact residual reuse kernel changes because those kernels were not changed
- target-model movement should be read against same-device control movement
- cache hit/miss differences are documented in the CSV notes and affect
  end-to-end interpretation
- `fit_model` and kernel-profile counters are more relevant than full pipeline
  wall-clock when cache status differs
- one local laptop run per model is useful validation evidence, but not a
  portable runtime claim
- the control movement is high enough that this validation cannot support a
  hard speed interpretation for the target kernels

## Acceptance Decision

Decision: `ACCEPT_AS_EXACT_CLEANUP`.

Rationale:

- same-contract checks pass for `asvdpp` and `cb_asvdpp`
- `asymmetric_svd` passes with a documented cache-status note; command-level
  cache policy and model contract match
- all three target models preserve quality at floating-point noise scale
- all three control models preserve quality exactly in the collected metrics
- target runtime readouts do not show a target regression
- control models move substantially, so the runtime readout is too noisy for a
  hard target-kernel speed attribution
- the implementation remains a narrow exact cleanup: raw residual values are
  reused within the same rating update, no scaled products are cached, no pass
  fusion was introduced, and rating/history/update order remains unchanged

No kernel is selected for revert by this evidence.

## Claim Boundary

This validation may support an acceptance decision for the narrow Step 16b
implementation slice on the local laptop profile. It does not establish a
portable runtime claim, a dataset-general claim, or a cross-device claim.

Synthetic tiny benchmark values from Step 16b are not used as speed evidence.

## Next Step

Step 16c is accepted as `ACCEPT_AS_EXACT_CLEANUP`.

The next red-thread step is:

17. CB Kernel Specialization

The next step should start as an audit/plan, not as an immediate kernel
rewrite:

17a. CB Kernel Specialization Audit and Plan V1.
