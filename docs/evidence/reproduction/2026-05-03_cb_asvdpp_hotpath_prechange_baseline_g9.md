# CB-ASVD++ Hotpath Pre-Change Baseline G9

- date: `2026-05-03`
- status: `pass_for_clean_prechange_baseline`
- dataset: `ml100k`
- model: `cb_asvdpp`
- contract:
  `docs/evidence/reproduction/2026-05-03_cb_asvdpp_hotpath_remediation_contract_g8.md`
- config: `configs/models/tuned/ml100k_cb_asvdpp_stage1.yaml`
- split_family: `benchmark_random_v1`
- split seed: `1`
- model seed: `1`
- git commit: `bc966e42f4fc2cf3d09c7f7194e17a81c93617cc`
- git dirty: `false`

## Purpose

This note records the clean pre-change baseline required by the `cb_asvdpp`
hot-path remediation contract. It is the comparison baseline for a later
post-change run. It is not a remediation result and does not unlock a speed,
scalability, quality, production-readiness, SOTA, paper-faithfulness, or
large-dataset claim.

## Command

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main train-cb-asvdpp `
  data\processed\ml100k\ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json `
  --model-config configs\models\tuned\ml100k_cb_asvdpp_stage1.yaml `
  --runtime-config configs\runtime\base.yaml `
  --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml `
  --split-family benchmark_random_v1 `
  --train-ratio 0.8 `
  --validation-ratio 0.1 `
  --split-seed 1 `
  --model-seed 1 `
  --cluster-artifact-cache
```

## Artifact Readout

- run artifact:
  `artifacts/runs/2026-05-03T123549Z_ml100k_cb_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/run_manifest.json`
- run status: `completed`
- run git dirty: `false`
- profiler version: `stage_profile_v1`
- stage count: `12`
- cluster artifact cache status: `hit`
- user-cluster history cache status: `hit`
- runtime device profile: `local_i5_2500k_24gb`
- dtype: `float32`

## Metrics

- train RMSE: `0.6848969434499206`
- validation RMSE: `0.9134162708331054`
- test RMSE: `0.9102128098774724`
- main training wall-clock seconds: `122.91284980002092`
- total profiled wall-clock seconds: `125.95959799998673`
- peak memory MB: `326.8203125`
- train ratings processed: `1600600`
- ratings per second train: `13022.234881089931`

## Stage Profile

| Stage | Seconds |
| --- | ---: |
| `data_load` | `1.691132599953562` |
| `split_resolution` | `0.16474759997799993` |
| `config_build` | `0.00007229996845126152` |
| `cluster_induction` | `0.19053179997717962` |
| `explicit_feedback_index_build` | `0.03358640003716573` |
| `user_history_index_build` | `0.03569789999164641` |
| `user_cluster_history_build` | `0.1214509000419639` |
| `model_initialization` | `0.00006890000076964498` |
| `main_training` | `122.91283029998885` |
| `inference_train` | `0.30609590001404285` |
| `inference_validation` | `0.2562440999899991` |
| `inference_test` | `0.24713930004509166` |

## Comparison Contract

The future post-change run may be compared against this baseline only if it
matches:

- dataset
- processed manifest
- split family
- split seed
- model seed
- model config
- runtime config
- device config
- dtype
- cache policy
- clean git state

Acceptance remains governed by the contract:

- `main_training_wall_clock_seconds` must decrease by at least `1.0%`
- `train_rmse`, `validation_rmse`, and `test_rmse` absolute drift must be at
  most `1e-6`
- no speed claim is allowed if runtime does not improve or if metric drift
  exceeds the declared bound

## Claim Boundary

Allowed:

- use this artifact as the clean pre-change baseline for the approved
  `cb_asvdpp` exact work-buffer remediation attempt

Not allowed:

- no speed claim
- no scalability claim
- no production-readiness claim
- no SOTA claim
- no paper-faithfulness claim
- no large-dataset claim
- no quality claim
- no claim that a remediation has already improved runtime

## Evidence Gate Readout

Executed after documenting this baseline:

- Ruff: `All checks passed!`
- Mypy: `Success: no issues found in 62 source files`
- Pytest: `139 passed`
