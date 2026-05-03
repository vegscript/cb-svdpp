# CB-ASVD++ Hotpath Decision G7

- date: `2026-05-03`
- status: `pass_for_hotpath_prioritization_not_remediation`
- dataset: `ml100k`
- model: `cb_asvdpp`
- config: `configs/models/tuned/ml100k_cb_asvdpp_stage1.yaml`
- split_family: `benchmark_random_v1`
- split seed: `1`
- model seed: `1`
- git commit: `2edc8a3be8f64c657df9519befc371d9e7accfd3`
- git dirty: `false`

## Purpose

This note decides whether `cb_asvdpp` hot-path remediation is justified as a
separate work item. It is a single-run profiling decision, not a benchmark anchor,
not a tuning result, and not a performance-improvement claim.

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
  `artifacts/runs/2026-05-03T121942Z_ml100k_cb_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/run_manifest.json`
- run status: `completed`
- run git dirty: `false`
- profiler version: `stage_profile_v1`
- stage count: `12`
- cache status:
  - cluster artifacts: `miss`
  - user-cluster history: `miss`
- runtime device profile: `local_i5_2500k_24gb`

## Metrics

Quality metrics are recorded only to preserve the run readout. They are not a
claimable benchmark anchor because this is a single split-seed profiling run.

- train RMSE: `0.6848969434499206`
- validation RMSE: `0.9134162708331054`
- test RMSE: `0.9102128098774724`
- peak memory MB: `301.21875`
- train ratings processed: `1600600`
- ratings per second train: `13905.732530074107`

## Stage Profile

| Stage | Seconds | Share Of Profiled Wall Clock |
| --- | ---: | ---: |
| `data_load` | `0.7252324000000954` | `0.58%` |
| `split_resolution` | `0.13881730003049597` | `0.11%` |
| `config_build` | `0.00007190002361312509` | `0.00%` |
| `cluster_induction` | `6.17832319997251` | `4.96%` |
| `explicit_feedback_index_build` | `0.027955199999269098` | `0.02%` |
| `user_history_index_build` | `0.023112300026696175` | `0.02%` |
| `user_cluster_history_build` | `1.4987650000257418` | `1.20%` |
| `model_initialization` | `0.00005779997445642948` | `0.00%` |
| `main_training` | `115.10358980001183` | `92.44%` |
| `inference_train` | `0.3255631000502035` | `0.26%` |
| `inference_validation` | `0.24484830000437796` | `0.20%` |
| `inference_test` | `0.2466058999998495` | `0.20%` |

Total profiled wall-clock seconds: `124.51294220011914`.

## Decision

`cb_asvdpp` hot-path remediation is justified as a separate work item because
`main_training` dominates the profiled run at about `92.44%` of profiled wall
clock.

The next work item must be a remediation contract before code changes. That
contract must define:

- exact implementation target, for example Numba work-buffer reuse in
  `train_cb_asvdpp_epoch_numba`
- deterministic equivalence tests against the current kernel
- metric drift bound before any benchmark is run
- before/after benchmark command on the same dataset, split, config, seed,
  dtype, cache policy, device profile, and clean git state
- acceptance rule that rejects the candidate if runtime is not improved or if
  metrics drift beyond the predeclared bound

## Claim Boundary

Allowed:

- `cb_asvdpp` main training is the dominant measured stage in this single
  bounded `ml100k` profiling context.
- `cb_asvdpp` hot-path remediation is prioritized for a separate exactness and
  before/after benchmark gate.

Not allowed:

- no speed claim
- no scalability claim
- no production-readiness claim
- no SOTA claim
- no paper-faithfulness claim
- no large-dataset claim
- no claim that a remediation has already improved runtime
- no model-quality claim from this single profiling run

## Evidence Gate Readout

Executed after documenting this decision:

- Ruff: `All checks passed!`
- Mypy: `Success: no issues found in 62 source files`
- Pytest: `137 passed`
