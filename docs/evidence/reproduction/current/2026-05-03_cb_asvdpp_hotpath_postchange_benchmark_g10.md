# CB-ASVD++ Hotpath Post-Change Benchmark G10

- date: `2026-05-03`
- status: `pass_for_exact_workbuffer_remediation_context`
- dataset: `ml100k`
- model: `cb_asvdpp`
- contract:
  `docs/evidence/reproduction/current/2026-05-03_cb_asvdpp_hotpath_remediation_contract_g8.md`
- pre-change baseline:
  `docs/evidence/reproduction/current/2026-05-03_cb_asvdpp_hotpath_prechange_baseline_g9.md`
- config: `configs/models/selected/ml100k/ml100k_cb_asvdpp_stage1.yaml`
- split_family: `benchmark_random_v1`
- split seed: `1`
- model seed: `1`
- git commit: `e6b77c7f9bc5a87259a5e18e618dc18941a3a9e3`
- git dirty: `false`

## Purpose

This note records the post-change benchmark for the exact `cb_asvdpp`
work-buffer remediation in `train_cb_asvdpp_epoch_numba`. The code change
reuses fixed-size work arrays across rating updates inside one epoch kernel
call. It does not change the predictor, objective, update order, split,
configuration, clustering policy, cache policy, dtype, or device profile.

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
  `artifacts/runs/2026-05-03T124801Z_ml100k_cb_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/run_manifest.json`
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
- main training wall-clock seconds: `113.81238390004728`
- total profiled wall-clock seconds: `115.6799853000557`
- peak memory MB: `211.81640625`
- train ratings processed: `1600600`
- ratings per second train: `14063.49594966471`

## Before/After Comparison

| Metric | G9 pre-change | G10 post-change | Delta |
| --- | ---: | ---: | ---: |
| train RMSE | `0.6848969434499206` | `0.6848969434499206` | `0.0` |
| validation RMSE | `0.9134162708331054` | `0.9134162708331054` | `0.0` |
| test RMSE | `0.9102128098774724` | `0.9102128098774724` | `0.0` |
| main training wall-clock seconds | `122.91284980002092` | `113.81238390004728` | `-9.10046589997364` |
| main training wall-clock percent change | `0.0%` | `-7.403998780257792%` | `pass` |

Acceptance thresholds from the G8 contract:

- maximum absolute metric drift: `1e-6`
- required post-change main-training wall-clock threshold:
  `121.68372130202071` seconds
- observed post-change main-training wall-clock:
  `113.81238390004728` seconds

## Stage Profile

| Stage | Seconds |
| --- | ---: |
| `data_load` | `0.7031782999983989` |
| `split_resolution` | `0.12197350000496954` |
| `config_build` | `0.00007100001676008105` |
| `cluster_induction` | `0.09914250002475455` |
| `explicit_feedback_index_build` | `0.027518499991856515` |
| `user_history_index_build` | `0.022522500017657876` |
| `user_cluster_history_build` | `0.09150720003526658` |
| `model_initialization` | `0.00006379996193572879` |
| `main_training` | `113.81238390004728` |
| `inference_train` | `0.3124461999977939` |
| `inference_validation` | `0.24509959999704733` |
| `inference_test` | `0.24407829996198416` |

## Claim Boundary

Allowed:

- the exact `ml100k cb_asvdpp` benchmark context above ran faster in the
  `main_training` stage after the work-buffer remediation
- the allowed statement is limited to this dataset, split family, split seed,
  model seed, config, dtype, device profile, cache policy, and clean git state

Not allowed:

- no broad or unqualified speed claim
- no scalability claim
- no production-readiness claim
- no SOTA claim
- no paper-faithfulness claim
- no large-dataset claim
- no claim that other `cb_asvdpp` configs, datasets, seeds, or devices improve
- no claim that this removes the need for `ml1m`, `ml10m`, or `ml20m`
  benchmark evidence

## Evidence Gate Readout

Executed for this remediation and documentation gate:

- Focused CB-ASVD++/CB-SVD++ gate:
  `8 passed`
- Ruff: `All checks passed!`
- Mypy: `Success: no issues found in 62 source files`
- Pytest before this evidence note: `140 passed`
- Pytest after this evidence note and guardrail test: `141 passed`
