# Evidence Note

- date: `2026-05-01`
- scope: `ml10m` clean multi-split-seed matched benchmark
- model: `cb_svdpp`
- config: `configs/models/tuned/ml10m_cb_svdpp_stage0_transfer.yaml`
- git_commit: `b70904985d1b84bad5c3ef6d0b69592a0b4fa8b0`
- git_dirty: `false`

## Purpose

Promote `ml10m cb_svdpp` from single-epoch feasibility evidence to a clean
three-split-seed matched benchmark anchor under the canonical
`benchmark_random_v1` contract. This creates a claim-eligible `ml10m`
comparison candidate against the already documented `ml10m biased_mf`
baseline anchor.

This evidence does not create a general model-family superiority claim, a
`paper-faithful` claim, or a scalability claim.

## Inputs

- processed manifest:
  `data/processed/ml10m/ml10m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config:
  `configs/models/tuned/ml10m_cb_svdpp_stage0_transfer.yaml`
- runtime config:
  `configs/runtime/base.yaml`
- device config:
  `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- split family:
  `benchmark_random_v1`
- split contract:
  `train_ratio=0.8`, `validation_ratio=0.1`
- split seeds:
  `1,2,3`
- model seed:
  `1`
- training index cache:
  `disabled`

## Artifacts

- benchmark directory:
  `artifacts/benchmarks/2026-05-01T061215Z_ml10m_benchmark_random_v1_cb_svdpp_multiseed_s001_s002_s003_modelseed_s001_local_i5_2500k_24gb/`
- benchmark manifest:
  `artifacts/benchmarks/2026-05-01T061215Z_ml10m_benchmark_random_v1_cb_svdpp_multiseed_s001_s002_s003_modelseed_s001_local_i5_2500k_24gb/benchmark_manifest.json`
- summary:
  `artifacts/benchmarks/2026-05-01T061215Z_ml10m_benchmark_random_v1_cb_svdpp_multiseed_s001_s002_s003_modelseed_s001_local_i5_2500k_24gb/summary.json`
- run manifests:
  `artifacts/runs/2026-04-30T184417Z_ml10m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/run_manifest.json`
  `artifacts/runs/2026-05-01T003440Z_ml10m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s002_s001/run_manifest.json`
  `artifacts/runs/2026-05-01T033826Z_ml10m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s003_s001/run_manifest.json`

## Method

The repository was clean and synchronized on `main` before every claim-eligible
run and during aggregation. Each completed run used the canonical CLI from the
locked `uv` environment.

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main train-cb-svdpp data\processed\ml10m\ml10m_benchmark_random_v1_explicit_v1_float32_manifest.json --model-config configs\models\tuned\ml10m_cb_svdpp_stage0_transfer.yaml --runtime-config configs\runtime\base.yaml --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml --split-family benchmark_random_v1 --train-ratio 0.8 --validation-ratio 0.1 --split-seed 1 --model-seed 1 --disable-training-index-cache
.venv\Scripts\python.exe -m recsys_lab.cli.main train-cb-svdpp data\processed\ml10m\ml10m_benchmark_random_v1_explicit_v1_float32_manifest.json --model-config configs\models\tuned\ml10m_cb_svdpp_stage0_transfer.yaml --runtime-config configs\runtime\base.yaml --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml --split-family benchmark_random_v1 --train-ratio 0.8 --validation-ratio 0.1 --split-seed 2 --model-seed 1 --disable-training-index-cache
.venv\Scripts\python.exe -m recsys_lab.cli.main train-cb-svdpp data\processed\ml10m\ml10m_benchmark_random_v1_explicit_v1_float32_manifest.json --model-config configs\models\tuned\ml10m_cb_svdpp_stage0_transfer.yaml --runtime-config configs\runtime\base.yaml --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml --split-family benchmark_random_v1 --train-ratio 0.8 --validation-ratio 0.1 --split-seed 3 --model-seed 1 --disable-training-index-cache
```

The completed runs were aggregated with explicit run-manifest paths to exclude
one interrupted `split_seed=2` start artifact that never produced metrics.

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main benchmark-random-multiseed data\processed\ml10m\ml10m_benchmark_random_v1_explicit_v1_float32_manifest.json configs\models\tuned\ml10m_cb_svdpp_stage0_transfer.yaml --runtime-config configs\runtime\base.yaml --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml --split-seeds 1,2,3 --model-seed 1 --run-manifest-paths artifacts\runs\2026-04-30T184417Z_ml10m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001\run_manifest.json,artifacts\runs\2026-05-01T003440Z_ml10m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s002_s001\run_manifest.json,artifacts\runs\2026-05-01T033826Z_ml10m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s003_s001\run_manifest.json
```

## Readout

- status:
  `completed`
- benchmark id:
  `2026-05-01T061215Z_ml10m_benchmark_random_v1_cb_svdpp_multiseed_s001_s002_s003_modelseed_s001_local_i5_2500k_24gb`
- Git dirty:
  `false`
- interactions:
  `10000054`
- split seeds:
  `1,2,3`
- model seed:
  `1`
- validation RMSE mean:
  `0.790782`
- validation RMSE std:
  `0.001257`
- test RMSE mean:
  `0.791315`
- test RMSE std:
  `0.001045`
- training wall clock seconds mean:
  `8986.11`
- training wall clock seconds std:
  `175.73`
- peak memory MB mean:
  `12701.16`
- peak memory MB std:
  `3.24`

Per-split readout:

| Split Seed | Validation RMSE | Test RMSE | Fit Time (s) | Peak Memory MB |
| --- | ---: | ---: | ---: | ---: |
| `1` | `0.790366` | `0.790454` | `8917.04` | `12704.52` |
| `2` | `0.789785` | `0.792477` | `9185.88` | `12700.92` |
| `3` | `0.792194` | `0.791015` | `8855.42` | `12698.06` |

## Comparison Boundary

The matched `ml10m biased_mf` baseline anchor reports validation RMSE mean
`0.787190`, test RMSE mean `0.787738`, fit-time mean `147.45s`, and peak memory
mean `6583.41 MB` under the same split-seed and model-seed contract.

Under this exact `ml10m` contract, the documented `cb_svdpp` profile has higher
validation RMSE and higher test RMSE than the matched `biased_mf` baseline.
Stated as a guardrail phrase: higher validation RMSE, higher test RMSE,
materially higher fit cost, and higher peak memory. This is a bounded
comparison for these two profiles only.

## Interpretation

This is a clean three-split-seed `ml10m cb_svdpp` benchmark anchor under the
repo's `benchmark_random_v1` contract. It can support statements about this
specific `cb_svdpp stage0_transfer` profile on `ml10m`.

It is not a tuned `ml10m` optimum, not an exact paper reproduction claim, not a
large-dataset `cb_svdpp` superiority claim, not a speed claim, and not evidence
for `ml20m`. Guardrail phrase: not evidence for `ml20m`.

The interrupted earlier `split_seed=2` start artifact is not used in the
aggregation because it remained at manifest status `started` and produced no
`metrics.json`.

## Decision

- Promote `ml10m cb_svdpp` from `single_epoch_feasibility` to
  `clean_multiseed_matched_anchor`.
- Allow only the bounded `ml10m biased_mf` versus `ml10m cb_svdpp` comparison
  stated above.
- Keep final `ml20m` model-comparison claims blocked until matching
  clustering-model evidence exists under the same split-seed and budget
  discipline.
