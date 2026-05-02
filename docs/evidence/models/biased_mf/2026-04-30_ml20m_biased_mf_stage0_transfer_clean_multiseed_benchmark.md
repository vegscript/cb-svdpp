# Evidence Note

- date: `2026-04-30`
- scope: `ml20m` clean multi-split-seed baseline benchmark
- model: `biased_mf`
- config: `configs/models/tuned/ml20m_biased_mf_stage0_transfer.yaml`
- git_commit: `e9ce60ed0ff895f14a1f899e966a5b724c8f54c1`
- git_dirty: `false`

## Purpose

Promote the previous `ml20m biased_mf` single-seed feasibility readout to a
clean three-split-seed baseline anchor under the canonical `benchmark_random_v1`
contract. This establishes a stronger baseline row for `ml20m`, but it does not
create a final `ml20m` model-comparison claim because the clustering model
evidence remains single-epoch and budget-unmatched.

## Inputs

- processed manifest:
  `data/processed/ml20m/ml20m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config:
  `configs/models/tuned/ml20m_biased_mf_stage0_transfer.yaml`
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

## Artifacts

- benchmark directory:
  `artifacts/benchmarks/2026-04-30T135639Z_ml20m_benchmark_random_v1_biased_mf_multiseed_s001_s002_s003_modelseed_s001_local_i5_2500k_24gb/`
- benchmark manifest:
  `artifacts/benchmarks/2026-04-30T135639Z_ml20m_benchmark_random_v1_biased_mf_multiseed_s001_s002_s003_modelseed_s001_local_i5_2500k_24gb/benchmark_manifest.json`
- summary:
  `artifacts/benchmarks/2026-04-30T135639Z_ml20m_benchmark_random_v1_biased_mf_multiseed_s001_s002_s003_modelseed_s001_local_i5_2500k_24gb/summary.json`
- run manifests:
  `artifacts/runs/2026-04-30T133542Z_ml20m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/run_manifest.json`
  `artifacts/runs/2026-04-30T134218Z_ml20m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s002_s001/run_manifest.json`
  `artifacts/runs/2026-04-30T134844Z_ml20m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s003_s001/run_manifest.json`

## Method

The repository was clean and synchronized on `main` before the runs. Each run
used the canonical CLI from the locked `uv` environment.

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main train-biased-mf data\processed\ml20m\ml20m_benchmark_random_v1_explicit_v1_float32_manifest.json --model-config configs\models\tuned\ml20m_biased_mf_stage0_transfer.yaml --runtime-config configs\runtime\base.yaml --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml --train-ratio 0.8 --validation-ratio 0.1 --split-seed 1 --model-seed 1
.venv\Scripts\python.exe -m recsys_lab.cli.main train-biased-mf data\processed\ml20m\ml20m_benchmark_random_v1_explicit_v1_float32_manifest.json --model-config configs\models\tuned\ml20m_biased_mf_stage0_transfer.yaml --runtime-config configs\runtime\base.yaml --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml --train-ratio 0.8 --validation-ratio 0.1 --split-seed 2 --model-seed 1
.venv\Scripts\python.exe -m recsys_lab.cli.main train-biased-mf data\processed\ml20m\ml20m_benchmark_random_v1_explicit_v1_float32_manifest.json --model-config configs\models\tuned\ml20m_biased_mf_stage0_transfer.yaml --runtime-config configs\runtime\base.yaml --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml --train-ratio 0.8 --validation-ratio 0.1 --split-seed 3 --model-seed 1
```

The completed runs were then aggregated with the canonical multi-seed
benchmark command:

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main benchmark-random-multiseed data\processed\ml20m\ml20m_benchmark_random_v1_explicit_v1_float32_manifest.json configs\models\tuned\ml20m_biased_mf_stage0_transfer.yaml --runtime-config configs\runtime\base.yaml --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml --split-seeds 1,2,3 --model-seed 1 --run-manifest-paths artifacts\runs\2026-04-30T133542Z_ml20m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001\run_manifest.json,artifacts\runs\2026-04-30T134218Z_ml20m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s002_s001\run_manifest.json,artifacts\runs\2026-04-30T134844Z_ml20m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s003_s001\run_manifest.json
```

## Readout

- status:
  `completed`
- benchmark id:
  `2026-04-30T135639Z_ml20m_benchmark_random_v1_biased_mf_multiseed_s001_s002_s003_modelseed_s001_local_i5_2500k_24gb`
- Git dirty:
  `false`
- interactions:
  `20000263`
- split seeds:
  `1,2,3`
- model seed:
  `1`
- validation RMSE mean:
  `0.775339`
- validation RMSE std:
  `0.000683`
- test RMSE mean:
  `0.775803`
- test RMSE std:
  `0.000197`
- training wall clock seconds mean:
  `323.44`
- training wall clock seconds std:
  `8.59`
- peak memory MB mean:
  `13029.49`
- peak memory MB std:
  `13.75`

Per-split readout:

| Split Seed | Validation RMSE | Test RMSE | Fit Time (s) | Peak Memory MB |
| --- | ---: | ---: | ---: | ---: |
| `1` | `0.774734` | `0.775594` | `333.28` | `13036.39` |
| `2` | `0.776079` | `0.775986` | `319.61` | `13013.66` |
| `3` | `0.775204` | `0.775828` | `317.43` | `13038.41` |

## Interpretation

This is a clean three-split-seed `ml20m biased_mf` baseline anchor under the
repo's `benchmark_random_v1` contract. It can support statements about this
specific `biased_mf stage0_transfer` baseline profile on `ml20m`.

This is not a final `ml20m` model-comparison claim. It must not be compared as
a final result against the current `ml20m cb_svdpp` probe because that probe is
single-seed, single-epoch, and budget-unmatched. It is also not a tuned
`ml20m` optimum, not an exact paper reproduction claim, and not a scalability
claim.

## Decision

- Promote `ml20m biased_mf` from `single_seed_feasibility` to
  `clean_multiseed_baseline_anchor`.
- Keep final `ml20m` model-comparison claims blocked until matching
  clustering-model evidence exists under the same split-seed and budget
  discipline.
- Treat the current `ml20m cb_svdpp` row as feasibility/resource evidence only.
