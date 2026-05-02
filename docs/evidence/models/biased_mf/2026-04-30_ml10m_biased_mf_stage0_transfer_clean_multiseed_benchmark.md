# Evidence Note

- date: `2026-04-30`
- scope: `ml10m` clean multi-split-seed baseline benchmark
- model: `biased_mf`
- config: `configs/models/tuned/ml10m_biased_mf_stage0_transfer.yaml`
- git_commit: `bbe5f816a3fdab6757fb5ba8a457ad7389b32cde`
- git_dirty: `false`

## Purpose

Promote the previous `ml10m biased_mf` single-seed feasibility readout to a
clean three-split-seed baseline anchor under the canonical `benchmark_random_v1`
contract. This establishes a stronger baseline row for `ml10m`, but it does not
create a final `ml10m` model-comparison claim because the clustering model
evidence is still budget-unmatched.

## Inputs

- processed manifest:
  `data/processed/ml10m/ml10m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config:
  `configs/models/tuned/ml10m_biased_mf_stage0_transfer.yaml`
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
  `artifacts/benchmarks/2026-04-30T085703Z_ml10m_benchmark_random_v1_biased_mf_multiseed_s001_s002_s003_modelseed_s001_local_i5_2500k_24gb/`
- benchmark manifest:
  `artifacts/benchmarks/2026-04-30T085703Z_ml10m_benchmark_random_v1_biased_mf_multiseed_s001_s002_s003_modelseed_s001_local_i5_2500k_24gb/benchmark_manifest.json`
- summary:
  `artifacts/benchmarks/2026-04-30T085703Z_ml10m_benchmark_random_v1_biased_mf_multiseed_s001_s002_s003_modelseed_s001_local_i5_2500k_24gb/summary.json`
- run manifests:
  `artifacts/runs/2026-04-30T084717Z_ml10m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/run_manifest.json`
  `artifacts/runs/2026-04-30T085024Z_ml10m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s002_s001/run_manifest.json`
  `artifacts/runs/2026-04-30T085331Z_ml10m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s003_s001/run_manifest.json`

## Method

The repository was clean and synchronized on `main` before the runs. Each run
used the canonical CLI from the locked `uv` environment.

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main train-biased-mf data\processed\ml10m\ml10m_benchmark_random_v1_explicit_v1_float32_manifest.json --model-config configs\models\tuned\ml10m_biased_mf_stage0_transfer.yaml --runtime-config configs\runtime\base.yaml --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml --train-ratio 0.8 --validation-ratio 0.1 --split-seed 1 --model-seed 1
.venv\Scripts\python.exe -m recsys_lab.cli.main train-biased-mf data\processed\ml10m\ml10m_benchmark_random_v1_explicit_v1_float32_manifest.json --model-config configs\models\tuned\ml10m_biased_mf_stage0_transfer.yaml --runtime-config configs\runtime\base.yaml --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml --train-ratio 0.8 --validation-ratio 0.1 --split-seed 2 --model-seed 1
.venv\Scripts\python.exe -m recsys_lab.cli.main train-biased-mf data\processed\ml10m\ml10m_benchmark_random_v1_explicit_v1_float32_manifest.json --model-config configs\models\tuned\ml10m_biased_mf_stage0_transfer.yaml --runtime-config configs\runtime\base.yaml --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml --train-ratio 0.8 --validation-ratio 0.1 --split-seed 3 --model-seed 1
```

The completed runs were then aggregated with the canonical multi-seed
benchmark command:

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main benchmark-random-multiseed data\processed\ml10m\ml10m_benchmark_random_v1_explicit_v1_float32_manifest.json configs\models\tuned\ml10m_biased_mf_stage0_transfer.yaml --runtime-config configs\runtime\base.yaml --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml --split-seeds 1,2,3 --model-seed 1 --run-manifest-paths artifacts\runs\2026-04-30T084717Z_ml10m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001\run_manifest.json,artifacts\runs\2026-04-30T085024Z_ml10m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s002_s001\run_manifest.json,artifacts\runs\2026-04-30T085331Z_ml10m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s003_s001\run_manifest.json
```

## Readout

- status:
  `completed`
- benchmark id:
  `2026-04-30T085703Z_ml10m_benchmark_random_v1_biased_mf_multiseed_s001_s002_s003_modelseed_s001_local_i5_2500k_24gb`
- Git dirty:
  `false`
- interactions:
  `10000054`
- split seeds:
  `1,2,3`
- model seed:
  `1`
- validation RMSE mean:
  `0.787190`
- validation RMSE std:
  `0.001495`
- test RMSE mean:
  `0.787738`
- test RMSE std:
  `0.000916`
- training wall clock seconds mean:
  `147.45`
- training wall clock seconds std:
  `2.95`
- peak memory MB mean:
  `6583.41`
- peak memory MB std:
  `12.87`

Per-split readout:

| Split Seed | Validation RMSE | Test RMSE | Fit Time (s) | Peak Memory MB |
| --- | ---: | ---: | ---: | ---: |
| `1` | `0.786906` | `0.786747` | `146.45` | `6576.65` |
| `2` | `0.785857` | `0.788554` | `145.13` | `6575.32` |
| `3` | `0.788807` | `0.787912` | `150.77` | `6598.25` |

## Interpretation

This is a clean three-split-seed `ml10m biased_mf` baseline anchor under the
repo's `benchmark_random_v1` contract. It can support statements about this
specific `biased_mf stage0_transfer` baseline profile on `ml10m`.

This is not a final `ml10m` model-comparison claim. It must not be compared as a
final result against the current `ml10m cb_svdpp` probe because that probe is
single-seed, single-epoch, and budget-unmatched. It is also not a tuned `ml10m`
optimum, not an exact paper reproduction claim, and not a scalability claim.

## Decision

- Promote `ml10m biased_mf` from `single_seed_feasibility` to
  `clean_multiseed_baseline_anchor`.
- Keep final `ml10m` model-comparison claims blocked until matching
  clustering-model evidence exists under the same split-seed and budget
  discipline.
- The next evidence step, if local budget is approved, should target a matched
  `ml10m cb_svdpp` campaign or a documented reason why that campaign remains
  deferred on the local device profile.
