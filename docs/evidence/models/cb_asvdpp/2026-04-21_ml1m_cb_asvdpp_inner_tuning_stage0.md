# Evidence Note

- date: `2026-04-21`
- scope: `ml1m` clean reduced-budget inner tuning
- model: `cb_asvdpp`
- tuning_config: `configs/experiments/tuning/ml1m_cb_asvdpp_stage0.yaml`
- git_commit: `e515d20f6f78d1bdc88d89a13876e0ea6272bd0e`
- git_dirty: `false`

## Purpose

Establish whether the `ml1m` `cb_asvdpp` path has any reduced-budget local
stage0 candidate that is clearly strong enough to justify a later outer
benchmark on the default `local_i5_2500k_24gb` device.

## Inputs

- processed manifest:
  `data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- tuning config:
  `configs/experiments/tuning/ml1m_cb_asvdpp_stage0.yaml`
- base model config:
  `configs/models/tuned/ml1m_cb_asvdpp_stage0_transfer.yaml`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- split family: `benchmark_random_v1`
- split contract: `train_ratio=0.8`, `validation_ratio=0.1`
- split seeds: `1,2`
- model seed: `1`

## Artifacts

- clean inner tuning benchmark:
  `artifacts/benchmarks/2026-04-21T204336Z_ml1m_inner_tuning_cb_asvdpp_stage0_local_i5_2500k_24gb/`
- candidate run:
  `artifacts/runs/2026-04-21T204337Z_ml1m_cb_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/`
- candidate run:
  `artifacts/runs/2026-04-21T204750Z_ml1m_cb_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s002_s001/`
- candidate run:
  `artifacts/runs/2026-04-21T205236Z_ml1m_cb_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/`
- candidate run:
  `artifacts/runs/2026-04-21T205639Z_ml1m_cb_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s002_s001/`
- candidate run:
  `artifacts/runs/2026-04-21T210252Z_ml1m_cb_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/`
- candidate run:
  `artifacts/runs/2026-04-21T210724Z_ml1m_cb_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s002_s001/`

## Candidate Readout

| Candidate | Validation RMSE Mean | Validation RMSE Std | Effective Fit Time Mean (s) |
| --- | ---: | ---: | ---: |
| `rank064_uc064_ic064_a010_lr0075_reg0025_e002` | 0.910164 | 0.000841 | 236.66 |
| `rank064_uc080_ic080_a010_lr0075_reg0025_e002` | 0.910204 | 0.000757 | 256.65 |
| `rank064_uc080_ic080_a015_lr0075_reg0025_e002` | 0.912088 | 0.000685 | 297.32 |

## Winner Delta Versus Transfer-Like Baseline

- winner:
  `rank064_uc064_ic064_a010_lr0075_reg0025_e002`
- baseline comparison candidate:
  `rank064_uc080_ic080_a010_lr0075_reg0025_e002`
- validation RMSE delta: `-0.000040`
- effective fit-time delta seconds: `-19.99`

## Interpretation

- This is the first clean `ml1m` inner-tuning benchmark for `cb_asvdpp` in the
  repository.
- The smaller `64/64`, `alpha=0.10` candidate ranks first, but only by a very
  small margin over the transferred `80/80`, `alpha=0.10` profile.
- That margin is too small to justify a promoted `ml1m` tuned benchmark
  profile or a stronger quality claim without later outer confirmation.
- The `alpha=0.15` candidate is clearly dominated in this local study because
  it is both slower and worse on validation RMSE than the top two candidates.

## Decision

- Record this artifact as clean selection evidence for the `ml1m` `cb_asvdpp`
  path.
- Do not promote a new versioned `ml1m` `cb_asvdpp` benchmark config yet.
- If later outer benchmarking is justified, use
  `rank064_uc064_ic064_a010_lr0075_reg0025_e002`
  as the first benchmark candidate.

## Next Step

- Prefer final report consolidation unless a stronger reason emerges to spend
  additional `ml1m` budget on `cb_asvdpp`.
