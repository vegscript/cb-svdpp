# Evidence Note

- date: `2026-04-15`
- scope: `ml100k` clean multi-seed benchmark
- model: `biased_mf`
- config: `artifacts/local/config_snapshots/2026-04-13_ml100k_stage1_clean_fb1fcbc/ml100k_biased_mf_stage1_clean_fb1fcbc.yaml`
- git_commit: `fb1fcbc53796fbe27e515a526096bba03ffbb41f`
- git_dirty: `false`

## Purpose

Establish a clean, internally consistent three-seed benchmark anchor for the
current `biased_mf` stage1-tuned profile on official `MovieLens 100K` folds
`u1` to `u5`.

## Inputs

- processed manifest:
  `data/processed/ml100k/ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- model seeds: `1, 2, 3`

## Source Benchmarks

- `artifacts/benchmarks/2026-04-13T110922Z_ml100k_paper_faithful_biased_mf_local_i5_2500k_24gb`
- `artifacts/benchmarks/2026-04-13T114252Z_ml100k_paper_faithful_biased_mf_local_i5_2500k_24gb`
- `artifacts/benchmarks/2026-04-13T122237Z_ml100k_paper_faithful_biased_mf_local_i5_2500k_24gb`

## Aggregate Result

- multi-seed benchmark:
  `artifacts/benchmarks/2026-04-13T130402Z_ml100k_paper_faithful_biased_mf_multiseed_s001_s002_s003_local_i5_2500k_24gb`
- seed-level test RMSE mean: `0.937111`
- seed-level test RMSE std: `0.001492`
- fold-run-level test RMSE mean: `0.937111`
- fold-run-level test RMSE std: `0.006523`
- seed-level training wall clock mean seconds: `277.24`

## Interpretation

- The stage1-tuned `biased_mf` profile is stable across three clean seeds.
- This is the preferred `biased_mf` benchmark anchor for `ml100k` on commit
  `fb1fcbc`.
