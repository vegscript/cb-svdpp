# Evidence Note

- date: `2026-04-15`
- scope: `ml100k` clean multi-seed benchmark
- model: `svdpp`
- config: `artifacts/local/config_snapshots/2026-04-13_ml100k_stage1_clean_fb1fcbc/ml100k_svdpp_stage1_clean_fb1fcbc.yaml`
- git_commit: `fb1fcbc53796fbe27e515a526096bba03ffbb41f`
- git_dirty: `false`

## Purpose

Establish a clean, internally consistent three-seed benchmark anchor for the
current `svdpp` stage1-tuned profile on official `MovieLens 100K` folds
`u1` to `u5`.

## Inputs

- processed manifest:
  `data/processed/ml100k/ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- model seeds: `1, 2, 3`

## Source Benchmarks

- `artifacts/benchmarks/2026-04-13T130840Z_ml100k_paper_faithful_svdpp_local_i5_2500k_24gb`
- `artifacts/benchmarks/2026-04-14T235611Z_ml100k_paper_faithful_svdpp_local_i5_2500k_24gb`
- `artifacts/benchmarks/2026-04-15T005436Z_ml100k_paper_faithful_svdpp_local_i5_2500k_24gb`

## Aggregate Result

- multi-seed benchmark:
  `artifacts/benchmarks/2026-04-15T030305Z_ml100k_paper_faithful_svdpp_multiseed_s001_s002_s003_local_i5_2500k_24gb`
- seed-level test RMSE mean: `0.924015`
- seed-level test RMSE std: `0.000461`
- fold-run-level test RMSE mean: `0.924015`
- fold-run-level test RMSE std: `0.007650`
- seed-level training wall clock mean seconds: `1386.62`

## Interpretation

- The stage1-tuned `svdpp` profile is stable across three clean seeds.
- `svdpp` remains clearly better than the clean multi-seed `biased_mf` anchor
  on `ml100k`.
- This is now the preferred `svdpp` benchmark anchor for `ml100k` on commit
  `fb1fcbc`.
