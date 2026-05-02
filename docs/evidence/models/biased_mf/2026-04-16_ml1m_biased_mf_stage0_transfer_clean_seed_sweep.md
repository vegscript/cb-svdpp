# Evidence Note

- date: `2026-04-16`
- scope: `ml1m` clean exploratory seed sweep
- model: `biased_mf`
- config: `configs/models/tuned/ml1m_biased_mf_stage0_transfer.yaml`
- git_commit: `240afd8da05450a00814142b4fa574bd9280f298`
- git_dirty: `false`

## Purpose

Restore canonical config truth for the first clean `ml1m` scaling readouts of
`biased_mf` and capture the currently available three-seed exploratory results
without overstating them as a final benchmark anchor.

## Inputs

- processed manifest:
  `data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- split family: `benchmark_random_v1`
- split contract: `train_ratio=0.8`, `validation_ratio=0.1`
- model seeds: `1,2,3`

## Source Runs

- `artifacts/runs/2026-04-16T090057Z_ml1m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001`
- `artifacts/runs/2026-04-16T090240Z_ml1m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s002_s001`
- `artifacts/runs/2026-04-16T090444Z_ml1m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s003_s001`

## Readout

- train RMSE mean: `0.642804`
- train RMSE std: `0.000351`
- validation RMSE mean: `0.866357`
- validation RMSE std: `0.001873`
- test RMSE mean: `0.866615`
- test RMSE std: `0.001616`
- train time total mean seconds: `57.78`
- train time total std seconds: `8.58`

## Interpretation

- The transferred `ml1m` `biased_mf` profile executes cleanly across three
  random split seeds on a clean repo state.
- This is a valid exploratory scaling readout for `ml1m`, not yet a promoted
  benchmark anchor.
- No leakagesafe `ml1m` tuning study or canonical multiseed benchmark artifact
  exists yet for this profile.

## Decision

- `configs/models/tuned/ml1m_biased_mf_stage0_transfer.yaml` is accepted as the
  canonical config reference for these clean exploratory `ml1m` runs.

## Next Step

- Add canonical `ml1m` aggregation tooling and only then promote clean
  multiseed readouts into benchmark-level evidence.
