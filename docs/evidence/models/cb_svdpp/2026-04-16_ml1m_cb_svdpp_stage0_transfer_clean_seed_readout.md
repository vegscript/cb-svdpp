# Evidence Note

- date: `2026-04-16`
- scope: `ml1m` clean exploratory single-seed readout
- model: `cb_svdpp`
- config: `configs/models/tuned/ml1m_cb_svdpp_stage0_transfer.yaml`
- git_commit: `240afd8da05450a00814142b4fa574bd9280f298`
- git_dirty: `false`

## Purpose

Restore canonical config truth for the first clean `ml1m` scaling readout of
`cb_svdpp` and record the currently available single-seed exploratory result
without promoting it to a benchmark anchor.

## Inputs

- processed manifest:
  `data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- split family: `benchmark_random_v1`
- split contract: `train_ratio=0.8`, `validation_ratio=0.1`
- model seed: `1`

## Source Run

- `artifacts/runs/2026-04-16T090913Z_ml1m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001`

## Readout

- train RMSE: `0.724115`
- validation RMSE: `0.857911`
- test RMSE: `0.859314`
- cluster induction wall clock seconds: `90.99`
- main training wall clock seconds: `3902.78`
- train time total seconds: `3993.77`

## Interpretation

- The transferred `ml1m` `cb_svdpp` profile executes cleanly on the first
  random split seed under a clean repo state.
- On this single seed, it improves over the matching clean `biased_mf` seed-1
  readout on both validation RMSE (`0.866678` to `0.857911`) and test RMSE
  (`0.868475` to `0.859314`).
- This is still only a single-seed exploratory readout and must not be treated
  as a benchmark-level conclusion.
- Fit-time interpretation must include both cluster induction and main training.

## Decision

- `configs/models/tuned/ml1m_cb_svdpp_stage0_transfer.yaml` is accepted as the
  canonical config reference for the current clean exploratory `ml1m` run.

## Next Step

- Produce clean additional seeds or a clean tuning-plus-aggregation path before
  making any model-ranking claim for `ml1m`.
