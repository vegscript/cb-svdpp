# Evidence Note

## Scope

First leakagesafe stage1 inner-tuning study for `cb_asvdpp` on `ml100k` using
the canonical `paper_faithful_ml100k_inner_v1` contract.

## Claim Or Question

Does `cb_asvdpp` produce a benchmark-relevant tuned profile on `ml100k`, and if
so, does the first stage1 search outperform both its own draft baseline and the
current `cb_svdpp` stage1 winner on the same inner-validation protocol?

## Inputs And Artifacts

- processed dataset manifest:
  `data/processed/ml100k/ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json`
- tuning config:
  `configs/experiments/tuning/ml100k_cb_asvdpp_stage1.yaml`
- base model config: `configs/models/cb_asvdpp.yaml`
- benchmark directory:
  `artifacts/benchmarks/2026-04-15T174215Z_ml100k_inner_tuning_cb_asvdpp_stage1_local_i5_2500k_24gb/`
- benchmark manifest:
  `artifacts/benchmarks/2026-04-15T174215Z_ml100k_inner_tuning_cb_asvdpp_stage1_local_i5_2500k_24gb/benchmark_manifest.json`
- benchmark summary:
  `artifacts/benchmarks/2026-04-15T174215Z_ml100k_inner_tuning_cb_asvdpp_stage1_local_i5_2500k_24gb/summary.json`
- comparison study:
  `artifacts/benchmarks/2026-04-15T074345Z_ml100k_inner_tuning_cb_svdpp_stage1_local_i5_2500k_24gb/summary.json`

## Method

- use the no-leakage `paper_faithful_ml100k_inner_v1` protocol
- derive inner validation splits only from `u1.base` and `u2.base`
- rank candidates by mean validation RMSE across the two inner folds
- count fit time as `training_wall_clock_seconds` plus
  `cluster_induction_wall_clock_seconds` when present
- evaluate four candidates:
  - `baseline_k050_uc100_ic100_a010_lr0100_reg0020_e020`
  - `alpha05_k050_uc100_ic100_lr0100_reg0020_e020`
  - `alpha15_k050_uc100_ic100_lr0100_reg0020_e020`
  - `rank064_uc080_ic080_a010_lr0075_reg0025_e020`

## Readout

- status: `completed`
- manifest validation: `valid`
- winning candidate: `rank064_uc080_ic080_a010_lr0075_reg0025_e020`
- winning validation RMSE mean: `0.913004`
- winning validation RMSE std: `0.005457`
- winning fit-time mean: `404.974576` seconds
- baseline candidate validation RMSE mean: `0.925254`
- baseline candidate fit-time mean: `359.381545` seconds
- current `cb_svdpp` stage1 winning validation RMSE mean: `0.915080`

## Interpretation

The first `cb_asvdpp` stage1 study produces a real tuning signal rather than a
trivial alpha-only fluctuation.

Observed comparison:

- versus the `cb_asvdpp` baseline candidate:
  - validation RMSE improved from `0.925254` to `0.913004`
  - absolute gain: `0.012250`
  - fit-time mean increased from `359.381545` to `404.974576` seconds
- versus the current `cb_svdpp` stage1 winner:
  - validation RMSE improved from `0.915080` to `0.913004`
  - absolute gain: `0.002076`

This is strong enough to justify promotion into a versioned `stage1_tuned`
profile and a first official outer benchmark. It is not enough to claim a
final hierarchy shift yet, because the result is still an inner-validation
selection signal and the model continues to inherit the accepted detached
explicit residual contract from `D-003`.

## Decision Or Next Step

- promote the winning candidate into `configs/models/tuned/ml100k_cb_asvdpp_stage1.yaml`
- run the first official `ml100k` outer benchmark for the promoted profile
- treat any resulting outer readout as provisional until it is repeated on a
  clean Git state and, ideally, across multiple seeds
