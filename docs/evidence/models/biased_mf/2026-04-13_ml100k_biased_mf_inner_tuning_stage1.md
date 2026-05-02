# Evidence Note

## Scope

First `stage1` inner-validation tuning study for `biased_mf` on `MovieLens 100K`
using `paper_faithful_ml100k_inner_v1`.

## Claim Or Question

Can a leakagesafe inner-validation search on official `u1/u2` training folds
identify a materially stronger `biased_mf` configuration than the current draft
baseline?

## Inputs And Artifacts

- tuning config:
  `configs/experiments/tuning/ml100k_biased_mf_stage1.yaml`
- processed dataset manifest:
  `data/processed/ml100k/ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- tuning benchmark directory:
  `artifacts/benchmarks/2026-04-13T050523Z_ml100k_inner_tuning_biased_mf_stage1_local_i5_2500k_24gb/`

## Method

- use only official outer folds `u1.base` and `u2.base`
- derive inner `train/validation` splits from the outer training partition only
- keep the official outer test partitions untouched and unevaluated
- evaluate three manually chosen candidate profiles
- select by lowest mean validation RMSE across the two outer folds

## Readout

- selection stage: `stage1`
- outer folds used for tuning: `u1`, `u2`
- inner validation ratio: `0.1`
- inner seed: `17`
- winner: `rank064_lr0075_reg0030_e025`
- winning validation RMSE mean: `0.933395`
- winning validation RMSE std: `0.004680`

Candidate ranking:

- `rank064_lr0075_reg0030_e025`: `0.933395`
- `rank080_lr0050_reg0015_e030`: `0.945506`
- `baseline_k050_lr0100_reg0020_e020`: `0.953546`

## Interpretation

The inner-validation path produced a clear winner, and the baseline candidate
was the weakest of the three. The selected profile is more regularized and
slightly larger than the baseline, which is consistent with the much lower
validation error.

This is a real tuning result, but not benchmark-final:

- only outer folds `u1/u2` were used
- only one model seed was used
- the search space is still small and manually designed

## Decision Or Next Step

- promote the winning profile to a versioned repo config
- benchmark that config on the official `u1` to `u5` outer test folds
- keep the result labeled as `stage1_tuned`, not final
