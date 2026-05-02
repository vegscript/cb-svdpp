# Evidence Note

## Scope

Second bounded inner-tuning study for `cb_svdpp` on `MovieLens 100K`, focused on
the local neighborhood of the current `stage1_tuned` winner.

## Claim Or Question

Can a more targeted `stage2` search around the current tuned `cb_svdpp` profile
produce a materially better candidate before spending more outer-benchmark
budget?

## Inputs And Artifacts

- tuning config:
  `configs/experiments/tuning/ml100k_cb_svdpp_stage2.yaml`
- processed manifest:
  `data/processed/ml100k/ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json`
- stage2 tuning benchmark:
  `artifacts/benchmarks/2026-04-15T151654Z_ml100k_inner_tuning_cb_svdpp_stage2_local_i5_2500k_24gb/`
- current clean official `cb_svdpp` anchor for context:
  `artifacts/benchmarks/2026-04-15T150207Z_ml100k_paper_faithful_cb_svdpp_multiseed_s001_s002_s003_local_i5_2500k_24gb/`

## Method

- inner tuning on `paper_faithful_ml100k_inner_v1`
- outer base folds used for selection: `u1`, `u2`, `u3`
- validation ratio: `0.1`
- inner seed: `17`
- model seed: `1`
- targeted search around the stage1 winner with five candidates varying:
  - `alpha`
  - learning rate
  - regularization
  - epochs
  - cluster count

## Readout

- status: `completed`
- git commit: `d76e9d44693420d40f2125a68cfda013853e7168`
- git dirty: `true`
- best candidate:
  `rank064_uc080_ic080_a015_lr0075_reg0025_e020`
- best validation RMSE mean: `0.915126`
- best validation RMSE std: `0.003240`
- best fit-time mean: `222.36` seconds
- stage2 baseline candidate:
  `rank064_uc080_ic080_a010_lr0075_reg0025_e020`
- stage2 baseline validation RMSE mean: `0.915911`
- absolute gain vs. stage2 baseline: `0.000785`

## Interpretation

The stage2 search found a slightly better candidate than the current stage1-like
baseline under the stage2 protocol, and the winner keeps the same rank and
cluster counts while preferring `alpha=0.15` instead of `0.10`.

That improvement is real inside this bounded study, but it is small. The search
does not justify immediate promotion on integrity grounds because:

- the tuning benchmark ran with `git.dirty=true`
- the gain over the stage2 baseline is only `0.000785`
- no official outer benchmark has yet confirmed that this small inner gain
  survives on the held-out test folds

This is therefore a directional search result, not a new benchmark anchor.

## Decision Or Next Step

- keep the stage2 winner as a promising candidate, not yet as a promoted tuned
  profile
- do not create `ml100k_cb_svdpp_stage2.yaml` yet
- next step should be either:
  - one confirmatory official outer benchmark for the stage2 winner, if we want
    to test whether the tiny inner gain survives, or
  - move to `cb_asvdpp` if we judge the marginal search return on `cb_svdpp`
    too small
