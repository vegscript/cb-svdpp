# Evidence Note

## Scope

First full `u1` to `u5` official `MovieLens 100K` benchmark for `biased_mf`
under the canonical `paper_faithful_ml100k_v1` split family.

## Claim Or Question

What is the first full five-fold `ml100k` baseline for `biased_mf` under the
official split files, and what homogeneous training-time profile does it show
on the default local device profile?

## Inputs And Artifacts

- processed dataset manifest:
  `data/processed/ml100k/ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json`
- split family: `paper_faithful_ml100k_v1`
- folds: `u1` to `u5`
- model config: `configs/models/biased_mf.yaml`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- benchmark directory:
  `artifacts/benchmarks/2026-04-13T034022Z_ml100k_paper_faithful_biased_mf_local_i5_2500k_24gb/`
- benchmark manifest:
  `artifacts/benchmarks/2026-04-13T034022Z_ml100k_paper_faithful_biased_mf_local_i5_2500k_24gb/benchmark_manifest.json`
- benchmark summary:
  `artifacts/benchmarks/2026-04-13T034022Z_ml100k_paper_faithful_biased_mf_local_i5_2500k_24gb/summary.json`

## Method

- use the official `u1.base/u1.test` to `u5.base/u5.test` files directly
- train one `biased_mf` run per official fold with model seed `1`
- disable reuse of older folds because the workspace is dirty and benchmark
  reuse is not allowed in that state
- aggregate fold-level `train_rmse`, `test_rmse`, and
  `training_wall_clock_seconds`

## Readout

- status: `completed`
- benchmark scope: `paper_faithful_ml100k_v1_biased_mf_u1_u5`
- folds completed: `5`
- test RMSE mean: `0.952430`
- test RMSE std: `0.005070`
- test RMSE min/max: `0.948407` / `0.959989`
- train RMSE mean: `0.556664`
- train RMSE std: `0.002425`
- training wall-clock mean: `105.933591` seconds
- training wall-clock std: `5.733264` seconds

Per-fold test RMSE:

- `u1`: `0.959989`
- `u2`: `0.955340`
- `u3`: `0.948407`
- `u4`: `0.949567`
- `u5`: `0.948845`

## Interpretation

The first full official `ml100k` baseline for `biased_mf` is now available as
an aggregate benchmark rather than a single-fold anecdote. The RMSE spread
across folds is modest, and the homogeneous rerun eliminates the earlier timing
caveat introduced by historical fold reuse.

This benchmark is still not benchmark-final in the publication sense:

- the config remains a documented draft profile
- only one model seed has been used
- no hyperparameter tuning has been applied yet

## Decision Or Next Step

- keep this benchmark as the official current `biased_mf` comparison anchor on
  `ml100k`
- compare `svdpp` against this five-fold aggregate, not just against a single
  fold
- use this result to decide whether the remaining RMSE gap is mostly tuning or
  requires model-level audit
