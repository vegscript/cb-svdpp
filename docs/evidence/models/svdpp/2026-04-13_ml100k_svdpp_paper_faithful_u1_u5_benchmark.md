# Evidence Note

## Scope

First full `u1` to `u5` official `MovieLens 100K` benchmark for `svdpp`
under the canonical `paper_faithful_ml100k_v1` split family.

## Claim Or Question

Does `svdpp` outperform the full five-fold official `biased_mf` baseline on
`ml100k`, and what is the corresponding training-cost profile on the default
local CPU target?

## Inputs And Artifacts

- processed dataset manifest:
  `data/processed/ml100k/ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json`
- split family: `paper_faithful_ml100k_v1`
- folds: `u1` to `u5`
- model config: `configs/models/svdpp.yaml`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- benchmark directory:
  `artifacts/benchmarks/2026-04-13T035011Z_ml100k_paper_faithful_svdpp_local_i5_2500k_24gb/`
- benchmark manifest:
  `artifacts/benchmarks/2026-04-13T035011Z_ml100k_paper_faithful_svdpp_local_i5_2500k_24gb/benchmark_manifest.json`
- benchmark summary:
  `artifacts/benchmarks/2026-04-13T035011Z_ml100k_paper_faithful_svdpp_local_i5_2500k_24gb/summary.json`
- comparison benchmark:
  `artifacts/benchmarks/2026-04-13T034022Z_ml100k_paper_faithful_biased_mf_local_i5_2500k_24gb/`

## Method

- use the official `u1.base/u1.test` to `u5.base/u5.test` files directly
- train one `svdpp` run per official fold with model seed `1`
- define the implicit neighborhood from training-rated items only
- disable reuse of older folds because the workspace is dirty and benchmark
  reuse is not allowed in that state
- aggregate fold-level `train_rmse`, `test_rmse`, and
  `training_wall_clock_seconds`

## Readout

- status: `completed`
- benchmark scope: `paper_faithful_ml100k_v1_svdpp_u1_u5`
- folds completed: `5`
- test RMSE mean: `0.944529`
- test RMSE std: `0.005283`
- test RMSE min/max: `0.939967` / `0.952367`
- train RMSE mean: `0.544582`
- train RMSE std: `0.002368`
- training wall-clock mean: `606.908673` seconds
- training wall-clock std: `101.903401` seconds

Per-fold test RMSE:

- `u1`: `0.952367`
- `u2`: `0.947357`
- `u3`: `0.939967`
- `u4`: `0.942647`
- `u5`: `0.940306`

Comparison against the current official five-fold `biased_mf` benchmark:

- absolute mean test RMSE improvement: `0.007901`
- relative train-time multiplier: about `5.73x`

## Interpretation

The first full official five-fold benchmark confirms that `svdpp` improves
predictive quality over `biased_mf` on `ml100k` under identical official folds.
The gain is modest in absolute RMSE terms but stable enough to be meaningful in
this model family.

The cost side is also now clear: on the default local CPU profile, `svdpp`
requires substantially more training time than `biased_mf`. This makes the
quality-cost tradeoff explicit instead of anecdotal.

This result is benchmark-relevant but still not benchmark-final:

- the config remains a documented draft profile
- only one model seed has been used
- no tuning has been applied yet

## Decision Or Next Step

- keep this as the official current `svdpp` benchmark anchor on `ml100k`
- move next to controlled hyperparameter search on `biased_mf` and `svdpp`
- only after tuned baselines decide whether further RMSE gaps indicate tuning
  debt, implementation debt, or model-family limits
