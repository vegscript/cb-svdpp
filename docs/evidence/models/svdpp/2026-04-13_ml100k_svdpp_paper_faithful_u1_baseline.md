# Evidence Note

## Scope

First official `MovieLens 100K` run for `svdpp` on the canonical
`paper_faithful_ml100k_v1` split path using fold `u1`.

## Claim Or Question

Does `svdpp` improve over the first official `biased_mf` baseline on the same
official `ml100k` fold when both are evaluated on the exact same
`paper_faithful_ml100k_v1` split path?

## Inputs And Artifacts

- processed dataset manifest:
  `data/processed/ml100k/ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json`
- split family: `paper_faithful_ml100k_v1`
- fold: `u1`
- model config: `configs/models/svdpp.yaml`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- run directory:
  `artifacts/runs/2026-04-13T021540Z_ml100k_svdpp_local_i5_2500k_24gb_s001/`
- run manifest:
  `artifacts/runs/2026-04-13T021540Z_ml100k_svdpp_local_i5_2500k_24gb_s001/run_manifest.json`
- run metrics:
  `artifacts/runs/2026-04-13T021540Z_ml100k_svdpp_local_i5_2500k_24gb_s001/metrics.json`
- comparison baseline:
  `artifacts/runs/2026-04-13T020700Z_ml100k_biased_mf_local_i5_2500k_24gb_s001/`

## Method

- use the official `u1.base` and `u1.test` files from the raw `ml100k` package
- define the implicit neighborhood from the training-rated items only
- train `svdpp` with `latent_dim=50`, `epochs=20`, `learning_rate=0.01`,
  `lambda_b=lambda_p=lambda_q=lambda_y=0.02`, model seed `1`, and `float32`
- report only `train` and `test` metrics because the official split path has no
  canonical validation partition

## Readout

- status: `completed`
- split family: `paper_faithful_ml100k_v1`
- fold: `u1`
- train rows: `80,000`
- test rows: `20,000`
- validation rows: `0`
- train RMSE: `0.546113`
- test RMSE: `0.952367`
- training wall-clock time: `850.939886` seconds
- users with training history: `943`
- mean history size: `84.8356`
- max history size: `685`

## Interpretation

On the first official `ml100k` fold, `svdpp` improves over the corresponding
`biased_mf` baseline:

- test RMSE improved from `0.959989` to `0.952367`
- training time increased from `276.497542` seconds to `850.939886` seconds

This is the first clean benchmark-relevant sign inside the repository that the
expected direction of the model family still holds under the official dataset
path. At the same time, this is still not enough for a strong reproduction
claim, because:

- only fold `u1` has been run
- only one model seed has been run
- no tuning has been done yet

## Decision Or Next Step

- keep this run as the first official `ml100k` `svdpp` baseline artifact
- expand the comparison to all official `u1` to `u5` folds
- only after that decide whether the remaining RMSE gap is mostly tuning,
  implementation quality, or a deeper methodological issue
