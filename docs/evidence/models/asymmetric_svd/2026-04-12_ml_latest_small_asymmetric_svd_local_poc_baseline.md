# Evidence Note

## Scope

First completed local proof-of-concept run for `asymmetric_svd` on the local
`ml_latest_small` dataset contract.

## Claim Or Question

Can the repository execute a complete `asymmetric_svd` run on the canonical
processed-data and artifact contracts, while making the detached residual-weight
optimizer contract explicit and operational?

## Inputs And Artifacts

- processed dataset manifest:
  `data/processed/ml_latest_small/ml_latest_small_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config: `configs/models/asymmetric_svd.yaml`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- completed run directory:
  `artifacts/runs/2026-04-12T215954Z_ml_latest_small_asymmetric_svd_local_i5_2500k_24gb_s001/`
- completed run manifest:
  `artifacts/runs/2026-04-12T215954Z_ml_latest_small_asymmetric_svd_local_i5_2500k_24gb_s001/run_manifest.json`
- completed run metrics:
  `artifacts/runs/2026-04-12T215954Z_ml_latest_small_asymmetric_svd_local_i5_2500k_24gb_s001/metrics.json`
- cancelled pre-optimization run:
  `artifacts/runs/2026-04-12T211559Z_ml_latest_small_asymmetric_svd_local_i5_2500k_24gb_s001/`

## Method

- use the canonical `benchmark_random_v1` split with `train_ratio=0.8`,
  `validation_ratio=0.1`, and split seed `1`
- define both `R(u)` and `N(u)` from the unique training-rated items of each
  user
- implement the accepted repo contract
  `residual_weight_contract=detached`
- train `asymmetric_svd` with `latent_dim=50`, `epochs=20`,
  `learning_rate=0.01`, `lambda_b=lambda_q=lambda_x=lambda_y=0.04`, model seed
  `1`, and `float32`
- execute the hot path through a Numba kernel while keeping the mathematical
  update contract unchanged

## Readout

- status: `completed`
- train rows: `81,365`
- validation rows: `10,083`
- test rows: `9,388`
- train RMSE: `0.772916`
- validation RMSE: `0.853331`
- test RMSE: `0.872972`
- training wall-clock time: `460.518888` seconds
- users with explicit history: `610`
- mean explicit history size: `133.3852`
- max explicit history size: `2154`

## Interpretation

The first pure-Python local run exceeded the 30-minute execution budget and was
therefore cancelled. After moving the `asymmetric_svd` hot path into a Numba
kernel, the same mathematical contract completed successfully on the default
local device profile.

On this local POC dataset and seed, the completed `asymmetric_svd` run improves
validation and test RMSE over the current `biased_mf` and `svdpp` POC baselines.
However, this model remains under the accepted deviation `D-003`, so the result
must be described as a repo-defined detached-residual implementation rather than
as an exact optimizer-faithful reconstruction of Koren 2008.

## Decision Or Next Step

- treat this run as the first stable local `asymmetric_svd` baseline artifact
- keep the cancelled pre-kernel run as negative engineering evidence, not as a
  reportable result
- next implementation target: `asvdpp`, reusing the same detached-residual
  contract and the same processed-data, split, manifest, and report structure
