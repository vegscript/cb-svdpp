# Evidence Note

## Scope

First completed local proof-of-concept run for `asvdpp` on the local
`ml_latest_small` dataset contract.

## Claim Or Question

Can the repository execute a complete `asvdpp` run on the canonical processed
data and artifact contracts, and does the combined free-user plus asymmetric
feedback block improve the current local POC baselines?

## Inputs And Artifacts

- processed dataset manifest:
  `data/processed/ml_latest_small/ml_latest_small_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config: `configs/models/asvdpp.yaml`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- run directory:
  `artifacts/runs/2026-04-12T223030Z_ml_latest_small_asvdpp_local_i5_2500k_24gb_s001/`
- run manifest:
  `artifacts/runs/2026-04-12T223030Z_ml_latest_small_asvdpp_local_i5_2500k_24gb_s001/run_manifest.json`
- run metrics:
  `artifacts/runs/2026-04-12T223030Z_ml_latest_small_asvdpp_local_i5_2500k_24gb_s001/metrics.json`
- comparison baselines:
  `artifacts/runs/2026-04-12T201114Z_ml_latest_small_biased_mf_local_i5_2500k_24gb_s001/`
  `artifacts/runs/2026-04-12T203658Z_ml_latest_small_svdpp_local_i5_2500k_24gb_s001/`
  `artifacts/runs/2026-04-12T215954Z_ml_latest_small_asymmetric_svd_local_i5_2500k_24gb_s001/`

## Method

- use the canonical `benchmark_random_v1` split with `train_ratio=0.8`,
  `validation_ratio=0.1`, and split seed `1`
- define both `R(u)` and `N(u)` from the unique training-rated items of each
  user
- apply the accepted repo optimizer contract
  `residual_weight_contract=detached`
- train `asvdpp` with `latent_dim=50`, `epochs=20`, `learning_rate=0.01`,
  `lambda_b=lambda_p=lambda_q=lambda_x=lambda_y=0.02`, model seed `1`, and
  `float32`
- execute the hot path through a Numba kernel while keeping the mathematical
  update contract unchanged

## Readout

- status: `completed`
- train rows: `81,365`
- validation rows: `10,083`
- test rows: `9,388`
- train RMSE: `0.438575`
- validation RMSE: `0.870787`
- test RMSE: `0.888086`
- training wall-clock time: `1478.631258` seconds
- users with explicit history: `610`
- mean explicit history size: `133.3852`
- max explicit history size: `2154`

## Interpretation

The repository now supports a full local `asvdpp` implementation on the same
artifact and governance contracts as the earlier model family members. On this
local POC dataset and seed, however, `asvdpp` does not improve over the current
`svdpp` or `asymmetric_svd` baselines.

Observed local comparison:

- versus `biased_mf`:
  - validation RMSE worsened slightly from `0.870649` to `0.870787`
  - test RMSE improved slightly from `0.890874` to `0.888086`
- versus `svdpp`:
  - validation RMSE worsened from `0.867045` to `0.870787`
  - test RMSE worsened from `0.885889` to `0.888086`
- versus `asymmetric_svd`:
  - validation RMSE worsened from `0.853331` to `0.870787`
  - test RMSE worsened from `0.872972` to `0.888086`

This is therefore a valid negative local engineering result: the added `p_u`
block and combined feedback structure do not automatically dominate the simpler
variants under the current draft configuration on `ml_latest_small`.

Because `asvdpp` inherits the accepted detached-residual contract `D-003`, the
result must be described as a repo-defined engineering baseline, not as an exact
optimizer-faithful source reproduction.

## Decision Or Next Step

- keep this run as the first stable local `asvdpp` baseline artifact
- retain the negative result explicitly in the report instead of silently
  preferring only the stronger baselines
- next implementation target: `cb_svdpp`, but only after the CB design
  decisions and `D-004` boundary are tightened enough for a defensible first
  implementation
