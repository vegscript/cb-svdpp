# Evidence Note

## Scope

First completed local proof-of-concept run for `cb_asvdpp` on the local
`ml_latest_small` dataset contract under the accepted detached explicit
residual contract and the accepted CB v1 repo contract.

## Claim Or Question

Can the repository execute a full end-to-end `cb_asvdpp` run with:

- train-only cluster induction
- fixed cluster assignments during CB training
- detached explicit residual weights
- diagnostic-only `R_star`

and does this first compositional CB variant improve the current local POC
baselines?

## Inputs And Artifacts

- processed dataset manifest:
  `data/processed/ml_latest_small/ml_latest_small_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config: `configs/models/cb_asvdpp.yaml`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- run directory:
  `artifacts/runs/2026-04-15T170715Z_ml_latest_small_cb_asvdpp_local_i5_2500k_24gb_s001/`
- run manifest:
  `artifacts/runs/2026-04-15T170715Z_ml_latest_small_cb_asvdpp_local_i5_2500k_24gb_s001/run_manifest.json`
- run metrics:
  `artifacts/runs/2026-04-15T170715Z_ml_latest_small_cb_asvdpp_local_i5_2500k_24gb_s001/metrics.json`
- comparison baselines:
  `artifacts/runs/2026-04-12T201114Z_ml_latest_small_biased_mf_local_i5_2500k_24gb_s001/`
  `artifacts/runs/2026-04-12T203658Z_ml_latest_small_svdpp_local_i5_2500k_24gb_s001/`
  `artifacts/runs/2026-04-12T215954Z_ml_latest_small_asymmetric_svd_local_i5_2500k_24gb_s001/`
  `artifacts/runs/2026-04-12T223030Z_ml_latest_small_asvdpp_local_i5_2500k_24gb_s001/`
  `artifacts/runs/2026-04-13T003438Z_ml_latest_small_cb_svdpp_local_i5_2500k_24gb_s001/`

## Method

- use the canonical `benchmark_random_v1` split with `train_ratio=0.8`,
  `validation_ratio=0.1`, and split seed `1`
- fit a train-only `biased_mf` induction model on the training split
- cluster user latents and item latents separately with `KMeans`
  using `100` user clusters, `100` item clusters, and `kmeans_n_init=10`
- build `R_star` only from the training ratings and treat it as a diagnostic
  artifact, not as a second optimization target
- keep cluster assignments fixed during the subsequent `cb_asvdpp` training run
- train `cb_asvdpp` with `latent_dim=50`, `epochs=20`, `learning_rate=0.01`,
  `lambda_b=lambda_p=lambda_q=lambda_x=lambda_y=lambda_pC=lambda_qC=lambda_xC=lambda_yC=0.02`,
  `alpha=0.1`, model seed `1`, `float32`, and
  `residual_weight_contract=detached`

## Readout

- status: `completed`
- manifest validation: `valid`
- train rows: `81,365`
- validation rows: `10,083`
- test rows: `9,388`
- train RMSE: `0.528843`
- validation RMSE: `0.856718`
- test RMSE: `0.873943`
- cluster induction wall-clock time: `73.980353` seconds
- main training wall-clock time: `1341.400062` seconds
- end-to-end fit time: `1415.380415` seconds
- user clusters: `100`
- item clusters: `100`
- `R_star` non-empty cluster pairs: `7,943 / 10,000`
- `R_star` density: `0.794300`
- induction-model train RMSE: `0.492186`

## Interpretation

The repository now supports the full `cb_asvdpp` ladder endpoint on the same
artifact and governance contracts as the earlier models.

Observed local comparison:

- versus `biased_mf`:
  - validation RMSE improved from `0.870649` to `0.856718`
  - test RMSE improved from `0.890874` to `0.873943`
- versus `svdpp`:
  - validation RMSE improved from `0.867045` to `0.856718`
  - test RMSE improved from `0.885889` to `0.873943`
- versus `asymmetric_svd`:
  - validation RMSE worsened from `0.853331` to `0.856718`
  - test RMSE worsened from `0.872972` to `0.873943`
- versus `asvdpp`:
  - validation RMSE improved from `0.870787` to `0.856718`
  - test RMSE improved from `0.888086` to `0.873943`
- versus `cb_svdpp`:
  - validation RMSE worsened from `0.854943` to `0.856718`
  - test RMSE worsened from `0.872354` to `0.873943`
  - end-to-end fit time increased from `601.739194` to `1415.380415` seconds

On this single-seed local POC run, `cb_asvdpp` is stable and materially better
than the non-CB `biased_mf`, `svdpp`, and `asvdpp` draft baselines, but it does
not beat either `asymmetric_svd` or the current `cb_svdpp` local anchor. It is
also much more expensive than `cb_svdpp` in end-to-end fit time.

Because this model uses both the accepted detached explicit residual contract
from `D-003` and the accepted CB v1 contract from `D-004`, the result must be
described as source-grounded and methodically documented, but not as an exact
paper-faithful reproduction.

## Decision Or Next Step

- keep this run as the first stable local `cb_asvdpp` baseline artifact
- do not promote the current draft `cb_asvdpp` profile over the clean
  `cb_svdpp` anchor
- next benchmark-relevant step: tune `cb_asvdpp` on `ml100k` under the same
  leakagesafe inner-tuning contract before any official outer benchmark claim
