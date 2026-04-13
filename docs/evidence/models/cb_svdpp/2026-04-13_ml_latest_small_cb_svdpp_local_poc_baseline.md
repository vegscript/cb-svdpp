# Evidence Note

## Scope

First completed local proof-of-concept run for `cb_svdpp` on the local
`ml_latest_small` dataset contract under the accepted CB v1 repo contract.

## Claim Or Question

Can the repository execute a full clustering-based `svdpp` run with train-only
cluster induction, fixed cluster assignments, and diagnostic-only `R_star`, and
does this first CB variant improve the current local POC baselines?

## Inputs And Artifacts

- processed dataset manifest:
  `data/processed/ml_latest_small/ml_latest_small_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config: `configs/models/cb_svdpp.yaml`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- run directory:
  `artifacts/runs/2026-04-13T003438Z_ml_latest_small_cb_svdpp_local_i5_2500k_24gb_s001/`
- run manifest:
  `artifacts/runs/2026-04-13T003438Z_ml_latest_small_cb_svdpp_local_i5_2500k_24gb_s001/run_manifest.json`
- run metrics:
  `artifacts/runs/2026-04-13T003438Z_ml_latest_small_cb_svdpp_local_i5_2500k_24gb_s001/metrics.json`
- comparison baselines:
  `artifacts/runs/2026-04-12T201114Z_ml_latest_small_biased_mf_local_i5_2500k_24gb_s001/`
  `artifacts/runs/2026-04-12T203658Z_ml_latest_small_svdpp_local_i5_2500k_24gb_s001/`
  `artifacts/runs/2026-04-12T215954Z_ml_latest_small_asymmetric_svd_local_i5_2500k_24gb_s001/`
  `artifacts/runs/2026-04-12T223030Z_ml_latest_small_asvdpp_local_i5_2500k_24gb_s001/`

## Method

- use the canonical `benchmark_random_v1` split with `train_ratio=0.8`,
  `validation_ratio=0.1`, and split seed `1`
- fit a train-only `biased_mf` induction model on the training split
- cluster user latents and item latents separately with `KMeans`
  using `100` user clusters, `100` item clusters, and `kmeans_n_init=10`
- build `R_star` only from the training ratings and treat it as a diagnostic
  artifact, not as a second optimization target
- keep cluster assignments fixed during the subsequent `cb_svdpp` training run
- train `cb_svdpp` with `latent_dim=50`, `epochs=20`, `learning_rate=0.01`,
  `lambda_b=lambda_p=lambda_q=lambda_y=lambda_pC=lambda_qC=lambda_yC=0.02`,
  `alpha=0.1`, model seed `1`, and `float32`

## Readout

- status: `completed`
- train rows: `81,365`
- validation rows: `10,083`
- test rows: `9,388`
- train RMSE: `0.554827`
- validation RMSE: `0.854943`
- test RMSE: `0.872354`
- cluster induction wall-clock time: `123.233366` seconds
- main training wall-clock time: `478.505828` seconds
- end-to-end fit time: `601.739194` seconds
- user clusters: `100`
- item clusters: `100`
- `R_star` non-empty cluster pairs: `7,943 / 10,000`
- `R_star` density: `0.794300`
- induction-model train RMSE: `0.492186`

## Interpretation

The repository now supports the first clustering-based model family member on
the same artifact and governance contracts as the earlier baselines.

Observed local comparison:

- versus `biased_mf`:
  - validation RMSE improved from `0.870649` to `0.854943`
  - test RMSE improved from `0.890874` to `0.872354`
- versus `svdpp`:
  - validation RMSE improved from `0.867045` to `0.854943`
  - test RMSE improved from `0.885889` to `0.872354`
- versus `asymmetric_svd`:
  - validation RMSE worsened slightly from `0.853331` to `0.854943`
  - test RMSE improved slightly from `0.872972` to `0.872354`
- versus `asvdpp`:
  - validation RMSE improved from `0.870787` to `0.854943`
  - test RMSE improved from `0.888086` to `0.872354`

On this single-seed local POC run, `cb_svdpp` produces the best test RMSE seen
so far in the repository, but not the best validation RMSE. That means the
result is promising, not settled. It is strong enough to justify continuation
of the CB branch, but not strong enough to claim that clustering is now
decisively superior on this setup.

Because this model uses the accepted CB v1 contract from `D-004`, the result
must be described as a source-grounded predictor with repo-defined
optimization, not as an exact paper reproduction.

## Decision Or Next Step

- keep this run as the first stable local `cb_svdpp` baseline artifact
- preserve both the cluster induction time and the main training time in all
  future comparisons; do not collapse them into a single unlabeled fit cost
- next implementation target: `cb_asvdpp` on the same fixed-assignment and
  diagnostic-only `R_star` contract
