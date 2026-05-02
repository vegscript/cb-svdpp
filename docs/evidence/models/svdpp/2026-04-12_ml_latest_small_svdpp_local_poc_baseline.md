# Evidence Note

## Scope

First end-to-end local proof-of-concept run for `svdpp` on the local
`ml_latest_small` dataset contract.

## Claim Or Question

Can the repository execute a full, manifest-valid `svdpp` training run on the
same processed-data, split, and artifact contracts that were already validated
for `biased_mf`?

## Inputs And Artifacts

- processed dataset manifest:
  `data/processed/ml_latest_small/ml_latest_small_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config: `configs/models/svdpp.yaml`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- run directory:
  `artifacts/runs/2026-04-12T203658Z_ml_latest_small_svdpp_local_i5_2500k_24gb_s001/`
- run manifest:
  `artifacts/runs/2026-04-12T203658Z_ml_latest_small_svdpp_local_i5_2500k_24gb_s001/run_manifest.json`
- metrics:
  `artifacts/runs/2026-04-12T203658Z_ml_latest_small_svdpp_local_i5_2500k_24gb_s001/metrics.json`
- comparison baseline:
  `artifacts/runs/2026-04-12T201114Z_ml_latest_small_biased_mf_local_i5_2500k_24gb_s001/`

## Method

- load the canonical processed Parquet interactions for `ml_latest_small`
- construct a `benchmark_random_v1` split with `train_ratio=0.8`,
  `validation_ratio=0.1`, and split seed `1`
- define the `svdpp` implicit set as the unique training-rated items per user
  only
- train `svdpp` with `latent_dim=50`, `epochs=20`, `learning_rate=0.01`,
  `lambda_b=lambda_p=lambda_q=lambda_y=0.02`, model seed `1`, and `float32`
- write `config_snapshot.yaml`, `metrics.json`, `stdout.log`, and a validated
  `run_manifest.json`

## Readout

- status: `completed`
- train rows: `81,365`
- validation rows: `10,083`
- test rows: `9,388`
- train RMSE: `0.460875`
- validation RMSE: `0.867045`
- test RMSE: `0.885889`
- training wall-clock time: `1192.526193` seconds
- users with training history: `610`
- mean history size: `133.3852`
- max history size: `2154`

## Interpretation

The repository now supports a second full model family member with the same
artifact and governance contracts as the `biased_mf` baseline. On this local
POC dataset and seed, `svdpp` improves validation RMSE and test RMSE over the
existing `biased_mf` baseline, but the gain is small relative to the much higher
training cost.

Observed local delta versus the current `biased_mf` POC baseline:

- validation RMSE improved from `0.870649` to `0.867045`
- test RMSE improved from `0.890874` to `0.885889`
- training time increased from `112.448709` seconds to `1192.526193` seconds

This comparison is useful for local engineering direction, but it is still not
an official benchmark result for the final dataset ladder.

## Decision Or Next Step

- treat this run as the first stable local `svdpp` baseline artifact
- keep the `biased_mf` artifact as the direct regression reference
- next implementation target: `asymmetric_svd`, reusing the same processed-data,
  split, manifest, and report contracts
