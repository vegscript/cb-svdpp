# Evidence Note

## Scope

First end-to-end local proof-of-concept run for `biased_mf` on the local
`ml_latest_small` dataset contract.

## Claim Or Question

Can the repository execute a full, manifest-valid `biased_mf` training run from
processed Parquet input to stable run artifacts on the default local device
profile without violating the repo contracts?

## Inputs And Artifacts

- processed dataset manifest:
  `data/processed/ml_latest_small/ml_latest_small_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config: `configs/models/biased_mf.yaml`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- run directory:
  `artifacts/runs/2026-04-12T201114Z_ml_latest_small_biased_mf_local_i5_2500k_24gb_s001/`
- run manifest:
  `artifacts/runs/2026-04-12T201114Z_ml_latest_small_biased_mf_local_i5_2500k_24gb_s001/run_manifest.json`
- metrics:
  `artifacts/runs/2026-04-12T201114Z_ml_latest_small_biased_mf_local_i5_2500k_24gb_s001/metrics.json`

## Method

- load processed Parquet interactions from the canonical processed dataset
  manifest
- construct a `benchmark_random_v1` split with `train_ratio=0.8`,
  `validation_ratio=0.1`, and split seed `1`
- enforce train coverage for all users and rated items
- train `biased_mf` with `latent_dim=50`, `epochs=20`, `learning_rate=0.01`,
  `lambda_b=lambda_p=lambda_q=0.02`, model seed `1`, and `float32`
- write `config_snapshot.yaml`, `metrics.json`, `stdout.log`, and a validated
  `run_manifest.json`

## Readout

- status: `completed`
- train rows: `81,365`
- validation rows: `10,083`
- test rows: `9,388`
- train RMSE: `0.492186`
- validation RMSE: `0.870649`
- test RMSE: `0.890874`
- training wall-clock time: `112.448709` seconds

## Interpretation

The repo now supports a full local baseline execution path for `biased_mf` with
manifest-valid run outputs. This is a local POC result only. It is suitable for
pipeline validation, artifact inspection, and regression tracking, but it is not
an official benchmark result for the final dataset ladder.

The relatively large train-versus-validation gap is expected at this stage
because the current configuration is still a draft baseline and has not been
tuned.

## Decision Or Next Step

- treat this run as the first stable local baseline artifact
- use it as the regression reference for later `biased_mf` refactors
- next implementation target: `svdpp` on the same processed-data and artifact
  contracts
