# Evidence Note

## Scope

First official `MovieLens 100K` run for `biased_mf` on the canonical
`paper_faithful_ml100k_v1` split path using fold `u1`.

## Claim Or Question

Can the repository execute `biased_mf` on the official `ml100k` split files
without introducing a synthetic validation partition, and what baseline test
RMSE does that produce under the current draft configuration?

## Inputs And Artifacts

- processed dataset manifest:
  `data/processed/ml100k/ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json`
- split family: `paper_faithful_ml100k_v1`
- fold: `u1`
- model config: `configs/models/biased_mf.yaml`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- run directory:
  `artifacts/runs/2026-04-13T020700Z_ml100k_biased_mf_local_i5_2500k_24gb_s001/`
- run manifest:
  `artifacts/runs/2026-04-13T020700Z_ml100k_biased_mf_local_i5_2500k_24gb_s001/run_manifest.json`
- run metrics:
  `artifacts/runs/2026-04-13T020700Z_ml100k_biased_mf_local_i5_2500k_24gb_s001/metrics.json`

## Method

- use the official `u1.base` and `u1.test` files from the raw `ml100k` package
- map them back to the processed interaction table through the raw
  `(user, item, rating, timestamp)` identity
- train `biased_mf` with `latent_dim=50`, `epochs=20`, `learning_rate=0.01`,
  `lambda_b=lambda_p=lambda_q=0.02`, model seed `1`, and `float32`
- report only `train` and `test` metrics because the official split path has no
  canonical validation partition

## Readout

- status: `completed`
- split family: `paper_faithful_ml100k_v1`
- fold: `u1`
- train rows: `80,000`
- test rows: `20,000`
- validation rows: `0`
- train RMSE: `0.558853`
- test RMSE: `0.959989`
- training wall-clock time: `276.497542` seconds

## Interpretation

The repository now has its first official `ml100k` baseline on the actual split
family that belongs to this dataset package. This matters more than the
absolute number alone, because it removes the earlier `ml_latest_small`
comparability objection.

The result is still only a first draft baseline:

- only fold `u1` has been run so far
- only one model seed has been used
- the model config is still a draft default profile, not a tuned benchmark
  profile

So this run is benchmark-relevant, but not benchmark-final.

## Decision Or Next Step

- keep this run as the first official `ml100k` `biased_mf` baseline artifact
- use it as the direct comparison anchor for official `svdpp` runs on the same
  fold
- next step: compare `svdpp` on the same `u1` split, then expand to the full
  `u1` to `u5` ladder
