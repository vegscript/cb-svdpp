# Evidence Note

- date: `2026-04-21`
- scope: `ml1m` clean multi-seed benchmark anchor
- model: `cb_svdpp`
- config: `configs/models/tuned/ml1m_cb_svdpp_stage0_transfer.yaml`
- git_commit: `a9a45b996f81a8229b53ca2b52a3a0969371302c`
- git_dirty: `false`

## Purpose

Promote the transferred `ml1m` `cb_svdpp` profile from historical exploratory
and single-seed evidence into a canonical clean multi-seed benchmark artifact
on the same clean commit as the matched `biased_mf` control family.

## Inputs

- processed manifest:
  `data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config:
  `configs/models/tuned/ml1m_cb_svdpp_stage0_transfer.yaml`
- matched control config:
  `configs/models/tuned/ml1m_biased_mf_stage0_transfer.yaml`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- split family: `benchmark_random_v1`
- split contract: `train_ratio=0.8`, `validation_ratio=0.1`
- split seeds: `1,2,3`
- model seed: `1`

## Artifacts

- clean seed run:
  `artifacts/runs/2026-04-21T183928Z_ml1m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/`
- clean seed run:
  `artifacts/runs/2026-04-21T185653Z_ml1m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s002_s001/`
- clean seed run:
  `artifacts/runs/2026-04-21T191700Z_ml1m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s003_s001/`
- clean multi-seed benchmark:
  `artifacts/benchmarks/2026-04-21T193859Z_ml1m_benchmark_random_v1_cb_svdpp_multiseed_s001_s002_s003_modelseed_s001_local_i5_2500k_24gb/`
- matched clean control benchmark:
  `artifacts/benchmarks/2026-04-21T193857Z_ml1m_benchmark_random_v1_biased_mf_multiseed_s001_s002_s003_modelseed_s001_local_i5_2500k_24gb/`

## Readout

- train RMSE mean: `0.724537`
- train RMSE std: `0.000468`
- validation RMSE mean: `0.857005`
- validation RMSE std: `0.001719`
- test RMSE mean: `0.857365`
- test RMSE std: `0.001689`
- train time total mean seconds: `1082.96`
- train time total std seconds: `85.67`
- peak memory mean MB: `1455.98`

## Matched Control Delta: `cb_svdpp - biased_mf`

- validation RMSE delta: `-0.009352`
- test RMSE delta: `-0.009251`
- fit-time ratio: `52.91x`
- peak-memory ratio: `1.87x`

## Interpretation

- This is the first clean benchmark-level `ml1m` comparison in the repository
  that is both multi-seed and matched on the same clean commit.
- Under that contract, the transferred `cb_svdpp` profile beats the matched
  `biased_mf` control on both validation RMSE and test RMSE across three split
  seeds.
- The systems tradeoff remains substantial: `cb_svdpp` is far more expensive in
  fit time and materially heavier in memory on the same device profile.
- The current fit-time readout is much lower than the older clean single-seed
  `cb_svdpp` comparison from commit `2c6ffa1`. That difference should not be
  narrated as a validated speed improvement without a dedicated cross-commit
  performance study.

## Decision

- `configs/models/tuned/ml1m_cb_svdpp_stage0_transfer.yaml` is now promoted to
  the preferred clean `ml1m` `cb_svdpp` benchmark anchor in the repository.
- `cb_svdpp` may now be compared against the matched clean `biased_mf` baseline
  at benchmark level for `ml1m`, as long as quality and systems tradeoffs stay
  explicitly paired.

## Next Step

- Decide whether the next repo priority is
  `cb_asvdpp` on `ml1m`,
  later-dataset ladder closure,
  or final comparison tables for the report.
