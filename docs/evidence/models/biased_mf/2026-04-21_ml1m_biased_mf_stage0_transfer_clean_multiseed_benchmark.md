# Evidence Note

- date: `2026-04-21`
- scope: `ml1m` clean multi-seed benchmark anchor
- model: `biased_mf`
- config: `configs/models/tuned/ml1m_biased_mf_stage0_transfer.yaml`
- git_commit: `a9a45b996f81a8229b53ca2b52a3a0969371302c`
- git_dirty: `false`

## Purpose

Promote the transferred `ml1m` `biased_mf` profile from historical exploratory
readouts into a canonical clean benchmark artifact on the current repository
state and refresh the anchor on the same clean commit used for the matched
`cb_svdpp` comparison.

## Inputs

- processed manifest:
  `data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config:
  `configs/models/tuned/ml1m_biased_mf_stage0_transfer.yaml`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- split family: `benchmark_random_v1`
- split contract: `train_ratio=0.8`, `validation_ratio=0.1`
- split seeds: `1,2,3`
- model seed: `1`

## Artifacts

- clean seed run:
  `artifacts/runs/2026-04-21T193703Z_ml1m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/`
- clean seed run:
  `artifacts/runs/2026-04-21T193730Z_ml1m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s002_s001/`
- clean seed run:
  `artifacts/runs/2026-04-21T193756Z_ml1m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s003_s001/`
- clean multi-seed benchmark:
  `artifacts/benchmarks/2026-04-21T193857Z_ml1m_benchmark_random_v1_biased_mf_multiseed_s001_s002_s003_modelseed_s001_local_i5_2500k_24gb/`

## Readout

- train RMSE mean: `0.642804`
- train RMSE std: `0.000351`
- validation RMSE mean: `0.866357`
- validation RMSE std: `0.001873`
- test RMSE mean: `0.866615`
- test RMSE std: `0.001616`
- train time total mean seconds: `20.47`
- train time total std seconds: `4.43`
- peak memory mean MB: `777.09`

## Interpretation

- This artifact is the first canonical clean `ml1m` benchmark anchor on the
  current `main` branch that is explicitly matched to the clean `cb_svdpp`
  comparison commit.
- The quality readout is closely aligned with the older clean exploratory
  `biased_mf` seed sweep from commit `240afd8`, so the transferred baseline
  remains stable under the current repo state.
- The fit-time readout still differs materially from the older clean
  exploratory sweep. That is an observed benchmark difference across clean
  snapshots, not yet a validated optimization claim.

## Decision

- `configs/models/tuned/ml1m_biased_mf_stage0_transfer.yaml` is now the
  preferred clean `ml1m` baseline anchor in the repository.
- Future `ml1m` model comparisons should use this clean benchmark artifact as
  the reference baseline until a stronger baseline supersedes it.

## Next Step

- Keep this artifact as the matched clean control family for the current
  `ml1m` `cb_svdpp` benchmark anchor and any follow-up comparison study on the
  same contract.
