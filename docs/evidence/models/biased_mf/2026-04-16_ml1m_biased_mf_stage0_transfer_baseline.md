# Evidence Note

## Scope

First `ml1m` scaling readout for the transferred `biased_mf` baseline profile.

## Claim Or Question

Does the cleanly transferred `biased_mf` profile from `ml100k` remain
operationally practical on `ml1m`, and what first quality and performance
baseline does it establish on the default local device?

## Inputs And Artifacts

- processed dataset manifest:
  `data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config:
  `configs/models/tuned/ml1m_biased_mf_stage0_transfer.yaml`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- run directory:
  `artifacts/runs/2026-04-16T013358Z_ml1m_biased_mf_local_i5_2500k_24gb_s001/`

## Method

- use the canonical `benchmark_random_v1` split family with `train=0.8`, `validation=0.1`, `test=0.1`
- keep seed discipline fixed at split seed `1` and model seed `1`
- run the transferred `ml100k` stage1-tuned `biased_mf` profile without any `ml1m`-specific tuning

## Readout

- status: `completed`
- manifest validation: `valid`
- Git state: dirty
- train RMSE: `0.643027`
- validation RMSE: `0.866678`
- test RMSE: `0.868475`
- training wall clock seconds: `56.582216`
- inference wall clock seconds: `2.102440`
- peak memory MB: `843.535156`
- peak memory delta MB: `634.671875`
- ratings per second during training: `353553.23`

## Interpretation

This is a strong operational baseline for `ml1m` on the default local device.
The transferred `biased_mf` profile remains fully runnable in well under two
minutes of training time and stays comfortably below the repository's local RAM
budget.

This result should still be treated as a dirty stage0 transfer readout, not as
a benchmark-final `ml1m` anchor. It exists to establish the first realistic
quality and runtime floor before any larger-factor or clustering-based model is
promoted on this dataset.

## Decision Or Next Step

- keep this run as the active `ml1m` baseline reference point
- use it as the cost and quality floor for clustering-based `ml1m` scaling probes
- next step: compare with bounded `cb_svdpp` and `cb_asvdpp` probes before attempting expensive full transfers again
