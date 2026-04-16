# Evidence Note

## Scope

Clean control baseline for `biased_mf` on `MovieLens 1M` under the canonical
`benchmark_random_v1` split.

## Claim Or Question

Provide a claim-faehige clean control run on the same dataset, split, device,
and seed contract as the clean `cb_svdpp` confirmatory run.

## Inputs And Artifacts

- processed dataset manifest:
  `data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config:
  `configs/models/tuned/ml1m_biased_mf_stage0_transfer.yaml`
- run directory:
  `artifacts/runs/2026-04-16T081023Z_ml1m_biased_mf_local_i5_2500k_24gb_s001/`
- run manifest validation: `valid`
- clean snapshot commit: `2c6ffa1e1b19ac9cc0952f96b8ccac8dbb1ff656`

## Method

- run `biased_mf` on `ml1m` with the canonical random benchmark split
- use `train_ratio=0.8`, `validation_ratio=0.1`, split seed `1`, and model seed `1`
- run on device profile `local_i5_2500k_24gb`
- use the clean cloned snapshot with `dirty=false`

## Readout

- status: `completed`
- Git dirty: `false`
- train RMSE: `0.643027`
- validation RMSE: `0.866678`
- test RMSE: `0.868475`
- training wall clock seconds: `43.291659`
- inference wall clock seconds: `1.919164`
- peak memory MB: `849.289063`
- model size MB: `2.416573`

## Interpretation

This run provides the clean control point for direct comparison against the
clean `cb_svdpp` confirmatory run on the same `ml1m` split and device setup.
It is a valid single-seed baseline, but not yet a multi-seed final result.

## Decision Or Next Step

- use this run as the clean `biased_mf` control baseline for all immediate
  `ml1m` single-seed comparisons
- keep any stronger cross-model claim conservative until multi-seed clean runs
  exist for the compared models