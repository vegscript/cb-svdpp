# Evidence Note

- date: `2026-05-01`
- scope: `ml20m cb_svdpp stage0_transfer split-seed-2 readout`
- model: `cb_svdpp`
- config: `configs/models/tuned/ml20m_cb_svdpp_stage0_transfer.yaml`
- git_commit: `001f86f70298ed1848fbb4c8f5daeca24f62bb96`
- git_dirty: `false`
- status: `completed_single_split_seed`

## Purpose

Record the second authorized `ml20m cb_svdpp` matched-campaign split under the
2026-05-01 split-seed-2-only continuation gate. This is an intermediate
campaign readout, not a final benchmark anchor and not a model-comparison
claim.

## Input Contract

- processed manifest:
  `data/processed/ml20m/ml20m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config:
  `configs/models/tuned/ml20m_cb_svdpp_stage0_transfer.yaml`
- runtime config:
  `configs/runtime/base.yaml`
- device config:
  `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- split family:
  `benchmark_random_v1`
- split contract:
  `train_ratio=0.8`, `validation_ratio=0.1`
- split seed:
  `2`
- model seed:
  `1`
- training index cache:
  `disabled`

## Artifacts

- run directory:
  `artifacts/runs/2026-05-01T123151Z_ml20m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s002_s001/`
- run manifest:
  `artifacts/runs/2026-05-01T123151Z_ml20m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s002_s001/run_manifest.json`
- metrics:
  `artifacts/runs/2026-05-01T123151Z_ml20m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s002_s001/metrics.json`

## Readout

- run id:
  `2026-05-01T123151Z_ml20m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s002_s001`
- manifest status:
  `completed`
- validation RMSE:
  `0.781702`
- test RMSE:
  `0.781773`
- train RMSE:
  `0.717938`
- cluster induction wall clock seconds:
  `429.808874`
- main training wall clock seconds:
  `20286.319446`
- fit-time total seconds:
  `20716.128320`
- train time per epoch:
  `1014.315972`
- peak memory MB:
  `19128.371094`
- model size MB:
  `223.440536`

## Guardrail Assessment

The run completed with `git.dirty=false`, produced a valid run manifest and
`metrics.json`, and stayed below the local 80 percent RAM guardrail. The peak
memory margin is narrow, so split seed `3` may only run as a monitored final
single split.

## Claim Boundary

This readout does not unlock a final `ml20m` model-comparison claim. A
claim-eligible `ml20m cb_svdpp` anchor still requires completed split seeds
`1,2,3`, model seed `1`, canonical `benchmark-random-multiseed` aggregation,
and synchronized claim-matrix/report/evidence updates.
