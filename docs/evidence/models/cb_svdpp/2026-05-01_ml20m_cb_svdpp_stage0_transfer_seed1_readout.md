# Evidence Note

- date: `2026-05-01`
- scope: `ml20m cb_svdpp stage0_transfer split-seed-1 readout`
- model: `cb_svdpp`
- config: `configs/models/tuned/ml20m_cb_svdpp_stage0_transfer.yaml`
- git_commit: `aa167dbe8df0a4dee0933612a3ea7c0c0dec7ffd`
- git_dirty: `false`
- status: `completed_single_split_seed`

## Purpose

Record the first authorized `ml20m cb_svdpp` matched-campaign split under the
2026-05-01 split-seed-1-only budget gate. This is an intermediate campaign
readout, not a final benchmark anchor and not a model-comparison claim.

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
  `1`
- model seed:
  `1`
- training index cache:
  `disabled`

## Artifacts

- run directory:
  `artifacts/runs/2026-05-01T063252Z_ml20m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/`
- run manifest:
  `artifacts/runs/2026-05-01T063252Z_ml20m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/run_manifest.json`
- metrics:
  `artifacts/runs/2026-05-01T063252Z_ml20m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json`

## Readout

- run id:
  `2026-05-01T063252Z_ml20m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001`
- manifest status:
  `completed`
- validation RMSE:
  `0.780558`
- test RMSE:
  `0.781255`
- train RMSE:
  `0.718271`
- cluster induction wall clock seconds:
  `417.503998`
- main training wall clock seconds:
  `19700.131893`
- fit-time total seconds:
  `20117.635891`
- train time per epoch:
  `985.006595`
- peak memory MB:
  `18586.136719`
- model size MB:
  `223.582520`

## Guardrail Assessment

The run completed with `git.dirty=false`, produced a valid run manifest and
`metrics.json`, and stayed below the local 80 percent RAM guardrail. The peak
memory readout is still close enough to the guardrail that the remaining split
seeds must be run sequentially and monitored.

## Claim Boundary

This readout does not unlock a final `ml20m` model-comparison claim. A
claim-eligible `ml20m cb_svdpp` anchor still requires completed split seeds
`1,2,3`, model seed `1`, canonical `benchmark-random-multiseed` aggregation,
and synchronized claim-matrix/report/evidence updates.
