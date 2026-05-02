# Evidence Note

- date: `2026-05-02`
- scope: `ml20m cb_svdpp stage0_transfer split-seed-3 guardrail breach`
- model: `cb_svdpp`
- config: `configs/models/tuned/ml20m_cb_svdpp_stage0_transfer.yaml`
- git_commit: `1cb39de0fbd8ca05644f62d03903eeb172fa9fee`
- git_dirty: `false`
- status: `completed_guardrail_breach_negative_evidence`

## Purpose

Record the third authorized `ml20m cb_svdpp` matched-campaign split under the
2026-05-01 split-seed-3-only continuation gate. The CLI run completed and
produced a valid manifest, but the measured peak memory crossed the documented
local 80 percent RAM guardrail. Under the campaign contract, this stop gate
demotes the run to negative resource evidence.

This note is not a final benchmark anchor and does not unlock any final
`ml20m` model-comparison claim.

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
  `3`
- model seed:
  `1`
- training index cache:
  `disabled`

## Artifacts

- wrapper stdout:
  `artifacts/runs/ml20m_cb_svdpp_stage0_transfer_split_seed3_model_seed1_logs/train_20260501_203305.out.log`
- wrapper stderr:
  `artifacts/runs/ml20m_cb_svdpp_stage0_transfer_split_seed3_model_seed1_logs/train_20260501_203305.err.log`
- monitor log:
  `artifacts/runs/ml20m_cb_svdpp_stage0_transfer_split_seed3_model_seed1_logs/train_20260501_203305.monitor.csv`
- run directory:
  `artifacts/runs/2026-05-01T183320Z_ml20m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s003_s001/`
- run manifest:
  `artifacts/runs/2026-05-01T183320Z_ml20m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s003_s001/run_manifest.json`
- metrics:
  `artifacts/runs/2026-05-01T183320Z_ml20m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s003_s001/metrics.json`

## Readout

- wrapper exit code:
  `0`
- run id:
  `2026-05-01T183320Z_ml20m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s003_s001`
- manifest status:
  `completed`
- validation RMSE:
  `0.781010`
- test RMSE:
  `0.781511`
- train RMSE:
  `0.718401`
- cluster induction wall clock seconds:
  `481.187380`
- main training wall clock seconds:
  `19884.330198`
- fit-time total seconds:
  `20365.517578`
- train time per epoch:
  `994.216510`
- inference wall clock seconds:
  `217.306353`
- peak memory MB:
  `19898.871094`
- model size MB:
  `223.013821`

## Guardrail Assessment

The run completed with `git.dirty=false`, produced a valid run manifest and
`metrics.json`, and emitted no stderr. However, the local device has
`24559.11 MB` visible RAM, so the repository's documented 80 percent guardrail
is approximately `19647.29 MB`. The run's measured `peak_memory_mb` was
`19898.871094 MB`, exceeding the guardrail by approximately `251.58 MB`.

No system instability was observed in the monitor log, but the campaign
contract treats a peak-memory crossing as a stop gate. Therefore this run is
negative resource evidence and must not be folded into a final benchmark
matrix or used for a final `ml20m` model ranking.

## Claim Boundary

This readout does not unlock a final `ml20m` model-comparison claim. The local
`ml20m cb_svdpp stage0_transfer` matched campaign is not claim-eligible under
the current local 24 GB device contract because split seed `3` breached the
memory guardrail before any canonical final aggregation step.

Any future `ml20m cb_svdpp` promotion attempt requires a stronger device
profile or a documented lower-memory profile, plus a new explicit campaign
contract and fresh clean-run evidence.
