# Evidence Note

- date: `2026-04-24`
- scope: `ml20m` single-epoch clustering-model feasibility probe
- model: `cb_svdpp`
- config: `configs/models/tuned/ml20m_cb_svdpp_stage0_probe_e001.yaml`
- git_commit: `6ccef25dd6799cf25999f24f73d0e32b797fd626`
- git_dirty: `false`

## Purpose

Measure whether the clustering-based `cb_svdpp` path can complete a bounded
`ml20m` run on the local `local_i5_2500k_24gb` device profile, after `ml20m`
data ingestion and the `ml20m biased_mf` feasibility baseline were completed.

## Inputs

- processed manifest:
  `data/processed/ml20m/ml20m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config:
  `configs/models/tuned/ml20m_cb_svdpp_stage0_probe_e001.yaml`
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
- split cache:
  auto policy, effective disabled
- training index cache:
  disabled

## Artifacts

- run directory:
  `artifacts/runs/2026-04-24T181239Z_ml20m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/`
- run manifest:
  `artifacts/runs/2026-04-24T181239Z_ml20m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/run_manifest.json`
- metrics:
  `artifacts/runs/2026-04-24T181239Z_ml20m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json`
- run manifest validation:
  `valid`

## Method

- commit and push the explicit `ml20m` probe config before execution
- verify `main` is clean and synchronized before execution
- run one bounded epoch to measure local feasibility without treating the probe
  as a full-transfer benchmark
- run the canonical wrapper command:
  `.\scripts\train_cb_svdpp.ps1 data\processed\ml20m\ml20m_benchmark_random_v1_explicit_v1_float32_manifest.json --model-config configs\models\tuned\ml20m_cb_svdpp_stage0_probe_e001.yaml --runtime-config configs\runtime\base.yaml --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml --split-family benchmark_random_v1 --train-ratio 0.8 --validation-ratio 0.1 --split-seed 1 --model-seed 1 --disable-training-index-cache`
- monitor the process during execution because RAM usage is part of the
  feasibility question
- validate the generated run manifest through the canonical manifest validator

## Readout

- status:
  `completed`
- Git dirty:
  `false`
- interactions:
  `20000263`
- train rows:
  `16001106`
- validation rows:
  `2000026`
- test rows:
  `1999131`
- epochs:
  `1`
- user clusters:
  `80`
- item clusters:
  `80`
- `r_star` density:
  `0.987812`
- induction train RMSE:
  `0.866077`
- train RMSE:
  `0.857491`
- validation RMSE:
  `0.863001`
- test RMSE:
  `0.863991`
- cluster induction wall clock seconds:
  `195.149008`
- main training wall clock seconds:
  `983.076083`
- effective fit time seconds:
  `1178.225090`
- inference wall clock seconds:
  `277.892600`
- peak memory MB:
  `17876.066406`
- model size MB:
  `212.180618`
- ratings per second train:
  `16276.569315`
- ratings per second inference:
  `71971.196708`

## Interpretation

This run proves that the current `cb_svdpp` implementation can complete one
bounded `ml20m` clustering-model probe on the local device with a valid run
manifest and a clean Git snapshot.

This is not a final benchmark anchor. It is single-seed, single-epoch,
not tuned on `ml20m`, and not matched in training budget to the completed
`ml20m biased_mf` feasibility baseline. Therefore the RMSE values are
diagnostic only and must not be used for a final model-comparison claim.

The peak memory readout is below the local 24 GB device capacity and below the
80 percent guardrail of about 19.2 GB, but it is close enough that deeper
`ml20m` clustering runs should not be started casually on this machine. The
measured effective fit time is about 19.64 minutes for one epoch plus
one-epoch induction; full-transfer feasibility remains a separate hardware and
time-budget decision.

## Decision

- Mark `ml20m cb_svdpp` as `single_epoch_feasibility`.
- Keep final `ml20m` model-comparison claims blocked.
- Do not run a full `ml20m cb_svdpp` transfer on the local device unless the
  plan explicitly accepts a multi-hour run with elevated RAM pressure or moves
  the full transfer to a stronger device.
- The next project step should be a final claim/readiness matrix or an explicit
  hardware deferral decision for deeper large-scale clustering runs.
