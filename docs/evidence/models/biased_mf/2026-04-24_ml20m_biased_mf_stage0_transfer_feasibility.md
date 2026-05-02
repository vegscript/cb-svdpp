# Evidence Note

- date: `2026-04-24`
- scope: `ml20m` single-seed feasibility baseline
- model: `biased_mf`
- config: `configs/models/tuned/ml20m_biased_mf_stage0_transfer.yaml`
- git_commit: `e49f36dabc2c040df30e1079a0dd9111f9ebeb8b`
- git_dirty: `false`

## Purpose

Establish the first clean `ml20m` model-run feasibility readout under the
canonical `benchmark_random_v1` split, using the cheapest baseline family in
the current publish-scope plan.

## Inputs

- processed manifest:
  `data/processed/ml20m/ml20m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config:
  `configs/models/tuned/ml20m_biased_mf_stage0_transfer.yaml`
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

## Artifacts

- run directory:
  `artifacts/runs/2026-04-24T175832Z_ml20m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/`
- run manifest:
  `artifacts/runs/2026-04-24T175832Z_ml20m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/run_manifest.json`
- metrics:
  `artifacts/runs/2026-04-24T175832Z_ml20m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json`
- run manifest validation:
  `valid`

## Method

- commit and push the explicit `ml20m` transfer config before execution
- verify `main` is clean and synchronized before execution
- run the canonical wrapper command:
  `.\scripts\train_biased_mf.ps1 -ProcessedManifest data\processed\ml20m\ml20m_benchmark_random_v1_explicit_v1_float32_manifest.json -ModelConfig configs\models\tuned\ml20m_biased_mf_stage0_transfer.yaml -RuntimeConfig configs\runtime\base.yaml -DeviceConfig configs\runtime\devices\local_i5_2500k_24gb.yaml -TrainRatio 0.8 -ValidationRatio 0.1 -SplitSeed 1 -ModelSeed 1`
- validate the generated run manifest through the canonical manifest validator

## Readout

- status:
  `completed`
- Git dirty:
  `false`
- training backend effective:
  `numba`
- interactions:
  `20000263`
- users:
  `138493`
- rated items:
  `26744`
- train rows:
  `16001106`
- validation rows:
  `2000026`
- test rows:
  `1999131`
- train RMSE:
  `0.671047`
- validation RMSE:
  `0.774734`
- test RMSE:
  `0.775594`
- training wall clock seconds:
  `329.954495`
- inference wall clock seconds:
  `10.681993`
- train time per epoch seconds:
  `13.198180`
- peak memory MB:
  `13055.300781`
- model size MB:
  `40.971394`
- ratings per second train:
  `1212372.180613`
- ratings per second inference:
  `1872334.421790`

## Interpretation

This run proves that the current `biased_mf` stage0 transfer profile can
complete one clean `ml20m` training/evaluation run on the local
`local_i5_2500k_24gb` device profile with a valid run manifest.

This is not a final benchmark anchor. It is single-seed only, not tuned on
`ml20m`, not a multi-seed estimate, not a clustering-model comparison, and not
a paper-faithful reproduction claim.

The peak memory readout remains below the local 24 GB device capacity for this
specific baseline run. It must not be generalized to `svdpp`, `cb_svdpp`,
`cb_asvdpp`, or any deeper `ml20m` model run without separate measurements.

## Decision

- Mark `ml20m biased_mf` benchmark evidence as `single_seed_feasibility`.
- Keep final `ml20m` model-comparison claims blocked until matched clean
  evidence exists for the compared model family and seed policy.
- Use this run as the baseline feasibility readout when deciding whether a
  bounded `ml20m` clustering-model probe is acceptable on local hardware.
