# Evidence Note

- date: `2026-04-24`
- scope: `ml10m` single-seed feasibility baseline
- model: `biased_mf`
- config: `configs/models/tuned/ml10m_biased_mf_stage0_transfer.yaml`
- git_commit: `56d9161fe528786a50768d03762da4ccac1f8334`
- git_dirty: `false`

## Purpose

Establish the first clean `ml10m` model-run feasibility readout under the
canonical `benchmark_random_v1` split, using the cheapest baseline family in
the current publish-scope plan.

## Inputs

- processed manifest:
  `data/processed/ml10m/ml10m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config:
  `configs/models/tuned/ml10m_biased_mf_stage0_transfer.yaml`
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
  `artifacts/runs/2026-04-24T171706Z_ml10m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/`
- run manifest:
  `artifacts/runs/2026-04-24T171706Z_ml10m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/run_manifest.json`
- metrics:
  `artifacts/runs/2026-04-24T171706Z_ml10m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json`
- run manifest validation:
  `valid`

## Method

- commit and push the explicit `ml10m` transfer config before execution
- verify `main` is clean and synchronized before execution
- run the canonical wrapper command:
  `.\scripts\train_biased_mf.ps1 -ProcessedManifest data\processed\ml10m\ml10m_benchmark_random_v1_explicit_v1_float32_manifest.json -ModelConfig configs\models\tuned\ml10m_biased_mf_stage0_transfer.yaml -RuntimeConfig configs\runtime\base.yaml -DeviceConfig configs\runtime\devices\local_i5_2500k_24gb.yaml -TrainRatio 0.8 -ValidationRatio 0.1 -SplitSeed 1 -ModelSeed 1`
- validate the generated run manifest through the canonical manifest validator

## Readout

- status:
  `completed`
- Git dirty:
  `false`
- training backend effective:
  `numba`
- interactions:
  `10000054`
- train rows:
  `8000065`
- validation rows:
  `1000005`
- test rows:
  `999984`
- train RMSE:
  `0.665679`
- validation RMSE:
  `0.786906`
- test RMSE:
  `0.786747`
- training wall clock seconds:
  `157.857385`
- inference wall clock seconds:
  `5.621301`
- train time per epoch seconds:
  `6.314295`
- peak memory MB:
  `6634.933594`
- model size MB:
  `19.974041`
- ratings per second train:
  `1266976.674842`
- ratings per second inference:
  `1778957.188397`

## Interpretation

This run proves that the current `biased_mf` stage0 transfer profile can
complete one clean `ml10m` training/evaluation run on the local
`local_i5_2500k_24gb` device profile with a valid run manifest.

This is not a final benchmark anchor. It is single-seed only, not tuned on
`ml10m`, not a multi-seed estimate, not a model comparison, and not a
paper-faithful reproduction claim.

The peak memory readout supports local feasibility for this specific baseline
run. It must not be generalized to clustering-based models, `svdpp`,
`cb_svdpp`, `cb_asvdpp`, or `ml20m` without separate measurements.

## Decision

- Mark `ml10m biased_mf` benchmark evidence as `single_seed_feasibility`.
- Keep final `ml10m` model-comparison claims blocked until matched clean
  evidence exists for the compared model family and seed policy.
- Use this run to decide whether the next `ml10m` model step should be a
  cautious clustering-model feasibility probe or whether `ml20m` data evidence
  should be prioritized first.
