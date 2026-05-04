# ML10M CB-SVD++ Large Stage Profile G2

- date: `2026-05-02`
- status: `pass_for_bounded_large_dataset_stage_profile`
- scope: `G2 large-dataset CB stage profiling`
- model: `cb_svdpp`
- dataset: `ml10m`
- gate: `docs/roadmaps/2026-05-02_claim_unlock_and_scalability_plan.md`

## Purpose

This note records a bounded large-dataset stage profile after the G4 cache
plumbing. It identifies the measured hot path for the next remediation step. It
does not unlock any speed, scalability, quality, tuning, large-dataset model
comparison, production-readiness, or paper-faithfulness claim.

## Pre-Run Integrity Fixes

Two discarded local starts are not used as evidence:

- A first start had an incorrectly quoted `RECSYS_CACHE_ROOT` and was stopped.
- A second start exposed a Windows path-length failure in the training-index
  cache path.

Before the evidence run, both issues were fixed and pushed:

- `f184a03 Make CB run commands replayable`
- `2a48609 Shorten training index cache paths`

Only the completed run on commit `2a48609` is used below.

## Command

Environment:

```powershell
$env:RECSYS_CACHE_ROOT = "artifacts/local/g2ml10m_2a48609"
```

Command:

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main train-cb-svdpp data\processed\ml10m\ml10m_benchmark_random_v1_explicit_v1_float32_manifest.json --model-config configs\models\tuned\ml10m_cb_svdpp_stage0_probe_e001.yaml --runtime-config configs\runtime\base.yaml --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml --split-family benchmark_random_v1 --train-ratio 0.8 --validation-ratio 0.1 --split-seed 1 --model-seed 1 --split-cache enable --training-index-cache --cluster-artifact-cache
```

The run manifest command is replayable and contains the positional processed
manifest path plus explicit split ratios and cache flags.

## Run Artifacts

- run directory:
  `artifacts/runs/2026-05-02T101352Z_ml10m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001`
- run manifest:
  `artifacts/runs/2026-05-02T101352Z_ml10m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/run_manifest.json`
- metrics:
  `artifacts/runs/2026-05-02T101352Z_ml10m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json`
- cache root:
  `artifacts/local/g2ml10m_2a48609`
- manifest validation:
  `valid`

## Run Context

- status: `completed`
- git commit: `2a4860999e53d1463d7f0d80d16b97aab41f0ae5`
- git dirty: `false`
- device profile: `local_i5_2500k_24gb`
- profile contract: `claim_eligible=true`
- interactions: `10000054`
- train rows: `8000065`
- validation rows: `1000005`
- test rows: `999984`
- epochs: `1`
- latent dimension: `64`
- user clusters: `80`
- item clusters: `80`
- split cache status: `miss`
- training user-history cache status: `miss`
- cluster artifact cache status: `miss`
- user-cluster-history cache status: `miss`

## Stage Readout

The run manifest and metrics contain `profile_version: stage_profile_v1` with
`11` completed stages:

| Stage | Wall Clock Seconds | RSS Delta MB | Cache Status |
| --- | ---: | ---: | --- |
| `data_load` | `0.950285` | `285.355469` | |
| `split_resolution` | `21.969240` | `81.441406` | `miss` |
| `config_build` | `0.000069` | `0.000000` | |
| `cluster_induction` | `112.783847` | `117.917969` | `miss` |
| `training_index_resolution` | `19.807804` | `33.289062` | `miss` |
| `user_cluster_history_build` | `3.515554` | `17.867188` | `miss` |
| `model_initialization` | `0.000077` | `0.000000` | |
| `main_training` | `498.060299` | `20.644531` | |
| `inference_train` | `15.146952` | `105.828125` | |
| `inference_validation` | `1.079297` | `15.265625` | |
| `inference_test` | `0.954000` | `15.265625` | |

Aggregate readout:

- stage count: `11`
- total profiled wall-clock seconds: `674.267424`
- effective fit time seconds: `610.844200`
- peak process memory: `12693.964844 MB`
- train RMSE: `0.865231`
- validation RMSE: `0.872094`
- test RMSE: `0.871385`

## Interpretation

Measured hot-path order for this bounded run:

- `main_training`: `498.060299` seconds, about `73.87%` of profiled time
- `cluster_induction`: `112.783847` seconds, about `16.73%` of profiled time
- split and index/cache construction combined: about `45.292599` seconds

This makes `main_training` the next remediation target. Cluster induction is
still material, but the G4 cache path already makes repeated lookup explicit and
manifest-visible. Any claim that a change is faster requires a clean
before/after benchmark against this or a deliberately equivalent baseline.

## Verification

Commands run before the evidence run on the same code lineage:

```powershell
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy src
.venv\Scripts\python.exe -m pytest
```

Readout:

- Ruff: `All checks passed!`
- Mypy source gate: `Success: no issues found in 62 source files`
- Pytest: `127 passed`

Manifest validation:

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main validate-manifest artifacts\runs\2026-05-02T101352Z_ml10m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001\run_manifest.json
```

Readout:

- status: `valid`

## Claim Boundary

Allowed claim:

- A bounded `ml10m cb_svdpp` run now has manifest-valid stage-level profiling
  on a clean commit.
- The measured next remediation target is `main_training`.

Explicit non-claims:

- no speed claim
- no scalability claim
- no production-readiness claim
- no SOTA claim
- no final `ml10m` model-comparison claim
- no hyperparameter-tuning claim
- no paper-faithfulness claim
