# CB Stage Profile G2

- date: `2026-05-02`
- status: `pass_for_instrumentation_and_ml100k_smoke`
- scope: `G2 stage-level CB profiling`
- gate: `docs/roadmaps/2026-05-02_claim_unlock_and_scalability_plan.md`

## Purpose

This note documents stage-level profiling instrumentation for `cb_svdpp` and
`cb_asvdpp`. It does not unlock any speed, scalability, quality, tuning,
large-dataset, or paper-faithfulness claim.

## Implementation

Implemented:

- reusable `StageProfiler`
- per-stage wall-clock timing
- per-stage RSS start/end/delta memory readout
- `profiling` block in `metrics.json`
- `profiling` block in `run_manifest.json`
- run-manifest schema support for stage profiling
- synthetic integration tests for `cb_svdpp` and `cb_asvdpp`
- development-only `ml100k` smoke profile:
  `configs/models/development/ml100k_cb_svdpp_stage_profile_smoke.yaml`

## ML100K Smoke Run

Run:

```text
artifacts/runs/2026-05-02T014708Z_ml100k_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001
```

Run manifest:

```text
artifacts/runs/2026-05-02T014708Z_ml100k_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/run_manifest.json
```

Command:

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main train-cb-svdpp data\processed\ml100k\ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json --model-config configs\models\development\ml100k_cb_svdpp_stage_profile_smoke.yaml --runtime-config configs\runtime\base.yaml --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml --split-family benchmark_random_v1 --train-ratio 0.8 --validation-ratio 0.1 --split-seed 1 --model-seed 1 --split-cache disable --disable-training-index-cache
```

Run context:

- dataset: `ml100k`
- model: `cb_svdpp`
- profile: `development-only stage_profile_smoke`
- split family: `benchmark_random_v1`
- split seed: `1`
- model seed: `1`
- epochs: `1`
- latent dimension: `16`
- user clusters: `8`
- item clusters: `8`
- device profile: `local_i5_2500k_24gb`
- git commit: `90ca7c3b08c0c106de4f2246a3403ddf1a3098b8`
- git dirty: `false`

## Stage Readout

The run manifest and metrics contain `profile_version: stage_profile_v1` with
`11` completed stages:

| Stage | Wall Clock Seconds | RSS Delta MB |
| --- | ---: | ---: |
| `data_load` | `0.668754` | `14.285156` |
| `split_resolution` | `0.123978` | `4.550781` |
| `config_build` | `0.000060` | `0.000000` |
| `cluster_induction` | `1.405690` | `30.457031` |
| `training_index_resolution` | `0.032132` | `0.011719` |
| `user_cluster_history_build` | `0.023521` | `0.218750` |
| `model_initialization` | `0.000061` | `0.000000` |
| `main_training` | `0.677701` | `0.167969` |
| `inference_train` | `0.095416` | `3.250000` |
| `inference_validation` | `0.002403` | `0.152344` |
| `inference_test` | `0.002252` | `0.000000` |

Aggregate profiler readout:

- stage count: `11`
- total profiled wall-clock seconds: `3.031969`
- peak process memory in system metrics: `208.347656 MB`

## Verification

Commands:

```powershell
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy src
.venv\Scripts\python.exe -m pytest
```

Readout:

- Ruff: `All checks passed!`
- Mypy source gate: `Success: no issues found in 60 source files`
- Pytest: `121 passed`

## Claim Boundary

Allowed claim:

- `G2` stage-level CB profiling instrumentation exists and writes stage
  profiles for `cb_svdpp` and `cb_asvdpp` runs.
- A clean `ml100k` development smoke run demonstrates the manifest and metrics
  readout.

Explicit non-claims:

- no speed claim
- no scalability claim
- no large-dataset profiling claim yet
- no model-quality claim from the smoke profile
- no tuning claim
- no paper-faithfulness claim
