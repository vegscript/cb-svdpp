# Tune-Inner Cache Controls And Bounded CB-SVD++ Selection G5

- date: `2026-05-02`
- status: `pass_for_bounded_validation_only_selection_probe`
- scope: `G5 tune-inner cache controls and bounded cb_svdpp alpha/cluster tuning`
- gate: `docs/roadmaps/2026-05-02_claim_unlock_and_scalability_plan.md`

## Purpose

This note records generic `tune-inner` cache controls and one bounded
validation-only `ml100k cb_svdpp` selection probe. It does not unlock a final
quality, large-dataset, scalability, production-readiness, SOTA, or
paper-faithfulness claim.

## Implementation

Implemented:

- `tune-inner` and `tune-ml100k-inner` expose explicit `--split-cache`,
  `--training-index-cache`, and `--cluster-artifact-cache` controls.
- `run_inner_tuning` forwards cache controls only to model runners that support
  them.
- Unsupported cache controls are rejected instead of silently ignored.
- Tuning summaries and config snapshots include the effective cache policy.
- Candidate config filenames are shortened deterministically for Windows path
  safety.
- Bounded validation-only config:
  `configs/experiments/tuning/ml100k_cb_svdpp_g5_bounded_alpha_cluster.yaml`

Code commits:

- `6b15a08 Add tune-inner cache controls`
- `ca80624 Shorten tune-inner candidate config paths`
- `29bb460 Tighten tune-inner candidate path guard`

## Pre-Run Integrity Notes

Two discarded starts are not evidence:

- A first clean-worktree start used the editable environment without
  `PYTHONPATH` pinned to the clean worktree, so it imported the main worktree
  source path.
- A second clean-worktree start exposed the Windows path-length issue in
  candidate config filenames.

Only the completed run on commit `29bb460` is used below.

## Command

The evidence run used a separate clean Git worktree. The local absolute
worktree path is intentionally omitted from this public note; the clean
worktree was pinned to commit `29bb460`.

```text
<clean-worktree>
```

Environment:

```powershell
$env:PYTHONPATH = "<clean-worktree>\src"
$env:RECSYS_CACHE_ROOT = "artifacts/local/g5_ml100k_tune_29bb460"
```

Command:

```powershell
<python> -m recsys_lab.cli.main tune-inner configs\experiments\tuning\ml100k_cb_svdpp_g5_bounded_alpha_cluster.yaml data\processed\ml100k\ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json --runtime-config configs\runtime\base.yaml --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml --split-cache enable --training-index-cache --cluster-artifact-cache
```

The completed artifacts were copied unchanged from the clean worktree into the
main repo's ignored `artifacts/` tree for local evidence reference.

## Run Artifacts

- benchmark directory:
  `artifacts/benchmarks/2026-05-02T111728Z_ml100k_inner_tuning_cb_svdpp_g5_bounded_alpha_cluster_local_i5_2500k_24gb`
- benchmark manifest:
  `artifacts/benchmarks/2026-05-02T111728Z_ml100k_inner_tuning_cb_svdpp_g5_bounded_alpha_cluster_local_i5_2500k_24gb/benchmark_manifest.json`
- summary:
  `artifacts/benchmarks/2026-05-02T111728Z_ml100k_inner_tuning_cb_svdpp_g5_bounded_alpha_cluster_local_i5_2500k_24gb/summary.json`
- manifest validation: `valid`

## Run Context

- status: `completed`
- git commit: `29bb4600006d2af59fafbef868524cb5dc2dd19b`
- git dirty: `false`
- dataset: `ml100k`
- model: `cb_svdpp`
- split family: `benchmark_random_v1`
- selection unit: `s001`
- train ratio: `0.8`
- validation ratio: `0.1`
- model seed: `1`
- latent dimension: `32`
- epochs per candidate: `2`
- objective: `validation_rmse_mean`
- split cache: `enable`
- training-index cache: `enabled`
- cluster-artifact cache: `enabled`
- candidates: `6`

## Validation-Only Guard

Every candidate run reports:

- `test_metrics_available`: `false`
- `test_rmse`: `None`
- `git.dirty`: `false`

Therefore this run is selection evidence only. Test data was not used for model
selection or ranking.

## Candidate Readout

| Rank | Candidate | Validation RMSE Mean | Mean Fit Seconds |
| --- | --- | ---: | ---: |
| 1 | `rank032_uc064_ic064_a000_lr0100_reg0020_e002` | `0.960717726` | `7.262972` |
| 2 | `rank032_uc064_ic064_a005_lr0100_reg0020_e002` | `0.961989637` | `4.568154` |
| 3 | `rank032_uc032_ic032_a010_lr0100_reg0020_e002` | `0.962415506` | `5.215376` |
| 4 | `rank032_uc064_ic064_a010_lr0100_reg0020_e002` | `0.962828462` | `2.690385` |
| 5 | `rank032_uc080_ic080_a010_lr0100_reg0020_e002` | `0.962860503` | `3.524465` |
| 6 | `rank032_uc064_ic064_a015_lr0100_reg0020_e002` | `0.963316563` | `2.471847` |

Bounded selection winner:

- candidate: `rank032_uc064_ic064_a000_lr0100_reg0020_e002`
- validation RMSE mean: `0.960717726`

## Cache Readout

The six candidate runs reported these cache statuses:

| Candidate Order | Cluster Artifacts | Training User History | User Cluster History |
| ---: | --- | --- | --- |
| 1 | `miss` | `miss` | `miss` |
| 2 | `hit` | `hit` | `hit` |
| 3 | `hit` | `hit` | `hit` |
| 4 | `hit` | `hit` | `hit` |
| 5 | `miss` | `hit` | `miss` |
| 6 | `miss` | `hit` | `miss` |

This verifies that repeated alpha-only candidates reuse split, training-index,
cluster, and cluster-history artifacts, while cluster-count changes correctly
invalidate the cluster-dependent caches.

## Verification

Focused commands before the evidence run:

```powershell
.venv\Scripts\python.exe -m ruff check src\recsys_lab\cli\main.py src\recsys_lab\experiments\ml100k_inner_tuning.py tests\unit\test_cli_main.py tests\integration\test_ml100k_inner_tuning.py
.venv\Scripts\python.exe -m pytest tests\unit\test_cli_main.py tests\integration\test_ml100k_inner_tuning.py
```

Readout:

- Ruff: `All checks passed!`
- Pytest: `13 passed`

Focused path-guard verification after the failed path-length start:

```powershell
.venv\Scripts\python.exe -m ruff check src\recsys_lab\experiments\ml100k_inner_tuning.py tests\integration\test_ml100k_inner_tuning.py
.venv\Scripts\python.exe -m pytest tests\integration\test_ml100k_inner_tuning.py
```

Readout:

- Ruff: `All checks passed!`
- Pytest: `7 passed`

Manifest validation:

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main validate-manifest artifacts\benchmarks\2026-05-02T111728Z_ml100k_inner_tuning_cb_svdpp_g5_bounded_alpha_cluster_local_i5_2500k_24gb\benchmark_manifest.json
```

Readout:

- status: `valid`

Full verification commands after the evidence and roadmap updates:

```powershell
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy src
.venv\Scripts\python.exe -m pytest
```

Readout:

- Ruff: `All checks passed!`
- Mypy source gate: `Success: no issues found in 62 source files`
- Pytest: `131 passed`

## Claim Boundary

Allowed claim:

- `tune-inner` now has explicit, tested cache controls for supported model
  families.
- A bounded validation-only `ml100k cb_svdpp` probe ranked alpha and cluster
  candidates without evaluating test data.
- Cache hits and invalidations are manifest-visible in the candidate runs.

Explicit non-claims:

- no final `ml100k` quality claim
- no `ml10m` or `ml20m` tuning claim
- no test-set model-selection claim
- no scalability claim
- no production-readiness claim
- no SOTA claim
- no paper-faithfulness claim
