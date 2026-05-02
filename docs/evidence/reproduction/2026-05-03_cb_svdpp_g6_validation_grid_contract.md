# CB-SVD++ G6 Validation Grid Contract

- date: `2026-05-03`
- status: `contract_ready_g5_to_g6_validation_grid`
- scope: `ml100k cb_svdpp validation-only promotion`
- governed_by: `docs/roadmaps/2026-05-02_claim_unlock_and_scalability_plan.md`

## Purpose

This note records the decision to promote the bounded G5 `ml100k cb_svdpp`
winner into a larger validation-only grid before any test-set rerun or final
quality claim.

This is a run contract, not a benchmark result.

## Decision

Promote the G5 winner:

- source config:
  `configs/experiments/tuning/ml100k_cb_svdpp_g5_bounded_alpha_cluster.yaml`
- source evidence:
  `docs/evidence/reproduction/2026-05-02_tune_inner_cache_controls_g5.md`
- selected candidate:
  `rank032_uc064_ic064_a000_lr0100_reg0020_e002`

The promotion target is:

- config:
  `configs/experiments/tuning/ml100k_cb_svdpp_g6_validation_grid.yaml`
- objective: `validation_rmse_mean`
- split family: `benchmark_random_v1`
- split seeds: `1,2,3`
- model seed: `1`
- candidates: `12`
- planned validation-only runs: `36`

## Method Boundary

The G6 validation grid keeps the method boundary unchanged:

- no test-set evaluation during selection
- no `R_star` objective, coefficient, or quality claim
- no MiniBatchKMeans substitution
- no final benchmark claim from the selection run
- no `faster`, `scalable`, `production-ready`, SOTA, or paper-faithfulness claim

## Execution Command

The contract requires a clean worktree and explicit cache controls:

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main tune-inner configs\experiments\tuning\ml100k_cb_svdpp_g6_validation_grid.yaml data\processed\ml100k\ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json --runtime-config configs\runtime\base.yaml --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml --split-cache enable --training-index-cache --cluster-artifact-cache
```

## Acceptance Gates For A Future Run

The future run is selection evidence only if all of these are true:

- `git.dirty=false` in the benchmark manifest
- `test_metrics_available=false` or equivalent absence of test metrics for all
  candidate selection runs
- `test_rmse=None` in candidate run metrics where the run schema exposes it
- cache policy is present in the benchmark summary
- all candidates use validation RMSE as the selection objective
- the selected candidate is frozen before any outer test evaluation

## Verification

Focused contract tests:

```powershell
.venv\Scripts\python.exe -m pytest tests\integration\test_cb_svdpp_g6_validation_grid_contract.py
```

Readout:

- Pytest: `1 passed`

Full local gates:

```powershell
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy src
.venv\Scripts\python.exe -m pytest
```

Readout:

- Ruff: `All checks passed!`
- Mypy: `Success: no issues found in 62 source files`
- Pytest: `134 passed`

## Claim Boundary

Allowed claim:

- The repo now contains a tested validation-only G6 promotion contract for
  `ml100k cb_svdpp`.

Explicit non-claims:

- no G6 tuning result yet
- no final `ml100k cb_svdpp` quality claim from this contract
- no test-set result from this contract
- no `ml10m` or `ml20m` tuning claim
- no speed, scalability, production-readiness, SOTA, or paper-faithfulness claim
