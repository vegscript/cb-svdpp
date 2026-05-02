# CB-SVD++ G6 Validation Grid Run Evidence

- date: `2026-05-03`
- status: `pass_for_validation_only_selection`
- scope: `ml100k cb_svdpp g6 validation grid`
- contract:
  `docs/evidence/reproduction/2026-05-03_cb_svdpp_g6_validation_grid_contract.md`

## Purpose

This note records the completed G6 validation-only selection run for
`ml100k cb_svdpp`.

This is selection evidence only. It is not a final benchmark, not a test-set
result, and not a quality, speed, scalability, production-readiness, SOTA, or
paper-faithfulness claim.

## Command

The command was executed from a clean `main` worktree:

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main tune-inner configs\experiments\tuning\ml100k_cb_svdpp_g6_validation_grid.yaml data\processed\ml100k\ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json --runtime-config configs\runtime\base.yaml --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml --split-cache enable --training-index-cache --cluster-artifact-cache
```

## Artifact Readout

- benchmark id:
  `2026-05-02T223510Z_ml100k_inner_tuning_cb_svdpp_g6_validation_grid_local_i5_2500k_24gb`
- benchmark manifest:
  `artifacts/benchmarks/2026-05-02T223510Z_ml100k_inner_tuning_cb_svdpp_g6_validation_grid_local_i5_2500k_24gb/benchmark_manifest.json`
- summary:
  `artifacts/benchmarks/2026-05-02T223510Z_ml100k_inner_tuning_cb_svdpp_g6_validation_grid_local_i5_2500k_24gb/summary.json`
- git commit: `9a483363123a417d21e6dff42c8f6bf61f53e312`
- git dirty: `false`
- benchmark status: `completed`
- manifest validation: `valid`
- candidate count: `12`
- split seeds: `1,2,3`
- candidate run count: `36`
- non-null `test_rmse` count across candidate metrics: `0`
- split cache: `enable`
- training-index cache: `true`
- cluster-artifact cache: `true`

## Selection Result

Selected candidate:

- candidate id: `rank032_uc100_ic100_a0000_lr0100_reg0020_e002`
- validation RMSE mean: `0.9566122815305916`
- validation RMSE std: `0.002372317660216579`
- validation RMSE count: `3`
- train RMSE mean: `0.9253765314125518`
- fit-time mean: `5.577742766647134` seconds

Frozen selected config:

- `configs/models/tuned/ml100k_cb_svdpp_g6_validation_selected.yaml`

Top selection table:

| Rank | Candidate | Validation RMSE Mean | Validation RMSE Std | Fit Time Mean (s) |
| --- | --- | ---: | ---: | ---: |
| 1 | `rank032_uc100_ic100_a0000_lr0100_reg0020_e002` | `0.956612` | `0.002372` | `5.58` |
| 2 | `rank032_uc064_ic064_a0000_lr0100_reg0020_e002` | `0.956932` | `0.003279` | `6.15` |
| 3 | `rank032_uc080_ic080_a0000_lr0100_reg0020_e002` | `0.957108` | `0.003093` | `5.46` |
| 4 | `rank032_uc032_ic032_a0000_lr0100_reg0020_e002` | `0.957145` | `0.002982` | `5.25` |

Interpretation boundary:

- The selected candidate is the validation-RMSE winner in this G6 grid.
- The winning margin over rank 2 is small, so no broad model-quality claim is
  allowed from this selection evidence alone.
- No test-set claim is allowed until the frozen selected config is evaluated in
  a separate clean outer benchmark run.

## Verification

Artifact checks:

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main validate-manifest artifacts\benchmarks\2026-05-02T223510Z_ml100k_inner_tuning_cb_svdpp_g6_validation_grid_local_i5_2500k_24gb\benchmark_manifest.json
```

Readout:

- status: `valid`
- git dirty: `false`
- candidate runs: `36`
- non-null `test_rmse` count: `0`

Focused tests:

```powershell
.venv\Scripts\python.exe -m ruff check tests\integration\test_cb_svdpp_g6_validation_grid_contract.py tests\integration\test_release_evidence_integrity.py tests\integration\test_publish_readiness_plan.py
.venv\Scripts\python.exe -m pytest tests\integration\test_cb_svdpp_g6_validation_grid_contract.py tests\integration\test_release_evidence_integrity.py tests\integration\test_publish_readiness_plan.py
```

Readout:

- Ruff: `All checks passed!`
- Pytest: `9 passed`

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

- The G6 `ml100k cb_svdpp` validation-only grid completed cleanly and selected
  `rank032_uc100_ic100_a0000_lr0100_reg0020_e002` by validation RMSE mean.

Explicit non-claims:

- no final `ml100k cb_svdpp` quality claim
- no test-set result or test-set comparison
- no `ml10m` or `ml20m` tuning claim
- no speed, scalability, production-readiness, SOTA, or paper-faithfulness claim
