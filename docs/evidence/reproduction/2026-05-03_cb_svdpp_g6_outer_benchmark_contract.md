# CB-SVD++ G6 Outer Benchmark Contract

- date: `2026-05-03`
- status: `approved_for_clean_outer_benchmark_contract`
- dataset: `ml100k`
- model: `cb_svdpp`
- split_family: `benchmark_random_v1`
- selected_config:
  `configs/models/tuned/ml100k_cb_svdpp_g6_validation_selected.yaml`
- selection_evidence:
  `docs/evidence/reproduction/2026-05-03_cb_svdpp_g6_validation_grid_run.md`
- governed_by:
  `docs/roadmaps/2026-05-02_claim_unlock_and_scalability_plan.md`

## Decision

The frozen G6-selected `ml100k cb_svdpp` config is approved for one clean
outer benchmark campaign.

This approval exists because the G6 selection campaign was validation-only,
used split seeds `1,2,3`, selected by `validation_rmse_mean`, and recorded
`0` non-null `test_rmse` values across candidate metrics. The selected config
is now frozen before any outer test-set evaluation.

This note is a run contract, not a benchmark result.

## Frozen Inputs

- processed manifest:
  `data/processed/ml100k/ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config:
  `configs/models/tuned/ml100k_cb_svdpp_g6_validation_selected.yaml`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- split family: `benchmark_random_v1`
- train ratio: `0.8`
- validation ratio: `0.1`
- split seeds: `1,2,3`
- model seed: `1`
- cache policy:
  `--split-cache enable --training-index-cache --cluster-artifact-cache`

## Planned Commands

Run these commands only from a clean worktree on the committed version of this
contract.

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main train-cb-svdpp `
  data\processed\ml100k\ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json `
  --model-config configs\models\tuned\ml100k_cb_svdpp_g6_validation_selected.yaml `
  --runtime-config configs\runtime\base.yaml `
  --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml `
  --split-family benchmark_random_v1 `
  --train-ratio 0.8 `
  --validation-ratio 0.1 `
  --split-seed 1 `
  --model-seed 1 `
  --split-cache enable `
  --training-index-cache `
  --cluster-artifact-cache
```

Repeat the command for `--split-seed 2` and `--split-seed 3` without changing
any other input.

After all three run manifests exist on the same clean git commit, aggregate:

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main benchmark-random-multiseed `
  data\processed\ml100k\ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json `
  configs\models\tuned\ml100k_cb_svdpp_g6_validation_selected.yaml `
  --runtime-config configs\runtime\base.yaml `
  --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml `
  --split-seeds 1,2,3 `
  --model-seed 1
```

## Acceptance Gates

- The worktree must be clean before the first outer run starts.
- This contract must already be committed before the first outer run starts.
- The three run manifests must report `git.dirty=false`.
- The three run manifests must report the same git commit as the aggregation process.
- The three run manifests must use the frozen selected config listed above.
- The three run manifests must use the same processed manifest, runtime config,
  device config, split family, train ratio, validation ratio, model seed, dtype,
  and cache policy.
- The aggregate benchmark must be created without editing model config,
  runtime config, device config, data, split logic, or benchmark code after
  seeing any outer test metric.
- If any run fails or the worktree becomes dirty before aggregation, the
  campaign is invalid for final G6 promotion and must be restarted from a clean
  committed state.

## Claim Boundary

Allowed after a passing aggregation:

- a clean outer `ml100k cb_svdpp` benchmark readout for this frozen G6-selected
  profile, limited to the documented split family, split seeds, model seed,
  runtime profile, device profile, and commit

Still blocked after this contract alone:

- final `ml100k cb_svdpp` quality claim
- any `ml1m`, `ml10m`, or `ml20m` claim
- speed claim
- scalability claim
- production-readiness claim
- SOTA claim
- paper-faithfulness claim
- `R_star` optimization claim

## Evidence Gate Readout

Executed after adding this contract:

- Ruff: `All checks passed!`
- Mypy: `Success: no issues found in 62 source files`
- Pytest: `135 passed`

These gates prove repository consistency for the contract. They do not prove
the outer benchmark result, because the outer run has not been executed yet.
