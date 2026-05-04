# CB-SVD++ Hot-Path Remediation G3

- date: `2026-05-02`
- status: `pass_for_cb_svdpp_workbuffer_candidate`
- scope: `G3 cb_svdpp training hot-path remediation`
- gate: `docs/roadmaps/2026-05-02_claim_unlock_and_scalability_plan.md`

## Purpose

This note records an exact `cb_svdpp` Numba-kernel hot-path change and its
clean before/after benchmark. It does not unlock a broad scalability,
production-readiness, SOTA, large-dataset, or paper-faithfulness claim.

## Implemented Change

The accepted kernel change reuses fixed-size old-value and work buffers inside
`train_cb_svdpp_epoch_numba` instead of constructing the per-rating temporary
vectors inside the rating loop.

Implemented in:

- `src/recsys_lab/models/kernels.py`
- `tests/unit/test_cb_svdpp.py`

Code commits:

- `e5460db` tried a scalar re-read variant and was rejected by the `ml100k`
  benchmark below.
- `f9dcf40` keeps the old-value vector semantics and reuses the work buffers.

## Equivalence Guard

Focused test command before the clean `ml100k` benchmark:

```powershell
.venv\Scripts\python.exe -m pytest tests\unit\test_cb_svdpp.py
```

Readout:

- Pytest: `4 passed`
- The added test forces the Python reference path and compares model
  parameters plus unclipped predictions against the Numba path with
  `rtol=1e-12` and `atol=1e-12`.

Full verification commands after the evidence and roadmap updates:

```powershell
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy src
.venv\Scripts\python.exe -m pytest
```

Readout:

- Ruff: `All checks passed!`
- Mypy source gate: `Success: no issues found in 62 source files`
- Pytest: `128 passed`

Benchmark drift guard:

- expected RMSE drift for the accepted exact candidate: `0.0` at printed
  precision
- observed train RMSE drift: `0.0`
- observed validation RMSE drift: `0.0`
- observed test RMSE drift: `0.0`

## Benchmark Command

The same command structure, split, config, seeds, dtype, and cache policy were
used for the baseline and the accepted candidate. Each run used a fresh cache
root, so cache readouts are comparable `miss` states.

Baseline environment:

```powershell
git switch --detach bc7ef19a294a7dbd35075793fe359fdd6dc3c266
$env:RECSYS_CACHE_ROOT = "artifacts/local/g3_ml100k_baseline_bc7ef19"
```

Accepted candidate environment:

```powershell
git switch main
$env:RECSYS_CACHE_ROOT = "artifacts/local/g3_ml100k_workbuffer_f9dcf40"
```

Command:

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main train-cb-svdpp data\processed\ml100k\ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json --model-config configs\models\tuned\ml100k_cb_svdpp_stage1.yaml --runtime-config configs\runtime\base.yaml --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml --split-family benchmark_random_v1 --train-ratio 0.8 --validation-ratio 0.1 --split-seed 1 --model-seed 1 --split-cache enable --training-index-cache --cluster-artifact-cache
```

Manifest validation command:

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main validate-manifest <run_manifest.json>
```

Readout:

- baseline manifest: `valid`
- accepted candidate manifest: `valid`

## Run Context

Common context:

- dataset: `ml100k`
- model: `cb_svdpp`
- config: `configs/models/selected/ml100k/ml100k_cb_svdpp_stage1.yaml`
- device profile: `local_i5_2500k_24gb`
- split family: `benchmark_random_v1`
- train ratio: `0.8`
- validation ratio: `0.1`
- split seed: `1`
- model seed: `1`
- epochs: `20`
- latent dimension: `64`
- dtype: `float32`
- alpha: `0.10`
- user clusters: `80`
- item clusters: `80`
- train rows: `80030`
- validation rows: `10000`
- test rows: `9970`
- split cache status: `miss`
- training user-history cache status: `miss`
- cluster artifact cache status: `miss`
- user-cluster-history cache status: `miss`

## Benchmark Readout

| Candidate | Commit | Git Dirty | Run | Main Training Seconds | Total Profiled Seconds | Train Ratings/Sec | Validation RMSE | Test RMSE |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | `bc7ef19` | `false` | `2026-05-02T104718Z_ml100k_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `52.348266` | `67.052602` | `30575.977507` | `0.915353916` | `0.910395702` |
| scalar re-read candidate | `e5460db` | `false` | `2026-05-02T104951Z_ml100k_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `53.277411` | `68.070856` | `30042.740087` | `0.915353916` | `0.910395702` |
| accepted workbuffer candidate | `f9dcf40` | `false` | `2026-05-02T105418Z_ml100k_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `51.257735` | `66.368540` | `31226.493862` | `0.915353916` | `0.910395702` |

Relative to the baseline:

- scalar re-read candidate: `+0.929145` seconds in `main_training`, or
  `+1.774929%`; rejected as a performance candidate
- accepted workbuffer candidate: `-1.090531` seconds in `main_training`, or
  `-2.083223%`; accepted only for this measured `ml100k` Stage1 context

## Claim Boundary

Allowed claim:

- In this clean `ml100k cb_svdpp` Stage1 run, the accepted workbuffer kernel
  reduced measured `main_training` wall-clock time from `52.348266` seconds to
  `51.257735` seconds with unchanged printed RMSE metrics.
- The rejected scalar re-read candidate is documented as negative performance
  evidence.

Explicit non-claims:

- no broad speed claim outside this benchmark context
- no `ml10m` or `ml20m` speed claim
- no `cb_asvdpp` hot-path claim
- no scalability claim
- no production-readiness claim
- no SOTA claim
- no paper-faithfulness claim
- no quality-improvement claim
