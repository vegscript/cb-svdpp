# CB-ASVD++ Hotpath Remediation Contract G8

- date: `2026-05-03`
- status: `approved_for_exact_remediation_contract`
- dataset: `ml100k`
- model: `cb_asvdpp`
- governed_by:
  `docs/evidence/reproduction/current/2026-05-03_cb_asvdpp_hotpath_decision_g7.md`
- target code:
  `src/recsys_lab/models/kernels.py::train_cb_asvdpp_epoch_numba`
- target tests:
  `tests/unit/test_cb_asvdpp.py`

## Purpose

This contract authorizes one exact `cb_asvdpp` hot-path remediation attempt.
It does not authorize a speed, scalability, quality, production-readiness,
SOTA, or paper-faithfulness claim.

The previous profiling decision showed that `main_training` consumed about
`92.44%` of profiled wall-clock in a bounded `ml100k cb_asvdpp` run. That
prioritizes the training kernel, but it does not prove that any implementation
change is faster or safe.

## Approved Implementation Target

The only approved first remediation target is fixed-size work-buffer reuse
inside `train_cb_asvdpp_epoch_numba`.

Allowed changes:

- allocate per-epoch work arrays once before the rating loop where Numba
  semantics permit it
- reuse buffers for:
  - `p_old`
  - `p_cluster_old`
  - `q_old`
  - `q_cluster_old`
  - `q_mix_old`
  - `context`
- preserve old-value SGD semantics for user factors, item factors, cluster
  factors, explicit factors, implicit factors, and biases
- keep the model name `cb_asvdpp` only if exactness gates pass

Disallowed changes:

- no reordered SGD updates
- no user-blocked, item-blocked, delayed-refresh, cached-context, or approximate
  algorithm under the existing `cb_asvdpp` label
- no change to `alpha`, cluster counts, regularization, dtype, split logic,
  cache policy, inference code, or config defaults
- no MiniBatchKMeans substitution
- no `R_star` objective integration

## Exactness Gates

Before any performance claim or benchmark promotion:

- add a deterministic unit test that compares the remediated Numba epoch
  against the current Python fallback semantics for a toy `cb_asvdpp` problem
- the test must cover non-empty explicit feedback, implicit history, cluster
  history, non-zero `alpha`, multiple clusters, and repeated items
- all trainable arrays must match within:
  - absolute tolerance: `1e-6`
  - relative tolerance: `1e-6`
- the test must compare:
  - user bias
  - item bias
  - user factors
  - item factors
  - explicit item factors
  - implicit item factors
  - user-cluster factors
  - item-cluster factors
  - explicit-cluster factors
  - implicit-cluster factors
- if exactness fails, the remediation must be rejected or renamed as a new
  approximate variant with a separate math note and claim boundary

## Benchmark Gates

The existing G7 profiling run is prioritization evidence only. It is not a
before baseline for a remediation claim because it was produced before this
contract existed.

Required sequence:

1. Commit this contract.
2. On the clean contract commit, run a fresh pre-change baseline with the exact
   benchmark command below.
3. Implement the remediation.
4. Run exactness tests and full repository gates.
5. On the clean remediation commit, rerun the same benchmark command.
6. Compare only the pre-change and post-change artifacts that match this
   contract.

Benchmark command:

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main train-cb-asvdpp `
  data\processed\ml100k\ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json `
  --model-config configs\models\tuned\ml100k_cb_asvdpp_stage1.yaml `
  --runtime-config configs\runtime\base.yaml `
  --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml `
  --split-family benchmark_random_v1 `
  --train-ratio 0.8 `
  --validation-ratio 0.1 `
  --split-seed 1 `
  --model-seed 1 `
  --cluster-artifact-cache
```

Benchmark acceptance rule:

- same dataset, split family, split seed, model seed, model config, runtime
  config, device config, dtype, cache policy, and clean git state
- `main_training_wall_clock_seconds` must decrease by at least `1.0%`
- `train_rmse`, `validation_rmse`, and `test_rmse` absolute drift must be at most `1e-6`
- stage count and manifest schema validation must still pass
- no speed claim is allowed if runtime does not improve or if metric drift
  exceeds the declared bound

## Claim Boundary

Allowed after this contract:

- implement one exact `cb_asvdpp` work-buffer remediation attempt under the
  gates above

Still blocked after this contract alone:

- speed claim
- scalability claim
- production-readiness claim
- SOTA claim
- paper-faithfulness claim
- large-dataset claim
- quality claim
- claim that `cb_asvdpp` is remediated

## Evidence Gate Readout

Executed after adding this contract:

- Ruff: `All checks passed!`
- Mypy: `Success: no issues found in 62 source files`
- Pytest: `138 passed`
