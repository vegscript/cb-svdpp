# Exact Residual Reuse Optimization V1 Evidence

## Branch

`exact-residual-reuse-v1`

## Goal

Implement the first narrow `EXACT` optimization slice from the residual /
history duplication audit:

- cache raw explicit residual weights within one rating update in the explicit
  history kernels that recomputed them
- cache explicit `history_cluster` IDs within one rating update in
  `cb_asvdpp`

This step changes only redundant per-rating recomputation in selected kernels.
It does not change model formulas, data order, update order, or history layout.

## Audit Source

Primary audit:

- `docs/performance/residual_history_duplication_audit_v1.md`

Supporting evidence:

- `docs/evidence/performance/residual_history_duplication_audit_v1.md`
- `artifacts/benchmarks/kernel/residual_history_duplication_baseline_v1/kernel_benchmark_summary.csv`

The audit classified raw explicit residual reuse and explicit
`history_cluster` reuse as `EXACT`. Scaled products and pass fusion were not
classified as exact for this implementation slice.

## Kernels Changed

Only these kernels changed:

- `train_asymmetric_svd_epoch_numba`
- `train_asvdpp_epoch_numba`
- `train_cb_asvdpp_epoch_numba`

## Kernels Not Changed

These kernels were intentionally left unchanged:

- `train_biased_mf_epoch_numba`
- `train_svdpp_epoch_numba`
- `train_cb_svdpp_epoch_numba`

## What Changed Exactly

### `train_asymmetric_svd_epoch_numba`

- Computes the maximum explicit history length once before the rating loop.
- Allocates one `explicit_residual_workspace` outside the rating loop.
- Stores the raw explicit `residual_weight` during the explicit context pass.
- Reuses the stored raw `residual_weight` during the explicit-factor update
  pass.

### `train_asvdpp_epoch_numba`

- Uses the same raw explicit residual workspace pattern as
  `asymmetric_svd`.
- Leaves `p_old`, `q_old`, context construction, implicit history traversal,
  and factor update order unchanged.

### `train_cb_asvdpp_epoch_numba`

- Computes the maximum explicit history length once before the rating loop.
- Allocates one raw explicit residual workspace and one explicit cluster
  workspace outside the rating loop.
- Stores raw `residual_weight` and raw `history_cluster` during the explicit
  context pass.
- Reuses those stored values during the explicit-factor and
  explicit-cluster-factor update pass.

## What Explicitly Did Not Change

- No scaled products are cached.
- No `explicit_norm * residual_weight` values are cached.
- No alpha-scaled values are cached.
- No pass fusion was introduced.
- No implicit history traversal changed.
- No cluster history traversal changed.
- No rating order changed.
- No explicit history order changed.
- No history layout changed.
- No model formula changed.
- No update order changed.
- No cache is shared across ratings or epochs.
- No `biased_mf`, `svdpp`, or `cb_svdpp` logic changed.
- No benchmark or experiment code is imported into model hotpath code.

## Exactness Argument

Raw residuals are computed and reused within the same rating update. No
residuals are shared across ratings or epochs.

Only the raw residual value is stored. Scaled products such as
`explicit_norm * residual_weight`, alpha-weighted terms, and norm/product
combinations are not cached.

The explicit update loop remains a separate loop. Context construction and
explicit-factor updates are not fused.

The explicit history order is unchanged. The rating order is unchanged.

For `history_item == item_id`, the existing `item_bias_old` semantics are
preserved because the raw residual is computed in the same location where the
old expression selected `item_bias_old`.

For `history_item != item_id`, the same bias value is used because, between
the explicit context pass and the explicit update pass, only the current item
bias can be changed. Non-current history item biases are not updated in that
interval.

In `cb_asvdpp`, `history_cluster` is an immutable lookup from `item_clusters`.
Caching it within the same rating update reuses the same ID without changing
the individual or cluster contribution formulas.

## Test Results

Focused checks run during this step:

```bash
python -m ruff check tests/unit/test_exact_residual_reuse.py src/recsys_lab/models/kernels.py
python -m pytest tests/unit/test_exact_residual_reuse.py
python -m ruff check scripts/compare_kernel_benchmarks.py
```

Results:

- `ruff check tests/unit/test_exact_residual_reuse.py src/recsys_lab/models/kernels.py`: passed
- `pytest tests/unit/test_exact_residual_reuse.py`: 12 passed
- `ruff check scripts/compare_kernel_benchmarks.py`: passed

The focused tests verify that target and non-target synthetic cases run through
the existing kernel benchmark harness, mutated state remains finite, and the
payload contract remains valid for target cases. They do not include runtime
thresholds or speedup assertions.

Full Step 16b gates are still to be run in the final gate phase.

## Benchmark Artifacts

Baseline artifact from Step 16a:

- `artifacts/benchmarks/kernel/residual_history_duplication_baseline_v1/kernel_benchmark_summary.csv`
- `artifacts/benchmarks/kernel/residual_history_duplication_baseline_v1/<benchmark_id>/kernel_benchmark.json`

After-change synthetic tiny artifact:

- `artifacts/benchmarks/kernel/exact_residual_reuse_v1/kernel_benchmark_summary.csv`
- `artifacts/benchmarks/kernel/exact_residual_reuse_v1/<benchmark_id>/kernel_benchmark.json`

Comparison helper output:

- `artifacts/benchmarks/kernel/exact_residual_reuse_v1/comparison_vs_baseline.csv`

Command used for the after-change synthetic tiny benchmark:

```bash
python scripts/run_kernel_benchmarks.py --case all --warmup-repeats 1 --timed-repeats 5 --output-dir artifacts/benchmarks/kernel/exact_residual_reuse_v1
```

Command used for the CSV comparison:

```bash
python scripts/compare_kernel_benchmarks.py --before artifacts/benchmarks/kernel/residual_history_duplication_baseline_v1/kernel_benchmark_summary.csv --after artifacts/benchmarks/kernel/exact_residual_reuse_v1/kernel_benchmark_summary.csv --output artifacts/benchmarks/kernel/exact_residual_reuse_v1/comparison_vs_baseline.csv
```

These benchmark artifacts are local machine artifacts and are ignored by git.

## Synthetic Benchmark Readout

The local comparison CSV contains six model rows and `metadata_match=True` for
all six rows.

Synthetic tiny after/before mean wall-time ratios from the local comparison
CSV:

- `asymmetric_svd`: `0.4505748960206684`
- `asvdpp`: `2.3230091168805456`
- `cb_asvdpp`: `0.44264186101675884`
- `biased_mf`: `1.8448926747353338`
- `svdpp`: `2.304344841721764`
- `cb_svdpp`: `1.3719701471138455`

These are synthetic tiny sanity readouts only. They are not portable
performance results and are not interpreted as evidence that any kernel is
generally faster or slower.

## Claim Boundary

No portable or general runtime claim is made. This evidence records a narrow
exact implementation slice, focused tests, and local synthetic tiny benchmark
artifacts. It does not claim a speedup, deployment status, or general runtime
behavior on real datasets or other machines.

## Known Limitations

- Numerical equivalence was not tested against an in-branch copy of the old
  kernels to avoid adding duplicate hotpath formulas to tests.
- The synthetic benchmark artifacts use tiny deterministic cases and are useful
  as harness sanity checks, not as model-level performance evidence.
- Full Step 16b gates remain pending until the final gate phase.
- Large dataset benchmarks were intentionally not run in this step.

## Recommended Next Step

Run the full Step 16b gates, including the boundary tests, kernel benchmark
harness tests, script smoke, claim check, and focused exact residual reuse tests.
