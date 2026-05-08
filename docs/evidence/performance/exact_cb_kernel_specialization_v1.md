# Exact CB Kernel Specialization V1 Evidence

## Branch

`exact-cb-kernel-specialization-v1`

## Goal

Implement the first minimal CB specialization proof:

- model family: `cb_svdpp`
- endpoint: `alpha == 0.0`
- specialized kernel: `train_cb_svdpp_alpha0_epoch_numba`

The goal is parameter-identical behavior versus the generic
`train_cb_svdpp_epoch_numba(..., alpha=0.0, ...)` reference on deterministic
synthetic inputs. This step does not add production dispatch.

## Audit Source

This implementation follows:

- `docs/performance/cb_kernel_specialization_audit_v1.md`
- `docs/evidence/performance/cb_kernel_specialization_audit_v1.md`

The audit classified endpoint-alpha specialization as acceptable only when all
mutated parameter arrays remain equivalent to the generic kernel, including
regularization effects for zero-gradient factor families.

## Kernel Added

- `src/recsys_lab/models/kernels.py`
  - `train_cb_svdpp_alpha0_epoch_numba`

The new kernel omits the runtime `alpha` argument and is valid only for
`alpha == 0.0`.

## Kernels Not Changed

No intended logic change was made to:

- `train_cb_svdpp_epoch_numba`
- `train_cb_asvdpp_epoch_numba`
- `train_svdpp_epoch_numba`
- `train_asvdpp_epoch_numba`
- `train_asymmetric_svd_epoch_numba`
- `train_biased_mf_epoch_numba`

No model wrapper dispatch was added in:

- `src/recsys_lab/models/cb_svdpp.py`
- `src/recsys_lab/models/cb_asvdpp.py`
- `src/recsys_lab/models/registry.py`
- `src/recsys_lab/experiments/unified_runner.py`

## What Changed

- Added a deterministic endpoint-alpha synthetic case:
  - `cb_svdpp_alpha0`
  - `alpha == 0.0`
  - int32 ids, indices, indptr, counts, and cluster assignments
  - float32 ratings, norms, and factor arrays
  - contiguous arrays
  - nonzero cluster factor arrays
- Added benchmark-harness dispatch for diagnostic measurement:
  - `cb_svdpp` runs the generic kernel
  - `cb_svdpp_alpha0` runs the specialized alpha0 kernel
- Added equivalence tests comparing generic alpha0 state to specialized alpha0
  state.

## What Explicitly Did Not Change

- No dispatch to `train_svdpp_epoch_numba`.
- No productively integrated model-wrapper dispatch.
- No runner, registry, or requirements dispatch.
- No CB-ASVD++ specialization.
- No `alpha == 1` specialization.
- No non-endpoint alpha specialization.
- No pass fusion.
- No rating or history reordering.
- No scalar-product hoists accepted as evidence.
- No model formula, objective, or update-order change.
- No skipping of cluster-factor regularization.

## Exactness Argument

The specialized kernel is valid only for `alpha == 0.0`.

The reference standard is the generic
`train_cb_svdpp_epoch_numba(..., alpha=0.0, ...)` kernel. The implementation is
kept close to the generic operation order and preserves all mutated parameter
families:

- `user_bias`
- `item_bias`
- `user_factors`
- `item_factors`
- `implicit_factors`
- `user_cluster_factors`
- `item_cluster_factors`
- `implicit_cluster_factors`

Cluster-factor regularization is not skipped. Even though cluster prediction
and gradient contributions are zero at `alpha == 0.0`, the current generic
kernel still mutates cluster factor arrays through regularization. The
specialized kernel preserves that behavior.

No prediction-only equivalence is accepted. Metrics-only equivalence is also
insufficient. The test contract compares all mutated arrays from a generic
alpha0 run and a specialized alpha0 run. The current comparison uses:

```text
rtol = 0.0
atol = 0.0
```

No production model dispatch was introduced. The specialized kernel is reachable
only from tests and the coldpath benchmark harness.

## Equivalence Tests

Test file:

- `tests/unit/test_cb_kernel_specialization.py`

Tests:

- `test_cb_svdpp_alpha0_specialized_matches_generic_all_mutated_arrays`
- `test_cb_svdpp_alpha0_specialized_preserves_cluster_regularization`
- `test_cb_svdpp_alpha0_specialized_does_not_change_shapes_or_dtypes`
- `test_cb_svdpp_alpha0_specialized_case_has_nonzero_cluster_factors`
- `test_cb_svdpp_alpha0_specialized_benchmark_payload_contract`

Final gate commands:

```bash
ruff check .
pytest tests/unit/test_cb_kernel_specialization.py
pytest tests/unit/test_kernel_benchmark_harness.py
pytest tests/integration/test_kernel_benchmark_harness_tiny.py
pytest tests/unit/test_hotpath_coldpath_boundaries.py
pytest tests/unit
pytest tests/integration/test_unified_pipeline_smoke_all_models.py
pytest
```

Results:

- `ruff check .`: passed
- `pytest tests/unit/test_cb_kernel_specialization.py`: 5 passed
- `pytest tests/unit/test_kernel_benchmark_harness.py`: 19 passed
- `pytest tests/integration/test_kernel_benchmark_harness_tiny.py`: 1 passed
- `pytest tests/unit/test_hotpath_coldpath_boundaries.py`: 13 passed
- `pytest tests/unit`: 217 passed
- `pytest tests/integration/test_unified_pipeline_smoke_all_models.py`: 1 passed
- `pytest`: 291 passed, 2 skipped
- requested claim-pattern check over `docs src tests`: completed; matches were
  pre-existing governance, claim-boundary, and test strings, with no matches in
  this 17b evidence file

## Benchmark Artifacts

Diagnostic smoke artifact generated during harness integration:

- `artifacts/benchmarks/kernel/cb_svdpp_alpha0_smoke_v1/kernel_benchmark_summary.csv`
- `artifacts/benchmarks/kernel/cb_svdpp_alpha0_smoke_v1/<benchmark_id>/kernel_benchmark.json`

Final Step 17b benchmark artifacts:

- `artifacts/benchmarks/kernel/exact_cb_kernel_specialization_v1/kernel_benchmark_summary.csv`
- `artifacts/benchmarks/kernel/exact_cb_kernel_specialization_v1/tiny_cb_svdpp_float32_wr1_tr5_epr1/kernel_benchmark.json`
- `artifacts/benchmarks/kernel/exact_cb_kernel_specialization_v1/tiny_cb_svdpp_alpha0_float32_wr1_tr5_epr1/kernel_benchmark.json`

Command:

```bash
python scripts/run_kernel_benchmarks.py --case cb_svdpp_alpha0 --warmup-repeats 1 --timed-repeats 2 --output-dir artifacts/benchmarks/kernel/cb_svdpp_alpha0_smoke_v1
python scripts/run_kernel_benchmarks.py --case all --warmup-repeats 1 --timed-repeats 5 --output-dir artifacts/benchmarks/kernel/exact_cb_kernel_specialization_v1
python scripts/run_kernel_benchmarks.py --case cb_svdpp_alpha0 --warmup-repeats 1 --timed-repeats 5 --output-dir artifacts/benchmarks/kernel/exact_cb_kernel_specialization_v1
```

Local execution note:

- In this local shell, bare `python` resolves to Python 3.10 without the repo
  dependencies. The smoke was run with the Python 3.12 interpreter used by
  pytest and `PYTHONPATH=src`.

The generated values are synthetic sanity readouts only.

## Claim Boundary

No statement is made that the specialized kernel is faster.

This evidence only establishes:

- a specialized alpha0 kernel exists
- generic alpha0 and specialized alpha0 match on deterministic synthetic
  mutated-array state
- the coldpath benchmark harness can execute the specialized case

No portable runtime, dataset-general, or device-general performance claim is
made.

## Known Limitations

- No production model dispatch exists yet.
- No real ML1M benchmark has been run for this specialization.
- Only `cb_svdpp` at `alpha == 0.0` is covered.
- No CB-ASVD++ or `alpha == 1` specialization is included.
- Equivalence is currently established on deterministic synthetic tiny inputs.

## Next Step

Recommended next step:

`17c. Validate Exact CB Kernel Specialization V1`

That step should decide whether to add production model-wrapper dispatch after
running the required gates and device-scoped validation. It should not broaden
the specialization scope without a separate audit and equivalence test contract.
