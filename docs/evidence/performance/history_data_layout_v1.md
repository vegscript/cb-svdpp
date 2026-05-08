# History Data Layout V1 Evidence

## Branch

`history-data-layout-v1`

## Goal

History Data Layout V1 makes the history structures used by SVD++,
ASVD++, CB-SVD++, and CB-ASVD++ explicit hotpath data contracts.

The goal is data-layout hardening only:

- CSR-like in-memory layout
- contiguous arrays
- `int32` IDs, indices, counts, and `indptr` where safely representable
- `float32` or `float64` ratings and norms matching the configured dtype
- centralized validation of invariants
- cache-compatible layout migration

This step does not optimize kernels and does not change model formulas.

## Affected Index Types

The contract covers:

- `UserHistoryIndex`
- `UserExplicitFeedbackIndex`
- `UserClusterCountIndex`

The affected builders are:

- `build_user_history_index`
- `build_user_explicit_feedback_index`
- `build_user_cluster_count_index`

The affected validators are:

- `validate_user_history_index`
- `validate_user_explicit_feedback_index`
- `validate_user_cluster_count_index`

## Dtype And Contiguity Rules

For `UserHistoryIndex`:

- `indptr`: 1D, C-contiguous, `int32`
- `item_indices`: 1D, C-contiguous, `int32`
- `counts`: 1D, C-contiguous, `int32`
- `norms`: 1D, C-contiguous, `float32` or `float64`

For `UserExplicitFeedbackIndex`:

- `indptr`: 1D, C-contiguous, `int32`
- `item_indices`: 1D, C-contiguous, `int32`
- `ratings`: 1D, C-contiguous, `float32` or `float64`
- `counts`: 1D, C-contiguous, `int32`
- `norms`: 1D, C-contiguous, `float32` or `float64`

For `UserClusterCountIndex`:

- `indptr`: 1D, C-contiguous, `int32`
- `cluster_ids`: 1D, C-contiguous, `int32`
- `counts`: 1D, C-contiguous, `int32`

Common invariants:

- `indptr[0] == 0`
- `indptr` is monotonic nondecreasing
- `indptr[-1]` matches the payload length
- history `counts == np.diff(indptr)` where counts are per user
- stored cluster counts are positive
- IDs are non-negative
- optional `n_items` and `n_clusters` bounds are enforced
- ratings and norms are finite
- empty history users have norm `0`
- non-empty history users have norm `1 / sqrt(count)`

## Overflow Strategy

The layout uses `int32` for hotpath IDs, indices, counts, and `indptr`.

To avoid silent overflow:

- counts are validated as non-negative integer arrays
- cumulative sums are computed with `np.cumsum(..., dtype=np.int64)`
- the final cumulative value is checked against `np.iinfo(np.int32).max`
- only after the bounds check is the `indptr` cast to `int32`

If values do not fit the V1 contract, the helpers raise `OverflowError` or
`ValueError`. There is no unsafe silent cast.

## Cache Migration And Invalidation

Training-index caches and user-cluster-history caches now include explicit
layout metadata:

- `layout_version = "history_data_layout_v1"`
- `index_dtype = "int32"`
- `value_dtype = "float32"` or `"float64"` for training-index caches
- `count_dtype = "int32"` for user-cluster-history caches

The layout version is also part of the cache path for V1 history caches. This
keeps newly generated V1 cache artifacts separate from older cache artifacts.

Cache-hit behavior:

- a cache hit requires matching layout metadata
- older manifests without the V1 layout metadata are treated as cache misses
- invalid cached arrays, including old `int64 indptr` arrays, are rejected by the
  validators and rebuilt through the normal cache miss path
- historical artifacts are not rewritten in place by this documentation step

## Tests And Gates

Focused tests added or hardened:

- `tests/unit/test_history_data_layout.py`
- `tests/unit/test_histories.py`
- `tests/unit/test_explicit_feedback_index.py`
- `tests/unit/test_clustering.py`
- `tests/unit/test_training_index_cache.py`
- `tests/unit/test_cluster_cache.py`
- `tests/unit/test_kernel_profile.py`
- `tests/unit/test_kernel_benchmark_harness.py`
- `tests/integration/test_history_data_layout_pipeline.py`
- `tests/integration/test_kernel_benchmark_harness_tiny.py`
- `tests/unit/test_hotpath_coldpath_boundaries.py`

Focused gates run during this step:

- `ruff check ...`: passed for changed Python files checked so far
- `pytest tests/unit/test_history_data_layout.py tests/unit/test_training_index_cache.py tests/unit/test_cluster_cache.py`: passed
- `pytest tests/integration/test_history_data_layout_pipeline.py`: passed
- `pytest tests/unit/test_hotpath_coldpath_boundaries.py`: passed
- `pytest tests/unit/test_kernel_benchmark_harness.py`: passed
- `pytest tests/integration/test_kernel_benchmark_harness_tiny.py`: passed
- `python scripts/run_kernel_benchmarks.py --case all --warmup-repeats 1 --timed-repeats 2 --output-dir artifacts/benchmarks/kernel`: passed as a harness smoke

The script smoke only verifies that the benchmark harness still runs and writes
its expected tiny synthetic artifacts. It is not used for a performance claim.

## What Did Not Change

This step did not change:

- model formulas
- objective functions
- regularization terms
- update rules
- Numba kernel implementation logic
- ratings order semantics
- split semantics
- clustering method semantics
- benchmark interpretation rules
- production run claims

## Hotpath / Coldpath Boundary

`src/recsys_lab/data/histories.py` is classified as hotpath preparation. It may
use NumPy and in-memory `RatingsData`, but it must not own file IO, manifests,
paths, reporting, CLI, evidence, or experiment orchestration.

`src/recsys_lab/data/training_index_cache.py` and
`src/recsys_lab/clustering/cache.py` remain boundary/coldpath cache modules.
They may use `Path`, JSON, hashing, and atomic IO for cache metadata and
artifacts before model fit.

The hotpath/coldpath boundary tests were extended so `histories.py` is checked
as stricter hotpath preparation, while cache modules are not incorrectly checked
as pure hotpath files.

## Claim Boundary

This step makes no claim that any model, kernel, pipeline, or benchmark has a
runtime, quality, scalability, or deployment-readiness improvement.

The result is a stricter data layout and validation contract for history
structures. Any future performance claim requires a separate benchmark and
evidence record.

## Next Red-Thread Step

16. Residual / History Duplication Optimization
