# Kernel Benchmark Harness V1 Evidence

## Branch

`kernel-benchmark-harness-v1`

## Goal

Kernel Benchmark Harness V1 adds coldpath diagnostic tooling for isolated,
warm-run measurement of the Numba training kernels in
`src/recsys_lab/models/kernels.py`.

The goal is measurement infrastructure only:

- build deterministic synthetic tiny kernel inputs
- isolate kernel calls from pipeline and runner noise
- separate warmup from timed repeats
- keep state mutation controlled across repeats
- write JSON and CSV benchmark artifacts

## What Is Measured

The harness measures direct calls to these training kernels:

- `train_biased_mf_epoch_numba`
- `train_svdpp_epoch_numba`
- `train_asymmetric_svd_epoch_numba`
- `train_asvdpp_epoch_numba`
- `train_cb_svdpp_epoch_numba`
- `train_cb_asvdpp_epoch_numba`

Each timed repeat starts from a freshly cloned mutable model state and records
only wall-clock seconds for the kernel call region. If `epochs_per_repeat > 1`,
the same mutable state is used across those kernel calls inside the repeat.

## What Is Not Measured

The harness does not measure:

- full pipeline runtime
- model class `fit(...)` runtime
- dataset loading or split creation
- config loading
- manifest writing
- report generation
- JSON or CSV writing time
- state-copy time
- Numba compile time as part of timed repeats

No model formula, update rule, kernel implementation, hyperparameter, split,
history layout, or production runner behavior is changed by this step.

## Warmup And Compile Rule

Warmup repeats run before timed repeats.

Warmup seconds are recorded separately as `warmup_wall_seconds` and are marked
with `warmup_excluded_from_timed = true`. Timed statistics are computed only from
`repeat_wall_seconds`.

The payload records `compile_excluded = true`. The first warmup run is the place
where Numba compilation can occur for the synthetic signature. Timed repeats are
therefore intended as warm-run kernel measurements for the same synthetic case.

## State-Copy Rule

Training kernels mutate parameter arrays in place. The harness copies only the
arrays that kernels mutate:

- biases
- user and item latent factors
- explicit and implicit feedback factors
- cluster-level latent factors

Immutable input arrays such as `order`, `user_ids`, `item_ids`, `ratings`,
`indptr`, `items`, `norms`, `cluster_ids`, `cluster_counts`, `user_clusters`,
and `item_clusters` are reused.

State copying happens before the timed region. The payload records
`state_copy_excluded = true`.

After every warmup and timed repeat, the harness checks that mutated parameters
remain finite. The payload records this under `state_checks`.

## Output Artifacts

The CLI writes benchmark artifacts to:

- `artifacts/benchmarks/kernel/<benchmark_id>/kernel_benchmark.json`
- `artifacts/benchmarks/kernel/kernel_benchmark_summary.csv`

JSON payloads contain aggregate metadata, repeat timings, history summaries,
estimated structural work, and state-check results. They do not contain the raw
NumPy arrays used by the synthetic cases.

The CSV summary is intentionally flat and contains aggregate columns only.

## Synthetic Tiny Cases

The synthetic tiny cases are deterministic, small, and in-memory only:

- `n_users = 4`
- `n_items = 5`
- `n_ratings = 16`
- `latent_dim = 3`
- `dtype = float32`
- integer IDs and index arrays use `int32`

History-enabled cases include implicit, explicit, and cluster index arrays as
required by the corresponding kernels. The case builder validates that arrays
are contiguous, finite, have valid IDs, and do not contain empty per-user
history ranges where histories are required.

The synthetic cases provide inputs only. They do not duplicate model formulas or
update rules.

## Structural Work Readout

`history_structure` and `estimated_factor_touches` reuse the existing Kernel
Cost Anatomy V1 helpers in `src/recsys_lab/experiments/kernel_profile.py`.

The estimated touch counts are structural diagnostics for consistent harness
payloads. They are not CPU instruction counts and are not performance claims.

## Tests And Gates

Focused checks run for this step:

- `ruff check ...`: passed
- `python -m pytest tests/unit/test_kernel_benchmark_harness.py`: passed, 17 tests
- `python -m pytest tests/integration/test_kernel_benchmark_harness_tiny.py`: passed, 1 test
- `python -m pytest tests/unit/test_hotpath_coldpath_boundaries.py`: passed, 11 tests

The integration test runs all six synthetic tiny cases with
`warmup_repeats = 1` and `timed_repeats = 2`, writes artifacts under `tmp_path`,
and checks the output contract. It uses no speedup threshold and no absolute
runtime threshold.

## Hotpath / Coldpath Boundary

The harness is coldpath diagnostic tooling. It is allowed to import
`recsys_lab.models.kernels` and call the Numba kernels directly.

The reverse direction is forbidden:

- `src/recsys_lab/models/kernels.py` must not import `recsys_lab.benchmarks`
- model hotpath files under `src/recsys_lab/models/` must not import
  `recsys_lab.benchmarks`

The boundary unit tests include this import-direction guard.

## Claim Boundary

This step makes no statement that any kernel, model, or pipeline is faster,
slower, better, scalable, production-ready, or improved.

Any tiny benchmark output from this harness is a synthetic sanity readout for
the measurement infrastructure only. It is not evidence of dataset-level or
device-level performance behavior.

## Next Red-Thread Step

15. History Data Layout V1
