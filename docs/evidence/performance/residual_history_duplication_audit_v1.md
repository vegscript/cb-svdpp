# Residual / History Duplication Audit V1 Evidence

## Branch

`residual-history-duplication-audit-v1`

## Goal

Audit the six Numba training kernels for duplicated residual and history work,
classify optimization candidates by exactness, and create a synthetic tiny
baseline for later before/after sanity comparison.

This step does not implement a kernel optimization and does not change model
semantics.

## Files Audited

- `src/recsys_lab/models/kernels.py`
- `src/recsys_lab/experiments/kernel_profile.py`
- `src/recsys_lab/experiments/duplication_profile.py`
- `src/recsys_lab/benchmarks/kernel_harness.py`
- `src/recsys_lab/benchmarks/synthetic_kernel_cases.py`
- `docs/performance/residual_history_duplication_audit_v1.md`
- `artifacts/reports/kernel_cost_anatomy.csv`, if present locally

## Duplication Findings

- `biased_mf`: no residual or history duplication candidate.
- `svdpp`: implicit history is traversed for context construction and later for
  implicit-factor updates. Pass fusion is not the first exact slice.
- `asymmetric_svd`: explicit residual weights are recomputed in the explicit
  context pass and explicit-factor update pass.
- `asvdpp`: same explicit residual duplication as `asymmetric_svd`, plus the
  existing implicit context/update traversal split.
- `cb_svdpp`: implicit and cluster histories are traversed in separate context
  and update passes; cluster id/count lookups are repeated.
- `cb_asvdpp`: combines explicit residual duplication, explicit item-cluster
  lookup duplication, implicit traversal duplication, and cluster traversal
  duplication.

## Exactness Classification Summary

- `EXACT`: cache raw explicit residual weights per rating in
  `asymmetric_svd`, `asvdpp`, and `cb_asvdpp`; cache explicit
  `history_cluster` IDs in `cb_asvdpp`; cache raw cluster history IDs/counts in
  CB kernels.
- `EXACT_BUT_ORDER_SENSITIVE`: cache scaled products such as
  `explicit_norm * residual_weight` or alpha/norm/count products.
- `APPROXIMATION`: fuse context construction with explicit, implicit, or
  cluster update passes; cache full contexts or residuals across ratings/epochs.
- `RESEARCH_CHANGE`: reorder ratings or histories, change residual formulas,
  change current-item bias semantics, or change history layout contracts.

## Baseline Benchmark Artifacts

Command:

```bash
python scripts/run_kernel_benchmarks.py --case all --warmup-repeats 1 --timed-repeats 5 --output-dir artifacts/benchmarks/kernel/residual_history_duplication_baseline_v1
```

Artifacts:

- `artifacts/benchmarks/kernel/residual_history_duplication_baseline_v1/kernel_benchmark_summary.csv`
- `artifacts/benchmarks/kernel/residual_history_duplication_baseline_v1/<benchmark_id>/kernel_benchmark.json`

Artifact check:

- six payloads generated
- all six synthetic tiny cases covered
- `warmup_repeats=1`
- `timed_repeats=5`
- `compile_excluded=true`
- `state_copy_excluded=true`

The values are synthetic tiny baseline readouts for later controlled
before/after comparison only.

## Tests/Gates

Focused checks run:

```bash
ruff check tests/unit/test_duplication_profile.py src/recsys_lab/experiments/duplication_profile.py
pytest tests/unit/test_duplication_profile.py
rg <claim-check-pattern> docs/performance/residual_history_duplication_audit_v1.md src/recsys_lab/experiments/duplication_profile.py tests/unit/test_duplication_profile.py
```

Results:

- `ruff check ...`: passed
- `pytest tests/unit/test_duplication_profile.py`: 7 passed
- focused claim check: no matches

Full gates run for Step 16a:

- `ruff check .`: passed
- `pytest tests/unit/test_hotpath_coldpath_boundaries.py`: 13 passed
- `pytest tests/unit/test_kernel_benchmark_harness.py`: 18 passed
- `pytest tests/integration/test_kernel_benchmark_harness_tiny.py`: 1 passed
- `pytest tests/unit`: 199 passed
- `pytest tests/integration/test_unified_pipeline_smoke_all_models.py`: 1 passed
- `pytest`: 272 passed, 3 skipped
- `python scripts/run_kernel_benchmarks.py --case all --warmup-repeats 1 --timed-repeats 5 --output-dir artifacts/benchmarks/kernel/residual_history_duplication_baseline_v1`:
  passed and wrote 6 payloads plus summary CSV

Claim scan note:

- Full scan over `docs src tests` reported existing guardrail strings and older
  evidence references from previous steps.
- Focused scan of the Step 16a audit, evidence, helper, and tests reported no
  matches.

## Claim Boundary

No performance claim is made. This evidence records an audit, an exactness
classification, aggregate duplication counts, and a synthetic tiny baseline. It
does not claim that any kernel is faster or that any optimization has been
implemented.

## Recommended Next Implementation Slice

Implement only `EXACT` candidates:

- cache raw explicit residual weights in:
  - `train_asymmetric_svd_epoch_numba`
  - `train_asvdpp_epoch_numba`
  - `train_cb_asvdpp_epoch_numba`
- cache explicit `history_cluster` IDs in `train_cb_asvdpp_epoch_numba`
- keep rating order, history order, factor update order, formulas, and history
  layout unchanged
- add deterministic before/after equality tests for all mutated arrays in the
  target kernels
- rerun the Kernel Benchmark Harness as a synthetic sanity check, not as a
  portable runtime claim
