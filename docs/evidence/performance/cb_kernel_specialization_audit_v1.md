# CB Kernel Specialization Audit V1 Evidence

## Branch

`cb-kernel-specialization-audit-v1`

## Goal

Audit the clustering-based training kernels and define a conservative
specialization plan before any kernel implementation work.

This step is analysis and evidence only:

- no kernel specialization implemented
- no new Numba kernels added
- no model formula changed
- no update order changed
- no performance claim made

## Files Audited

- `src/recsys_lab/models/kernels.py`
- `src/recsys_lab/models/cb_svdpp.py`
- `src/recsys_lab/models/cb_asvdpp.py`
- `src/recsys_lab/benchmarks/kernel_harness.py`
- `src/recsys_lab/benchmarks/synthetic_kernel_cases.py`
- `src/recsys_lab/data/histories.py`
- `src/recsys_lab/experiments/duplication_profile.py`
- `docs/evidence/performance/history_data_layout_v1.md`
- `docs/evidence/performance/kernel_benchmark_harness_v1.md`
- `docs/performance/residual_history_duplication_audit_v1.md`
- `docs/evidence/performance/exact_residual_reuse_v1.md`
- `docs/evidence/performance/exact_residual_reuse_validation_v1.md`
- `docs/performance/cb_kernel_specialization_audit_v1.md`

## CB Kernel Anatomy Summary

Audited kernels:

- `train_cb_svdpp_epoch_numba`
- `train_cb_asvdpp_epoch_numba`

Reference kernels:

- `train_svdpp_epoch_numba`
- `train_asvdpp_epoch_numba`

The CB kernels mix individual and cluster factor families with:

- `alpha`
- `one_minus_alpha = 1.0 - alpha`
- `q_mix_old = one_minus_alpha * q_old + alpha * q_cluster_old`
- context initialized from individual and cluster user factors

Individual factor families:

- `user_factors`
- `item_factors`
- `implicit_factors`
- `explicit_factors` in CB-ASVD++

Cluster factor families:

- `user_cluster_factors`
- `item_cluster_factors`
- `implicit_cluster_factors`
- `explicit_cluster_factors` in CB-ASVD++

History traversals:

- implicit user history
- explicit user history in CB-ASVD++
- cluster-count history in both CB kernels

## Candidate Classification Summary

Endpoint alpha candidates are exact only under a strict parameter-state rule:

- `alpha == 0` can simplify cluster-weighted prediction/context terms only if
  cluster-factor regularization updates remain equivalent.
- `alpha == 1` can simplify individual-weighted prediction/context terms only
  if individual-factor regularization updates remain equivalent.

Summary:

| Candidate family | Classification | Reason |
| --- | --- | --- |
| Endpoint alpha with all generic parameter mutations preserved | `EXACT` | Parameter-identical to the generic kernel. |
| Endpoint alpha that skips inactive factor-family updates | `RESEARCH_CHANGE` | Prediction may match at fixed alpha, but stored parameters differ. |
| Local scalar hoists such as `implicit_norm * alpha` | `EXACT_BUT_ORDER_SENSITIVE` | Product grouping can alter floating-point rounding. |
| Pass fusion of context and update loops | `APPROXIMATION` | Error timing and update order change. |
| Dispatching `alpha == 0` to plain SVD++/ASVD++ | `RESEARCH_CHANGE` | Changes CB model state and semantics. |
| Generic `0 < alpha < 1` path | keep current kernel | Both individual and cluster paths are semantically active. |
| No-cluster-history fast path | no specialization recommended | Empty CSR ranges already skip loop bodies. |
| Fixed dtype/layout dispatch | no first-slice recommendation | Step 15 validators and Numba signatures already cover this boundary. |

Dispatch boundary:

- future specialization dispatch should live in `CBSVDppRecommender.fit(...)`
  and `CBASVDppRecommender.fit(...)`
- runner, registry, and model requirements should not branch on hotpath kernel
  implementation variants
- benchmark harness dispatch can remain coldpath diagnostic tooling

## Parameter-Update Visibility Findings

The main exactness trap is parameter visibility.

At `alpha == 0`, cluster factors do not contribute to predictions, but the
generic kernel still applies regularization updates to cluster factor arrays.
Skipping those updates is not parameter-identical.

At `alpha == 1`, individual factors do not contribute to predictions, but the
generic kernel still applies regularization updates to individual factor
arrays. Skipping those updates is not parameter-identical.

The audit therefore separates:

- `parameter-identical exact`
- `prediction-identical only`
- `observable-metrics-identical only`
- `not exact`

For the first implementation slice, the accepted default is
`parameter-identical exact`. Prediction-equivalent but parameter-different
specialization is not accepted unless a separate model contract proves the
changed parameters are never visible or relevant.

## Cost Model Summary

No new cost helper was added. Existing contracts provide the needed aggregate
inputs.

Sources:

- Step 16c ML1M after-run `kernel_profile.json` files
- `src/recsys_lab/experiments/duplication_profile.py`
- Kernel Benchmark Harness summary CSVs
- `performance_profile.json` for stage context

The cost model records:

- `implicit_history_visits`
- `explicit_history_visits`
- `cluster_history_visits`
- `estimated_cluster_factor_touches`
- `estimated_individual_factor_touches`
- endpoint skippable work under strict parameter-identical semantics

ML1M structural readout used in the audit:

| Model | alpha | implicit visits | explicit visits | cluster visits | cluster factor touches | individual factor touches | Parameter-identical endpoint skippable work |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cb_svdpp` | `0.1` | `4,994,765,860` | `0` | `877,047,720` | `112,262,108,160` | `639,330,030,080` | `0` |
| `cb_asvdpp` | `0.1` | `4,994,765,860` | `4,994,765,860` | `877,047,720` | `112,262,108,160` | `1,278,660,060,160` | `0` |

Interpretation:

- the ML1M validation configs use `alpha = 0.1`, not endpoint alpha
- endpoint costs are planning structure, not measured endpoint results
- under strict parameter-identical semantics, no endpoint work can simply be
  dropped unless equivalent regularization updates are preserved

## Baseline Benchmark Artifacts

Synthetic baseline command:

```bash
python scripts/run_kernel_benchmarks.py --case all --warmup-repeats 1 --timed-repeats 5 --output-dir artifacts/benchmarks/kernel/cb_specialization_baseline_v1
```

Artifacts:

- `artifacts/benchmarks/kernel/cb_specialization_baseline_v1/kernel_benchmark_summary.csv`
- `artifacts/benchmarks/kernel/cb_specialization_baseline_v1/<benchmark_id>/kernel_benchmark.json`

Artifact check:

- six kernel benchmark payloads written
- summary CSV written
- `cb_svdpp` baseline present:
  `tiny_cb_svdpp_float32_wr1_tr5_epr1`
- `cb_asvdpp` baseline present:
  `tiny_cb_asvdpp_float32_wr1_tr5_epr1`

The baseline values are synthetic sanity readouts for future specialized kernel
comparisons only.

## Tests/Gates

Commands run:

```bash
ruff check .
pytest tests/unit/test_hotpath_coldpath_boundaries.py
pytest tests/unit/test_kernel_benchmark_harness.py
pytest tests/integration/test_kernel_benchmark_harness_tiny.py
pytest tests/unit
pytest tests/integration/test_unified_pipeline_smoke_all_models.py
pytest
python scripts/run_kernel_benchmarks.py --case all --warmup-repeats 1 --timed-repeats 5 --output-dir artifacts/benchmarks/kernel/cb_specialization_baseline_v1
rg <claim-check-pattern> docs src tests
```

Results:

- `ruff check .`: passed
- `pytest tests/unit/test_hotpath_coldpath_boundaries.py`: 13 passed
- `pytest tests/unit/test_kernel_benchmark_harness.py`: 18 passed
- `pytest tests/integration/test_kernel_benchmark_harness_tiny.py`: 1 passed
- `pytest tests/unit`: 211 passed
- `pytest tests/integration/test_unified_pipeline_smoke_all_models.py`: 1 passed
- `pytest`: 285 passed, 2 skipped
- synthetic benchmark baseline: passed and wrote six payloads plus summary CSV
- claim-pattern check on the new 17a audit/evidence documents: no matches

Execution note:

- In this local shell, bare `python` resolves to Python 3.10 without the repo
  dependencies. The benchmark baseline was therefore executed with the Python
  3.12 interpreter used by pytest and `PYTHONPATH=src`.
- The full `rg <claim-check-pattern> docs src tests` command reports existing
  governance, claim-boundary, and test strings. No new 17a document contains
  the claim-check pattern.

## Claim Boundary

No statement is made that specialization is faster.

This evidence records:

- audit findings
- candidate classification
- parameter-visibility constraints
- structural cost estimates
- synthetic baseline artifacts

It does not claim runtime improvement, cross-device behavior, or dataset-general
performance.

## Recommended Next Implementation Slice

Do not implement a new kernel until this audit is accepted.

Recommended Step 17b preparation:

1. Add endpoint-alpha synthetic cases for CB-SVD++ and CB-ASVD++.
2. Add generic-vs-specialized equivalence tests that compare all mutated arrays.
3. Only then implement the first endpoint specialization.

Preferred first target:

- `train_cb_svdpp_epoch_numba` at `alpha == 0`

Constraint:

- preserve parameter-identical behavior, including cluster-factor
  regularization updates, or classify the implementation as a research change
  instead of an exact specialization.
