# Residual / History Duplication Audit and Optimization Plan V1

## Executive Summary

This audit reviews the six Numba training kernels in
`src/recsys_lab/models/kernels.py` for duplicated residual and history work.

The main findings are:

- `biased_mf` has no history or residual-history duplication.
- `svdpp` has two implicit-history traversals per rating: one to build context
  and one to update implicit factors. This is structurally duplicated traversal,
  but merging it would change update timing and is not the first exact slice.
- `asymmetric_svd` and `asvdpp` compute explicit residual weights twice for the
  same user/rating row: once for context construction and once for explicit
  factor updates.
- `cb_svdpp` has separate implicit-history and cluster-history context/update
  traversals. It also repeats cluster id/count loads between those passes.
- `cb_asvdpp` combines all duplication classes: explicit residual weights,
  explicit item-cluster lookups, implicit history traversal, and cluster-history
  traversal.

Recommended first implementation slice:

1. Add benchmark/equality baselines for the current kernels using the existing
   warm-run harness.
2. Implement an exact local-cache slice only for explicit residual weights and
   explicit item-cluster ids inside `cb_asvdpp`, plus the equivalent residual
   cache in `asymmetric_svd` and `asvdpp`.
3. Keep rating order, history order, and all factor update loops unchanged.

This document is a plan. It does not implement a kernel optimization and makes
no runtime or quality claim.

## Method

Inputs inspected:

- `src/recsys_lab/models/kernels.py`
- `src/recsys_lab/benchmarks/kernel_harness.py`
- `src/recsys_lab/benchmarks/synthetic_kernel_cases.py`
- `src/recsys_lab/experiments/kernel_profile.py`
- `src/recsys_lab/experiments/duplication_profile.py`
- `docs/kernel_optimization_plan.md`
- `docs/evidence/performance/kernel_benchmark_harness_v1.md`
- `docs/evidence/performance/history_data_layout_v1.md`
- `artifacts/reports/kernel_cost_anatomy.csv`, if present locally

Method:

- Read each kernel body and identified per-rating loops.
- Classified repeated work by whether it recomputes the same scalar, repeats the
  same history range traversal, repeats the same ID lookup, or repeats factor
  vector work.
- Checked whether a proposed removal would preserve the current SGD update
  order and the source values used by the update.
- Classified candidates as `EXACT`, `EXACT_BUT_ORDER_SENSITIVE`,
  `APPROXIMATION`, or `RESEARCH_CHANGE`.
- Added a coldpath aggregate helper in
  `src/recsys_lab/experiments/duplication_profile.py` to convert
  `estimated_kernel_work` counters into duplication-cost counters. The helper
  emits aggregates only and does not serialize arrays.

Definitions:

- `EXACT`: mathematically and update-order identical. No formula, update timing,
  history order, rating order, or floating-point operation grouping changes
  beyond trivial scalar or ID reuse.
- `EXACT_BUT_ORDER_SENSITIVE`: mathematically intended to be identical, but
  floating-point operation grouping or pass structure can produce small numeric
  differences even if the same conceptual values are used.
- `APPROXIMATION`: changes update timing, uses stale values, changes
  context/update semantics, or removes traversals through a non-identical order.
- `RESEARCH_CHANGE`: introduces new model behavior and must not be treated as a
  kernel optimization.

## Kernel-by-kernel Duplication Table

| Kernel | Explicit residual duplication | Implicit history duplication | Cluster history duplication | Other repeated work | First safe action |
| --- | --- | --- | --- | --- | --- |
| `train_biased_mf_epoch_numba` | None | None | None | None beyond expected old-vector reuse | No residual/history optimization needed |
| `train_svdpp_epoch_numba` | None | Context pass and implicit-factor update pass traverse the same user history | None | `history_start`, `history_end`, `norm` are loaded once and reused | Keep as baseline; pass fusion is not `EXACT` |
| `train_asymmetric_svd_epoch_numba` | `residual_weight` recomputed in explicit context pass and explicit-factor update pass | Context pass and implicit-factor update pass traverse the same user history | None | `explicit_norm * residual_weight` recomputed in factor loop | Cache raw explicit residual weights only; keep update pass order |
| `train_asvdpp_epoch_numba` | Same duplication as `asymmetric_svd` | Same implicit traversal duplication as `asymmetric_svd` | None | `p_old`, `q_old`, context already use per-epoch workspace | Cache raw explicit residual weights only; keep update pass order |
| `train_cb_svdpp_epoch_numba` | None | Context pass and implicit-factor update pass traverse the same user history | Context pass and cluster-factor update pass traverse the same cluster history | `history_cluster`, `history_cluster_count`, `implicit_norm * alpha * count`, `implicit_norm * one_minus_alpha` repeated | Defer pass-merging; consider local scalar hoists only after equality baseline |
| `train_cb_asvdpp_epoch_numba` | `residual_weight` recomputed in explicit context pass and explicit-factor update pass | Context pass and implicit-factor update pass traverse the same user history | Context pass and cluster-factor update pass traverse the same cluster history | `item_clusters[history_item]`, `history_cluster`, `history_cluster_count`, alpha/norm products repeated | First target: cache raw explicit residual weights and explicit history cluster ids; keep loop order |

## Duplication Cost Estimate

The cost estimate is expressed in the same aggregate units used by the
`kernel_profile.json` contract:

- `E`: `explicit_history_visits`
- `I`: `implicit_history_visits`
- `C`: `cluster_history_visits`

The duplicated counters below count the removable second computation or second
lookup/traversal candidate, not total work. For example, if a residual is
computed twice for each explicit history visit, the duplicated residual count is
`E`, not `2E`.

| Model | explicit_history_visits | implicit_history_visits | cluster_history_visits | duplicated_explicit_residual_computations | duplicated_history_cluster_lookups | duplicated_history_traversals | potential exact reuse scope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `biased_mf` | 0 | 0 | 0 | 0 | 0 | 0 | none |
| `svdpp` | 0 | I | 0 | 0 | 0 | I | none |
| `asymmetric_svd` | E | I | 0 | E | 0 | E + I | raw explicit residual weights |
| `asvdpp` | E | I | 0 | E | 0 | E + I | raw explicit residual weights |
| `cb_svdpp` | 0 | I | C | 0 | C | I + C | cluster history ids/counts |
| `cb_asvdpp` | E | I | C | E | E + C | E + I + C | raw explicit residual weights; explicit history cluster ids; cluster history ids/counts |

Local cost-anatomy readout:

- Source: `artifacts/reports/kernel_cost_anatomy.csv`, latest local `ml1m` row
  per model where present.
- Purpose: audit sizing only. These are visit-count aggregates from an existing
  report, not a new benchmark and not a runtime claim.

| Model | implicit_history_visits | explicit_history_visits | cluster_history_visits | duplicated_explicit_residual_computations | duplicated_history_cluster_lookups | duplicated_history_traversals | potential exact reuse scope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `biased_mf` | 0 | 0 | 0 | 0 | 0 | 0 | none |
| `svdpp` | 4,994,765,860 | 0 | 0 | 0 | 0 | 4,994,765,860 | none |
| `asymmetric_svd` | 4,994,765,860 | 4,994,765,860 | 0 | 4,994,765,860 | 0 | 9,989,531,720 | raw explicit residual weights |
| `asvdpp` | 4,994,765,860 | 4,994,765,860 | 0 | 4,994,765,860 | 0 | 9,989,531,720 | raw explicit residual weights |
| `cb_svdpp` | 4,994,765,860 | 0 | 877,047,720 | 0 | 877,047,720 | 5,871,813,580 | cluster history ids/counts |
| `cb_asvdpp` | 4,994,765,860 | 4,994,765,860 | 877,047,720 | 4,994,765,860 | 5,871,813,580 | 10,866,579,440 | raw explicit residual weights; explicit history cluster ids; cluster history ids/counts |

## Explicit Residual Duplication Findings

### `train_asymmetric_svd_epoch_numba`

The explicit history is traversed during context construction. For each explicit
history entry, the kernel computes:

- `history_item`
- `history_item_bias`
- `residual_weight = explicit_rating - (global_mean + user_bias_old + history_item_bias)`

The explicit history is traversed again during `explicit_factors` updates and
the same `history_item_bias` and `residual_weight` are recomputed.

The recomputation uses `user_bias_old` and a guarded `history_item_bias`.
For `history_item == item_id`, it intentionally uses `item_bias_old` because
`item_bias[item_id]` has already been updated before the explicit-factor update
pass. For other history items, `item_bias[history_item]` is read in both passes.

Exactness note:

- Caching the residual weight computed in the first explicit pass and replaying
  it in the second explicit pass is exact only if the current read semantics are
  preserved.
- The cache must be per rating and indexed in the same explicit history order.
- The explicit-factor update loop must remain separate and ordered exactly as it
  is today.

### `train_asvdpp_epoch_numba`

The explicit residual duplication is the same as in `asymmetric_svd`.

Additional user-factor updates do not invalidate the residual cache because the
residual formula uses `user_bias_old`, `item_bias_old` for the current item, and
item biases for history items. It does not use `user_factors`.

Exactness note:

- Same exact local residual-cache candidate as `asymmetric_svd`.
- Do not merge explicit context construction with explicit-factor updates.

### `train_cb_asvdpp_epoch_numba`

The explicit context pass computes:

- `history_item`
- `history_cluster = item_clusters[history_item]`
- `history_item_bias`
- `residual_weight`
- explicit individual contribution
- explicit cluster contribution

The explicit update pass recomputes:

- `history_item`
- `history_cluster = item_clusters[history_item]`
- `history_item_bias`
- `residual_weight`

Then it updates both `explicit_factors` and `explicit_cluster_factors`.

Exactness note:

- Caching `residual_weight` and `history_cluster` for the explicit history range
  is the most direct exact candidate.
- Caching `explicit_norm * residual_weight` and possibly alpha-scaled variants
  may also be exact, but should be treated as a second micro-slice because it
  changes floating expression grouping.
- The explicit update loop must remain separate and in the same order.

## Implicit History Duplication Findings

### `train_svdpp_epoch_numba`

The implicit history is traversed once to build `z_user` and again to update
`implicit_factors`.

This is duplicated traversal, but the first pass reads `implicit_factors` before
the prediction/error and before updates. The second pass mutates
`implicit_factors` after user/item updates.

Classification: `APPROXIMATION`.

Do not merge these passes in the first slice. A local cache of `history_item`
would save only an array lookup and add workspace/memory traffic. It is unlikely
to be the strongest first candidate without benchmark evidence.

### `train_asymmetric_svd_epoch_numba`

The implicit history is traversed once to add implicit factors to `context` and
again to update `implicit_factors`.

Classification: `APPROXIMATION`.

The same reasoning applies: the context pass reads old implicit factors; the
update pass mutates them after prediction/error and item-factor update.

### `train_asvdpp_epoch_numba`

The implicit traversal pattern is the same as `asymmetric_svd`.

Classification: `APPROXIMATION`.

### `train_cb_svdpp_epoch_numba`

The implicit item history is traversed once for individual implicit contribution
to `context` and once to update `implicit_factors`.

Classification: `APPROXIMATION`.

Potential exact local scalar hoists:

- `implicit_norm * one_minus_alpha`
- `error * implicit_norm * one_minus_alpha`

These do not remove traversal, but they reduce repeated scalar multiplication.
They should be benchmarked after the explicit residual-cache slice.

### `train_cb_asvdpp_epoch_numba`

The implicit item history is traversed once for individual implicit contribution
to `context` and once to update `implicit_factors`.

Classification: `APPROXIMATION`.

As with `cb_svdpp`, scalar hoists may be exact but are not the primary
duplication target.

## Cluster History Duplication Findings

### `train_cb_svdpp_epoch_numba`

The cluster history is traversed once to add cluster implicit factors to
`context` and again to update `implicit_cluster_factors`.

The second pass repeats:

- `history_cluster = int(cluster_ids[cluster_pos])`
- `history_cluster_count = float(cluster_counts[cluster_pos])`

Classification:

- Reusing `history_cluster` and `history_cluster_count` in local per-rating
  buffers is `EXACT` if order is unchanged.
- Merging the context and update passes is `APPROXIMATION` because
  `implicit_cluster_factors` are read before prediction and mutated after error
  computation and main factor updates.

### `train_cb_asvdpp_epoch_numba`

The cluster history duplication is the same as `cb_svdpp`.

Additionally, explicit history uses `item_clusters[history_item]` in both the
explicit context and explicit update passes. That explicit item-cluster lookup is
a stronger first candidate than cluster-history pass merging.

Classification:

- Caching explicit `history_cluster = item_clusters[history_item]` is `EXACT`
  if the explicit history order is unchanged.
- Caching cluster-history ids/counts is `EXACT` but likely lower priority than
  explicit residual caching.
- Merging cluster context and update passes is `APPROXIMATION`.

## Exactness Classification

### EXACT Candidates

These candidates can preserve current semantics if implemented as local
per-rating workspace reuse and if all loop orders remain unchanged:

1. Cache explicit residual weights in `asymmetric_svd`.
2. Cache explicit residual weights in `asvdpp`.
3. Cache explicit residual weights in `cb_asvdpp`.
4. Cache explicit history cluster ids in `cb_asvdpp`.
5. Cache per-rating cluster-history ids/counts in `cb_svdpp` and `cb_asvdpp`.

Only the first four `EXACT` candidates are recommended for the first real
optimization step. Candidate 5 is exact but lower priority because it targets
smaller scalar/id reloads rather than duplicated residual computation.

### EXACT_BUT_ORDER_SENSITIVE Candidates

These candidates keep the same mathematical intent but can change
floating-point operation grouping. They are not allowed in the first
implementation slice.

1. Hoist products into new precomputed scaled terms if doing so changes
   expression grouping, for example:
   - `implicit_norm * one_minus_alpha`
   - `implicit_norm * alpha`
   - `explicit_norm * one_minus_alpha`
   - `explicit_norm * alpha`
2. Cache scaled residual products such as `explicit_norm * residual_weight`
   instead of caching the raw `residual_weight`.
3. Combine alpha, norm, residual, and error terms into a single cached multiplier
   before the current factor loop.

These may pass tolerance-based comparisons, but they are not strict first-slice
work because they can change floating-point evaluation order.

### APPROXIMATION Candidates

These candidates remove larger traversals or reuse contexts by changing when
mutable values are read or updated. They are not implementation candidates for
Step 16 optimization.

1. Merge context construction and implicit-factor update traversals.
2. Merge context construction and cluster-factor update traversals.
3. Convert per-rating history passes into per-user precomputed contexts across
   multiple ratings.
4. Cache full context vectors across repeated user ratings inside an epoch.
5. Reuse residual weights across ratings or epochs.
6. Reuse contexts across ratings for the same user when mutable factors may have
   changed.
7. Replace residual weights with approximate or stale values.

Reason: the kernels mutate factors after each rating. Reusing a context computed
before another rating update can silently change SGD semantics.

### RESEARCH_CHANGE Candidates

These candidates introduce new model behavior and must be handled as method
changes, not optimizations:

1. Reorder ratings, explicit histories, implicit histories, or cluster histories.
2. Update explicit, implicit, or cluster factors during context construction.
3. Change the use of `item_bias_old` for the current item in explicit residual
   formulas after `item_bias[item_id]` has been updated.
4. Pre-aggregate duplicate history entries in a way that changes current update
   multiplicity or order.
5. Change history deduplication, residual definitions, norm formulas, or cluster
   contribution formulas.

## Optimization Candidates

### Candidate A: Explicit Residual Workspace

Classification: `EXACT`.

Target kernels:

- `train_asymmetric_svd_epoch_numba`
- `train_asvdpp_epoch_numba`
- `train_cb_asvdpp_epoch_numba`

Plan:

- Allocate a per-rating workspace sized to the maximum explicit history length
  supported by the kernel call or use a local array sized from the current
  explicit range if Numba behavior and allocation cost are acceptable.
- During explicit context construction, store raw `residual_weight` at the same
  relative history position.
- During explicit update pass, load the cached residual instead of recomputing.
- Keep explicit update order unchanged.

Exactness checks:

- `user_bias_old` is captured before the prediction and before the bias update.
  Both residual computations currently use this old value.
- `item_bias_old` is captured before the current item bias update. For
  `history_item == item_id`, the residual formula intentionally uses
  `item_bias_old` in both passes.
- For `history_item != item_id`, both passes read `item_bias[history_item]`.
  The first-slice implementation is exact only if no intervening update can
  change those non-current history item biases before the second pass. In the
  current kernels, the only item bias updated between context construction and
  explicit-factor update is `item_bias[item_id]`.
- The cached value must be the raw `residual_weight`, not a scaled product.
  Reusing the raw scalar preserves the existing multiplication grouping inside
  the later factor update expression.
- The workspace must be per rating and indexed by the existing explicit history
  order. It must not sort, deduplicate, or reuse entries across ratings.

Answer:

- Yes, caching the raw residual weights is `EXACT` for the three target kernels
  under the constraints above. It reuses the same values that the second pass
  currently recomputes after the current item bias update, including the
  deliberate `item_bias_old` branch for `history_item == item_id`.

Risk:

- Variable-length workspace can add memory traffic.
- Per-rating allocation would defeat part of the purpose; prefer reusable
  workspace if feasible.

Evidence needed:

- exact output comparison against baseline for synthetic cases
- warm-run harness benchmark before/after
- no new portable runtime claim

### Candidate B: Explicit Item-Cluster Workspace

Classification: `EXACT`.

Target kernel:

- `train_cb_asvdpp_epoch_numba`

Plan:

- During explicit context construction, cache `history_cluster`.
- Reuse it during explicit individual and cluster factor update pass.
- Keep explicit history order unchanged.

Exactness checks:

- `item_clusters` is immutable during training.
- `history_item` comes from the explicit CSR-like history arrays and is not
  mutated inside the kernel.
- The cached value is an integer ID. It does not change floating-point operation
  order.

Answer:

- Yes, caching `history_cluster = item_clusters[history_item]` in
  `train_cb_asvdpp_epoch_numba` is `EXACT` if the cache is per rating and keyed
  by the unchanged explicit history position.

Risk:

- Small scalar lookup savings may be offset by workspace writes.
- Best bundled with Candidate A in `cb_asvdpp` because the same explicit range
  is already being cached.

### Candidate C: Precompute Explicit Norm Times Residual

Classification: `EXACT_BUT_ORDER_SENSITIVE`.

Target kernels:

- `train_asymmetric_svd_epoch_numba`
- `train_asvdpp_epoch_numba`
- `train_cb_asvdpp_epoch_numba`

Plan:

- Precompute `explicit_norm * residual_weight` for each explicit history entry
  inside one rating update.
- Reuse the scaled value during context construction and explicit-factor update.

Exactness checks:

- The mathematical scalar is intended to be the same within the rating update.
- However, the current explicit-factor update expression multiplies
  `error * explicit_norm * residual_weight * q_mix_old[...]` or the non-CB
  equivalent. Precomputing `explicit_norm * residual_weight` can change the
  grouping and rounding path relative to the current expression.
- This is stricter than Candidate A because it caches a rounded product, not the
  raw residual input to the existing expression.

Answer:

- Treat Candidate C as `EXACT_BUT_ORDER_SENSITIVE`, not as a first-slice exact
  optimization. It may be acceptable later with tolerance-based numerical
  evidence, but it is outside the strict first implementation slice.

Risk:

- It may produce tiny floating-point differences even though the mathematical
  intent is unchanged.
- It is not needed to remove the explicit residual recomputation; Candidate A is
  the stricter path.

### Candidate D: Fuse Context Construction and Explicit Update Pass

Classification: `APPROXIMATION`.

Target kernels:

- `train_asymmetric_svd_epoch_numba`
- `train_asvdpp_epoch_numba`
- `train_cb_asvdpp_epoch_numba`

Plan:

- Fuse explicit context construction with explicit-factor updates to avoid a
  second explicit-history traversal.

Exactness checks:

- The explicit context is needed before prediction.
- The prediction error is known only after the context vector has been fully
  constructed and the prediction has been computed.
- Explicit-factor updates currently happen after prediction and after user/item
  bias updates.
- Moving explicit-factor updates into the context pass would either require a
  stale or unavailable `error`, or it would defer writes in a way that recreates
  the second pass.

Answer:

- Candidate D is not exact. Fusing these passes changes update timing or requires
  a deferred update structure that is no longer a simple pass fusion. Classify it
  as `APPROXIMATION` unless a future design proves identical ordering with a
  separate evidence protocol.

Risk:

- It can silently change SGD semantics.
- It is not allowed in the next implementation slice.

### Candidate E: Fuse Implicit Context Construction and Implicit Update Pass

Classification: `APPROXIMATION`.

Target kernels:

- `train_svdpp_epoch_numba`
- `train_asymmetric_svd_epoch_numba`
- `train_asvdpp_epoch_numba`
- `train_cb_svdpp_epoch_numba`
- `train_cb_asvdpp_epoch_numba`

Plan:

- Fuse implicit context construction with implicit-factor updates to avoid a
  second implicit-history traversal.

Exactness checks:

- The implicit context contribution is part of the prediction input.
- The implicit-factor update uses `error`, which is only available after
  prediction.
- The update also uses `q_old` or `q_mix_old`, which is intentionally captured
  before item-factor updates.
- Performing implicit updates during context construction changes when mutable
  implicit factors are read and written within the rating update.

Answer:

- Candidate E is not exact. It is an `APPROXIMATION` because the fused pass cannot
  both build the old-state context and update factors using the later error
  without changing the pass structure or update timing.

Risk:

- It targets a large traversal count, but it is not a strict optimization
  candidate for Step 16.
- It belongs in a later research or approximation track if ever considered.

### Candidate F: Cluster History Counts and Scalars

Classification: `EXACT` for raw IDs/counts; `EXACT_BUT_ORDER_SENSITIVE` for
scaled products.

Target kernels:

- `train_cb_svdpp_epoch_numba`
- `train_cb_asvdpp_epoch_numba`

Plan:

- Cache raw `history_cluster = cluster_ids[cluster_pos]` and
  `history_cluster_count = cluster_counts[cluster_pos]` for the current
  cluster-history range.
- Optionally consider scalar products such as
  `implicit_norm * alpha * history_cluster_count` only after the raw-id/count
  case has evidence.

Exactness checks:

- `cluster_ids` and `cluster_counts` are immutable during the kernel call.
- Caching raw integer IDs and raw counts preserves lookup values and does not
  change floating-point grouping.
- Caching precomputed products involving `implicit_norm`, `alpha`, or
  `one_minus_alpha` can change grouping in expressions such as
  `error * implicit_norm * alpha * history_cluster_count * q_mix_old[...]`.

Answer:

- Raw `history_cluster` and `history_cluster_count` reuse is `EXACT`.
- Precomputed scaled products are `EXACT_BUT_ORDER_SENSITIVE`.
- This is lower priority than Candidate A/B because it removes scalar/id reloads
  rather than duplicated residual computations.

Risk:

- Workspace writes may exceed scalar lookup savings for small cluster histories.
- It should be benchmarked separately from Candidate A/B if implemented later.

## No-go Candidates

The following should not be implemented in Step 16:

- `APPROXIMATION`: residual caching across ratings
- `APPROXIMATION`: residual caching across epochs
- `APPROXIMATION`: context caching across ratings for the same user
- `APPROXIMATION`: user-level precomputed explicit/implicit context vectors
- `RESEARCH_CHANGE`: history traversal reordering
- `RESEARCH_CHANGE`: cluster-history reordering
- `RESEARCH_CHANGE`: moving factor updates into context construction passes
- `RESEARCH_CHANGE`: changing residual formulas
- `RESEARCH_CHANGE`: changing how current-item bias uses `item_bias_old`
- `RESEARCH_CHANGE`: changing history deduplication or layout contracts from Step 15

These would either change semantics or require a separate method-change
proposal.

## Required Benchmark/Evidence Protocol

Before any implementation:

1. Run baseline warm-run harness for all six synthetic tiny cases:

   `python scripts/run_kernel_benchmarks.py --case all --warmup-repeats 1 --timed-repeats 5 --output-dir artifacts/benchmarks/kernel/residual_history_duplication_baseline_v1`

2. Add or use deterministic before/after equality tests that compare mutated
   arrays after one epoch for the target kernels:

   - user bias
   - item bias
   - user factors where present
   - item factors
   - explicit factors where present
   - implicit factors where present
   - cluster factors where present

3. Run focused tests:

   - `pytest tests/unit/test_duplication_profile.py`
   - `pytest tests/unit/test_kernel_benchmark_harness.py`
   - `pytest tests/integration/test_kernel_benchmark_harness_tiny.py`
   - `pytest tests/unit/test_hotpath_coldpath_boundaries.py`
   - target kernel equivalence tests added for the optimization slice

4. Run the script smoke again after implementation.

5. Record evidence in a new performance evidence file. Tiny benchmark values may
   be reported only as synthetic sanity readouts, not as performance claims.

6. Do not use absolute runtime thresholds in tests.

Baseline generated for this audit:

- Command:
  `python scripts/run_kernel_benchmarks.py --case all --warmup-repeats 1 --timed-repeats 5 --output-dir artifacts/benchmarks/kernel/residual_history_duplication_baseline_v1`
- Summary:
  `artifacts/benchmarks/kernel/residual_history_duplication_baseline_v1/kernel_benchmark_summary.csv`
- JSON payloads:
  `artifacts/benchmarks/kernel/residual_history_duplication_baseline_v1/<benchmark_id>/kernel_benchmark.json`
- Coverage:
  `biased_mf`, `svdpp`, `asymmetric_svd`, `asvdpp`, `cb_svdpp`, `cb_asvdpp`
- Contract check:
  six payloads, `timed_repeats=5`, `warmup_repeats=1`,
  `compile_excluded=true`, and `state_copy_excluded=true`.
- Interpretation:
  synthetic tiny baseline only for later before/after sanity comparison; no
  performance claim.

Audit helper tests added:

- `test_duplication_profile_counts_explicit_residual_duplication`
- `test_duplication_profile_counts_cb_asvdpp_cluster_lookup_duplication`
- `test_duplication_profile_has_no_explicit_duplication_for_biased_mf`
- `test_duplication_profile_has_no_explicit_duplication_for_svdpp`

## Recommended First Implementation Slice

First slice:

- implement explicit residual local caching for:
  - `train_asymmetric_svd_epoch_numba`
  - `train_asvdpp_epoch_numba`
  - `train_cb_asvdpp_epoch_numba`
- in `train_cb_asvdpp_epoch_numba`, cache explicit `history_cluster` in the same
  explicit-history workspace
- include only candidates classified as `EXACT`
- keep all factor update loops and all history iteration order unchanged
- do not merge implicit or cluster history passes
- do not hoist scaled residual or alpha/norm products in this first slice
- do not change synthetic cases, history layout contracts, or model formulas

Why this slice:

- it targets the clearest duplicated scalar work
- it does not require changing update order
- it is directly exercisable by the Step 14 benchmark harness
- it is supported by the Step 15 contiguous `int32` history layout
- it avoids higher-risk context caching and pass fusion

Acceptance criteria for the first implementation slice:

- equivalence tests pass for all mutated arrays in target kernels
- Kernel Benchmark Harness still runs for all six cases
- hotpath/coldpath boundary tests remain green
- no model formula, rating order, history order, or cache layout changes
- evidence states only what was measured and avoids portable runtime claims
