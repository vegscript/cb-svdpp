# CB Kernel Specialization Audit V1

## Executive Summary

This audit reviews the clustering-based training kernels for specialization
opportunities. It does not implement a new kernel and does not change model
semantics.

The main finding is that `alpha` controls a large amount of redundant work in
the generic CB kernels, but endpoint specialization is only exact if it
preserves the current update semantics, including regularization updates for
zero-gradient branches. Skipping inactive branches at `alpha == 0` or
`alpha == 1` would change behavior unless the model contract is explicitly
changed.

The recommended first implementation slice is therefore conservative:

- add an explicit dispatch/audit plan for exact alpha-specialized kernels
- start with a minimal `alpha == 0` and/or `alpha == 1` specialization only if
  equivalence tests compare all mutated arrays against the generic kernel
- preserve rating order, history order, update order, and regularization
  semantics

No performance claim follows from this audit.

## Method

Files reviewed:

- `src/recsys_lab/models/kernels.py`
- `src/recsys_lab/benchmarks/kernel_harness.py`
- `src/recsys_lab/benchmarks/synthetic_kernel_cases.py`
- `src/recsys_lab/data/histories.py`
- `docs/evidence/performance/history_data_layout_v1.md`
- `docs/evidence/performance/kernel_benchmark_harness_v1.md`
- `docs/performance/residual_history_duplication_audit_v1.md`
- `docs/evidence/performance/exact_residual_reuse_v1.md`
- `docs/evidence/performance/exact_residual_reuse_validation_v1.md`

Kernels inspected:

- reference kernels:
  - `train_svdpp_epoch_numba`
  - `train_asvdpp_epoch_numba`
- CB kernels:
  - `train_cb_svdpp_epoch_numba`
  - `train_cb_asvdpp_epoch_numba`

The audit classifies specialization candidates by exactness:

- `EXACT`: identical model behavior for a strictly defined configuration case.
- `EXACT_BUT_ORDER_SENSITIVE`: same mathematical intent, but possible
  floating-point order differences.
- `APPROXIMATION`: semantics, update timing, weighting, or traversal behavior
  changes.
- `RESEARCH_CHANGE`: new model behavior, not an optimization.

This audit also distinguishes parameter visibility levels:

- `parameter-identical exact`: all mutated arrays match the generic kernel.
- `prediction-identical only`: predictions match for a fixed configuration,
  but stored or serialized parameters can differ.
- `observable-metrics-identical only`: metrics match in tested runs, but neither
  predictions nor parameters are guaranteed identical.
- `not exact`: update rules, parameter state, prediction semantics, or ordering
  differ without a strict equivalence argument.

For implementation planning, only `parameter-identical exact` candidates are
safe by default. A `prediction-identical only` candidate can be considered only
if the changed parameters are proven never to be read, serialized, reused,
reported, or made relevant by later config changes. No such proof exists for
the current CB factor families.

## Current CB Kernel Anatomy

### `train_cb_svdpp_epoch_numba`

Inputs include individual factors:

- `user_factors`
- `item_factors`
- `implicit_factors`

Inputs include cluster factors:

- `user_cluster_factors`
- `item_cluster_factors`
- `implicit_cluster_factors`

Cluster assignments and count histories:

- `user_clusters`
- `item_clusters`
- `cluster_indptr`
- `cluster_ids`
- `cluster_counts`

Per rating, the kernel:

- reads current user and item IDs
- reads user and item cluster IDs
- snapshots individual and cluster user/item factors
- computes `one_minus_alpha = 1.0 - alpha`
- computes `q_mix_old = one_minus_alpha * q_old + alpha * q_cluster_old`
- initializes context from individual and cluster user factors
- traverses implicit item history for individual implicit context
- traverses cluster-count history for cluster implicit context
- computes prediction and error
- updates individual user/item factors
- updates cluster user/item factors
- traverses implicit item history again for individual implicit-factor updates
- traverses cluster-count history again for cluster implicit-factor updates

### `train_cb_asvdpp_epoch_numba`

This kernel includes all `cb_svdpp` inputs and additionally:

- `explicit_indptr`
- `explicit_items`
- `explicit_ratings`
- `explicit_norms`
- `explicit_factors`
- `explicit_cluster_factors`

Per rating, it additionally:

- traverses explicit history for individual explicit context
- traverses explicit history for cluster explicit context
- stores raw explicit residual weights from Step 16b
- stores explicit `history_cluster` IDs from Step 16b
- traverses explicit history again for individual and cluster explicit-factor
  updates

### Alpha-Dependent Work

These paths directly depend on `alpha` or `one_minus_alpha`:

- `q_mix_old`
- initial context mixing
- individual implicit/explicit context terms
- cluster implicit/explicit context terms
- individual user/item updates
- cluster user/item updates
- individual implicit/explicit factor updates
- cluster implicit/explicit factor updates

### Endpoint Observation

At `alpha == 0`, cluster gradient terms are zero, but current cluster updates
still apply regularization terms such as `-lambda_pC * p_cluster_old`. Skipping
cluster updates is therefore not exact under the current formula.

At `alpha == 1`, individual gradient terms are zero, but current individual
updates still apply regularization terms such as `-lambda_p * p_old`. Skipping
individual updates is therefore not exact under the current formula.

## Candidate Specialization Table

| Candidate | Scope | Description | Classification | Notes |
| --- | --- | --- | --- | --- |
| A | CB-SVD++ `alpha == 0` | Specialized kernel simplifies mix/context terms to individual path while preserving current cluster regularization updates. | `EXACT` only if parameter-identical | Exact only if all mutated arrays match generic kernel, including cluster-factor shrinkage. Skipping cluster updates is not exact. |
| B | CB-ASVD++ `alpha == 0` | Same as A, plus explicit individual path remains active and explicit cluster path has zero-gradient but current regularization behavior. | `EXACT` only if parameter-identical | Must preserve Step 16b residual reuse and update order. Skipping explicit/implicit cluster factor updates is not exact. |
| C | CB-SVD++ `alpha == 1` | Specialized kernel simplifies mix/context terms to cluster path while preserving current individual regularization updates. | `EXACT` only if parameter-identical | Exact only if individual factor shrinkage remains identical. Skipping individual updates is not exact. |
| D | CB-ASVD++ `alpha == 1` | Same as C with explicit cluster path active and explicit individual path zero-gradient plus regularization. | `EXACT` only if parameter-identical | Must preserve explicit history order and residual semantics. Skipping individual explicit/implicit updates is not exact. |
| E | CB-SVD++ common constant alpha | Specialized kernel for a fixed non-endpoint alpha, replacing runtime alpha with compile-time constant. | `EXACT` | Exact if operations are kept in the same order; benefit may be small. |
| F | CB-ASVD++ common constant alpha | Same as E for explicit + implicit + cluster history. | `EXACT` | Must not reorder products or combine loops. |
| G | Hoist local scalar products | Reuse products such as `implicit_norm * alpha` or `implicit_norm * one_minus_alpha` within the same rating update. | `EXACT_BUT_ORDER_SENSITIVE` | Algebra is equivalent, but product grouping can change floating-point rounding. |
| H | Split individual and cluster update loops | Separate alpha-active and inactive branches into distinct loops while preserving order within each branch. | `EXACT_BUT_ORDER_SENSITIVE` | Risk is changed interleaving and floating-point/order differences. |
| I | Skip cluster history traversal when `alpha == 0` | Do not visit cluster history at all for endpoint alpha. | `APPROXIMATION` | Not exact unless equivalent cluster-factor regularization updates are still applied. |
| J | Skip individual history traversal when `alpha == 1` | Do not visit individual history at all for endpoint alpha. | `APPROXIMATION` | Not exact unless equivalent individual-factor regularization updates are still applied. |
| K | Freeze inactive factor families | At endpoints, do not update factors multiplied by zero-gradient terms. | `RESEARCH_CHANGE` | Changes regularization/training semantics. |
| L | Treat `alpha == 0` as plain SVD++/ASVD++ | Dispatch to non-CB kernels and ignore cluster parameters. | `RESEARCH_CHANGE` | Useful ablation idea, not equivalent to current CB kernel semantics. |
| M | Treat `alpha == 1` as cluster-only model | Train only cluster factors and ignore individual factors. | `RESEARCH_CHANGE` | New model variant, not a specialization of existing behavior. |
| N | Pass fusion for context and history updates | Fuse context construction with update passes. | `APPROXIMATION` | Error is not known until after prediction; update timing/order changes. |

## Exactness Classification

## Required Candidate Checks

### Candidate A: `alpha == 0` fast path for CB-SVD++

Question: can cluster factor reads and updates be fully avoided without
changing visible model semantics?

Finding:

- At `alpha == 0`, cluster factors do not contribute to `q_mix_old`, context,
  prediction, individual-factor gradients, or implicit-factor gradients.
- The generic kernel still reads:
  - `user_clusters`
  - `item_clusters`
  - `user_cluster_factors`
  - `item_cluster_factors`
  - `cluster_indptr`
  - `cluster_ids`
  - `cluster_counts`
  - `implicit_cluster_factors`
- The generic kernel still updates cluster factors through regularization:
  - `user_cluster_factors`: gradient term is zero, regularization remains
  - `item_cluster_factors`: gradient term is zero, regularization remains
  - `implicit_cluster_factors`: gradient term is zero, regularization remains

Classification:

- Avoiding cluster reads for prediction/context only can be `EXACT`.
- Avoiding cluster updates entirely is not fully exact under the current model
  state contract because cluster parameter values would differ.
- It may be prediction-equivalent while `alpha` remains `0`, but that is only
  observationally exact for predictions, not exact for all mutated arrays or
  serialized model state.

Conclusion:

- A fully exact `alpha == 0` CB-SVD++ kernel must preserve cluster-factor
  regularization effects or explicitly document that it is only
  prediction-equivalent.
- A fast path that leaves unused cluster parameters unchanged is
  `RESEARCH_CHANGE` unless the model contract is changed.
- Under this audit's strict rule, the implementation-safe subset is
  `parameter-identical exact`. A prediction-identical but parameter-different
  endpoint fast path is not accepted as the first implementation slice.

### Candidate B: `alpha == 0` fast path for CB-ASVD++

Question: same as A, with explicit and implicit histories plus explicit cluster
factors.

Finding:

- Individual explicit and implicit paths remain active.
- Cluster explicit and implicit contributions are zero-weighted in prediction.
- The generic kernel still updates:
  - `user_cluster_factors`
  - `item_cluster_factors`
  - `explicit_cluster_factors`
  - `implicit_cluster_factors`
- Those updates still include regularization terms.
- Step 16b raw residual workspace and explicit cluster workspace do not change
  this conclusion; they only avoid recomputing exact per-rating values.

Classification:

- Simplifying zero-weighted cluster context terms is `EXACT` only if the same
  cluster regularization updates are preserved.
- Skipping all cluster-factor work is prediction-equivalent at fixed
  `alpha == 0`, but not exact for mutated arrays.

Conclusion:

- A fully exact CB-ASVD++ `alpha == 0` specialization must preserve
  regularization effects for all cluster factor families.
- Dropping explicit cluster factor updates is `RESEARCH_CHANGE`, not a safe
  optimization.
- Prediction-identical behavior at fixed `alpha == 0` is insufficient because
  cluster parameters remain stored model state and can become relevant if
  `alpha` changes for later prediction, inspection, serialization, or
  diagnostics.

### Candidate C: `alpha == 1` cluster-only path for CB-SVD++

Question: can individual factor work be removed?

Finding:

- At `alpha == 1`, individual factors do not contribute to `q_mix_old`,
  context, prediction, or cluster gradients.
- The generic kernel still reads and updates:
  - `user_factors`
  - `item_factors`
  - `implicit_factors`
- Individual updates have zero gradient contribution but nonzero
  regularization terms.

Classification:

- Removing individual contribution calculations from context/prediction can be
  `EXACT` if individual regularization updates are preserved.
- Skipping individual factor updates is prediction-equivalent at fixed
  `alpha == 1`, but not exact for model state.

Conclusion:

- A true exact `alpha == 1` specialization must still mutate individual
  factors according to the generic regularization behavior.
- A cluster-only training kernel is a `RESEARCH_CHANGE`.
- Prediction-identical behavior at fixed `alpha == 1` does not imply
  parameter-identical exactness.

### Candidate D: `alpha == 1` cluster-only path for CB-ASVD++

Question: same as C, with explicit and implicit histories.

Finding:

- Cluster explicit and implicit paths remain active.
- Individual explicit and implicit paths are zero-weighted in prediction.
- Generic updates still regularize:
  - `user_factors`
  - `item_factors`
  - `explicit_factors`
  - `implicit_factors`

Classification:

- Removing zero-weighted individual context contributions can be `EXACT` only
  with equivalent individual regularization updates.
- Skipping individual factor-family updates is a `RESEARCH_CHANGE`.

Conclusion:

- A cluster-only CB-ASVD++ kernel is not an exact optimization of the current
  model unless it preserves all individual-family regularization mutations.
- Individual factors, explicit factors, and implicit factors remain visible
  stored state; leaving them unchanged is not exact under the current contract.

### Candidate E: keep generic kernel for `0 < alpha < 1`

Question: should the generic kernel remain for true mix cases?

Finding:

- For non-endpoint alpha values, both individual and cluster paths contribute
  to context, prediction, and updates.
- Removing either path changes the model.
- Constant-alpha specialization may remove runtime scalar handling but does not
  remove a full factor family.

Classification:

- Keeping the generic kernel for `0 < alpha < 1` is the correct baseline.
- Specialized constant-alpha kernels for common non-endpoint values can be
  `EXACT` only if expression order and update order stay identical.

Conclusion:

- The generic CB kernels should remain the canonical implementation for mixed
  alpha values.
- Non-endpoint specialization is lower priority than endpoint audit/test
  scaffolding.

### Candidate F: no-cluster-history fast path

Question: if a user has no cluster history, can the cluster-history loop be
skipped?

Finding:

- The current CSR-like layout already makes the loop empty when
  `cluster_start == cluster_end`.
- No body work occurs for empty cluster-history ranges.
- The per-rating boundary reads remain, but they are cheap relative to history
  traversal.

Classification:

- Adding an explicit branch for empty cluster history is not needed for
  correctness.
- A separate specialized kernel for this case is not justified by the current
  audit.

Conclusion:

- No separate no-cluster-history specialization is recommended.
- If later profiling shows boundary reads matter, a scalar guard could be
  considered as `EXACT`, but it is not a first slice.

### Candidate G: fixed dtype/layout specialized dispatch

Question: can Step 15's int32/contiguous/float32 contract support dtype/layout
specialized kernels?

Finding:

- Step 15 already constrains hotpath histories to int32, contiguous arrays and
  float32/float64 value arrays.
- Numba specializes compiled signatures by argument dtype and layout.
- The benchmark synthetic cases already use int32 and float32.

Classification:

- Adding separate Python-level dispatch solely for int32/float32 is unlikely to
  change semantics, but it is not a clear kernel specialization win.
- Relying on Numba signatures plus layout validators is the current exact and
  lower-risk approach.

Conclusion:

- Do not add dtype/layout dispatch as a first CB specialization.
- Keep dtype/layout guarantees in validators and tests.

### Candidate H: model-level dispatch based on alpha

Question: where should dispatch live?

Finding:

- The runner should not know kernel specialization details.
- Registry and model requirements should not branch on hotpath implementation
  variants.
- `CBSVDppRecommender.fit(...)` and `CBASVDppRecommender.fit(...)` already own
  the call to the Numba kernel and have access to `self.config.alpha`.

Classification:

- Model-wrapper dispatch is the cleanest exact boundary:
  - `if alpha == 0.0: call alpha0 kernel`
  - `elif alpha == 1.0: call alpha1 kernel`
  - `else: call generic kernel`
- This dispatch is `EXACT` only if each specialized kernel is exact.

Conclusion:

- Dispatch should live in the model wrapper, not the runner, registry, or
  requirements layer.
- The benchmark harness can dispatch explicitly for diagnostic cases, but that
  must remain coldpath-only.

### Candidate I: CB-ASVD++ explicit-residual specialized path

Question: after Step 16b, is there another exact specialization only for the
explicit cluster path?

Finding:

- Step 16b already caches raw residual weights and explicit history cluster IDs
  per rating.
- At `alpha == 0`, explicit cluster context and update gradient terms are
  zero-weighted, but explicit cluster factor regularization still mutates
  `explicit_cluster_factors`.
- At `alpha == 1`, explicit individual context and update gradient terms are
  zero-weighted, but explicit individual factor regularization still mutates
  `explicit_factors`.

Classification:

- Endpoint simplification of explicit context terms can be `EXACT` if
  regularization-equivalent updates remain.
- Skipping explicit cluster or explicit individual factor updates is
  `RESEARCH_CHANGE`.
- Hoisting explicit scalar products is `EXACT_BUT_ORDER_SENSITIVE`.

Conclusion:

- There is no safe standalone explicit-residual specialization beyond Step 16b
  that should precede endpoint-alpha equivalence tests.
- CB-ASVD++ should follow CB-SVD++ after the endpoint test scaffold proves the
  approach.

## Exactness Classification

### `EXACT`

Exact candidates must preserve:

- rating order
- explicit history order
- implicit history order
- cluster history order
- prediction formula
- error computation point
- update order
- regularization behavior
- mutated array set
- parameter values for every mutated array

Endpoint specialization can be exact only when zero-gradient branches still
receive the same regularization updates as the generic kernel. This is the main
constraint on `alpha == 0` and `alpha == 1` kernels.

The core trap is that a zero-weighted prediction term can still have a
nonzero parameter update through regularization. For example, at `alpha == 0`,
the generic `user_cluster_factors` update keeps the regularization part:

```text
p_cluster_old + learning_rate * (-lambda_pC * p_cluster_old)
```

A specialized kernel that drops this update may be prediction-identical while
`alpha` remains zero, but it is not parameter-identical and is not exact under
the current model-state contract.

Non-endpoint constant-alpha specialization can be exact if it keeps expression
order identical. Its expected benefit is uncertain and must be measured with
the kernel harness before any broader benchmark.

### `EXACT_BUT_ORDER_SENSITIVE`

Scalar hoists and loop restructuring are order-sensitive because they can
change floating-point multiplication grouping or interleaving. They should not
be the first implementation slice unless tests explicitly tolerate
floating-point-level array deltas and the evidence documents the tolerance.

### `APPROXIMATION`

Skipping traversals, dropping update passes, or fusing context/update loops can
change update timing or regularization behavior. These candidates are not valid
as exact cleanup.

Prediction-identical but parameter-different endpoint paths are classified as
not exact for the first implementation unless a separate model contract proves
the changed parameters are never visible or relevant.

### `RESEARCH_CHANGE`

Freezing inactive factor families or dispatching endpoint alpha to non-CB
baseline kernels changes the model being trained. These are valid research
variants only if introduced as named model variants with separate methodology
documentation.

## No-go Candidates

Do not implement in this step:

- any kernel specialization before this audit is accepted
- pass fusion
- history reordering
- rating reordering
- dropping regularization updates for zero-gradient branches
- dispatching CB endpoint alpha to non-CB kernels
- changing alpha semantics
- freezing inactive individual or cluster factor families
- changing Step 16b residual reuse behavior
- adding benchmark/experiment imports to `src/recsys_lab/models/`

## Cost Model

No new helper was added for this audit. The existing contracts already expose
the required aggregate structure:

- `kernel_profile.json` provides `implicit_history_visits`,
  `explicit_history_visits`, `cluster_history_visits`,
  `estimated_factor_touches`, `latent_dim`, and touch factors.
- `duplication_profile.py` already records history duplication aggregates from
  the same `kernel_profile.json` work contract.
- `kernel_benchmark_summary.csv` validates that synthetic tiny CB cases can be
  measured through the harness, but it is not used for real-data cost sizing.
- `performance_profile.json` confirms the `fit_model` stage is the dominant
  profiled stage in ML1M validation runs, but does not by itself classify
  specialization exactness.

The table below uses the Step 16c local ML1M after-run `kernel_profile.json`
files:

- `artifacts/runs/2026-05-08T131921Z_ml1m_cb_svdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001/kernel_profile.json`
- `artifacts/runs/2026-05-08T130305Z_ml1m_cb_asvdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001/kernel_profile.json`

Touch estimate formula:

```text
estimated_individual_factor_touches =
  (implicit_history_visits * implicit_touch_factor
   + explicit_history_visits * explicit_touch_factor)
  * latent_dim

estimated_cluster_factor_touches =
  cluster_history_visits * cluster_touch_factor * latent_dim
```

The base per-rating touch term is excluded from the individual/cluster split
because the current contract exposes it as one aggregate
`base_rating_touch_factor`. The totals still reconcile with
`estimated_factor_touches` when the base term is added back.

| Model | alpha | implicit_history_visits | explicit_history_visits | cluster_history_visits | estimated_cluster_factor_touches | estimated_individual_factor_touches | potential_alpha0_skippable_work | potential_alpha1_skippable_work | classification |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `cb_svdpp` | `0.1` | `4,994,765,860` | `0` | `877,047,720` | `112,262,108,160` | `639,330,030,080` | `0` parameter-identical; up to `112,262,108,160` prediction-identical-only cluster-history touches | `0` parameter-identical; up to `639,330,030,080` prediction-identical-only individual-history touches | Endpoint fast paths require parameter-identical regularization preservation. |
| `cb_asvdpp` | `0.1` | `4,994,765,860` | `4,994,765,860` | `877,047,720` | `112,262,108,160` | `1,278,660,060,160` | `0` parameter-identical; up to `112,262,108,160` prediction-identical-only cluster-history touches | `0` parameter-identical; up to `1,278,660,060,160` prediction-identical-only individual explicit+implicit history touches | Endpoint fast paths require parameter-identical regularization preservation. |

Interpretation:

- The ML1M validation configs use `alpha = 0.1`, so endpoint-specialization
  costs are planning estimates, not measured endpoint run results.
- The structural cluster-history component is much smaller than individual
  history work for these ML1M runs.
- Under the strict parameter-identical rule, the immediately skippable endpoint
  work is `0` unless the specialized kernel performs equivalent regularization
  updates for the zero-gradient factor families.
- Prediction-identical-only skipped work is not accepted as an exact first
  implementation slice because CB factor arrays remain stored, inspectable, and
  potentially relevant if `alpha` changes for later prediction.

## Benchmark Protocol

Required before any implementation acceptance:

1. Add target synthetic cases to the Kernel Benchmark Harness for endpoint
   alpha values:
   - `cb_svdpp_alpha0`
   - `cb_svdpp_alpha1`
   - `cb_asvdpp_alpha0`
   - `cb_asvdpp_alpha1`
2. Add same-branch generic-vs-specialized equality tests for all mutated arrays
   on deterministic synthetic cases.
3. Verify quality stability through real ML1M laptop runs only after synthetic
   equivalence passes.
4. Keep runtime readouts device-scoped and avoid broad claims.
5. Keep controls:
   - `svdpp`
   - `asvdpp`
   - unchanged generic CB path where applicable
6. Run gates:
   - `ruff check .`
   - focused specialization tests
   - kernel benchmark harness tests
   - hotpath/coldpath boundary tests
   - `pytest tests/unit`
   - targeted integration smoke
   - full `pytest` where feasible

Benchmark artifacts should remain under:

- `artifacts/benchmarks/kernel/`
- `artifacts/reports/`
- `artifacts/runs/`

## Recommended First Implementation Slice

Recommended first slice: implement no kernel yet until this plan is accepted.

For Step 17b, the narrowest plausible implementation slice is:

1. Add synthetic endpoint-alpha benchmark cases.
2. Add generic-vs-specialized equivalence test scaffolding.
3. Implement exactly one endpoint specialized kernel only after the test
   scaffold exists.

Preferred first target:

- `train_cb_svdpp_epoch_numba` with `alpha == 0`

Reasoning:

- smaller than `cb_asvdpp`
- no explicit residual path
- still exercises individual and cluster factor regularization behavior
- can validate whether endpoint specialization can be kept exact before
  touching explicit-history CB-ASVD++

The first implementation must preserve cluster-factor regularization updates.
If the intended optimization requires skipping those updates, it is not this
exact specialization slice and must be treated as a research change.

Equivalence tests must compare all mutated arrays, not only metrics or
predictions. Metrics-only equality is insufficient for acceptance.

## Acceptance Criteria

This audit step is successful if:

- `docs/performance/cb_kernel_specialization_audit_v1.md` exists
- CB-SVD++ and CB-ASVD++ alpha-dependent paths are documented
- individual and cluster factor arrays are identified
- implicit, explicit, and cluster history traversals are identified
- candidate specializations are classified as `EXACT`,
  `EXACT_BUT_ORDER_SENSITIVE`, `APPROXIMATION`, or `RESEARCH_CHANGE`
- no kernel code is changed
- no performance claim is made
- hotpath/coldpath boundary remains intact
- the recommended first implementation slice is conservative and testable
