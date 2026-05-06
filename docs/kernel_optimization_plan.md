# Kernel Optimization Plan V1

This document is a planning artifact. It does not implement kernel changes, change
model formulas, change hyperparameters, or make new performance claims. All
priorities below are hypotheses derived from the current Performance Forensics V1
and Kernel Cost Anatomy V1 artifacts.

## Executive Summary

The current ML1M artifact set shows `fit_model` as the dominant measured stage
for all six model families. Runner, serialization, config, split, prediction,
and reporting stages are visible in the profile data, but they are not the first
optimization target for the kernel phase.

The kernel anatomy reports explain the dominant stage structurally:

- `biased_mf` has only base rating updates.
- `svdpp` adds implicit history visits.
- `asymmetric_svd` and `asvdpp` add explicit and implicit history visits.
- `cb_svdpp` adds implicit and cluster-history visits.
- `cb_asvdpp` combines explicit, implicit, and cluster-history visits and is the
  largest measured `fit_model` stage in the current ML1M run set.

The first recommended optimization phase should be limited to exact changes.
The lowest-risk candidate is workspace reuse for per-rating temporary arrays in
the non-CB kernels, starting with `train_svdpp_epoch_numba` after a small
`biased_mf` guardrail if needed. The CB kernels already allocate their main
workspace arrays outside the per-rating loop, so the first CB changes should not
be attempted before the exact non-CB allocation candidate has been benchmarked.

## Inputs Analyzed

- `artifacts/reports/performance_hotspots.csv`
- `artifacts/reports/performance_stage_breakdown.csv`
- `artifacts/reports/kernel_cost_anatomy.csv`
- `src/recsys_lab/models/kernels.py`
- `src/recsys_lab/models/biased_mf.py`
- `src/recsys_lab/models/svdpp.py`
- `src/recsys_lab/models/asymmetric_svd.py`
- `src/recsys_lab/models/asvdpp.py`
- `src/recsys_lab/models/cb_svdpp.py`
- `src/recsys_lab/models/cb_asvdpp.py`
- `docs/performance_forensics.md`
- `docs/kernel_cost_anatomy.md`

The concrete numbers below refer to the latest complete ML1M runs present in the
current reports for split seed 1 and model seed 1 on
`local_i5_2500k_24gb_benchmark`. They are diagnostic inputs for planning, not
portable performance claims.

## Current Top Bottlenecks

| Rank | Dataset | Model | Stage | Wall clock seconds | Share of profiled time |
| ---: | --- | --- | --- | ---: | ---: |
| 1 | ml1m | cb_asvdpp | fit_model | 1958.666 | 0.974934 |
| 2 | ml1m | asvdpp | fit_model | 1019.779 | 0.990898 |
| 3 | ml1m | asymmetric_svd | fit_model | 1017.104 | 0.989617 |
| 4 | ml1m | cb_svdpp | fit_model | 744.180 | 0.950525 |
| 5 | ml1m | svdpp | fit_model | 426.552 | 0.986317 |

The next visible non-fit costs in the current ML1M reports are CB artifact
construction stages, for example `build_fit_artifacts` and
`build_cluster_artifacts` around 16 to 21 seconds for the CB models. They remain
secondary for the first kernel optimization phase because the measured
`fit_model` stages are larger by one to two orders of magnitude in the same run
set.

## Model Anatomy

### biased_mf

Current ML1M anatomy:

- `train_rows`: 800193
- `epochs`: 25
- `latent_dim`: 64
- `fit_seconds_total`: 16.146
- `estimated_factor_touches`: 5121235200
- `implicit_history_visits`: 0
- `explicit_history_visits`: 0
- `cluster_history_visits`: 0

Cost hypothesis: this kernel is the clean base case. Its cost comes from rating
updates and factor-vector updates. The current kernel allocates per-rating
temporary arrays for old user and item vectors. Because there are no history
loops, it is a useful guardrail for exact workspace reuse and dtype/signature
checks.

First candidates:

- `EXACT`: allocate old-vector workspaces once per epoch call and reuse them for
  each rating update.
- `EXACT`: assert or normalize contiguous arrays and expected dtypes before
  entering the compiled kernel.
- `EXACT`: separate Numba cold-start timing from warm-run kernel timing in the
  benchmark protocol.

### svdpp

Current ML1M anatomy:

- `train_rows`: 800193
- `epochs`: 20
- `latent_dim`: 50
- `fit_seconds_total`: 426.523
- `estimated_factor_touches`: 502677358000
- `implicit_history_visits`: 4994765860
- `explicit_history_visits`: 0
- `cluster_history_visits`: 0

Cost hypothesis: implicit history traversal explains most structural work. The
kernel also allocates per-rating workspaces for old user factors, old item
factors, and the implicit context vector. History traversal happens once to build
the implicit context and again to update implicit factors.

First candidates:

- `EXACT`: reuse `user_vector_old`, `item_vector_old`, and implicit-context
  workspace arrays inside the epoch kernel.
- `EXACT`: replace per-rating `np.zeros` context allocation with a reused buffer
  that is explicitly reset before each rating.
- `EXACT`: harden contiguous array and dtype guarantees at the Python/kernel
  boundary.

Medium-risk candidates:

- `EXACT`: cache per-rating context state in an auxiliary buffer during the
  first history traversal and reuse it in the update traversal. This is exact
  only if rating order, history order, and update order remain unchanged, and it
  must be benchmarked against added memory traffic.

### asymmetric_svd

Current ML1M anatomy:

- `train_rows`: 800193
- `epochs`: 20
- `latent_dim`: 50
- `fit_seconds_total`: 1017.080
- `estimated_factor_touches`: 1002153944000
- `implicit_history_visits`: 4994765860
- `explicit_history_visits`: 4994765860
- `cluster_history_visits`: 0

Cost hypothesis: this kernel combines explicit and implicit history costs.
Per-rating allocation is present for the old item vector and the context vector.
Explicit residual weights are computed during context construction and again
during the explicit-factor update pass.

First candidates:

- `EXACT`: hoist and reuse `q_old` and context buffers.
- `EXACT`: reset reused context buffers explicitly instead of allocating them per
  rating.
- `EXACT`: enforce stable dtype and contiguity before compiled execution.

Medium-risk candidates:

- `EXACT`: preserve explicit residual weights from the context traversal for the
  update traversal. This can be exact only if it does not change the floating
  point order or the timing of factor updates. It also introduces variable-length
  workspace pressure.

### asvdpp

Current ML1M anatomy:

- `train_rows`: 800193
- `epochs`: 20
- `latent_dim`: 50
- `fit_seconds_total`: 1019.747
- `estimated_factor_touches`: 1002153944000
- `implicit_history_visits`: 4994765860
- `explicit_history_visits`: 4994765860
- `cluster_history_visits`: 0

Cost hypothesis: this kernel has the same explicit and implicit history volume
as `asymmetric_svd`, plus the user-factor update path. It allocates per-rating
workspaces for old user factors, old item factors, and context.

First candidates:

- `EXACT`: hoist and reuse `p_old`, `q_old`, and context buffers.
- `EXACT`: keep update order unchanged while resetting reused context buffers.
- `EXACT`: add dtype and contiguity guardrails before the Numba call.

Medium-risk candidates:

- `EXACT`: reuse explicit residual weights between the context and update
  traversals if the benchmark shows memory traffic does not offset saved work.

### cb_svdpp

Current ML1M anatomy:

- `train_rows`: 800193
- `epochs`: 20
- `latent_dim`: 64
- `fit_seconds_total`: 744.147
- `estimated_factor_touches`: 755689126400
- `implicit_history_visits`: 4994765860
- `explicit_history_visits`: 0
- `cluster_history_visits`: 877047720

Cost hypothesis: the kernel cost comes from implicit history traversal, cluster
history traversal, and mixed base/cluster factor updates. Unlike the non-CB
kernels, the main workspace arrays are already allocated outside the per-rating
loop.

First candidates:

- `EXACT`: verify contiguous array and dtype contracts for all cluster-history
  arrays.
- `EXACT`: remove redundant casts only after Numba signatures are locked and
  tested.
- `EXACT`: keep Numba cold-start timing separate from warm-run training timing.

Medium-risk candidates:

- `EXACT`: cache active cluster ids and normalized cluster weights for reuse
  between cluster-context and cluster-update passes. This needs memory and
  ordering checks before it can be accepted as exact.

### cb_asvdpp

Current ML1M anatomy:

- `train_rows`: 800193
- `epochs`: 20
- `latent_dim`: 64
- `fit_seconds_total`: 1958.626
- `estimated_factor_touches`: 1395019156480
- `implicit_history_visits`: 4994765860
- `explicit_history_visits`: 4994765860
- `cluster_history_visits`: 877047720

Cost hypothesis: this is the largest current `fit_model` stage because it
combines explicit feedback, implicit feedback, and cluster-history update paths.
The kernel already hoists its main workspace arrays, so the first exact
optimization is less obvious than for non-CB kernels.

First candidates:

- `EXACT`: verify dtype and contiguity guarantees across rating, explicit,
  implicit, and cluster-history arrays.
- `EXACT`: remove redundant casts only after typed signatures and regression
  tests prove the same behavior.

Medium-risk candidates:

- `EXACT`: cache explicit residual weights and cluster lookup state across
  repeated traversals. This must preserve all update order and factor read/write
  timing.
- `EXACT_BUT_ORDER_SENSITIVE`: re-layout cluster-history traversal for more
  contiguous access if it changes only traversal mechanics but not intended
  mathematical order. Any changed accumulation order needs numerical acceptance
  checks.

## Cost Hypotheses Per Kernel

| Kernel | Primary structural cost | Current allocation signal | First priority |
| --- | --- | --- | --- |
| `train_biased_mf_epoch_numba` | Base rating updates | Per-rating old user/item vectors | Reuse workspaces |
| `train_svdpp_epoch_numba` | Implicit history context and updates | Per-rating old vectors and context | Reuse workspaces |
| `train_asymmetric_svd_epoch_numba` | Explicit plus implicit history traversal | Per-rating old item vector and context | Reuse workspaces |
| `train_asvdpp_epoch_numba` | Explicit plus implicit history traversal and user-factor updates | Per-rating old user/item vectors and context | Reuse workspaces |
| `train_cb_svdpp_epoch_numba` | Implicit plus cluster-history traversal | Main workspaces already hoisted | Dtype/contiguity/signature checks |
| `train_cb_asvdpp_epoch_numba` | Explicit, implicit, and cluster-history traversal | Main workspaces already hoisted | Dtype/contiguity/signature checks |

## Exact Low-Risk Optimization Candidates

These candidates are allowed for the first optimization phase if they pass the
benchmark contract below.

### 1. Hoist per-rating temporary workspaces in non-CB kernels

Classification: `EXACT`

Applicable kernels:

- `train_biased_mf_epoch_numba`
- `train_svdpp_epoch_numba`
- `train_asymmetric_svd_epoch_numba`
- `train_asvdpp_epoch_numba`

Plan:

- Allocate temporary old-vector and context buffers once per epoch-kernel call.
- Reuse them for each rating.
- Reset context buffers explicitly before accumulation.
- Preserve rating order, history order, update order, and factor read/write
  timing.

Reason this is low risk: it changes temporary storage lifetime only. It should
not change the mathematical formula or update order. Any result delta beyond the
defined numerical tolerance blocks the change.

### 2. Harden contiguous array guarantees before kernel calls

Classification: `EXACT`

Plan:

- Ensure arrays passed into Numba kernels are contiguous where kernels assume
  dense contiguous memory.
- Prefer explicit boundary checks or conversions in the model fit setup over
  implicit behavior inside hot loops.
- Record any conversion in code comments or tests if it can allocate.

Reason this is low risk: it does not change model behavior if dtype and values
are unchanged. It can change memory behavior, so benchmark memory must be
recorded.

### 3. Harden dtype consistency

Classification: `EXACT`

Plan:

- Keep rating/user/item id arrays on the documented integer dtype.
- Keep factor and rating arrays on the configured floating dtype.
- Add tests around kernel input preparation so dtype drift is caught before a
  benchmark run.

Reason this is low risk: it makes existing assumptions explicit. It must not
silently switch configured precision.

### 4. Separate Numba cold-start from warm-run timing

Classification: `EXACT`

Plan:

- Add a benchmark convention that records compile/cold-start cost separately
  from warm training cost.
- Do not compare an optimized warm run against a baseline cold run.

Reason this is low risk: it is measurement hygiene only.

### 5. Remove unnecessary casts after signature lock

Classification: `EXACT`

Plan:

- First lock or inspect the compiled signatures for the kernel inputs.
- Remove casts inside hot loops only where input dtypes are guaranteed.
- Keep regression tests for numerical metrics and kernel profile production.

Reason this is lower priority: casts are visible in the code, but changing them
without stable signatures can accidentally alter inferred Numba behavior.

### 6. Measure prediction and evaluation overhead

Classification: `EXACT`

Plan:

- Keep prediction and metric stages in Performance Forensics.
- Do not optimize them before fit kernels unless a new profile shows they are a
  primary bottleneck for a target workload.

Reason this is lower priority: current ML1M profiles show prediction/evaluation
costs are much smaller than `fit_model` for the kernel-heavy models.

## Medium-Risk Optimization Candidates

These are not first-phase candidates unless the exact low-risk work is complete
and the benchmark contract is in place.

### Cache repeated explicit residual weights

Classification: `EXACT` if update order and factor read timing are preserved.

Applicable kernels:

- `train_asymmetric_svd_epoch_numba`
- `train_asvdpp_epoch_numba`
- `train_cb_asvdpp_epoch_numba`

Risk: residuals are tied to current factor values and the current rating update.
Caching is exact only within the same rating update and only if the cached values
are produced at the same point where the existing code reads them.

### Cache cluster lookup state inside a rating update

Classification: `EXACT` if update order and cluster-weight semantics are
preserved.

Applicable kernels:

- `train_cb_svdpp_epoch_numba`
- `train_cb_asvdpp_epoch_numba`

Risk: this can reduce repeated lookup work, but it adds variable workspace and
memory traffic. It must not change cluster weighting, alpha mixing, or update
timing.

### Re-layout history or cluster-history access

Classification: `EXACT` or `EXACT_BUT_ORDER_SENSITIVE`, depending on whether
the traversal order is unchanged.

Risk: changing memory layout can be exact if the logical order stays identical.
If accumulation order changes, floating point differences are expected and must
stay within the acceptance tolerance.

### Introduce explicit Numba signatures

Classification: `EXACT`

Risk: signatures can stabilize compilation and remove inference ambiguity, but
they also make dtype contracts stricter. This should happen only after input
preparation tests cover all six models.

## Approximation-Risk Candidates

These are not allowed in the first optimization phase.

### Truncate or sample histories

Classification: `APPROXIMATION`

This changes the training signal and the objective behavior.

### Use stale or precomputed contexts across rating updates

Classification: `APPROXIMATION`

Contexts depend on factor values that change during SGD. Reusing stale contexts
changes update semantics.

### Parallelize SGD updates without preserving serial order

Classification: `APPROXIMATION`

Parallel writes can change update timing and floating point order. This is not
an exact optimization of the current training semantics.

### Reorder ratings or histories for locality

Classification: `EXACT_BUT_ORDER_SENSITIVE` at best, often `APPROXIMATION`

If SGD rating order changes, model behavior changes. If only inner accumulation
order changes, numerical differences must be tested.

## Research-Change Candidates

These are outside the scope of Kernel Optimization V1.

- New loss functions or regularization terms.
- New clustering definitions.
- New negative sampling or history weighting schemes.
- New training schedules.
- New model variants that change prediction formulas.

Classification: `RESEARCH_CHANGE`

## No-Go Changes

- Do not change model formulas.
- Do not change update timing or SGD order for an exact optimization.
- Do not change hyperparameters, default profiles, split policy, cache policy,
  seeds, or dtype as part of an optimization patch.
- Do not train clusters on validation or test data.
- Do not remove Performance Forensics or Kernel Cost Anatomy artifacts.
- Do not claim a runtime or memory improvement without before/after artifacts.
- Do not compare runs that differ in dataset, split, config, seed, dtype, device
  profile, cache policy, or number of epochs.

## Low-Risk Candidate Checklist

| Candidate | Current assessment | First phase status |
| --- | --- | --- |
| Per-rating `np.empty` / `np.zeros` allocations | Present in non-CB kernels | Allowed as `EXACT` |
| Workspace arrays outside inner loops | Already present in CB kernels, missing in non-CB kernels | Allowed as `EXACT` |
| Contiguous array guarantees | Should be made explicit at boundaries | Allowed as `EXACT` |
| Dtype consistency: `float32` / `int32` | Should be guarded before kernels | Allowed as `EXACT` |
| Unnecessary casts inside hot loops | Candidate after signature/dtype lock | Allowed later as `EXACT` |
| Numba signature stability | Needed before cast cleanup | Measurement and test prerequisite |
| Numba cold-start vs warm-run separation | Required for valid before/after comparison | Benchmark prerequisite |
| Duplicated history traversal | Structurally visible, but exact caching is harder | Medium risk |
| Prediction/evaluation overhead | Visible but not top priority in current ML1M reports | Defer |

## Benchmark Protocol For Before/After Comparison

### Required Dataset And Model Matrix

Minimum benchmark matrix:

- `ml100k`: all six models
- `ml1m`: all six models

Optional larger benchmark matrix:

- `ml10m`: `biased_mf`, `svdpp`, `cb_svdpp`, `cb_asvdpp`

### Required Before/After Invariants

Every before/after pair must use:

- same dataset
- same split
- same model config
- same model seed
- same dtype
- same device profile
- same cache policy
- same number of epochs
- same code path for artifact collection

In a dirty workspace, benchmark reuse is not valid unless the exact code and
configuration match can be proven from run metadata.

### Required Metrics

Each comparison must report:

- `test_rmse`
- `test_mae`
- `train_time_total`
- `fit_model_seconds`
- `ratings_per_second_train`
- `estimated_factor_touches`
- `fit_seconds_per_million_estimated_factor_touches`
- `peak_memory_mb`

The comparison must keep the raw run artifacts:

- `metrics.json`
- `run_manifest.json`
- `config_snapshot.yaml`
- `performance_profile.json`
- `kernel_profile.json`

The report collectors should then regenerate:

- `artifacts/reports/performance_stage_breakdown.csv`
- `artifacts/reports/performance_hotspots.csv`
- `artifacts/reports/kernel_cost_anatomy.csv`

### Cold-Start And Warm-Run Rule

Numba compile/cold-start cost must not be mixed into warm-run kernel comparison.
If cold-start behavior is relevant, record it separately. The acceptance
comparison for kernel optimization should use warm runs on both baseline and
candidate code.

## Acceptance Criteria

For `EXACT` changes:

- No `test_rmse` or `test_mae` regression beyond the predeclared tiny numerical
  tolerance. A practical starting tolerance is absolute delta `<= 1e-7` for
  aggregate metrics unless an existing model test defines a stricter contract.
- No changed run manifest semantics.
- No missing `performance_profile.json` or `kernel_profile.json`.
- Runtime change must be measured, not assumed.
- Memory change must be measured, not assumed.
- Any changed output metric beyond tolerance blocks the patch until explained.

For `EXACT_BUT_ORDER_SENSITIVE` changes:

- The intended mathematical operation must be unchanged.
- Floating point order differences must be documented.
- The same before/after matrix is required.
- Numerical deltas must remain within the accepted tolerance before the change
  can be considered for merge.

For `APPROXIMATION` and `RESEARCH_CHANGE` candidates:

- They are excluded from the first optimization phase.
- They require separate methodology documentation and evaluation protocol before
  implementation.

## First Recommended Optimization

Start with `EXACT` workspace reuse in the non-CB Numba kernels, with
`train_svdpp_epoch_numba` as the first history-bearing target and
`train_biased_mf_epoch_numba` as the smallest guardrail case if a minimal pilot
is useful.

Why this is the first recommendation:

- It addresses a concrete code-level allocation pattern visible in
  `src/recsys_lab/models/kernels.py`.
- It does not require formula changes, rating reordering, history truncation, or
  changed update timing.
- It applies to multiple kernels that are prominent in the ML1M fit-stage data.
- The CB kernels already have this pattern partially implemented, so starting
  with non-CB kernels avoids the more subtle cluster-history and residual-cache
  risks.

The first implementation task should include only one small kernel slice, a
numerical regression check, full profile artifact generation, and the benchmark
matrix required above before any broader rollout.
