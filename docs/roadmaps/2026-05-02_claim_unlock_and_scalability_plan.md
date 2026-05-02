# Claim Unlock And Scalability Plan

- date: `2026-05-02`
- governed_by: `docs/project_master_plan.md`
- scope: `cb_svdpp`, `cb_asvdpp`, large-dataset readiness, tuning discipline
- status: `active_work_queue`

## Purpose

This document converts the current no-claims into an executable remediation
plan. It does not unlock any claim by itself. A claim is unlocked only by the
gates defined below and by the normal evidence rules in
`docs/evaluation_protocol.md`, `docs/manifest_contract.md`, and
`docs/publish_readiness_matrix.md`.

The current repo state is useful, but it is not yet a production-grade,
scalable, or exact paper-faithful CB reproduction. The implemented CB models are
documented source-grounded predictors with repo-defined optimization under
`D-004`. The plan below keeps that boundary explicit while defining the work
needed to make the implementation faster, more reusable, and more defensible.

## Keep

The following parts are worth retaining and extending:

| Area | Keep | Why |
| --- | --- | --- |
| Governance | Project master plan, evidence notes, readiness matrix, manifest contracts | They prevent unsupported claims and dirty benchmark reuse. |
| Dataset scope | `ml100k`, `ml1m`, `ml10m`, `ml20m` | Scope is already frozen; failures become evidence, not silent removals. |
| Split discipline | Train/validation/test separation and train-only clustering | This is mandatory to avoid leakage. |
| Model ladder | `biased_mf`, `svdpp`, `asymmetric_svd`, `asvdpp`, `cb_svdpp`, `cb_asvdpp` | It gives controlled baselines and reductions. |
| Mathematical boundary | `D-004` and CB math specs | They correctly block exact paper-faithful claims where the source is underspecified. |
| Numba kernels | Existing compiled training kernels | They are a valid baseline for hot-path optimization. |
| Training index cache | Precomputed user/item histories keyed by split and config fingerprints | This is the right direction for repeated large runs. |
| Inference caches | User-context and mixed item-factor caching in CB predictors | They reduce repeated inference work and should be preserved. |
| Negative evidence | `ml20m cb_svdpp` guardrail breach | This is important evidence, not a failure to hide. |

## Change

The following parts must change before stronger claims are allowed:

| Area | Required change | Claim impact |
| --- | --- | --- |
| Runtime language | Stop describing the CB path as scalable or production-ready without benchmarks. | Keeps `scalable`, `faster`, and `ready` locked. |
| Device profile | Replace draft HPC placeholders with concrete CPU/RAM/thread/cache profiles before large final CB runs. | Required before any stronger large-dataset campaign. |
| Hot path | Profile and optimize CB training around actual per-stage timings, not guesses. | Required before any speed or scalability claim. |
| Memory guardrails | Treat the local 24 GB profile as insufficient for final `ml20m cb_svdpp` promotion under the current profile. | Keeps `ml20m` model-comparison claims blocked locally. |
| Cluster cache | Persist train-only cluster assignments and cluster-history indexes with leakage-safe fingerprints. | Required for efficient tuning campaigns. |
| KMeans layer | Add explicit `MiniBatchKMeans` support only as a measured candidate, not as a silent replacement. | Avoids hidden method changes. |
| Hyperparameter tuning | Promote `alpha`, cluster counts, learning rates, regularization, epochs, and early stopping into documented validation-only campaigns. | Required before tuned CB quality claims. |
| `R_star` usage | Keep `R_star` diagnostic unless a new math spec, objective, updates, tests, and ablation are added. | Blocks paper-faithful or `R_star`-optimized claims. |
| CLI ergonomics | Expose cache flags and tuning controls consistently across generic tuning commands. | Needed for repeatable large tuning. |
| Early stopping | Add validation-only early stopping and best-checkpoint restore as an explicit config option. | May reduce wasted epochs, but no quality-preserving speed claim without benchmark. |

## Implement

### 1. Concrete HPC And Runtime Contract

Add concrete non-draft runtime profiles before any further large final CB
campaign:

- CPU model, physical cores, logical threads, RAM, storage class, and cache path
- BLAS/OpenMP/threadpool settings
- expected RAM guardrail
- dtype and precision profile
- manifest readout showing the effective profile
- preflight check that rejects null HPC profile fields for claim-eligible runs

Acceptance gate:

- status: `implemented_g1_preflight`
- evidence: `docs/evidence/reproduction/2026-05-02_runtime_profile_contract_g1.md`
- a focused test fails if a claim-eligible HPC profile still contains null
  thread or RAM fields
- `validate-runtime-profile --claim-eligible` rejects the draft `hpc_cpu`
  template and accepts the concrete local reference profile
- run manifests include `runtime.device_profile_contract`

### 2. Stage-Level CB Profiler

Add a dedicated profiler for `cb_svdpp` and `cb_asvdpp` runs. It must measure at
least:

- data load and split resolution
- split-cache lookup or creation
- training-index cache lookup or creation
- induction model training
- KMeans or MiniBatchKMeans
- `R_star` diagnostic build
- user-cluster-history build
- each main training epoch
- validation and test inference
- peak memory for each major stage where feasible

Acceptance gate:

- status: `implemented_g2_instrumentation_ml100k_and_ml10m_profile`
- evidence: `docs/evidence/reproduction/2026-05-02_cb_stage_profile_g2.md`
- large-dataset evidence: `docs/evidence/reproduction/2026-05-02_ml10m_cb_svdpp_large_stage_profile_g2.md`
- one synthetic integration test verifies that stage timings are written
- one `ml100k` evidence note records the profiler output
- one bounded `ml10m` evidence note records the large-dataset profiler output
- no speed claim is made until a before/after benchmark exists

### 3. CB Training Hot-Path Remediation

The current CB kernels repeatedly construct per-rating temporary arrays for user
context, item context, and cluster context. This is a plausible bottleneck, but
it remains a hypothesis until measured.

Remediation candidates:

- remove per-rating allocations in Numba kernels
- reuse fixed-size work buffers where Numba semantics allow it
- precompute repeated normalizers and pointer ranges
- avoid redundant history scans where exact SGD semantics are preserved
- add kernel-level counters or microbenchmarks for allocation-sensitive paths

Acceptance gate:

- toy deterministic equivalence test against the current kernel
- `ml100k` before/after benchmark with same config, split, seed, dtype, and
  clean git state
- metric drift bound documented before running the benchmark
- no `faster` claim unless runtime improves in the measured benchmark

Status:

- status: `implemented_g3_cb_svdpp_workbuffer_ml100k`
- evidence:
  `docs/evidence/reproduction/2026-05-02_cb_svdpp_hotpath_g3.md`
- accepted change: reuse fixed-size `cb_svdpp` Numba work buffers while
  preserving old-value vector semantics
- rejected change: scalar re-read candidate, because the clean `ml100k` Stage1
  benchmark measured slower `main_training`
- measured boundary: only the accepted `ml100k cb_svdpp` Stage1 context may be
  described as faster; no `ml10m`, `ml20m`, `cb_asvdpp`, scalability,
  production-readiness, quality, SOTA, or paper-faithfulness claim is unlocked

### 4. Algorithmic Acceleration Track

The dominant CB-SVD++ cost is structurally tied to scanning user histories and
cluster histories during rating updates. Micro-optimization alone may not be
enough for `ml20m`.

Candidate tracks:

- per-user or blocked training order experiments
- cached context vectors with exact or explicitly approximate update semantics
- delayed context refresh with a new method label if exact equivalence is lost
- sparse-aware cluster-history storage
- optional lower-memory profile with fewer clusters or smaller latent dimension,
  selected only through validation

Acceptance gate:

- every exact track needs equivalence tests
- every approximate track needs a new model variant name and math note
- no approximate result may be mixed into existing `cb_svdpp` claims

### 5. Leakage-Safe Cluster Artifact Cache

Persist train-only clustering artifacts under `artifacts/local` or an equivalent
ignored cache location:

- induction latent fingerprint
- induction config fingerprint
- split fingerprint
- clustering algorithm and seed
- `n_user_clusters`, `n_item_clusters`, `alpha` where relevant
- user and item assignments
- cluster means/counts and `R_star` diagnostics
- user-cluster-history index

Acceptance gate:

- cache is invalidated when split, induction config, cluster config, dtype, or
  source data changes
- tests prove validation/test ratings do not affect the cached artifacts
- manifests report cache hit/miss and cache fingerprint

Status:

- status: `implemented_g4_cluster_artifact_cache`
- evidence:
  `docs/evidence/reproduction/2026-05-02_cluster_artifact_cache_g4.md`
- `cb_svdpp` and `cb_asvdpp` run manifests expose top-level `caches`
- `--cluster-artifact-cache/--disable-cluster-artifact-cache` is available for
  both CB experiment CLIs
- no speed, scalability, quality, large-dataset, or paper-faithfulness claim is
  unlocked by this cache gate

### 6. Methodical Hyperparameter Tuning

The following parameters must be handled by validation-only campaigns before
quality claims can be strengthened:

- `alpha`
- user cluster count
- item cluster count
- latent dimension
- learning rate
- user/item/feedback/cluster regularization
- induction epochs and induction regularization
- main training epochs or early-stopping patience
- `R_star` coefficient only if a new `R_star` objective is implemented

Initial search axes should be budgeted, not exhaustive:

- `alpha`: `0.0`, `0.025`, `0.05`, `0.10`, `0.15`, `0.25`
- cluster counts: `32`, `64`, `80`, `100`, `128`
- latent dimensions: `32`, `64`
- early stopping: disabled baseline, then patience-based candidate

Acceptance gate:

- all tuning uses validation only
- test data is evaluated only after the selection is frozen
- candidate manifests include dataset, split family, split seeds, model seeds,
  full config fingerprints, cache policy, and device profile
- reduced-budget tuning is labelled selection evidence, not final benchmark

Status:

- status: `implemented_g5_bounded_validation_only_selection_probe`
- evidence:
  `docs/evidence/reproduction/2026-05-02_tune_inner_cache_controls_g5.md`
- `tune-inner` and `tune-ml100k-inner` expose explicit cache controls for
  split cache, supported training-index cache, and supported CB
  cluster-artifact cache
- unsupported cache-control/model combinations are rejected instead of silently
  ignored
- bounded `ml100k cb_svdpp` alpha/cluster-count selection ran with
  `test_metrics_available=false` and `test_rmse=None` for every candidate
- reduced-budget selection winner:
  `rank032_uc064_ic064_a000_lr0100_reg0020_e002`
- no final quality, large-dataset, scalability, production-readiness, SOTA, or
  paper-faithfulness claim is unlocked

Promotion contract:

- status: `contract_ready_g5_to_g6_validation_grid`
- evidence:
  `docs/evidence/reproduction/2026-05-03_cb_svdpp_g6_validation_grid_contract.md`
- config:
  `configs/experiments/tuning/ml100k_cb_svdpp_g6_validation_grid.yaml`
- decision: promote the bounded G5 winner into a larger validation-only
  `ml100k cb_svdpp` grid before any outer test rerun
- planned scope: `12` candidates times `3` split seeds, all selected by
  validation RMSE only
- no G6 tuning result, final quality claim, test-set claim, large-dataset
  claim, speed claim, scalability claim, SOTA claim, production-readiness claim,
  or paper-faithfulness claim is unlocked by this contract

### 7. `R_star` Decision Track

`R_star` currently remains diagnostic in the repo-defined CB v1 pipeline. That
is acceptable, but it must stay explicit.

Allowed paths:

- Path A: keep `R_star` diagnostic and continue calling the model
  source-grounded/repo-defined, not paper-faithful
- Path B: introduce a new explicitly named variant, for example
  `cb_svdpp_rstar_aux`, with math spec, objective, updates, implementation,
  tests, tuning, and ablation

Rejected path:

- silently feeding `R_star` into the objective or tuning loop while preserving
  the old model name

Acceptance gate:

- no `R_star` quality claim without Path B
- no paper-faithful claim while `D-004` remains unresolved

### 8. Release And Claim Gates

A stronger CB release requires all of these gates:

| Gate | Requirement |
| --- | --- |
| `G1` | Concrete runtime profile with no draft/null HPC fields for claim-eligible large runs. |
| `G2` | Stage-level profiler evidence on at least `ml100k` and one large dataset. |
| `G3` | Hot-path remediation has equivalence tests and before/after benchmarks. |
| `G4` | Cluster artifact cache is leakage-safe and manifest-visible. |
| `G5` | Hyperparameter tuning campaign is validation-only and manifest-backed. |
| `G6` | Final selected configs are rerun cleanly on outer benchmark seeds. |
| `G7` | `ml10m` and `ml20m` remain in scope, with either benchmark evidence or explicit negative evidence. |
| `G8` | README, report, readiness matrix, and evidence notes agree on allowed and blocked claims. |

## Do Not Do

- Do not tune on test data.
- Do not call current CB large-dataset behavior `scalable`.
- Do not rerun `ml20m cb_svdpp` locally under the same memory-risk profile for
  claim promotion.
- Do not replace KMeans with MiniBatchKMeans without a new config, manifest
  readout, and benchmark.
- Do not introduce an approximate acceleration under the existing exact model
  label.
- Do not claim `R_star` is optimized unless a new objective and update path
  exists.

## Immediate Next Sequence

1. Run the G6 `ml100k cb_svdpp` validation-only promotion grid only from a clean
   worktree with explicit split, training-index, and cluster-artifact cache
   controls.
2. Freeze the selected G6 candidate before any outer test rerun.
3. Keep `cb_asvdpp` hot-path remediation as separate work if profiling shows it
   is decision-critical.
4. Reassess `ml10m` and `ml20m` only after the profiler and cache work produce
   clean evidence.

## Current Progress Estimate

Approximate engineering readiness for a publishable, stronger CB-focused repo:
`80%`.

This percentage is not a benchmark result. It is a planning estimate based on
completed governance, data, model, benchmark scaffolding, runtime-profile
preflight, stage profiling on `ml100k` and bounded `ml10m`, and leakage-safe
cluster-cache plumbing, plus a measured `cb_svdpp` hot-path workbuffer
remediation on `ml100k`, plus explicit tune-inner cache controls and one
bounded validation-only `ml100k cb_svdpp` alpha/cluster selection probe, versus
the still missing larger validation grid decision, `cb_asvdpp` hot-path decision
if needed, final large-dataset reassessment, and final release-claim gates
above.
