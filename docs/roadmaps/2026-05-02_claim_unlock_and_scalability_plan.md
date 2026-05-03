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

- status: `completed_g6_validation_only_selection`
- evidence:
  `docs/evidence/reproduction/2026-05-03_cb_svdpp_g6_validation_grid_contract.md`
- run evidence:
  `docs/evidence/reproduction/2026-05-03_cb_svdpp_g6_validation_grid_run.md`
- config:
  `configs/experiments/tuning/ml100k_cb_svdpp_g6_validation_grid.yaml`
- decision: promote the bounded G5 winner into a larger validation-only
  `ml100k cb_svdpp` grid before any outer test rerun
- planned scope: `12` candidates times `3` split seeds, all selected by
  validation RMSE only
- selected candidate:
  `rank032_uc100_ic100_a0000_lr0100_reg0020_e002`
- frozen selected config:
  `configs/models/tuned/ml100k_cb_svdpp_g6_validation_selected.yaml`
- no final quality claim, test-set claim, large-dataset claim, speed claim,
  scalability claim, SOTA claim, production-readiness claim, or
  paper-faithfulness claim is unlocked by this selection evidence

Outer benchmark contract:

- status: `approved_for_clean_outer_benchmark_contract`
- evidence:
  `docs/evidence/reproduction/2026-05-03_cb_svdpp_g6_outer_benchmark_contract.md`
- decision: run one clean outer `ml100k cb_svdpp` benchmark for the frozen
  G6-selected config
- selected config:
  `configs/models/tuned/ml100k_cb_svdpp_g6_validation_selected.yaml`
- planned scope: split seeds `1,2,3`, model seed `1`, split family
  `benchmark_random_v1`
- required order: commit this contract first, then run the three outer runs
  and aggregate on the same clean commit
- no outer benchmark result, final quality claim, speed claim, scalability
  claim, SOTA claim, production-readiness claim, or paper-faithfulness claim is
  unlocked by this contract alone

Outer benchmark run:

- status: `completed_g6_clean_outer_benchmark`
- evidence:
  `docs/evidence/reproduction/2026-05-03_cb_svdpp_g6_outer_benchmark_run.md`
- benchmark artifact:
  `artifacts/benchmarks/2026-05-03T120906Z_ml100k_benchmark_random_v1_cb_svdpp_multiseed_s001_s002_s003_modelseed_s001_local_i5_2500k_24gb/benchmark_manifest.json`
- git commit: `67570ed8fde4c158848ab80494524ab203b40df5`
- git dirty: `false`
- split seeds: `1,2,3`
- model seed: `1`
- validation RMSE mean: `0.9566122815305916`
- validation RMSE std: `0.002372317660216579`
- test RMSE mean: `0.9595668222022953`
- test RMSE std: `0.011253690215412724`
- allowed claim: clean outer `ml100k cb_svdpp` benchmark readout for this
  frozen G6-selected profile under `benchmark_random_v1`
- no cross-split-family comparison, speed claim, scalability claim, SOTA claim,
  production-readiness claim, paper-faithfulness claim, or large-dataset claim
  is unlocked by this run

### 6.1. `cb_asvdpp` Hotpath Decision

Status:

- status: `pass_for_hotpath_prioritization_not_remediation`
- evidence:
  `docs/evidence/reproduction/2026-05-03_cb_asvdpp_hotpath_decision_g7.md`
- run artifact:
  `artifacts/runs/2026-05-03T121942Z_ml100k_cb_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/run_manifest.json`
- git commit: `2edc8a3be8f64c657df9519befc371d9e7accfd3`
- git dirty: `false`
- profile: `ml100k cb_asvdpp`, `benchmark_random_v1`, split seed `1`,
  model seed `1`
- `main_training`: `115.10358980001183` seconds
- total profiled wall-clock: `124.51294220011914` seconds
- `main_training` share: about `92.44%`
- decision: `cb_asvdpp` hot-path remediation is justified as a separate work
  item, but no code change may be made before a remediation contract defines
  equivalence tests, metric-drift bounds, and a clean before/after benchmark
- no speed, scalability, production-readiness, SOTA, paper-faithfulness,
  large-dataset, or quality claim is unlocked by this profiling decision

Remediation contract:

- status: `approved_for_exact_remediation_contract`
- evidence:
  `docs/evidence/reproduction/2026-05-03_cb_asvdpp_hotpath_remediation_contract_g8.md`
- target:
  `src/recsys_lab/models/kernels.py::train_cb_asvdpp_epoch_numba`
- approved first target: fixed-size work-buffer reuse inside the exact Numba
  epoch kernel
- exactness gate: deterministic toy comparison against current Python fallback
  semantics with tolerance `1e-6` absolute and relative on all trainable arrays
- benchmark gate: fresh clean pre-change baseline on the committed contract,
  then same-command post-change benchmark after remediation
- acceptance rule: at least `1.0%` lower `main_training_wall_clock_seconds`
  with `train_rmse`, `validation_rmse`, and `test_rmse` absolute drift at most
  `1e-6`
- no code change, speed claim, scalability claim, production-readiness claim,
  SOTA claim, paper-faithfulness claim, large-dataset claim, or quality claim is
  unlocked by this contract alone

Pre-change baseline:

- status: `pass_for_clean_prechange_baseline`
- evidence:
  `docs/evidence/reproduction/2026-05-03_cb_asvdpp_hotpath_prechange_baseline_g9.md`
- run artifact:
  `artifacts/runs/2026-05-03T123549Z_ml100k_cb_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/run_manifest.json`
- git commit: `bc966e42f4fc2cf3d09c7f7194e17a81c93617cc`
- git dirty: `false`
- `main_training_wall_clock_seconds`: `122.91284980002092`
- train RMSE: `0.6848969434499206`
- validation RMSE: `0.9134162708331054`
- test RMSE: `0.9102128098774724`
- cache status: cluster artifact cache `hit`, user-cluster history cache `hit`
- no code change, speed claim, scalability claim, production-readiness claim,
  SOTA claim, paper-faithfulness claim, large-dataset claim, or quality claim is
  unlocked by this baseline alone

Post-change benchmark:

- status: `pass_for_exact_workbuffer_remediation_context`
- evidence:
  `docs/evidence/reproduction/2026-05-03_cb_asvdpp_hotpath_postchange_benchmark_g10.md`
- run artifact:
  `artifacts/runs/2026-05-03T124801Z_ml100k_cb_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/run_manifest.json`
- git commit: `e6b77c7f9bc5a87259a5e18e618dc18941a3a9e3`
- git dirty: `false`
- `main_training_wall_clock_seconds`: `113.81238390004728`
- required post-change threshold: `121.68372130202071`
- observed main-training wall-clock change: `-7.403998780257792%`
- train RMSE drift: `0.0`
- validation RMSE drift: `0.0`
- test RMSE drift: `0.0`
- allowed claim: only this exact `ml100k cb_asvdpp` benchmark context ran
  faster in the `main_training` stage after exact work-buffer remediation
- no broad speed claim, scalability claim, production-readiness claim, SOTA
  claim, paper-faithfulness claim, large-dataset claim, or cross-device claim is
  unlocked

### 6.2. `ml20m cb_svdpp` Lower-Memory Reassessment Contract

Status:

- status: `contract_ready_g11_ml20m_lower_memory_validation_reassessment`
- evidence:
  `docs/evidence/reproduction/2026-05-03_ml20m_cb_svdpp_g11_lower_memory_validation_contract.md`
- config:
  `configs/experiments/tuning/ml20m_cb_svdpp_g11_lower_memory_validation_grid.yaml`
- purpose: define the next allowed local `ml20m cb_svdpp` reassessment after
  the previous full `stage0_transfer` campaign breached the local RAM guardrail
- planned scope: `8` lower-memory candidates times split seeds `1,2`, all
  validation-only
- resource gate: reject any candidate with a run above `19660.8 MB` peak memory
  on `local_i5_2500k_24gb`
- selection rule: select by `validation_rmse_mean` only after all candidate
  runs pass the resource gate
- no test-set evaluation, final `ml20m cb_svdpp` benchmark claim, model
  comparison claim, speed claim, scalability claim, production-readiness claim,
  SOTA claim, or paper-faithfulness claim is unlocked by this contract

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

1. Keep the G10 `cb_asvdpp` speed claim limited to its exact benchmark context.
2. Run the G11 `ml20m cb_svdpp` lower-memory validation grid only from a clean
   committed state, with explicit cache controls and resource-gate readout.
3. If G11 passes, freeze one selected config and write a separate outer
   benchmark contract before any test-set or comparison claim.
4. If G11 breaches the RAM guardrail, document negative resource evidence and
   keep final `ml20m cb_svdpp` claims blocked locally.

## Current Progress Estimate

Approximate engineering readiness for a publishable, stronger CB-focused repo:
`95%`.

This percentage is not a benchmark result. It is a planning estimate based on
completed governance, data, model, benchmark scaffolding, runtime-profile
preflight, stage profiling on `ml100k` and bounded `ml10m`, leakage-safe
cluster-cache plumbing, a measured `cb_svdpp` hot-path workbuffer remediation
on `ml100k`, explicit tune-inner cache controls, a bounded validation-only
`ml100k cb_svdpp` alpha/cluster selection probe, and a completed G6
validation-only `ml100k cb_svdpp` grid, a documented clean outer benchmark
contract for the frozen G6 selection, and a completed clean outer benchmark
readout for that frozen profile, and a `cb_asvdpp` profiling decision,
exact-remediation contract, clean pre-change baseline, exact work-buffer
implementation, equivalence guard, and clean post-change benchmark that passed
the G8 metric-drift and wall-clock gates in the exact `ml100k cb_asvdpp`
context, plus a G11 lower-memory validation contract for `ml20m cb_svdpp`,
versus the still missing G11 execution, final large-dataset reassessment, and
final release-claim gates above.
