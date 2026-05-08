# SOTA Tuning Framework Architecture V1

## Executive Summary

Tuning must not be a single grid script. The target architecture is a framework
made of explicit search spaces, candidate generation, candidate
materialization, run execution, evidence capture, artifact reuse, optional
pruning, and multi-objective selection.

Step 19a defines the architecture and contracts only. It does not implement a
new optimizer, does not run large tuning studies, does not change model
semantics, and does not change KMeans or kernel behavior.

The framework should wrap the existing `run_unified_experiment(...)` execution
primitive instead of replacing it. The new layer should make studies
cache-aware before execution so CB tuning can vary target-model parameters
without rebuilding split, history, Cluster-Artifact, or user-cluster-history
artifacts unnecessarily.

## Goals

- Represent tuning studies as first-class artifacts, not ad-hoc script runs.
- Separate search-space definition from generated candidates.
- Materialize every candidate into an effective model config snapshot.
- Execute trials through the unified runner and preserve existing manifests.
- Group candidates by expensive artifact identities before scheduling.
- Record cache policies, cache statuses, and artifact reuse groups.
- Support validation-only inner tuning without test-set leakage.
- Support resource gates and failure states as explicit trial outcomes.
- Prepare for later multi-fidelity pruning without changing model semantics.
- Prepare for later multi-objective and Pareto selection.
- Preserve claim boundaries: tuning selects candidates; outer benchmarks make
  benchmark claims.

## Non-goals

- No full tuning engine implementation in Step 19a.
- No Optuna, TPE, Sobol, Successive Halving, or pruning dependency is added.
- No large ML1M, ML10M, ML20M, or Netflix-scale tuning run is started.
- No model formula, kernel, KMeans algorithm, KMeans default, or induction
  default is changed.
- No migration of all existing tuning configs is forced.
- No historical artifact is overwritten.
- No performance or quality claim is introduced.
- No compatibility bridge or new monolithic script is created.

## Current Tuning Inventory

Inventory command:

```powershell
rg "tuning|grid|candidate|alpha|search_space|study|trial|sweep|optuna|halving|pruning" src scripts configs tests docs
```

### Active Tuning Configs

| Config | Dataset | Model | Stage | Units | Candidates | Base model | Notes |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| `configs/experiments/tuning/active/ml100k_cb_svdpp_g6_validation_grid.yaml` | `ml100k` | `cb_svdpp` | `g6_validation_grid` | split seeds `1,2,3` | 12 | `configs/models/cb_svdpp.yaml` | Validation-only grid around a previous bounded-selection winner. Varies cluster counts and `alpha`; includes `alpha=0` ablation candidates. |
| `configs/experiments/tuning/active/ml1m_cb_svdpp_stage0.yaml` | `ml1m` | `cb_svdpp` | `stage0` | split seeds `1,2` | 3 | `configs/models/selected/ml1m/ml1m_cb_svdpp_stage0_transfer.yaml` | Small draft Stage0 grid. Varies `alpha` and cluster counts with a fixed two-epoch budget. |
| `configs/experiments/tuning/active/ml1m_cb_asvdpp_stage0.yaml` | `ml1m` | `cb_asvdpp` | `stage0` | split seeds `1,2` | 3 | `configs/models/selected/ml1m/ml1m_cb_asvdpp_stage0_transfer.yaml` | Small draft Stage0 grid. Varies `alpha` and cluster counts with a fixed two-epoch budget. |
| `configs/experiments/tuning/active/ml20m_cb_svdpp_g11_lower_memory_validation_grid.yaml` | `ml20m` | `cb_svdpp` | `g11_lower_memory_validation_grid` | split seeds `1,2` | 8 | `configs/models/cb_svdpp.yaml` | Resource-gated lower-memory validation grid. Varies latent dim, cluster counts, and `alpha`; includes explicit memory guardrail metadata. |

### Archived And Template Configs

Archived configs under `configs/experiments/tuning/archive/`:

- `ml100k_biased_mf_stage1.yaml`
- `ml100k_svdpp_stage1.yaml`
- `ml100k_cb_svdpp_stage1.yaml`
- `ml100k_cb_svdpp_stage2.yaml`
- `ml100k_cb_svdpp_g5_bounded_alpha_cluster.yaml`
- `ml100k_cb_asvdpp_stage1.yaml`

Templates under `configs/experiments/tuning/templates/`:

- `cb_svdpp_grid_template.yaml`
- `cb_asvdpp_grid_template.yaml`

These are useful inventory and evidence inputs. They should not be deleted in
Step 19a, but they should not become the future framework contract.

### Existing Entry Points And Outputs

CLI commands:

- `recsys-lab tune-inner`
- `recsys-lab tune-ml100k-inner`

Script wrapper:

- `scripts/tune_ml100k_inner.ps1`

Implementation:

- `src/recsys_lab/experiments/ml100k_inner_tuning.py`

Current output directory:

```text
artifacts/benchmarks/<timestamp>_<dataset>_inner_tuning_<model>_<stage>_<device>/
```

Current files:

- `benchmark_manifest.json`
- `config_snapshot.yaml`
- `summary.json`
- `summary.md`
- `stdout.log`
- `candidates/<candidate-config>.yaml`
- run manifests referenced from `benchmark_manifest.json`

The current outputs are useful but benchmark-shaped. They do not yet provide a
study-level manifest, candidate manifests, trial manifests, or an artifact
reuse plan.

### Monolithic Parts

`src/recsys_lab/experiments/ml100k_inner_tuning.py` currently combines:

- config loading and validation
- selection-unit resolution
- candidate config materialization
- unified-runner dispatch
- metric aggregation
- resource-gate enforcement
- candidate selection
- manifest writing
- summary rendering
- failure handling

This is workable for small hand-written grids. It is not a scalable
multi-objective or multi-fidelity tuning architecture.

### Reusable Parts

The future framework should reuse:

- `run_unified_experiment(...)` as the trial execution primitive
- model config validation through `validate_model_config_payload(...)`
- the current candidate override merge behavior
- split-unit resolution for `benchmark_random_v1` and
  `paper_faithful_ml100k_inner_v1`
- benchmark measurement metadata helpers
- manifest validation helpers
- timestamped artifact directory conventions
- existing split, training-index, explicit-feedback, Cluster-Artifact, and
  user-cluster-history cache systems
- resource-gate concepts and memory guardrail enforcement
- active-config validation tests

## Core Concepts

Study:
A reproducible tuning campaign with dataset, split policy, model family,
search-space contract, budget, seed policy, objective contract, resource
policy, cache policy, and claim boundary. A study is the unit that owns
candidate generation, trial execution, aggregation, and final selection.

Search space:
A declarative space of tunable dimensions and constraints. It is not a list of
run configs. It describes how candidates may be generated.

Candidate:
One concrete materialized model/run configuration inside a study. A candidate
has deterministic candidate id, effective model config, search-space
coordinates, budget assignment, and expected artifact reuse groups.

Trial:
One executed candidate evaluation at one fidelity level on one selection unit,
for example one split seed or one official fold. A trial has a status such as
`completed`, `failed`, `pruned`, or `skipped`, and it points to run artifacts
and metrics.

Fidelity:
One declared resource level for a trial, such as epochs, dataset tier,
latent-dim cap, split count, or sample fraction. The first MVP only documents
fidelity contracts and may use a single full-fidelity level; it does not
implement multi-fidelity execution.

Promotion:
The decision to move a candidate from a smaller fidelity level to a larger
fidelity level. Promotion must be based only on declared inner-validation
metrics and resource gates, never on test metrics.

Pruning:
The decision to stop a candidate early or not evaluate it at a higher fidelity.
Pruning is not a failure if it follows the study pruning policy. Every pruned
candidate must retain evidence showing the fidelity level, metrics, and rule
that caused pruning.

Selection unit:
The inner-validation unit over which a candidate is evaluated. Current units
are benchmark-random split seeds and ML100K official inner folds.

Artifact identity:
The cache-relevant identity of expensive artifacts such as splits, history
indices, explicit-feedback indices, Cluster-Artifacts, and user-cluster-history.

Artifact reuse group:
A group of candidates and trials that may share expensive artifacts because
their artifact identities match. Example: CB candidates with the same train
split, same `n_user_clusters`, same `n_item_clusters`, and same induction
config may share Cluster-Artifacts even when target-model `alpha`, lambdas, or
epochs differ.

Objective:
The declared evaluation function used for selection. The MVP can use primary
validation RMSE with tie-breakers, but the architecture must allow objectives
that combine or compare quality, cost, resource use, and stability.

Pareto candidate:
A candidate that is not dominated on the declared quality/cost/resource
dimensions. For example, another candidate must not be better or equal on all
objectives and strictly better on at least one objective. Pareto status is an
analysis output, not automatically a claim.

Study result:
The aggregate objective and resource summary over all completed trials, plus
selection decision, tie-break reasoning, and claim boundary.

## Tuning Lifecycle

1. Load study config.
2. Validate dataset, split, model family, objective, cache, and resource policy.
3. Expand or sample the search space into candidate descriptors.
4. Materialize effective model configs for candidates.
5. Compute candidate hashes and expected artifact identity groups.
6. Write `study_manifest.json`, candidate manifests, and artifact-reuse plan.
7. Schedule candidates in cache-aware order.
8. Execute trials through `run_unified_experiment(...)`.
9. Write one `trial_manifest.json` per candidate and selection unit.
10. Update study progress and resource-gate status.
11. Optionally prune candidates only at declared fidelity boundaries.
12. Aggregate objectives and resource metrics.
13. Compute Pareto and tie-break summaries where enabled.
14. Write `study_summary.json`, `study_summary.csv`, and human summary.
15. Mark the study completed, failed, or partially completed.

The framework should never use the test split for candidate selection.

## Search Space Contract

Search-space configs should be declarative and versioned:

```yaml
search_space_version: tuning_search_space_v1

study:
  name: ml1m_cb_svdpp_alpha_regularization_v1
  dataset: ml1m
  split_family: benchmark_random_v1
  model: cb_svdpp
  seed: 1

base_model_config: configs/models/selected/ml1m/ml1m_cb_svdpp_stage0_transfer.yaml

budget:
  max_candidates: 24
  max_parallel: 1
  max_wall_seconds: null

generator:
  type: grid
  deterministic_order: true

search_space:
  alpha:
    target_path: clustering.alpha
    type: float
    distribution: uniform
    low: 0.05
    high: 0.95
    constraints:
      - productive_cb_alpha

  learning_rate:
    target_path: training.learning_rate
    type: float
    distribution: loguniform
    low: 0.001
    high: 0.05

  lambda_p:
    target_path: training.lambda_p
    type: float
    distribution: loguniform
    low: 1.0e-4
    high: 1.0e-1

  lambda_q:
    target_path: training.lambda_q
    type: float
    distribution: loguniform
    low: 1.0e-4
    high: 1.0e-1

  latent_dim:
    target_path: training.latent_dim
    type: categorical
    values: [32, 64]

artifact_reuse:
  cluster_artifacts:
    reuse_across:
      - alpha
      - learning_rate
      - lambda_p
      - lambda_q
      - lambda_y
      - epochs
    invalidate_on:
      - n_user_clusters
      - n_item_clusters
      - induction_config
      - dataset
      - split
      - train_fingerprint

objective:
  primary:
    metric: validation_rmse
    aggregation: mean
    direction: minimize
  secondary:
    - metric: validation_mae
      aggregation: mean
      direction: minimize
    - metric: fit_model_seconds
      aggregation: mean
      direction: minimize
    - metric: peak_memory_mb
      aggregation: max
      direction: minimize
    - metric: cluster_total_seconds
      aggregation: mean
      direction: minimize
```

Required fields:

- `search_space_version`
- `study.name`
- `study.dataset`
- `study.split_family`
- `study.model`
- `study.seed`
- `base_model_config`
- `budget`
- generator type and deterministic ordering policy
- `search_space` dimensions with target paths, types, distributions, bounds,
  values, or constraints
- `artifact_reuse`
- `objective`

Supported MVP generator:

- deterministic grid over explicit categorical values

Future generators:

- random
- quasi-random
- Bayesian/TPE
- successive halving / Hyperband style schedulers

Dimension rules:

- `type: categorical` requires non-empty `values`.
- `type: int` requires integer `low` and `high`, or explicit `values`.
- `type: float` requires `distribution` and numeric `low` and `high`.
- `distribution: uniform` samples or enumerates on the original scale.
- `distribution: loguniform` samples or enumerates on the logarithmic scale and
  requires `low > 0`.
- Every dimension needs a `target_path` into the effective model config.
- Candidate generation must be deterministic for a fixed study config and
  generator seed.

CB endpoint alpha policy:

- Productive `cb_svdpp` and `cb_asvdpp` search spaces must constrain
  `clustering.alpha` to `0 < alpha < 1`.
- `alpha == 0` is an SVD++/ASVD++ baseline or CB-disabled ablation, not a
  normal productive CB candidate.
- `alpha == 1` is a research variant candidate, not a normal productive CB
  candidate.
- If a study intentionally includes endpoint alpha values, it must mark them as
  `ablation` or `research_variant` and must not mix them into productive CB
  candidate selection without an explicit policy decision.

The future schema validator should reject productive CB search spaces that
include endpoint alpha values unless the study declares a non-productive
endpoint policy.

## Candidate Contract

Each generated candidate needs a manifest with:

- `candidate_id`
- `candidate_index`
- `search_space_version`
- search-space coordinate map
- base model config ref
- effective model config ref
- effective model config SHA256
- model name and family
- dtype
- target training knobs
- cluster artifact identity knobs
- induction config identity
- expected artifact reuse group ids
- claim boundary

Candidate IDs must be deterministic for the study config and coordinate map.
They must not encode unbounded floating-point strings without normalization.

Candidate materialization must write the full effective model config before any
trial starts.

## Study Manifest Contract

`study_manifest.json` should be the top-level machine-readable contract.

Required fields:

- `manifest_version`
- `kind: tuning_study_manifest`
- `study_id`
- `study_status`
- `generated_at_utc`
- `git` snapshot
- runtime and device profile
- processed manifest ref
- dataset and split contract
- model family and base config ref
- search-space ref and hash
- candidate generation policy
- objective contract
- resource policy
- cache policy
- artifact reuse plan ref
- candidate manifest refs
- trial manifest refs
- claim boundary

The manifest should be immutable for completed studies except for status and
progress fields written through explicit updates.

## Study And Candidate Artifacts

Future tuning studies should write under:

```text
artifacts/tuning/<study_id>/
  study_manifest.json
  search_space.yaml
  artifact_reuse_plan.json
  candidates/
    <candidate_id>/
      candidate_config.yaml
      candidate_manifest.json
      runs/
        <run_id>/
          metrics.json
          performance_profile.json
          kernel_profile.json
          run_manifest.json
          trial_manifest.json
  reports/
    candidate_summary.csv
    candidate_summary.json
    pareto_frontier.csv
    pareto_frontier.json
    artifact_reuse_summary.csv
    artifact_reuse_summary.json
    pruning_summary.csv
    pruning_summary.json
    study_summary.md
```

Study manifests must not embed large arrays. They may store paths, IDs,
hashes, config snapshots or refs, aggregate metrics, runtime summaries, memory
summaries, cache statuses, and evidence refs.

### Required `study_manifest.json` Fields

- `manifest_version`
- `kind: tuning_study_manifest`
- `study_id`
- `study_status`
- `generated_at_utc`
- `updated_at_utc`
- `git`
- `runtime`
- `device_profile`
- `command`
- `cwd`
- `claim_boundary`
- `inputs.processed_manifest`
- `inputs.base_model_config`
- `inputs.search_space`
- `inputs.runtime_config`
- `inputs.device_config`
- `dataset.dataset_short_name`
- `dataset.split_family`
- `dataset.train_ratio`
- `dataset.validation_ratio`
- `dataset.selection_units`
- `model.name`
- `model.family`
- `budget.max_candidates`
- `budget.max_parallel`
- `budget.max_wall_seconds`
- `objective`
- `cache_policy`
- `resource_policy`
- `candidate_count`
- `candidate_manifest_paths`
- `trial_manifest_paths`
- `artifact_reuse_plan`
- `reports`

### Required `candidate_manifest.json` Fields

- `manifest_version`
- `kind: tuning_candidate_manifest`
- `study_id`
- `candidate_id`
- `candidate_index`
- `candidate_status`
- `search_space_version`
- `search_coordinates`
- `base_model_config`
- `candidate_config`
- `candidate_config_sha256`
- `effective_model_name`
- `effective_model_family`
- `effective_dtype`
- `selection_units`
- `fidelity_plan`
- `artifact_reuse_groups`
- `expected_trial_count`
- `trial_manifest_paths`
- `resource_gate_status`
- `objective_status`
- `claim_boundary`

### Required `candidate_summary.csv` Columns

- `study_id`
- `candidate_id`
- `candidate_status`
- `rank`
- `is_selected`
- `is_pareto_candidate`
- `primary_objective_metric`
- `primary_objective_mean`
- `primary_objective_std`
- `validation_rmse_mean`
- `validation_rmse_std`
- `validation_mae_mean`
- `validation_mae_std`
- `fit_model_seconds_mean`
- `fit_model_seconds_std`
- `train_time_total_mean`
- `peak_memory_mb_max`
- `completed_trial_count`
- `failed_trial_count`
- `pruned_trial_count`
- `resource_gate_passed`
- `cluster_artifact_miss_count`
- `cluster_artifact_hit_count`
- `artifact_reuse_group_ids`
- `candidate_config`
- `candidate_manifest`
- `notes`

### Required `artifact_reuse_summary.csv` Columns

- `study_id`
- `artifact_type`
- `reuse_group_id`
- `identity_hash`
- `candidate_count`
- `trial_count`
- `expected_cold_build_count`
- `actual_cache_miss_count`
- `actual_cache_hit_count`
- `cache_statuses`
- `reuse_across`
- `invalidate_on`
- `artifact_paths`
- `evidence_source`
- `notes`

The study artifacts should reference run-level `metrics.json`,
`performance_profile.json`, `kernel_profile.json`, and `run_manifest.json`
instead of copying those payloads wholesale.

## Objective Contract

The objective contract defines what the framework optimizes and how ties are
resolved.

MVP primary objective:

```yaml
objective:
  primary:
    metric: validation_rmse
    aggregation: mean
    direction: minimize
  tie_breakers:
    - metric: validation_rmse
      aggregation: std
      direction: minimize
    - metric: training_wall_clock_seconds
      aggregation: mean
      direction: minimize
    - metric: peak_memory_mb
      aggregation: max
      direction: minimize
  required_guards:
    - validation_mae
    - fit_model_seconds
    - total_wall_seconds
    - peak_memory_mb
    - cluster_cache_status
    - cluster_total_seconds
    - cb_claim_eligible
    - cb_diagnostics
```

Rules:

- Test metrics must not be part of inner tuning selection.
- Training metrics may be diagnostics or tie-breakers only when declared.
- Resource gates can make a candidate ineligible.
- Missing objective values must be explicit failure states, not silent zeros.
- Objective names, aggregations, and directions must be serialized in the study
  manifest and summary.
- Ranking metrics must not appear in the tuning objective unless a ranking
  evaluation protocol has first been defined and documented.
- A performance claim requires separate before/after or study evidence under
  the evaluation protocol. The presence of runtime guard metrics in tuning does
  not create a performance claim.

Required guard metrics for MVP:

| Metric | Required For | Purpose | Selection Use |
| --- | --- | --- | --- |
| `validation_mae` | all rating-prediction studies | Secondary quality sanity check | Guard or declared tie-breaker only |
| `fit_model_seconds` | all studies | Candidate cost visibility; for CB this must include required train-only artifact stages where available | Tie-breaker only when declared |
| `total_wall_seconds` | all studies | End-to-end local resource visibility | Guard/reporting only |
| `peak_memory_mb` | when available | Resource guard and scale-risk visibility | Resource gate or tie-breaker only when declared |
| `cluster_cache_status` | CB studies | Confirms cache reuse behavior for Cluster-Artifacts | Guard/reporting only |
| `cluster_total_seconds` | CB studies | Measures Cluster-Artifact cold/warm cost | Guard/reporting only unless a future cost objective declares it |
| `cb_claim_eligible` | CB studies where available | Preserves CB methodology/claim boundary | Guard/reporting only |
| `cb_diagnostics` | CB studies where available | Records CB-specific diagnostics such as alpha and cluster metadata | Guard/reporting only |

Not allowed:

- ranking metrics without a ranking protocol
- `test_rmse`, `test_mae`, or any test metric as a tuning objective
- selecting candidates from test-set values
- broad performance claims from validation-only tuning
- hiding failed, pruned, or resource-gated candidates from study evidence

Future scalar cost objective:

```text
score = validation_rmse + cost_penalty
```

This is future scope. A cost penalty must specify units, normalization,
weights, device profile, and whether it uses `fit_model_seconds`,
`total_wall_seconds`, `peak_memory_mb`, or Cluster-Artifact stage costs.

Future Pareto objective dimensions:

- `validation_rmse`
- `validation_mae`
- `fit_model_seconds`
- `peak_memory_mb`
- `cluster_total_seconds`

Pareto reporting should identify non-dominated candidates but should not make
an automatic quality or performance claim. The selected candidate must still
state the decision rule used to choose from the Pareto frontier.

## Artifact Reuse Contract

The framework must distinguish target-model knobs from artifact-identity knobs.

Reuse-safe target-model knobs for Cluster-Artifacts, based on Step 18a:

- `clustering.alpha`
- target-model lambdas
- target-model epochs when the induction config is unchanged

Cluster-Artifact rebuild knobs:

- `clustering.n_user_clusters`
- `clustering.n_item_clusters`
- induction config
- split id
- train fingerprint
- processed manifest fingerprint

The artifact reuse plan should record expected groups:

- split group
- user-history group
- explicit-feedback group
- Cluster-Artifact group
- user-cluster-history group

Each group should include:

- identity payload hash
- expected candidate ids
- expected trial count
- expected cold build count
- expected warm hit count
- actual cache statuses after execution

This is the core cache-aware tuning contract.

| Artifact Type | Reuse Across | Invalidate On | Evidence Source |
| --- | --- | --- | --- |
| split cache | Model hyperparameters, model family, candidate id, target-model seed when the split seed and processed data are unchanged | dataset, processed manifest/fingerprint, split family, train/validation/test ratio, split seed, split cache version | `src/recsys_lab/experiments/split_cache.py`; `docs/data_and_split_contract.md`; existing unified-runner cache metadata |
| training indices | Target-model learning rate, lambdas, epochs, alpha, model seed, and candidate id when train rows and dtype are unchanged | dataset, split id, train fingerprint, dtype, history layout version, index kind, user/item id arrays | `src/recsys_lab/data/training_index_cache.py`; `docs/evidence/performance/history_data_layout_v1.md` |
| explicit feedback index | Target-model learning rate, lambdas, epochs, alpha, model seed, and candidate id when explicit train data and dtype are unchanged | dataset, split id, train fingerprint, dtype, history layout version, explicit feedback values/order | `src/recsys_lab/data/training_index_cache.py`; `docs/evidence/performance/history_data_layout_v1.md` |
| user history index | Target-model learning rate, lambdas, epochs, alpha, model seed, and candidate id when implicit policy and train data are unchanged | dataset, split id, train fingerprint, dtype, history layout version, implicit policy, user/item ids | `src/recsys_lab/data/training_index_cache.py`; `docs/evidence/performance/history_data_layout_v1.md` |
| cluster artifacts | `alpha`, CB lambdas, target model epochs, target model learning rate, target model latent regularization when induction config is unchanged | `n_user_clusters`, `n_item_clusters`, induction config, `kmeans_n_init`, clustering algorithm, dataset/split/train fingerprint, processed manifest fingerprint | `docs/evidence/performance/cluster_artifact_kmeans_audit_v1.md`; `docs/evidence/performance/kmeans_candidate_strategy_v1.md`; `src/recsys_lab/clustering/cache.py` |
| user cluster history index | Target-model alpha/lambdas/epochs/learning rate when item cluster assignments and user history are unchanged | item cluster assignments, cluster cache key/fingerprint, train fingerprint, history layout version, `n_item_clusters`, dataset/split identity | `docs/evidence/performance/cluster_artifact_kmeans_audit_v1.md`; `tests/unit/test_cluster_artifact_cache_reuse.py`; `src/recsys_lab/clustering/cache.py` |
| kernel benchmark artifacts | Same code commit, same synthetic case, same dtype, same warmup/timed repeat contract, same benchmark harness version; useful only as local diagnostic comparison | code changes to kernels or harness, case definitions, dtype/layout, repeat contract, benchmark version, device/profile changes for interpreted comparison | `docs/evidence/performance/kernel_benchmark_harness_v1.md`; `src/recsys_lab/benchmarks/kernel_harness.py` |
| performance profiles | Same run id and run manifest context only; profiles are evidence for a completed run, not reusable inputs for another run | any new run, changed config, changed cache status, changed device, changed code commit, changed data/split/seed | `src/recsys_lab/experiments/performance.py`; `src/recsys_lab/experiments/unified_runner.py`; `docs/evaluation_protocol.md` |

Cluster-Artifact reuse rule for tuning:

- Reuse is allowed across target-model `alpha`, CB lambdas, target-model
  epochs, target-model learning rate, and target-model latent regularization
  only when the induction config is unchanged.
- Reuse is not allowed across changes to `n_user_clusters`,
  `n_item_clusters`, induction config, `kmeans_n_init`, clustering algorithm,
  dataset, split, train fingerprint, or processed manifest fingerprint.
- The study planner should report expected Cluster-Artifact cold-build count
  before execution and actual miss/hit counts after execution.

This table reflects the Step 18a cache-reuse contract and the Step 18c ML1M
diagnostic decision to proceed to tuning without changing KMeans or induction
algorithms.

## Cache-Aware Scheduling

Scheduling should minimize repeated cold artifact work without changing
selection semantics.

MVP scheduling:

- group candidates by split and Cluster-Artifact identity
- run one candidate/trial in each group first to populate expensive caches
- schedule remaining candidates in the same group while cache locality is high
- preserve deterministic order inside groups
- record the planned and actual order

The scheduler may reorder execution for efficiency, but it must not alter:

- candidate ids
- selection units
- objective aggregation
- pruning policy
- claim boundary

Cache-aware scheduling should be optional but enabled by default for CB studies.

## Multi-Fidelity / Pruning Design

Multi-fidelity is future scope, but the architecture should reserve contracts.

Fidelity axes:

- epochs
- split count
- candidate subset
- possibly smaller dataset slices only if separately documented

Pruning rules:

- pruning can happen only at declared fidelity boundaries
- every pruned candidate must have a `pruned` trial or candidate state
- pruning criteria must be serialized before execution
- pruned candidates are not failures
- pruned results must not be mixed with full-fidelity results without labels
- no test metrics may participate in pruning

MVP Step 19a should not implement pruning. It should define the manifest fields
needed later.

## Multi-Objective / Pareto Design

The future framework should support multi-objective analysis without forcing it
as the MVP selection rule.

Possible objectives:

- validation RMSE
- validation MAE
- fit time including train-only CB artifact stages
- peak memory
- model size
- cache cold-build count
- warm-cache reuse rate

Pareto outputs:

- `pareto_front.csv`
- `pareto_front.json`
- objective values and directions
- dominated/non-dominated flags
- selected candidate and reason

MVP should keep one primary objective plus tie-breakers, while writing enough
fields for later Pareto analysis.

## Extension Points

Future modules should be separable:

- search-space loader
- candidate generator
- candidate materializer
- artifact identity planner
- scheduler
- trial executor
- metric collector
- objective evaluator
- resource gate evaluator
- pruning policy
- Pareto analyzer
- study reporter

Optimizer backends such as Optuna/TPE/Sobol should plug into candidate
generation later. They should not own execution, artifact reuse, or manifests.

### Search And Evidence Extensions

| Extension | Purpose | Inputs | Outputs | Artifact reuse implications | Risks | MVP status |
| --- | --- | --- | --- | --- | --- | --- |
| `AlphaSearchExtension` | Vary productive CB mixture weight for `cb_svdpp` and `cb_asvdpp`. | Alpha dimension bounds or values; CB endpoint alpha policy; model family. | Candidate coordinates and `clustering.alpha` overrides. | Cluster-Artifacts are reusable across alpha when cluster counts and induction config stay fixed. | Endpoint alpha values can become hidden ablations or research variants if policy is not enforced. | included |
| `RegularizationSearchExtension` | Vary target-model regularization terms such as `lambda_b`, `lambda_p`, `lambda_q`, `lambda_y`, and CB lambdas. | Regularization dimension definitions; allowed target paths; log scale policy. | Candidate coordinates and training regularization overrides. | Cluster-Artifacts are reusable across target regularization when induction config remains unchanged. | If induction config is derived from target regularization, reuse can be invalidated unless the boundary is explicit. | included |
| `LearningRateSearchExtension` | Vary target-model learning rate. | Learning-rate dimension definition; scale policy; target path. | Candidate coordinates and `training.learning_rate` override. | Cluster-Artifacts are reusable across target learning rate when induction config remains unchanged. | Learning rate can affect stability and resource waste; failed candidates must remain visible. | included |
| `LatentDimSearchExtension` | Vary target-model latent dimension. | Latent-dim categorical values; target path; optional max cap. | Candidate coordinates and `training.latent_dim` override. | If induction config uses the same latent dim, Cluster-Artifacts are invalidated. If induction config is frozen separately, target latent dim can be reuse-safe only under an explicit contract. | High memory/runtime impact; can accidentally change induction identity. | included with explicit reuse classification required |
| `ArtifactReuseTracking` | Compute and report expected versus actual artifact reuse groups. | Candidate manifests; split identity; history identity; Cluster-Artifact identity; cache statuses. | `artifact_reuse_plan.json`, `artifact_reuse_summary.csv`, reuse group ids in candidate manifests. | Central mechanism for preventing unnecessary rebuilds. | Incorrect grouping can reuse invalid artifacts or hide rebuild cost. | included |
| `KMeansClusterSearchExtension` | Vary `n_user_clusters`, `n_item_clusters`, `kmeans_n_init`, or clustering algorithm. | Cluster count dimensions; KMeans parameter dimensions; algorithm choices. | Candidate coordinates and clustering overrides. | Cluster-Artifacts rebuild when cluster counts, `kmeans_n_init`, algorithm, or induction config changes. | Can multiply expensive cold builds; changes cluster quality/stability; algorithm changes need separate evidence. | deferred |
| `InductionConfigSearchExtension` | Vary Biased-MF induction profile independently from target model config. | Induction latent dim, epochs, learning rate, regularization, seed, dtype. | Candidate coordinates and explicit induction config identity. | Induction config changes invalidate Cluster-Artifacts. Frozen induction config enables reuse across target-model knobs. | Changes cluster quality and downstream CB behavior; needs quality/stability evidence. | deferred |
| `EpochBudgetExtension` | Vary target training epochs or assign fidelity budgets. | Epoch values; budget/fidelity policy; promotion rules. | Candidate epoch overrides and fidelity levels. | Target epochs are Cluster-Artifact reuse-safe when induction config is unchanged. | Lower epochs are lower-fidelity evidence, not final quality evidence. | included for fixed candidate budgets; multi-fidelity deferred |
| `PruningExtension` | Stop or stop promoting candidates based on declared early evidence. | Fidelity levels; pruning metric; threshold or ranking rule; minimum observations. | Pruned candidate/trial states and `pruning_summary.csv`. | Pruning should not change artifact identity; it changes how many trials are executed. | Biases study if policy is not fixed before execution; must not use test metrics. | deferred |
| `ParetoEvidenceExtension` | Report non-dominated candidates across quality, cost, memory, and cluster-stage dimensions. | Objective values; metric directions; guard metrics; candidate summaries. | `pareto_frontier.csv`, `pareto_frontier.json`, Pareto flags in candidate summary. | No direct reuse impact; may expose cache-cost tradeoffs. | Pareto status can be misread as a claim; selection rule must remain explicit. | deferred |

MVP included extensions:

- `AlphaSearchExtension`
- `RegularizationSearchExtension`
- `LearningRateSearchExtension`
- `LatentDimSearchExtension`
- `EpochBudgetExtension` for fixed per-candidate epoch budgets only
- `ArtifactReuseTracking`

Deferred extensions:

- `KMeansClusterSearchExtension`
- `InductionConfigSearchExtension`
- multi-fidelity `EpochBudgetExtension`
- `PruningExtension`
- `ParetoEvidenceExtension`

## Minimal Schema Scaffold

Step 19a includes a small contract-only scaffold in `src/recsys_lab/tuning/`.
It is intentionally not an execution engine:

- no candidate runner
- no optimizer dependency
- no integration with `unified_runner`
- no file IO
- no model or kernel imports

The scaffold defines Pydantic contracts for:

- `StudySpec`
- `SearchSpaceSpec`
- `ObjectiveSpec`
- artifact reuse declarations

The initial validation surface is deliberately narrow:

- unknown fields are rejected
- productive CB search spaces reject `alpha <= 0` and `alpha >= 1`
- test metrics are rejected as primary tuning objectives
- the default Cluster-Artifact reuse contract documents reuse across `alpha`

## Minimal MVP Scope

Recommended next step:

`19b. Cache-aware Tuning MVP V1`

This should be the first framework MVP, not the final SOTA tuning system.
Recommended scope:

- extend the Step 19a schema scaffold into `tuning_study_manifest_v1`
- define candidate manifest schema
- dataset scope: ML100K first, ML1M optional only if the local run budget is
  explicitly acceptable
- model scope: `cb_svdpp` first
- search method: deterministic small grid, or Latin Hypercube without an
  external optimizer dependency
- candidate budget: small, for example 8-16 candidates
- objective: validation RMSE
- tracked metrics: validation MAE, fit-model seconds, total wall time, cluster
  cache status, and cluster total seconds
- no pruning yet
- no KMeans search yet
- support deterministic grid generation over categorical dimensions
- materialize candidate configs without executing large studies
- compute artifact reuse groups for CB studies
- reuse Cluster-Artifacts across alpha, lambda, and learning-rate candidates
  when the Step 18a/18c identity contract allows it
- write a dry-run study plan with candidate and artifact-reuse manifests
- write full study manifest, candidate manifests, and candidate summary CSV
- add unit tests for schema validation, candidate ids, and artifact grouping

The MVP may include a tiny synthetic or ML100K smoke execution only if it is
needed to prove manifest wiring. It should not run a large tuning campaign.

## Future Scope

Later phases can add:

- generic `recsys-lab tune-study` CLI
- execution of planned studies
- resume/retry behavior
- pruning at declared fidelity boundaries
- Pareto front outputs
- optimizer plugins
- study-level dashboards or report collectors
- ML1M/ML10M/ML20M tuning campaigns
- cache IO optimization based on repeated study access patterns
- induction artifact cache if repeated real studies show induction rebuilds
  dominate

## Acceptance Gates

Architecture acceptance for Step 19a:

- document existing tuning inventory
- define lifecycle and core concepts
- define search-space, candidate, study, objective, and artifact-reuse
  contracts
- define cache-aware scheduling behavior
- define multi-fidelity and multi-objective extension points
- define minimal MVP scope
- add a minimal contract-only schema scaffold
- no production tuning engine implemented
- no new optimizer dependency added
- no large tuning study run
- no model, kernel, KMeans, or induction semantics changed
- hotpath/coldpath boundary remains intact
- claim boundary remains explicit

Future implementation gates:

- schema/unit tests for all new manifests
- dry-run artifact-reuse plan tests
- active config compatibility tests
- no test-set leakage tests
- cache identity grouping tests for alpha/lambda/epoch tuning
- resource-gate tests
- claim-check over `docs`, `src`, and `tests`
