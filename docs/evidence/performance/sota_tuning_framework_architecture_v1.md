# SOTA Tuning Framework Architecture V1 Evidence

## Branch

`sota-tuning-framework-architecture-v1`

## Goal

Define the architecture, contracts, artifact layout, and MVP scope for a
cache-aware tuning framework.

This step is architectural and contract-only. It does not implement a tuning
engine, add an optimizer dependency, run a tuning study, change model formulas,
change kernels, change KMeans behavior, or change induction defaults.

## Inputs Reviewed

- `docs/evidence/performance/cluster_artifact_kmeans_audit_v1.md`
- `docs/evidence/performance/kmeans_candidate_strategy_v1.md`
- `docs/architecture/adr_cb_endpoint_alpha_policy_v1.md`
- `docs/evidence/performance/cb_endpoint_alpha_policy_v1.md`
- `docs/evidence/performance/exact_residual_reuse_validation_v1.md`
- `configs/experiments/tuning/active/`
- `configs/experiments/tuning/archive/`
- `configs/experiments/tuning/templates/`
- `scripts/tune_ml100k_inner.ps1`
- `src/recsys_lab/experiments/ml100k_inner_tuning.py`
- `src/recsys_lab/cli/main.py`
- `src/recsys_lab/tuning/schemas.py`
- `tests/unit/test_tuning_schemas.py`

Inventory command:

```powershell
rg "tuning|grid|candidate|alpha|search_space|study|trial|sweep|optuna|halving|pruning" src scripts configs tests docs
```

## Current Tuning Inventory

Active tuning configs:

- `configs/experiments/tuning/active/ml100k_cb_svdpp_g6_validation_grid.yaml`
- `configs/experiments/tuning/active/ml1m_cb_svdpp_stage0.yaml`
- `configs/experiments/tuning/active/ml1m_cb_asvdpp_stage0.yaml`
- `configs/experiments/tuning/active/ml20m_cb_svdpp_g11_lower_memory_validation_grid.yaml`

Existing tuning entry points:

- `recsys-lab tune-inner`
- `recsys-lab tune-ml100k-inner`
- `scripts/tune_ml100k_inner.ps1`

Existing implementation:

- `src/recsys_lab/experiments/ml100k_inner_tuning.py`

Current outputs are benchmark-shaped under:

```text
artifacts/benchmarks/<timestamp>_<dataset>_inner_tuning_<model>_<stage>_<device>/
```

They include benchmark manifests, config snapshots, summaries, logs, candidate
configs, and referenced run artifacts. They do not yet define a stable
study-level manifest, candidate manifest, trial manifest, artifact reuse plan,
or Pareto/pruning summaries.

The current tuning implementation is useful for small hand-written grids, but
it combines config loading, candidate materialization, runner dispatch, metric
aggregation, resource gates, selection, and reporting in one monolithic module.

## Architecture Summary

The Step 19a architecture is documented in:

`docs/architecture/sota_tuning_framework_architecture_v1.md`

Core architecture decisions:

- Tuning is a framework of Study, Search Space, Candidate, Trial, Fidelity,
  Artifact Reuse Group, Objective, and Study Result.
- The framework should wrap `run_unified_experiment(...)` as the execution
  primitive instead of replacing it.
- Candidate generation, candidate materialization, artifact identity planning,
  trial execution, objective evaluation, pruning, and evidence reporting are
  separate responsibilities.
- Large study execution, optimizer integration, pruning, and Pareto selection
  are deferred.

Step 19a also adds a minimal schema scaffold:

- `src/recsys_lab/tuning/__init__.py`
- `src/recsys_lab/tuning/schemas.py`

This scaffold is contract-only:

- no execution engine
- no candidate runner
- no optimizer dependency
- no integration with `unified_runner`
- no file IO
- no model or kernel imports

## Search-Space Contract

The architecture defines a versioned YAML search-space contract with:

- `search_space_version`
- `study`
- `base_model_config`
- `budget`
- `generator`
- `search_space`
- `artifact_reuse`
- `objective`

The minimal schema scaffold validates:

- unknown fields are rejected
- numeric dimensions define `low`, `high`, and `distribution`
- categorical dimensions define non-empty `values`
- productive CB search spaces for `cb_svdpp` and `cb_asvdpp` reject endpoint
  alpha values

Endpoint alpha policy:

- productive CB candidates require `0 < alpha < 1`
- `alpha == 0` is an individual-only baseline/ablation policy, not a normal
  productive CB candidate
- `alpha == 1` is a research variant policy, not a normal productive CB
  candidate

## Artifact-Reuse Contract

The architecture defines artifact reuse tables for:

- split cache
- training indices
- explicit feedback index
- user history index
- Cluster-Artifacts
- user-cluster-history index
- kernel benchmark artifacts
- performance profiles

Cluster-Artifact tuning contract inherited from Steps 18a/18c:

Reuse across:

- `alpha`
- CB lambdas
- target model epochs
- target model learning rate
- target model latent regularization when induction config is unchanged

Invalidate on:

- `n_user_clusters`
- `n_item_clusters`
- induction config
- `kmeans_n_init`
- clustering algorithm
- dataset/split/train fingerprint
- processed manifest fingerprint

The schema scaffold exposes this contract through:

- `DEFAULT_CLUSTER_ARTIFACT_REUSE_ACROSS`
- `DEFAULT_CLUSTER_ARTIFACT_INVALIDATE_ON`
- `default_cluster_artifact_reuse_spec()`

## Objective Contract

MVP objective:

- primary objective: `validation_rmse`, minimize

Required guard metrics:

- `validation_mae`
- `fit_model_seconds`
- `total_wall_seconds`
- `peak_memory_mb`, when available
- `cluster_cache_status`, for CB
- `cluster_total_seconds`, for CB
- CB diagnostics and claim-eligibility fields when available

Explicitly not allowed:

- test metrics as tuning objectives
- ranking metrics without a ranking protocol
- performance claims from validation tuning alone
- hidden failed, skipped, resource-gated, or pruned candidates

The schema scaffold rejects `test_rmse`, `test_mae`, and other `test_*`
metrics as primary tuning objectives.

## Extension Points

MVP-included extension points:

- `AlphaSearchExtension`
- `RegularizationSearchExtension`
- `LearningRateSearchExtension`
- `LatentDimSearchExtension`
- fixed-budget `EpochBudgetExtension`
- `ArtifactReuseTracking`

Deferred extension points:

- `KMeansClusterSearchExtension`
- `InductionConfigSearchExtension`
- multi-fidelity `EpochBudgetExtension`
- `PruningExtension`
- `ParetoEvidenceExtension`

Deferred means the architecture defines the contract shape, but Step 19a does
not implement execution or optimizer behavior.

## MVP Recommendation

Recommended next step:

`19b. Cache-aware Tuning MVP V1`

Scope:

- extend the Step 19a schema scaffold into study and candidate manifest
  schemas
- dataset: ML100K first, ML1M optional only if the local run budget is
  explicitly acceptable
- model: `cb_svdpp` first
- search method: deterministic small grid or Latin Hypercube without an
  external optimizer dependency
- max candidates: small, for example 8-16
- no pruning yet
- no KMeans search yet
- Cluster-Artifacts reused across alpha, lambda, and learning-rate candidates
  when cluster counts and induction config stay fixed
- full study manifest, candidate manifests, and candidate summary CSV
- objective: validation RMSE
- tracked metrics: validation MAE, fit-model seconds, total wall time, cluster
  cache status, and cluster total seconds
- support deterministic grid generation over categorical dimensions
- materialize candidate configs without running large studies
- compute artifact reuse groups for CB studies
- write a dry-run study plan with candidate and artifact-reuse manifests
- add unit tests for schema validation, candidate ids, and artifact grouping

19b is not SOTA-final. It is the first framework MVP. It should remain dry-run
first; a tiny synthetic or ML100K smoke can be added only if needed to prove
manifest wiring.

## Tests/Gates

Step 19a gates:

- `ruff check src/recsys_lab/tuning tests/unit/test_tuning_schemas.py` passed.
- `pytest tests/unit/test_tuning_schemas.py` passed: 6 passed.
- `ruff check .` passed.
- `pytest tests/unit/test_hotpath_coldpath_boundaries.py` passed: 13 passed.
- `pytest tests/unit` passed: 235 passed.
- `pytest tests/integration/test_unified_pipeline_smoke_all_models.py` passed:
  1 passed.
- `pytest tests/integration/test_release_evidence_integrity.py` passed:
  5 passed.
- `pytest` passed: 310 passed, 2 skipped.
- The requested claim-check command was run. It returned existing
  claim-boundary and claim-lock references, not a new Step 19a performance or
  quality claim.

Schema tests cover:

- unknown search-space fields are rejected
- productive CB `alpha == 0` is rejected
- productive CB `alpha == 1` is rejected
- productive CB alpha values strictly between zero and one are accepted
- Cluster-Artifact reuse across `alpha` is documented in the contract
- `test_rmse` is rejected as primary tuning objective

## Claim Boundary

No claim is made that tuning is faster.

No claim is made that any model is better.

This evidence only states:

- the tuning framework architecture is defined
- search-space, objective, artifact-reuse, study, and candidate contracts are
  defined
- a minimal schema scaffold exists for the first contract checks
- the MVP scope for Step 19b is defined
- Cluster-Artifact reuse findings from Steps 18a/18c are represented in the
  tuning contract

## Next Step

Step 19a gates are complete. Proceed to:

`19b. Cache-aware Tuning MVP V1`

The next step should remain dry-run and manifest-focused before executing any
large tuning study.
