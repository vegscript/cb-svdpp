# Cluster Artifact / KMeans Audit V1 Evidence

## Branch

`cluster-artifact-kmeans-audit-v1`

## Goal

Make cluster induction, cluster artifact cache reuse, and user-cluster-history
cache reuse measurable and contract-checkable before any KMeans or artifact
optimization work.

This step adds profiling and cache-reuse evidence only. It does not change
KMeans behavior, model formulas, CB kernels, model dispatch, or clustering
artifact semantics.

## Files Audited

- `src/recsys_lab/clustering/latent_kmeans.py`
- `src/recsys_lab/clustering/cache.py`
- `src/recsys_lab/data/training_index_cache.py`
- `src/recsys_lab/data/histories.py`
- `src/recsys_lab/experiments/unified_runner.py`
- `src/recsys_lab/models/cb_svdpp.py`
- `src/recsys_lab/models/cb_asvdpp.py`
- `tests/unit/test_cluster_cache.py`
- `tests/unit/test_cluster_artifact_cache_reuse.py`
- `tests/integration/test_cluster_artifact_cache_smoke.py`

## Cluster Artifact Lifecycle

Cluster artifacts remain train-only coldpath artifacts:

1. Load train data.
2. Build the Biased-MF induction model.
3. Predict train rows with the induction model.
4. Compute induction train RMSE diagnostics.
5. Fit user-factor KMeans.
6. Fit item-factor KMeans.
7. Compute `r_star_means` and `r_star_counts`.
8. Validate cluster artifact arrays.
9. Write or load cluster artifacts through the cluster cache.
10. Build, validate, write, or load user-cluster-history through its cache.

The profile is attached to coldpath cache result objects. No profiling code was
added to `src/recsys_lab/models/kernels.py`.

## Profiling Fields Added

Cluster artifact stage timings:

- `cluster_total_seconds`
- `cluster_cache_read_seconds`
- `cluster_cache_write_seconds`
- `induction_fit_seconds`
- `induction_predict_seconds`
- `induction_train_rmse_seconds`
- `user_kmeans_seconds`
- `item_kmeans_seconds`
- `r_star_seconds`
- `cluster_artifact_validation_seconds`

User-cluster-history stage timings:

- `user_cluster_history_total_seconds`
- `user_cluster_history_cache_read_seconds`
- `user_cluster_history_cache_write_seconds`
- `user_cluster_history_build_seconds`
- `user_cluster_history_validation_seconds`

Profile metadata:

- dataset, split, model label
- train rows, user/item counts where available
- cluster counts, algorithm, `kmeans_n_init`
- induction seed, latent dimension, epochs, dtype
- cluster cache key/fingerprint
- user-cluster-history cache key/fingerprint
- cache status fields

Convention: Stages that do not run for a path are reported as `0.0`.

## Cache Reuse Contract

The cache identity contract is tested in
`tests/unit/test_cluster_artifact_cache_reuse.py`.

Cluster artifact cache key is stable when only these target-model tuning values
change:

- CB `alpha`
- CB cluster lambdas
- CB target-model epochs, when the induction config is unchanged

Cluster artifact cache key changes when these identity inputs change:

- `n_user_clusters`
- `n_item_clusters`
- induction config
- split id
- train fingerprint

User-cluster-history cache identity:

- depends on item cluster assignments
- includes `history_data_layout_v1`
- records `int32` index/count layout

Profiling metadata is not part of cache identity. Runtime timings, model label,
and cache status are report metadata only.

## Cold/Warm Cache Smoke Result

`tests/integration/test_cluster_artifact_cache_smoke.py` validates cold/warm
behavior on a small synthetic dataset:

- first cluster artifact resolution: cache `miss`
- second cluster artifact resolution: cache `hit`
- cold/warm cluster artifacts are exactly equal for:
  - `user_clusters`
  - `item_clusters`
  - `user_cluster_sizes`
  - `item_cluster_sizes`
  - `r_star_means`
  - `r_star_counts`
- first user-cluster-history resolution: cache `miss`
- second user-cluster-history resolution: cache `hit`
- cold/warm user-cluster-history arrays are exactly equal for:
  - `indptr`
  - `cluster_ids`
  - `counts`

## Tests/Gates

Focused checks:

- `ruff check src\recsys_lab\clustering\cache.py tests\unit\test_cluster_cache.py`
- `pytest tests/unit/test_cluster_cache.py tests/unit/test_clustering.py`
- `ruff check tests\unit\test_cluster_artifact_cache_reuse.py`
- `pytest tests/unit/test_cluster_artifact_cache_reuse.py`
- `ruff check tests\integration\test_cluster_artifact_cache_smoke.py`
- `pytest tests/integration/test_cluster_artifact_cache_smoke.py`
- `ruff check scripts\profile_cluster_artifacts.py tests\unit\test_profile_cluster_artifacts_script.py`
- `pytest tests/unit/test_profile_cluster_artifacts_script.py`

Final Step 18a gates:

- `ruff check .` passed.
- `pytest tests/unit/test_cluster_artifact_cache_reuse.py` passed.
- `pytest tests/integration/test_cluster_artifact_cache_smoke.py` passed.
- `pytest tests/unit/test_hotpath_coldpath_boundaries.py` passed.
- `pytest tests/unit` passed.
- `pytest tests/integration/test_unified_pipeline_smoke_all_models.py` passed.
- `pytest` passed: 297 passed, 2 skipped.
- `python scripts/profile_cluster_artifacts.py --help` passed.
- The requested claim-check command was run. It returned existing
  claim-boundary and claim-lock references, not a new Step 18a performance
  claim.

## Findings

- The repo can measure cluster induction in named coldpath stages.
- Cache-reuse identity is contract-checked for alpha/lambda/epoch tuning
  scenarios.
- Cold/warm cache behavior is validated on a small synthetic dataset.
- Cluster artifact cache identity remains separate from runtime profiling
  metadata.
- Existing cache-key semantics were not changed by profiling fields.
- No cache identity or profiling blocker was found that requires a separate
  cache-hardening step before the next audit.
- The synthetic smoke does not establish which stage dominates on real ML data;
  it only proves that the profiler and cache paths report the needed fields.

## Limitations

- The synthetic smoke validates behavior, not large-dataset runtime.
- The reporter currently supports `benchmark_random_v1` only.
- The current evidence does not rank real ML1M bottlenecks yet; it establishes
  the profiling mechanism and cache contract.
- No KMeans alternative, sampling strategy, or cache-hardening migration was
  implemented.

## Claim Boundary

No performance improvement claim is made.

Allowed conclusions:

- The repo can measure Cluster-Induction in stages.
- Cache reuse is contract-checked for alpha/lambda/epoch tuning.
- Cold/warm cache behavior is validated on a small test dataset.

Disallowed conclusions:

- KMeans is faster.
- Cluster induction is optimized.
- Large-dataset cache reuse is fully characterized.

## Recommended Next Step

Decision: cache reuse is clean under the tested contract, and no concrete 18b
cache-hardening blocker was found.

Recommended next step:

`18c. KMeans Algorithm / Candidate Strategy V1`

The next step should start by using the new profiler on local claim-ineligible
diagnostic runs to rank induction fit, user KMeans, item KMeans, `r_star`, cache
IO, and user-cluster-history costs before proposing any algorithmic change.
