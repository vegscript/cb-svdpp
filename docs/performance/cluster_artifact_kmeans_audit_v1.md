# Cluster Artifact / KMeans Audit V1

## Purpose

Define the coldpath profiling contract for cluster induction, cluster artifact
cache reuse, and user-cluster-history cache reuse.

This document defines the profiling concept and records the first low-risk
additive implementation. It does not change KMeans behavior, model formulas,
CB kernels, model dispatch, or cache identity semantics.

## Current Pipeline Boundary

Cluster artifacts are resolved in coldpath experiment orchestration:

- `src/recsys_lab/clustering/latent_kmeans.py`
- `src/recsys_lab/clustering/cache.py`
- `src/recsys_lab/experiments/unified_runner.py`

The hotpath kernels remain outside this profiling scope. No timer, JSON,
reporting, cache, or experiment logic belongs in `src/recsys_lab/models/kernels.py`.

## Stage Report Contract

Future profiling should emit one structured cluster profile per CB run. The
profile should be embedded in run evidence or written as a coldpath artifact;
it must not be produced from model kernels.

Required timing fields:

| Field | Meaning |
| --- | --- |
| `cluster_total_seconds` | End-to-end cluster artifact resolution time. |
| `cluster_cache_read_seconds` | Time spent attempting to read cluster artifacts from cache. |
| `cluster_cache_write_seconds` | Time spent writing cluster artifacts after a miss. |
| `cluster_cache_status` | `hit`, `miss`, or `disabled` from cluster artifact cache metadata. |
| `induction_fit_seconds` | Time spent fitting the Biased-MF induction model. |
| `induction_predict_seconds` | Time spent predicting train rows with the induction model. |
| `induction_train_rmse_seconds` | Time spent computing train RMSE for induction diagnostics. |
| `user_kmeans_seconds` | Time spent fitting/predicting user factor KMeans. |
| `item_kmeans_seconds` | Time spent fitting/predicting item factor KMeans. |
| `r_star_seconds` | Time spent computing `r_star_means` and `r_star_counts`. |
| `cluster_artifact_validation_seconds` | Time spent validating cluster artifacts loaded or built. |
| `user_cluster_history_total_seconds` | End-to-end user-cluster-history resolution time. |
| `user_cluster_history_cache_read_seconds` | Time spent attempting to read user-cluster history from cache. |
| `user_cluster_history_cache_write_seconds` | Time spent writing user-cluster history after a miss. |
| `user_cluster_history_build_seconds` | Time spent building user-cluster history after a miss or disabled cache. |
| `user_cluster_history_validation_seconds` | Time spent validating user-cluster history loaded from cache. |
| `user_cluster_history_cache_status` | `hit`, `miss`, or `disabled` from user-cluster-history cache metadata. |

Required metadata fields:

| Field | Source |
| --- | --- |
| `dataset_short_name` | Unified runner dataset context. |
| `split_family` | Unified runner split context. |
| `split_id` | Cache split identity. |
| `model` | Unified model adapter name, for example `cb_svdpp` or `cb_asvdpp`. |
| `n_users` | Train data shape. |
| `n_items` | Train data shape. |
| `train_rows` | Train data shape / cache fingerprint row count. |
| `n_user_clusters` | Validated clustering config. |
| `n_item_clusters` | Validated clustering config. |
| `algorithm` | Validated clustering config, currently `kmeans`. |
| `kmeans_n_init` | Validated clustering config. |
| `induction_seed` | Biased-MF induction config seed. |
| `induction_latent_dim` | Biased-MF induction config latent dimension. |
| `induction_epochs` | Biased-MF induction config epochs. |
| `induction_dtype` | Biased-MF induction config dtype. |
| `cluster_cache_key` | Cluster cache metadata. |
| `cluster_cache_fingerprint_sha256` | Cluster cache metadata. |
| `user_cluster_history_cache_key` | User-cluster-history cache metadata. |
| `user_cluster_history_cache_fingerprint_sha256` | User-cluster-history cache metadata. |

## Measurement Semantics

Cache timing should separate read, build, and write work:

- cache hit:
  - read time records lookup and artifact loading
  - validation time records layout/content validation separately
  - build and write time should be zero
- cache miss:
  - read time records the failed lookup/validation attempt
  - build time records induction or history construction
  - validation time records validation of newly built artifacts
  - write time records artifact persistence
- cache disabled:
  - read and write time should be zero
  - build time records direct construction
  - validation time records validation of newly built artifacts

Stages that do not run for a path use `0.0`. For example, cache-hit paths keep
build and write timing fields at `0.0`, while cache-disabled paths keep read
and write timing fields at `0.0`.

Cluster artifact timing should decompose the current monolithic
`induce_train_only_clusters(...)` path:

1. `induction_fit_seconds`
2. `induction_predict_seconds`
3. `induction_train_rmse_seconds`
4. `user_kmeans_seconds`
5. `item_kmeans_seconds`
6. `r_star_seconds`
7. `cluster_artifact_validation_seconds`

User-cluster-history timing should decompose:

1. cache read
2. build via `build_user_cluster_count_index(...)`
3. cache write
4. validation

## Cache Reuse Questions

The audit should make these questions answerable from artifacts:

- Was the cluster artifact cache hit, missed, or disabled?
- Was the user-cluster-history cache hit, missed, or disabled?
- Are cache keys and fingerprints stable across tuning candidates that share
  the same dataset, split, induction config, cluster counts, algorithm, and
  KMeans settings?
- Which cluster stage dominates when cache is disabled or missed?
- Does tuning pay cluster induction repeatedly when cache reuse should have
  been possible?

## Non-Goals

Do not use this profiling contract to introduce:

- MiniBatchKMeans
- sampling
- alternative clustering algorithms
- KMeans parameter changes
- model formula changes
- CB kernel changes
- model-wrapper dispatch
- tuning-engine changes
- runtime improvement claims

## Low-Risk Implementation Added

Implemented coldpath pieces:

- `ClusterInductionProfile` in `latent_kmeans.py`
  - records induction fit, prediction, train RMSE, user KMeans, item KMeans,
    and `r_star` timings
- `ClusterStageTimings` in `clustering/cache.py`
  - records the stage timing fields with zero defaults
- `ClusterArtifactProfile` in `clustering/cache.py`
  - carries timing fields, cache status, cache keys/fingerprints, dataset and
    induction metadata
- `ClusterArtifactsCacheResult.profile`
  - additive optional field
- `UserClusterHistoryCacheResult.profile`
  - additive optional field
- `scripts/profile_cluster_artifacts.py`
  - coldpath local reporter for targeted cluster artifact and
    user-cluster-history profiling
  - writes `artifacts/reports/cluster_artifact_profile_v1.json`
  - writes `artifacts/reports/cluster_artifact_profile_v1.csv`
  - does not serialize cluster arrays and does not make performance claims

Compatibility:

- Existing productive call sites continue to receive the same primary
  artifacts and metadata.
- New profile fields are additive and default-compatible.
- No timers were added to Numba kernels.
- No cache identity, KMeans parameter, model formula, or dispatch behavior was
  changed.

Focused tests:

- `pytest tests/unit/test_clustering.py tests/unit/test_cluster_cache.py`
- `pytest tests/unit/test_cluster_artifact_cache_reuse.py`
- `pytest tests/integration/test_cluster_artifact_cache_smoke.py`
- `pytest tests/unit/test_profile_cluster_artifacts_script.py`

Reporter usage:

```bash
python scripts/profile_cluster_artifacts.py \
  --processed-manifest data/processed/ml1m/manifest.json \
  --model-config configs/models/selected/ml1m/ml1m_cb_svdpp_stage0_transfer.yaml \
  --runtime-config configs/runtime/base.yaml \
  --model cb_svdpp \
  --split-family benchmark_random_v1 \
  --split-seed 1 \
  --model-seed 1 \
  --repeats 2
```

The script currently supports `benchmark_random_v1` to keep the reporter
bounded. It is intended for local diagnostic profiling only.

## Implementation Guidance

The preferred implementation path is coldpath-only:

- add a structured profiler object or payload builder under
  `src/recsys_lab/clustering/` or `src/recsys_lab/experiments/`
- keep timers around cache and induction stages, not inside model kernels
- surface the profile through `unified_runner.py` artifacts and manifests
- add unit tests for payload contract and cache-status timing semantics
- add a small integration smoke that proves profiles are written for a tiny CB
  run without relying on runtime thresholds

Later phases should attach these profiles to unified runner artifacts and add
integration evidence for cache reuse across tuning candidates.
