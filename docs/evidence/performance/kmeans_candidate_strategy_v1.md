# KMeans Candidate Strategy V1 Evidence

## Branch

`kmeans-candidate-strategy-v1`

## Goal

Use the Step 18a Cluster-Artifact profiler on local ML1M diagnostic runs to
rank cluster-induction costs and decide whether to implement a KMeans,
induction, `r_star`, or cache-layout optimization before the tuning framework.

This step is diagnostic only. It does not implement MiniBatchKMeans, change
KMeans defaults, change induction defaults, alter model formulas, or modify CB
kernels.

## Inputs Reviewed

- `docs/evidence/performance/cluster_artifact_kmeans_audit_v1.md`
- `docs/performance/cluster_artifact_kmeans_audit_v1.md`
- `docs/performance/kmeans_candidate_strategy_v1.md`
- `src/recsys_lab/clustering/cache.py`
- `src/recsys_lab/clustering/latent_kmeans.py`
- `scripts/profile_cluster_artifacts.py`
- `tests/unit/test_cluster_artifact_cache_reuse.py`
- `tests/integration/test_cluster_artifact_cache_smoke.py`

## Diagnostic Run Plan

Profile command inputs:

- dataset: `ml1m`
- processed manifest:
  `data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config:
  `configs/models/archive/tuned/ml1m_cb_svdpp_stage0_probe_e003.yaml`
- model: `cb_svdpp`
- split family: `benchmark_random_v1`
- train ratio: `0.8`
- validation ratio: `0.1`
- split seed: `1`
- model seed: `1`
- user clusters: `80`
- item clusters: `80`
- algorithm: `kmeans`
- `kmeans_n_init`: `10`
- induction seed: `1`
- induction latent dim: `64`
- induction epochs: `3`
- induction dtype: `float32`
- device profile in local runtime override: `local_u300_24gb`

The run used an isolated cache root under `artifacts/local/` so repeat 1
exercised cold miss/build and repeat 2 exercised warm hit/load.

Planned diagnostic scope:

- P1 required: ML1M cold-cache profile and warm-cache profile.
- P2 optional: ML100K smoke only if the ML1M profiler path failed or needed a
  quick sanity check.
- P3 optional: ML10M only as an explicit follow-up, not started in this step.

## Runs Executed

One ML1M local diagnostic run was executed through:

`scripts/profile_cluster_artifacts.py`

The run used `--repeats 2`:

- repeat 1: isolated cache miss/build path
- repeat 2: warm cache hit/load path

Final diagnostic command:

```powershell
C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe scripts/profile_cluster_artifacts.py --processed-manifest data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json --model-config configs/models/archive/tuned/ml1m_cb_svdpp_stage0_probe_e003.yaml --runtime-config artifacts/reports/kmeans_candidate_strategy_v1/runtime_local_diagnostic_ml1m.yaml --model cb_svdpp --split-family benchmark_random_v1 --train-ratio 0.8 --validation-ratio 0.1 --split-seed 1 --model-seed 1 --repeats 2 --output-dir artifacts/reports/kmeans_candidate_strategy_v1 --output-stem cluster_artifact_profile_ml1m
```

Generated artifacts:

- `artifacts/reports/kmeans_candidate_strategy_v1/cluster_artifact_profile_ml1m.json`
- `artifacts/reports/kmeans_candidate_strategy_v1/cluster_artifact_profile_ml1m.csv`
- `artifacts/reports/kmeans_candidate_strategy_v1/cluster_artifact_profile_ml1m_summary.csv`
- `artifacts/reports/kmeans_candidate_strategy_v1/cluster_stage_ranking.csv`

These artifacts are local diagnostics and are not versioned.

## Stage Ranking

- repeat 1 cluster artifacts: `cluster_cache_status == "miss"`
- repeat 1 user-cluster-history: `user_cluster_history_cache_status == "miss"`
- repeat 2 cluster artifacts: `cluster_cache_status == "hit"`
- repeat 2 user-cluster-history: `user_cluster_history_cache_status == "hit"`

Local diagnostic totals:

- `cold_cluster_total_seconds`: `8.638122900`
- `warm_cluster_total_seconds`: `0.272404600`
- `cold_vs_warm_ratio`: `31.710635210`

Selected stage shares:

- `cluster_cache_write_share_of_cold_total`: `0.359808564`
- `user_kmeans_share_of_cold_total`: `0.230664697`
- `induction_fit_share_of_cold_total`: `0.213262479`
- `item_kmeans_share_of_cold_total`: `0.128295581`
- `induction_predict_share_of_cold_total`: `0.055204945`
- `r_star_share_of_cold_total`: `0.004075029`
- `cluster_cache_read_share_of_warm_total`: `0.675378463`
- `user_cluster_history_build_share`: `0.136577741`
- `user_cluster_history_cache_read_share`: `0.777945681`

These values are a single local ML1M diagnostic readout on the laptop profile,
not a stable runtime generalization.

The full stage ranking is stored in:

`artifacts/reports/kmeans_candidate_strategy_v1/cluster_stage_ranking.csv`

## Cold/Warm Interpretation

On this local ML1M diagnostic profile:

- The cold Cluster-Artifact build was split across cache write, user KMeans,
  induction fit, item KMeans, and induction prediction.
- The warm Cluster-Artifact path loaded from cache and did not run induction
  fit, induction prediction, KMeans, or `r_star` stages.
- The warm user-cluster-history path loaded from cache and did not rebuild the
  history index.
- Cache IO is visible in both cluster and user-cluster-history paths, but this
  single local readout is not enough to justify a cache-format or layout change.

## Candidate Classification

| Candidate | Classification |
| --- | --- |
| A. No algorithm change; proceed to SOTA tuning | `READY_FOR_IMPLEMENTATION` |
| B. Induction reuse / induction profile freezing | `NEEDS_REAL_DATA_EVIDENCE` |
| C. Lower-cost induction config as explicit strategy | `RESEARCH_CHANGE` |
| D. KMeans parameter strategy | `NEEDS_REAL_DATA_EVIDENCE` |
| E. MiniBatchKMeans as explicit alternative strategy | `RESEARCH_CHANGE` |
| F. `r_star` vectorization/layout optimization | `REJECT_FOR_NOW` |
| G. Cache IO/layout optimization | `NEEDS_REAL_DATA_EVIDENCE` |
| H. User-cluster-history build optimization | `REJECT_FOR_NOW` |

## Decision

Decision: `PROCEED_TO_TUNING_FRAMEWORK`

Rationale:

- ML1M data was available, so the decision is not blocked by missing real-data
  diagnostics.
- Cold/warm cache behavior was observed on the local ML1M profile.
- Cache reuse is clean for the alpha/lambda/target-epoch tuning case under the
  Step 18a cache-identity tests.
- The local cold path is split across cache write, user KMeans, induction fit,
  and item KMeans rather than one clearly dominant algorithmic stage.
- `r_star` is not a meaningful bottleneck in this local readout.
- Cache IO is visible, but cache-format or layout changes should wait for
  tuning access-pattern evidence.

Rejected implementation decisions:

- `IMPLEMENT_INDUCTION_ARTIFACT_CACHE`
- `IMPLEMENT_KMEANS_STRATEGY_EXPERIMENT`
- `IMPLEMENT_R_STAR_OPTIMIZATION`
- `IMPLEMENT_CACHE_IO_OPTIMIZATION`
- `INSUFFICIENT_REAL_DATA_FOR_ALGORITHM_CHANGE`

## What Was Not Changed

- No KMeans algorithm was changed.
- No MiniBatchKMeans or alternative clustering implementation was added.
- No KMeans default parameter was changed.
- No induction config default was changed.
- No CB model formula was changed.
- No CB kernel or hotpath file was changed.
- No model wrapper, registry, or runner dispatch was changed.
- No historical result artifact was overwritten.

## Tests/Gates

Focused checks already run for the Step 18c reporter extension:

- `ruff check scripts/profile_cluster_artifacts.py tests/unit/test_profile_cluster_artifacts_script.py` passed.
- `pytest tests/unit/test_profile_cluster_artifacts_script.py` passed: 9 passed.
- `python scripts/profile_cluster_artifacts.py --help` passed.
- `git diff --check` passed after documentation/report changes.

Reporter unit coverage includes:

- CLI help contract.
- invalid CLI config rejection.
- JSON/CSV output contract.
- custom output stem validation.
- synthetic summary CSV output.
- stage ranking descending order.

The local ML1M `cluster_stage_ranking.csv` was regenerated through the tested
`write_stage_ranking_csv(...)` helper.

Final Step 18c gates:

- `ruff check .` passed.
- `pytest tests/unit/test_profile_cluster_artifacts_script.py` passed: 9 passed.
- `pytest tests/unit/test_cluster_artifact_cache_reuse.py` passed: 9 passed.
- `pytest tests/integration/test_cluster_artifact_cache_smoke.py` passed: 1 passed.
- `pytest tests/unit/test_hotpath_coldpath_boundaries.py` passed: 13 passed.
- `pytest tests/unit` passed: 229 passed.
- `pytest tests/integration/test_unified_pipeline_smoke_all_models.py` passed: 1 passed.
- `pytest` passed: 304 passed, 2 skipped.
- `python scripts/profile_cluster_artifacts.py --help` passed.
- The final ML1M diagnostic command above passed and regenerated the local
  report artifacts.
- The requested claim-check command was run. It returned existing
  claim-boundary and claim-lock references, not a new Step 18c performance
  claim.

## Claim Boundary

No performance improvement claim is made.

Allowed conclusion:

- On this local diagnostic profile, `cluster_cache_write` was the largest
  Cluster-Artifact cold-build cost block.
- Cache reuse works in the tested contract.
- The local diagnostic supports a strategy decision for the next development
  step.

Disallowed conclusions:

- KMeans is optimized.
- Cluster induction is faster.
- MiniBatchKMeans is better.
- ML1M performance is generally improved.
- The local timing shares generalize across machines, datasets, or tuning
  profiles.

## Recommended Next Step

`19. SOTA Tuning Framework`

The tuning framework should preserve Cluster-Artifact cache reuse and continue
recording Cluster-Artifact profiling fields so later cache IO, induction, or
KMeans strategy work is based on repeated real tuning access patterns.
