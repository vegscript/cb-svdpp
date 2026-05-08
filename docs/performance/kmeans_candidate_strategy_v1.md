# KMeans Candidate Strategy V1

## Status

Draft diagnostic plan for Step 18c.

This document defines local diagnostic runs for ranking Cluster-Induction and
Cluster-Artifact costs. It does not report runtime results, implement a new
KMeans strategy, change defaults, or make a performance claim.

## Inputs Reviewed

- `docs/evidence/performance/cluster_artifact_kmeans_audit_v1.md`
- `docs/performance/cluster_artifact_kmeans_audit_v1.md`
- `src/recsys_lab/clustering/cache.py`
- `src/recsys_lab/clustering/latent_kmeans.py`
- `scripts/profile_cluster_artifacts.py`
- `tests/unit/test_cluster_artifact_cache_reuse.py`
- `tests/integration/test_cluster_artifact_cache_smoke.py`

## Available Profiler Contract

The Step 18a reporter writes one JSON and one CSV payload with one row per
profile kind and repeat:

- `cluster_artifacts`
- `user_cluster_history`

The current reporter supports `benchmark_random_v1` only.

Cluster artifact stages:

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

User-cluster-history stages:

- `user_cluster_history_total_seconds`
- `user_cluster_history_cache_read_seconds`
- `user_cluster_history_cache_write_seconds`
- `user_cluster_history_build_seconds`
- `user_cluster_history_validation_seconds`

Cache statuses are reported as profile metadata:

- `cluster_cache_status`
- `user_cluster_history_cache_status`

Timings, model labels, and cache statuses are report metadata, not cache
identity fields.

## Local Data Availability

The following processed manifests are present locally and use
`benchmark_random_v1`:

- `data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- `data/processed/ml100k/ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json`
- `data/processed/ml10m/ml10m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- `data/processed/ml20m/ml20m_benchmark_random_v1_explicit_v1_float32_manifest.json`

Step 18c must not start ML10M, ML20M, or other large diagnostics
automatically. Those are follow-up candidates only after the ML1M readout is
understood.

## P1 ML1M Diagnostic Plan

Minimum required diagnostic:

- one ML1M cold-cache Cluster-Artifact profile
- one ML1M warm-cache Cluster-Artifact profile

Selected profile:

- model config:
  `configs/models/archive/tuned/ml1m_cb_svdpp_stage0_probe_e003.yaml`
- processed manifest:
  `data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model: `cb_svdpp`
- split family: `benchmark_random_v1`
- train ratio: `0.8`
- validation ratio: `0.1`
- split seed: `1`
- model seed: `1`

Selection rationale:

- It uses the real local ML1M processed dataset and the supported
  `benchmark_random_v1` split family.
- It is a CB-SVD++ cluster profile, so it exercises the Cluster-Artifact and
  user-cluster-history lifecycle.
- It keeps the transferred ML1M cluster scale (`80` user clusters and `80`
  item clusters) while reducing induction epochs to `3`, which is appropriate
  for local diagnosis.
- It avoids using the full 20-epoch transfer profile before the profiler has
  ranked the stage costs.

The cold/warm diagnostic should use an isolated local cache root so repeat 1
is a cache miss/build and repeat 2 is a cache hit/load. The runtime override is
local output under `artifacts/` and is not versioned.

Example PowerShell setup:

```powershell
$diagnosticRoot = "artifacts/reports/kmeans_candidate_strategy_v1/ml1m_cb_svdpp_probe_e003"
$cacheRoot = "artifacts/local/kmeans_candidate_strategy_v1/ml1m_cb_svdpp_probe_e003"
New-Item -ItemType Directory -Force $diagnosticRoot | Out-Null
@"
metadata:
  status: local_diagnostic
  owner: local
  purpose: kmeans_candidate_strategy_v1

runtime:
  project_slug: recsys_paper_lab
  default_device_profile: local_u300_24gb
  default_precision_profile: performance_float32
  setup_tool: uv
  data_root: data
  artifact_root: artifacts
  cache_root: $cacheRoot
  naming_contract: docs/naming_conventions.md
  manifest_contract: docs/manifest_contract.md

threading_defaults:
  omp_num_threads: 6
  blas_threads: 6

precision_profiles:
  performance_float32:
    dtype: float32
    purpose: local_diagnostic
  reference_float64:
    dtype: float64
    purpose: numerical_validation
"@ | Set-Content -Encoding UTF8 "$diagnosticRoot/runtime_local_diagnostic.yaml"
```

Example profile command:

```powershell
python scripts/profile_cluster_artifacts.py `
  --processed-manifest data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json `
  --model-config configs/models/archive/tuned/ml1m_cb_svdpp_stage0_probe_e003.yaml `
  --runtime-config artifacts/reports/kmeans_candidate_strategy_v1/ml1m_cb_svdpp_probe_e003/runtime_local_diagnostic.yaml `
  --model cb_svdpp `
  --split-family benchmark_random_v1 `
  --train-ratio 0.8 `
  --validation-ratio 0.1 `
  --split-seed 1 `
  --model-seed 1 `
  --repeats 2 `
  --output-dir artifacts/reports/kmeans_candidate_strategy_v1 `
  --output-stem cluster_artifact_profile_ml1m
```

Expected report files:

- `artifacts/reports/kmeans_candidate_strategy_v1/cluster_artifact_profile_ml1m.json`
- `artifacts/reports/kmeans_candidate_strategy_v1/cluster_artifact_profile_ml1m.csv`

Expected status pattern for a fresh isolated cache root:

- repeat 1 `cluster_artifacts`: `cluster_cache_status == "miss"`
- repeat 2 `cluster_artifacts`: `cluster_cache_status == "hit"`
- repeat 1 `user_cluster_history`: `user_cluster_history_cache_status == "miss"`
- repeat 2 `user_cluster_history`: `user_cluster_history_cache_status == "hit"`

If the first repeat is already a hit, the cache root was not fresh. Use a new
local cache-root suffix instead of deleting shared cache content.

## P2 Optional ML100K Sanity

Optional quick sanity profile:

- model config:
  `configs/models/archive/development/ml100k_cb_svdpp_stage_profile_smoke.yaml`
- processed manifest:
  `data/processed/ml100k/ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model: `cb_svdpp`

This is only a profiler sanity readout. It is not a substitute for the ML1M
diagnostic because it uses a development smoke profile (`latent_dim=16`,
`epochs=1`, `8x8` clusters, `kmeans_n_init=2`).

## P3 Optional ML10M Follow-Up

ML10M processed data is locally present, and a probe profile exists:

- `configs/models/archive/tuned/ml10m_cb_svdpp_stage0_probe_e001.yaml`
- `data/processed/ml10m/ml10m_benchmark_random_v1_explicit_v1_float32_manifest.json`

Do not run this automatically in Step 18c. It is a follow-up only if the ML1M
diagnostic shows that a larger-scale readout is necessary and local runtime is
acceptable.

## Ranking Method

Use the ML1M cold-cache row to rank build-path costs:

- `induction_fit_seconds`
- `induction_predict_seconds`
- `induction_train_rmse_seconds`
- `user_kmeans_seconds`
- `item_kmeans_seconds`
- `r_star_seconds`
- `cluster_artifact_validation_seconds`
- `cluster_cache_write_seconds`
- `user_cluster_history_build_seconds`
- `user_cluster_history_validation_seconds`
- `user_cluster_history_cache_write_seconds`

Use the ML1M warm-cache row to rank load-path costs:

- `cluster_cache_read_seconds`
- `cluster_artifact_validation_seconds`
- `user_cluster_history_cache_read_seconds`
- `user_cluster_history_validation_seconds`

Report both absolute seconds and local percentage of the relevant total:

- `cluster_total_seconds` for Cluster-Artifact rows
- `user_cluster_history_total_seconds` for user-cluster-history rows

Do not interpret the local percentages as portable performance claims. They are
diagnostic ranking evidence for the laptop profile only.

## ML1M Diagnostic Readout

Artifacts produced locally:

- `artifacts/reports/kmeans_candidate_strategy_v1/cluster_artifact_profile_ml1m.json`
- `artifacts/reports/kmeans_candidate_strategy_v1/cluster_artifact_profile_ml1m.csv`
- `artifacts/reports/kmeans_candidate_strategy_v1/cluster_artifact_profile_ml1m_summary.csv`
- `artifacts/reports/kmeans_candidate_strategy_v1/cluster_stage_ranking.csv`

Cold/warm status:

- cold repeat: cluster cache `miss`, user-cluster-history cache `miss`
- warm repeat: cluster cache `hit`, user-cluster-history cache `hit`

Local diagnostic totals:

- `cold_cluster_total_seconds`: `8.638122900`
- `warm_cluster_total_seconds`: `0.272404600`
- `cold_vs_warm_ratio`: `31.710635210`

Cold cluster-stage shares:

- `cluster_cache_write_share_of_cold_total`: `0.359808564`
- `user_kmeans_share_of_cold_total`: `0.230664697`
- `induction_fit_share_of_cold_total`: `0.213262479`
- `item_kmeans_share_of_cold_total`: `0.128295581`
- `induction_predict_share_of_cold_total`: `0.055204945`
- `r_star_share_of_cold_total`: `0.004075029`

Warm cluster-stage share:

- `cluster_cache_read_share_of_warm_total`: `0.675378463`

User-cluster-history shares:

- `user_cluster_history_build_share`: `0.136577741`
- `user_cluster_history_cache_read_share`: `0.777945681`

These values are a single local ML1M diagnostic readout on the laptop profile.
They are not a stable runtime generalization.

## Strategy Candidate Classification

Classification labels:

- `READY_FOR_IMPLEMENTATION`: evidence is sufficient for a narrow next
  engineering slice.
- `NEEDS_REAL_DATA_EVIDENCE`: plausible, but requires more real-data quality,
  stability, or before/after evidence before implementation.
- `DEFER_TO_TUNING_FRAMEWORK`: better handled as part of the upcoming tuning
  system instead of a standalone optimization.
- `REJECT_FOR_NOW`: not supported by the current diagnostic readout.
- `RESEARCH_CHANGE`: changes modeling behavior or requires a separate research
  protocol.

| Candidate | Classification | Diagnostic basis | Next handling |
| --- | --- | --- | --- |
| A. No algorithm change; proceed to SOTA tuning | `READY_FOR_IMPLEMENTATION` | The cold path is measurable, warm cache works, and cache reuse removes induction/KMeans/r_star work on repeat tuning paths. The cold readout is split across cache write, user KMeans, induction fit, and item KMeans instead of one single algorithmic bottleneck. | Proceed to the tuning framework with the current Cluster-Artifact cache contract. Keep Cluster-Artifact profiling enabled for later tuning evidence. |
| B. Induction reuse / induction profile freezing | `NEEDS_REAL_DATA_EVIDENCE` | `induction_fit_seconds` is material locally (`0.213262479` of cold cluster total) but does not dominate alone. | Do not implement now. Revisit if tuning generates many distinct Cluster-Artifact keys with the same induction config, or if larger datasets show induction fit dominance. |
| C. Lower-cost induction config as explicit strategy | `RESEARCH_CHANGE` | Reducing induction epochs or latent dimension may reduce cold-build cost, but it changes the induced factors and therefore cluster quality. | Treat as an explicit candidate strategy requiring RMSE/MAE, cluster stability, and downstream CB quality evidence. Do not silently change defaults. |
| D. KMeans parameter strategy | `NEEDS_REAL_DATA_EVIDENCE` | User and item KMeans together are material locally (`0.230664697` + `0.128295581` of cold cluster total), but KMeans does not dominate alone and `kmeans_n_init` affects stability. | Do not change defaults now. Evaluate candidate grids for `kmeans_n_init`, seed stability, and algorithm choices inside the tuning framework or a dedicated 18c follow-up. |
| E. MiniBatchKMeans as explicit alternative strategy | `RESEARCH_CHANGE` | MiniBatchKMeans would change the clustering algorithm and likely cluster assignments. The current ML1M readout does not justify a silent replacement. | Keep as a named alternative strategy only. It needs cluster-quality, downstream RMSE/MAE, and stability evidence before any product path. |
| F. r_star vectorization/layout optimization | `REJECT_FOR_NOW` | `r_star_seconds` is small in the local readout (`0.004075029` of cold cluster total). | No implementation slice now. Keep as monitoring only if larger datasets show a different profile. |
| G. Cache IO/layout optimization | `NEEDS_REAL_DATA_EVIDENCE` | Cold cluster cache write is the largest local stage (`0.359808564` of cold cluster total), and warm cluster cache read dominates warm total (`0.675378463`). User-cluster-history cache IO also dominates its own warm path. | Investigate after tuning needs are clear. Candidate checks: `.npy` layout, number of small file reads/writes, mmap behavior, and avoiding compression. Do not change cache format in Step 18c. |
| H. User-cluster-history build optimization | `REJECT_FOR_NOW` | User-cluster-history build is visible but not dominant locally (`0.136577741` of user-cluster-history cold total), and its total is much smaller than cold Cluster-Artifact build. | No implementation slice now. Keep History Data Layout V1 as the current contract and monitor on larger data. |

Recommended Step 18c decision:

`Candidate A` is the only candidate ready for the mainline next step. The
diagnostic does not justify a KMeans algorithm replacement or induction-config
change before SOTA tuning. Keep `B`, `D`, and `G` as evidence-backed follow-up
questions; classify `C` and `E` as research changes; reject `F` and `H` for now
based on this local readout.

## Decision

Decision: `PROCEED_TO_TUNING_FRAMEWORK`

Rationale:

- ML1M data and configs were available, so the decision is not blocked by
  missing real-data diagnostics.
- The cold/warm diagnostic produced the required cache paths: cold cluster
  artifacts and user-cluster-history were cache misses, while the second repeat
  loaded both from cache.
- Cache reuse is the relevant protection for alpha/lambda/target-epoch tuning:
  those target-model changes do not require rebuilding Cluster-Artifacts under
  the Step 18a cache-identity contract.
- The cold ML1M readout does not show one algorithmic stage dominating enough
  to justify an immediate KMeans or induction strategy implementation. The
  largest local cold stages were cache write, user KMeans, induction fit, and
  item KMeans.
- `r_star` and user-cluster-history build were not material enough in this
  local readout to justify immediate implementation work.
- Cache IO is visible and should remain a follow-up question, but cache-format
  or layout changes need more evidence from actual tuning access patterns.

Rejected 18c decisions:

- `IMPLEMENT_INDUCTION_ARTIFACT_CACHE`: induction fit is material but not a
  standalone dominant bottleneck in this local ML1M diagnostic.
- `IMPLEMENT_KMEANS_STRATEGY_EXPERIMENT`: KMeans work is material, but a
  strategy experiment would affect cluster stability and needs quality evidence.
- `IMPLEMENT_R_STAR_OPTIMIZATION`: `r_star_seconds` is small in this readout.
- `IMPLEMENT_CACHE_IO_OPTIMIZATION`: cache IO is visible, but changing cache
  layout before observing tuning access patterns is premature.
- `INSUFFICIENT_REAL_DATA_FOR_ALGORITHM_CHANGE`: not applicable because ML1M
  processed data and a real ML1M diagnostic profile were available.

No algorithm change should be implemented from Step 18c.

## Decision Questions

The Phase 3+ evidence should answer:

- Is the dominant cold-cache cost induction fit, KMeans, `r_star`, cache IO, or
  user-cluster-history build?
- Is the warm-cache path dominated by cache read, validation, or another stage?
- Is cache reuse sufficiently clean for alpha/lambda/epoch tuning on ML1M?
- Is there a remaining cache-hardening blocker that warrants `18b`?
- If no cache blocker remains, should the next implementation work target
  induction cost, KMeans candidate strategy, `r_star`, or defer algorithmic
  changes?

## Claim Boundary

This plan permits only local diagnostic statements after the commands are run.
It does not permit claims that KMeans is optimized, that one algorithm is
faster, or that any result is portable across machines or datasets.
