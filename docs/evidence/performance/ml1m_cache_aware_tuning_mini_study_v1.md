# ML1M Cache-aware Tuning Mini Study V1

## Branch

`ml1m-cache-aware-tuning-mini-study-v1`

## Goal

Run a small local ML1M cache-aware tuning mini study to verify that planned tuning candidates can execute sequentially, share one cluster artifact reuse group, and record cold/warm cache behavior in Study reports.

## Search Space Used

`configs/experiments/tuning/active/ml1m_cb_svdpp_cache_aware_mini_study_v1.yaml`

- Dataset: `ml1m`
- Split family: `benchmark_random_v1`
- Model: `cb_svdpp`
- Candidates: 2
- Candidate coordinate varied: `clustering.alpha`
- Alpha values: `0.2`, `0.5`
- Base model config: `configs/models/selected/ml1m/ml1m_cb_svdpp_mini_study_e003.yaml`

## Execution Command

```bash
python scripts/run_tuning_mini_study.py \
  --search-space configs/experiments/tuning/active/ml1m_cb_svdpp_cache_aware_mini_study_v1.yaml \
  --output-dir artifacts/tuning \
  --study-id ml1m_cb_svdpp_cache_aware_mini_study_v1 \
  --processed-manifest data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json \
  --runtime-config configs/runtime/base.yaml \
  --device-config configs/runtime/devices/local_u300_24gb.yaml \
  --max-candidates 2 \
  --overwrite
```

The local run used the repository Python 3.12 environment that has the project dependencies installed.

Artifacts were written locally under:

`artifacts/tuning/ml1m_cb_svdpp_cache_aware_mini_study_v1/`

## Candidates Executed

| Candidate | Alpha | Execution status | Cluster reuse group |
|---|---:|---|---|
| `cand_0000_c8b577e70ac7` | 0.2 | `succeeded` | `cluster_rg_56b3da70a7c2` |
| `cand_0001_04309f83d373` | 0.5 | `succeeded` | `cluster_rg_56b3da70a7c2` |

## Cache Reuse Result

Cold/warm cluster reuse was observed within the local ML1M mini study:

| Candidate | Cluster cache | User cluster history cache | Notes |
|---|---|---|---|
| `cand_0000_c8b577e70ac7` | `miss` | `miss` | `cold_cache_build` |
| `cand_0001_04309f83d373` | `hit` | `hit` | `warm_cache_hit` |

`mini_study_summary.json` reports:

- `cache_reuse_observed: true`
- `cold_candidate_id: cand_0000_c8b577e70ac7`
- `warm_candidate_ids: [cand_0001_04309f83d373]`
- `candidate_count: 2`
- `executed_candidate_count: 2`
- `cluster_reuse_group_count: 1`

## Metrics Table

| Candidate | Alpha | Validation RMSE | Validation MAE |
|---|---:|---:|---:|
| `cand_0000_c8b577e70ac7` | 0.2 | 0.9095895670578061 | 0.7199936929078307 |
| `cand_0001_04309f83d373` | 0.5 | 0.9110140941162271 | 0.7210676646745169 |

## Runtime/Cache Table

| Candidate | fit_model_seconds | total_wall_seconds | cluster_total_seconds | Cluster cache | User cluster history cache |
|---|---:|---:|---:|---|---|
| `cand_0000_c8b577e70ac7` | 52.86696970000048 | 92.51553860002605 | 9.060479900013888 | `miss` | `miss` |
| `cand_0001_04309f83d373` | 52.970114199997624 | 61.22787599997537 | 0.337670299995807 | `hit` | `hit` |

## Tests/Gates

Phase 10 gates:

- `ruff check .`: passed
- Focused tuning and boundary tests: 81 passed
- `pytest tests/unit`: 297 passed
- `pytest tests/integration/test_unified_pipeline_smoke_all_models.py`: 1 passed
- `pytest`: 373 passed, 2 skipped
- `python scripts/run_tuning_mini_study.py --help`: passed
- Local ML1M mini-study command listed above: 2 candidates succeeded
- Claim-check returned only existing claim-boundary or negative-claim references.

## Limitations

- This is one local laptop mini study with two candidates.
- It proves execution wiring and cold/warm cache reuse for this Study contract only.
- It is not a tuning campaign and does not establish model ranking.
- Generated Study artifacts remain local and are not versioned.

## Claim Boundary

No claim is made that tuning is generally faster, that one candidate is better, that ML1M results generalize, or that large-scale behavior has been reached. The allowed readout is limited to: this local ML1M mini study observed cold-to-warm cluster reuse, collected candidate metrics, and updated Study reports for sequential candidate execution.

## Decision

`CACHE_REUSE_CONFIRMED_PROCEED_TO_SMALL_TUNING`
