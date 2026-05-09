# Target / Induction Config Separation V1

## Branch

`target-induction-config-separation-v1`

## Problem

Step 19e planned all inner-loop `cb_svdpp` candidates into one cluster reuse group, but real ML1M runs rebuilt cluster artifacts for several candidates. The planned reuse contract did not match the actual cluster cache identity.

## Root Cause

CB cluster induction used a `BiasedMFConfig` derived from target model training parameters. Changing target `training.learning_rate` or `training.lambda_q` therefore changed the induction config inside the real cluster cache identity.

## Code / Config Change

- CB model profiles now support explicit `clustering.induction`.
- Unified CB-SVD++ and CB-ASVD++ induction config resolution uses `clustering.induction`.
- The unified runner passes the validated model profile into induction config resolution.
- The planner preserves `clustering.induction.*` coordinates in cluster reuse group identity, while excluding target-only coordinates such as `training.learning_rate` and `training.lambda_q`.
- The ML1M mini/small-study base config now contains a stable explicit induction config.
- The ML1M small-study search space varies target training fields only and does not target `clustering.induction.*`.

## Tests

- `pytest tests/unit/test_cb_induction_config_resolution.py tests/unit/test_tuning_cluster_reuse_identity.py tests/unit/test_tuning_candidates.py`: passed, 32 tests
- `pytest tests/unit/test_model_framework.py tests/integration/test_active_tuning_config_validation.py tests/integration/test_unified_pipeline_smoke_all_models.py`: passed, 36 tests
- `pytest tests/unit/test_tuning_candidates.py::test_ml1m_small_study_target_only_variation_uses_one_cluster_reuse_group tests/integration/test_active_tuning_config_validation.py`: passed, 2 tests

## ML1M Two-Candidate Reuse Validation

Command:

```powershell
python scripts/run_tuning_mini_study.py --search-space configs/experiments/tuning/active/ml1m_cb_svdpp_target_induction_reuse_validation_v1.yaml --output-dir artifacts/tuning --study-id ml1m_target_induction_separation_validation_v1 --processed-manifest data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json --runtime-config configs/runtime/base.yaml --device-config configs/runtime/devices/local_u300_24gb.yaml --max-candidates 2 --overwrite
```

| Candidate | alpha | learning_rate | lambda_q | validation_rmse | validation_mae | fit_model_seconds | cluster_total_seconds | cluster_cache_status | user_cluster_history_cache_status | reuse_group |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `cand_0000_62990cda9b0a` | 0.2 | 0.0075 | 0.025 | 0.9095895670578061 | 0.7199936929078307 | 124.84118509999826 | 8.245795700000599 | miss | miss | `cluster_rg_56b3da70a7c2` |
| `cand_0001_b73c33122a6b` | 0.2 | 0.01 | 0.04 | 0.9073643264979903 | 0.7177134630541534 | 121.66307949999464 | 0.43299160001333803 | hit | hit | `cluster_rg_56b3da70a7c2` |

Both candidates used ML1M, `cb_svdpp`, the same explicit `clustering.induction` config, and the same `cluster_reuse_group_id`. Candidate 1 built the cluster artifacts in the isolated study cache; candidate 2 loaded them.

## Decision

`TARGET_INDUCTION_SEPARATION_CONFIRMED`

No performance or quality claim is made from this two-candidate validation.
