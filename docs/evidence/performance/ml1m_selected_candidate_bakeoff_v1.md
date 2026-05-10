# ML1M Selected Candidate Bake-off V1

## Branch

`ml1m-selected-candidate-bakeoff-v1`

## Compared Configs

| Role | Label | Config |
|---|---|---|
| Baseline | `baseline_stage0_transfer` | `configs/models/selected/ml1m/ml1m_cb_svdpp_stage0_transfer.yaml` |
| Candidate | `small_study_v2_candidate` | `configs/models/selected/ml1m/ml1m_cb_svdpp_small_study_v2_candidate.yaml` |

## Command

```powershell
python scripts/run_ml1m_selected_candidate_bakeoff.py --baseline-config configs/models/selected/ml1m/ml1m_cb_svdpp_stage0_transfer.yaml --candidate-config configs/models/selected/ml1m/ml1m_cb_svdpp_small_study_v2_candidate.yaml --processed-manifest data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json --runtime-config configs/runtime/base.yaml --device-config configs/runtime/devices/local_u300_24gb.yaml --output-dir artifacts/bakeoff/ml1m_selected_candidate_bakeoff_v1 --overwrite
```

## Results

| Label | validation_rmse | validation_mae | fit_model_seconds | total_wall_seconds | cluster_total_seconds | cache status |
|---|---:|---:|---:|---:|---:|---|
| `baseline_stage0_transfer` | 0.8579114910589735 | 0.672440913185854 | 264.2113745000097 | 313.56624359995476 | 14.523418400000082 | miss/miss |
| `small_study_v2_candidate` | 0.9065428380888158 | 0.7170014349707604 | 44.358329699985916 | 72.3161927999754 | 8.335115800000494 | miss/miss |

## Decision

`REJECT_SELECTED_CONFIG`

The candidate validation RMSE is higher than the baseline validation RMSE by `0.04863134702984229`. Selection uses validation metrics only; test metrics were not evaluated and were not used.

## Adopted Config Path

Not applicable. No final selected config was created.

## Gates

- `python scripts/run_ml1m_selected_candidate_bakeoff.py ...`: passed
- `bakeoff_results.csv`: `artifacts/bakeoff/ml1m_selected_candidate_bakeoff_v1/bakeoff_results.csv`
- `bakeoff_decision.json`: `artifacts/bakeoff/ml1m_selected_candidate_bakeoff_v1/bakeoff_decision.json`

## Limitations

- This is one local ML1M selected-candidate bake-off.
- No claim is made for ML10M, ML20M, Netflix-scale, or global optimality.
- No test metric was used for selection.
