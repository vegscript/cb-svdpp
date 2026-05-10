# ML1M Final Bakeoff V1

## Branch

`ml1m-final-bakeoff-v1`

## Compared Configs

| Label | Config |
|---|---|
| `baseline_stage0_transfer` | `configs/models/selected/ml1m/ml1m_cb_svdpp_stage0_transfer.yaml` |
| `fidelity_promotion_selected` | `configs/models/selected/ml1m/ml1m_cb_svdpp_fidelity_promotion_selected.yaml` |

Both configs use `cb_svdpp`, ML1M, `training.epochs=20`, `training.latent_dim=64`,
and explicit `clustering.induction.*` with `epochs=20`.

## Command

```powershell
python scripts/run_ml1m_final_bakeoff.py `
  --baseline-config configs/models/selected/ml1m/ml1m_cb_svdpp_stage0_transfer.yaml `
  --selected-config configs/models/selected/ml1m/ml1m_cb_svdpp_fidelity_promotion_selected.yaml `
  --processed-manifest data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json `
  --runtime-config configs/runtime/base.yaml `
  --device-config configs/runtime/devices/local_u300_24gb.yaml `
  --output-dir artifacts/final_bakeoff/ml1m_final_bakeoff_v1 `
  --overwrite
```

## Results Table

| Label | validation_rmse | validation_mae | fit_model_seconds | total_wall_seconds | cluster_total_seconds | cache status |
|---|---:|---:|---:|---:|---:|---|
| `baseline_stage0_transfer` | 0.8579114910589735 | 0.672440913185854 | 267.4543166999938 | 316.2147137999418 | 14.715487399982521 | miss/miss |
| `fidelity_promotion_selected` | 0.853703596758597 | 0.6695381079447467 | 269.50239099998726 | 278.32886579993647 | 0.4742633999849204 | hit/hit |

Artifacts:

- `artifacts/final_bakeoff/ml1m_final_bakeoff_v1/final_bakeoff_results.csv`
- `artifacts/final_bakeoff/ml1m_final_bakeoff_v1/final_bakeoff_decision.json`

## Decision

`ADOPT_PROMOTED_CONFIG`

The selected config had lower validation RMSE and validation MAE than the
baseline in this local ML1M final bakeoff. No test metric was used for
selection.

## Config Status

`configs/models/selected/ml1m/ml1m_cb_svdpp_fidelity_promotion_selected.yaml`
remains the selected ML1M CB-SVD++ config.

Its metadata provenance includes:

- `final_bakeoff: docs/evidence/performance/ml1m_final_bakeoff_v1.md`
- `decision: ADOPT_PROMOTED_CONFIG`

## Gates

- Phase-local script run: passed
- `ruff check scripts/run_ml1m_final_bakeoff.py`: passed
- `pytest tests/unit/test_ml1m_final_bakeoff.py`: passed, 1 test
- `ruff check .`: passed
- `pytest tests/unit/test_ml1m_final_bakeoff.py`: passed, 7 tests
- `pytest tests/unit`: passed, 347 tests
- `pytest tests/integration/test_unified_pipeline_smoke_all_models.py`: passed
- `pytest`: passed, 423 tests, 2 skipped
- `python scripts/run_ml1m_final_bakeoff.py --help`: passed
- Real ML1M final bakeoff: passed, 2 runs
- Claim-check found only existing claim-boundary and claim-lock references.

## Limitations

- This is one local ML1M final bakeoff.
- No ML100K, synthetic, ML10M, ML20M, or Netflix-scale benchmark was used for selection.
- Cache status differs between the two runs, so `total_wall_seconds` is context only.

## Claim Boundary

This is a local ML1M final bakeoff.
No claim is made for ML10M, ML20M, Netflix-scale, or global optimality.
No test metric was used for selection.
Runtime interpretation is limited to same-run conditions; no broad speedup claim.
