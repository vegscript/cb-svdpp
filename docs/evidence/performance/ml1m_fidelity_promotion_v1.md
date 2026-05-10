# ML1M Fidelity Promotion V1

## Branch

`ml1m-fidelity-promotion-v1`

## Promoted Candidates

| Label | Source candidate | Config | alpha | learning_rate | lambda_q |
|---|---|---|---:|---:|---:|
| `promotion_p1` | `cand_0007_7111cfbbc311` | `configs/models/candidates/ml1m/ml1m_cb_svdpp_fidelity_promotion_p1.yaml` | 0.2 | 0.01 | 0.025 |
| `promotion_p2` | `cand_0001_9b2f164d471c` | `configs/models/candidates/ml1m/ml1m_cb_svdpp_fidelity_promotion_p2.yaml` | 0.1 | 0.0075 | 0.025 |
| `promotion_p3` | `cand_0011_77783cd2cfdc` | `configs/models/candidates/ml1m/ml1m_cb_svdpp_fidelity_promotion_p3.yaml` | 0.25 | 0.01 | 0.04 |

All promoted configs use baseline fidelity from
`configs/models/selected/ml1m/ml1m_cb_svdpp_stage0_transfer.yaml`:
`training.epochs=20`, `training.latent_dim=64`, and unchanged
`clustering.induction.*`.

## Commands

```powershell
python scripts/run_ml1m_fidelity_promotion.py `
  --baseline-config configs/models/selected/ml1m/ml1m_cb_svdpp_stage0_transfer.yaml `
  --promoted-config configs/models/candidates/ml1m/ml1m_cb_svdpp_fidelity_promotion_p1.yaml `
  --promoted-config configs/models/candidates/ml1m/ml1m_cb_svdpp_fidelity_promotion_p2.yaml `
  --promoted-config configs/models/candidates/ml1m/ml1m_cb_svdpp_fidelity_promotion_p3.yaml `
  --label promotion_p1 `
  --label promotion_p2 `
  --label promotion_p3 `
  --baseline-label baseline_stage0_transfer `
  --processed-manifest data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json `
  --runtime-config configs/runtime/base.yaml `
  --device-config configs/runtime/devices/local_u300_24gb.yaml `
  --output-dir artifacts/fidelity_promotion/ml1m_fidelity_promotion_v1 `
  --overwrite
```

## Results Table

| Rank | Label | Role | validation_rmse | validation_mae | fit_model_seconds | cluster_total_seconds | cache status |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `promotion_p1` | promoted | 0.853703596758597 | 0.6695381079447467 | 255.95572810000158 | 0.6602143000054639 | hit/hit |
| 2 | `baseline_stage0_transfer` | baseline | 0.8579114910589735 | 0.672440913185854 | 271.8543551999901 | 12.995302399998764 | miss/miss |
| 3 | `promotion_p2` | promoted | 0.8579114910589735 | 0.672440913185854 | 250.0020692999824 | 0.9805498999776319 | hit/hit |
| 4 | `promotion_p3` | promoted | 0.8605166711730309 | 0.6764545254374795 | 253.77147229999537 | 0.574858400010271 | hit/hit |

Artifacts:

- `artifacts/fidelity_promotion/ml1m_fidelity_promotion_v1/promotion_results.csv`
- `artifacts/fidelity_promotion/ml1m_fidelity_promotion_v1/promotion_decision.json`

## Decision

`PROMOTED_CANDIDATE_READY_FOR_FINAL_BAKEOFF`

`promotion_p1` had lower validation RMSE than the baseline in this local ML1M
fidelity-promotion run. The selected config written for the next step is:

`configs/models/selected/ml1m/ml1m_cb_svdpp_fidelity_promotion_selected.yaml`

## Gates

- `ruff check scripts/run_ml1m_fidelity_promotion.py`: passed
- `python -m py_compile scripts/run_ml1m_fidelity_promotion.py`: passed
- `python scripts/run_ml1m_fidelity_promotion.py --help`: passed with Python 3.12
- `ruff check .`: passed
- `pytest tests/unit/test_ml1m_fidelity_promotion.py`: passed, 7 tests
- `pytest tests/unit`: passed, 340 tests
- `pytest tests/integration/test_unified_pipeline_smoke_all_models.py`: passed
- `pytest`: passed, 416 tests, 2 skipped
- `python scripts/run_ml1m_fidelity_promotion.py --help`: passed
- Real ML1M promotion run with `--candidate-config`: passed, 4 runs
- Claim-check found only existing claim-boundary and claim-lock references.

## Limitations

- This is one local ML1M fidelity-promotion study.
- No ML100K, synthetic, ML10M, ML20M, or Netflix-scale benchmark was used for the decision.
- The first run built the cache; follow-up promoted configs reused cluster artifacts.
- Generated artifacts remain local under `artifacts/`.

## Claim Boundary

This is a local ML1M fidelity-promotion study.
No claim is made for ML10M, ML20M, Netflix-scale, or global optimality.
No test metric was used for selection.
