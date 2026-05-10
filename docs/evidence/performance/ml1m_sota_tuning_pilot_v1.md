# ML1M SOTA Tuning Pilot V1

## Branch

`ml1m-sota-tuning-pilot-v1`

## Pilot Scope

- Dataset: ML1M
- Split: `benchmark_random_v1`
- Model: `cb_svdpp`
- Base config: `configs/models/selected/ml1m/ml1m_cb_svdpp_fidelity_promotion_selected.yaml`
- Execution: sequential only
- Stages: `16 -> 4 -> 1`

## Search Space

Active search space:

`configs/experiments/tuning/active/ml1m_cb_svdpp_sota_pilot_v1.yaml`

Generator:

- `latin_hypercube`
- seed: `11`
- candidates: `16`

Varied target parameters:

- `clustering.alpha`
- `training.learning_rate`
- `training.lambda_p`
- `training.lambda_q`
- `training.lambda_y`
- `training.lambda_pC`
- `training.lambda_qC`
- `training.lambda_yC`

Cluster and induction parameters were not varied.

## Stage Results Summary

| Stage | Planned | Succeeded | Cache pattern |
| --- | ---: | ---: | --- |
| Stage 1, epochs 3 | 16 | 16 | 1 miss/miss, 15 hit/hit |
| Stage 2, epochs 10 | 4 | 4 | 4 hit/hit |
| Stage 3, epochs 20 | 1 | 1 | 1 hit/hit |

All executed candidates used cluster reuse group `cluster_rg_8edb24092572`.

Reports:

- `artifacts/tuning/ml1m_cb_svdpp_sota_pilot_v1/reports/stage1_results.csv`
- `artifacts/tuning/ml1m_cb_svdpp_sota_pilot_v1/reports/stage2_results.csv`
- `artifacts/tuning/ml1m_cb_svdpp_sota_pilot_v1/reports/stage3_results.csv`
- `artifacts/tuning/ml1m_cb_svdpp_sota_pilot_v1/reports/sota_pilot_summary.csv`
- `artifacts/tuning/ml1m_cb_svdpp_sota_pilot_v1/reports/sota_pilot_decision.json`

## Final Candidate

- Candidate id: `prom_0001_3c1f3956f93b`
- Config: `artifacts/tuning/ml1m_cb_svdpp_sota_pilot_v1/promotions/stage3_full_fidelity/candidates/prom_0001_3c1f3956f93b/candidate_config.yaml`
- Source Stage-1 candidate: `cand_0015_1ef101956b4f`
- `alpha`: `0.26136005806237755`
- `learning_rate`: `0.01940222593411329`
- `lambda_p`: `0.008823599142678131`
- `lambda_q`: `0.038329613132722415`
- `lambda_y`: `0.047185732597354674`
- validation RMSE: `0.8555598728153794`
- validation MAE: `0.6689958129042556`

## Incumbent Comparison Context

Incumbent config:

`configs/models/selected/ml1m/ml1m_cb_svdpp_fidelity_promotion_selected.yaml`

Known 19h incumbent values:

- validation RMSE: `0.853703596758597`
- validation MAE: `0.6695381079447467`

Pilot final candidate deltas:

- RMSE delta vs incumbent: `+0.0018562760567824022`
- MAE delta vs incumbent: `-0.0005422950404910809`

## Decision

`SOTA_PILOT_COMPLETED_INCUMBENT_STILL_REFERENCE`

The staged pilot completed technically, but the final candidate did not beat the
incumbent on validation RMSE. No selected config was written.

## Gates

Completed before or during this step:

```bash
ruff check .
pytest tests/unit/test_run_sota_tuning_pilot_script.py tests/unit/test_tuning_staged_planner.py tests/unit/test_plan_sota_tuning_study_script.py
pytest tests/integration/test_active_tuning_config_validation.py
pytest tests/unit
pytest tests/integration/test_unified_pipeline_smoke_all_models.py
pytest
py -3.12 scripts/run_sota_tuning_pilot.py --help
```

`tests/unit/test_sota_tuning_pilot_results.py` was not present; the pilot
result contract tests live in `tests/unit/test_run_sota_tuning_pilot_script.py`.
The local default `python` points at Python 3.10 without the project
dependencies, so the script smoke and ML1M run used the Python 3.12 project
environment.

Real ML1M pilot command:

```bash
py -3.12 scripts/run_sota_tuning_pilot.py --search-space configs/experiments/tuning/active/ml1m_cb_svdpp_sota_pilot_v1.yaml --output-dir artifacts/tuning --study-id ml1m_cb_svdpp_sota_pilot_v1 --processed-manifest data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json --runtime-config configs/runtime/base.yaml --device-config configs/runtime/devices/local_u300_24gb.yaml --overwrite
```

Claim check: the requested repository-wide claim-regex check only returned
existing claim-boundary and governance references.

## Limitations

- This was one local ML1M staged pilot.
- No ML10M, ML20M, Netflix-scale, or global optimality claim is made.
- No test metric was used for promotion or selection.
- No broad runtime speedup claim is made.
- The pilot does not adopt a new selected config.

## Claim Boundary

This is a local ML1M staged SOTA tuning pilot. No claim is made for ML10M,
ML20M, Netflix-scale, or global optimality. No test metric was used for
promotion or selection. No broad runtime speedup claim is made.
