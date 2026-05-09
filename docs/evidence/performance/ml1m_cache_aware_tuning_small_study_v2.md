# ML1M Cache-aware Tuning Small Study V2

## Branch

`ml1m-cache-aware-tuning-small-study-v2`

## Command

```powershell
python scripts/run_tuning_mini_study.py --search-space configs/experiments/tuning/active/ml1m_cb_svdpp_small_study_v2.yaml --output-dir artifacts/tuning --study-id ml1m_cb_svdpp_small_study_v2 --processed-manifest data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json --runtime-config configs/runtime/base.yaml --device-config configs/runtime/devices/local_u300_24gb.yaml --max-candidates 12 --overwrite
```

## Candidate Count

- Planned candidates: 12
- Executed candidates: 12
- Succeeded candidates: 12
- Dataset: ML1M
- Model: `cb_svdpp`
- Study artifacts: `artifacts/tuning/ml1m_cb_svdpp_small_study_v2/`

## Reuse Validation

- Planned cluster reuse groups: 1
- Reuse group: `cluster_rg_56b3da70a7c2`
- Candidate 1 cluster status: `miss`
- Candidate 1 user-cluster-history status: `miss`
- Follow-up cluster hits: 11 of 11
- Follow-up user-cluster-history hits: 11 of 11

## Ranking Top 5

| Rank | Candidate | alpha | lr | lambda_q | validation_rmse | validation_mae | fit_model_seconds | cluster_total_seconds | cache status |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `cand_0007_7111cfbbc311` | 0.2 | 0.01 | 0.025 | 0.9065428380888158 | 0.7170014349707604 | 46.84750299999723 | 0.3469156000064686 | hit/hit |
| 2 | `cand_0001_9b2f164d471c` | 0.1 | 0.0075 | 0.025 | 0.9074399459326296 | 0.7181076485725564 | 47.59733740000229 | 0.3596262999926694 | hit/hit |
| 3 | `cand_0011_77783cd2cfdc` | 0.25 | 0.01 | 0.04 | 0.908097455826485 | 0.71833967078778 | 47.80686599999899 | 0.28915739999501966 | hit/hit |
| 4 | `cand_0002_ddf20b2626fa` | 0.15 | 0.0075 | 0.025 | 0.9088536548028712 | 0.7193484501664578 | 49.1069453000091 | 0.2518876999965869 | hit/hit |
| 5 | `cand_0008_2f8b6a176a66` | 0.2 | 0.0075 | 0.015 | 0.9093105868309379 | 0.7197551605952758 | 46.65753020001284 | 0.3634425999916857 | hit/hit |

## Selected Candidate

- Candidate: `cand_0007_7111cfbbc311`
- Parameters: `alpha=0.2`, `learning_rate=0.01`, `lambda_q=0.025`
- Selected config: `artifacts/tuning/ml1m_cb_svdpp_small_study_v2/selected/selected_candidate_config.yaml`
- Selection reason: lowest validation RMSE among succeeded candidates; tie-breakers are validation MAE and fit time.

## V1 Context

The old 19e selected point was `alpha=0.25`, `learning_rate=0.01`, `lambda_q=0.04` with validation RMSE `0.9039941318794485`. In V2 the same parameter point ranked third with validation RMSE `0.908097455826485`. This comparison is context only; V1 ran before target/induction config separation and is superseded for bake-off selection.

## Decision

`SELECTED_CANDIDATE_READY_FOR_BAKEOFF`

## Gates

- `python scripts/run_tuning_mini_study.py ... --max-candidates 12 --overwrite`: passed
- Focused tests before the run: `pytest tests/unit/test_tuning_selection.py tests/unit/test_run_tuning_mini_study_script.py tests/unit/test_tuning_cluster_reuse_identity.py`: passed, 25 tests

## Limitations

- This is one local ML1M small tuning study, not a global search.
- Generated study artifacts remain local under `artifacts/` and are not versioned.
- No ML100K, synthetic, ML10M, ML20M, or Netflix-scale benchmark is used for this decision.

No claim is made that tuning is generally faster, that the selected candidate is globally optimal, or that these values generalize beyond this local ML1M small study.
