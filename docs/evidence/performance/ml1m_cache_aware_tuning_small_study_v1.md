# ML1M Cache-aware Tuning Small Study V1

## Branch

`ml1m-cache-aware-tuning-small-study-v1`

## Command

```powershell
python scripts/run_tuning_mini_study.py --search-space configs/experiments/tuning/active/ml1m_cb_svdpp_small_study_v1.yaml --output-dir artifacts/tuning --study-id ml1m_cb_svdpp_small_study_v1 --processed-manifest data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json --runtime-config configs/runtime/base.yaml --device-config configs/runtime/devices/local_u300_24gb.yaml --max-candidates 12 --overwrite
```

## Candidate Count

- Planned candidates: 12
- Executed candidates: 12
- Succeeded candidates: 12
- Dataset: ML1M
- Model: `cb_svdpp`
- Study artifacts: `artifacts/tuning/ml1m_cb_svdpp_small_study_v1/`

## Ranking Table

| Rank | Candidate | alpha | lr | validation_rmse | validation_mae | fit_model_seconds | cluster_total_seconds | cache status | selected |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `cand_0011_3f9308e00fb3` | 0.25 | 0.01 | 0.9039941318794485 | 0.7147700425022206 | 52.787563400008366 | 8.092954100007773 | miss/miss | true |
| 2 | `cand_0007_52163c37bf3f` | 0.2 | 0.01 | 0.9045566875151262 | 0.7152891391218138 | 53.37537500000326 | 8.270614099994418 | miss/miss | false |
| 3 | `cand_0001_5845c4cd2a90` | 0.1 | 0.0075 | 0.9074399459326296 | 0.7181076485725564 | 74.21376670000609 | 0.3053900000086287 | hit/hit | false |
| 4 | `cand_0009_8c0588a5cdff` | 0.2 | 0.0075 | 0.908725629498338 | 0.7193127774943109 | 25059.229446800004 | 8.796255900000688 | miss/miss | false |
| 5 | `cand_0002_34e4744d1b7a` | 0.15 | 0.0075 | 0.9088536548028712 | 0.7193484501664578 | 73.984472799988 | 0.39226559999224264 | hit/hit | false |
| 6 | `cand_0000_9e3d005416e7` | 0.2 | 0.0075 | 0.9095895670578061 | 0.7199936929078307 | 53.82749850000255 | 8.9478853999899 | miss/miss | false |
| 7 | `cand_0008_80f29bbbc9e5` | 0.2 | 0.0075 | 0.9098856620570333 | 0.720273384292797 | 53.66482609999366 | 7.846858900011284 | miss/miss | false |
| 8 | `cand_0003_1f56ad14dbb9` | 0.25 | 0.0075 | 0.9099143777038134 | 0.7202793743903024 | 63.371918399992865 | 0.39475019999372307 | hit/hit | false |
| 9 | `cand_0004_624374bdbc90` | 0.3 | 0.0075 | 0.9100826168756632 | 0.7204217625476866 | 52.15948380000191 | 0.31741360000160057 | hit/hit | false |
| 10 | `cand_0005_ce0af1888616` | 0.35 | 0.0075 | 0.9102603775678674 | 0.7205547355984238 | 52.76632300000347 | 0.34115120000205934 | hit/hit | false |
| 11 | `cand_0010_6023ac6aa25f` | 0.15 | 0.005 | 0.9144322491770167 | 0.7246531443645938 | 76.57283839999582 | 14.75334219999786 | miss/miss | false |
| 12 | `cand_0006_f09a3d267642` | 0.2 | 0.005 | 0.9146110096961678 | 0.7248362078929044 | 53.51798250000866 | 8.094146999996156 | miss/miss | false |

## Selected Candidate

- Candidate: `cand_0011_3f9308e00fb3`
- Selected config: `artifacts/tuning/ml1m_cb_svdpp_small_study_v1/selected/selected_candidate_config.yaml`
- Selection reason: lowest validation RMSE among succeeded candidates; tie-breakers are validation MAE and fit time.

## Decision

`SELECTED_CANDIDATE_READY_FOR_BAKEOFF`

## Gates

- `ruff check tests/unit/test_tuning_selection.py tests/unit/test_run_tuning_mini_study_script.py scripts/run_tuning_mini_study.py`: passed
- `pytest tests/unit/test_tuning_selection.py tests/unit/test_run_tuning_mini_study_script.py tests/unit/test_tuning_mini_study_reports.py`: passed, 16 tests
- `ruff check .`: passed
- `pytest tests/unit`: passed, 304 tests
- `pytest tests/integration/test_unified_pipeline_smoke_all_models.py`: passed
- `pytest`: passed, 380 tests, 2 skipped

## Limitations

- This is one local ML1M small tuning study, not a global search.
- The final script invocation exceeded the tool-call timeout, but the local process completed and wrote all 12 candidate results.
- Several candidates rebuilt cluster artifacts because the current real cluster-cache identity includes the induction config derived from target `learning_rate` / `lambda_q`.
- Generated study artifacts remain local under `artifacts/` and are not versioned.

## Claim Boundary

No claim that tuning is generally faster.
No claim that the selected candidate is globally optimal.
No claim from ML100K or synthetic data.
This is a local ML1M small tuning study used to select a candidate for the next bake-off.
