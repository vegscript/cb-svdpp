# ML10M Run Index

- date: `2026-05-05`
- status: `controlled_single_seed_runs_recorded_with_prior_timeout_attempt`
- dataset: `ml10m`
- split_family: `benchmark_random_v1`
- split: `train_ratio=0.8`, `validation_ratio=0.1`, `split_seed=1`
- model_seed: `1`
- device_profile: `local_i5_2500k_24gb`
- claim_relevance: single-seed run evidence only; not a final
  model-comparison claim

## Runs

| Run ID | Model | Config | Status | Test RMSE | Test MAE | Runtime Seconds | Peak RSS MB | Resource Notes | Claim Relevance |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `2026-05-05T083836Z_ml10m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `biased_mf` | `configs/models/selected/ml10m/ml10m_biased_mf_stage0_transfer.yaml` | `completed` | `0.786747` | `0.603147` | `188.60` | `688.80` | split cache enabled | controlled single-seed evidence only |
| `2026-05-05T084213Z_ml10m_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `svdpp` | `configs/models/svdpp.yaml` | `completed` | `0.793940` | `0.607161` | `5081.44` | `754.74` | base profile, training-index cache enabled | controlled single-seed evidence only |
| `2026-05-05T121413Z_ml10m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `cb_svdpp` | `configs/models/selected/ml10m/ml10m_cb_svdpp_stage0_transfer.yaml` | `completed` | `0.790454` | `0.607096` | `8537.05` | `813.46` | training-index and cluster-artifact caches enabled; completed on the controlled rerun | controlled single-seed evidence only; `cb_claim_eligible=false` |
| `2026-05-05T143714Z_ml10m_cb_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `cb_asvdpp` | `configs/models/cb_asvdpp.yaml` | `completed` | `0.787362` | `0.602819` | `21071.61` | `918.85` | base profile; training-index and cluster-artifact caches enabled | controlled single-seed evidence only; `cb_claim_eligible=false` |

## Notes

Earlier attempt
`2026-05-05T100750Z_ml10m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001`
exceeded the initial 2-hour tool window and was stopped. It has no
`metrics.json` and remains non-result evidence only. The later
`2026-05-05T121413Z...` run is the completed `cb_svdpp` result in this index.
