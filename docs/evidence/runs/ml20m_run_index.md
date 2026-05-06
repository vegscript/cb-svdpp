# ML20M Run Index

- date: `2026-05-05`
- status: `partial_controlled_runs_with_cb_resource_boundary`
- dataset: `ml20m`
- intended_split_family: `benchmark_random_v1`
- intended_split: `train_ratio=0.8`, `validation_ratio=0.1`, `split_seed=1`
- intended_model_seed: `1`
- device_profile: `local_i5_2500k_24gb`
- claim_relevance: partial single-seed run evidence only; no final
  model-comparison claim

## Runs

| Run ID | Model | Config | Status | Test RMSE | Test MAE | Runtime Seconds | Peak RSS MB | Resource Notes | Claim Relevance |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `2026-05-05T202912Z_ml20m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `biased_mf` | `configs/models/selected/ml20m/ml20m_biased_mf_stage0_transfer.yaml` | `completed` | `0.775594` | `0.590539` | `390.60` | `1205.13` | split cache enabled | controlled single-seed evidence only |
| `2026-05-05T203616Z_ml20m_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `svdpp` | `configs/models/svdpp.yaml` | `completed` | `0.779429` | `0.592203` | `12041.99` | `944.36` | base profile, training-index cache enabled | controlled single-seed evidence only |
| `2026-05-05T235949Z_ml20m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `cb_svdpp` | `configs/models/cb_svdpp.yaml` | `started_without_metrics` |  |  |  |  | basis-profile attempt returned without `metrics.json`; manifest remains `started`; no active selected `ml20m cb_svdpp` profile exists | resource/time-constrained attempt, not model-quality evidence |
|  | `cb_asvdpp` | `configs/models/cb_asvdpp.yaml` | `not_started_after_cb_svdpp_boundary` |  |  |  |  | not started after `ml20m cb_svdpp` failed to produce metrics under the controlled attempt | no result claim |

## Notes

The `ml20m cb_svdpp` attempt is not a completed run. It has a manifest and
config snapshot only, with no `metrics.json`. It must not be used for RMSE, MAE,
runtime, memory, or model comparison claims. `ml20m cb_asvdpp` was not started
after this CB boundary was observed.
