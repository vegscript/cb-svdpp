# ML100K Run Index

- date: `2026-05-05`
- status: `controlled_single_seed_runs_recorded`
- dataset: `ml100k`
- split_family: `benchmark_random_v1`
- split: `train_ratio=0.8`, `validation_ratio=0.1`, `split_seed=1`
- model_seed: `1`
- device_profile: `local_i5_2500k_24gb`
- claim_relevance: single-seed controlled run evidence only; not a final
  multi-seed benchmark claim

## Runs

| Run ID | Model | Config | Status | Test RMSE | Test MAE | Runtime Seconds | Peak RSS MB | Resource Notes | Claim Relevance |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `2026-05-05T071611Z_ml100k_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `biased_mf` | `configs/models/selected/ml100k/ml100k_biased_mf_stage1.yaml` | `completed` | `0.932512` | `0.730806` | `7.55` | `197.98` | split cache enabled | controlled single-seed evidence only |
| `2026-05-05T071638Z_ml100k_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `svdpp` | `configs/models/selected/ml100k/ml100k_svdpp_stage1.yaml` | `completed` | `0.912009` | `0.715632` | `41.34` | `204.98` | training-index cache enabled | controlled single-seed evidence only |
| `2026-05-05T071738Z_ml100k_asymmetric_svd_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `asymmetric_svd` | `configs/models/asymmetric_svd.yaml` | `completed` | `0.915839` | `0.720305` | `59.84` | `208.55` | base profile, training-index cache enabled | controlled single-seed evidence only |
| `2026-05-05T071857Z_ml100k_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `asvdpp` | `configs/models/asvdpp.yaml` | `completed` | `0.934610` | `0.726536` | `61.47` | `211.41` | base profile, training-index cache enabled | controlled single-seed evidence only |
| `2026-05-05T071125Z_ml100k_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `cb_svdpp` | `configs/models/selected/ml100k/ml100k_cb_svdpp_stage1.yaml` | `completed` | `0.910396` | `0.713885` | `56.17` | `212.44` | training-index and cluster-artifact caches enabled | controlled single-seed evidence only; `cb_claim_eligible=false` |
| `2026-05-05T071240Z_ml100k_cb_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `cb_asvdpp` | `configs/models/selected/ml100k/ml100k_cb_asvdpp_stage1.yaml` | `completed` | `0.910213` | `0.712082` | `133.29` | `217.49` | training-index and cluster-artifact caches enabled | controlled single-seed evidence only; `cb_claim_eligible=false` |

## Notes

An initial invocation attempted to enable all caches for every model. The
Unified Runner rejected non-applicable cache options for non-CB models. Those
CLI errors did not produce completed run artifacts and are not counted as model
results in this index.
