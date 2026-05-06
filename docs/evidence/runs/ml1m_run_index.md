# ML1M Run Index

- date: `2026-05-05`
- status: `controlled_single_seed_runs_recorded`
- dataset: `ml1m`
- split_family: `benchmark_random_v1`
- split: `train_ratio=0.8`, `validation_ratio=0.1`, `split_seed=1`
- model_seed: `1`
- device_profile: `local_i5_2500k_24gb`
- claim_relevance: single-seed controlled run evidence only; not a final
  multi-seed benchmark claim

## Runs

| Run ID | Model | Config | Status | Test RMSE | Test MAE | Runtime Seconds | Peak RSS MB | Resource Notes | Claim Relevance |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `2026-05-05T072047Z_ml1m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `biased_mf` | `configs/models/selected/ml1m/ml1m_biased_mf_stage0_transfer.yaml` | `completed` | `0.868475` | `0.679173` | `23.76` | `241.84` | split cache enabled | controlled single-seed evidence only |
| `2026-05-05T072131Z_ml1m_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `svdpp` | `configs/models/svdpp.yaml` | `completed` | `0.882945` | `0.686760` | `475.75` | `254.63` | base profile, training-index cache enabled | controlled single-seed evidence only |
| `2026-05-05T072946Z_ml1m_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `asvdpp` | `configs/models/asvdpp.yaml` | `completed` | `0.881981` | `0.686823` | `1107.92` | `263.86` | base profile, training-index cache enabled | controlled single-seed evidence only |
| `2026-05-05T074832Z_ml1m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `cb_svdpp` | `configs/models/selected/ml1m/ml1m_cb_svdpp_stage0_transfer.yaml` | `completed` | `0.859314` | `0.673340` | `813.82` | `275.08` | training-index and cluster-artifact caches enabled | controlled single-seed evidence only; `cb_claim_eligible=false` |
| `2026-05-05T080224Z_ml1m_cb_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `cb_asvdpp` | `configs/models/selected/ml1m/ml1m_cb_asvdpp_stage0_transfer.yaml` | `completed` | `0.858497` | `0.672333` | `2115.51` | `277.56` | training-index and cluster-artifact caches enabled | controlled single-seed evidence only; `cb_claim_eligible=false` |

## Notes

The batch command reached the shell timeout while `cb_asvdpp` was still running.
The specific process was allowed to finish and produced a complete
`metrics.json` and `run_manifest.json`.
