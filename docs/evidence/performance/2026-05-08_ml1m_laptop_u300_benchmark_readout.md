# ML1M Laptop U300 Benchmark Readout

## Scope

This note records the local laptop baseline generated on 2026-05-08 for the six
training models. The raw report files are local artifacts and are not intended
to be versioned in Git.

## Device And Dataset

- Device profile: `configs/runtime/devices/local_u300_24gb.yaml`
- Device name: `local_u300_24gb`
- Dataset: `ml1m`
- Processed manifest:
  `data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- Split family: `benchmark_random_v1`
- Train/validation split: `0.8` / `0.1`
- Split seed: `1`
- Model seed: `1`
- Runtime config: `configs/runtime/base.yaml`

## Local Artifacts

Local-only report directory:

```text
artifacts/reports/ml1m_laptop_u300_2026-05-08/
```

Expected local files:

- `ml1m_laptop_u300_six_model_run_index.csv`
- `ml1m_laptop_u300_six_model_run_index.json`
- `performance_stage_breakdown.csv`
- `performance_hotspots.csv`
- `kernel_cost_anatomy.csv`

Each run directory under `artifacts/runs/` contains:

- `run_manifest.json`
- `config_snapshot.yaml`
- `metrics.json`
- `performance_profile.json`
- `kernel_profile.json`
- `stdout.log`

## Run IDs

| Model | Run ID |
| --- | --- |
| `biased_mf` | `2026-05-08T004817Z_ml1m_biased_mf_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` |
| `svdpp` | `2026-05-08T004906Z_ml1m_svdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` |
| `asymmetric_svd` | `2026-05-08T005445Z_ml1m_asymmetric_svd_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` |
| `asvdpp` | `2026-05-08T010531Z_ml1m_asvdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` |
| `cb_svdpp` | `2026-05-08T011336Z_ml1m_cb_svdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` |
| `cb_asvdpp` | `2026-05-08T011957Z_ml1m_cb_asvdpp_local_u300_24gb_benchmark_random_v1_tr080_va010_s001_s001` |

## Raw Readout

These values are copied from the local run index for orientation only.

| Model | Fit seconds | Profiled wall seconds | Validation RMSE |
| --- | ---: | ---: | ---: |
| `biased_mf` | 14.677 | 22.188 | 0.866678 |
| `svdpp` | 278.900 | 298.543 | 0.880023 |
| `asymmetric_svd` | 608.582 | 628.440 | 0.897034 |
| `asvdpp` | 438.470 | 444.740 | 0.879784 |
| `cb_svdpp` | 291.575 | 332.076 | 0.857911 |
| `cb_asvdpp` | 777.101 | 786.022 | 0.856611 |

## PC Comparison Context

The existing PC readout uses `local_i5_2500k_24gb`. Laptop and PC timings must
be read as separate device baselines. They are not evidence of code-level
progress or regression.

The cost-anatomy counters matched on train rows and history visit counts for
the compared rows, so the readout is useful for device-context planning. It is
not a substitute for same-device before/after measurement.

## Claim Boundary

This note does not make a code-level improvement claim. It records local device
baseline runs and the artifact locations needed to compare later laptop work
against the same `local_u300_24gb` context.
