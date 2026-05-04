# CB-SVD++ G6 Outer Benchmark Run

- date: `2026-05-03`
- status: `pass_for_clean_outer_benchmark`
- dataset: `ml100k`
- model: `cb_svdpp`
- selected_config:
  `configs/models/selected/ml100k/ml100k_cb_svdpp_g6_validation_selected.yaml`
- contract:
  `docs/evidence/reproduction/current/2026-05-03_cb_svdpp_g6_outer_benchmark_contract.md`
- selection_evidence:
  `docs/evidence/reproduction/current/2026-05-03_cb_svdpp_g6_validation_grid_run.md`
- git commit: `67570ed8fde4c158848ab80494524ab203b40df5`
- git dirty: `false`

## Purpose

This note records the clean outer benchmark for the frozen G6-selected
`ml100k cb_svdpp` profile. The selected config was frozen before any outer
test-set evaluation, and the G6 selection evidence used validation metrics
only.

This is an outer benchmark readout for one documented profile. It is not a
SOTA, scalability, speed, production-readiness, or paper-faithfulness claim.

## Commands

Each run used the same processed manifest, selected model config, runtime
config, device config, split family, model seed, and cache policy. Only
`--split-seed` changed.

Migration note: the command blocks preserve the original pre-archive selected
config path as execution provenance. The moved config is now
`configs/models/selected/ml100k/ml100k_cb_svdpp_g6_validation_selected.yaml`.

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main train-cb-svdpp `
  data\processed\ml100k\ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json `
  --model-config configs\models\tuned\ml100k_cb_svdpp_g6_validation_selected.yaml `
  --runtime-config configs\runtime\base.yaml `
  --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml `
  --split-family benchmark_random_v1 `
  --train-ratio 0.8 `
  --validation-ratio 0.1 `
  --split-seed 1 `
  --model-seed 1 `
  --split-cache enable `
  --training-index-cache `
  --cluster-artifact-cache
```

The same command was repeated for split seeds `2` and `3`.

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main benchmark-random-multiseed `
  data\processed\ml100k\ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json `
  configs\models\tuned\ml100k_cb_svdpp_g6_validation_selected.yaml `
  --runtime-config configs\runtime\base.yaml `
  --device-config configs\runtime\devices\local_i5_2500k_24gb.yaml `
  --split-seeds 1,2,3 `
  --model-seed 1
```

## Artifact Readout

- benchmark artifact:
  `artifacts/benchmarks/2026-05-03T120906Z_ml100k_benchmark_random_v1_cb_svdpp_multiseed_s001_s002_s003_modelseed_s001_local_i5_2500k_24gb/benchmark_manifest.json`
- run manifests:
  `artifacts/runs/2026-05-03T120705Z_ml100k_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/run_manifest.json`
- run manifests:
  `artifacts/runs/2026-05-03T120739Z_ml100k_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s002_s001/run_manifest.json`
- run manifests:
  `artifacts/runs/2026-05-03T120813Z_ml100k_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s003_s001/run_manifest.json`
- benchmark status: `completed`
- benchmark git dirty: `false`
- measured sample count: `3`
- split seeds: `1,2,3`
- model seed: `1`
- runtime device profile: `local_i5_2500k_24gb`
- device profile claim eligible: `true`
- dtype: `float32`
- cache status: split cache, training-index cache, cluster-artifact cache, and
  user-cluster-history cache were enabled and manifest-visible.

## Metrics

Aggregate:

| Metric | Mean | Std | Min | Median | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation RMSE | `0.9566122815305916` | `0.002372317660216579` | `0.9550286612769201` | `0.9554683744130718` | `0.9593398089017833` |
| test RMSE | `0.9595668222022953` | `0.011253690215412724` | `0.9466147041028611` | `0.9651331994295013` | `0.9669525630745236` |
| train RMSE | `0.9253765314125518` | `0.0009672273895100401` | `0.9245197352038855` | `0.9251844828909535` | `0.9264253761428164` |
| training wall-clock seconds | `4.767408333330725` | `0.18289051624149263` | `4.55747609998798` | `4.852492699981667` | `4.89225620002253` |
| peak memory MB | `237.70703125` | `1.8478792148767293` | `236.0703125` | `237.33984375` | `239.7109375` |

Per split seed:

| Split Seed | Validation RMSE | Test RMSE | Training Seconds | Peak Memory MB |
| --- | ---: | ---: | ---: | ---: |
| `1` | `0.9593398089017833` | `0.9466147041028611` | `4.55747609998798` | `239.7109375` |
| `2` | `0.9554683744130718` | `0.9669525630745236` | `4.852492699981667` | `237.33984375` |
| `3` | `0.9550286612769201` | `0.9651331994295013` | `4.89225620002253` | `236.0703125` |

## Claim Boundary

Allowed:

- clean outer `ml100k cb_svdpp` benchmark readout for this frozen G6-selected
  profile under `benchmark_random_v1`, split seeds `1,2,3`, model seed `1`,
  commit `67570ed`, and device profile `local_i5_2500k_24gb`

Not allowed:

- comparison against the older `paper_faithful_ml100k_v1` anchor rows without
  explicitly stating the split-family difference
- model-family superiority claim
- speed claim
- scalability claim
- production-readiness claim
- SOTA claim
- paper-faithfulness claim
- `R_star` optimization claim
- any `ml1m`, `ml10m`, or `ml20m` claim

## Evidence Gate Readout

Executed after documenting this run:

- Ruff: `All checks passed!`
- Mypy: `Success: no issues found in 62 source files`
- Pytest: `136 passed`
