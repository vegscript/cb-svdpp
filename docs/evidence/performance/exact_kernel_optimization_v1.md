# Exact Kernel Optimization V1 Evidence

This evidence note records the first exact kernel optimization benchmark. It is
not a broad performance claim. The only implemented optimization is workspace
reuse for avoidable per-rating temporary arrays in the non-CB Numba kernels.

## Scope

Implemented EXACT optimization:

- `train_biased_mf_epoch_numba`: old user/item factor workspaces moved outside
  the per-rating loop.
- `train_svdpp_epoch_numba`: old user/item factor workspaces and the implicit
  context workspace moved outside the per-rating loop.
- `train_asymmetric_svd_epoch_numba`: old item-factor workspace and context
  workspace moved outside the per-rating loop; context is reset explicitly per
  rating.
- `train_asvdpp_epoch_numba`: old user/item factor workspaces and context
  workspace moved outside the per-rating loop.

Intentionally not implemented:

- No context caching across rating updates.
- No residual-weight caching.
- No cluster-history caching.
- No parallel SGD.
- No rating or history reordering.
- No formula, hyperparameter, split, dtype, or tuning changes.
- No CB kernel changes; the CB kernels already had their main workspace arrays
  outside the per-rating loop.

## Benchmark Contract

All before/after pairs use:

- dataset: `ml1m`
- processed manifest:
  `data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- split family: `benchmark_random_v1`
- split id: `benchmark_random_v1_tr080_va010_s001`
- split seed: `1`
- model seed: `1`
- train ratio: `0.8`
- validation ratio: `0.1`
- dtype: `float32`
- device profile: `local_i5_2500k_24gb`
- runtime config: `configs/runtime/base.yaml`
- split cache: disabled
- training index cache: disabled
- cluster artifact cache: disabled

Same-config confirmation was checked from `run_manifest.json` for dataset
manifest, model config path, split id, split family, model seed, dtype, and
device profile. All six pairs matched.

Per-model contract details:

| Model | Model config | Epochs before | Epochs after | Train rows before | Train rows after | Same split id | Same dtype | Same device |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `biased_mf` | `configs/models/selected/ml1m/ml1m_biased_mf_stage0_transfer.yaml` | 25 | 25 | 800193 | 800193 | true | true | true |
| `svdpp` | `configs/models/svdpp.yaml` | 20 | 20 | 800193 | 800193 | true | true | true |
| `asymmetric_svd` | `configs/models/asymmetric_svd.yaml` | 20 | 20 | 800193 | 800193 | true | true | true |
| `asvdpp` | `configs/models/asvdpp.yaml` | 20 | 20 | 800193 | 800193 | true | true | true |
| `cb_svdpp` | `configs/models/selected/ml1m/ml1m_cb_svdpp_stage0_transfer.yaml` | 20 | 20 | 800193 | 800193 | true | true | true |
| `cb_asvdpp` | `configs/models/selected/ml1m/ml1m_cb_asvdpp_stage0_transfer.yaml` | 20 | 20 | 800193 | 800193 | true | true | true |

The workspace was dirty for both before and after runs. These runs are valid for
this local diagnostic before/after readout because the same local artifact
contract and configuration references were used, but they should not be promoted
to final benchmark claims.

## Run IDs

| Model | Before run id | After run id |
| --- | --- | --- |
| `biased_mf` | `2026-05-06T193752Z_ml1m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `2026-05-06T213413Z_ml1m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` |
| `svdpp` | `2026-05-06T211124Z_ml1m_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `2026-05-06T213445Z_ml1m_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` |
| `asymmetric_svd` | `2026-05-06T194745Z_ml1m_asymmetric_svd_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `2026-05-06T214148Z_ml1m_asymmetric_svd_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` |
| `asvdpp` | `2026-05-06T200507Z_ml1m_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `2026-05-06T220041Z_ml1m_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` |
| `cb_svdpp` | `2026-05-06T202231Z_ml1m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `2026-05-06T221715Z_ml1m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` |
| `cb_asvdpp` | `2026-05-06T203535Z_ml1m_cb_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | `2026-05-06T223014Z_ml1m_cb_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` |

## Artifact Check

For each before and after run, these artifacts were present:

- `metrics.json`
- `run_manifest.json`
- `config_snapshot.yaml`
- `performance_profile.json`
- `kernel_profile.json`

Artifact locations follow this convention for every run id listed above:

```text
artifacts/runs/<run_id>/metrics.json
artifacts/runs/<run_id>/run_manifest.json
artifacts/runs/<run_id>/config_snapshot.yaml
artifacts/runs/<run_id>/performance_profile.json
artifacts/runs/<run_id>/kernel_profile.json
```

Collectors were rerun after the benchmark:

- `scripts/collect_performance_profiles.py`
- `scripts/collect_kernel_profiles.py`

Updated report outputs:

- `artifacts/reports/performance_stage_breakdown.csv`
- `artifacts/reports/performance_hotspots.csv`
- `artifacts/reports/kernel_cost_anatomy.csv`

## Before/After Metrics

| Model | Same contract | RMSE before | RMSE after | MAE before | MAE after | fit before s | fit after s | fit ratio before/after | train before s | train after s | estimated factor touches before | estimated factor touches after |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `biased_mf` | true | 0.868475231036 | 0.868475231036 | 0.679172638770 | 0.679172638770 | 16.178 | 11.911 | 1.358 | 16.178 | 11.911 | 5121235200 | 5121235200 |
| `svdpp` | true | 0.882945480720 | 0.882945480720 | 0.686760147697 | 0.686760147697 | 426.552 | 401.816 | 1.062 | 426.552 | 401.816 | 502677358000 | 502677358000 |
| `asymmetric_svd` | true | 0.899189422151 | 0.899189422151 | 0.710399805347 | 0.710399805347 | 1017.104 | 1107.096 | 0.919 | 1017.104 | 1107.096 | 1002153944000 | 1002153944000 |
| `asvdpp` | true | 0.881981273881 | 0.881981273881 | 0.686823193480 | 0.686823193480 | 1019.779 | 970.167 | 1.051 | 1019.779 | 970.167 | 1002153944000 | 1002153944000 |
| `cb_svdpp` | true | 0.859313721176 | 0.859313721176 | 0.673339978940 | 0.673339978940 | 744.180 | 737.836 | 1.009 | 744.180 | 737.836 | 755689126400 | 755689126400 |
| `cb_asvdpp` | true | 0.858497453970 | 0.858497453970 | 0.672333240005 | 0.672333240005 | 1958.666 | 2109.758 | 0.928 | 1958.666 | 2109.758 | 1395019156480 | 1395019156480 |

## Secondary Performance Readout

| Model | Peak MB before | Peak MB after | Ratings/s before | Ratings/s after | fit seconds per million estimated touches before | fit seconds per million estimated touches after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `biased_mf` | 832.945 | 808.609 | 1236510.269 | 1679475.610 | 0.003152830259 | 0.002320144035 |
| `svdpp` | 1139.246 | 1157.652 | 37519.092 | 39828.858 | 0.000848502180 | 0.000799289607 |
| `asymmetric_svd` | 1177.559 | 1178.047 | 15734.733 | 14455.709 | 0.001014893726 | 0.001104691776 |
| `asvdpp` | 1168.098 | 1182.422 | 15693.451 | 16495.976 | 0.001017555163 | 0.000968049099 |
| `cb_svdpp` | 1458.699 | 1467.625 | 21505.350 | 21690.268 | 0.000984726778 | 0.000976330948 |
| `cb_asvdpp` | 1446.090 | 1454.574 | 8170.797 | 7585.638 | 0.001404013720 | 0.001512323792 |

## Numerical Stability

All six before/after pairs produced identical recorded `test_rmse` and
`test_mae` values at the precision shown above. Estimated factor touches were
unchanged for every model.

This supports the intended exactness of the workspace-lifetime change for this
diagnostic run set. It does not prove behavior on every possible dataset or
configuration.

## Observed Fit-Time Readout

Observed `fit_model` deltas in this local diagnostic run:

- `biased_mf`: 16.178s to 11.911s
- `svdpp`: 426.552s to 401.816s
- `asymmetric_svd`: 1017.104s to 1107.096s
- `asvdpp`: 1019.779s to 970.167s
- `cb_svdpp`: 744.180s to 737.836s
- `cb_asvdpp`: 1958.666s to 2109.758s

The CB models were not changed, so their small positive/negative movement should
be treated as run-to-run noise in this local diagnostic comparison. The
`asymmetric_svd` regression means workspace reuse is not uniformly beneficial in
this single ML1M run and needs a focused follow-up before broader claims.

## Known Caveats

- This is a single split-seed/model-seed ML1M diagnostic run, not the full
  benchmark contract from `docs/kernel_optimization_plan.md`.
- The workspace was dirty for both before and after runs because the profiling
  and kernel-anatomy changes are still in the same development stack.
- Numba cold-start and warm-run timing were not separately isolated beyond using
  the same benchmark path before and after.
- The observed fit-time readout is model- and machine-specific.
- No memory-improvement claim follows from this evidence; peak memory moved in
  both directions across models.

## Next Step

Before promoting this optimization as a broader exact performance improvement,
run the full benchmark contract from `docs/kernel_optimization_plan.md`:

- `ml100k`: all six models
- `ml1m`: all six models
- optional `ml10m`: `biased_mf`, `svdpp`, `cb_svdpp`, `cb_asvdpp`

The immediate technical follow-up should inspect why `asymmetric_svd` slowed down
in this single run despite identical metrics and unchanged estimated work. If
that result persists, keep workspace reuse for the kernels where it is
beneficial and revert or specialize the `asymmetric_svd` workspace change.
