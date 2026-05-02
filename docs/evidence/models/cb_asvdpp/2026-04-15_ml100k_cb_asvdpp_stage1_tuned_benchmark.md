# Evidence Note

## Scope

First official `ml100k` outer benchmark for the promoted `cb_asvdpp`
`stage1_tuned` profile on the canonical `paper_faithful_ml100k_v1` fold family.

## Claim Or Question

Does the first promoted `cb_asvdpp` profile remain competitive on the official
`ml100k` outer folds, and how does its first official readout compare with the
current `biased_mf`, `svdpp`, and `cb_svdpp` anchors?

## Inputs And Artifacts

- processed dataset manifest:
  `data/processed/ml100k/ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config: `configs/models/tuned/ml100k_cb_asvdpp_stage1.yaml`
- runtime config: `configs/runtime/base.yaml`
- device config: `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- benchmark directory:
  `artifacts/benchmarks/2026-04-15T184526Z_ml100k_paper_faithful_cb_asvdpp_local_i5_2500k_24gb/`
- benchmark manifest:
  `artifacts/benchmarks/2026-04-15T184526Z_ml100k_paper_faithful_cb_asvdpp_local_i5_2500k_24gb/benchmark_manifest.json`
- comparison anchors:
  `artifacts/benchmarks/2026-04-13T130402Z_ml100k_paper_faithful_biased_mf_multiseed_s001_s002_s003_local_i5_2500k_24gb/`
  `artifacts/benchmarks/2026-04-15T030305Z_ml100k_paper_faithful_svdpp_multiseed_s001_s002_s003_local_i5_2500k_24gb/`
  `artifacts/benchmarks/2026-04-15T124046Z_ml100k_paper_faithful_cb_svdpp_local_i5_2500k_24gb/`
  `artifacts/benchmarks/2026-04-15T150207Z_ml100k_paper_faithful_cb_svdpp_multiseed_s001_s002_s003_local_i5_2500k_24gb/`

## Method

- use the canonical official five-fold `paper_faithful_ml100k_v1` protocol
- evaluate the promoted `stage1_tuned` `cb_asvdpp` config on folds `u1` to `u5`
- report benchmark fit time as cluster induction plus main training
- keep the result provisional because the current workspace state is dirty and
  because the model inherits the accepted detached residual contract from
  `D-003`

## Readout

- status: `completed`
- manifest validation: `valid`
- Git state: dirty
- test RMSE mean: `0.917367`
- test RMSE std: `0.007451`
- fit-time mean: `477.524850` seconds
- train RMSE mean: `0.686120`

## Interpretation

This first official `cb_asvdpp` readout is strong, but it is not yet a clean
anchor.

Observed comparison:

- versus clean multi-seed `biased_mf`:
  - test RMSE improved from `0.937111` to `0.917367`
  - absolute gain: `0.019744`
- versus clean multi-seed `svdpp`:
  - test RMSE improved from `0.924015` to `0.917367`
  - absolute gain: `0.006648`
- versus clean multi-seed `cb_svdpp`:
  - test RMSE improved from `0.918968` to `0.917367`
  - absolute gain: `0.001601`
- versus single-seed clean `cb_svdpp`:
  - test RMSE improved from `0.919292` to `0.917367`
  - absolute gain: `0.001925`

The outer result is therefore better than every currently documented anchor at
the RMSE level, including the clean `cb_svdpp` baseline family. However, this
must still be treated as provisional for two reasons:

- the benchmark was executed with `git.dirty=true`
- the model depends on the accepted detached explicit residual contract from
  `D-003`, so it remains source-grounded but not optimizer-faithful

## Decision Or Next Step

- keep this benchmark as the first official `cb_asvdpp` result
- do not yet promote it to the repository's clean best-result claim
- next step: rerun the same `stage1_tuned` profile from a clean Git state and
  then extend to a clean multi-seed confirmation
