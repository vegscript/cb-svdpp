# Evidence Note

## Scope

First clean confirmatory full run for `cb_svdpp` on `MovieLens 1M`, plus direct
comparison to the clean `biased_mf` control run from the same snapshot.

## Claim Or Question

Does the selected `ml1m` `cb_svdpp` stage-0 profile beat the clean `biased_mf`
control baseline on the same split, device, and seed setup?

## Inputs And Artifacts

- processed dataset manifest:
  `data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- confirmed model config:
  `configs/models/tuned/ml1m_cb_svdpp_stage0_transfer.yaml`
- clean control baseline config:
  `configs/models/tuned/ml1m_biased_mf_stage0_transfer.yaml`
- confirmatory run directory:
  `artifacts/runs/2026-04-16T065748Z_ml1m_cb_svdpp_local_i5_2500k_24gb_s001/`
- clean control run directory:
  `artifacts/runs/2026-04-16T081023Z_ml1m_biased_mf_local_i5_2500k_24gb_s001/`
- run manifest validation: `2/2 valid`
- clean snapshot commit: `2c6ffa1e1b19ac9cc0952f96b8ccac8dbb1ff656`

## Method

- run the selected `cb_svdpp` profile for the full `20` epochs
- use `train_ratio=0.8`, `validation_ratio=0.1`, split seed `1`, and model seed `1`
- execute from a clean cloned snapshot with `dirty=false`
- disable training-index cache reuse so the claim does not depend on hidden
  cache state
- compare against a fresh clean `biased_mf` control run on the same setup
- treat `cb_svdpp` fit time fairly as cluster induction plus main training

## Readout

### Clean `cb_svdpp` Confirmatory Run

- status: `completed`
- Git dirty: `false`
- train RMSE: `0.724115`
- validation RMSE: `0.857911`
- test RMSE: `0.859314`
- cluster induction wall clock seconds: `195.566152`
- main training wall clock seconds: `3193.527693`
- effective fit time seconds: `3389.093845`
- inference wall clock seconds: `7.869697`
- peak memory MB: `1477.734375`
- model size MB: `13.068722`

### Clean `biased_mf` Control Run

- status: `completed`
- Git dirty: `false`
- validation RMSE: `0.866678`
- test RMSE: `0.868475`
- training wall clock seconds: `43.291659`
- peak memory MB: `849.289063`

### Direct Delta: `cb_svdpp - biased_mf`

- validation RMSE delta: `-0.008766`
- test RMSE delta: `-0.009162`

## Interpretation

On this clean single-seed `ml1m` comparison, the selected `cb_svdpp` profile
beats the clean `biased_mf` control on both validation and test RMSE. This is
our first clean evidence that the clustering-based path can outperform the
matrix-factorization control on `ml1m` under the repo's canonical random split.

The tradeoff is substantial systems cost. Relative to the clean `biased_mf`
control, `cb_svdpp` is much slower and heavier on the same machine. The quality
result is therefore positive, but it does not justify any blanket performance
claim in favor of `cb_svdpp`.

This remains a single-seed result. It is valid evidence, but not yet a
multi-seed final benchmark claim.

## Decision Or Next Step

- treat `configs/models/tuned/ml1m_cb_svdpp_stage0_transfer.yaml` as the
  currently validated `ml1m` `cb_svdpp` candidate
- next clean evaluation step: add at least `2` more split seeds for a
  comparison-grade readout against the same clean `biased_mf` control family
- keep quality and systems tradeoffs explicitly paired in any summary table