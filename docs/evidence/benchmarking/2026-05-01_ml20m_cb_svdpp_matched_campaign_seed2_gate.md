# Evidence Note

- date: `2026-05-01`
- scope: `ml20m cb_svdpp matched-campaign split-seed-2 gate`
- status: `approved_for_split_seed_2_only`
- current_release_marker: `submission-2026-05-01-r9`
- required_profile: `configs/models/tuned/ml20m_cb_svdpp_stage0_transfer.yaml`
- required_commit_state: clean `main`

## Purpose

Authorize the next single `ml20m cb_svdpp` matched-campaign split after split
seed `1` completed under the previous budget gate. This note does not authorize
split seed `3` and does not create a benchmark result by itself.

## Prior Split Readout

The completed split-seed-1 readout is documented in:

`docs/evidence/models/cb_svdpp/2026-05-01_ml20m_cb_svdpp_stage0_transfer_seed1_readout.md`

Central readout:

- validation RMSE:
  `0.780558`
- test RMSE:
  `0.781255`
- fit-time total seconds:
  `20117.635891`
- peak memory MB:
  `18586.136719`
- run manifest status:
  `completed`
- git dirty:
  `false`

The memory readout stayed below the local 80 percent RAM guardrail, but remains
close enough that the campaign must continue sequentially with monitoring.

## Authorized Next Run

Only the second split is authorized by this gate:

- dataset:
  `ml20m`
- processed manifest:
  `data/processed/ml20m/ml20m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- model config:
  `configs/models/tuned/ml20m_cb_svdpp_stage0_transfer.yaml`
- runtime config:
  `configs/runtime/base.yaml`
- device config:
  `configs/runtime/devices/local_i5_2500k_24gb.yaml`
- split family:
  `benchmark_random_v1`
- split contract:
  `train_ratio=0.8`, `validation_ratio=0.1`
- split seed:
  `2`
- model seed:
  `1`
- training index cache:
  `disabled`

Split seed `3` is not automatically authorized by this note. It may only start
after split seed `2` completes with a valid manifest, clean git state, no
guardrail breach, and an updated readout that confirms the campaign remains
locally acceptable.

## Stop Gates

Abort or demote the run to negative evidence if any of the following occurs:

- repo is not clean on `main` before the run starts
- peak memory crosses the documented 80 percent local RAM guardrail or causes
  system instability
- the run lacks a valid `run_manifest.json`
- `git.dirty` in the run manifest is not `false`
- the model, runtime, dataset, split seed, or model seed differs from this note
- the process exits without `metrics.json`

## Claim Boundary

This gate does not unlock any final `ml20m` model-comparison claim. It only
allows a controlled attempt at `split_seed=2`.
