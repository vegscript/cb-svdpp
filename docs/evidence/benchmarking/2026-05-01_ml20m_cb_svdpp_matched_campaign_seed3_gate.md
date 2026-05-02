# Evidence Note

- date: `2026-05-01`
- scope: `ml20m cb_svdpp matched-campaign split-seed-3 gate`
- status: `approved_for_split_seed_3_only`
- current_release_marker: `submission-2026-05-01-r9`
- required_profile: `configs/models/tuned/ml20m_cb_svdpp_stage0_transfer.yaml`
- required_commit_state: clean `main`

## Purpose

Authorize the final single `ml20m cb_svdpp` matched-campaign split after split
seeds `1` and `2` completed under their sequential gates. This note does not
create a benchmark result by itself.

Guardrail phrase: final single-split gate.

## Prior Split Readouts

Split seed `1`:

- evidence:
  `docs/evidence/models/cb_svdpp/2026-05-01_ml20m_cb_svdpp_stage0_transfer_seed1_readout.md`
- validation RMSE:
  `0.780558`
- test RMSE:
  `0.781255`
- peak memory MB:
  `18586.136719`

Split seed `2`:

- evidence:
  `docs/evidence/models/cb_svdpp/2026-05-01_ml20m_cb_svdpp_stage0_transfer_seed2_readout.md`
- validation RMSE:
  `0.781702`
- test RMSE:
  `0.781773`
- peak memory MB:
  `19128.371094`

Both completed with valid manifests and `git.dirty=false`. The split-seed-2
memory readout stayed below the local 80 percent RAM guardrail but left only a
narrow margin. The final split must therefore be monitored and must stop or
demote to negative evidence if the guardrail is crossed.

## Authorized Next Run

Only the third split is authorized by this gate:

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
  `3`
- model seed:
  `1`
- training index cache:
  `disabled`

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
allows a controlled attempt at `split_seed=3`. Final claims remain blocked until
all three completed runs are aggregated with `benchmark-random-multiseed` and
the claim matrix, report, and evidence notes are synchronized.
