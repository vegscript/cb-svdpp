# Evidence Note

- date: `2026-05-01`
- scope: `ml20m cb_svdpp matched-campaign budget gate`
- status: `approved_for_split_seed_1_only`
- current_release_marker: `submission-2026-05-01-r9`
- required_profile: `configs/models/tuned/ml20m_cb_svdpp_stage0_transfer.yaml`
- required_commit_state: clean `main`

## Purpose

Record the explicit budget and device gate for starting the next `ml20m`
`cb_svdpp` matched-campaign step. This note exists because the standing
large-dataset campaign contract forbids an automatic local `ml20m cb_svdpp`
matched campaign without an explicit budget/device decision.

This is a budget gate and run authorization note, not a benchmark result.

## Prior Evidence

The only `ml20m cb_svdpp` runtime basis before this gate is the clean one-epoch
feasibility probe:

- evidence:
  `docs/evidence/models/cb_svdpp/2026-04-24_ml20m_cb_svdpp_stage0_probe_e001_feasibility.md`
- validation RMSE:
  `0.863001`
- test RMSE:
  `0.863991`
- one-epoch effective fit time:
  `1178.225090s`
- peak memory:
  `17876.066406 MB`

The accepted matched-campaign contract estimates:

- 20-epoch estimate per split:
  `19856.671s` / `5.52h`
- three-split estimate:
  `16.54h` before overhead
- local device profile:
  `local_i5_2500k_24gb`

These figures are planning estimates, not benchmark results.

## Authorized Next Run

Only the first split is authorized by this gate:

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
  `1`
- model seed:
  `1`
- training index cache:
  `disabled`

Split seeds `2` and `3` are not automatically authorized by this note. They may
only start after split seed `1` completes with a valid manifest, clean git
state, no guardrail breach, and an updated readout that confirms the campaign
remains locally acceptable.

## Stop Gates

Abort or demote the run to negative evidence if any of the following occurs:

- repo is not clean on `main` before the run starts
- `configs/models/tuned/ml20m_cb_svdpp_stage0_transfer.yaml` is not committed
- peak memory crosses the documented 80 percent local RAM guardrail or causes
  system instability
- the run lacks a valid `run_manifest.json`
- `git.dirty` in the run manifest is not `false`
- the model, runtime, dataset, split seed, or model seed differs from this note
- the process exits without `metrics.json`

## Claim Boundary

This gate does not unlock any final `ml20m` model-comparison claim. It only
allows a controlled attempt at `split_seed=1`. A claim-eligible `ml20m`
comparison still requires all three split seeds `1,2,3`, model seed `1`, clean
completed run manifests, canonical `benchmark-random-multiseed` aggregation,
and synchronized updates to the publish-readiness matrix, report, and evidence
notes.
