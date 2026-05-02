# Evidence Note

- date: `2026-04-24`
- scope: `large-dataset cb_svdpp run-budget decision`
- status: `accepted`
- decision: defer deeper local `ml10m` and `ml20m` `cb_svdpp` runs

## Claim Or Question

Should the repo spend the current local `local_i5_2500k_24gb` budget on deeper
`ml10m` or `ml20m` `cb_svdpp` runs after the clean one-epoch feasibility probes?

## Inputs And Artifacts

- `ml10m biased_mf` feasibility evidence:
  `docs/evidence/models/biased_mf/2026-04-24_ml10m_biased_mf_stage0_transfer_feasibility.md`
- `ml10m cb_svdpp` feasibility evidence:
  `docs/evidence/models/cb_svdpp/2026-04-24_ml10m_cb_svdpp_stage0_probe_e001_feasibility.md`
- `ml20m biased_mf` feasibility evidence:
  `docs/evidence/models/biased_mf/2026-04-24_ml20m_biased_mf_stage0_transfer_feasibility.md`
- `ml20m cb_svdpp` feasibility evidence:
  `docs/evidence/models/cb_svdpp/2026-04-24_ml20m_cb_svdpp_stage0_probe_e001_feasibility.md`
- reference transfer profile:
  `configs/models/tuned/ml1m_cb_svdpp_stage0_transfer.yaml`

## Method

Use only clean-run evidence already recorded in the repo. Do not reuse dirty
workspace artifacts and do not start another long run before the decision is
documented.

For rough local budget sizing, use the measured one-epoch cluster-model probes
as a conservative planning input:

- one clean `ml10m cb_svdpp` probe measured one cluster-induction phase plus
  one main epoch
- one clean `ml20m cb_svdpp` probe measured one cluster-induction phase plus
  one main epoch
- the `ml1m` transfer profile uses `20` epochs, so a 20-epoch large-dataset
  local run would be at least multi-hour if the measured main-epoch cost scales
  linearly

This extrapolation is a planning estimate, not a benchmark result.

## Readout

Measured clean `ml10m cb_svdpp` one-epoch probe:

- git dirty: `false`
- validation RMSE: `0.872094`
- test RMSE: `0.871385`
- cluster induction wall clock seconds: `78.425572`
- main training wall clock seconds: `422.423619`
- effective fit time seconds: `500.849191`
- peak memory MB: `12730.062500`

Measured clean `ml20m cb_svdpp` one-epoch probe:

- git dirty: `false`
- validation RMSE: `0.863001`
- test RMSE: `0.863991`
- cluster induction wall clock seconds: `195.149008`
- main training wall clock seconds: `983.076083`
- effective fit time seconds: `1178.225090`
- peak memory MB: `17876.066406`

Planning estimate if a 20-epoch transfer-style run used the measured
one-epoch main-training cost as a lower-bound planning input:

- `ml10m`: `78.425572 + 20 * 422.423619 = 8526.898` seconds, about
  `2.37` hours for one split/model-seed run
- `ml20m`: `195.149008 + 20 * 983.076083 = 19856.671` seconds, about
  `5.52` hours for one split/model-seed run
- three split-seed runs would multiply those wall-clock budgets before any
  retry, tuning, or report work

Memory risk is also material on `ml20m`: the one-epoch probe measured
`17876.066406 MB`, which is below but close to the documented 80 percent RAM
guardrail for the local 24 GB profile.

## Interpretation

The current evidence is sufficient to claim local feasibility for bounded
one-epoch `cb_svdpp` probes on `ml10m` and `ml20m`.

The evidence is not sufficient to justify final model-comparison claims on
`ml10m` or `ml20m`, because the clustering probes are single-seed,
single-epoch, and budget-unmatched against the `biased_mf` feasibility
baselines.

Starting deeper local `cb_svdpp` runs now would spend multi-hour local budget
without closing the publish-readiness gap unless the repo also commits to
matched seed policy, matched budget, rerun discipline, and report integration.
On the current local hardware, that is not the highest-integrity next step.

## Decision

- Defer deeper `ml10m` and `ml20m` `cb_svdpp` runs on the current local
  `local_i5_2500k_24gb` device.
- Keep the existing `ml10m` and `ml20m` `cb_svdpp` rows at
  `single_epoch_feasibility`.
- Keep all final `ml10m` and `ml20m` model-comparison claims blocked.
- Resume deeper large-dataset CB runs only if one of these is true:
  - a stronger device profile is available and documented
  - an explicit multi-hour local budget is approved for matched reruns
  - the final report scope requires a negative-result section backed by a
    deliberately bounded failed or cancelled run

## Next Step

Move to report condensation and reproduction-readiness evidence instead of
starting another large local CB run.
