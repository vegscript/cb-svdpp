# Evidence Note

- date: `2026-04-30`
- scope: `large-dataset cb_svdpp matched-campaign contract after baseline anchors`
- status: `accepted`
- git_commit_at_decision: `8148648`
- git_dirty_at_decision: `false`

## Purpose

Define the exact conditions under which a future `ml10m` or `ml20m`
`cb_svdpp` campaign may be promoted from feasibility evidence to a final
model-comparison candidate. This note updates the earlier large-run deferral
decision after both `ml10m` and `ml20m` gained clean multi-split-seed
`biased_mf` baseline anchors.

This is a run contract, not a benchmark result. The contract has since been satisfied for `ml10m` by the 2026-05-01 clean multi-split-seed benchmark evidence listed below; it remains active for `ml20m`.

## Current Evidence State

Clean baseline anchors now exist for both large datasets:

- `ml10m biased_mf stage0_transfer`, split seeds `1,2,3`, model seed `1`:
  `docs/evidence/models/biased_mf/2026-04-30_ml10m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md`
- `ml20m biased_mf stage0_transfer`, split seeds `1,2,3`, model seed `1`:
  `docs/evidence/models/biased_mf/2026-04-30_ml20m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md`

The `ml10m cb_svdpp` matched campaign has completed:

- `ml10m cb_svdpp stage0_transfer`, split seeds `1,2,3`, model seed `1`,
  20 epochs:
  `docs/evidence/models/cb_svdpp/2026-05-01_ml10m_cb_svdpp_stage0_transfer_clean_multiseed_benchmark.md`

The remaining large-dataset `cb_svdpp` evidence for `ml20m` is still
feasibility-only:

- `ml20m cb_svdpp stage0_probe_e001`, split seed `1`, model seed `1`,
  one epoch:
  `docs/evidence/models/cb_svdpp/2026-04-24_ml20m_cb_svdpp_stage0_probe_e001_feasibility.md`

Therefore, `ml10m` may now use only the bounded matched profile comparison
documented in the 2026-05-01 evidence note. Final `ml20m` model-comparison
claims remain blocked.

## Minimum Campaign Required To Unblock A Dataset

For any remaining dataset-specific `cb_svdpp` promotion, a claim-eligible
campaign requires:

- dataset:
  same processed manifest as the corresponding `biased_mf` baseline anchor
- split family:
  `benchmark_random_v1`
- split contract:
  `train_ratio=0.8`, `validation_ratio=0.1`
- split seeds:
  `1,2,3`
- model seed:
  `1`
- repo state:
  clean `main` before every run and during aggregation
- config state:
  committed model/runtime/device configs before running
- run artifacts:
  one valid run manifest per split seed
- aggregation:
  one `benchmark-random-multiseed` benchmark manifest over exactly those three
  runs
- reporting:
  update `docs/publish_readiness_matrix.md`,
  `docs/report/project_report.md`, and the relevant Evidence note in the same
  change set

A single-seed, single-epoch, dirty, cancelled, or manually aggregated run is not
a final benchmark anchor.

## Local Budget Gate

The measured one-epoch probes remain the only local planning basis. The
following estimates use the same lower-bound extrapolation as the earlier
deferral note: one cluster-induction phase plus `20` measured main epochs.
This is a planning estimate, not a benchmark result.

| Dataset | One-Epoch Effective Fit | 20-Epoch Estimate Per Split | 3-Split Estimate | Peak Memory Probe |
| --- | ---: | ---: | ---: | ---: |
| `ml10m` | `500.849191s` | `8526.898s` / `2.37h` | `7.11h` before overhead | `12730.062500 MB` |
| `ml20m` | `1178.225090s` | `19856.671s` / `5.52h` | `16.54h` before overhead | `17876.066406 MB` |

On the local `local_i5_2500k_24gb` device, an automatic `ml20m cb_svdpp`
matched campaign is not acceptable without an explicit budget decision because
the one-epoch memory probe is close to the repository's 80 percent RAM
guardrail and the three-split time budget is materially larger than a normal
interactive task.

The current split-seed-1-only budget gate is documented in
`docs/evidence/benchmarking/2026-05-01_ml20m_cb_svdpp_matched_campaign_budget_gate.md`.
It does not authorize split seeds `2` or `3` and does not unlock an `ml20m`
model-comparison claim.

After split seed `1` completed, the split-seed-2-only continuation gate was
documented in
`docs/evidence/benchmarking/2026-05-01_ml20m_cb_svdpp_matched_campaign_seed2_gate.md`.
It does not authorize split seed `3`.

After split seed `2` completed, the split-seed-3-only continuation gate was
documented in
`docs/evidence/benchmarking/2026-05-01_ml20m_cb_svdpp_matched_campaign_seed3_gate.md`.
This is the final single-split gate before any canonical aggregation may be
attempted.

Split seed `3` completed as a CLI run, but its measured `peak_memory_mb`
crossed the documented local 80 percent RAM guardrail. The readout is
documented as negative resource evidence in
`docs/evidence/models/cb_svdpp/2026-05-02_ml20m_cb_svdpp_stage0_transfer_seed3_guardrail_breach.md`.
Under this contract, the local `ml20m cb_svdpp stage0_transfer` campaign is
not claim-eligible and must not be folded into the final benchmark matrix.

The completed `ml10m cb_svdpp` campaign confirms that the local profile can run
the 20-epoch three-split campaign, but at multi-hour cost. This observation is
not transferable to `ml20m` because the `ml20m` one-epoch memory and time probes
are materially larger.

## Stop Gates

Abort or demote the campaign to feasibility/negative evidence if any of the
following occurs:

- the repo becomes dirty before aggregation
- a run is interrupted, manually resumed outside the canonical CLI, or lacks a
  valid run manifest
- the effective config differs across split seeds
- peak memory crosses the documented local guardrail or causes system
  instability
- any split seed cannot complete under the declared run profile
- aggregation reuses artifacts that do not match commit, config, dataset,
  split, seed, dtype, and device contracts

If a stop gate triggers, document the result as a bounded failed or negative
run. Do not fold it into the final benchmark matrix.

## Allowed Claims After Completion

If a dataset-specific matched campaign completes, the repo may consider a
bounded comparison claim only for that dataset, split family, model profiles,
seed set, and device profile.

Even after completion, the following remain non-claims unless separate evidence
exists:

- exact paper reproduction
- exact optimizer-faithful CB training
- general model superiority
- unqualified scalability
- unqualified speed claims
- claims about other datasets or model families

## Decision

- Keep final `ml20m` model-comparison claims blocked for the current release
  marker. Guardrail phrase: current release marker.
- Treat the completed local `ml20m cb_svdpp stage0_transfer` split-seed-3 run
  as negative resource evidence because it crossed the memory guardrail.
- Treat `ml10m` as unblocked only for the bounded `biased_mf stage0_transfer`
  versus `cb_svdpp stage0_transfer` profile comparison documented on
  2026-05-01.
- Treat this note as the campaign contract for any future large-dataset
  `cb_svdpp` matched campaign.
- Do not start a local `ml20m cb_svdpp` matched campaign without an explicit
  budget/device decision.
