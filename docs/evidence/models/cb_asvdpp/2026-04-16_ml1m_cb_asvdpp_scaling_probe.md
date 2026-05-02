# Evidence Note

## Scope

First `ml1m` scaling study for the clustering-based residual family, covering a
cancelled full-transfer `cb_asvdpp` attempt and a completed bounded probe.

## Claim Or Question

Can the transferred `cb_asvdpp` profile from `ml100k` be executed practically on
`ml1m` on the default local device, and if not, what bounded probe best
quantifies the true runtime scale?

## Inputs And Artifacts

- processed dataset manifest:
  `data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- cancelled full-transfer config:
  `configs/models/tuned/ml1m_cb_asvdpp_stage0_transfer.yaml`
- probe config:
  `configs/models/tuned/ml1m_cb_asvdpp_stage0_probe_e003.yaml`
- cancelled run directory:
  `artifacts/runs/2026-04-16T013743Z_ml1m_cb_asvdpp_local_i5_2500k_24gb_s001/`
- probe run directory:
  `artifacts/runs/2026-04-16T025201Z_ml1m_cb_asvdpp_local_i5_2500k_24gb_s001/`

## Method

- first attempt the direct `ml100k` transfer profile with `20` epochs to test end-to-end feasibility
- treat the timed-out full run as non-claimable and mark it explicitly as `cancelled`
- then run a bounded `3`-epoch probe under the same split, clustering, dtype, and seed contract
- use the measured cluster induction and mean epoch duration to estimate the cost of the original full-transfer profile

## Readout

### Cancelled Full Transfer

- status: `cancelled`
- manifest validation: `valid`
- reason: external timeout on the default local device
- claimability: none

### Completed Probe

- status: `completed`
- manifest validation: `valid`
- Git state: dirty
- epochs: `3`
- train RMSE: `0.879001`
- validation RMSE: `0.901483`
- test RMSE: `0.904128`
- cluster induction wall clock seconds: `171.937315`
- main training wall clock seconds: `1578.600502`
- effective fit time seconds: `1750.537817`
- mean epoch duration seconds: `526.200167`
- peak memory MB: `1465.628906`
- peak memory delta MB: `1255.710938`
- projected full-transfer fit time for `20` epochs: about `10695.94` seconds, or about `2.97` hours

## Interpretation

The transferred `cb_asvdpp` profile is operationally too expensive for direct
full-scale `ml1m` execution on the default local device. The bounded probe is
still valuable because it quantifies the real scale rather than leaving the
failed full run as an anecdote.

At the probe budget, `cb_asvdpp` remains much worse than the current fully run
`ml1m` `biased_mf` baseline in both validation and test RMSE. That comparison is
not a fair final model comparison because the budgets differ materially, but it
is enough to show that blind transfer of the full `ml100k` winner is not an
efficient next step on this hardware.

## Decision Or Next Step

- do not rerun the full `20`-epoch transfer profile on the default local device without a stronger reason
- treat the bounded probe as the current trustworthy scaling signal for `cb_asvdpp` on `ml1m`
- prefer either reduced-budget `ml1m` tuning or a stronger device before attempting a full `cb_asvdpp` transfer benchmark again
