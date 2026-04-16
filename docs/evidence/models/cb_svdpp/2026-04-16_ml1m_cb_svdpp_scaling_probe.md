# Evidence Note

## Scope

First `ml1m` scaling study for `cb_svdpp`, covering a cancelled full-transfer
attempt and a completed bounded probe.

## Claim Or Question

Is `cb_svdpp` a more practical clustering-based scaling path than `cb_asvdpp` on
`ml1m` for the default local device, and what runtime should be expected from
the full transferred profile?

## Inputs And Artifacts

- processed dataset manifest:
  `data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- cancelled full-transfer config:
  `configs/models/tuned/ml1m_cb_svdpp_stage0_transfer.yaml`
- probe config:
  `configs/models/tuned/ml1m_cb_svdpp_stage0_probe_e003.yaml`
- cancelled run directory:
  `artifacts/runs/2026-04-16T032643Z_ml1m_cb_svdpp_local_i5_2500k_24gb_s001/`
- probe run directory:
  `artifacts/runs/2026-04-16T043108Z_ml1m_cb_svdpp_local_i5_2500k_24gb_s001/`

## Method

- first attempt the direct `ml100k` transfer profile with `20` epochs to test full local feasibility
- mark the timed-out full run as `cancelled` and non-claimable
- then run a bounded `3`-epoch probe under the same split, clustering, dtype, and seed contract
- derive a practical full-transfer runtime estimate from cluster induction plus mean epoch duration

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
- train RMSE: `0.886755`
- validation RMSE: `0.907440`
- test RMSE: `0.910138`
- cluster induction wall clock seconds: `167.661262`
- main training wall clock seconds: `614.684187`
- effective fit time seconds: `782.345448`
- mean epoch duration seconds: `204.894729`
- peak memory MB: `1461.585938`
- peak memory delta MB: `1251.890625`
- projected full-transfer fit time for `20` epochs: about `4265.56` seconds, or about `71.09` minutes

## Interpretation

`cb_svdpp` is materially more practical than `cb_asvdpp` on `ml1m` for the
default local device. Under the same bounded `3`-epoch budget, it finishes in
less than half the fit time of `cb_asvdpp` and projects to a full-transfer
runtime of about `1.18` hours rather than about `2.97` hours.

Even so, the full transferred profile still exceeds the current one-hour local
runtime budget used for interactive scaling steps. At the probe budget, it also
remains clearly worse than the fully trained `ml1m` `biased_mf` baseline. The
important positive result is therefore not yet quality, but tractability: if a
clustering-based `ml1m` path is pursued on this hardware, `cb_svdpp` is the
more practical family to tune first.

## Decision Or Next Step

- treat `cb_svdpp` as the more viable clustering-based `ml1m` candidate on the default local device
- do not reuse the cancelled full-transfer run for any benchmark claim
- next step: run a reduced-budget `ml1m` tuning study for `cb_svdpp`, or move the full transfer to a stronger device
