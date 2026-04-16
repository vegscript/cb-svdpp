# Evidence Note

## Scope

Clean official `u1` to `u5` `MovieLens 100K` benchmark for the promoted
`stage1_tuned` `cb_svdpp` profile.

## Claim Or Question

Can the previously observed strong `stage1_tuned` `cb_svdpp` signal be
confirmed on a genuinely clean Git state, and how does the clean readout compare
to the current clean `biased_mf` and `svdpp` anchors?

## Inputs And Artifacts

- tuned model config:
  `configs/models/tuned/ml100k_cb_svdpp_stage1.yaml`
- clean official benchmark:
  `artifacts/benchmarks/2026-04-15T124046Z_ml100k_paper_faithful_cb_svdpp_local_i5_2500k_24gb/`
- supporting fold runs:
  - `artifacts/runs/2026-04-15T124054Z_ml100k_cb_svdpp_local_i5_2500k_24gb_s001/`
  - `artifacts/runs/2026-04-15T124816Z_ml100k_cb_svdpp_local_i5_2500k_24gb_s001/`
  - `artifacts/runs/2026-04-15T125430Z_ml100k_cb_svdpp_local_i5_2500k_24gb_s001/`
  - `artifacts/runs/2026-04-15T130144Z_ml100k_cb_svdpp_local_i5_2500k_24gb_s001/`
  - `artifacts/runs/2026-04-15T131021Z_ml100k_cb_svdpp_local_i5_2500k_24gb_s001/`
- prior provisional tuned benchmark:
  `artifacts/benchmarks/2026-04-15T085851Z_ml100k_paper_faithful_cb_svdpp_local_i5_2500k_24gb/`
- clean tuned `biased_mf` multi-seed anchor:
  `artifacts/benchmarks/2026-04-13T130402Z_ml100k_paper_faithful_biased_mf_multiseed_s001_s002_s003_local_i5_2500k_24gb/`
- clean tuned `svdpp` multi-seed anchor:
  `artifacts/benchmarks/2026-04-15T030305Z_ml100k_paper_faithful_svdpp_multiseed_s001_s002_s003_local_i5_2500k_24gb/`

## Method

- use the official outer folds `u1` to `u5`
- run one benchmark pass with model seed `1`
- execute the benchmark from an isolated clean clone at commit `d76e9d4`
- require `git.dirty=false` both in the benchmark manifest and in every fold
  run manifest
- use benchmark fit time that includes cluster induction plus main training

## Readout

- status: `completed`
- git commit: `d76e9d44693420d40f2125a68cfda013853e7168`
- git dirty: `false`
- all five fold run manifests: `dirty=false`
- test RMSE mean: `0.919292`
- test RMSE std: `0.007626`
- fit-time mean: `406.41` seconds
- absolute delta vs. prior provisional tuned benchmark: `0.000000`
- absolute improvement vs. clean tuned `biased_mf`: `0.017819`
- absolute improvement vs. clean tuned `svdpp`: `0.004723`
- fit-time multiplier vs. clean tuned `biased_mf`: about `1.47x`
- fit-time multiplier vs. clean tuned `svdpp`: about `0.29x`

## Interpretation

The clean rerun confirms the earlier provisional signal exactly on predictive
quality: the clean benchmark reproduces the same mean and standard deviation as
the earlier dirty `stage1_tuned` readout.

This is now the strongest clean official `cb_svdpp` result in the repository.
It outperforms the current clean `biased_mf` and `svdpp` anchors on `ml100k`
while remaining far cheaper than clean tuned `svdpp`.

The result is still not a final CB anchor because the outer benchmark has only
been confirmed for one model seed. The next methodological step is therefore
multi-seed confirmation on the same frozen config path.

## Decision Or Next Step

- promote this benchmark from provisional to clean single-seed official
  `cb_svdpp` evidence
- keep the earlier dirty benchmark only as historical precursor evidence
- next step should be clean multi-seed confirmation for `cb_svdpp`
