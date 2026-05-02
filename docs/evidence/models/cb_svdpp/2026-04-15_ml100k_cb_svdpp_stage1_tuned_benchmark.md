# Evidence Note

## Scope

Official `u1` to `u5` `MovieLens 100K` benchmark for the promoted
`stage1_tuned` `cb_svdpp` profile.

## Claim Or Question

Does the first promoted `stage1_tuned` `cb_svdpp` config materially improve over
the draft `cb_svdpp` baseline, and how does it compare to the current clean
`biased_mf` and `svdpp` anchors?

## Inputs And Artifacts

- tuned model config:
  `configs/models/tuned/ml100k_cb_svdpp_stage1.yaml`
- inner tuning study:
  `artifacts/benchmarks/2026-04-15T074345Z_ml100k_inner_tuning_cb_svdpp_stage1_local_i5_2500k_24gb/`
- draft official `cb_svdpp` benchmark:
  `artifacts/benchmarks/2026-04-15T043209Z_ml100k_paper_faithful_cb_svdpp_local_i5_2500k_24gb/`
- tuned official benchmark:
  `artifacts/benchmarks/2026-04-15T085851Z_ml100k_paper_faithful_cb_svdpp_local_i5_2500k_24gb/`
- clean tuned `biased_mf` anchor:
  `artifacts/benchmarks/2026-04-13T130402Z_ml100k_paper_faithful_biased_mf_multiseed_s001_s002_s003_local_i5_2500k_24gb/`
- clean tuned `svdpp` anchor:
  `artifacts/benchmarks/2026-04-15T030305Z_ml100k_paper_faithful_svdpp_multiseed_s001_s002_s003_local_i5_2500k_24gb/`

## Method

- use the official outer folds `u1` to `u5`
- run one benchmark pass with model seed `1`
- use benchmark fit time that includes cluster induction plus main training
- compare against:
  - the draft official `cb_svdpp` readout
  - the current clean tuned `biased_mf` anchor
  - the current clean tuned `svdpp` anchor

## Readout

- status: `completed`
- git commit: `59d9e8b2d20ffea789f9e7fc95ebc21a9dc61b30`
- git dirty: `true`
- test RMSE mean: `0.919292`
- test RMSE std: `0.007626`
- fit-time mean: `510.15` seconds
- absolute improvement vs. draft `cb_svdpp`: `0.006609`
- absolute improvement vs. clean tuned `biased_mf`: `0.017819`
- absolute improvement vs. clean tuned `svdpp`: `0.004723`
- fit-time multiplier vs. draft `cb_svdpp`: about `1.18x`
- fit-time multiplier vs. clean tuned `biased_mf`: about `1.84x`
- fit-time multiplier vs. clean tuned `svdpp`: about `0.37x`

## Interpretation

This is the strongest `cb_svdpp` result currently present in the repo. The
stage1-tuned profile materially improves over the draft official `cb_svdpp`
baseline and, on this dirty one-seed readout, it also outperforms the current
clean tuned `svdpp` anchor while staying much cheaper than `svdpp`.

That is a strong research signal, but not yet a clean final claim. The benchmark
was executed with `git.dirty=true`, and the result is still only a one-seed
outer readout.

## Decision Or Next Step

- keep this benchmark as the current provisional tuned `cb_svdpp` result
- do not promote it to a clean anchor yet
- next step should be a clean rerun and then multi-seed confirmation
