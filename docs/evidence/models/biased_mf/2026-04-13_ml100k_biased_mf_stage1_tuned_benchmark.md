# Evidence Note

## Scope

Official `u1` to `u5` `MovieLens 100K` benchmark for the promoted
`stage1_tuned` `biased_mf` profile.

## Claim Or Question

Does the first promoted `stage1_tuned` `biased_mf` config materially improve
over the draft baseline on the official outer test folds?

## Inputs And Artifacts

- tuned model config:
  `configs/models/tuned/ml100k_biased_mf_stage1.yaml`
- comparison baseline benchmark:
  `artifacts/benchmarks/2026-04-13T034022Z_ml100k_paper_faithful_biased_mf_local_i5_2500k_24gb/`
- tuned benchmark directory:
  `artifacts/benchmarks/2026-04-13T060536Z_ml100k_paper_faithful_biased_mf_local_i5_2500k_24gb/`

## Method

- use the official outer folds `u1` to `u5`
- train one run per fold with the promoted `stage1_tuned` profile
- compare aggregate test RMSE and training time against the current draft
  `biased_mf` benchmark

## Readout

- status: `completed`
- test RMSE mean: `0.937461`
- test RMSE std: `0.007740`
- training wall-clock mean: `115.907983` seconds
- training wall-clock std: `3.715289` seconds
- absolute improvement vs. draft baseline mean test RMSE: `0.014969`
- relative training-time multiplier vs. draft baseline: about `1.09x`

## Interpretation

The first promoted `stage1_tuned` `biased_mf` profile materially improves over
the draft baseline on the official outer test folds. This is a strong signal
that the earlier gap was not primarily caused by a broken `biased_mf`
implementation, but by weak default hyperparameters.

This result is still not benchmark-final because the tuning stage used only
outer folds `u1/u2` and one seed, but it is already a credible improvement
under a no-leakage protocol.

## Decision Or Next Step

- keep this benchmark as the current tuned `biased_mf` anchor on `ml100k`
- compare tuned `svdpp` against this tuned baseline, not against the old draft
  baseline
