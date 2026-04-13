# Evidence Note

## Scope

Official `u1` to `u5` `MovieLens 100K` benchmark for the promoted
`stage1_tuned` `svdpp` profile.

## Claim Or Question

Does the first promoted `stage1_tuned` `svdpp` config materially improve over
the draft baseline and over the tuned `biased_mf` anchor on the official outer
test folds?

## Inputs And Artifacts

- tuned model config:
  `configs/models/tuned/ml100k_svdpp_stage1.yaml`
- comparison baseline benchmark:
  `artifacts/benchmarks/2026-04-13T035011Z_ml100k_paper_faithful_svdpp_local_i5_2500k_24gb/`
- tuned benchmark directory:
  `artifacts/benchmarks/2026-04-13T061611Z_ml100k_paper_faithful_svdpp_local_i5_2500k_24gb/`
- tuned `biased_mf` benchmark:
  `artifacts/benchmarks/2026-04-13T060536Z_ml100k_paper_faithful_biased_mf_local_i5_2500k_24gb/`

## Method

- use the official outer folds `u1` to `u5`
- train one run per fold with the promoted `stage1_tuned` profile
- compare aggregate test RMSE and training time against the current draft
  `svdpp` benchmark and the current tuned `biased_mf` benchmark

## Readout

- status: `completed`
- test RMSE mean: `0.923483`
- test RMSE std: `0.008131`
- training wall-clock mean: `658.866663` seconds
- training wall-clock std: `286.055341` seconds
- absolute improvement vs. draft baseline mean test RMSE: `0.021046`
- absolute improvement vs. tuned `biased_mf` mean test RMSE: `0.013978`
- relative training-time multiplier vs. tuned `biased_mf`: about `5.68x`

## Interpretation

The first promoted `stage1_tuned` `svdpp` profile materially improves over both
the draft `svdpp` baseline and the tuned `biased_mf` anchor. This is the first
point in the repo where the stronger implicit-feedback model family begins to
look convincingly competitive rather than merely directionally correct.

The training-time variance is still high across folds, so the cost conclusion
is directionally strong but not yet performance-final. The accuracy result is
much stronger than before and points clearly toward hyperparameter debt rather
than a fundamentally broken `svdpp` implementation.

## Decision Or Next Step

- keep this benchmark as the current tuned `svdpp` anchor on `ml100k`
- next step should be a broader or more systematic search and then a repeat
  benchmark with multiple model seeds
