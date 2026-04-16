# Evidence Note

## Scope

Clean multi-seed official `u1` to `u5` `MovieLens 100K` benchmark for the
promoted `stage1_tuned` `cb_svdpp` profile.

## Claim Or Question

Does `cb_svdpp` remain stronger than the current clean `biased_mf` and `svdpp`
anchors when the official outer benchmark is confirmed across model seeds
`1,2,3` on an identical clean Git state?

## Inputs And Artifacts

- tuned model config:
  `configs/models/tuned/ml100k_cb_svdpp_stage1.yaml`
- clean seed benchmarks:
  - `artifacts/benchmarks/2026-04-15T124046Z_ml100k_paper_faithful_cb_svdpp_local_i5_2500k_24gb/`
  - `artifacts/benchmarks/2026-04-15T135449Z_ml100k_paper_faithful_cb_svdpp_local_i5_2500k_24gb/`
  - `artifacts/benchmarks/2026-04-15T143114Z_ml100k_paper_faithful_cb_svdpp_local_i5_2500k_24gb/`
- clean multi-seed benchmark:
  `artifacts/benchmarks/2026-04-15T150207Z_ml100k_paper_faithful_cb_svdpp_multiseed_s001_s002_s003_local_i5_2500k_24gb/`
- clean tuned `biased_mf` multi-seed anchor:
  `artifacts/benchmarks/2026-04-13T130402Z_ml100k_paper_faithful_biased_mf_multiseed_s001_s002_s003_local_i5_2500k_24gb/`
- clean tuned `svdpp` multi-seed anchor:
  `artifacts/benchmarks/2026-04-15T030305Z_ml100k_paper_faithful_svdpp_multiseed_s001_s002_s003_local_i5_2500k_24gb/`

## Method

- use the official outer folds `u1` to `u5`
- run clean official seed benchmarks for model seeds `1`, `2`, and `3`
- execute all seed benchmarks from an isolated clean clone at commit `d76e9d4`
- require `git.dirty=false` for every seed benchmark and every fold run
- aggregate seeds only via explicit `benchmark_manifest_paths`
- use seed-level means as the primary clean multi-seed comparison unit

## Readout

- status: `completed`
- git commit: `d76e9d44693420d40f2125a68cfda013853e7168`
- git dirty: `false`
- model seeds: `1,2,3`
- seed-level test RMSE mean: `0.918968`
- seed-level test RMSE std: `0.000917`
- seed-level fit-time mean: `357.33` seconds
- fold-run-level test RMSE mean: `0.918968`
- fold-run-level test RMSE std: `0.006624`
- absolute improvement vs. clean tuned `biased_mf`: `0.018143`
- absolute improvement vs. clean tuned `svdpp`: `0.005047`
- fit-time multiplier vs. clean tuned `biased_mf`: about `1.29x`
- fit-time multiplier vs. clean tuned `svdpp`: about `0.26x`

## Interpretation

This is the first clean multi-seed official `cb_svdpp` benchmark in the
repository, and it materially strengthens the earlier CB claim. The clustering
extension now remains ahead of both clean tuned anchor families even after
multi-seed confirmation.

The result is especially important on the quality-cost axis. Clean multi-seed
`cb_svdpp` is not only better than clean multi-seed `svdpp` by `0.0050` RMSE;
it also runs at only about one quarter of `svdpp`'s mean fit time on the
default local CPU target.

At the same time, the repo must stay conservative about generalization. This is
currently a clean `ml100k` result, not yet a claim about `ml1m` or larger
datasets, and not yet a proof that the same ordering survives broader CB search
spaces.

## Decision Or Next Step

- promote this artifact to the current clean official `cb_svdpp` anchor on
  `ml100k`
- use this anchor for future CB-family comparisons and for `cb_asvdpp`
  evaluation
- next step should be either a broader clean `cb_svdpp` search or clean
  `cb_asvdpp` implementation and benchmarking on the same protocol
