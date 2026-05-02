# Evidence Note

## Scope

Clean multi-seed official `u1` to `u5` `MovieLens 100K` benchmark for the
promoted `stage1_tuned` `cb_asvdpp` profile.

## Claim Or Question

Does `cb_asvdpp` remain stronger than the current clean `biased_mf`, `svdpp`,
and `cb_svdpp` anchors when the official outer benchmark is confirmed across
model seeds `1,2,3` on an identical clean Git state?

## Inputs And Artifacts

- tuned model config:
  `configs/models/tuned/ml100k_cb_asvdpp_stage1.yaml`
- clean seed benchmarks:
  - `artifacts/benchmarks/2026-04-15T194023Z_ml100k_paper_faithful_cb_asvdpp_local_i5_2500k_24gb/`
  - `artifacts/benchmarks/2026-04-15T202913Z_ml100k_paper_faithful_cb_asvdpp_local_i5_2500k_24gb/`
  - `artifacts/benchmarks/2026-04-15T210922Z_ml100k_paper_faithful_cb_asvdpp_local_i5_2500k_24gb/`
- clean multi-seed benchmark:
  `artifacts/benchmarks/2026-04-15T215411Z_ml100k_paper_faithful_cb_asvdpp_multiseed_s001_s002_s003_local_i5_2500k_24gb/`
- clean tuned `biased_mf` multi-seed anchor:
  `artifacts/benchmarks/2026-04-13T130402Z_ml100k_paper_faithful_biased_mf_multiseed_s001_s002_s003_local_i5_2500k_24gb/`
- clean tuned `svdpp` multi-seed anchor:
  `artifacts/benchmarks/2026-04-15T030305Z_ml100k_paper_faithful_svdpp_multiseed_s001_s002_s003_local_i5_2500k_24gb/`
- clean tuned `cb_svdpp` multi-seed anchor:
  `artifacts/benchmarks/2026-04-15T150207Z_ml100k_paper_faithful_cb_svdpp_multiseed_s001_s002_s003_local_i5_2500k_24gb/`

## Method

- use the official outer folds `u1` to `u5`
- run clean official seed benchmarks for model seeds `1`, `2`, and `3`
- execute all seed benchmarks from an isolated clean clone at commit `3fc9993`
- require `git.dirty=false` for every seed benchmark and every fold run
- aggregate seeds only via explicit `benchmark_manifest_paths`
- use seed-level means as the primary clean multi-seed comparison unit

## Readout

- status: `completed`
- git commit: `3fc9993580de0ff8daa1091f157fc2cedb20470d`
- git dirty: `false`
- model seeds: `1,2,3`
- seed-level test RMSE mean: `0.916839`
- seed-level test RMSE std: `0.001334`
- fold-run-level test RMSE mean: `0.916839`
- fold-run-level test RMSE std: `0.006572`
- seed-level training wall clock mean seconds: `477.95`
- absolute improvement vs. clean multi-seed `biased_mf`: `0.020272`
- absolute improvement vs. clean multi-seed `svdpp`: `0.007176`
- absolute improvement vs. clean multi-seed `cb_svdpp`: `0.002129`
- fit-time multiplier vs. clean multi-seed `biased_mf`: about `1.72x`
- fit-time multiplier vs. clean multi-seed `svdpp`: about `0.34x`
- fit-time multiplier vs. clean multi-seed `cb_svdpp`: about `1.34x`

## Interpretation

This is the first clean multi-seed official `cb_asvdpp` benchmark in the
repository, and it materially strengthens the earlier single-seed signal. The
composed CB family now remains ahead of the current clean `biased_mf`,
`svdpp`, and `cb_svdpp` anchors even after multi-seed confirmation.

The result is especially important because it changes the clean official
ordering under the repository's current modeling contracts. `cb_asvdpp` is no
longer only a plausible extension; it is now the best-performing clean
multi-seed official model on `ml100k` presently documented in the repo.

The repo must still stay precise about scope. This is:

- a clean official `ml100k` result under the current repo contracts
- not yet a claim about `ml1m` or larger datasets
- not an optimizer-faithful paper-reproduction claim because `D-003` remains
  active

## Decision Or Next Step

- promote this artifact to the current clean multi-seed official `cb_asvdpp`
  anchor on `ml100k`
- use this anchor as the leading CB-family reference under the current repo
  contracts
- next step should be either broader `cb_asvdpp` search or scale-out to `ml1m`
