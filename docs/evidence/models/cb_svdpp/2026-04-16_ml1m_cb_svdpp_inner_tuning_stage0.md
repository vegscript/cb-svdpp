# Evidence Note

## Scope

First reduced-budget `stage0` inner-tuning study for `cb_svdpp` on `MovieLens 1M`
under the canonical `benchmark_random_v1` split family.

## Claim Or Question

Which locally feasible `cb_svdpp` candidate should be carried forward on `ml1m`
before paying for a full `20`-epoch confirmatory run?

## Inputs And Artifacts

- tuning config:
  `configs/experiments/tuning/ml1m_cb_svdpp_stage0.yaml`
- processed dataset manifest:
  `data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json`
- base model config:
  `configs/models/tuned/ml1m_cb_svdpp_stage0_transfer.yaml`
- tuning benchmark directory:
  `artifacts/benchmarks/2026-04-16T054923Z_ml1m_inner_tuning_cb_svdpp_stage0_local_i5_2500k_24gb/`
- benchmark manifest validation: `valid`
- run manifest validation: `6/6 valid`

## Method

- evaluate `3` candidate profiles under a bounded `2`-epoch budget
- use split seeds `1` and `2`
- fix model seed `1`
- use the canonical random benchmark split contract with `train_ratio=0.8`,
  `validation_ratio=0.1`, and an unused holdout test remainder
- disable test evaluation during selection to keep the study purely
  validation-driven
- compare candidates by validation RMSE mean, then validation RMSE std, then
  effective fit time mean
- measure effective fit time as cluster induction plus main training so that the
  clustering overhead is not hidden

## Readout

- benchmark status: `completed`
- Git state: dirty
- claimability: development-only selection evidence, not benchmark-final
- measured candidate runs: `6`
- benchmark wall clock: about `50.5` minutes
- winner: `rank064_uc080_ic080_a010_lr0075_reg0025_e002`
- winning validation RMSE mean: `0.914705`
- winning validation RMSE std: `0.000766`
- winning effective fit time mean seconds: `456.33`

Candidate ranking:

| Rank | Candidate | Validation RMSE Mean | Validation RMSE Std | Effective Fit Time Mean (s) |
| --- | --- | ---: | ---: | ---: |
| 1 | `rank064_uc080_ic080_a010_lr0075_reg0025_e002` | 0.914705 | 0.000766 | 456.33 |
| 2 | `rank064_uc064_ic064_a015_lr0075_reg0025_e002` | 0.915358 | 0.001009 | 484.08 |
| 3 | `rank064_uc080_ic080_a015_lr0075_reg0025_e002` | 0.915431 | 0.000779 | 479.63 |

## Interpretation

The reduced-budget tuning study did produce a consistent winner, but the result
is conservative rather than surprising: the best candidate is exactly the same
hyperparameter profile already stored in
`configs/models/tuned/ml1m_cb_svdpp_stage0_transfer.yaml`, namely rank `64`,
`80/80` clusters, `alpha=0.10`, learning rate `0.0075`, and regularization
`0.025`.

This means the tuning study does not justify promoting a new `ml1m` `cb_svdpp`
config. Instead, it narrows uncertainty around the transferred profile and shows
that the tested `alpha=0.15` and `64/64` alternatives do not beat it even under
a shorter local budget.

Because the benchmark was executed in a dirty workspace, the result is valid as
development evidence for candidate selection, but not as a benchmark-final claim.

## Decision Or Next Step

- keep `configs/models/tuned/ml1m_cb_svdpp_stage0_transfer.yaml` as the active
  `ml1m` stage-0 `cb_svdpp` candidate
- do not create a duplicate promoted config for the same hyperparameter profile
- next methodologically clean step: execute the full `20`-epoch confirmatory
  run for that selected profile from a clean git snapshot