# Project Report

## Title

Reproduction and Engineering Study of Clustering-Based Factorized Collaborative
Filtering

## Abstract

This project rebuilds a clustering-based factorized collaborative filtering
pipeline for explicit rating prediction. The repository implements a layered
model family from `biased_mf` through `svdpp`, `asymmetric_svd`, `asvdpp`,
`cb_svdpp`, and `cb_asvdpp`, and evaluates the claimable subset on official
MovieLens datasets from 100K to 20M interactions. The final report uses only
evidence-backed claims from the frozen publish-readiness matrix. Clean
multi-seed benchmark anchors exist for the full current `ml100k` ladder, for a
matched `ml1m` comparison between `biased_mf` and `cb_svdpp`, and for a matched
`ml10m` comparison between the documented `biased_mf` and `cb_svdpp` transfer
profiles. `ml20m` has a clean `biased_mf` baseline anchor, while its
clustering-model row remains feasibility and negative resource evidence only.

## 1. Introduction

Factorized recommender models remain useful for studying the relationship
between predictive quality, model structure, and computational cost. They are
also a suitable project target because their mathematical components can be
specified explicitly and implemented without relying on black-box recommender
libraries.

The source motivation is the clustering-based factorized collaborative
filtering work by Mirbakhsh and Ling [1], interpreted alongside Koren's
factorization family [2]. The project goal is not to assert exact reproduction
where the source material is under-specified. Instead, the repository defines a
controlled research platform with explicit data contracts, mathematical
specifications, run manifests, and claim gates.

## 2. Research Objective And Scope

The objective is to reconstruct and extend the paper-family model ladder under
one evaluation and engineering contract:

1. `biased_mf`
2. `svdpp`
3. `asymmetric_svd`
4. `asvdpp`
5. `cb_svdpp`
6. `cb_asvdpp`

The official dataset scope is fixed to `ml100k`, `ml1m`, `ml10m`, and `ml20m`.
`ml_latest_small` is retained only as a local proof-of-concept and integration
dataset.

The current report claims are bounded by
`docs/publish_readiness_matrix.md`. In particular, `paper-faithful`,
`exact optimizer-faithful`, `reproduced`, `scalable`, `ready`, and unqualified
`faster` claims are not used. The clustering-based models are reported as
source-grounded predictors with repo-defined optimization under the accepted
`D-004` boundary.

## 3. Data

All official datasets are prepared through canonical raw and processed dataset
contracts. Processed interactions use deterministic dense user and item indices,
typed ratings, and dataset manifests that record preprocessing family, dtype,
counts, rating range, and artifact paths.

| Dataset | Interactions | Users | Rated Items | Catalog Items | Data Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| `ml100k` | 100000 | 943 | 1682 | 1682 | `docs/evidence/data/2026-04-13_ml100k_official_ingestion.md` |
| `ml1m` | 1000209 | 6040 | 3706 | 3883 | `docs/evidence/data/2026-04-16_ml1m_official_ingestion.md` |
| `ml10m` | 10000054 | 69878 | 10677 | 10681 | `docs/evidence/data/2026-04-24_ml10m_processed_ingestion.md` |
| `ml20m` | 20000263 | 138493 | 26744 | 27278 | `docs/evidence/data/2026-04-24_ml20m_official_ingestion.md` |

The `ml100k` path additionally supports the official GroupLens `u1` to `u5`
folds for the `paper_faithful_ml100k_v1` split family. The larger datasets use
the repository's canonical `benchmark_random_v1` split family for the current
evidence state.

## 4. Methodology

The methodology separates model mathematics, split policy, and claim policy.
Mathematical specifications live in `docs/math/`; deviations from source
papers live in `docs/methodology/deviations_from_paper.md`; evidence-backed
claim boundaries live in `docs/publish_readiness_matrix.md`.

`biased_mf` is the baseline factorization model with user and item bias terms.
`svdpp` adds an implicit item-history block. `asymmetric_svd` and `asvdpp`
introduce item-side user representations. The accepted `D-003` deviation
detaches explicit residual weights inside each SGD step for the asymmetric
families, so those models are not reported as exact optimizer-faithful
implementations.

For `cb_svdpp` and `cb_asvdpp`, user and item clusters are induced only from
training-derived latent vectors. Cluster assignments remain fixed during the
main model fit. `R_star` is computed as a train-only cluster diagnostic rather
than as a separate optimization target. This is the accepted `D-004` contract:
source-grounded predictor, repo-defined optimization.

Evaluation uses RMSE as the primary rating-prediction metric. System metrics
include fit time, inference time, throughput, peak memory, and model size when
available. For clustering models, comparable fit time includes mandatory
train-only cluster induction plus main training.

## 5. System Design And Implementation

The repository is structured as a research platform rather than a notebook
collection:

- `src/` contains active implementation code.
- `tests/` contains unit, integration, and documentation guardrails.
- `configs/` contains dataset, runtime, model, and experiment profiles.
- `docs/` contains governance, math, evidence, and report material.
- `artifacts/` contains generated run and benchmark outputs.

Run and benchmark manifests are mandatory for claimable evidence. They record
the command, cwd, git commit, dirty status, dataset, split family, seeds,
runtime profile, dtype, threading, and output artifact paths. This makes each
reported result traceable to a concrete execution context.

The default local device profile is CPU-first:

- CPU: Intel Core i5-2500K
- logical threads: 4
- RAM: 24 GB
- GPU: GT 1030, not part of the default hot path
- Python: 3.10.7
- dtype default for performance runs: `float32`

## 6. Experimental Setup

`ml100k` benchmark anchors use the official `u1` to `u5` outer folds under
`paper_faithful_ml100k_v1`. Clean multi-seed anchors use model seeds `1,2,3`
and aggregate seed-level results from clean benchmark manifests.

`ml1m` benchmark anchors use `benchmark_random_v1`, split seeds `1,2,3`, and
model seed `1`. The current `ml1m` comparison is valid only for matched
`biased_mf` and `cb_svdpp` profiles on the same clean commit.

`ml10m` currently has processed data evidence plus clean three-split-seed
`biased_mf` and `cb_svdpp` anchors under the same split-seed and model-seed
contract. `ml20m` has processed data evidence plus a clean three-split-seed
`biased_mf` baseline anchor; its local `cb_svdpp` matched-campaign attempt
crossed the memory guardrail and is therefore not a final comparison anchor.

## 7. Results

### 7.1 Clean Benchmark Anchors

The clean `ml100k` anchor table covers the full current benchmark ladder. All
rows are clean, three-seed official-fold results under the current repo
contracts.

| Dataset | Model | Profile | Seeds | Test RMSE Mean | Seed Std | Fit Time Mean (s) | Git State | Evidence |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `ml100k` | `biased_mf` | `stage1_tuned` | `1,2,3` | 0.937111 | 0.001492 | 277.24 | `fb1fcbc`, clean | `docs/evidence/models/biased_mf/2026-04-15_ml100k_biased_mf_stage1_tuned_clean_multiseed.md` |
| `ml100k` | `svdpp` | `stage1_tuned` | `1,2,3` | 0.924015 | 0.000461 | 1386.62 | `fb1fcbc`, clean | `docs/evidence/models/svdpp/2026-04-15_ml100k_svdpp_stage1_tuned_clean_multiseed.md` |
| `ml100k` | `cb_svdpp` | `stage1_tuned` | `1,2,3` | 0.918968 | 0.000917 | 357.33 | `d76e9d4`, clean | `docs/evidence/models/cb_svdpp/2026-04-15_ml100k_cb_svdpp_stage1_tuned_clean_multiseed.md` |
| `ml100k` | `cb_asvdpp` | `stage1_tuned` | `1,2,3` | 0.916839 | 0.001334 | 477.95 | `3fc9993`, clean | `docs/evidence/models/cb_asvdpp/2026-04-15_ml100k_cb_asvdpp_stage1_tuned_clean_multiseed.md` |

Within this `ml100k` evidence boundary, the two clustering-based models have
lower mean test RMSE than the non-clustering anchors. `cb_asvdpp` has the
lowest documented `ml100k` RMSE among the current clean anchor set. This is not
an exact paper-reproduction claim and not a larger-dataset claim.

The clean `ml1m` anchor table is narrower. It supports only the matched
`biased_mf` versus `cb_svdpp` comparison.

| Dataset | Model | Profile | Seeds | Validation RMSE Mean | Test RMSE Mean | Fit Time Mean (s) | Peak Memory Mean (MB) | Git State | Evidence |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `ml1m` | `biased_mf` | `stage0_transfer`, seeds `1,2,3` | split seeds `1,2,3`, model seed `1` | 0.866357 | 0.866615 | 20.47 | 777.09 | `a9a45b9`, clean | `docs/evidence/models/biased_mf/2026-04-21_ml1m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md` |
| `ml1m` | `cb_svdpp` | `stage0_transfer`, seeds `1,2,3` | split seeds `1,2,3`, model seed `1` | 0.857005 | 0.857365 | 1082.96 | 1455.98 | `a9a45b9`, clean | `docs/evidence/models/cb_svdpp/2026-04-21_ml1m_cb_svdpp_stage0_transfer_clean_multiseed_benchmark.md` |

Under this matched `ml1m` contract, `cb_svdpp` has lower validation and test
RMSE than `biased_mf`, but at a much higher fit-time cost. This supports a
bounded quality-cost tradeoff statement for this pair only.

The large-dataset anchor table now includes a matched `ml10m` comparison for
the documented `biased_mf` and `cb_svdpp` transfer profiles. `ml20m` remains
baseline-only because its local clustering-model matched-campaign attempt
breached the memory guardrail.

| Dataset | Model | Profile | Seeds | Validation RMSE Mean | Test RMSE Mean | Fit Time Mean (s) | Peak Memory Mean (MB) | Git State | Evidence |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `ml10m` | `biased_mf` | `stage0_transfer` | split seeds `1,2,3`, model seed `1` | 0.787190 | 0.787738 | 147.45 | 6583.41 | `bbe5f81`, clean | `docs/evidence/models/biased_mf/2026-04-30_ml10m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md` |
| `ml10m` | `cb_svdpp` | `stage0_transfer` | split seeds `1,2,3`, model seed `1` | 0.790782 | 0.791315 | 8986.11 | 12701.16 | `b709049`, clean | `docs/evidence/models/cb_svdpp/2026-05-01_ml10m_cb_svdpp_stage0_transfer_clean_multiseed_benchmark.md` |
| `ml20m` | `biased_mf` | `stage0_transfer` | split seeds `1,2,3`, model seed `1` | 0.775339 | 0.775803 | 323.44 | 13029.49 | `e9ce60e`, clean | `docs/evidence/models/biased_mf/2026-04-30_ml20m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md` |

Under the matched `ml10m` contract, the documented `cb_svdpp` profile has
higher validation RMSE, higher test RMSE, materially higher fit cost, and
higher peak memory than the matched `biased_mf` baseline. This is a bounded
profile-specific result, not a general claim against clustering models.

### 7.2 Feasibility, Selection, And Deferral Evidence

The following evidence is useful for project interpretation but is not a final
benchmark anchor.

| Dataset | Model / Study | Evidence Type | Central Readout | Evidence |
| --- | --- | --- | --- | --- |
| `ml1m` | `cb_asvdpp stage0` | clean reduced-budget selection | best validation RMSE margin only `0.000040`; no profile promoted | `docs/evidence/models/cb_asvdpp/2026-04-21_ml1m_cb_asvdpp_inner_tuning_stage0.md` |
| `ml10m` | `cb_svdpp stage0_probe_e001` | historical single-epoch feasibility, superseded for anchoring | validation RMSE `0.872094`, effective fit time `500.849191s`, peak memory `12730.062500 MB` | `docs/evidence/models/cb_svdpp/2026-04-24_ml10m_cb_svdpp_stage0_probe_e001_feasibility.md` |
| `ml20m` | `cb_svdpp stage0_probe_e001` | single-epoch feasibility | validation RMSE `0.863001`, effective fit time `1178.225090s`, peak memory `17876.066406 MB` | `docs/evidence/models/cb_svdpp/2026-04-24_ml20m_cb_svdpp_stage0_probe_e001_feasibility.md` |
| `ml20m` | `cb_svdpp stage0_transfer` seed-3 readout | negative resource evidence | validation RMSE `0.781010`, test RMSE `0.781511`, fit time `20365.517578s`, peak memory `19898.871094 MB`; crosses local 80 percent RAM guardrail | `docs/evidence/models/cb_svdpp/2026-05-02_ml20m_cb_svdpp_stage0_transfer_seed3_guardrail_breach.md` |

The large-dataset evidence now includes clean `ml10m` matched `biased_mf` and
`cb_svdpp` anchors, a clean `ml20m biased_mf` baseline anchor, and bounded
CB-SVD++ feasibility and resource evidence. It supports only the documented
`ml10m` profile comparison and still does not support final `ml20m` model
rankings.
The current large-dataset `cb_svdpp` run boundary is documented in
`docs/evidence/benchmarking/2026-04-30_large_cb_svdpp_matched_campaign_contract.md`.
It requires clean split seeds `1,2,3`, a committed run profile, canonical
multi-seed aggregation, and explicit budget/device approval before any
remaining large `cb_svdpp` campaign can be treated as a final comparison
candidate. The local `ml20m cb_svdpp stage0_transfer` attempt triggered this
boundary because split seed `3` crossed the memory guardrail.

### 7.3 Current Non-Claims

The report intentionally does not claim:

- exact paper reproduction
- exact optimizer-faithful CB training
- final `ml20m` model-comparison results
- general `ml10m` model-family superiority beyond the documented matched
  profile comparison
- `ml1m cb_asvdpp` benchmark status
- unqualified speed, scalability, or publish-readiness
- superiority of any model outside the specific datasets, splits, seeds, and
  evidence rows listed above

## 8. Discussion

The clean `ml100k` results support the modeling intuition that adding
implicit-feedback and clustering structure can reduce rating-prediction error
under controlled conditions. The current lowest `ml100k` RMSE belongs to
`cb_asvdpp`, but that model's optimizer boundary remains repo-defined.

The `ml1m` result is narrower but important: the matched clean comparison shows
`cb_svdpp` below `biased_mf` in RMSE, with a substantial systems cost. This is a
quality-cost tradeoff, not a blanket model-family ranking.

The large-dataset probes and anchors prevent a silent scope drop. `ml10m` and
`ml20m` remain in scope, have processed data evidence, and now have clean
baseline anchors. The completed `ml10m cb_svdpp` campaign is especially useful
because it shows that the transferred clustering profile is not automatically
better at larger scale: under the matched local contract it has higher RMSE and
much higher fit cost than `biased_mf`. `ml20m` now additionally shows that the
same local CB profile is resource-risky at 20M scale: the seed-3 run completed
but crossed the local memory guardrail. Any future `ml20m` model-comparison
claim requires a stronger device profile or a lower-memory matched profile.

## 9. Limitations

The poster source does not fully specify the regularized CB objective,
`R_star` optimization role, or exact update rules. The repository resolves this
with a documented v1 contract, but that contract limits claim wording.

The current final benchmark evidence is strongest on `ml100k`, narrower on
`ml1m`, profile-specific on `ml10m`, and baseline-only on `ml20m`. Ranking
metrics such as NDCG or Recall@K are not currently part of the claimable
evidence layer.

Hardware also matters. The local 24 GB CPU-first profile completed the
three-split `ml10m cb_svdpp` campaign with peak memory mean `12701.16 MB` and
fit-time mean `8986.11s` per split, which is usable but costly. The
`ml20m cb_svdpp` one-epoch probe already reached `17876.066406 MB` peak memory,
which is below but close to the repository's 80 percent RAM guardrail. The
subsequent local matched-campaign attempt reached split seed `3`, but that run
reported `19898.871094 MB` peak memory, above the approximately `19647.29 MB`
80 percent guardrail on the local 24 GB profile. This is negative resource
evidence, not a claimable `ml20m cb_svdpp` benchmark anchor.
The matched-campaign contract estimates of about `16.54h` before overhead
remain planning context only; the completed seed-3 memory readout is the
binding local viability evidence.

## 10. Conclusion

The repository now contains a modular, tested, evidence-backed implementation
of the target factorization ladder. The clean benchmark evidence supports a
full current `ml100k` anchor table, a matched `ml1m` `biased_mf` versus
`cb_svdpp` comparison, and a matched `ml10m` profile comparison. On `ml10m`,
the documented `cb_svdpp` transfer profile has higher RMSE and much higher fit
cost than the matched `biased_mf` baseline. The large MovieLens datasets remain
in scope; `ml20m` still has only a clean `biased_mf` baseline anchor plus
CB-SVD++ feasibility and negative resource evidence.

For the current handoff, the repository is a claim-limited release candidate.
The release marker is `submission-2026-05-02-r10`, and any final interpretation
must follow the claim matrix in `docs/publish_readiness_matrix.md`. Additional
large-dataset benchmark campaigns, especially `ml20m cb_svdpp`, should only be
started if a stronger device profile or documented lower-memory matched profile
is available first.

## 11. References

[1] N. Mirbakhsh and C. X. Ling. Clustering-Based Factorized Collaborative
Filtering. RecSys 2013 Poster.

[2] Y. Koren. Factorization Meets the Neighborhood: a Multifaceted
Collaborative Filtering Model. KDD 2008.

[3] F. M. Harper and J. A. Konstan. The MovieLens Datasets: History and
Context. ACM Transactions on Interactive Intelligent Systems, 2015.

## 12. Appendix: Evidence Map

Final claim matrix:

- `docs/publish_readiness_matrix.md`

`ml100k` selection and anchor evidence:

- `docs/evidence/models/biased_mf/2026-04-13_ml100k_biased_mf_inner_tuning_stage1.md`
- `docs/evidence/models/biased_mf/2026-04-13_ml100k_biased_mf_stage1_tuned_benchmark.md`
- `docs/evidence/models/biased_mf/2026-04-15_ml100k_biased_mf_stage1_tuned_clean_multiseed.md`
- `docs/evidence/models/svdpp/2026-04-13_ml100k_svdpp_inner_tuning_stage1.md`
- `docs/evidence/models/svdpp/2026-04-13_ml100k_svdpp_stage1_tuned_benchmark.md`
- `docs/evidence/models/svdpp/2026-04-15_ml100k_svdpp_stage1_tuned_clean_multiseed.md`
- `docs/evidence/models/cb_svdpp/2026-04-15_ml100k_cb_svdpp_inner_tuning_stage1.md`
- `docs/evidence/models/cb_svdpp/2026-04-15_ml100k_cb_svdpp_stage1_tuned_benchmark.md`
- `docs/evidence/models/cb_svdpp/2026-04-15_ml100k_cb_svdpp_stage1_tuned_clean_multiseed.md`
- `docs/evidence/models/cb_asvdpp/2026-04-15_ml100k_cb_asvdpp_inner_tuning_stage1.md`
- `docs/evidence/models/cb_asvdpp/2026-04-15_ml100k_cb_asvdpp_stage1_tuned_benchmark.md`
- `docs/evidence/models/cb_asvdpp/2026-04-15_ml100k_cb_asvdpp_stage1_tuned_clean_multiseed.md`

`ml1m` current anchor and selection evidence:

- `docs/evidence/models/biased_mf/2026-04-21_ml1m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md`
- `docs/evidence/models/cb_svdpp/2026-04-21_ml1m_cb_svdpp_stage0_transfer_clean_multiseed_benchmark.md`
- `docs/evidence/models/cb_asvdpp/2026-04-21_ml1m_cb_asvdpp_inner_tuning_stage0.md`

Historical but superseded `ml1m` exploratory evidence:

- `docs/evidence/models/biased_mf/2026-04-16_ml1m_biased_mf_stage0_transfer_clean_seed_sweep.md`
- `docs/evidence/models/cb_svdpp/2026-04-16_ml1m_cb_svdpp_stage0_transfer_clean_seed_readout.md`

Large-dataset feasibility and deferral evidence:

- `docs/evidence/data/2026-04-24_ml10m_processed_ingestion.md`
- `docs/evidence/models/biased_mf/2026-04-24_ml10m_biased_mf_stage0_transfer_feasibility.md`
- `docs/evidence/models/biased_mf/2026-04-30_ml10m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md`
- `docs/evidence/models/cb_svdpp/2026-04-24_ml10m_cb_svdpp_stage0_probe_e001_feasibility.md`
- `docs/evidence/models/cb_svdpp/2026-05-01_ml10m_cb_svdpp_stage0_transfer_clean_multiseed_benchmark.md`
- `docs/evidence/data/2026-04-24_ml20m_official_ingestion.md`
- `docs/evidence/models/biased_mf/2026-04-24_ml20m_biased_mf_stage0_transfer_feasibility.md`
- `docs/evidence/models/biased_mf/2026-04-30_ml20m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md`
- `docs/evidence/models/cb_svdpp/2026-04-24_ml20m_cb_svdpp_stage0_probe_e001_feasibility.md`
- `docs/evidence/models/cb_svdpp/2026-05-02_ml20m_cb_svdpp_stage0_transfer_seed3_guardrail_breach.md`
- `docs/evidence/benchmarking/2026-04-24_large_cb_svdpp_deeper_run_deferral.md`
- `docs/evidence/benchmarking/2026-04-30_large_cb_svdpp_matched_campaign_contract.md`
