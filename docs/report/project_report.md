# Project Report

## Title

Reproduction and Engineering Study of Clustering-Based Factorized Collaborative
Filtering

Release marker: `submission-2026-05-02-r10`

## Abstract

This project implements and evaluates a rating-prediction-only recommender
system ladder around matrix factorization and clustering-based extensions. The
active system uses the Unified Experiment Framework: each productive run goes
through `run_unified_experiment`, the `MODEL_REGISTRY`, strict Pydantic model
profiles, `RatingsData`, generated fit artifacts, metrics, profiling, caches,
and run manifests.

The report is claim-limited. It does not evaluate ranking quality, side
features, genres, tags, content metadata, online serving, or production
readiness. Current result tables are generated from `metrics.json` and
`run_manifest.json` artifacts through `scripts/collect_results.py`. The current
controlled runs cover all planned `ml100k`, `ml1m`, and `ml10m` rows. On
`ml20m`, `biased_mf` and `svdpp` completed, while `cb_svdpp` produced a
manifest without `metrics.json`; `cb_asvdpp` was not started after that
resource/time boundary.

## 1. Introduction

Factorized recommender models are useful for studying how predictive structure,
optimization contracts, and system cost interact. The source motivation is the
clustering-based factorized collaborative filtering work by Mirbakhsh and Ling
[1], interpreted alongside Koren's factorization family [2].

This project does not claim exact paper reproduction. The source material does
not fully specify every clustering-based objective and update detail. The repo
therefore implements a documented, source-grounded but repo-defined
optimization contract, with explicit mathematical deviations and claim gates.

## 2. Scope And Non-Scope

The evaluation scope is explicit rating prediction only. The primary metric is
RMSE, with MAE and error-distribution diagnostics reported as supporting rating
metrics. No ranking metrics such as NDCG, Recall@K, HitRate@K, Precision@K, or
Coverage are evaluated in the current claim layer.

No side information is used. Genres, tags, titles, timestamps as features, and
other content metadata are not part of the model inputs. The MovieLens
interactions are used as explicit ratings after the repository's processed data
contract has mapped users, items, and ratings into `RatingsData`.

The current report does not claim:

- general superiority of clustering-based models
- proof that `alpha > 0` creates a meaningful cluster contribution
- ranking performance
- genre, tag, or content-feature usage
- production readiness
- exact paper-faithful optimization
- broad speed or scalability

## 3. Model Ladder

The active model ladder contains exactly six registry-backed models:

1. `biased_mf`
2. `svdpp`
3. `asymmetric_svd`
4. `asvdpp`
5. `cb_svdpp`
6. `cb_asvdpp`

`biased_mf` is the baseline rating factorization model. `svdpp` adds an
implicit item-history block. `asymmetric_svd` and `asvdpp` use item-side user
representations; the accepted `D-003` deviation detaches explicit residual
weights inside each SGD step. `cb_svdpp` and `cb_asvdpp` add train-only cluster
artifacts and fixed cluster assignments.

The `CB` prefix means clustering-based, not content-based.

## 4. Unified Experiment Framework

All productive runs are routed through the Unified Experiment Framework. The
core interfaces are:

- `run_unified_experiment`
- `MODEL_REGISTRY`
- `ModelRequirements`
- `FitArtifacts`
- Pydantic model profile schemas
- `RatingsData`
- train-only history, explicit-feedback, and cluster-artifact builders
- metrics, manifests, profiling, split caches, training-index caches, and
  cluster-artifact caches

The framework keeps model-specific requirements in the registry rather than in
parallel runner branches. CB models request cluster artifacts through
`ModelRequirements`; SVD++-family models request history or explicit feedback
indices where required. This is why cache options are model-specific and why
the runner rejects non-applicable cache combinations.

## 5. Data And Experimental Setup

The official dataset scope is `ml100k`, `ml1m`, `ml10m`, and `ml20m`.
`ml_latest_small` exists only as a local integration dataset and is not part of
the report claim boundary.

Current controlled runs use:

- split family: `benchmark_random_v1`
- train ratio: `0.8`
- validation ratio: `0.1`
- split seed: `1`
- model seed: `1`
- device profile: `local_i5_2500k_24gb`
- dtype: `float32`
- test evaluation enabled

Historical clean multi-seed benchmark anchors remain documented in the evidence
map. The current central results table below is the latest artifact-derived
single-seed run table and is not a replacement for final multi-seed benchmark
claims.

## 6. Tuning Protocol

The tuning protocol is documented in `docs/evaluation_protocol.md`. It is an
inner-validation procedure, not a benchmark claim procedure. Active tuning
currently applies only to `cb_svdpp` and `cb_asvdpp` because the cluster counts
and `alpha` affect both model behavior and resources.

The active selection rule is:

1. lowest mean validation RMSE
2. lower validation RMSE stability/std when means are practically tied
3. lower train time when validation quality and stability do not decide
4. better memory/resource status, with configured resource gates treated as
   hard constraints

`alpha=0` is an explicit ablation candidate. `alpha>0` activates the cluster
channel, but does not by itself prove useful cluster contribution and does not
make a run `cb_claim_eligible`.

## 7. Metric Contract

The central rating metric contract reports:

- RMSE
- MAE
- residual mean and standard deviation
- absolute error p50, p90, p95, and max
- prediction mean, standard deviation, min, and max
- prediction below/above/out-of-range rates

`metrics.json` keeps backward-compatible flat fields such as `train_rmse`,
`validation_rmse`, and `test_rmse`, while also writing structured split metrics
under `metrics.train`, `metrics.validation`, and `metrics.test`.

## 8. Results

### 7.1 Clean Benchmark Anchors

Historical clean multi-seed anchors remain part of the evidence base. They are
not regenerated in this report section, but the corresponding evidence remains
traceable. These rows are the release-facing benchmark anchors guarded by
`docs/publish_readiness_matrix.md`; they are distinct from the current
single-seed artifact table below.

- `ml100k`: clean multi-seed anchors for `biased_mf`, `svdpp`, `cb_svdpp`, and
  `cb_asvdpp`.
- `ml1m`: The clean `ml1m` anchor table is narrower and covers a matched
  `biased_mf` versus `cb_svdpp` comparison.
- `ml10m`: prior clean multi-split-seed evidence covers `biased_mf` and
  `cb_svdpp` transfer profiles.
- `ml20m`: prior clean multi-split-seed evidence covers `biased_mf` only;
  clustering evidence remains feasibility or resource-boundary evidence.

Compatibility evidence references retained for the repo guardrails:

| Dataset | Model | Profile | Evidence / Boundary |
| --- | --- | --- | --- |
| `ml100k` | `biased_mf` | `stage1_tuned` | `docs/evidence/models/biased_mf/2026-04-13_ml100k_biased_mf_inner_tuning_stage1.md`; `docs/evidence/models/biased_mf/2026-04-13_ml100k_biased_mf_stage1_tuned_benchmark.md`; `docs/evidence/models/biased_mf/2026-04-15_ml100k_biased_mf_stage1_tuned_clean_multiseed.md` |
| `ml100k` | `svdpp` | `stage1_tuned` | `docs/evidence/models/svdpp/2026-04-13_ml100k_svdpp_inner_tuning_stage1.md`; `docs/evidence/models/svdpp/2026-04-13_ml100k_svdpp_stage1_tuned_benchmark.md`; `docs/evidence/models/svdpp/2026-04-15_ml100k_svdpp_stage1_tuned_clean_multiseed.md` |
| `ml100k` | `cb_svdpp` | `stage1_tuned` | `docs/evidence/models/cb_svdpp/2026-04-15_ml100k_cb_svdpp_inner_tuning_stage1.md`; `docs/evidence/models/cb_svdpp/2026-04-15_ml100k_cb_svdpp_stage1_tuned_benchmark.md`; `docs/evidence/models/cb_svdpp/2026-04-15_ml100k_cb_svdpp_stage1_tuned_clean_multiseed.md` |
| `ml100k` | `cb_asvdpp` | `stage1_tuned` | `docs/evidence/models/cb_asvdpp/2026-04-15_ml100k_cb_asvdpp_inner_tuning_stage1.md`; `docs/evidence/models/cb_asvdpp/2026-04-15_ml100k_cb_asvdpp_stage1_tuned_benchmark.md`; `docs/evidence/models/cb_asvdpp/2026-04-15_ml100k_cb_asvdpp_stage1_tuned_clean_multiseed.md` |
| `ml1m` | `biased_mf` | `stage0_transfer`, seeds `1,2,3` | `docs/evidence/models/biased_mf/2026-04-21_ml1m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md` |
| `ml1m` | `cb_svdpp` | `stage0_transfer`, seeds `1,2,3` | `docs/evidence/models/cb_svdpp/2026-04-21_ml1m_cb_svdpp_stage0_transfer_clean_multiseed_benchmark.md` |
| `ml10m` | `biased_mf` | `stage0_transfer`, seeds `1,2,3` | Clean baseline anchor: `docs/evidence/models/biased_mf/2026-04-30_ml10m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md` |
| `ml10m` | `cb_svdpp` | `stage0_transfer`, seeds `1,2,3` | Clean matched comparison candidate: higher validation RMSE, higher test RMSE, and materially higher fit cost than matched `biased_mf`; evidence `docs/evidence/models/cb_svdpp/2026-05-01_ml10m_cb_svdpp_stage0_transfer_clean_multiseed_benchmark.md` |
| `ml20m` | `biased_mf` | `stage0_transfer`, seeds `1,2,3` | Clean baseline-only anchor: `docs/evidence/models/biased_mf/2026-04-30_ml20m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md` |

### 7.2 Feasibility, Selection, And Deferral Evidence

Rows in this subsection are evidence-backed, but they are not final benchmark
anchors. They may support feasibility, resource, or selection statements only
and must not be mixed into final model-ranking tables.

The matched-campaign contract estimates remain planning context, not completed
benchmark evidence. The current local evidence still does not support final
`ml20m` model rankings.

The large-dataset `cb_svdpp` run boundary is resource-controlled: prior
matched-campaign estimates placed the local `ml20m cb_svdpp` campaign at
`16.54h` before overhead, and the completed guardrail readout remains negative
resource evidence rather than a publishable comparison row.

| Dataset | Model Profile | Evidence / Boundary |
| --- | --- | --- |
| `ml1m` | `cb_asvdpp stage0` inner tuning | clean reduced-budget selection evidence only: `docs/evidence/models/cb_asvdpp/2026-04-21_ml1m_cb_asvdpp_inner_tuning_stage0.md`; no final `ml1m cb_asvdpp` benchmark-anchor claim follows from this tuning run. |
| `ml10m` | `cb_svdpp stage0_probe_e001` | Historical single-epoch feasibility/resource readout: `docs/evidence/models/cb_svdpp/2026-04-24_ml10m_cb_svdpp_stage0_probe_e001_feasibility.md`; superseded for final `ml10m cb_svdpp` anchoring by the clean matched benchmark. |
| `ml20m` | `cb_svdpp stage0_probe_e001` | Clean single-epoch feasibility/resource readout: `docs/evidence/models/cb_svdpp/2026-04-24_ml20m_cb_svdpp_stage0_probe_e001_feasibility.md`; not comparable to full-budget anchors. |
| `ml20m` | `cb_svdpp stage0_transfer` seed-3 guardrail readout | Negative resource evidence: `docs/evidence/models/cb_svdpp/2026-05-02_ml20m_cb_svdpp_stage0_transfer_seed3_guardrail_breach.md`; not a final benchmark anchor and not eligible for final aggregation under the current campaign contract. |
| `ml20m` | `cb_svdpp current basis-profile attempt` | Current run artifact produced `run_manifest.json` but no `metrics.json`: `artifacts/runs/2026-05-05T235949Z_ml20m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/run_manifest.json`; no RMSE, MAE, runtime, or memory result is reported from this attempt. |

### 7.3 Current Non-Claims

The report does not claim general CB superiority, ranking performance,
side-feature usage, production readiness, exact paper-faithful optimization, or
that `alpha>0` proves a useful cluster contribution. It also does not claim any
final `ml20m` model comparison.

### 8.1 Current Artifact-Derived Rating Results

The following table is produced from run artifacts through
`python scripts/collect_results.py`. Each row lists the artifact path that backs
the central numbers in that row.

| Dataset | Model | Validation RMSE | Test RMSE | Validation MAE | Test MAE | Test abs error p90 | Out-of-range rate | Train time (s) | Peak memory (MB) | Alpha | CB claim eligible | Diagnostic claim ready | Status | Run artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `ml100k` | `biased_mf` | `0.934196` | `0.932512` | `0.730964` | `0.730806` | `1.516285` | `0.000000` | `5.628736` | `230.765625` |  |  |  | `completed` | `artifacts/runs/2026-05-05T071611Z_ml100k_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json` |
| `ml100k` | `svdpp` | `0.922242` | `0.912009` | `0.724252` | `0.715632` | `1.473456` | `0.000000` | `40.199607` | `313.242188` |  |  |  | `completed` | `artifacts/runs/2026-05-05T071638Z_ml100k_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json` |
| `ml100k` | `asymmetric_svd` | `0.922156` | `0.915839` | `0.726031` | `0.720305` | `1.478442` | `0.000000` | `58.307840` | `287.660156` |  |  |  | `completed` | `artifacts/runs/2026-05-05T071738Z_ml100k_asymmetric_svd_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json` |
| `ml100k` | `asvdpp` | `0.956244` | `0.934610` | `0.746389` | `0.726536` | `1.529629` | `0.000000` | `60.640834` | `291.105469` |  |  |  | `completed` | `artifacts/runs/2026-05-05T071857Z_ml100k_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json` |
| `ml100k` | `cb_svdpp` | `0.915354` | `0.910396` | `0.718246` | `0.713885` | `1.469087` | `0.000000` | `51.940735` | `327.375000` | `0.100000` | `false` | `false` | `completed` | `artifacts/runs/2026-05-05T071125Z_ml100k_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json` |
| `ml100k` | `cb_asvdpp` | `0.913416` | `0.910213` | `0.715743` | `0.712082` | `1.471394` | `0.000000` | `129.337318` | `314.617188` | `0.100000` | `false` | `false` | `completed` | `artifacts/runs/2026-05-05T071240Z_ml100k_cb_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json` |
| `ml1m` | `biased_mf` | `0.866678` | `0.868475` | `0.678380` | `0.679173` | `1.409559` | `0.000000` | `19.274583` | `806.570312` |  |  |  | `completed` | `artifacts/runs/2026-05-05T072047Z_ml1m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json` |
| `ml1m` | `svdpp` | `0.880023` | `0.882945` | `0.685434` | `0.686760` | `1.440126` | `0.000000` | `470.972271` | `1161.703125` |  |  |  | `completed` | `artifacts/runs/2026-05-05T072131Z_ml1m_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json` |
| `ml1m` | `asvdpp` | `0.879784` | `0.881981` | `0.685844` | `0.686823` | `1.441329` | `0.000000` | `1102.149094` | `1170.746094` |  |  |  | `completed` | `artifacts/runs/2026-05-05T072946Z_ml1m_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json` |
| `ml1m` | `cb_svdpp` | `0.857911` | `0.859314` | `0.672441` | `0.673340` | `1.392877` | `0.000000` | `809.692809` | `1461.789062` | `0.100000` | `false` | `false` | `completed` | `artifacts/runs/2026-05-05T074832Z_ml1m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json` |
| `ml1m` | `cb_asvdpp` | `0.856611` | `0.858497` | `0.670970` | `0.672333` | `1.392284` | `0.000000` | `2111.574233` | `1418.363281` | `0.100000` | `false` | `false` | `completed` | `artifacts/runs/2026-05-05T080224Z_ml1m_cb_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json` |
| `ml10m` | `biased_mf` | `0.786906` | `0.786747` | `0.602975` | `0.603147` | `1.272961` | `0.000000` | `165.526697` | `6586.140625` |  |  |  | `completed` | `artifacts/runs/2026-05-05T083836Z_ml10m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json` |
| `ml10m` | `svdpp` | `0.793454` | `0.793940` | `0.606447` | `0.607161` | `1.285547` | `0.000000` | `5052.771135` | `9981.566406` |  |  |  | `completed` | `artifacts/runs/2026-05-05T084213Z_ml10m_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json` |
| `ml10m` | `cb_svdpp` | `0.790366` | `0.790454` | `0.606632` | `0.607096` | `1.276449` | `0.000000` | `8517.180747` | `12669.632812` | `0.100000` | `false` | `false` | `completed` | `artifacts/runs/2026-05-05T121413Z_ml10m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json` |
| `ml10m` | `cb_asvdpp` | `0.786807` | `0.787362` | `0.602313` | `0.602819` | `1.273839` | `0.000000` | `21024.462149` | `10130.804688` | `0.100000` | `false` | `false` | `completed` | `artifacts/runs/2026-05-05T143714Z_ml10m_cb_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json` |
| `ml20m` | `biased_mf` | `0.774734` | `0.775594` | `0.590142` | `0.590539` | `1.250785` | `0.000000` | `334.323293` | `13034.949219` |  |  |  | `completed` | `artifacts/runs/2026-05-05T202912Z_ml20m_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json` |
| `ml20m` | `svdpp` | `0.778699` | `0.779429` | `0.592091` | `0.592203` | `1.258909` | `0.000000` | `11874.412670` | `18163.230469` |  |  |  | `completed` | `artifacts/runs/2026-05-05T203616Z_ml20m_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/metrics.json` |
| `ml20m` | `cb_svdpp` |  |  |  |  |  |  |  |  | `0.100000` | `false` |  | `started_missing_metrics` | `artifacts/runs/2026-05-05T235949Z_ml20m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/run_manifest.json` |

`ml20m cb_svdpp` is not a result row. It is listed because the artifact
collector found a manifest, but there is no `metrics.json`; therefore it has no
RMSE, MAE, runtime, or memory result. `ml20m cb_asvdpp` was not started after
this boundary.

## 9. Runtime And Memory Tradeoff

The current controlled runs show that model structure changes system cost
substantially. On `ml10m`, `biased_mf` completed with `165.526697` train-time
seconds and `6586.140625` MB peak memory, while `cb_svdpp` completed with
`8517.180747` train-time seconds and `12669.632812` MB peak memory. These
numbers come from the corresponding `ml10m` artifact rows in Table 8.1.

This is not a broad scalability claim. It is a device- and profile-specific
runtime/memory tradeoff under `local_i5_2500k_24gb`, split seed `1`, model seed
`1`, and the listed configs. Larger CB runs are especially expensive because
their comparable fit time includes train-only cluster induction and main model
training.

On `ml20m`, `biased_mf` and `svdpp` completed. The `cb_svdpp` attempt did not
produce `metrics.json`, so it cannot be used for runtime, memory, RMSE, or MAE.

## 10. CB Semantics

CB semantics are explicit in both manifests and metrics. `alpha=0.1` in the
current CB rows means the cluster channel is configured on. It does not prove
that the cluster contribution is causally useful, and it does not automatically
make a row claim-eligible.

In the current completed CB rows, `cb_claim_eligible=false` and
`diagnostic_claim_ready=false`. This is intentional: the runner distinguishes
between a configured cluster channel, diagnostic artifacts, and a claim-ready
CB result. The report therefore treats CB rows as rating-prediction runs under
the documented repo contract, not as proof of a general clustering benefit.

## 11. Limitations

The main limitations are:

- rating-only evaluation; no ranking metrics were evaluated
- no side features such as genres, tags, or content metadata
- single-seed current controlled runs in the central artifact table
- historical multi-seed anchors exist, but are governed by their own evidence
  files and commits
- exact paper-faithful optimization is not claimed
- `alpha>0` is not evidence of useful clustering by itself
- `ml20m cb_svdpp` is incomplete in the current artifact table
- no production-readiness claim

The local hardware profile is also a material limitation. The `ml20m cb_svdpp`
manifest without metrics is a resource/time-boundary observation, not a model
quality conclusion.

## 12. Appendix: Evidence Map

Current generated run indexes:

- `docs/evidence/runs/ml100k_run_index.md`
- `docs/evidence/runs/ml1m_run_index.md`
- `docs/evidence/runs/ml10m_run_index.md`
- `docs/evidence/runs/ml20m_run_index.md`

Current collector:

- `scripts/collect_results.py`
- `src/recsys_lab/reporting/collect_results.py`

Current config and tuning evidence:

- `docs/evidence/current_config_freeze.md`
- `docs/evaluation_protocol.md`
- `docs/evidence/current_evidence_index.md`

Final claim matrix:

- `docs/publish_readiness_matrix.md`

Large-dataset feasibility and deferral evidence:

- `docs/evidence/data/2026-04-24_ml10m_processed_ingestion.md`
- `docs/evidence/models/biased_mf/2026-04-30_ml10m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md`
- `docs/evidence/models/cb_svdpp/2026-04-24_ml10m_cb_svdpp_stage0_probe_e001_feasibility.md`
- `docs/evidence/models/cb_svdpp/2026-05-01_ml10m_cb_svdpp_stage0_transfer_clean_multiseed_benchmark.md`
- `docs/evidence/data/2026-04-24_ml20m_official_ingestion.md`
- `docs/evidence/models/biased_mf/2026-04-30_ml20m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md`
- `docs/evidence/models/cb_svdpp/2026-04-24_ml20m_cb_svdpp_stage0_probe_e001_feasibility.md`
- `docs/evidence/models/cb_svdpp/2026-05-02_ml20m_cb_svdpp_stage0_transfer_seed3_guardrail_breach.md`
- `docs/evidence/benchmarking/2026-04-24_large_cb_svdpp_deeper_run_deferral.md`
- `docs/evidence/benchmarking/2026-04-30_large_cb_svdpp_matched_campaign_contract.md`

Historical but superseded `ml1m` exploratory evidence:

- `docs/evidence/models/biased_mf/2026-04-16_ml1m_biased_mf_stage0_transfer_clean_seed_sweep.md`
- `docs/evidence/models/cb_svdpp/2026-04-16_ml1m_cb_svdpp_stage0_transfer_clean_seed_readout.md`

## 13. Conclusion

The repository now contains a unified, registry-backed implementation of the
six-model factorization ladder with strict configs, manifests, structured
rating metrics, profiling, and cache metadata. Current controlled runs provide
artifact-backed rating-prediction tables for `ml100k`, `ml1m`, and `ml10m`, and
partial `ml20m` coverage for `biased_mf` and `svdpp`.

The evidence supports careful, bounded statements about the observed runs and
their runtime/memory costs. It does not support general CB superiority,
ranking-performance claims, side-feature claims, or production-readiness
claims.

## 14. References

[1] N. Mirbakhsh and C. X. Ling. Clustering-Based Factorized Collaborative
Filtering. RecSys 2013 Poster.

[2] Y. Koren. Factorization Meets the Neighborhood: a Multifaceted
Collaborative Filtering Model. KDD 2008.

[3] F. M. Harper and J. A. Konstan. The MovieLens Datasets: History and
Context. ACM Transactions on Interactive Intelligent Systems, 2015.
