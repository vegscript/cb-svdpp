# Publish Readiness Matrix

- last_assessed: `2026-05-03`
- canonical_plan: `docs/project_master_plan.md`
- status: `release_candidate_claim_limited`
- current_branch_requirement: `main`

## Purpose

This matrix operationalizes the publish readiness gates from
`docs/project_master_plan.md`. It is a control document, not a report section.
Its job is to show which gates are satisfied, which are blocked, and what the
next action must be.

Status vocabulary:

- `pass`: requirement is satisfied by current repo evidence
- `partial`: requirement is partly satisfied but not enough for publish
- `blocked`: requirement cannot pass until missing work is completed
- `pending`: requirement has not yet been executed or documented

## Gate Summary

| Gate | Status | Current Readout | Required Next Action |
| --- | --- | --- | --- |
| Gate 1: Scope Freeze | `pass` | Official scope is fixed as `ml100k`, `ml1m`, `ml10m`, and `ml20m`; `ml10m` and `ml20m` may not be silently removed. | Keep this scope visible in README, report, and final claim matrix. |
| Gate 2: Dataset Evidence | `pass` | `ml100k`, `ml1m`, `ml10m`, and `ml20m` have processed manifests and data evidence. | Keep dataset evidence links visible in the final claim matrix and report appendix. |
| Gate 3: Benchmark Evidence | `partial` | Clean benchmark anchors exist for `ml100k`; `ml100k cb_svdpp` also has a completed G6 validation-only selection, a frozen selected config, and a completed clean outer benchmark readout under `benchmark_random_v1`; clean `ml1m` anchors exist for `biased_mf` and `cb_svdpp`; `ml1m cb_asvdpp` is selection-only; `ml100k cb_asvdpp` has a bounded hot-path profiling decision, an exact-remediation contract, and a clean pre-change baseline but no post-change remediation evidence yet; `ml10m` has matched clean `biased_mf` and `cb_svdpp` multi-split-seed anchors; `ml20m` has a clean `biased_mf` baseline anchor plus `cb_svdpp` feasibility and a local matched-campaign attempt that breached the memory guardrail. | Keep final `ml20m` model-comparison claims blocked until a stronger device profile or lower-memory matched profile has clean evidence; restrict `ml10m` claims to the documented `biased_mf` vs `cb_svdpp` profile comparison; keep the new G6 `ml100k cb_svdpp` readout separate from older `paper_faithful_ml100k_v1` anchors; implement `cb_asvdpp` remediation only under the exactness and before/after gates. |
| Gate 4: Claim Freeze | `pass` | The final claim matrix exists below and separates benchmark anchors from selection and feasibility evidence. | Keep the matrix synchronized whenever new evidence is added; the report may only use claims explicitly allowed below. |
| Gate 5: Report Ready | `pass` | Report is condensed around final claim boundaries, clean benchmark anchors, feasibility evidence, limitations, and an evidence map. | Keep future report edits integrated into the existing sections; do not reintroduce chronological work-log sections. |
| Gate 6: Reproduction Ready | `pass` | Current `main` is clean; `uv.lock` is present; `uv sync --extra dev --locked` completed after dev type-stub updates; Ruff, Mypy, focused regression tests, and the full test suite pass from the `uv` environment. | Keep `uv.lock` versioned and rerun the same setup/smoke/quality sequence before final tagging if dependencies change. |
| Gate 7: Release Hygiene | `pass` | README and report state release scope, setup path, claimable results, artifact/data policy, non-claims, and release marker `submission-2026-05-02-r10`. | Keep release-facing language synchronized if claim boundaries or dependencies change. |

## Dataset Evidence Matrix

| Dataset | Publish Scope | Processed Manifest | Data Evidence | Benchmark Evidence | Current Publish Status | Next Action |
| --- | --- | --- | --- | --- | --- | --- |
| `ml100k` | `in_scope` | `pass` | `pass` | `pass_for_current_anchor_set_plus_g6_outer_benchmark` | `benchmark_evidence_ready_g6_outer_anchor_documented` | Keep current clean anchors for final claims; the G6 `cb_svdpp` outer benchmark may be used only as its own `benchmark_random_v1` readout and must not be merged into older `paper_faithful_ml100k_v1` comparisons. |
| `ml1m` | `in_scope` | `pass` | `pass` | `partial` | `benchmark_evidence_partial` | Keep `biased_mf` and `cb_svdpp` as clean anchors; keep `cb_asvdpp` selection-only unless an outer benchmark is run. |
| `ml10m` | `in_scope` | `pass` | `pass` | `matched_biased_mf_cb_svdpp_anchor` | `matched_profile_comparison_ready` | Data evidence: `docs/evidence/data/2026-04-24_ml10m_processed_ingestion.md`; baseline evidence: `docs/evidence/models/biased_mf/2026-04-30_ml10m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md`; matched CB evidence: `docs/evidence/models/cb_svdpp/2026-05-01_ml10m_cb_svdpp_stage0_transfer_clean_multiseed_benchmark.md`; historical CB feasibility: `docs/evidence/models/cb_svdpp/2026-04-24_ml10m_cb_svdpp_stage0_probe_e001_feasibility.md`. |
| `ml20m` | `in_scope` | `pass` | `pass` | `partial_baseline_anchor_plus_cb_negative_resource_evidence` | `baseline_anchor_ready_model_comparison_blocked` | Data evidence: `docs/evidence/data/2026-04-24_ml20m_official_ingestion.md`; baseline evidence: `docs/evidence/models/biased_mf/2026-04-30_ml20m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md`; CB feasibility: `docs/evidence/models/cb_svdpp/2026-04-24_ml20m_cb_svdpp_stage0_probe_e001_feasibility.md`; CB guardrail breach: `docs/evidence/models/cb_svdpp/2026-05-02_ml20m_cb_svdpp_stage0_transfer_seed3_guardrail_breach.md`; campaign contract: `docs/evidence/benchmarking/2026-04-30_large_cb_svdpp_matched_campaign_contract.md`. |

## Model Benchmark Matrix

| Dataset | `biased_mf` | `svdpp` | `cb_svdpp` | `cb_asvdpp` | Claim Boundary |
| --- | --- | --- | --- | --- | --- |
| `ml100k` | `clean_multiseed_anchor` | `clean_multiseed_anchor` | `clean_multiseed_anchor_plus_g6_outer_readout` | `clean_multiseed_anchor_plus_hotpath_decision_only` | Final claims may use the existing clean anchors if seed count and evidence links are stated; the G6 `cb_svdpp` readout must stay under `benchmark_random_v1`; the `cb_asvdpp` hot-path note is profiling evidence only, not remediation evidence. |
| `ml1m` | `clean_multiseed_anchor` | `missing_anchor` | `clean_multiseed_anchor` | `clean_selection_only` | Final cross-model claims are currently valid only for matched `biased_mf` vs `cb_svdpp`; `cb_asvdpp` is not benchmark-final. |
| `ml10m` | `clean_multiseed_baseline_anchor` | `missing` | `clean_multiseed_matched_anchor` | `missing` | Bounded matched `biased_mf` vs `cb_svdpp` comparison is allowed for the documented `stage0_transfer` profiles only. |
| `ml20m` | `clean_multiseed_baseline_anchor` | `missing` | `guardrail_breached_local_campaign` | `missing` | Baseline-only `biased_mf` statements are allowed; no final model-comparison claim is allowed because the local CB matched-campaign attempt breached the memory guardrail. |

Note: `asymmetric_svd` and `asvdpp` are implemented and tested, but they are
not currently part of the clean final anchor set. They must not be introduced
into final comparison tables unless matching evidence is added or the report
labels them as implementation/POC context only.

## Final Claim Matrix

This is the Gate 4 claim freeze for the current evidence state. Rows in this
section are the only rows that may support final benchmark claims in the report.
Every claim must stay within the stated dataset, split, seed, commit, and metric
boundary.

| Dataset | Model Profile | Split / Seeds | Git State | Central Metrics | Evidence | Allowed Claim | Explicit Non-Claims |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ml100k` | `biased_mf stage1_tuned` | `paper_faithful_ml100k_v1`, folds `u1-u5`, model seeds `1,2,3` | `fb1fcbc`, clean | test RMSE mean `0.937111`, seed std `0.001492`, train time mean `277.24s` | `docs/evidence/models/biased_mf/2026-04-15_ml100k_biased_mf_stage1_tuned_clean_multiseed.md` | Clean three-seed `ml100k` baseline anchor under the repo's current benchmark contract. | Not a claim about larger datasets, exact paper reproduction, or scalability. |
| `ml100k` | `svdpp stage1_tuned` | `paper_faithful_ml100k_v1`, folds `u1-u5`, model seeds `1,2,3` | `fb1fcbc`, clean | test RMSE mean `0.924015`, seed std `0.000461`, train time mean `1386.62s` | `docs/evidence/models/svdpp/2026-04-15_ml100k_svdpp_stage1_tuned_clean_multiseed.md` | Clean three-seed `ml100k` anchor; lower RMSE than matched `biased_mf` on this benchmark family. | Not a `faster` claim and not a claim outside `ml100k`. |
| `ml100k` | `cb_svdpp stage1_tuned` | `paper_faithful_ml100k_v1`, folds `u1-u5`, model seeds `1,2,3` | `d76e9d4`, clean | test RMSE mean `0.918968`, seed std `0.000917`, fit time mean `357.33s` | `docs/evidence/models/cb_svdpp/2026-04-15_ml100k_cb_svdpp_stage1_tuned_clean_multiseed.md` | Clean three-seed `ml100k` CB-SVD++ anchor; lower RMSE than the clean `biased_mf` and `svdpp` anchors under current repo contracts. | Not exact optimizer-faithful paper reproduction; not a claim about `ml1m`, `ml10m`, or `ml20m`. |
| `ml100k` | `cb_svdpp g6_validation_selected` | `benchmark_random_v1`, split seeds `1,2,3`, model seed `1` | `67570ed`, clean | validation RMSE mean `0.9566122815305916`, test RMSE mean `0.9595668222022953`, training wall-clock mean `4.767408333330725s`, peak memory mean `237.70703125 MB` | `docs/evidence/reproduction/2026-05-03_cb_svdpp_g6_outer_benchmark_run.md` | Clean outer `ml100k cb_svdpp` benchmark readout for the frozen G6-selected two-epoch profile under `benchmark_random_v1`. | Not comparable to the older `paper_faithful_ml100k_v1` anchor rows without stating the split-family difference; not a speed, scalability, SOTA, production-readiness, paper-faithfulness, or larger-dataset claim. |
| `ml100k` | `cb_asvdpp stage1_tuned` | `paper_faithful_ml100k_v1`, folds `u1-u5`, model seeds `1,2,3` | `3fc9993`, clean | test RMSE mean `0.916839`, seed std `0.001334`, train time mean `477.95s` | `docs/evidence/models/cb_asvdpp/2026-04-15_ml100k_cb_asvdpp_stage1_tuned_clean_multiseed.md` | Clean three-seed `ml100k` CB-ASVD++ anchor; lowest documented `ml100k` RMSE among the current clean anchor set. | Not exact paper reproduction because `D-003` and `D-004` remain active; not a larger-dataset claim. |
| `ml1m` | `biased_mf stage0_transfer` | `benchmark_random_v1`, split seeds `1,2,3`, model seed `1` | `a9a45b9`, clean | validation RMSE mean `0.866357`, test RMSE mean `0.866615`, train time mean `20.47s` | `docs/evidence/models/biased_mf/2026-04-21_ml1m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md` | Clean three-split-seed `ml1m` baseline anchor under `benchmark_random_v1`. | Not tuned on `ml1m` as a final optimum and not comparable to missing model families. |
| `ml1m` | `cb_svdpp stage0_transfer` | `benchmark_random_v1`, split seeds `1,2,3`, model seed `1` | `a9a45b9`, clean | validation RMSE mean `0.857005`, test RMSE mean `0.857365`, fit time mean `1082.96s` | `docs/evidence/models/cb_svdpp/2026-04-21_ml1m_cb_svdpp_stage0_transfer_clean_multiseed_benchmark.md` | Clean matched `ml1m` comparison: lower validation and test RMSE than matched `biased_mf`, with materially higher fit cost. | Not a blanket model-family superiority claim; not a speed, scalability, `svdpp`, `cb_asvdpp`, `ml10m`, or `ml20m` claim. |
| `ml10m` | `biased_mf stage0_transfer` | `benchmark_random_v1`, split seeds `1,2,3`, model seed `1` | `bbe5f81`, clean | validation RMSE mean `0.787190`, test RMSE mean `0.787738`, train time mean `147.45s`, peak memory mean `6583.41 MB` | `docs/evidence/models/biased_mf/2026-04-30_ml10m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md` | Clean three-split-seed `ml10m` baseline anchor under `benchmark_random_v1`. | Not tuned on `ml10m`, not a model-comparison claim, not a CB claim, and not a scalability claim. |
| `ml10m` | `cb_svdpp stage0_transfer` | `benchmark_random_v1`, split seeds `1,2,3`, model seed `1` | `b709049`, clean | validation RMSE mean `0.790782`, test RMSE mean `0.791315`, fit time mean `8986.11s`, peak memory mean `12701.16 MB` | `docs/evidence/models/cb_svdpp/2026-05-01_ml10m_cb_svdpp_stage0_transfer_clean_multiseed_benchmark.md` | Clean matched `ml10m` comparison candidate; under this exact profile, `cb_svdpp` has higher validation RMSE, higher test RMSE, and materially higher fit cost than the matched `biased_mf` baseline. | Not tuned on `ml10m`, not a model-family superiority claim, not a speed or scalability claim, and not evidence for `ml20m`. |
| `ml20m` | `biased_mf stage0_transfer` | `benchmark_random_v1`, split seeds `1,2,3`, model seed `1` | `e9ce60e`, clean | validation RMSE mean `0.775339`, test RMSE mean `0.775803`, train time mean `323.44s`, peak memory mean `13029.49 MB` | `docs/evidence/models/biased_mf/2026-04-30_ml20m_biased_mf_stage0_transfer_clean_multiseed_benchmark.md` | Clean three-split-seed `ml20m` baseline anchor under `benchmark_random_v1`. | Not tuned on `ml20m`, not a model-comparison claim, not a CB claim, and not a scalability claim. |

## Feasibility And Selection Evidence

Rows in this section are evidence-backed, but they are not final benchmark
anchors. They may support feasibility, resource, or selection statements only.
They must not be mixed into final model-ranking tables.

| Dataset | Model Profile | Split / Seeds | Git State | Central Metrics | Evidence | Allowed Claim | Explicit Non-Claims |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ml1m` | `cb_asvdpp stage0` inner tuning | `benchmark_random_v1`, split seeds `1,2`, model seed `1`, `2` epochs | `e515d20`, clean | best validation RMSE margin `0.000040`; selected candidate is not promoted | `docs/evidence/models/cb_asvdpp/2026-04-21_ml1m_cb_asvdpp_inner_tuning_stage0.md` | Clean reduced-budget selection evidence only. | Not a benchmark anchor and not a final `ml1m cb_asvdpp` quality claim. |
| `ml100k` | `cb_svdpp g6_validation_selected` | `benchmark_random_v1`, split seeds `1,2,3`, model seed `1`, `12` candidates, `2` epochs | `9a48336`, clean | selected validation RMSE mean `0.9566122815305916`, validation RMSE std `0.002372317660216579`, non-null `test_rmse` count `0` | `docs/evidence/reproduction/2026-05-03_cb_svdpp_g6_validation_grid_run.md` | Clean validation-only selection evidence; selected config is frozen at `configs/models/tuned/ml100k_cb_svdpp_g6_validation_selected.yaml` and was later evaluated by the documented clean outer benchmark. | Not itself a benchmark anchor, not a test-set result, not a speed or scalability claim, and not a claim outside `ml100k`. |
| `ml10m` | `cb_svdpp stage0_probe_e001` | `benchmark_random_v1`, split seed `1`, model seed `1`, `1` epoch | `c1d2e1d`, clean | validation RMSE `0.872094`, test RMSE `0.871385`, effective fit time `500.849191s`, peak memory `12730.062500 MB` | `docs/evidence/models/cb_svdpp/2026-04-24_ml10m_cb_svdpp_stage0_probe_e001_feasibility.md` | Historical single-epoch `ml10m` clustering feasibility/resource readout; superseded for final `ml10m cb_svdpp` anchoring by the clean 2026-05-01 matched benchmark. | Not a final result row and not comparable to full-budget anchors. |
| `ml20m` | `cb_svdpp stage0_probe_e001` | `benchmark_random_v1`, split seed `1`, model seed `1`, `1` epoch | `6ccef25`, clean | validation RMSE `0.863001`, test RMSE `0.863991`, effective fit time `1178.225090s`, peak memory `17876.066406 MB` | `docs/evidence/models/cb_svdpp/2026-04-24_ml20m_cb_svdpp_stage0_probe_e001_feasibility.md` | Clean single-epoch `ml20m` clustering feasibility/resource readout on the local 24 GB profile. | Not comparable to the `biased_mf` baseline quality because the epoch and budget are unmatched; not a final CB benchmark. |
| `ml20m` | `cb_svdpp stage0_transfer` seed-3 guardrail readout | `benchmark_random_v1`, split seed `3`, model seed `1`, `20` epochs | `1cb39de`, clean | validation RMSE `0.781010`, test RMSE `0.781511`, fit time `20365.517578s`, peak memory `19898.871094 MB` | `docs/evidence/models/cb_svdpp/2026-05-02_ml20m_cb_svdpp_stage0_transfer_seed3_guardrail_breach.md` | Negative resource evidence for the local 24 GB profile; the run completed but crossed the 80 percent RAM guardrail. | Not a final benchmark anchor, not a final `ml20m` model-comparison claim, and not eligible for final aggregation under the current campaign contract. |

## Global Claim Locks

- The repository is a claim-limited release candidate, not an unconstrained
  paper-faithful reproduction.
- No `paper-faithful`, `exact optimizer-faithful`, `reproduced`, `scalable`,
  `ready`, or unqualified `faster` claim is allowed.
- No final `ml20m` model-comparison claim is allowed with the current evidence
  because the attempted local clustering-model matched campaign breached the
  memory guardrail before claim-eligible aggregation.
- `ml10m` model-comparison claims are allowed only for the documented
  `biased_mf stage0_transfer` versus `cb_svdpp stage0_transfer` matched
  profiles, split family, seed set, and device profile.
- Further local `ml20m cb_svdpp` promotion attempts require a stronger device
  profile or documented lower-memory matched profile before rerun.
- The original local budget deferral is documented in
  `docs/evidence/benchmarking/2026-04-24_large_cb_svdpp_deeper_run_deferral.md`.
- The large-dataset `cb_svdpp` matched-campaign contract remains the governing
  document for any remaining promotion work:
  `docs/evidence/benchmarking/2026-04-30_large_cb_svdpp_matched_campaign_contract.md`.
- No final claim may use dirty, cancelled, exploratory, or selection-only runs
  as benchmark anchors.
- The G6 `ml100k cb_svdpp` validation grid remains selection evidence only;
  the separate clean outer benchmark run may support only its own
  `benchmark_random_v1` readout.
- The `ml100k cb_asvdpp` hot-path decision may support only prioritization of
  remediation work. It is not speed, scalability, production-readiness,
  quality, or large-dataset evidence.
- The `cb_asvdpp` remediation contract authorizes one exact work-buffer
  remediation attempt only after a fresh clean pre-change baseline exists. It
  is not remediation evidence and unlocks no speed claim.
- The `cb_asvdpp` pre-change baseline is comparison evidence only. It is not
  post-change remediation evidence and unlocks no speed claim.

## Post-Release Work Queue

1. If dependencies, data evidence, benchmarks, or claim boundaries change after
   the release marker `submission-2026-05-02-r10`, rerun the affected gates and
   move the release marker.
2. Keep the documented G6 outer readout separate from older
   `paper_faithful_ml100k_v1` anchor comparisons unless the split-family
   difference is explicit.
3. Implement the `cb_asvdpp` exact work-buffer remediation only under the G8
   contract, then compare the post-change run against the G9 pre-change
   baseline.

## Current Non-Claims

- The repo is not an unconstrained or exact paper-faithful reproduction.
- `ml10m` has a clean matched `biased_mf` versus `cb_svdpp` comparison for the
  documented profiles only; it is not a general CB superiority claim.
- `ml20m` has a clean `biased_mf` baseline anchor, but its local `cb_svdpp`
  matched-campaign attempt is negative resource evidence, not final
  model-comparison evidence.
- `ml1m cb_asvdpp` is not a benchmark anchor.
- G6 `ml100k cb_svdpp` is validation-only selection evidence, not a final
  benchmark anchor.
- No final `ml20m` model-comparison claim is allowed.
- No `faster`, `scalable`, or `ready` claim is allowed beyond the specific
  evidence-backed benchmark context where it is explicitly documented.
