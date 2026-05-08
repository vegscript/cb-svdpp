# Cache-aware Tuning MVP Contract Hardening V1 Evidence

## Branch

`cache-aware-tuning-mvp-hardening-v1`

## Goal

Harden two dry-run planner contracts before any execution layer is introduced:

- Candidate summary paths must reference the actual output directory.
- Manual CB alpha endpoint validation must follow alpha target-path mappings.

This step does not add a tuning engine, candidate execution, runner
integration, optimizer dependency, pruning, KMeans search, or model/kernel
changes.

## Inputs Reviewed

- `src/recsys_lab/tuning/writers.py`
- `src/recsys_lab/tuning/schemas.py`
- `tests/unit/test_plan_tuning_study_script.py`
- `tests/unit/test_tuning_schemas.py`
- `docs/evidence/performance/cache_aware_tuning_mvp_v1.md`

## Contract Hardening V1

Candidate summary path consistency:

- `candidate_summary.csv` now receives the actual study output directory.
- `candidate_config_path` points to
  `<output_dir>/<study_id>/candidates/<candidate_id>/candidate_config.yaml`.
- `candidate_manifest_path` points to
  `<output_dir>/<study_id>/candidates/<candidate_id>/candidate_manifest.json`.
- The CSV helper no longer hard-codes `artifacts/tuning/<study_id>/...`.

Manual alpha endpoint validation:

- Productive CB manual candidates now validate alpha endpoint policy through:
  - dimension key `alpha`
  - `target_path: alpha`
  - `target_path: clustering.alpha`
  - target-path leaf `alpha`
- `cb_svdpp` and `cb_asvdpp` reject manual candidates with `alpha <= 0` or
  `alpha >= 1`.
- Non-CB models are not subject to the CB endpoint alpha policy.

## Tests

Focused tests added:

- `test_candidate_summary_paths_respect_custom_output_dir`
- `test_candidate_summary_paths_match_written_files`
- `test_cb_manual_candidate_rejects_alpha_zero_via_target_path_alpha`
- `test_cb_manual_candidate_rejects_alpha_one_via_target_path_clustering_alpha`
- `test_cb_manual_candidate_accepts_alpha_between_zero_and_one_via_target_path`
- `test_cb_manual_candidate_alpha_policy_applies_to_cb_asvdpp`
- `test_non_cb_manual_candidate_not_subject_to_cb_alpha_policy`

Focused checks run during implementation:

- `ruff check src/recsys_lab/tuning tests/unit/test_tuning_schemas.py tests/unit/test_plan_tuning_study_script.py` passed.
- `pytest tests/unit/test_tuning_schemas.py tests/unit/test_plan_tuning_study_script.py` passed: 26 passed.

Final hardening gates:

- `ruff check .` passed.
- `pytest tests/unit/test_tuning_schemas.py` passed: 11 passed.
- `pytest tests/unit/test_tuning_candidates.py` passed: 18 passed.
- `pytest tests/unit/test_tuning_planner.py` passed: 3 passed.
- `pytest tests/unit/test_plan_tuning_study_script.py` passed: 15 passed.
- `pytest tests/unit/test_hotpath_coldpath_boundaries.py` passed: 13 passed.
- `pytest tests/unit` passed: 276 passed.
- `pytest tests/integration/test_unified_pipeline_smoke_all_models.py` passed:
  1 passed.
- `pytest` passed: 351 passed, 2 skipped.
- `python scripts/plan_tuning_study.py --help` passed in the Python 3.12 test
  environment.
- `python scripts/plan_tuning_study.py --search-space tests/fixtures/tuning/cb_svdpp_tuning_mvp.yaml --output-dir artifacts/tuning --study-id dry_run_cb_svdpp_tuning_mvp_v1 --overwrite` passed in the Python 3.12 test environment and wrote 6 candidate directories, `study_manifest.json`, `candidate_summary.csv`, and `artifact_reuse_summary.csv`.
- The requested claim-check command was run. It returned existing
  claim-boundary and claim-lock references, not a new Step 19b-hardening
  performance or quality claim.

## Claim Boundary

No claim is made that tuning is faster.

No claim is made that model quality is better.

This evidence only states that dry-run planner path contracts and manual
candidate alpha validation contracts were hardened.

## Recommended Next Step

`19c. Cache-aware Tuning Execution Smoke V1`

That step should remain a small execution wiring proof and should not introduce
large studies, pruning, optimizer backends, or performance/quality claims.
