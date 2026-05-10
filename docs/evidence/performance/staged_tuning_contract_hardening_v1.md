# Staged Tuning Contract Hardening V1

## Branch

`staged-tuning-contract-hardening-v1`

## Problem Fixed

Step 20a introduced staged planning and fake-result promotion, but the promotion
contract needed to be stricter before real ML1M candidate execution:

- Promotion must use an explicit source stage and target stage.
- Promotion count must come from `source_stage.promote_top_k`, not from
  `target_stage.max_candidates`.
- Promotion inputs must belong to the expected study and source stage when those
  fields are present.
- Stage-specific reuse keys must use only the currently planned stage overrides.

## Contract Changes

- Promotion now validates source and target stage names against the schedule.
- Promotion rejects same-stage and backward-stage movement.
- Promotion rejects target stages whose `max_candidates` is below the source
  stage `promote_top_k`.
- Promotion rejects result inputs with test metrics, wrong stage, wrong study,
  missing required fields, missing candidate configs, or insufficient succeeded
  candidates.
- Candidate manifests record `stage_name`.
- Study manifests record the schedule and current stage.
- Promoted candidate configs record source candidate, source stage, target stage,
  promotion rank, and applied stage overrides.
- Stage-aware cluster reuse keys include current outer stage overrides and ignore
  inner stage overrides and overrides from other stages.

## Tests/Gates

Focused Phase 6 tests:

```bash
ruff check tests/unit/test_tuning_staged_planner.py tests/unit/test_plan_sota_tuning_study_script.py tests/unit/test_tuning_search_roles.py tests/unit/test_tuning_schemas.py
pytest tests/unit/test_tuning_staged_planner.py tests/unit/test_plan_sota_tuning_study_script.py tests/unit/test_tuning_search_roles.py tests/unit/test_tuning_schemas.py
```

Result: `104 passed`.

## Dry-run Smoke

Stage 1 planning:

```bash
py -3.12 scripts/plan_sota_tuning_study.py --search-space configs/experiments/tuning/templates/ml1m_cb_svdpp_sota_tuning_v1.yaml --output-dir artifacts/tuning --study-id dry_run_ml1m_cb_svdpp_sota_tuning_v1_hardened --stage stage1_low_fidelity --overwrite
```

Observed:

- Stage-1 candidate configs: 48
- Stage-1 cluster reuse groups: 1
- No candidate execution was started.

Fake promotion:

```bash
py -3.12 scripts/plan_sota_tuning_study.py --search-space configs/experiments/tuning/templates/ml1m_cb_svdpp_sota_tuning_v1.yaml --output-dir artifacts/tuning --study-id dry_run_ml1m_cb_svdpp_sota_tuning_v1_hardened --promote-from-results artifacts/tuning/dry_run_ml1m_cb_svdpp_sota_tuning_v1_hardened/reports/fake_stage1_results.csv --from-stage stage1_low_fidelity --to-stage stage2_mid_fidelity --overwrite
```

Observed:

- Promoted Stage-2 configs: 12
- Promotion plan `from_stage`: `stage1_low_fidelity`
- Promotion plan `to_stage`: `stage2_mid_fidelity`
- Promotion plan stage override: `training.epochs = 10`

## Limitations

- No ML1M candidate execution was run in this hardening step.
- Fake results were used only to validate promotion planning and materialization.
- This step does not add pruning, Pareto selection, parallel execution, or a new
  runner.

## Decision

`STAGED_TUNING_CONTRACT_HARDENED`
