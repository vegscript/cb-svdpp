# SOTA Tuning Engine V1 Evidence

## Branch

`sota-tuning-engine-v1`

## Implemented Engine Capabilities

- Extended tuning schema supports `grid`, `manual`, `random`, and `latin_hypercube` generators.
- Search spaces can define staged fidelity schedules with per-stage candidate budgets and overrides.
- Dry-run planning can materialize one fidelity stage into study manifests, candidate manifests, candidate configs, candidate summary, and reuse summary.
- Promotion planning can read prior stage CSV/JSON results, rank succeeded candidates, and materialize promoted configs with next-stage overrides.
- No candidate execution engine, parallel runner, pruning engine, or optimizer backend was added.

## Samplers

- `random`: deterministic with local seeded RNG.
- `latin_hypercube`: deterministic with local seeded RNG, numeric strata, seeded per-dimension permutations, and deterministic categorical assignment.
- Supported dimension forms: categorical, int uniform, float uniform, float loguniform.
- Productive CB alpha bounds remain strictly inside `0 < alpha < 1`.

## Stage / Promotion Contract

- Stage planning uses the selected stage budget and writes candidate configs with stage overrides.
- Promotion input requires:
  - `candidate_id`
  - `execution_status`
  - `validation_rmse`
  - `validation_mae`
  - `fit_model_seconds`
  - `candidate_config_path`
- Promotion ranks only `succeeded` candidates by validation RMSE, then validation MAE, then fit time.
- `test_rmse` and `test_mae` are rejected in promotion inputs.
- Promoted configs apply the target stage overrides instead of copying source configs unchanged.

## Inner / Outer Search Separation

- Inner target params do not invalidate cluster reuse:
  - `clustering.alpha`
  - `training.learning_rate`
  - target lambdas
  - `training.epochs`
- Outer cluster / induction params invalidate cluster reuse:
  - `clustering.n_user_clusters`
  - `clustering.n_item_clusters`
  - `clustering.algorithm`
  - `clustering.kmeans_n_init`
  - `clustering.induction.*`
- Stage overrides follow the same role contract.

## Dry-run Artifacts

Command:

```bash
python scripts/plan_sota_tuning_study.py --search-space configs/experiments/tuning/templates/ml1m_cb_svdpp_sota_tuning_v1.yaml --output-dir artifacts/tuning --study-id dry_run_ml1m_cb_svdpp_sota_tuning_v1 --stage stage1_low_fidelity --overwrite
```

Observed dry-run output:

- 48 Stage-1 candidate configs materialized.
- Study manifest written.
- Candidate summary written.
- Artifact reuse summary written.
- One cluster reuse group planned.
- No ML1M execution was started.

Promotion smoke:

- 48 fake Stage-1 result rows were generated locally under `artifacts/tuning/...`.
- Promotion from `stage1_low_fidelity` to `stage2_mid_fidelity` materialized 12 promoted configs.
- `promotion_plan.json` was written.
- Generated artifacts remain unversioned under `artifacts/`.

## Tests / Gates

- `ruff check scripts/plan_sota_tuning_study.py tests/unit/test_plan_sota_tuning_study_script.py src/recsys_lab/tuning/staged_planner.py`
- `pytest tests/unit/test_plan_sota_tuning_study_script.py tests/unit/test_tuning_staged_planner.py`
- `ruff check tests/unit/test_tuning_sampling.py tests/unit/test_tuning_search_roles.py tests/unit/test_tuning_staged_planner.py tests/unit/test_plan_sota_tuning_study_script.py`
- `pytest tests/unit/test_tuning_sampling.py tests/unit/test_tuning_search_roles.py tests/unit/test_tuning_staged_planner.py tests/unit/test_plan_sota_tuning_study_script.py`
- Script help smoke passed with Python 3.12.

## Limitations

- No full tuning study was executed.
- No ML1M candidate execution was started in this step.
- No pruning, Pareto evaluation, optimizer backend, or parallel scheduling was implemented.
- On the local workstation, the bare `python` command points to an environment without project dependencies; script smokes used the Python environment that runs the test suite.

## Decision

`SOTA_TUNING_ENGINE_PRIMITIVES_READY`

## Next Step

20b should connect these primitives to a small controlled execution loop or harden promotion/resume behavior before any larger ML1M tuning campaign.
