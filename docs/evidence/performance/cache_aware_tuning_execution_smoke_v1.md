# Cache-aware Tuning Execution Smoke V1

## Branch

`cache-aware-tuning-execution-smoke-v1`

## Goal

Step 19c adds a narrow execution-wiring smoke for the cache-aware tuning
framework. The goal is to prove that one planned candidate can be handed to the
existing unified experiment runner and that tuning study artifacts can be
updated after the run.

This is not a tuning study.

## Inputs Reviewed

- `docs/architecture/sota_tuning_framework_architecture_v1.md`
- `docs/evidence/performance/cache_aware_tuning_mvp_v1.md`
- `docs/evidence/performance/cache_aware_tuning_mvp_hardening_v1.md`
- `src/recsys_lab/tuning/schemas.py`
- `src/recsys_lab/tuning/candidates.py`
- `src/recsys_lab/tuning/planner.py`
- `src/recsys_lab/tuning/manifests.py`
- `src/recsys_lab/tuning/writers.py`
- `src/recsys_lab/experiments/unified_runner.py`
- `tests/integration/test_unified_pipeline_smoke_all_models.py`

## Implemented Scope

- Added a small execution adapter in `src/recsys_lab/tuning/execution.py`.
- Added a single-candidate smoke script in
  `scripts/run_tuning_candidate_smoke.py`.
- Extended candidate manifests with additive execution result fields.
- Extended `candidate_summary.csv` with execution and selected metric/runtime
  columns.
- Added `execution_summary.csv`.
- Added unit tests for adapter/result contracts and script guardrails.
- Added a tiny integration smoke that executes exactly one `cb_svdpp`
  candidate on the existing synthetic pipeline-smoke dataset.

No execution engine, optimizer, pruning, Pareto engine, runner rewrite, kernel
change, or model change was added.

## Execution Adapter Contract

The adapter loads a `candidate_manifest.json`, resolves its
`candidate_config.yaml`, and calls the existing `run_unified_experiment(...)`
function. The adapter does not implement model fitting, metrics calculation,
split handling, cache handling, or artifact lifecycle logic.

The result contract is `CandidateExecutionResult`:

- `candidate_id`
- `study_id`
- `execution_status`
- `run_id`
- `run_dir`
- `metrics_path`
- `performance_profile_path`
- `kernel_profile_path`
- `run_manifest_path`
- `started_at_utc`
- `finished_at_utc`
- `error_message`

Supported statuses for this smoke are `succeeded`, `failed`, `skipped`, and
`not_executed`.

## Script Contract

`scripts/run_tuning_candidate_smoke.py` executes at most one candidate:

```bash
python scripts/run_tuning_candidate_smoke.py \
  --study-dir artifacts/tuning/<study_id> \
  --candidate-id <candidate_id>
```

Rules:

- If `--candidate-id` is omitted and the study has exactly one candidate, that
  candidate is selected.
- If the study has multiple candidates and `--candidate-id` is omitted, the
  script fails with a controlled error.
- There is no bulk execution option.
- There is no parallel execution option.
- There is no study loop.
- Real execution requires explicit runner inputs such as `--processed-manifest`,
  runtime config, device config, split parameters, and seed parameters.

## Study Artifact Updates

After execution, the following artifacts can be updated:

- `candidates/<candidate_id>/candidate_manifest.json`
- `reports/candidate_summary.csv`
- `reports/execution_summary.csv`

The candidate manifest stores the execution status and run artifact paths.

`candidate_summary.csv` now includes:

- `execution_status`
- `run_id`
- `validation_rmse`
- `validation_mae`
- `fit_model_seconds`
- `cluster_cache_status`
- `cluster_total_seconds`

Missing values are left blank rather than inferred.

`execution_summary.csv` includes:

- `study_id`
- `candidate_id`
- `execution_status`
- `run_id`
- `run_dir`
- `metrics_path`
- `performance_profile_path`
- `kernel_profile_path`
- `run_manifest_path`
- `error_message`

## Smoke Executed

Tiny integration smoke:

```bash
pytest tests/integration/test_tuning_execution_smoke.py
```

Result:

- `1 passed`

The smoke uses the existing synthetic all-model pipeline fixture:

- model: `cb_svdpp`
- candidates: exactly `1`
- epochs: `1`
- latent_dim: `2`
- dtype: `float32`
- cluster counts: small synthetic fixture values
- test evaluation: disabled for the smoke

The smoke verifies:

- exactly one candidate is present
- `candidate_manifest.json` is updated to `execution_status == "succeeded"`
- `run_manifest_path` exists
- `metrics_path` exists
- `candidate_summary.csv` is updated
- `execution_summary.csv` exists

Optional local ML100K script smoke:

```bash
python scripts/plan_tuning_study.py \
  --search-space tests/fixtures/tuning/cb_svdpp_tuning_mvp.yaml \
  --output-dir artifacts/tuning \
  --study-id local_ml100k_execution_smoke_v1 \
  --overwrite
```

Result:

- planned study: `local_ml100k_execution_smoke_v1`
- candidate_count: `6`
- artifact_reuse_group_count: `1`

Then a dry-run candidate selection was executed:

```bash
python scripts/run_tuning_candidate_smoke.py \
  --study-dir artifacts/tuning/local_ml100k_execution_smoke_v1 \
  --candidate-id cand_0000_493b27f01eea \
  --dry-run
```

Result:

- `execution_status == "not_executed"`

A real ML100K execution was not forced because the fixture references
`configs/models/cb_svdpp.yaml`, whose default profile is not a deliberately
tiny local smoke profile. The generated local planning artifacts were removed
after the check because they are local smoke outputs.

## Tests/Gates

Focused checks run during implementation:

```bash
ruff check scripts/run_tuning_candidate_smoke.py tests/unit/test_run_tuning_candidate_smoke_script.py src/recsys_lab/tuning
pytest tests/unit/test_run_tuning_candidate_smoke_script.py tests/unit/test_tuning_execution.py tests/unit/test_tuning_candidates.py
ruff check tests/unit/test_tuning_execution.py tests/unit/test_run_tuning_candidate_smoke_script.py tests/integration/test_tuning_execution_smoke.py
pytest tests/unit/test_tuning_execution.py tests/unit/test_run_tuning_candidate_smoke_script.py tests/integration/test_tuning_execution_smoke.py
python scripts/run_tuning_candidate_smoke.py --help
```

Observed results:

- focused ruff checks passed
- tuning execution unit tests passed
- single-candidate smoke script tests passed
- tiny integration execution smoke passed
- script help passed

Final Step 19c gates:

```bash
ruff check .
pytest tests/unit/test_tuning_schemas.py
pytest tests/unit/test_tuning_candidates.py
pytest tests/unit/test_tuning_planner.py
pytest tests/unit/test_plan_tuning_study_script.py
pytest tests/unit/test_tuning_execution.py
pytest tests/unit/test_run_tuning_candidate_smoke_script.py
pytest tests/unit/test_hotpath_coldpath_boundaries.py
pytest tests/unit
pytest tests/integration/test_unified_pipeline_smoke_all_models.py
pytest tests/integration/test_tuning_execution_smoke.py
pytest
python scripts/run_tuning_candidate_smoke.py --help
rg "guaranteed speedup|production-ready|SOTA speedup|broad performance claim" docs src tests
```

Observed final results:

- `ruff check .`: passed
- `pytest tests/unit/test_tuning_schemas.py`: 11 passed
- `pytest tests/unit/test_tuning_candidates.py`: 18 passed
- `pytest tests/unit/test_tuning_planner.py`: 3 passed
- `pytest tests/unit/test_plan_tuning_study_script.py`: 15 passed
- `pytest tests/unit/test_tuning_execution.py`: 7 passed
- `pytest tests/unit/test_run_tuning_candidate_smoke_script.py`: 5 passed
- `pytest tests/unit/test_hotpath_coldpath_boundaries.py`: 13 passed
- `pytest tests/unit`: 288 passed
- `pytest tests/integration/test_unified_pipeline_smoke_all_models.py`: 1 passed
- `pytest tests/integration/test_tuning_execution_smoke.py`: 1 passed
- `pytest`: 364 passed, 2 skipped
- `python scripts/run_tuning_candidate_smoke.py --help`: passed
- claim-check: only existing claim-boundary, claim-lock, or negative-claim
  references were returned

## Limitations

- Only one candidate is supported by the smoke script.
- There is no study execution engine.
- There is no resume, retry, scheduler, pruning, or Pareto reporting.
- ML100K real execution was not forced in this step.
- The smoke validates wiring and artifact updates only.
- It does not validate tuning effectiveness.

## Claim Boundary

No claim is made that tuning is faster.

No claim is made that model quality is better.

No claim is made that ML1M, ML20M, or larger datasets scale from this smoke.

This evidence only supports:

- one planned candidate can be executed in a controlled way
- candidate manifests can be updated after execution
- study reports can be updated after execution
- the executed validation was a single smoke run

## Recommended Next Step

19d. Tuning Study Execution MVP V1

Recommended scope:

- still small candidate counts
- explicit ML100K smoke profile if needed
- no optimizer dependency
- no pruning
- no performance or quality claims without study evidence
