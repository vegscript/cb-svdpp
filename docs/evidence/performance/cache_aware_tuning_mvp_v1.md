# Cache-aware Tuning MVP V1 Evidence

## Branch

`cache-aware-tuning-mvp-v1`

## Goal

Implement a minimal dry-run-first tuning planner that materializes deterministic
candidates from a search-space contract, computes cache-aware artifact reuse
groups, and writes study, candidate, and report artifacts.

This step does not implement a tuning engine, does not run a tuning study, does
not add an optimizer dependency, does not add pruning, does not search KMeans
or induction configuration, and does not change model, kernel, runner, or
dataset semantics.

## Inputs Reviewed

- `docs/architecture/sota_tuning_framework_architecture_v1.md`
- `docs/evidence/performance/sota_tuning_framework_architecture_v1.md`
- `docs/evidence/performance/cluster_artifact_kmeans_audit_v1.md`
- `docs/evidence/performance/kmeans_candidate_strategy_v1.md`
- `docs/architecture/adr_cb_endpoint_alpha_policy_v1.md`
- `src/recsys_lab/tuning/schemas.py`
- `tests/unit/test_tuning_schemas.py`
- existing tuning entry points under `src/recsys_lab/experiments/ml100k_inner_tuning.py`
- existing config loader and model config validation behavior

## Implemented Scope

New dry-run tuning package modules:

- `src/recsys_lab/tuning/candidates.py`
- `src/recsys_lab/tuning/planner.py`
- `src/recsys_lab/tuning/manifests.py`
- `src/recsys_lab/tuning/writers.py`

New planner script:

- `scripts/plan_tuning_study.py`

New test fixture:

- `tests/fixtures/tuning/cb_svdpp_tuning_mvp.yaml`

New or expanded tests:

- `tests/unit/test_tuning_candidates.py`
- `tests/unit/test_plan_tuning_study_script.py`
- existing `tests/unit/test_tuning_schemas.py` retained

The implementation is dry-run only. It writes candidate configs and manifests
but does not execute `run_unified_experiment(...)`.

## Study Planner Contract

The planner accepts a `SearchSpaceSpec` and produces a `StudyPlan` with:

- deterministic `study_id`
- candidate list
- artifact reuse groups
- objective status set to `planned`
- execution status represented as not executed in candidate manifests

The planner does not read datasets, build caches, run models, or inspect
runtime cache status. Artifact grouping is contract-level planning metadata.

## Candidate Generation Behavior

Supported generators:

- `grid`
- `manual`

Grid behavior:

- deterministic
- respects `budget.max_candidates`
- uses dimension-name sorting when `deterministic_order=true`
- accepts categorical dimensions with `values`
- accepts numeric dimensions only when explicit `values` are provided
- rejects continuous numeric sampling without explicit `values`

Continuous linspace/logspace sampling remains deferred. This avoids hidden
sampling policy in the first MVP.

Candidate IDs use:

```text
cand_0000_<short_hash>
```

The hash is based on study name, base model config, and parameter values.

Each candidate includes:

- `candidate_id`
- `study_id`
- zero-based `index`
- `parameter_values`
- `base_model_config`
- materialized config payload
- objective status `planned`

## Artifact Reuse Grouping Behavior

For CB candidates, Cluster-Artifact reuse groups are computed at contract
level.

Candidates share a Cluster-Artifact reuse group when only these fields change:

- `alpha`
- `learning_rate`
- `lambda_p`
- `lambda_q`
- `lambda_y`
- `lambda_pC`
- `lambda_qC`
- `lambda_yC`
- `epochs`

Candidates receive different groups when reuse-relevant fields change, for
example:

- `n_user_clusters`
- `n_item_clusters`
- induction config
- `kmeans_n_init`
- clustering algorithm
- dataset
- split
- train fingerprint, when available

Reuse group IDs use:

```text
cluster_rg_<short_hash>
```

The hash excludes declared `reuse_across` coordinates such as alpha, lambdas,
learning rate, and epochs.

## Written Artifact Layout

Dry-run output layout:

```text
artifacts/tuning/<study_id>/
  search_space.yaml
  study_manifest.json
  candidates/
    <candidate_id>/
      candidate_config.yaml
      candidate_manifest.json
  reports/
    candidate_summary.csv
    artifact_reuse_summary.csv
    artifact_reuse_summary.json
```

`study_manifest.json` includes:

- `study_id`
- `study_name`
- `search_space_version`
- `dataset`
- `split_family`
- `model`
- `seed`
- `base_model_config`
- `budget`
- `generator`
- `objective`
- `candidate_count`
- `artifact_reuse_contract`
- `created_at_utc`
- `schema_version`
- `claim_boundary`

`candidate_manifest.json` includes:

- `candidate_id`
- `candidate_index`
- `study_id`
- `parameter_values`
- `base_model_config`
- `candidate_config_path`
- `artifact_reuse_group_ids`
- `objective_status`
- `execution_status`
- `claim_boundary`

`candidate_summary.csv` includes:

- `candidate_id`
- `candidate_index`
- `study_id`
- `model`
- `dataset`
- `split_family`
- `alpha`
- `learning_rate`
- `latent_dim`
- `epochs`
- `cluster_reuse_group_id`
- `candidate_config_path`
- `candidate_manifest_path`
- `status`

Missing values are left empty rather than inferred.

## Dry-run Output Example

Example command:

```powershell
python scripts/plan_tuning_study.py --search-space tests/fixtures/tuning/cb_svdpp_tuning_mvp.yaml --output-dir artifacts/tuning --study-id ml100k_cb_svdpp_tuning_mvp_v1
```

The fixture produces:

- 6 planned candidates
- 1 Cluster-Artifact reuse group
- full study manifest
- candidate manifests
- candidate configs
- candidate summary CSV
- artifact reuse summary CSV/JSON

The command is a dry-run planner example only. It does not run training and does
not produce model quality or runtime evidence.

## Tests/Gates

Focused checks run during implementation:

- `ruff check src/recsys_lab/tuning tests/unit/test_tuning_candidates.py tests/unit/test_tuning_schemas.py` passed.
- `pytest tests/unit/test_tuning_candidates.py tests/unit/test_tuning_schemas.py` passed: 18 passed.
- `ruff check scripts/plan_tuning_study.py tests/unit/test_plan_tuning_study_script.py src/recsys_lab/tuning tests/unit/test_tuning_candidates.py tests/unit/test_tuning_schemas.py` passed.
- `pytest tests/unit/test_plan_tuning_study_script.py tests/unit/test_tuning_candidates.py tests/unit/test_tuning_schemas.py` passed: 37 passed.
- CLI smoke with Python 3.12 and `--example-synthetic` passed and wrote study manifest, candidate summary CSV, and artifact reuse summary CSV to a temporary directory.

Covered behavior:

- deterministic grid generation
- `max_candidates` enforcement
- stable candidate IDs
- manual candidate order
- endpoint alpha policy via existing schema tests
- continuous numeric grid sampling rejected for MVP
- materialized candidate config generation
- strict unknown override rejection
- study manifest fields
- candidate manifest fields
- candidate summary CSV fields
- artifact reuse summary CSV fields
- refusal to overwrite existing study output without `--overwrite`
- fixture-based dry-run script output

Final Step 19b gates:

- `ruff check .` passed.
- `pytest tests/unit/test_tuning_schemas.py` passed: 6 passed.
- `pytest tests/unit/test_tuning_candidates.py` passed: 18 passed.
- `pytest tests/unit/test_tuning_planner.py` passed: 3 passed.
- `pytest tests/unit/test_plan_tuning_study_script.py` passed: 13 passed.
- `pytest tests/unit/test_hotpath_coldpath_boundaries.py` passed: 13 passed.
- `pytest tests/unit` passed: 269 passed.
- `pytest tests/integration/test_unified_pipeline_smoke_all_models.py` passed:
  1 passed.
- `pytest` passed: 344 passed, 2 skipped.
- `python scripts/plan_tuning_study.py --help` passed in the Python 3.12 test
  environment.
- `python scripts/plan_tuning_study.py --search-space tests/fixtures/tuning/cb_svdpp_tuning_mvp.yaml --output-dir artifacts/tuning --study-id dry_run_cb_svdpp_tuning_mvp_v1 --overwrite` passed in the Python 3.12 test environment and wrote 6 candidate directories, `study_manifest.json`, `candidate_summary.csv`, and `artifact_reuse_summary.csv`.
- The requested claim-check command was run. It returned existing
  claim-boundary and claim-lock references, not a new Step 19b performance or
  quality claim.

## Limitations

- No candidate execution.
- No integration with `run_unified_experiment(...)`.
- No study resume/retry behavior.
- No pruning.
- No Pareto frontier computation.
- No optimizer backend.
- No KMeans or induction search.
- No continuous linspace/logspace sampling.
- No runtime cache status is observed because this is dry-run planning.
- Candidate config materialization is validated through strict merge behavior;
  full runner execution is deferred.

## Claim Boundary

No claim is made that tuning is faster.

No claim is made that any model quality is better.

This evidence only states:

- the planner produces deterministic candidates
- the planner writes study and candidate manifests
- the planner writes candidate configs and reports
- Cluster-Artifact reuse groups are planned for CB candidates
- dry-run artifacts are reproducible under the declared search-space contract

## Recommended Next Step

Recommended next step:

`19c. Cache-aware Tuning Execution Smoke V1`

That step should run at most one tiny candidate or one explicitly bounded
ML100K candidate to prove manifest-to-run wiring. It should remain separate
from the dry-run planner and should not introduce pruning, optimizer backends,
large studies, or performance/quality claims.
