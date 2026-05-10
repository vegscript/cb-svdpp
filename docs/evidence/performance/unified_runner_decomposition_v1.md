# Unified Runner Decomposition V1

## Branch

`unified-runner-decomposition-v1`

## Goal

Behavior-preserving internal decomposition of
`src/recsys_lab/experiments/unified_runner.py`. The public
`run_unified_experiment(...)` entry point, return payload, run artifact layout,
metrics, model behavior, kernels, config defaults, and performance stage names
must remain compatible.

## Phase 1 Inventory And Cut Plan

Current monolith:

- `run_unified_experiment(...)` owns the full orchestration path from path/config
  resolution through artifact writing and failure handling.
- Private helpers below the entry point already separate some behavior, but they
  still live in the same file.

Identified blocks:

| Block | Current location | Cut target |
| --- | --- | --- |
| Experiment/config/runtime resolution | `run_unified_experiment(...)` initial stages: `resolve_experiment_config`, `resolve_dataset_manifest`, `resolve_model_profile`, `validate_model_config`, `resolve_split_cache_policy` | `experiments/unified/config_resolution.py` and `context.py` |
| Dataset manifest + ratings loading | manifest resolution stage and later `load_ratings_data`; `_ratings_stage_metadata` | `experiments/unified/data_resolution.py` |
| Split/cache resolution | `_run_context_slug`, `_build_split`, split cache policy/use, `load_or_build_split_cache` | `experiments/unified/data_resolution.py` |
| Model profile validation / model config / induction config | `validate_model_config_payload`, adapter resolution, `build_model_config`, adapter `build_induction_config`, `_fit_stage_model_config_metadata`, `_cb_semantics_for_profile`, `_clustering_config` | `experiments/unified/config_resolution.py` |
| Fit artifact resolution | `_resolve_fit_artifacts`, `_cache_metadata_payload`, cluster/user-history/explicit-feedback cache handling | `experiments/unified/artifact_resolution.py` |
| Model instantiate / fit / predict | `initialize_model`, `fit_model`, kernel profile build/write, `predict_train`, `predict_validation`, `predict_test` | `experiments/unified/model_execution.py` |
| Metrics + diagnostics | `_prefixed_rating_metrics`, `_build_rating_metrics_payload`, system metrics assembly, `_build_cb_diagnostics` and related CB diagnostic helpers | `experiments/unified/evaluation.py` |
| Output writing / manifests / logs / performance profile | config snapshot, stdout log, run manifest writes, `_write_performance_profile`, `_build_performance_profile`, summaries, `_build_caches_payload`, `_build_model_payload`, `_default_command_string` | `experiments/unified/outputs.py` |
| Error handling | `try/except` block inside `run_unified_experiment(...)` writes failed manifest, log, performance profile, then re-raises | keep orchestrated from `unified_runner.py`, with output helpers delegated to `outputs.py` |

Refactor stance:

- Extract helpers first; keep stage names and stage boundaries stable.
- Keep `unified_runner.py` as the public import surface and thin orchestrator.
- Do not move model hotpath code or import tuning code into models.
- Add focused tests only around contracts that can regress during extraction,
  then rely on the all-model unified pipeline smoke for end-to-end compatibility.

## Claim Boundary

This is a decomposition plan and later behavior-preserving refactor only. No
performance, quality, or scalability claim is made.

## Phase 2 Context Types

Added `src/recsys_lab/experiments/unified/context.py` with small dataclasses for
values that are currently threaded through `run_unified_experiment(...)`:

- `UnifiedRunInputs`
- `ResolvedExperimentConfig`
- `RunPaths`
- `ResolvedModelProfile`
- `ResolvedSplitBundle`
- `FitArtifactResolution`
- `ModelExecutionResult`
- `EvaluationResult`

These are data containers only. No business logic, model logic, runner
semantics, metrics, or output contracts were changed.

## Phase 3 Config Resolution

Added `src/recsys_lab/experiments/unified/config_resolution.py` and routed
`run_unified_experiment(...)` through `resolve_unified_experiment_config(...)`.

Extracted behavior:

- repo-root and input path resolution
- runtime and device config loading
- runtime threading config resolution
- processed dataset manifest loading
- model config loading and profile validation
- split cache policy resolution
- cache option validation
- run context slug calculation
- device profile name resolution
- default command string construction
- CB semantics resolution

Compatibility checks:

```bash
ruff check src/recsys_lab/experiments/unified_runner.py src/recsys_lab/experiments/unified/config_resolution.py src/recsys_lab/experiments/unified/context.py
pytest tests/integration/test_unified_pipeline_smoke_all_models.py
```

Both checks passed. Public entry point and output contracts remain unchanged.

## Phase 4 Data And Split Resolution

Added `src/recsys_lab/experiments/unified/data_resolution.py` and routed the
data/split section of `run_unified_experiment(...)` through
`resolve_unified_data_split(...)`.

Extracted behavior:

- ratings data loading from the processed manifest
- split cache load/build orchestration
- train, validation, and test split extraction
- split-stage cache metadata updates
- ratings stage metadata helper
- `benchmark_random_v1`, `paper_faithful_ml100k_v1`, and
  `paper_faithful_ml100k_inner_v1` split dispatch

Compatibility checks:

```bash
ruff check src/recsys_lab/experiments/unified_runner.py src/recsys_lab/experiments/unified/data_resolution.py src/recsys_lab/experiments/unified/context.py
pytest tests/integration/test_unified_pipeline_smoke_all_models.py
```

Both checks passed. Split behavior and public runner behavior remain unchanged.
