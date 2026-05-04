# Unified Experiment Framework

## Architecture

Productive experiment execution uses one lifecycle:

```text
CLI / benchmark / tuning
  -> run_unified_experiment
  -> MODEL_REGISTRY + ModelAdapter
  -> ModelRequirements
  -> framework-built FitArtifacts
  -> model-specific fit / predict logic
  -> metrics, diagnostics, manifest, profiling, caches
```

`run_unified_experiment` owns split resolution, cache policy, artifact
construction, metrics writing, profiling, manifests, and runtime metadata. Model
adapters declare what each model needs via `ModelRequirements`; the framework
then builds or loads those artifacts and passes them as `FitArtifacts`.

The supported productive models are:

- `biased_mf`
- `svdpp`
- `asymmetric_svd`
- `asvdpp`
- `cb_svdpp`
- `cb_asvdpp`

`cb_svdpp` and `cb_asvdpp` use the same CB infrastructure. Their expected
requirements are:

- `cb_svdpp`: `user_history_index`, `cluster_artifacts`,
  `user_cluster_history_index`
- `cb_asvdpp`: `user_history_index`, `explicit_feedback_index`,
  `cluster_artifacts`, `user_cluster_history_index`

The models may differ in formulas, parameter blocks, update kernels,
requirements, and config schemas. They must not differ in experiment pipeline,
split logic, artifact generation, cache policy, manifest writing, or tuning
execution path.

## Config Contract

Productive YAML model configs must be validated through the Pydantic
`ModelProfile` schemas and built through the matching `ModelAdapter`.

The productive path is:

```text
YAML payload -> validate_model_config_payload -> ModelAdapter.build_model_config
```

Unknown fields, missing required fields, wrong model names, wrong model
families, and misspelled fields such as `lambda_yc` must fail validation.
Internal model dataclass defaults are allowed for focused model tests, but they
are not a productive config source. Productive CLI, benchmark, tuning, and
legacy-wrapper paths must not build model dataclasses directly from YAML dicts
or use `.get(..., default)` / `.setdefault(...)` fallbacks for model
parameters.

## CB Semantics

CB semantics are separated into configuration, diagnostics, and claim
governance:

- `alpha = 0`: the cluster channel is disabled; the run is a CB-disabled
  ablation.
- `alpha > 0`: the cluster channel is mathematically enabled in the
  configuration.
- `cb_claim_eligible`: remains `false` unless separate diagnostics and ablation
  evidence justify a CB quality claim.

`alpha > 0` is therefore not evidence that the cluster channel contributes
materially and not evidence for a scientific CB claim.

CB runs write `cb_semantics` and `cb_diagnostics` to `metrics.json`.
Diagnostics include cluster-artifact presence, cluster counts and sizes,
individual/cluster factor norm summaries, missing expected artifacts or model
fields, contribution diagnostics, and `diagnostic_claim_ready=false`.

## Legacy Commands

Legacy model-specific commands are kept only for backward compatibility. They
must remain thin wrappers and delegate to `run_unified_experiment`.

Prefer the unified command form for new work:

```bash
recsys-lab train --model cb_svdpp ...
recsys-lab train --model cb_asvdpp ...
```

Do not add split, cache, fit, predict, metrics, manifest, profiling, or config
merge logic to legacy wrappers.
