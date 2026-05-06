# Performance Forensics V1

## Purpose

Performance Forensics V1 makes productive unified experiment runs measurable at
pipeline-stage level. The goal is to identify where wall-clock time and process
RSS memory are spent in `run_unified_experiment`.

This is measurement infrastructure only. It does not optimize model formulas,
training kernels, cache policy, hyperparameters, split logic, or evaluation
metrics. A profile is diagnostic evidence, not a performance claim by itself.

## Measured Stages

The unified runner records meaningful pipeline blocks instead of timing every
line. Not every stage appears for every model. Non-applicable stages are not
written as placeholders.

Common stages include:

- `resolve_experiment_config`
- `resolve_dataset_manifest`
- `resolve_model_profile`
- `validate_model_config`
- `resolve_split_cache`
- `load_train_ratings`
- `load_validation_ratings`
- `load_test_ratings`
- `resolve_model_requirements`
- `build_fit_artifacts`
- `initialize_model`
- `fit_model`
- `predict_train`
- `predict_validation`
- `predict_test`
- `build_rating_metrics`
- `write_metrics_json`
- `write_run_manifest`
- `write_config_snapshot`
- `write_performance_profile`

Model-dependent stages include:

- `build_training_indices`
- `build_user_history_index`
- `build_explicit_feedback_index`
- `build_cluster_artifacts`
- `build_user_cluster_history_index`
- `build_cb_diagnostics`

For example, `biased_mf` does not write user-history or cluster stages.
`svdpp` writes user-history stages. `cb_svdpp` writes cluster-artifact and
user-cluster-history stages. `cb_asvdpp` additionally writes explicit-feedback
stages.

## Profile Artifact

Each completed or failed unified run writes:

```text
artifacts/runs/<run_id>/performance_profile.json
```

The payload contains:

- `profile_version`: currently `performance_forensics_v1`
- run identity: `run_id`, `dataset`, `model`, `device_profile`,
  `split_family`, `split_seed`, `model_seed`
- aggregate timing: `total_profiled_wall_clock_seconds`, `stage_count`
- `stages`: per-stage records with status, wall-clock seconds, RSS start/end,
  RSS delta, and small metadata
- `hotspots`: stages sorted descending by wall-clock seconds with share of
  total profiled time

Stage metadata is intentionally small and serializable. It may include row
counts, user/item counts, cache status, artifact names, cluster counts, `alpha`,
epochs, latent dimension, or training backend. It must not include NumPy arrays,
full config payloads, model objects, ratings arrays, or unnecessary absolute
local paths.

`metrics.json` includes a compact `performance_profile` summary with the
profile path, version, stage count, total profiled time, and top hotspots.
`run_manifest.json` lists the profile under `artifacts.performance_profile`.
The older `profiling` payload remains present for backward compatibility.

## Collecting Profiles

To collect CSV tables from all run folders:

```powershell
.venv\Scripts\python.exe scripts\collect_performance_profiles.py
```

Default outputs:

```text
artifacts/reports/performance_stage_breakdown.csv
artifacts/reports/performance_hotspots.csv
```

Custom locations can be passed explicitly:

```powershell
.venv\Scripts\python.exe scripts\collect_performance_profiles.py --runs-dir artifacts\runs --output-dir artifacts\reports
```

`performance_stage_breakdown.csv` contains one row per stage:

- dataset
- model
- run_id
- stage_name
- wall_clock_seconds
- share_of_profiled_time
- rss_start_mb
- rss_end_mb
- rss_delta_mb
- status

`performance_hotspots.csv` contains one row per run with the top three
wall-clock stages and total profiled wall-clock time.

## Non-Goals

Performance Forensics V1 does not:

- change model formulas or objective functions
- change training update rules
- introduce tuning logic
- alter split semantics
- optimize cache behavior
- replace KMeans or model kernels
- create speed, scalability, memory-efficiency, or readiness claims
- make previous benchmark results comparable to new ones without a clean
  benchmark contract

If a stage appears expensive, that observation is a profiling readout. It is
not evidence that a proposed change is faster or more memory-efficient.

## Deriving Next Steps

Use the profiles to decide which work needs a separate optimization contract.
A follow-up optimization should identify:

- the dominant stage or stages from `performance_hotspots.csv`
- the exact dataset, split, model profile, seed, dtype, and device profile
- whether the proposed change is exact or changes method semantics
- the regression tests needed to preserve metrics and model behavior
- the before/after benchmark command and acceptance criteria

Only after a clean before/after benchmark under matching conditions may a
runtime or memory claim be considered.
