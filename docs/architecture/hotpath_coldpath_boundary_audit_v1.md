# Hotpath / Coldpath Boundary Audit V1

## Purpose

This audit records the current architectural boundary between training hotpaths
and experiment coldpaths. It follows Performance Forensics V1, Kernel Cost
Anatomy V1, Kernel Optimization Plan V1, Exact Kernel Optimization V1, and the
Exact Kernel Optimization Acceptance decision.

The goal is boundary clarity, not optimization. This document does not introduce
new kernels, new tuning behavior, new model formulas, or performance claims.

## Definitions: Hotpath vs Coldpath

Hotpath code is code that runs inside, or directly prepares data for, model
fit/predict cores. It should be simple, array-oriented, and free of experiment
or reporting responsibilities.

Hotpath code may use:

- NumPy arrays
- contiguous memory expectations
- integer user/item ids
- configured floating dtypes
- CSR-like history indices
- Numba kernels
- model-local fit and predict logic
- small dataclasses that wrap arrays without file IO

Coldpath code is orchestration, persistence, reporting, validation, and
experiment lifecycle code.

Coldpath code may use:

- YAML and JSON
- Pydantic schemas
- manifests
- path discovery and path rendering
- artifact writing
- CLI orchestration
- reporting collectors
- evidence and documentation logic
- cache manifests and cache metadata
- profiling payload construction

## File Classification Table

| Path | Classification | Reason |
| --- | --- | --- |
| `src/recsys_lab/models/kernels.py` | HOTPATH | Numba training kernels. No YAML, JSON, paths, manifests, reporting, or experiment imports. |
| `src/recsys_lab/models/biased_mf.py` | HOTPATH | Model config dataclass, fit/predict core, and Numba dispatch. No coldpath IO or manifest responsibilities. |
| `src/recsys_lab/models/svdpp.py` | HOTPATH | Model fit/predict core plus history index consumption. No coldpath IO or reporting responsibilities. |
| `src/recsys_lab/models/asymmetric_svd.py` | HOTPATH | Model fit/predict core plus explicit/implicit history consumption. No coldpath IO or reporting responsibilities. |
| `src/recsys_lab/models/asvdpp.py` | HOTPATH | Model fit/predict core plus explicit/implicit history consumption. No coldpath IO or reporting responsibilities. |
| `src/recsys_lab/models/cb_svdpp.py` | HOTPATH | Model fit/predict core plus cluster-history array consumption. No coldpath IO or reporting responsibilities. |
| `src/recsys_lab/models/cb_asvdpp.py` | HOTPATH | Model fit/predict core plus explicit, implicit, and cluster-history array consumption. No coldpath IO or reporting responsibilities. |
| `src/recsys_lab/models/inference.py` | HOTPATH | Shared array-based inference cache construction. No coldpath IO or artifact writing. |
| `src/recsys_lab/data/histories.py` | HOTPATH PREPARATION | Builds CSR-like in-memory history structures from `RatingsData`. No file IO or manifests. |
| `src/recsys_lab/data/training_index_cache.py` | BOUNDARY / COLDPATH CACHE | Builds hotpath history arrays, but owns cache manifests, JSON, paths, hashing, and atomic IO. Must not run inside rating or epoch loops. |
| `src/recsys_lab/clustering/latent_kmeans.py` | BOUNDARY PREPARATION | Train-only clustering and dense-array preparation for CB models. Uses sklearn and a temporary BiasedMF model, so it is not a kernel hotpath. |
| `src/recsys_lab/clustering/cache.py` | BOUNDARY / COLDPATH CACHE | Orchestrates cluster artifact caches, manifests, JSON, paths, hashing, and atomic IO. Must remain outside model fit loops. |
| `src/recsys_lab/experiments/unified_runner.py` | COLDPATH | Experiment lifecycle, config snapshots, manifests, profiles, metrics JSON, artifact writing, and stage orchestration. |
| `src/recsys_lab/experiments/performance.py` | COLDPATH | Stage profiling, memory monitoring, system metrics, and performance profile payloads. |
| `src/recsys_lab/experiments/kernel_profile.py` | COLDPATH | Kernel anatomy payload construction from completed model state and fit artifacts. Runs after fit, not inside kernels. |
| `src/recsys_lab/config/loader.py` | COLDPATH | YAML loader/dumper and config parser boundary. |
| `src/recsys_lab/cli/main.py` | COLDPATH | CLI orchestration and command argument parsing. |
| `src/recsys_lab/reporting/` | COLDPATH | Result and profile collectors for report tables. |
| `scripts/` | COLDPATH | Shell and Python wrappers for CLI, reporting, preparation, and benchmarks. |
| `docs/` | COLDPATH | Evidence, architecture notes, methodology, and report documentation. |
| `configs/` | COLDPATH | Runtime, data, model, and experiment configuration files. |
| `schema/` | COLDPATH | Manifest and reporting schema contracts. |

## Allowed Dependencies For Hotpath

Hotpath modules may depend on:

- Python standard-library dataclasses and timing helpers where local model epoch
  durations are recorded.
- NumPy.
- Numba only in `models/kernels.py`.
- `RatingsData` as an in-memory data container.
- In-memory history dataclasses from `data/histories.py`.
- Other `recsys_lab.models.*` hotpath helpers such as kernels and inference.

## Forbidden Dependencies For Hotpath

Hotpath modules must not depend on:

- `recsys_lab.config.*`
- `recsys_lab.cli.*`
- `recsys_lab.experiments.*`
- `recsys_lab.reporting.*`
- `recsys_lab.utils.atomic_io`
- `recsys_lab.utils.manifests`
- `recsys_lab.utils.paths`
- Pydantic schemas
- Typer or CLI frameworks
- YAML or JSON loaders/writers
- `pathlib.Path` for artifact writing or path discovery
- Markdown/evidence/reporting logic
- Manifest validation or writing

Boundary modules such as `data/training_index_cache.py` and
`clustering/cache.py` may use JSON, paths, hashing, and atomic IO, but only as
coldpath cache orchestration. They must not be called inside rating loops or
epoch inner loops.

## Forbidden Hotpath Patterns

Forbidden patterns are evaluated by file category. A token such as `Path` is not
globally wrong in the repository: it is legitimate in cache/artifact boundary
code, but it is forbidden in the strict Numba kernel hotpath and should not
appear in model training loops.

### Strict Hotpath

Scope:

- `src/recsys_lab/models/kernels.py`

Rule:

`kernels.py` must contain only array-oriented kernel logic and Numba fallback
stubs. It must not import or call coldpath orchestration, filesystem, config,
manifest, reporting, or evidence logic.

Forbidden in strict hotpath:

- `yaml`
- `json`
- `pydantic`
- `Path`
- `open(`
- `write_json`
- `dump_yaml_file`
- `load_yaml_file`
- `validate_manifest_file`
- `discover_repo_root`
- `repo_path_string`
- `markdown`
- `report`
- `evidence`
- `argparse`
- `typer`
- `click`
- `print`
- logging calls

Additional strict rule: no logging or printing is allowed inside Numba kernels or
inner loops.

### Model Hotpath

Scope:

- `src/recsys_lab/models/biased_mf.py`
- `src/recsys_lab/models/svdpp.py`
- `src/recsys_lab/models/asymmetric_svd.py`
- `src/recsys_lab/models/asvdpp.py`
- `src/recsys_lab/models/cb_svdpp.py`
- `src/recsys_lab/models/cb_asvdpp.py`
- `src/recsys_lab/models/inference.py`

Rule:

Model hotpath modules may own model config dataclasses, NumPy arrays, fit/predict
state, local epoch duration recording, and dispatch to compiled kernels. They
must not load or write experiment configs, manifests, reports, evidence, or run
artifacts.

Forbidden in model hotpath:

- YAML loading or dumping: `yaml`, `load_yaml_file`, `dump_yaml_file`
- JSON loading or writing for artifacts/manifests: `json`, `write_json`
- Pydantic schema validation: `pydantic`
- Repository path discovery/rendering: `discover_repo_root`, `repo_path_string`
- Manifest validation/writing: `validate_manifest_file`, manifest writer logic
- CLI frameworks: `argparse`, `typer`, `click`
- Reporting/evidence/markdown logic: `report`, `evidence`, `markdown`
- `open(` for artifact or config IO
- `Path` for artifact, config, run-directory, or manifest responsibilities
- `print` or logging inside rating loops, history loops, prediction loops, or
  epoch loops

Allowed model-hotpath exceptions:

- Dataclass config objects that are already constructed by coldpath code.
- In-memory `RatingsData` and history dataclasses.
- `perf_counter` for local epoch duration telemetry.
- NumPy dtype conversion of in-memory arrays required by the configured model
  dtype.

### Data Hotpath / Artifact Builders

Scope:

- `src/recsys_lab/data/histories.py`
- `src/recsys_lab/data/training_index_cache.py`
- `src/recsys_lab/clustering/latent_kmeans.py`
- `src/recsys_lab/clustering/cache.py`

Rule:

Data hotpath preparation may build CSR-like arrays and train-only artifacts.
Cache/artifact builder modules may own cache manifests, JSON, paths, hashing,
and atomic IO, but only at the build/load boundary before model fit. They must
not own model training semantics, experiment reporting, or evidence decisions.

Allowed in cache/artifact boundary modules:

- `Path`
- `json`
- cache manifests
- hashing/fingerprints
- atomic array and JSON writes
- cache status metadata

Forbidden in data hotpath / artifact builders:

- Model training loop logic.
- Direct calls into experiment reporting or evidence writers.
- CLI parsing: `argparse`, `typer`, `click`.
- Markdown or evidence generation.
- Performance claim logic.
- Run manifest validation/writing outside explicit cache metadata contracts.
- Logging or printing inside array construction loops.

`data/histories.py` is stricter than the cache modules: it should remain
in-memory and should not grow JSON, `Path`, manifest, or atomic IO
responsibilities.

## Findings

1. `models/kernels.py` is cleanly hotpath-scoped. It imports NumPy and Numba
   only and contains no config, manifest, reporting, or filesystem logic.
2. The core model files remain hotpath-oriented. They own fit/predict state,
   model dataclass configs, epoch timings, and Numba dispatch, but do not load
   YAML, write JSON, write manifests, or touch evidence/reporting modules.
3. `models/config_schemas.py` and `models/registry.py` are not hotpath files.
   They contain Pydantic validation and adapter logic and should continue to be
   treated as coldpath/model-boundary infrastructure.
4. `data/histories.py` is hotpath preparation: it builds in-memory CSR-like
   structures from training arrays and does not write files.
5. `data/training_index_cache.py` is not pure hotpath. It wraps hotpath history
   builders with cache metadata, JSON manifests, paths, hashing, and atomic IO.
   This is acceptable only as boundary/coldpath cache code.
6. `clustering/latent_kmeans.py` is train-only preparation, not a Numba
   hotpath. It uses sklearn and a temporary BiasedMF model to create cluster
   artifacts before CB model fit.
7. `clustering/cache.py` is not pure hotpath. It owns cache metadata, JSON
   manifests, paths, hashing, and atomic IO for cluster artifacts and
   user-cluster history indices.
8. `experiments/unified_runner.py` correctly centralizes coldpath duties:
   config snapshots, run manifests, metrics JSON, performance profiles, kernel
   profiles, cache orchestration, split orchestration, and run lifecycle.
9. `experiments/kernel_profile.py` is coldpath by design. It computes structural
   summaries after fit using existing arrays and artifacts and does not enter
   Numba kernels.
10. No direct coldpath import was found in `models/kernels.py`.

## Required Fixes

No code fix is required for this classification phase.

Static checks found no strict/model hotpath imports of config loaders, manifest
writers, reporting/evidence modules, CLI frameworks, or filesystem IO helpers.
Ruff also found no unused imports in the audited hotpath and boundary files.

One case-insensitive search produced a false positive on
`RatingsDataFingerprint(` in `data/training_index_cache.py` because the token
contains the characters `Print(`. This is not a `print(...)` call and is not a
boundary violation.

The only required action from this phase is documentation of the boundary, the
identified boundary/cache modules, and the static guard added in
`tests/unit/test_hotpath_coldpath_boundaries.py`.

## Deferred Fixes

1. Add automated import-boundary tests that fail if hotpath modules import
   config, experiment, reporting, manifest, or filesystem-writing utilities.
2. Consider documenting `data/training_index_cache.py` and `clustering/cache.py`
   in module docstrings as coldpath cache-boundary modules.
3. Consider a small module ownership note for `models/config_schemas.py` and
   `models/registry.py`, because they live under `models/` but are not training
   hotpaths.
4. Review whether model epoch timing with `perf_counter` should remain in model
   classes or be treated as accepted lightweight hotpath-adjacent telemetry.

## Tests/guards Added

Added:

- `tests/unit/test_hotpath_coldpath_boundaries.py`

The guard checks:

- `src/recsys_lab/models/kernels.py` has no strict hotpath-forbidden terms.
- `src/recsys_lab/models/kernels.py` has no file IO patterns.
- `src/recsys_lab/models/kernels.py` has no JSON/YAML/Pydantic/manifest or
  reporting terms.
- model hotpath files do not import config loaders or manifest writers.
- `src/recsys_lab/experiments/unified_runner.py` is allowed to contain coldpath
  terms.
- reporting, scripts, and docs paths are not part of the hotpath forbidden-term
  checks.

No whitelist was needed for this phase because the audited strict and model
hotpath files had no legitimate forbidden-term hits.

## Claim Boundary

This audit does not claim a runtime improvement, memory improvement,
scalability, production readiness, or model quality improvement. It only records
the current hotpath/coldpath classification and boundary findings.
