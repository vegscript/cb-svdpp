# Hotpath / Coldpath Boundary

This document is a standing architecture rule. It is not evidence for a specific
benchmark and it does not make performance claims.

## Why This Boundary Exists

Training kernels and model fit/predict code must stay small, array-oriented, and
predictable. Experiment orchestration, configuration, reports, manifests, cache
metadata, and evidence writing are necessary, but they must not leak into the
rating update path.

The boundary exists to keep:

- model semantics easier to audit
- training loops free of filesystem and reporting work
- profiling artifacts interpretable
- future optimization work focused on kernels and arrays
- experiment lifecycle code reproducible but outside hot loops

## Hotpath Files

Strict hotpath:

- `src/recsys_lab/models/kernels.py`

Model hotpath:

- `src/recsys_lab/models/biased_mf.py`
- `src/recsys_lab/models/svdpp.py`
- `src/recsys_lab/models/asymmetric_svd.py`
- `src/recsys_lab/models/asvdpp.py`
- `src/recsys_lab/models/cb_svdpp.py`
- `src/recsys_lab/models/cb_asvdpp.py`
- `src/recsys_lab/models/inference.py`

Hotpath preparation:

- `src/recsys_lab/data/histories.py`

This module is stricter than cache-boundary modules: it owns in-memory
CSR-like history layout construction and validation only. It must not grow JSON,
YAML, `Path`, manifest, reporting, evidence, experiment, CLI, or atomic IO
responsibilities.

Boundary preparation/cache modules:

- `src/recsys_lab/data/training_index_cache.py`
- `src/recsys_lab/clustering/latent_kmeans.py`
- `src/recsys_lab/clustering/cache.py`

Boundary modules may build arrays or artifacts used by hotpath code, but their
cache IO and metadata work must happen before model training loops.

## Coldpath Files

Coldpath code owns orchestration, configuration, validation, profiling,
reporting, and persistence.

Coldpath examples:

- `src/recsys_lab/experiments/unified_runner.py`
- `src/recsys_lab/experiments/performance.py`
- `src/recsys_lab/experiments/kernel_profile.py`
- `src/recsys_lab/config/loader.py`
- `src/recsys_lab/cli/main.py`
- `src/recsys_lab/reporting/`
- `scripts/`
- `docs/`
- `configs/`
- `schema/`

## Forbidden Hotpath Dependencies

Strict hotpath code must not import or call:

- YAML or JSON loaders/writers
- Pydantic schemas
- `Path` or filesystem path discovery
- `open(`
- manifest validators or writers
- `write_json`
- `dump_yaml_file`
- `load_yaml_file`
- `validate_manifest_file`
- `discover_repo_root`
- `repo_path_string`
- Markdown, evidence, or reporting code
- CLI frameworks such as `argparse`, `typer`, or `click`
- `print` or logging inside kernels or inner loops

Model hotpath code may own dataclass config objects and in-memory arrays, but it
must not load YAML, write JSON, validate run manifests, generate evidence, or
write report artifacts.

Cache/artifact boundary modules may use `Path`, JSON, hashing, and atomic IO for
cache metadata. They must not own model training semantics or experiment report
logic.

## Adding New Model Features

New model features should be placed by responsibility:

- prediction formulas, factor state, and training updates belong in model
  hotpath code or Numba kernels
- array builders needed before fit belong in hotpath preparation or boundary
  preparation modules
- config validation belongs in model schema/registry coldpath infrastructure
- run orchestration belongs in `experiments/`
- reporting and evidence belong in `reporting/`, `docs/`, or `artifacts/`

If a feature needs both a new array and a new config option, split the
responsibilities: coldpath validates and constructs the config, preparation code
builds arrays, and model hotpath consumes already-built arrays.

## Adding New Tuning Features

Tuning engines are coldpath. They may select configs, schedule runs, collect
metrics, and write evidence. They must not add tuning decisions inside kernels,
rating loops, prediction loops, or model update formulas.

Allowed tuning responsibilities:

- candidate config generation
- run scheduling
- validation metric collection
- artifact and evidence writing
- cache policy selection before runs

Forbidden tuning responsibilities:

- dynamic hyperparameter changes inside rating loops unless explicitly
  documented as a model-method change
- test-set-driven selection
- report/evidence writes from model hotpath code
- hidden changes to dtype, split, seed, or update order

## Performance Forensics And Kernel Profiling

Performance Forensics and Kernel Cost Anatomy respect this boundary by running
outside kernels:

- `StageProfiler` measures experiment stages in the runner.
- `performance_profile.json` is written by coldpath code after or around stages.
- `kernel_profile.json` is built after `fit_model` from existing model state,
  epoch durations, training data, and fit artifacts.
- Kernel profile helpers summarize counts and estimated work; they do not enter
  Numba inner loops and do not change model behavior.

Profiling code may observe hotpath results, but it must not become part of the
rating update path.

## Guardrails

The boundary is guarded by:

- `docs/architecture/hotpath_coldpath_boundary_audit_v1.md`
- `tests/unit/test_hotpath_coldpath_boundaries.py`

The unit guard covers strict kernels, model hotpath files, benchmark back-import
direction, and `data/histories.py` as hotpath preparation. Future changes that
add coldpath imports or artifact logic to strict/model hotpath files or
`data/histories.py` should fail review unless they are moved to coldpath code or
documented as a deliberate architecture change.
