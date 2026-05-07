# Hotpath / Coldpath Boundary Audit V1 Evidence

## Branch

`hotpath-coldpath-boundary-audit-v1`

## Files Audited

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

Hotpath preparation and boundary/cache modules:

- `src/recsys_lab/data/histories.py`
- `src/recsys_lab/data/training_index_cache.py`
- `src/recsys_lab/clustering/latent_kmeans.py`
- `src/recsys_lab/clustering/cache.py`

Coldpath reference files and directories:

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

## Static Tests Added

Added `tests/unit/test_hotpath_coldpath_boundaries.py`.

The test guard checks that:

- `src/recsys_lab/models/kernels.py` contains no strict hotpath-forbidden
  coldpath imports or terms.
- `src/recsys_lab/models/kernels.py` contains no filesystem IO patterns.
- `src/recsys_lab/models/kernels.py` contains no JSON, YAML, Pydantic,
  manifest, evidence, or reporting terms.
- model hotpath files do not import config loaders, manifest writers,
  repository path rendering helpers, evidence, or reporting code.
- `src/recsys_lab/experiments/unified_runner.py` is allowed to contain
  coldpath terms.
- reporting, scripts, and docs paths are not included in the hotpath forbidden
  term scan.

Step 13b retained the token guard and added AST import guards:

- `_imported_modules(path: Path) -> set[str]` extracts `import x`,
  `import x.y`, `from x import y`, and `from x.y import z`.
- `src/recsys_lab/models/kernels.py` is guarded against AST imports from
  `recsys_lab.config`, `recsys_lab.cli`, `recsys_lab.experiments`,
  `recsys_lab.reporting`, selected `recsys_lab.utils.*` coldpath helpers,
  `pydantic`, `yaml`, `json`, `pathlib`, CLI frameworks, and `logging`.
- model hotpath files are guarded against AST imports from the same config,
  CLI, experiment, reporting, manifest/path utility, schema, YAML, JSON,
  filesystem path, CLI, and logging modules.
- negative unit tests create temporary files and prove that
  `from recsys_lab.experiments.performance import StageProfiler` is extracted
  and rejected without modifying real hotpath files.

No whitelist was added.

## Findings

1. `src/recsys_lab/models/kernels.py` is cleanly strict hotpath-scoped. It does
   not contain YAML, JSON, Pydantic, manifest, reporting, path discovery, or
   file IO logic.
2. The audited model hotpath files remain focused on model configs, in-memory
   arrays, fit/predict state, epoch timing, and kernel dispatch. They do not own
   config loading, manifest writing, evidence generation, or reporting.
3. `src/recsys_lab/data/histories.py` is hotpath preparation and remains
   in-memory.
4. `src/recsys_lab/data/training_index_cache.py` and
   `src/recsys_lab/clustering/cache.py` are boundary/cache modules. Their JSON,
   `Path`, hashing, and atomic IO responsibilities are accepted only as
   coldpath cache orchestration before training loops.
5. `src/recsys_lab/clustering/latent_kmeans.py` is boundary preparation, not a
   Numba kernel hotpath.
6. `src/recsys_lab/experiments/unified_runner.py`,
   `src/recsys_lab/experiments/performance.py`, and
   `src/recsys_lab/experiments/kernel_profile.py` remain coldpath owners for
   experiment lifecycle, stage profiling, kernel-profile payload construction,
   artifact writing, metrics, and manifests.
7. A case-insensitive manual scan produced one false positive on
   `RatingsDataFingerprint(` because the substring contains `Print(`. This is
   not a `print(...)` call and is not a boundary violation.

## Fixes Made

No production code fixes were made.

The change is limited to:

- documenting the audit result in
  `docs/architecture/hotpath_coldpath_boundary_audit_v1.md`
- adding the standing architecture note
  `docs/architecture/hotpath_coldpath_boundary.md`
- adding the token-based static unit guard
  `tests/unit/test_hotpath_coldpath_boundaries.py`
- adding the Step 13b AST import-boundary guard in
  `tests/unit/test_hotpath_coldpath_boundaries.py`
- recording this evidence note

No kernel, model, fit, predict, cache, manifest, split, dtype, seed, tuning, or
training semantics were changed.

## Deferred Items

1. Consider module docstrings for `src/recsys_lab/data/training_index_cache.py`
   and `src/recsys_lab/clustering/cache.py` stating that they are coldpath
   cache-boundary modules.
2. Consider an ownership note for `src/recsys_lab/models/config_schemas.py` and
   `src/recsys_lab/models/registry.py`, because they live under `models/` but
   are not model training hotpaths.
3. Revisit whether model-local epoch timing with `perf_counter` should remain
   accepted lightweight telemetry or move fully into coldpath profiling.

## Tests Run

Focused checks run during this audit:

- `pytest tests/unit/test_hotpath_coldpath_boundaries.py`
- `ruff check .`
- `pytest tests/unit`
- `pytest tests/integration/test_unified_pipeline_smoke_all_models.py`
- `pytest`
- `rg "guaranteed speedup|production-ready|SOTA speedup|broad performance claim" docs src tests`

Result recorded at the time of this note:

- boundary unit test: passed, 10 tests
- full ruff check: passed
- full unit suite: passed, 143 tests
- unified pipeline smoke for all models: passed, 1 test
- full pytest suite: passed, 214 tests and 3 skipped
- claim check: `rg` completed. Hits were existing claim-boundary or
  claim-prohibition contexts, not new performance claims.

## Gates

Gates run for this Step 13 branch:

- `ruff check .`: passed
- `python -m pytest tests/unit/test_hotpath_coldpath_boundaries.py`: passed
- `python -m pytest tests/unit`: passed
- `python -m pytest tests/integration/test_unified_pipeline_smoke_all_models.py`: passed
- `python -m pytest`: passed
- `rg "guaranteed speedup|production-ready|SOTA speedup|broad performance claim" docs src tests`:
  passed; hits were existing claim-boundary or claim-prohibition contexts

## Claim Boundary

This evidence note does not make a performance, memory, scalability, model
quality, or merge-readiness claim. It records a static architecture boundary
audit and the guard added to keep strict/model hotpath files free of coldpath
responsibilities.

## Next Red-Thread Step

14. Kernel Benchmark Harness
