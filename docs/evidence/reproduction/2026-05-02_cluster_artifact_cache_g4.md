# Cluster Artifact Cache G4

- date: `2026-05-02`
- status: `pass_for_synthetic_cache_gate`
- scope: `G4 leakage-safe cluster artifact cache`
- gate: `docs/roadmaps/2026-05-02_claim_unlock_and_scalability_plan.md`

## Purpose

This note documents the G4 remediation gate for persisted train-only CB cluster
artifacts. It does not unlock any speed, scalability, model-quality,
large-dataset, tuning, or paper-faithfulness claim.

## Implementation

Implemented:

- `load_or_build_cluster_artifacts`
- `load_or_build_user_cluster_history_index`
- local cache root below `artifacts/local/cb_clusters`
- short cache paths to avoid Windows path-length failures
- atomic `.npy` and JSON writes via shared atomic I/O helpers
- cache identity containing dataset, split, train fingerprint, induction config,
  cluster config, and processed-manifest reference
- relative artifact references inside cache manifests
- top-level `caches` payload in completed CB run manifests
- `caches` payload in CB metrics
- CLI flags:
  `--cluster-artifact-cache/--disable-cluster-artifact-cache`

## Cache Identity Boundary

The cluster-artifact cache identity is train-only:

- train rating fingerprint
- dataset short name
- split family and split id
- processed manifest reference
- induction `BiasedMFConfig`
- `n_user_clusters`
- `n_item_clusters`
- clustering algorithm
- `kmeans_n_init`

Validation and test ratings are not passed to the cache builder. A unit test
mutates a non-train rating while keeping the train subset fixed and verifies
that the second lookup remains a cache `hit` with the same cache key.

## Verification

Focused commands:

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/test_cluster_cache.py
.venv\Scripts\python.exe -m pytest tests/unit/test_training_index_cache.py tests/unit/test_cli_main.py
.venv\Scripts\python.exe -m pytest tests/integration/test_cb_svdpp_run.py
.venv\Scripts\python.exe -m pytest tests/integration/test_cb_asvdpp_run.py
.venv\Scripts\python.exe -m ruff check src/recsys_lab/clustering/cache.py src/recsys_lab/data/training_index_cache.py src/recsys_lab/experiments/cb_svdpp.py src/recsys_lab/experiments/cb_asvdpp.py src/recsys_lab/cli/main.py tests/unit/test_cluster_cache.py tests/unit/test_training_index_cache.py tests/unit/test_cli_main.py tests/integration/test_cb_svdpp_run.py tests/integration/test_cb_asvdpp_run.py
.venv\Scripts\python.exe -m mypy src
```

Focused readout:

- Cluster cache unit gate: `4 passed`
- Training-index and CLI unit gate: `8 passed`
- CB-SVD++ integration gate: `3 passed`
- CB-ASVD++ integration gate: `2 passed`
- Ruff focused gate: `All checks passed!`
- Mypy source gate: `Success: no issues found in 62 source files`

Full repo gate:

```powershell
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy src
.venv\Scripts\python.exe -m pytest
```

Readout:

- Ruff: `All checks passed!`
- Mypy source gate: `Success: no issues found in 62 source files`
- Pytest: `127 passed`

## Claim Boundary

Allowed claim:

- `G4` leakage-safe cluster artifact cache plumbing exists for `cb_svdpp` and
  `cb_asvdpp`, with manifest-visible cache status and fingerprints.
- Synthetic tests demonstrate cache `miss -> hit`, train-fingerprint
  invalidation, and non-train rating isolation.
- Marker: train-fingerprint invalidation.

Explicit non-claims:

- no speed claim
- no scalability claim
- no large-dataset cache benchmark claim
- no model-quality claim
- no validation-tuning claim
- no paper-faithfulness claim
