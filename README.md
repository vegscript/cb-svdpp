# Recommender System Reproduction Lab

## What This Repo Does

This repository is a research and engineering platform for controlled
rating-prediction experiments with factorized recommender models and
clustering-based extensions. Productive results must be backed by run artifacts,
manifests, dated evidence notes, and the publish-readiness matrix.

Release marker: `submission-2026-05-02-r10`

## Model Ladder

The active model ladder contains exactly six registry-backed models:

1. `biased_mf`
2. `svdpp`
3. `asymmetric_svd`
4. `asvdpp`
5. `cb_svdpp`
6. `cb_asvdpp`

## Rating-Only Scope

The current evaluation scope is explicit rating prediction. RMSE remains the
primary metric, with MAE, error-distribution metrics, prediction range metrics,
runtime, memory, cache, profiling, and manifest fields as supporting evidence.

The repo does not currently evaluate ranking metrics, genres, tags, content
features, online serving, or production readiness.

## Unified Experiment Framework

Productive model runs go through `run_unified_experiment` and the model
registry. The active framework surface is:

- `MODEL_REGISTRY`
- `ModelRequirements`
- `FitArtifacts`
- Pydantic model profiles in `configs/models/`
- `RatingsData`
- history, explicit-feedback, cluster-artifact, metrics, manifest, profiling,
  and cache builders

Legacy CLI and experiment wrappers are retained only as tested delegation paths
into the unified runner.

## Setup

The dependency source of truth is `pyproject.toml` and `uv.lock`.

```powershell
uv sync --extra dev --locked
```

Basic local checks:

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main bootstrap-check
.venv\Scripts\ruff.exe check .
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m mypy --ignore-missing-imports --follow-imports=skip --no-incremental src/recsys_lab/utils/paths.py src/recsys_lab/metrics/errors.py src/recsys_lab/config/loader.py
```

The Mypy command is a scoped smoke gate only. It is not a full repository
type-safety claim.

## Canonical Commands

Unified training entry point:

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main train --model biased_mf --processed-manifest data/processed/ml100k/manifest.json --model-config configs/models/biased_mf.yaml --runtime-config configs/runtime/base.yaml --device-config configs/runtime/devices/local_i5_2500k_24gb.yaml
```

Current artifact-derived report table:

```powershell
.venv\Scripts\python.exe scripts\collect_results.py
```

## Current Evidence

Use `docs/evidence/current_evidence_index.md` for current evidence navigation.
The README intentionally does not list every evidence note.

Current clean benchmark anchors:

| Dataset | Claimable Scope |
| --- | --- |
| `ml100k` | Clean multi-seed anchors for `biased_mf`, `svdpp`, `cb_svdpp`, and `cb_asvdpp`. |
| `ml1m` | Clean matched multi-seed comparison for `biased_mf` vs `cb_svdpp`. |
| `ml10m` | Clean matched multi-seed comparison for documented transfer profiles. |
| `ml20m` | Clean `biased_mf` baseline anchor plus archived/resource evidence for clustering-model attempts; no final model-comparison claim. |

Current generated run indexes:

- `docs/evidence/runs/ml100k_run_index.md`
- `docs/evidence/runs/ml1m_run_index.md`
- `docs/evidence/runs/ml10m_run_index.md`
- `docs/evidence/runs/ml20m_run_index.md`

## Claim Boundary

The final claim boundary is controlled by:

- `docs/publish_readiness_matrix.md`
- `docs/evaluation_protocol.md`
- `docs/evidence/current_evidence_index.md`
- `docs/report/project_report.md`

Non-claims remain explicit:

- no exact paper reproduction claim
- no exact optimizer-faithful CB training claim
- no final `ml20m` model-comparison claim
- no final `ml1m cb_asvdpp` benchmark-anchor claim
- no ranking-performance claim
- no genre, tag, or content-feature usage claim
- no production-readiness claim
- no unqualified `better`, `faster`, or `scalable` claim
