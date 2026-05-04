# Recommender System Reproduction Lab

## What This Repo Does

This repository is a research and engineering platform for controlled
experiments with factorized recommender models, including clustering-based
extensions. The current implementation is claim-limited: reported results must
match the publish-readiness matrix and dated evidence notes.

Release marker: `submission-2026-05-02-r10`

## Model Ladder

The active model ladder contains exactly six registry-backed models:

1. `biased_mf`
2. `svdpp`
3. `asymmetric_svd`
4. `asvdpp`
5. `cb_svdpp`
6. `cb_asvdpp`

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

Legacy CLI and experiment wrappers are retained only as delegation paths into
the unified runner.

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

Model-specific wrappers may exist for compatibility with previous command
records, but new productive runs should prefer `train --model`.

## Current Claim Boundary

The final claim boundary is controlled by:

- `docs/publish_readiness_matrix.md`
- `docs/evaluation_protocol.md`
- `docs/evidence/current_evidence_index.md`

Non-claims remain explicit:

- no exact paper reproduction claim
- no exact optimizer-faithful CB training claim
- no final `ml20m` model-comparison claim
- no final `ml1m cb_asvdpp` benchmark-anchor claim
- no unqualified `faster`, `scalable`, or production-readiness claim

Current clean benchmark anchors:

| Dataset | Claimable Scope |
| --- | --- |
| `ml100k` | Clean multi-seed anchors for `biased_mf`, `svdpp`, `cb_svdpp`, and `cb_asvdpp`; the G6-selected `cb_svdpp` profile is a separate `benchmark_random_v1` readout. |
| `ml1m` | Clean matched multi-seed comparison for `biased_mf` vs `cb_svdpp`. |
| `ml10m` | Clean matched multi-seed comparison for the documented transfer profiles. |
| `ml20m` | Clean `biased_mf` baseline anchor plus `cb_svdpp` blocked negative resource evidence; no final model-comparison claim. |

## Repository Structure

- `src/recsys_lab/`: active implementation code
- `tests/`: unit, integration, and documentation guardrails
- `configs/models/`: base profiles, selected profiles, and archived tuned profiles
- `configs/experiments/`: active, template, and archived experiment configs
- `docs/`: governance, math specs, evidence, and report
- `scripts/`: reproducible helper entry points
- `schema/`: run and benchmark manifest schemas
- `data/`: local data zones, ignored except placeholders
- `artifacts/`: generated run and benchmark outputs, ignored except placeholders

## Evidence Index

Use `docs/evidence/current_evidence_index.md` for current evidence navigation.
The README intentionally does not list every reproduction note.
