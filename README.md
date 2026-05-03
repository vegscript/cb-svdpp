# Recommender System Reproduction Lab

This repository is a modular research and engineering platform for a
methodologically controlled reproduction and extension of clustering-based
factorized collaborative filtering.

The implemented model ladder is:

1. `biased_mf`
2. `svdpp`
3. `asymmetric_svd`
4. `asvdpp`
5. `cb_svdpp`
6. `cb_asvdpp`

The current release boundary is claim-limited. It supports clean benchmark
claims only where the publish-readiness matrix explicitly allows them.

## Release Status

- Release marker: `submission-2026-05-02-r10`
- Current status: `release_candidate_claim_limited`
- Canonical claim matrix: `docs/publish_readiness_matrix.md`
- Project report: `docs/report/project_report.md`
- Reproduction setup evidence:
  `docs/evidence/reproduction/2026-05-02_public_clean_import.md`
- Public path hygiene evidence:
  `docs/evidence/reproduction/2026-05-03_public_path_hygiene.md`
- Validation-grid contract evidence:
  `docs/evidence/reproduction/2026-05-03_cb_svdpp_g6_validation_grid_contract.md`
- Validation-grid run evidence:
  `docs/evidence/reproduction/2026-05-03_cb_svdpp_g6_validation_grid_run.md`
- G6 outer benchmark contract evidence:
  `docs/evidence/reproduction/2026-05-03_cb_svdpp_g6_outer_benchmark_contract.md`
- G6 outer benchmark run evidence:
  `docs/evidence/reproduction/2026-05-03_cb_svdpp_g6_outer_benchmark_run.md`
- CB-ASVD++ hotpath decision evidence:
  `docs/evidence/reproduction/2026-05-03_cb_asvdpp_hotpath_decision_g7.md`
- CB-ASVD++ hotpath remediation contract evidence:
  `docs/evidence/reproduction/2026-05-03_cb_asvdpp_hotpath_remediation_contract_g8.md`
- CB-ASVD++ hotpath pre-change baseline evidence:
  `docs/evidence/reproduction/2026-05-03_cb_asvdpp_hotpath_prechange_baseline_g9.md`
- CB-ASVD++ hotpath post-change benchmark evidence:
  `docs/evidence/reproduction/2026-05-03_cb_asvdpp_hotpath_postchange_benchmark_g10.md`

This repository is not an unconstrained `paper-faithful` or `scalable`
reproduction. The clustering-based models are documented as source-grounded
predictors with repo-defined optimization under `D-004`.

## Setup

The canonical setup path is:

```powershell
uv sync --extra dev --locked
```

The dependency source of truth is:

- `pyproject.toml`
- `uv.lock`

The current local reproduction evidence used:

- Python `3.10.7`
- `uv 0.11.7`
- `uv sync --extra dev --locked`
- Ruff gate: `All checks passed!`
- Mypy source gate: `Success: no issues found in 62 source files`
- full test suite from the `uv` environment: `141 passed`

Basic smoke checks:

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main bootstrap-check
.venv\Scripts\python.exe -m pytest
```

Manifest validation examples in the evidence notes require the corresponding
local, ignored run or benchmark artifact to exist.

## Claimable Results

The final claim boundary is frozen in `docs/publish_readiness_matrix.md`.
The report may only use the claims listed there.

Current clean benchmark anchors:

| Dataset | Claimable Scope |
| --- | --- |
| `ml100k` | Clean multi-seed anchors for `biased_mf`, `svdpp`, `cb_svdpp`, and `cb_asvdpp`; additionally, a clean outer `benchmark_random_v1` readout exists for the frozen G6-selected `cb_svdpp` profile. |
| `ml1m` | Clean matched multi-seed comparison for `biased_mf` vs `cb_svdpp`. |
| `ml10m` | Clean matched multi-seed comparison for the documented `biased_mf` vs `cb_svdpp` transfer profiles. |
| `ml20m` | Clean multi-seed `biased_mf` baseline anchor plus `cb_svdpp` feasibility and negative resource evidence; no final model-comparison claim. |

Important non-claims:

- no exact paper reproduction claim
- no exact optimizer-faithful CB training claim
- no final `ml20m` model-comparison claim
- no general large-dataset `cb_svdpp` superiority claim; `ml10m` is limited to
  the documented matched profile comparison
- no final `ml1m cb_asvdpp` benchmark-anchor claim
- no unqualified `faster`, `scalable`, or unconstrained `ready` claim
- no additional large-dataset `cb_svdpp` matched-campaign claim until a
  stronger device profile or documented lower-memory matched profile satisfies
  a fresh campaign contract

## Data And Artifact Policy

Large or rebuildable data and outputs are intentionally not versioned in git:

- raw data lives under `data/raw/`
- processed data lives under `data/processed/`
- run artifacts live under `artifacts/runs/`
- benchmark artifacts live under `artifacts/benchmarks/`
- local caches live under `artifacts/local/`

These zones are guarded by `.gitignore`; only `.gitkeep` / README placeholders
and documentation are versioned. Evidence notes in `docs/evidence/` point to
the relevant artifact paths and summarize the claimable readouts.

## License

The repository code and documentation are licensed under the MIT License; see
`LICENSE`.

External datasets, papers, and generated local artifacts are not covered by
this repository license. They remain subject to their own source licenses and
usage terms.

## Repository Structure

- `src/recsys_lab/`: active implementation code
- `tests/`: unit, integration, and documentation guardrails
- `configs/`: canonical data, runtime, model, and experiment configs
- `docs/`: governance, math specs, evidence, and report
- `scripts/`: reproducible helper entry points
- `schema/`: run and benchmark manifest schemas
- `data/`: local data zones, ignored except placeholders
- `artifacts/`: generated run and benchmark outputs, ignored except placeholders

## Governance

Core contracts:

- `docs/project_master_plan.md`
- `docs/repo_governance.md`
- `docs/environment_contract.md`
- `docs/data_and_split_contract.md`
- `docs/evaluation_protocol.md`
- `docs/manifest_contract.md`
- `docs/report/report_contract.md`
- `docs/methodology/deviations_from_paper.md`

If a claim is not backed by a test, manifest, benchmark, or dated evidence note,
it must be treated as a hypothesis rather than as a result.
