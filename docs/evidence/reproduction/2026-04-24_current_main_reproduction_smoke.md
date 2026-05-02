# Evidence Note

- date: `2026-04-24`
- scope: `current main reproduction smoke`
- status: `pass`
- git_commit_before_evidence_update: `7298d85`
- git_dirty: `false`

## Claim Or Question

Can the current `main` branch be treated as reproduction-ready under Gate 6 of
`docs/project_master_plan.md`, including the canonical `uv sync --extra dev`
setup path?

## Inputs And Artifacts

- branch: `main`
- latest commit before this evidence update: `7298d85`
- repository root:
  current workspace root
- environment contract:
  `docs/environment_contract.md`
- manifest used for smoke validation:
  `artifacts/runs/2026-04-24T181239Z_ml20m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001/run_manifest.json`
- lockfile:
  `uv.lock`

## Method

Provision `uv`, generate the canonical lockfile, run the canonical setup path,
and execute non-training reproduction checks that exercise the installed
environment, repository entry point, manifest validation, and full test suite.

Commands executed:

```powershell
git status --short --branch
uv --version
Test-Path uv.lock
uv sync --extra dev --locked
.venv\Scripts\python.exe -m recsys_lab.cli.main bootstrap-check
.venv\Scripts\python.exe -m recsys_lab.cli.main validate-manifest artifacts\runs\2026-04-24T181239Z_ml20m_cb_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001\run_manifest.json
.venv\Scripts\python.exe -m pytest
uv lock --check
```

## Readout

- `git status --short --branch`:
  `## main...origin/main`
- Python version:
  `Python 3.10.7`
- `uv` provisioning:
  installed through Winget package `astral-sh.uv`
- effective `uv` version:
  `uv 0.11.7`
- `uv.lock`:
  present and validated by `uv lock --check`
- `uv sync --extra dev --locked`:
  completed successfully
- packages installed by `uv sync`:
  `54`
- setup warning:
  hardlink fallback to full copy because cache and target appear to be on
  different filesystems; this affects setup performance, not the dependency
  resolution result
- `bootstrap-check`:
  `healthy: true`
- required path check:
  `agents`, `artifacts`, `configs`, `data`, `docs`, `pyproject`, `schema`,
  `scripts`, `src`, and `tests` all present
- manifest validation:
  `status: valid`
- final full test suite from `.venv` after evidence updates:
  `104 passed in 28.46s`

## Interpretation

The current `main` branch now satisfies the Gate 6 reproduction requirement for
the local reference machine: the canonical `uv` setup path is available, the
lockfile is present and internally consistent, the project installs into a
fresh `.venv`, the CLI entry point can inspect repository health, a current run
manifest validates successfully, and the full test suite passes from the
`uv`-managed environment.

The setup took longer than ordinary test execution because the project lives on
a Google Drive-backed path and `uv` fell back from hardlinking to full file
copying. This is a setup-performance observation, not a benchmark claim.

## Decision Or Next Step

- Mark Gate 6 as `pass`.
- Version `uv.lock` as the canonical lockfile for the current dependency state.
- Move to release-facing README/report language and final release hygiene.
