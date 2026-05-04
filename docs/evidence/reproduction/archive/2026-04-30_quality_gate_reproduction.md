# 2026-04-30 Quality Gate Reproduction Evidence

- status: `pass`
- release_marker: `submission-2026-04-30-r8`
- scope: `post-ml10m-cb-svdpp-matched-profile quality gates for the claim-limited release candidate`
- environment: local `uv` environment
- python: `3.10.7`
- uv: `uv 0.11.7`

## Commands And Readouts

The following commands were executed from the repository root.

```powershell
uv sync --extra dev --locked
```

Readout: completed with the locked dev environment: `Resolved 60 packages` and
`Checked 57 packages`.

```powershell
.venv\Scripts\python.exe -m ruff check .
```

Readout: `All checks passed!`

```powershell
.venv\Scripts\python.exe -m mypy src
```

Readout: `Success: no issues found in 60 source files`

```powershell
.venv\Scripts\python.exe -m pytest tests\integration\test_publish_readiness_plan.py tests\integration\test_large_dataset_claim_locks.py tests\integration\test_release_evidence_integrity.py
```

Readout: `9 passed in 2.18s`

```powershell
.venv\Scripts\python.exe -m pytest
```

Readout: `111 passed in 28.71s`

```powershell
git diff --check
```

Readout: passed.

## Claim Impact

This evidence keeps the local Ruff and Mypy quality gates closed for the
current codebase and revalidates the full test suite. It incorporates the
committed `ml10m cb_svdpp` matched-campaign profile, but it does not unlock a final
large-dataset model-comparison claim. The repository remains a claim-limited
release candidate.
