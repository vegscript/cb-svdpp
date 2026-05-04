# Public Clean Import Evidence

- date: `2026-05-02`
- status: `pass_for_public_tree_hygiene`
- scope: `public repository hygiene and clean-root import`
- release marker: `submission-2026-05-02-r10`

## Purpose

This note records the checks used before replacing the public `main` history
with a clean-root import. It does not create any new benchmark, quality,
scalability, production-readiness, SOTA, or paper-faithfulness claim.

## Public-Hygiene Changes

Applied before the clean import:

- Added `LICENSE` with MIT terms for repository code and documentation.
- Added README license language that excludes external datasets, papers, and
  generated local artifacts from the repository license.
- Added package license metadata in `pyproject.toml`.
- Removed `clustering-based-factorized-cf.pdf` from the current tracked tree.
- Added `*.pdf` to `.gitignore`.

## Clean Import Meaning

The clean import replaces the public `main` branch with a new root commit whose
tree matches the current publishable repository state. This removes the old
commit graph from public branch history.

The old public tags must also be deleted, because tags pointing at old commits
would keep the old history publicly reachable.

This operation does not claim that GitHub has immediately garbage-collected all
unreferenced objects or external caches. It means no public repository ref
should continue to point at the old history.

## Verification

Commands run before the clean import:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy src
.venv\Scripts\python.exe -m recsys_lab.cli.main bootstrap-check
.venv\Scripts\python.exe -m recsys_lab.cli.main validate-runtime-profile configs\runtime\devices\local_i5_2500k_24gb.yaml --claim-eligible
```

Readout:

- Pytest: `131 passed`
- Ruff: `All checks passed!`
- Mypy source gate: `Success: no issues found in 62 source files`
- Bootstrap check: `healthy: true`
- Runtime profile: `status: valid`, `claim_eligible: true`

Public-tree scans:

- no tracked PDF, archive, tabular data, parquet, numpy, database, or jsonl
  payload files in the current index
- no high-signal secret-pattern hits in the current tracked tree
- no high-signal secret-pattern hits in the reachable pre-import Git history
- no local absolute workstation path hits in the current tracked tree

## Claim Boundary

Allowed claim:

- The public repository tree is prepared for a claim-limited code and
  documentation release after the clean-root import.

Explicit non-claims:

- no new model-quality claim
- no new runtime or scalability claim
- no paper-faithfulness claim
- no guarantee about third-party caches of old public objects
