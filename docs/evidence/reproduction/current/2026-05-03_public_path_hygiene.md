# Public Path Hygiene Evidence

- date: `2026-05-03`
- status: `pass_for_public_path_hygiene`
- scope: `public repository path hygiene`
- release marker line: `submission-2026-05-02-r10`

## Purpose

This note records a post-publication hygiene correction for release-facing and
governance documentation. It does not create any new benchmark, quality,
scalability, production-readiness, SOTA, or paper-faithfulness claim.

## Issue

After the public clean-root import, the tracked tree still contained local
absolute workstation path references in Markdown links and in one reproduction
evidence command block. These paths were not secrets, but they were not
appropriate for a public repository and could not be used by an independent
reader.

## Changes

Implemented:

- replaced local absolute Markdown link targets with repo-relative paths
- replaced local clean-worktree paths in G5 evidence with public-safe
  placeholders
- removed a README smoke command that referenced an ignored local run artifact
- added a release-integrity regression test that blocks local absolute
  workstation path patterns in the tracked tree
- updated the public clean-import note to include this path-hygiene scan

## Verification

Public path scan:

```powershell
.venv\Scripts\python.exe -m pytest tests\integration\test_release_evidence_integrity.py::test_public_tree_does_not_contain_local_absolute_paths
```

Readout:

- passed

Focused release-facing tests:

```powershell
.venv\Scripts\python.exe -m ruff check tests\integration\test_release_evidence_integrity.py
.venv\Scripts\python.exe -m pytest tests\integration\test_release_evidence_integrity.py tests\integration\test_large_dataset_claim_locks.py
```

Readout:

- Ruff: `All checks passed!`
- Pytest: `13 passed`

Full local gates:

```powershell
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy src
.venv\Scripts\python.exe -m pytest
```

Readout:

- Ruff: `All checks passed!`
- Mypy: `Success: no issues found in 62 source files`
- Pytest: `133 passed`

## Claim Boundary

Allowed claim:

- The tracked public tree no longer contains the checked local absolute path
  patterns.
- Release-facing tests include a regression guard for those patterns.

Explicit non-claims:

- no new model-quality claim
- no new runtime or scalability claim
- no paper-faithfulness claim
- no guarantee about third-party caches of older public objects
