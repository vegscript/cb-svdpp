# 2026-05-02 Release Hygiene Evidence

- status: `pass`
- release_marker: `submission-2026-05-02-r10`
- scope: `claim-limited public clean-root release hygiene`

## Checks

- README states the claim-limited release status and links the canonical claim
  matrix.
- README links the current public-clean reproduction evidence note:
  `docs/evidence/reproduction/current/2026-05-02_public_clean_import.md`.
- README links the post-publication path-hygiene evidence note:
  `docs/evidence/reproduction/current/2026-05-03_public_path_hygiene.md`.
- The project report states the current release marker.
- `docs/publish_readiness_matrix.md` keeps the final claim matrix, feasibility
  evidence, and explicit non-claims as the source of truth.
- `tests/integration/test_large_dataset_claim_locks.py` keeps the
  large-dataset claim boundaries explicit and checks release-marker consistency
  across release-facing documents.
- `tests/integration/test_release_evidence_integrity.py` verifies
  release-facing evidence links and prevents generated data/artifact outputs
  from being accidentally versioned.
- The same guard blocks local absolute workstation path patterns from the
  tracked public tree.
- The same guard verifies that the README full-suite count stays synchronized
  with the reproduction evidence.
- The public clean import removes the old branch history from public `main`.
- Old public `submission-*` tags must be deleted so they cannot continue to
  expose the previous history.

## Claim Impact

Mark Gate 7 as `pass` for release marker `submission-2026-05-02-r10`.

This remains a claim-limited release marker. It is not an exact paper
reproduction marker, not a final `ml20m` model-comparison marker, not a general
large-dataset CB-SVD++ superiority marker, and not an unconstrained
publish-ready claim.
