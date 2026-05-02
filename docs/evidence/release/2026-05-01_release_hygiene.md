# 2026-05-01 Release Hygiene Evidence

- status: `pass`
- release_marker: `submission-2026-05-01-r9`
- scope: `claim-limited release-candidate hygiene after ml10m cb_svdpp matched benchmark completion`

## Checks

- Evidence links in README, report, and the publish-readiness matrix resolve to
  versioned documentation notes.
- README states the claim-limited release status and links the canonical claim
  matrix.
- README links the current reproduction evidence note:
  `docs/evidence/reproduction/2026-05-01_quality_gate_reproduction.md`.
- The project report states the current release marker.
- `docs/publish_readiness_matrix.md` keeps the final claim matrix, feasibility
  evidence, and explicit non-claims as the source of truth.
- The new `ml10m cb_svdpp` benchmark row is matched to the documented
  `ml10m biased_mf` baseline anchor and explicitly limits claims to that
  profile comparison.
- The `ml20m biased_mf` benchmark row remains baseline-only and explicitly does
  not unlock final `ml20m` model-comparison claims.
- The large-dataset `cb_svdpp` matched-campaign contract is linked from the
  matrix/report and updated to state that `ml10m` has satisfied the contract
  while `ml20m` remains blocked without an explicit budget/device decision.
- `tests/integration/test_large_dataset_claim_locks.py` keeps the large-dataset
  claim boundaries explicit and checks release-marker consistency across
  release-facing documents.
- `tests/integration/test_release_evidence_integrity.py` verifies release-facing
  Evidence links and prevents generated data/artifact outputs from being
  accidentally versioned.
- The same guard verifies that the README full-suite count stays synchronized
  with the reproduction evidence.

## Claim Impact

Mark Gate 7 as `pass` for release marker `submission-2026-05-01-r9`.

This remains a claim-limited release marker. It is not an exact paper
reproduction marker, not a final `ml20m` model-comparison marker, not a general
large-dataset CB-SVD++ superiority marker, and not an unconstrained
publish-ready claim.
