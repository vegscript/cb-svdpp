# 2026-04-30 Release Hygiene Evidence

- status: `pass`
- release_marker: `submission-2026-04-30-r8`
- scope: `claim-limited release-candidate hygiene after ml10m cb_svdpp matched-profile preparation`

## Checks

- README states the claim-limited release status and links the canonical claim
  matrix.
- README links the current reproduction evidence note:
  `docs/evidence/reproduction/2026-04-30_quality_gate_reproduction.md`.
- The project report states the current release marker.
- `docs/publish_readiness_matrix.md` keeps the final claim matrix, feasibility
  evidence, and explicit non-claims as the source of truth.
- The new `ml10m biased_mf` and `ml20m biased_mf` benchmark rows are
  baseline-only and explicitly do not unlock final large-dataset
  model-comparison claims.
- The post-r3 large-dataset `cb_svdpp` matched-campaign contract is linked from
  the matrix and report, and keeps matched CB claims blocked until the campaign
  contract is satisfied.
- `tests/integration/test_large_dataset_claim_locks.py` keeps large-dataset
  `cb_svdpp` rows out of final anchor sections and checks release-marker
  consistency across release-facing documents.
- `tests/integration/test_release_evidence_integrity.py` verifies release-facing
  Evidence links and prevents generated data/artifact outputs from being
  accidentally versioned.
- The same guard verifies that the README full-suite count stays synchronized
  with the reproduction evidence.
- `configs/models/tuned/ml10m_cb_svdpp_stage0_transfer.yaml` is committed as
  the 20-epoch profile for a future `ml10m cb_svdpp` matched campaign; this
  profile alone does not unlock a final large-dataset model-comparison claim.

## Claim Impact

Mark Gate 7 as `pass` for release marker `submission-2026-04-30-r8`.

This remains a claim-limited release marker. It is not an exact paper
reproduction marker, not a final `ml10m` or `ml20m` model-comparison marker, and
not an unconstrained publish-ready claim.
