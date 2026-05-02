# Evidence Note

- date: `2026-04-24`
- scope: `release hygiene`
- status: `pass`
- release_marker: `submission-2026-04-24`

## Claim Or Question

Does the repository satisfy Gate 7 release-hygiene requirements for a
claim-limited submission handoff?

## Inputs And Artifacts

- release-facing README:
  `README.md`
- publish-readiness matrix:
  `docs/publish_readiness_matrix.md`
- project report:
  `docs/report/project_report.md`
- reproduction evidence:
  `docs/evidence/reproduction/2026-04-24_current_main_reproduction_smoke.md`
- git ignore policy:
  `.gitignore`

## Method

Check the Gate 7 requirements from `docs/project_master_plan.md` and update the
entry-point documentation without changing scientific claim boundaries.

Gate 7 requires:

- README describes scope, setup, main results, and limits
- artifact and data policy is understandable for external readers
- large or rebuildable outputs are not accidentally versioned
- known non-claims are visible
- the final commit is tagged or clearly marked as the submission commit

## Readout

- README now states the claim-limited release status.
- README states the canonical setup path:
  `uv sync --extra dev --locked`
- README states the current clean benchmark scopes:
  `ml100k` clean model ladder, `ml1m` matched `biased_mf` vs `cb_svdpp`,
  and `ml10m` / `ml20m` feasibility only.
- README lists non-claims, including no exact paper reproduction, no final
  `ml10m` / `ml20m` model-comparison claim, and no unqualified scalability
  claim.
- README summarizes ignored data and artifact zones.
- `.gitignore` keeps `.venv`, raw data, processed data, run artifacts,
  benchmark artifacts, figures, debug outputs, and local caches out of git
  except for placeholders.
- The report conclusion states the same claim-limited release boundary.
- The release marker is `submission-2026-04-24`.

## Interpretation

Gate 7 can be marked `pass` for a claim-limited release candidate. This does
not expand any benchmark claim. The scientific boundary remains governed by
`docs/publish_readiness_matrix.md`.

## Decision Or Next Step

- Mark Gate 7 as `pass`.
- Keep overall status as `release_candidate_claim_limited`.
- Use the git tag `submission-2026-04-24` as the release marker for this
  claim-limited handoff.
