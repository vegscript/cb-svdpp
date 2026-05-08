# CB Endpoint Alpha Policy V1 Evidence

## Branch

`cb-endpoint-alpha-policy-v1`

## Goal

Decide whether the Step 17b alpha0 CB-SVD++ specialized kernel should be
productized, and define the productive policy for CB endpoint alpha values.

This step is architecture and methodology evidence only. It does not implement
new kernels, model dispatch, or runner dispatch.

## Inputs Reviewed

Audit sources:

- `docs/performance/cb_kernel_specialization_audit_v1.md`
- `docs/evidence/performance/cb_kernel_specialization_audit_v1.md`

17b proof branch comparison:

```bash
git diff cb-kernel-specialization-audit-v1..exact-cb-kernel-specialization-v1 --stat
git diff cb-kernel-specialization-audit-v1..exact-cb-kernel-specialization-v1 -- src/recsys_lab/models/kernels.py
```

Observed 17b diff summary:

- 7 files changed
- 702 insertions
- 4 deletions
- `src/recsys_lab/models/kernels.py` gained
  `train_cb_svdpp_alpha0_epoch_numba`

Alpha validation/config scan:

```bash
rg alpha src configs tests
```

Relevant findings:

- `src/recsys_lab/models/config_schemas.py` currently validates clustering
  alpha with `ge=0.0, le=1.0`.
- `src/recsys_lab/models/cb_svdpp.py` and
  `src/recsys_lab/models/cb_asvdpp.py` currently validate alpha in the closed
  interval `[0, 1]`.
- Existing active and selected configs contain endpoint alpha values:
  - `configs/experiments/tuning/active/ml100k_cb_svdpp_g6_validation_grid.yaml`
  - `configs/experiments/tuning/active/ml20m_cb_svdpp_g11_lower_memory_validation_grid.yaml`
  - `configs/models/selected/ml100k/ml100k_cb_svdpp_g6_validation_selected.yaml`
- Existing tests document alpha-zero CB semantics as a CB-disabled ablation.

## 17b Summary

Step 17b was useful as a proof-of-concept and architectural test.

It showed:

- `train_cb_svdpp_alpha0_epoch_numba` is technically possible.
- The specialized kernel can be parameter-identical to the generic
  `train_cb_svdpp_epoch_numba(..., alpha=0.0, ...)`.
- Parameter identity requires preserving cluster-factor regularization even
  though cluster prediction contributions are zero at alpha0.
- The proof required additional hotpath code and diagnostic harness/test
  integration.
- No production model dispatch was added.

## Why Not Productize The Alpha0 Specialized Kernel

The 17b kernel is technically correct, but it is not a strong productive
scaling lever.

At `alpha == 0`, CB-SVD++ is prediction-level individual-only. The current CB
generic kernel is not parameter-state identical to SVD++ because inactive
cluster factors are still regularized. Preserving that internal state is what
makes the 17b specialization exact, but it also makes the specialized kernel
large and similar to the generic CB kernel.

Productizing this path would:

- add another hotpath kernel to `kernels.py`
- increase maintenance burden
- make endpoint semantics more prominent than they should be
- risk hidden model-variant confusion
- not address the main 100M+ scale pressure in true CB mixture runs

Decision:

```text
ACCEPT_17B_AS_EVIDENCE_ONLY
DO_NOT_PRODUCTIZE_ALPHA0_CB_SPECIALIZED_KERNEL
```

## Endpoint-Alpha Policy

Productive CB profiles should use:

```text
0 < alpha < 1
```

Endpoint policy:

- `alpha == 0`
  - Degeneration case.
  - Prediction-level individual-only behavior.
  - Use SVD++ or ASVD++ as the productive baseline.
  - CB alpha0 runs may remain documented as CB-disabled ablations.
  - Do not add productive CB alpha0 endpoint kernels.
- `alpha == 1`
  - Degeneration case.
  - Prediction-level cluster-only behavior.
  - Treat as a research-variant candidate, not as normal productive CB-SVD++
    or CB-ASVD++ behavior.
- `0 < alpha < 1`
  - True CB mixture.
  - Keep the generic CB hotpath canonical.

Precision note:

- `alpha == 0` is prediction-level degenerate to the individual path.
- It is not parameter-state identical to SVD++ or ASVD++ under the current
  generic CB kernel because inactive cluster factor families are still
  regularized.

## Impact On Configs/Tuning

Current validation allows `0 <= alpha <= 1`.

Directly hardening all validation to `0 < alpha < 1` in this step is not
risk-free because existing active, selected, archive, and test configs include
alpha0 candidates. Changing the schema immediately would break current config
inventory tests and historical evidence readability.

Policy accepted here:

- Productive CB model profiles should move to strict interior alpha.
- Existing endpoint alpha configs remain readable for now.
- Enforcement is deferred to a dedicated migration step that can separate:
  - productive CB profiles
  - historical evidence configs
  - CB-disabled ablations
  - future research variants

Follow-up migration:

```text
Enforce productive CB alpha interior policy after config inventory migration.
```

Minimum migration expectations:

- remove endpoint alpha from productive selected CB profiles
- keep old endpoint configs readable as archive/evidence
- add validation/tests for productive profile paths only
- update tuning docs to classify endpoint alpha as ablation or research variant

## What Code Was Not Merged

This branch starts from `cb-kernel-specialization-audit-v1`.

It does not include the 17b hotpath proof code:

- no `train_cb_svdpp_alpha0_epoch_numba`
- no `cb_svdpp_alpha0` benchmark harness dispatch
- no endpoint-alpha synthetic benchmark case
- no 17b equivalence test file
- no production model dispatch

## Tests/Gates

Commands:

```bash
ruff check .
pytest tests/unit/test_hotpath_coldpath_boundaries.py
pytest tests/unit
pytest tests/integration/test_unified_pipeline_smoke_all_models.py
pytest
rg <claim-check-pattern> docs src tests
```

Results:

- `ruff check .`: passed
- `pytest tests/unit/test_hotpath_coldpath_boundaries.py`: 13 passed
- `pytest tests/unit`: 211 passed
- `pytest tests/integration/test_unified_pipeline_smoke_all_models.py`: 1 passed
- `pytest`: 285 passed, 2 skipped
- claim-pattern check: completed; matches are pre-existing governance,
  claim-boundary, and test strings, with no matches in the new 17c ADR or
  17c evidence files

## Claim Boundary

No runtime improvement is claimed.

This evidence records an architecture decision:

- 17b is accepted as evidence only
- alpha0 CB endpoint specialization is not productized
- productive CB alpha policy is documented

It does not make a device-general, dataset-general, or runtime improvement
claim.

## Next Step

Recommended next red-thread step:

`18. Cluster Artifact / KMeans Optimization`

Before any future CB endpoint enforcement, run a separate config migration task
for productive-vs-archive alpha validation.
