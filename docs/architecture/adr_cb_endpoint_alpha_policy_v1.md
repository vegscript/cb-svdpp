# ADR: CB Endpoint Alpha Policy V1

## Status

Accepted

## Context

Step 17a audited CB kernel specialization opportunities for:

- `train_cb_svdpp_epoch_numba`
- `train_cb_asvdpp_epoch_numba`

The audit found that endpoint alpha values are methodologically delicate.
At `alpha == 0`, CB-SVD++ and CB-ASVD++ predictions reduce to the individual
factor path. At `alpha == 1`, predictions reduce to the cluster factor path.
However, the current generic CB kernels still regularize the inactive factor
families. Therefore endpoint alpha is not parameter-state identical to simply
using the corresponding non-CB baseline or a cluster-only variant.

Step 17b implemented a proof-of-concept
`train_cb_svdpp_alpha0_epoch_numba` on a separate branch. It showed that a
parameter-identical alpha0 specialization is technically possible if the
specialized kernel keeps the generic update semantics, including regularization
of zero-gradient cluster factors.

The proof also showed the architectural cost: preserving parameter identity
keeps much of the generic CB structure in the specialized kernel. Productizing
that path would add another hotpath kernel without a clear modeling or
scaling benefit.

## Decision

CB-SVD++ and CB-ASVD++ are productive CB models only for true mixture settings:

```text
0 < alpha < 1
```

Endpoint policy:

- `alpha == 0`
  - Degeneration case.
  - Prediction-level behavior is individual-only.
  - It should be treated methodologically as an SVD++ or ASVD++ baseline,
    not as a productive CB endpoint kernel.
  - It is not parameter-state identical to SVD++ or ASVD++ under the current
    generic CB kernel, because inactive cluster factor families are still
    regularized.
- `alpha == 1`
  - Degeneration case.
  - Prediction-level behavior is cluster-only.
  - It is a research-variant candidate, not a normal productive CB-SVD++ or
    CB-ASVD++ special case.
- `0 < alpha < 1`
  - True CB mixture.
  - The generic CB hotpath remains the canonical implementation.

Decision:

```text
ACCEPT_17B_AS_EVIDENCE_ONLY
DO_NOT_PRODUCTIZE_ALPHA0_CB_SPECIALIZED_KERNEL
```

## Consequences

- The 17b alpha0 specialization is not merged into the productive line.
- `kernels.py` should not grow endpoint-alpha CB kernels unless a later ADR
  reverses this policy with stronger evidence.
- Productive CB profiles should use strictly interior alpha values.
- If an experiment needs `alpha == 0`, it should use the non-CB baseline for
  productive comparison, or document the run as a CB-disabled ablation.
- If an experiment needs `alpha == 1`, it should be introduced as an explicit
  research variant with separate model semantics and evidence.
- Existing historical or active evidence configs that contain endpoint alpha
  values remain readable until a dedicated config migration is performed.

## Rejected Alternatives

### Productize `train_cb_svdpp_alpha0_epoch_numba`

Rejected.

The proof-of-concept was parameter-identical, but only by preserving cluster
regularization and much of the generic CB kernel structure. This adds hotpath
surface area without a clear scaling lever.

### Dispatch `alpha == 0` to SVD++

Rejected as an implementation shortcut.

Prediction-level degeneracy exists, but parameter-state identity does not hold
for the current generic CB kernel because inactive cluster factors are still
regularized. Dispatching silently to SVD++ would hide a semantic change.

### Freeze inactive factor families at endpoints

Rejected for productive CB models.

Freezing inactive factors changes training semantics. It may be a research
variant, but it is not an exact specialization of the current CB kernels.

### Add endpoint kernels for `alpha == 1`

Rejected for this productive policy.

Cluster-only endpoint behavior is a separate modeling question and should not
be introduced as a normal CB hotpath special case.

## Implementation Guidance

- Do not add endpoint-alpha CB kernels to `src/recsys_lab/models/kernels.py`.
- Do not add model-wrapper dispatch for CB endpoint alpha values.
- Keep the generic CB kernels canonical for `0 < alpha < 1`.
- Treat `alpha == 0` CB runs as CB-disabled ablations unless and until config
  migration removes them from productive CB profiles.
- Treat `alpha == 1` as a research-variant candidate requiring separate
  methodology documentation.
- A follow-up migration may harden productive CB profile validation to require
  `0 < alpha < 1`, but it must first account for existing active and archived
  configs that currently contain endpoint alpha values.
