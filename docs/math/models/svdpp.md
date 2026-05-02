# Model Spec: svdpp

## Status

- mathematical status: implementation-ready
- claim status: paper-grounded with repo-defined explicit objective

## Canonical Name

- repo name: `svdpp`

## Source Grounding

- Koren 2008, equation family for `SVD++`

## Predictor

```text
r_hat_ui = mu + b_u + b_i + q_i^T (p_u + norm_N(u) * sum_{j in N(u)} y_j)
```

## Parameter Blocks

- `b_u`, `b_i`
- `p_u`, `q_i`
- `y_j`

## Notes

- Im paper-faithful Profil gilt standardmaessig `N(u) = R(u)`, wenn nur Ratings
  vorliegen.
- Die kanonische Zielfunktion und die Update-Regeln dieses Repos stehen in
  `objective_functions.md` und `update_rules.md`.

## Reduction Role

- reduziert sich auf `biased_mf`, wenn `N(u)` leer ist oder alle `y_j = 0`
