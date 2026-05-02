# Model Spec: asymmetric_svd

## Status

- mathematical status: implementation-ready under detached-residual optimizer contract
- claim status: source-grounded predictor, optimizer-faithful claim currently restricted

## Canonical Name

- repo name: `asymmetric_svd`

## Source Grounding

- Koren 2008, asymmetric factor model

## Predictor

```text
r_hat_ui =
mu + b_u + b_i
+ q_i^T (
    norm_R(u) * sum_{j in R(u)} w_uj * x_j
  + norm_N(u) * sum_{j in N(u)} y_j
)
```

## Parameter Blocks

- `b_u`, `b_i`
- `q_i`
- `x_j`, `y_j`

## Critical Property

- kein freier User-Vektor `p_u`

## Notes

- Dieses Modell darf nicht mit `asvdpp` verwechselt werden.
- Der erste Repo-Optimizer behandelt `w_uj` im inneren Schritt als detachierte
  Groesse. Diese Entscheidung steht im Abweichungsregister.

## Reduction Role

- wird zu einer rein expliziten asymmetrischen Faktorisierung, wenn `N(u)` leer
  ist oder alle `y_j = 0`
