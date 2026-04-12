# Model Spec: asvdpp

## Status

- mathematical status: implementation-ready under repo optimization contract
- claim status: poster-grounded predictor, not identical to Koren 2008 `Asymmetric-SVD`

## Canonical Name

- repo name: `asvdpp`

## Source Grounding

- Mirbakhsh and Ling 2013 Poster, `Asymmetric-SVD++` predictor family

## Predictor

```text
r_hat_ui =
mu + b_u + b_i
+ q_i^T (
    p_u
  + norm_N(u) * sum_{j in N(u)} y_j
  + norm_R(u) * sum_{j in R(u)} w_uj * x_j
)
```

## Parameter Blocks

- `b_u`, `b_i`
- `p_u`, `q_i`
- `x_j`, `y_j`

## Naming Boundary

- `asvdpp` in diesem Repo ist nicht einfach ein Alias fuer Koren 2008
  `asymmetric_svd`
- `asvdpp` meint die vom Poster verwendete Familie mit freiem `p_u` plus
  explizitem item-seitigen Feedback-Block

## Reduction Role

- reduziert sich zu `svdpp`, wenn alle `x_j = 0`
- reduziert sich nicht zu `asymmetric_svd`, weil `p_u` erhalten bleibt
