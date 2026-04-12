# Model Spec: cb_asvdpp

## Status

- predictor status: source-grounded
- optimization status: repo-defined due poster under-specification
- claim status: not eligible for `exact paper reproduction` by default

## Canonical Name

- repo name: `cb_asvdpp`

## Source Grounding

- Mirbakhsh and Ling 2013 Poster, clustering-based `Asymmetric-SVD++`

## Predictor

```text
q_mix_i = (1 - alpha) * q_i + alpha * q_C[b(i)]
p_mix_u = (1 - alpha) * p_u + alpha * p_C[a(u)]
y_mix_j = (1 - alpha) * y_j + alpha * y_C[b(j)]
x_mix_j = (1 - alpha) * x_j + alpha * x_C[b(j)]

r_hat_ui =
mu + b_u + b_i
+ q_mix_i^T (
    p_mix_u
  + norm_N(u) * sum_{j in N(u)} y_mix_j
  + norm_R(u) * sum_{j in R(u)} w_uj * x_mix_j
)
```

## Parameter Blocks

- individual level: `p_u`, `q_i`, `x_j`, `y_j`
- cluster level: `p_C[a]`, `q_C[b]`, `x_C[b]`, `y_C[b]`
- assignments: `a(u)`, `b(i)`
- mix coefficient: `alpha`

## CB Contract

- Cluster-Assignments werden aus trainierten `biased_mf`-Latents und KMeans
  abgeleitet.
- `R_star` wird aus dem Trainingssatz berechnet.
- Die erste implementierbare Optimierung trainiert Cluster-Parameter gemeinsam
  mit den individuellen Parametern auf beobachteten Ratings.

## Reduction Role

- reduziert sich zu `asvdpp`, wenn `alpha = 0`

## Claim Boundary

- Ohne finale Entscheidung zur Rolle von `R_star` und zur CB-Optimierung darf
  dieses Modell nicht als exakte Paper-Reproduktion bezeichnet werden.
