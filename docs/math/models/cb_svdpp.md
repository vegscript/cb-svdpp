# Model Spec: cb_svdpp

## Status

- predictor status: source-grounded
- optimization status: repo-defined due poster under-specification
- claim status: not eligible for `exact paper reproduction` by default

## Canonical Name

- repo name: `cb_svdpp`

## Source Grounding

- Mirbakhsh and Ling 2013 Poster, clustering-based `SVD++`

## Predictor

```text
q_mix_i = (1 - alpha) * q_i + alpha * q_C[b(i)]
p_mix_u = (1 - alpha) * p_u + alpha * p_C[a(u)]
y_mix_j = (1 - alpha) * y_j + alpha * y_C[b(j)]

r_hat_ui =
mu + b_u + b_i
+ q_mix_i^T (
    p_mix_u
  + norm_N(u) * sum_{j in N(u)} y_mix_j
)
```

## Parameter Blocks

- individual level: `p_u`, `q_i`, `y_j`
- cluster level: `p_C[a]`, `q_C[b]`, `y_C[b]`
- assignments: `a(u)`, `b(i)`
- mix coefficient: `alpha`

## CB Contract

- Cluster-Assignments werden aus trainierten `biased_mf`-Latents und KMeans
  abgeleitet.
- `R_star` wird aus dem Trainingssatz berechnet.
- Die erste implementierbare Optimierung trainiert Cluster-Parameter gemeinsam
  mit den individuellen Parametern auf beobachteten Ratings.

## Claim Boundary

- Solange die Rolle von `R_star` in der Optimierung nicht final beschlossen ist,
  darf dieses Modell nur als paper-inspired implementation des publizierten
  Praediktors bezeichnet werden.
