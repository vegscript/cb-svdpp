# Model Spec: cb_svdpp

## Status

- predictor status: source-grounded
- optimization status: repo-defined v1 contract
- claim status: not eligible for `exact paper reproduction`

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

- Cluster-Assignments werden aus train-only `biased_mf`-Latents und separatem
  `KMeans` fuer User und Items abgeleitet.
- Cluster-Assignments bleiben waehrend des CB-Trainings fix.
- `R_star` wird nur aus dem Trainingssatz als Cluster-Mean-Matrix plus
  Cluster-Count-Matrix berechnet.
- `R_star` ist in v1 ein Diagnoseartefakt und kein eigener Loss-Term.
- Die Optimierung trainiert Cluster-Parameter gemeinsam mit den individuellen
  Parametern direkt auf beobachteten Ratings.
- `alpha` ist in v1 ein fester Hyperparameter pro Run.

## Reduction Role

- reduziert sich zu `svdpp`, wenn `alpha = 0`

## Claim Boundary

- Dieses Modell darf als source-grounded predictor mit repo-defined
  optimization bezeichnet werden.
- Dieses Modell darf nicht als exakte Paper-Reproduktion bezeichnet werden.
