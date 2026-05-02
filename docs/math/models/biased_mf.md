# Model Spec: biased_mf

## Status

- mathematical status: implementation-ready
- claim status: paper-grounded baseline

## Canonical Name

- repo name: `biased_mf`

## Source Grounding

- Koren 2008 regularized factorization family

## Predictor

```text
r_hat_ui = mu + b_u + b_i + p_u^T q_i
```

## Parameter Blocks

- `b_u`, `b_i`
- `p_u`, `q_i`

## Notes

- Dieses Modell ist die kanonische faktorisierte Baseline des Repos.
- Es soll im Repo nicht pauschal `svd` genannt werden.

## Reduction Role

- bildet die Referenz fuer `svdpp`
- liefert in der Poster-Familie die initialen Latents fuer das spaetere
  Clustering
