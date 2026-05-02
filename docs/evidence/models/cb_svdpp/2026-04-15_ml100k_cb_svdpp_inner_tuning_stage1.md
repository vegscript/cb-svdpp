# Evidence Note

## Scope

Leakagesafe inner tuning study for `cb_svdpp` on `MovieLens 100K` using the
canonical `paper_faithful_ml100k_inner_v1` protocol.

## Claim Or Question

Can a small, controlled `stage1` search improve the current draft `cb_svdpp`
profile on inner validation splits without touching the official outer test
folds?

## Inputs And Artifacts

- tuning config:
  `configs/experiments/tuning/ml100k_cb_svdpp_stage1.yaml`
- base model config:
  `configs/models/cb_svdpp.yaml`
- tuning benchmark directory:
  `artifacts/benchmarks/2026-04-15T074345Z_ml100k_inner_tuning_cb_svdpp_stage1_local_i5_2500k_24gb/`

## Method

- use only outer folds `u1` and `u2`
- derive inner validation splits from the corresponding `uX.base` partitions
- evaluate four candidates:
  - draft baseline
  - `alpha=0.05`
  - `alpha=0.15`
  - a moderated profile with `latent_dim=64`, `80/80` clusters,
    `learning_rate=0.0075`, and `reg=0.025`
- rank candidates by mean validation RMSE, then by validation RMSE standard
  deviation, then by mean fit time
- fit time includes cluster induction plus main training

## Readout

- status: `completed`
- winner:
  `rank064_uc080_ic080_a010_lr0075_reg0025_e020`
- winner validation RMSE mean: `0.915080`
- winner validation RMSE std: `0.004340`
- winner fit-time mean: `449.87` seconds
- draft baseline validation RMSE mean: `0.926495`
- absolute validation gain vs. draft baseline: `0.011416`

## Interpretation

The first controlled `cb_svdpp` tuning study succeeds. The winner is not one of
the alpha-only variants; the best result comes from a jointly moderated
profile: slightly larger factor rank, slightly fewer clusters, lower learning
rate, and stronger regularization.

This matters methodologically because it shows that `cb_svdpp` still had clear
hyperparameter debt after the first official draft benchmark. The tuned winner
is therefore a defensible candidate for the next outer-fold benchmark step.

## Decision Or Next Step

- promote the winner into a versioned config:
  `configs/models/tuned/ml100k_cb_svdpp_stage1.yaml`
- benchmark that promoted profile on the official `u1` to `u5` outer folds
- keep the result provisional until a clean workspace benchmark exists
