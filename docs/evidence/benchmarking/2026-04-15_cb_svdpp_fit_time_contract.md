# Evidence Note

- date: `2026-04-15`
- scope: `benchmarking`
- topic: `cb_svdpp fit-time comparability`
- status: `accepted`

## Context

The canonical `ml100k` benchmark aggregator previously summarized
`training_wall_clock_seconds` directly from per-run metrics.

This is fair for `biased_mf` and `svdpp`, where the model fit path is a single
training phase. It is not fair for `cb_svdpp`, where train-only cluster
induction is required before the main optimization phase.

## Decision

- For benchmark summaries, comparable fit time now includes all mandatory
  train-only stages needed to fit the model.
- For `cb_svdpp`, benchmark fit time is:
  `cluster_induction_wall_clock_seconds + training_wall_clock_seconds`
- For non-clustering models without a separate induction phase, benchmark fit
  time remains `training_wall_clock_seconds`.

## Why

- Excluding cluster induction would understate the true fit cost of
  clustering-based models.
- A benchmark table that compares raw MF training time against `cb_svdpp`
  training time without cluster induction is methodologically unsound.
- The benchmark layer must preserve apples-to-apples cost claims before any
  official `cb_svdpp` result is reported.

## Verification

- code path:
  - `src/recsys_lab/experiments/ml100k_paper_benchmark.py`
- targeted test:
  - `tests/integration/test_ml100k_paper_benchmark.py::test_run_ml100k_paper_benchmark_supports_cb_svdpp_and_counts_cluster_time`

## Impact

- Existing `biased_mf` and `svdpp` benchmark summaries are unaffected.
- Future `cb_svdpp` benchmark summaries will report a comparable fit-time value
  instead of silently dropping cluster induction cost.
