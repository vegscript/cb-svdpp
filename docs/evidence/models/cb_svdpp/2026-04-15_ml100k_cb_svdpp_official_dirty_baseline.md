# Evidence Note

- date: `2026-04-15`
- scope: `models/cb_svdpp`
- topic: `first official ml100k cb_svdpp benchmark`
- status: `accepted_with_limitations`

## Context

After the official `ml100k` benchmark dispatcher and fit-time contract were
extended to `cb_svdpp`, the next required step was a first end-to-end official
five-fold readout on `paper_faithful_ml100k_v1`.

The benchmark was executed from the current repo worktree, which was dirty at
run time because the `cb_svdpp` official benchmark wiring was still local and
uncommitted.

## Command

```text
python -m recsys_lab.cli.main benchmark-ml100k-paper cb_svdpp data/processed/ml100k/ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json configs/models/cb_svdpp.yaml --runtime-config configs/runtime/base.yaml --device-config configs/runtime/devices/local_i5_2500k_24gb.yaml --model-seed 1
```

## Artifact

- benchmark:
  `artifacts/benchmarks/2026-04-15T043209Z_ml100k_paper_faithful_cb_svdpp_local_i5_2500k_24gb`

## Result

- status: `completed`
- git commit: `fb1fcbc53796fbe27e515a526096bba03ffbb41f`
- git dirty: `true`
- folds: `u1` to `u5`
- model seed: `1`
- config status: `draft`
- mean test RMSE: `0.925901`
- test RMSE std: `0.006211`
- mean fit time: `431.17` seconds

Per-fold test RMSE:

- `u1`: `0.935645`
- `u2`: `0.928208`
- `u3`: `0.920158`
- `u4`: `0.923512`
- `u5`: `0.921983`

## Interpretation

- This is the first official `cb_svdpp` readout in the repo on the canonical
  `ml100k` split family.
- Under the new benchmark contract, the reported fit time already includes both
  cluster induction and main training.
- The result is materially better than the clean tuned `biased_mf` anchor
  (`0.937111`), but still slightly worse than the clean tuned `svdpp` anchor
  (`0.924015`).
- The compute profile is between those anchors: about `1.56x` the clean tuned
  `biased_mf` fit time and about `0.31x` the clean tuned `svdpp` fit time.

## Limitations

- This benchmark is not clean-final because `git.dirty=true`.
- The current `cb_svdpp` config is still a draft profile, not a tuned official
  candidate.
- The comparison against clean tuned anchors is informative, but not yet a
  final apples-to-apples model ranking.

## Decision

- Keep this benchmark as the official provisional `cb_svdpp` baseline on
  `ml100k`.
- Do not promote it to a clean anchor.
- The next methodologically correct step is controlled tuning of `alpha`,
  cluster counts, and possibly induction/training regularization, followed by a
  clean benchmark rerun.
