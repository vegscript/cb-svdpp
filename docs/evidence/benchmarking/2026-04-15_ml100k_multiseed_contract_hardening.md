# Evidence Note

- date: `2026-04-15`
- scope: benchmark governance and multi-seed aggregation
- git_commit_context: `fb1fcbc53796fbe27e515a526096bba03ffbb41f`

## Trigger

The repository had reached a point where historical seed-level benchmark
artifacts could coexist for the same model family and seed. Heuristic
multi-seed selection by config reference plus seed was therefore no longer
strict enough for high-confidence benchmark claims.

## Change

The multi-seed benchmark path was hardened in three ways:

- explicit `benchmark_manifest_paths` can now be passed through the CLI and
  runner
- duplicate heuristic matches now fail with an instruction to use explicit
  benchmark manifests
- selected seed benchmarks must share identical Git commit, branch, and dirty
  state before aggregation is allowed

## Verification

- targeted compile checks passed for the runner and CLI
- `tests/integration/test_ml100k_paper_multiseed_benchmark.py`: `3 passed`
- benchmark manifests for the clean `biased_mf` and clean `svdpp` multi-seed
  anchors validate against the canonical schema

## Interpretation

- Multi-seed aggregation is now explicit enough to survive the presence of
  historical seed artifacts.
- The repository benchmark contract is materially safer for long-running
  studies on the same dataset and config family.
