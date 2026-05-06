# Exact Kernel Optimization V1 Acceptance

This document records the acceptance decision for Exact Kernel Optimization V1.
It is an acceptance and merge-readiness note, not a new optimization task and
not a broad performance claim.

## Decision Summary

| Kernel | Change | Numerical stability | Performance readout | Decision | Reason |
| --- | --- | --- | --- | --- | --- |
| `train_biased_mf_epoch_numba` | Reuse old user/item factor workspaces outside the per-rating loop. | `test_rmse` and `test_mae` unchanged; estimated work unchanged. | ML1M `fit_model`: 16.178s to 11.911s. | ACCEPT | Evidence is complete, numerically stable, and the diagnostic fit readout is lower. |
| `train_svdpp_epoch_numba` | Reuse old user/item factor and implicit-context workspaces outside the per-rating loop. | `test_rmse` and `test_mae` unchanged; estimated work unchanged. | ML1M `fit_model`: 426.552s to 401.816s. | ACCEPT | Evidence is complete, numerically stable, and the diagnostic fit readout is lower. |
| `train_asymmetric_svd_epoch_numba` | Workspace reuse was attempted for old item factor and context buffers. | `test_rmse` and `test_mae` unchanged; estimated work unchanged. | ML1M `fit_model`: 1017.104s to 1107.096s. | REVERT | The slowdown is plausibly caused by the isolated workspace-lifetime change, with no counter-evidence proving run-to-run noise. |
| `train_asvdpp_epoch_numba` | Reuse old user/item factor and context workspaces outside the per-rating loop. | `test_rmse` and `test_mae` unchanged; estimated work unchanged. | ML1M `fit_model`: 1019.779s to 970.167s. | ACCEPT | Evidence is complete, numerically stable, and the diagnostic fit readout is lower. |
| `train_cb_svdpp_epoch_numba` | No V1 kernel change. | Not applicable to V1 change. | CB delta is treated as run-to-run noise because the kernel was not changed. | DEFER / NOT APPLICABLE | No CB-kernel optimization decision follows from this evidence. |
| `train_cb_asvdpp_epoch_numba` | No V1 kernel change. | Not applicable to V1 change. | CB delta is treated as run-to-run noise because the kernel was not changed. | DEFER / NOT APPLICABLE | No CB-kernel optimization decision follows from this evidence. |

## Evidence Sources

- `docs/kernel_optimization_plan.md`
- `docs/evidence/performance/exact_kernel_optimization_v1.md`
- `src/recsys_lab/models/kernels.py`
- `artifacts/reports/performance_hotspots.csv`
- `artifacts/reports/performance_stage_breakdown.csv`
- `artifacts/reports/kernel_cost_anatomy.csv`
- `artifacts/runs/<run_id>/metrics.json`
- `artifacts/runs/<run_id>/performance_profile.json`
- `artifacts/runs/<run_id>/kernel_profile.json`
- `artifacts/runs/<run_id>/run_manifest.json`

## Accepted Changes

Accepted kernel slices:

- `train_biased_mf_epoch_numba`
- `train_svdpp_epoch_numba`
- `train_asvdpp_epoch_numba`

Acceptance basis:

- The implemented changes alter temporary workspace lifetime only.
- Rating order, history order, update order, factor read/write timing, formulas,
  hyperparameters, dtype, split policy, and tuning policy were not changed.
- ML1M diagnostic before/after pairs used the same documented run contract.
- `test_rmse` and `test_mae` were unchanged at the documented precision.
- Estimated factor touches were unchanged.
- The diagnostic `fit_model` readout was lower for these three changed kernels.

Acceptance boundary: these slices are acceptable as exact low-risk workspace
reuse changes for this branch. This does not establish a cross-device,
cross-dataset, or broad speedup claim.

## Reverted Changes

Reverted kernel slice:

- `train_asymmetric_svd_epoch_numba`

Revert basis:

- The workspace reuse change for this kernel was isolated to old-item-vector and
  context workspace lifetime.
- Numerical metrics remained stable.
- Estimated factor touches were unchanged.
- The ML1M diagnostic `fit_model` readout moved from 1017.104s to 1107.096s.
- Ratings/s moved from 15734.733 to 14455.709.
- Fit seconds per million estimated touches moved from 0.001014893726 to
  0.001104691776.
- The workspace-lifetime change can plausibly affect Numba code generation or
  memory/cache behavior.
- No existing evidence shows that the slowdown is non-reproducible or purely
  run-to-run noise.

Decision: revert only this kernel slice to the previous per-rating allocation
behavior before merge. A different exact implementation can be investigated
later in a separate follow-up.

## Deferred / Not Applicable

Deferred / not applicable kernels:

- `train_cb_svdpp_epoch_numba`
- `train_cb_asvdpp_epoch_numba`

Reasoning:

- No implementation change was made to either CB kernel.
- Their main workspace arrays were already outside the per-rating loop before
  Exact Kernel Optimization V1.
- The observed CB fit deltas cannot be attributed to a CB-kernel change in this
  task.
- No CB kernel improvement, speedup, or merge acceptance claim follows from this
  evidence.

## Investigate Before Broader Claim

Required follow-up before any broader optimization claim:

- Re-check the accepted workspace reuse slices under the full benchmark contract
  from `docs/kernel_optimization_plan.md`.
- Investigate `asymmetric_svd` separately if another exact allocation cleanup is
  still desired.
- In that follow-up, test whether Numba code generation, explicit context reset,
  or cache behavior caused the diagnostic slowdown.
- Keep CB kernel work deferred until a separate exact candidate is defined and
  measured.

## What Changed

The implemented optimization moved reusable temporary workspace arrays outside
the per-rating loop for the non-CB Numba training kernels:

- `train_biased_mf_epoch_numba`
- `train_svdpp_epoch_numba`
- `train_asymmetric_svd_epoch_numba`
- `train_asvdpp_epoch_numba`

After the acceptance decision, the `train_asymmetric_svd_epoch_numba` slice was
selectively reverted. The accepted code scope is therefore limited to:

- `train_biased_mf_epoch_numba`
- `train_svdpp_epoch_numba`
- `train_asvdpp_epoch_numba`

## What Did Not Change

- No CB kernel implementation was changed.
- No model formula was changed.
- No residual-weight caching was introduced.
- No dynamic user-context caching was introduced.
- No cluster-history caching was introduced.
- No parallel SGD was introduced.
- No rating or history traversal order was changed.
- No hyperparameter, split, dtype, config, cache-policy, or tuning change was
  introduced.
- No historical result artifact was rewritten.

## Optional ML100K Recheck

A small optional ML100K recheck was run after the selective
`train_asymmetric_svd_epoch_numba` revert. This is acceptance hygiene only. It is
not a benchmark claim and is not used to replace the ML1M before/after evidence.

Recheck contract:

- dataset: `ml100k`
- processed manifest:
  `data/processed/ml100k/ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json`
- split family: `benchmark_random_v1`
- split seed: `1`
- model seed: `1`
- train ratio: `0.8`
- validation ratio: `0.1`
- device profile: `local_i5_2500k_24gb`
- split cache: disabled
- training index cache: disabled
- cluster artifact cache: disabled

Recheck readout:

| Model | Recheck run id | Status | RMSE | MAE | fit_model_seconds | Train rows | Estimated factor touches | Artifacts present |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `biased_mf` | `2026-05-06T233938Z_ml100k_biased_mf_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | completed | 0.945057639524 | 0.735141271765 | 3.304 | 80030 | 320120000 | true |
| `svdpp` | `2026-05-06T233958Z_ml100k_svdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | completed | 0.934431113079 | 0.726448078733 | 22.620 | 80030 | 26249108000 | true |
| `asymmetric_svd` | `2026-05-06T234038Z_ml100k_asymmetric_svd_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | completed | 0.915838548350 | 0.720304623327 | 49.367 | 80030 | 52178096000 | true |
| `asvdpp` | `2026-05-06T234144Z_ml100k_asvdpp_local_i5_2500k_24gb_benchmark_random_v1_tr080_va010_s001_s001` | completed | 0.934610012878 | 0.726535759629 | 48.802 | 80030 | 52178096000 | true |

For each recheck run, these artifacts were present:

- `metrics.json`
- `run_manifest.json`
- `config_snapshot.yaml`
- `performance_profile.json`
- `kernel_profile.json`

This recheck confirms that the four non-CB ML100K runs complete and produce the
required profiling artifacts on the acceptance branch after the selective
`asymmetric_svd` revert. It does not establish speedup, quality, scalability, or
cross-dataset claims.

## Tests And Gates

Acceptance-branch gates:

- `ruff check .`: passed
- `pytest tests/unit`: 133 passed
- `pytest tests/integration/test_unified_pipeline_smoke_all_models.py`: 1 passed
- `pytest`: 207 passed

Claim-string check:

- Requested `rg "broad performance claim|production-ready|guaranteed speedup|SOTA speedup" docs/evidence/performance`.
- Local `rg.exe` failed with access denied.
- Fallback was run with PowerShell `Select-String`.
- Hits were limited to explicit non-claim/caveat wording such as
  `not a broad performance claim` and `not ... production-ready`.
- No new overbroad speedup, production-readiness, or SOTA-speedup claim was
  found.

## Merge Recommendation

Recommended merge scope:

- Keep accepted exact workspace reuse for:
  - `train_biased_mf_epoch_numba`
  - `train_svdpp_epoch_numba`
  - `train_asvdpp_epoch_numba`
- Keep the selective revert for:
  - `train_asymmetric_svd_epoch_numba`
- Keep CB kernels unchanged.
- Do not include any CB-kernel optimization claim in this merge.
- Do not claim broad speedup from this branch.

Merge-readiness condition: the branch is merge-ready for the limited acceptance
scope above after the selective `asymmetric_svd` revert and the completed test
gates.

## Follow-Up Tasks

1. Keep the `train_asymmetric_svd_epoch_numba` workspace reuse slice reverted in
   this acceptance branch.
2. Open a separate follow-up investigation for `asymmetric_svd` if an exact
   allocation cleanup is still desired.
3. Run the full benchmark contract before making any broad performance claim:
   `ml100k` all six models and `ml1m` all six models.
4. Keep CB kernel work deferred until a separate exact candidate is defined and
   measured.
5. Preserve `performance_profile.json` and `kernel_profile.json` as mandatory
   evidence for all follow-up comparisons.

## Claim Boundary

This acceptance decision does not establish a broad performance claim. It only
records which exact workspace-reuse changes are safe enough to keep or require
further investigation under the current evidence.

It does not claim that Exact Kernel Optimization V1 is broadly faster, more
scalable, production-ready, or valid across devices and datasets. It only
accepts selected exact workspace reuse slices for the current branch, reverts
the `train_asymmetric_svd_epoch_numba` slice due to the observed diagnostic
slowdown, and defers CB-kernel optimization decisions.
