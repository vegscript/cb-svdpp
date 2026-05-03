# ML20M CB-SVD++ G11 Lower-Memory Validation Contract

- date: `2026-05-03`
- status: `contract_ready_g11_ml20m_lower_memory_validation_reassessment`
- dataset: `ml20m`
- model: `cb_svdpp`
- config:
  `configs/experiments/tuning/ml20m_cb_svdpp_g11_lower_memory_validation_grid.yaml`
- blocked campaign contract:
  `docs/evidence/benchmarking/2026-04-30_large_cb_svdpp_matched_campaign_contract.md`
- negative resource evidence:
  `docs/evidence/models/cb_svdpp/2026-05-02_ml20m_cb_svdpp_stage0_transfer_seed3_guardrail_breach.md`

## Purpose

This is a run contract, not a benchmark result. It defines the next allowed
local `ml20m cb_svdpp` reassessment after the previous full `stage0_transfer`
campaign crossed the local 80 percent RAM guardrail.

The contract intentionally uses a lower-memory, validation-only grid. It does
not authorize another local full-budget `ml20m cb_svdpp stage0_transfer`
promotion attempt under the same risk profile.

## Planned Command

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main tune-inner `
  configs\experiments\tuning\ml20m_cb_svdpp_g11_lower_memory_validation_grid.yaml `
  data\processed\ml20m\ml20m_benchmark_random_v1_explicit_v1_float32_manifest.json `
  configs\runtime\base.yaml `
  configs\runtime\devices\local_i5_2500k_24gb.yaml `
  --split-cache auto `
  --training-index-cache `
  --cluster-artifact-cache
```

## Selection Scope

- split family: `benchmark_random_v1`
- train ratio: `0.8`
- validation ratio: `0.1`
- split seeds: `1,2`
- model seed: `1`
- candidate count: `8`
- planned validation-only runs: `16`
- base model config: `configs/models/cb_svdpp.yaml`
- clustering algorithm: `kmeans`
- `MiniBatchKMeans`: not allowed in this contract
- `R_star`: diagnostic only, not an objective term and not a tuning coefficient
- test-set evaluation during selection: not allowed

## Candidate Axes

| Axis | Values |
| --- | --- |
| latent_dim | `16`, `32` |
| user clusters | `32`, `64` |
| item clusters | `32`, `64` |
| alpha | `0.0`, `0.025` |
| epochs | `1` |
| learning rate | `0.01` |
| regularization family | `0.02` for bias, factors, feedback, and cluster terms |

## Resource Gate

The local reference profile is `local_i5_2500k_24gb` with
`ram_guardrail_fraction=0.8`. The effective local memory threshold is
`19660.8 MB`.

A candidate is rejected if any of its runs crosses the local memory guardrail.
Selection by `validation_rmse_mean` is allowed only after all runs for that
candidate satisfy the resource gate.

If no candidate satisfies the resource gate, the correct outcome is negative
resource evidence. It must not be converted into a quality, speed,
scalability, production-readiness, SOTA, paper-faithfulness, or large-dataset
CB claim.

## Operational Enforcement

The `tune-inner` path now reads `resource_gate` from the tuning config. For each
candidate run it records `system_metrics.peak_memory_mb`, compares it with
`max_peak_memory_mb`, and stores the per-run resource-gate readout in the
summary.

If `reject_candidate_on_any_guardrail_breach=true`, a candidate is stopped
after its first resource-gate breach and excluded from validation-RMSE
selection. The best candidate is selected only from candidates that completed
all planned selection units without a resource-gate breach.

## Promotion Gate

This contract can produce only selection evidence. A final or comparison-ready
`ml20m cb_svdpp` claim still requires a separate committed outer benchmark
contract with:

- frozen selected config
- split seeds `1,2,3`
- model seed `1`
- clean git state for every run and aggregation
- matching processed manifest, split family, dtype, runtime config, device
  config, cache policy, and effective model config
- no RAM guardrail breach
- one canonical benchmark manifest from `benchmark-random-multiseed`

## Claim Boundary

Allowed:

- this contract authorizes one lower-memory, validation-only local reassessment
  of `ml20m cb_svdpp`
- future evidence may say whether the planned grid passed, failed, or was
  stopped by the resource gate

Not allowed:

- no final `ml20m cb_svdpp` benchmark claim
- no `ml20m` model-comparison claim
- no quality claim from this contract
- no speed claim from this contract
- no scalability claim from this contract
- no production-readiness claim from this contract
- no SOTA claim from this contract
- no paper-faithfulness claim from this contract
- no test-set claim from selection runs

## Evidence Gate Readout

This contract is claimable as a contract only after:

- its config loads through the repo YAML loader
- guardrail tests confirm validation-only scope, no test metric references, no
  `MiniBatchKMeans`, explicit `kmeans`, and explicit RAM rejection semantics
- integration tests confirm `tune-inner` excludes breached candidates from
  selection and keeps the lower-memory candidate eligible
- Ruff, Mypy, and the full test suite pass

Executed after documenting this contract:

- Ruff: `All checks passed!`
- Mypy: `Success: no issues found in 62 source files`
- Pytest: `143 passed`
