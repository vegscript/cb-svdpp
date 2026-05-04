# Runtime Profile Contract G1

- date: `2026-05-02`
- status: `pass`
- scope: `G1 concrete runtime profile validation`
- gate: `docs/roadmaps/2026-05-02_claim_unlock_and_scalability_plan.md`

## Purpose

This note documents the first claim-unlock remediation gate for runtime and
device-profile integrity. It does not unlock any model-quality, speed,
scalability, or paper-faithfulness claim.

## Change

Implemented:

- machine-readable device-profile contract assessment
- hard claim-eligible profile validation
- `validate-runtime-profile` CLI preflight
- `runtime.device_profile_contract` in run and benchmark manifests
- run-manifest and benchmark-manifest schema support for the new runtime field
- local reference profile marked as `validated_local_reference`
- draft `hpc_cpu.yaml` intentionally left non-claim-eligible

## Preflight Readout

Command:

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main validate-runtime-profile configs\runtime\devices\local_i5_2500k_24gb.yaml --claim-eligible
```

Readout:

- status: `valid`
- profile: `local_i5_2500k_24gb`
- compute class: `local_cpu`
- metadata status: `validated_local_reference`
- claim eligible: `true`
- blocking reasons: none
- RAM guardrail fraction: `0.8`

Command:

```powershell
.venv\Scripts\python.exe -m recsys_lab.cli.main validate-runtime-profile configs\runtime\devices\hpc_cpu.yaml --claim-eligible
```

Readout:

- status: expected failure
- reason: draft/template status plus unresolved CPU, thread, and RAM fields
- implication: the generic HPC template cannot be used for claim-eligible runs
  until it is replaced by a concrete cluster-specific profile

## Verification

Commands:

```powershell
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy src
.venv\Scripts\python.exe -m pytest
```

Readout:

- Ruff: `All checks passed!`
- Mypy source gate: `Success: no issues found in 60 source files`
- Pytest: `120 passed`

## Claim Boundary

Allowed claim:

- `G1` runtime-profile preflight exists and rejects non-concrete
  claim-eligible profiles.

Explicit non-claims:

- no runtime speed claim
- no stronger large-dataset CB claim
- no HPC readiness claim until a concrete cluster profile replaces the draft
  template
- no `scalable`, `production-ready`, or `publish-ready` claim
