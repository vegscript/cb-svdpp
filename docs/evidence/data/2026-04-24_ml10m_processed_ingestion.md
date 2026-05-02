# Evidence Note

## Scope

Official processed ingestion of `MovieLens 10M` into the repository's canonical
processed dataset contract.

## Claim Or Question

Can the repository prepare the full official `ml10m` ratings file through the
canonical CLI path after hardening the MovieLens preparation code for large
rating files?

## Inputs And Artifacts

- code commit used for processing:
  `f020ea937c42c2a5f74925fe51e3c79b5a0fde08`
- dataset config:
  `configs/data/movielens_10m.yaml`
- raw acquisition evidence:
  `docs/evidence/data/2026-04-24_ml10m_raw_acquisition.md`
- raw download manifest:
  `data/raw/ml10m/download_manifest.json`
- extracted raw directory:
  `data/raw/ml10m/ml-10M100K/`
- processed dataset manifest:
  `data/processed/ml10m/ml10m_benchmark_random_v1_explicit_v1_float32_manifest.json`

## Method

- keep `main` clean and synchronized before the run
- harden the MovieLens preparation path to avoid holding full large rating
  files as Python object dictionaries
- read rating columns through typed Arrow/NumPy arrays
- validate large raw files through streaming line counts rather than retaining
  all rows in memory
- run the canonical wrapper command:
  `.\scripts\prepare_dataset.ps1 -DatasetConfig configs\data\movielens_10m.yaml -Dtype float32 -Overwrite`

## Readout

- processed run completed on `2026-04-24`
- elapsed wall-clock time recorded by `Measure-Command`:
  `22.075` seconds
- raw format family:
  `legacy_10m`
- split family:
  `benchmark_random_v1`
- preprocessing family:
  `explicit_v1`
- dtype:
  `float32`
- interactions:
  `10000054`
- users:
  `69878`
- rated items:
  `10677`
- catalog items:
  `10681`
- tags:
  `95580`
- links:
  `0`
- rating range:
  `0.5` to `5.0`

## Verification

- focused regression before the real-data run:
  `pytest tests/integration/test_prepare_movielens.py`
- focused regression result:
  `4 passed`
- real-data preparation command completed with exit code `0`
- Git status remained clean after the data run because processed data artifacts
  are under the ignored `data/` artifact zone

## Interpretation

This closes the processed-data evidence gap for `ml10m`. It does not create any
model benchmark evidence and does not justify a `scalable`, `faster`,
`paper-faithful`, or `publish-ready` claim.

The recorded `22.075` seconds is only the dataset preparation wall-clock time on
the local environment. It is not a model runtime benchmark and must not be used
as model-performance evidence.

## Decision Or Next Step

- Treat `ml10m` as `processed_not_benchmarked`.
- Do not make model or benchmark claims on `ml10m`.
- Next action: run the cheapest clean `ml10m` feasibility baseline, starting
  with `biased_mf` under the canonical random split contract, and record the
  benchmark manifest and resource readout.
