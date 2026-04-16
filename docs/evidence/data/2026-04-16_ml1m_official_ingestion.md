# Evidence Note

## Scope

Official ingestion of `MovieLens 1M` into the repository's canonical raw and
processed dataset contracts.

## Claim Or Question

Can the repository now ingest the official GroupLens `ml-1m` release, validate
the legacy double-colon layout explicitly, and prepare it into the same typed
Parquet contract already used by the benchmark ladder?

## Inputs And Artifacts

- dataset config: `configs/data/movielens_1m.yaml`
- official archive:
  `data/raw/ml1m/ml-1m.zip`
- raw download manifest:
  `data/raw/ml1m/download_manifest.json`
- extracted raw directory:
  `data/raw/ml1m/ml-1m/`
- processed dataset manifest:
  `data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json`

## Method

- extend the MovieLens ingestion layer to support the `legacy_1m` layout
- download the official `ml-1m.zip` archive from GroupLens
- validate `ratings.dat`, `movies.dat`, and `users.dat` as a legacy double-colon dataset
- prepare the dataset through the canonical `prepare-dataset` CLI path into typed `float32` Parquet artifacts

## Readout

- raw download completed on `2026-04-16`
- official URL:
  `https://files.grouplens.org/datasets/movielens/ml-1m.zip`
- archive SHA-256:
  `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`
- raw format family: `legacy_1m`
- interactions: `1000209`
- users: `6040`
- rated items: `3706`
- catalog items: `3883`
- tags: `0`
- links: `0`
- rating range: `1.0` to `5.0`

## Interpretation

The repository now has its second official benchmark-eligible dataset source in
the canonical raw-manifest and processed-manifest contracts. This closes the
data-engineering gap for the first medium-scale MovieLens dataset and enables a
real scaling readout beyond `ml100k`.

## Decision Or Next Step

- treat `ml1m` as the active next scaling step in the benchmark ladder
- use transfer profiles first to establish runtime and memory behavior on the default local device
- do not promote any `ml1m` model result to a benchmark anchor until clean reruns and dataset-specific tuning exist
