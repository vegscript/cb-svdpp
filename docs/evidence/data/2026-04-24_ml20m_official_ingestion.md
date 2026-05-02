# Evidence Note

## Scope

Official ingestion of `MovieLens 20M` into the repository's canonical raw and
processed dataset contracts.

## Claim Or Question

Can the repository acquire the official GroupLens `ml-20m` archive, verify the
download against the official checksum, validate the modern CSV layout, and
prepare the full dataset through the canonical CLI path?

## Inputs And Artifacts

- dataset config:
  `configs/data/movielens_20m.yaml`
- official archive:
  `data/raw/ml20m/ml-20m.zip`
- official checksum file:
  `data/raw/ml20m/ml-20m.zip.md5`
- raw download manifest:
  `data/raw/ml20m/download_manifest.json`
- extracted raw directory:
  `data/raw/ml20m/ml-20m/`
- processed dataset manifest:
  `data/processed/ml20m/ml20m_benchmark_random_v1_explicit_v1_float32_manifest.json`

## Method

- verify the official GroupLens `MovieLens 20M` download page
- download `https://files.grouplens.org/datasets/movielens/ml-20m.zip`
- download the official `ml-20m.zip.md5` checksum file
- compute local MD5 and SHA-256 for the archive
- verify local MD5 against the official checksum file
- extract the archive under the canonical raw data zone
- validate the required modern CSV files and headers
- prepare the dataset through the canonical wrapper command:
  `.\scripts\prepare_dataset.ps1 -DatasetConfig configs\data\movielens_20m.yaml -Dtype float32 -Overwrite`

## Readout

- raw download completed on `2026-04-24`
- official URL:
  `https://files.grouplens.org/datasets/movielens/ml-20m.zip`
- source page:
  `https://grouplens.org/datasets/movielens/20m/`
- source README:
  `https://files.grouplens.org/datasets/movielens/ml-20m-README.html`
- archive MD5:
  `cd245b17a1ae2cc31bb14903e1204af3`
- official MD5 line:
  `MD5 (ml-20m.zip) = cd245b17a1ae2cc31bb14903e1204af3`
- MD5 verification:
  `pass`
- archive SHA-256:
  `96f243c338a8665f6bcc89c53edf6ee39162a846940de6b7c8c48aeada765ff3`
- archive size bytes:
  `198702078`
- raw format family:
  `modern_csv`
- processed run elapsed wall-clock time recorded by `Measure-Command`:
  `47.798` seconds
- split family:
  `benchmark_random_v1`
- preprocessing family:
  `explicit_v1`
- dtype:
  `float32`
- raw ratings rows:
  `20000263`
- raw movies rows:
  `27278`
- raw links rows:
  `27278`
- raw tags rows:
  `465564`
- users:
  `138493`
- processed interactions:
  `20000263`
- processed rated items:
  `26744`
- processed catalog items:
  `27278`
- rating range:
  `0.5` to `5.0`

## Verification

- raw directory validation:
  `validate_movielens_directory(..., format_family="modern_csv")`
- raw directory validation result:
  `ratings=20000263`, `movies=27278`, `links=27278`, `tags=465564`
- real-data preparation command completed with exit code `0`
- Git status remained clean after the data run because raw and processed data
  artifacts are under ignored data zones

## Interpretation

This closes the raw and processed dataset evidence gap for `ml20m`. It does not
create any model benchmark evidence and does not justify a `scalable`,
`faster`, `paper-faithful`, or `publish-ready` claim.

The recorded `47.798` seconds is only the dataset preparation wall-clock time on
the local environment. It is not a model runtime benchmark and must not be used
as model-performance evidence.

## Decision Or Next Step

- Treat `ml20m` as `processed_not_benchmarked`.
- Do not make model or benchmark claims on `ml20m`.
- Next action: run the cheapest clean `ml20m` feasibility baseline only if the
  plan accepts the expected local runtime and memory risk; otherwise document
  bounded infeasibility and move heavier model runs to a stronger device.
