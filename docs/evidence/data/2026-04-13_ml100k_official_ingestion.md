# Evidence Note

## Scope

Official ingestion of the first benchmark-eligible dataset, `MovieLens 100K`,
into the repository's canonical raw and processed data contracts.

## Claim Or Question

Can the repository now ingest the official `MovieLens 100K` release from
GroupLens, preserve its legacy layout explicitly, and convert it into the same
processed Parquet contract that the model pipeline already uses?

## Inputs And Artifacts

- dataset config: `configs/data/movielens_100k.yaml`
- official archive:
  `data/raw/ml100k/ml-100k.zip`
- raw download manifest:
  `data/raw/ml100k/download_manifest.json`
- extracted raw directory:
  `data/raw/ml100k/ml-100k/`
- processed dataset manifest:
  `data/processed/ml100k/ml100k_benchmark_random_v1_explicit_v1_float32_manifest.json`

## Method

- extend the MovieLens ingestion layer to support both `modern_csv` and the
  legacy `ml100k` layout
- download the official `ml-100k.zip` archive from GroupLens
- extract the archive into `data/raw/ml100k/`
- validate the raw file structure as `legacy_100k`
- prepare the dataset through the canonical `prepare-dataset` CLI path into
  typed Parquet artifacts with `float32`

## Readout

- raw download completed on `2026-04-13`
- official URL:
  `https://files.grouplens.org/datasets/movielens/ml-100k.zip`
- archive SHA-256:
  `50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229`
- raw format family: `legacy_100k`
- interactions: `100000`
- users: `943`
- rated items: `1682`
- catalog items: `1682`
- tags: `0`
- links: `0`
- official split files present in raw data:
  `u1.base/u1.test` through `u5.base/u5.test`, plus `ua` and `ub`

## Interpretation

The repository now has a benchmark-eligible official MovieLens dataset under a
clean raw-manifest and processed-manifest contract. This closes the biggest
comparison gap between the earlier local POC work on `ml_latest_small` and the
target paper scope.

However, this step alone does not yet make the results paper-comparable. The
processed manifest currently uses the repo's `benchmark_random_v1` split family.
The official `ml100k` split files are present in the raw directory but are not
yet wired into a canonical `paper_faithful` execution path.

## Decision Or Next Step

- treat `ml100k` as the first official benchmark-ready dataset source in the repo
- next implementation target: canonical support for the provided `u1` to `u5`
  split files as the `paper_faithful` split family
- after that, rerun `biased_mf` and `svdpp` on `ml100k` before making stronger
  claims about RMSE quality
