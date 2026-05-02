# Evidence Note

## Scope

Official raw acquisition of `MovieLens 10M` for the publish-scope dataset gate.

## Claim Or Question

Can the repository acquire the official GroupLens `ml-10m` archive, record a
raw download manifest, and confirm the expected legacy `10M/100K` file layout
before attempting memory-sensitive processing?

## Inputs And Artifacts

- dataset config: `configs/data/movielens_10m.yaml`
- official archive:
  `data/raw/ml10m/ml-10m.zip`
- raw download manifest:
  `data/raw/ml10m/download_manifest.json`
- extracted raw directory:
  `data/raw/ml10m/ml-10M100K/`

## Method

- verify the official GroupLens `MovieLens 10M` download page
- download `https://files.grouplens.org/datasets/movielens/ml-10m.zip`
- extract the archive under the canonical raw data zone
- compute the archive SHA-256
- count the raw `ratings.dat`, `movies.dat`, and `tags.dat` lines
- do not run the full processed-data preparation yet, because that step should
  be memory-reviewed for the 10M/20M publish-scope datasets before execution

## Readout

- raw download completed on `2026-04-24`
- official URL:
  `https://files.grouplens.org/datasets/movielens/ml-10m.zip`
- source page:
  `https://grouplens.org/datasets/movielens/10m/`
- archive SHA-256:
  `813c411ccb6122564edfe752e7f80c4dcc5aa25fa94c93622f6877a7ba252862`
- archive size bytes:
  `65566137`
- raw format family:
  `legacy_10m`
- extracted directory:
  `ml-10M100K`
- ratings rows:
  `10000054`
- movies rows:
  `10681`
- tags rows:
  `95580`
- links:
  `0`
- users:
  not provided as a separate file in this dataset layout

## Interpretation

This closes only the raw acquisition part of the `ml10m` dataset gate. It does
not yet make `ml10m` benchmark-ready. The processed manifest, schema validation,
and any model benchmark evidence remain open.

The layout-specific code path for `legacy_10m` is now implemented and tested on
a synthetic fixture, but the full real-data preparation has intentionally not
been run in this step. The preparation path should be memory-reviewed before
processing `10,000,054` ratings and later `ml20m`.

## Decision Or Next Step

- Treat `ml10m` as `raw_acquired`.
- Do not make model or benchmark claims on `ml10m`.
- Next action: harden or verify the processed-data preparation path for large
  MovieLens datasets, then create the canonical `ml10m` processed manifest.
