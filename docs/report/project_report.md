# Project Report

## Title

Reproduction and Engineering Study of Clustering-Based Factorized Collaborative Filtering

## Abstract

This project investigates the reproduction and structured re-implementation of
clustering-based factorized collaborative filtering models for explicit rating
prediction. The work focuses on a modular model family spanning `biased_mf`,
`svdpp`, `asymmetric_svd`, `asvdpp`, `cb_svdpp`, and `cb_asvdpp` across
MovieLens datasets from 100K to 20M interactions. The report documents dataset
selection, preprocessing, mathematical modeling, engineering design, evaluation
protocol, and empirical findings with a joint focus on predictive quality and
system performance.

## 1. Introduction

Recommender systems are a central application area of data science because they
combine statistical modeling, large-scale data processing, and system-level
tradeoffs. In the explicit-feedback setting, matrix factorization methods and
their extensions remain historically important because they offer a strong
combination of accuracy, interpretability of design choices, and computational
efficiency.

This project studies the clustering-based factorized collaborative filtering
poster by Mirbakhsh and Ling and places it in the broader factorization family
grounded in Koren's work on `SVD++` and `Asymmetric-SVD`. The goal is not an
uncritical one-to-one clone, but a conceptually faithful and methodologically
rigorous reconstruction that supports controlled ablations and cross-dataset
comparisons.

## 2. Research Objective And Reproduction Scope

The objective of the project is to build a research-grade recommender-system
pipeline that makes it possible to compare progressively richer factorization
models under identical data and evaluation conditions.

The project has three reproduction goals:

1. reconstruct the mathematical model family in a modular form
2. reproduce the core experimental logic of the source paper
3. evaluate both predictive accuracy and computational performance

The project does not claim exact reproduction where the source material is
under-specified. In such cases, implementation decisions are documented
explicitly and treated as controlled repo-level contracts rather than invisible
assumptions.

## 3. Data

The official dataset scope of this project consists of:

- MovieLens 100K
- MovieLens 1M
- MovieLens 10M
- MovieLens 20M

These datasets were selected because they provide a clean scaling ladder from
small to medium-large explicit-feedback recommender benchmarks while remaining
compatible with fair cross-model comparisons.

In addition to the official benchmark ladder, the repository maintains a local
proof-of-concept dataset contract for `ml_latest_small`. This local dataset is
used only to validate the engineering pipeline, manifest flow, and first model
implementations before moving to benchmark-eligible datasets. It is explicitly
not treated as a substitute for MovieLens 100K.

### 3.1 Dataset Suitability

This section should document:

- rating scale compatibility
- user and item counts
- interaction counts
- sparsity characteristics
- availability of timestamps
- practical suitability for explicit-feedback factor models

### 3.2 Data Preparation

The current ingestion path validates raw MovieLens CSV structure, checks file
presence, counts the core entities, and then produces a processed explicit
feedback representation in Parquet. User and item identifiers are remapped
deterministically to dense integer indices so that all downstream models can
operate on compact arrays and consistent sparse layouts.

The processed dataset manifest records:

- dataset identity
- preprocessing family
- dtype
- interaction and catalog counts
- rating range
- generated artifact paths

The initial local proof-of-concept path converts `ml_latest_small` into
`float32` Parquet artifacts under `data/processed/ml_latest_small/`.

### 3.3 Data Engineering Decisions

Parquet is used because it gives a compact columnar format with predictable
schema handling, efficient typed reads, and a clean boundary between raw data
and model-ready artifacts. The repository deliberately does not train directly
from CSV files. This avoids repeated parsing costs and keeps preprocessing
decisions explicit and inspectable.

Processed artifacts are versioned indirectly through their manifest metadata,
their dataset short name, their preprocessing family, and their dtype. Split
construction is performed after loading the processed interaction table, so the
raw-to-processed contract and the train-validation-test contract remain
separate.

## 4. Methodology

The methodological core of the project is a layered model family:

1. `biased_mf`
2. `svdpp`
3. `asymmetric_svd`
4. `asvdpp`
5. `cb_svdpp`
6. `cb_asvdpp`

This structure supports ablations that isolate the effect of:

- baseline factorization
- implicit feedback
- asymmetric user representation
- explicit item-side feedback aggregation
- clustering-based augmentation

For `asymmetric_svd` and later `asvdpp`, the repository now uses an accepted
repo-defined optimizer deviation: explicit residual weights are treated as
detached within each SGD step. This makes the implementation stable and
reproducible, but it also means these models must not be described as exact
optimizer-faithful reproductions of the source formulation.

For the clustering-based family, the repository now uses a separate accepted
v1 contract. User and item clusters are induced from train-only `biased_mf`
latents with separate `KMeans` models, the assignments remain fixed during CB
training, and `R_star` is treated as a train-only diagnostic artifact rather
than a second optimization target. This preserves the published predictor while
avoiding an unsupported extra factorization stage.

### 4.1 Mathematical Foundation

This section should summarize:

- the canonical notation
- the predictor functions
- the objective functions
- the update rules

Full mathematical detail belongs in `docs/math/`, while the report should
present the concise final narrative.

For the current explicit-feedback implementation path, `svdpp` uses the
training-only rated-item set of each user as the implicit neighborhood. This
keeps the implicit term leakage-safe during validation and test evaluation while
remaining consistent with the repo's current explicit-feedback contract.

### 4.2 Clustering-Based Extension

This section should explain:

- how latent vectors are used for clustering
- why clustering occurs after factorization
- how cluster-level factors are mixed with individual factors
- how `alpha` is used and optimized

### 4.3 Reproduction Boundaries

This section should explicitly state:

- which parts are source-grounded
- which parts are repo-defined implementation contracts
- which parts remain open or under-specified in the source paper

## 5. System Design And Implementation

The project is implemented as a modular research codebase rather than as a set
of notebooks or monolithic scripts.

### 5.1 Repository Structure

The repository follows a strict separation between source code, governance, and
generated artifacts:

- `src/` contains active implementation code
- `tests/` contains unit and integration guards
- `configs/` contains canonical runtime, dataset, model, and experiment configs
- `docs/` contains governance, mathematical specifications, evidence, and the
  continuously maintained project report
- `artifacts/` contains generated run outputs, benchmark outputs, debug data,
  and figures

The current active implementation path covers dataset preparation, processed
dataset loading, split generation with train coverage guarantees, `biased_mf`
training, RMSE evaluation, and run-manifest generation.

### 5.2 Portability And Environment

The project is configured for portable single-command setup through the repo
tooling and typed configuration files. Runtime defaults are encoded in
`configs/runtime/base.yaml`, while device-specific settings are separated into
dedicated device profiles such as `local_i5_2500k_24gb`.

This design avoids hidden machine-specific assumptions. A run artifact records
the exact device profile, Python version, dtype, threading settings, git state,
and configuration references that produced the result.

### 5.3 Performance Orientation

The repository is currently CPU-first. The local default hardware profile is an
Intel i5-2500K system with 24 GB RAM, so the initial engineering focus is on
typed arrays, deterministic ID compaction, Parquet-based preprocessing, and
moderate thread control rather than on GPU kernels or distributed execution.

```mermaid
flowchart LR
    A["Raw MovieLens CSV"] --> B["Validation"]
    B --> C["Deterministic ID Remapping"]
    C --> D["Processed Parquet + Manifest"]
    D --> E["Train/Validation/Test Split"]
    E --> F["Biased MF Training"]
    F --> G["Metrics + Run Manifest + Log"]
```

## 6. Experimental Setup

This section should record the final experimental protocol.

### 6.1 Hardware And Software

The first implemented baseline was executed on the default local device profile:

- CPU: Intel Core i5-2500K
- logical threads: 4
- RAM: 24 GB
- Python: 3.10.7
- dtype profile: `float32`
- thread profile: `omp_num_threads=4`, `blas_threads=4`

The current core stack is `NumPy`, `PyArrow`, `PyYAML`, `Typer`, `jsonschema`,
and `threadpoolctl`, with `pytest` for verification.

### 6.2 Evaluation Protocol

Document:

- split family
- seed policy
- primary metrics
- performance metrics
- ranking metrics, if enabled

### 6.3 Hyperparameter Strategy

Document:

- latent dimensions
- regularization ranges
- learning rates
- epoch strategy
- cluster counts
- `alpha` sweep design

## 7. Results

This section should contain only stable and evidence-backed results.

### 7.0 Initial Local POC Baseline

The first stable end-to-end results are local proof-of-concept baselines for
`biased_mf`, `svdpp`, `asymmetric_svd`, and `asvdpp` on `ml_latest_small`. These runs are
evidence-backed and manifest-valid, but they are not part of the final benchmark
ladder.

| Dataset | Model | Split | Train RMSE | Validation RMSE | Test RMSE | Train Time (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `ml_latest_small` | `biased_mf` | `benchmark_random_v1`, seed 1 | 0.4922 | 0.8706 | 0.8909 | 112.45 |
| `ml_latest_small` | `svdpp` | `benchmark_random_v1`, seed 1 | 0.4609 | 0.8670 | 0.8859 | 1192.53 |
| `ml_latest_small` | `asymmetric_svd` | `benchmark_random_v1`, seed 1 | 0.7729 | 0.8533 | 0.8730 | 460.52 |
| `ml_latest_small` | `asvdpp` | `benchmark_random_v1`, seed 1 | 0.4386 | 0.8708 | 0.8881 | 1478.63 |

Associated evidence note:

- `docs/evidence/models/biased_mf/2026-04-12_ml_latest_small_biased_mf_local_poc_baseline.md`
- `docs/evidence/models/svdpp/2026-04-12_ml_latest_small_svdpp_local_poc_baseline.md`
- `docs/evidence/models/asymmetric_svd/2026-04-12_ml_latest_small_asymmetric_svd_local_poc_baseline.md`
- `docs/evidence/models/asvdpp/2026-04-13_ml_latest_small_asvdpp_local_poc_baseline.md`

The current local comparison suggests that the implicit-feedback extension of
`svdpp` improves rating prediction quality slightly on `ml_latest_small`, but
with a much larger training cost on the default local CPU profile. This is a
directional engineering result, not yet a final benchmark conclusion.

The current local `asymmetric_svd` baseline improves validation and test RMSE
further while also training substantially faster than the current `svdpp`
implementation on the default local device profile. Because `asymmetric_svd`
uses the accepted detached-residual optimizer contract, this result is
reportable as a repo-defined engineering baseline, not as an exact
optimizer-faithful paper reproduction.

The current local `asvdpp` baseline does not improve over `svdpp` or
`asymmetric_svd` on `ml_latest_small`. This is an important negative result and
is kept explicitly in the report. It suggests that the additional free-user
factor block is not automatically beneficial under the present draft
hyperparameters and local POC dataset scale.

### 7.1 Main Model Comparisons

This subsection should compare models on identical datasets and split settings.

Recommended figure and table types:

- RMSE comparison tables
- train-time comparison tables
- throughput comparisons
- dataset-wise model comparison plots

### 7.2 Ablation Studies

This subsection should contain:

- `alpha` sweeps
- cluster-count sweeps
- comparisons with and without clustering
- comparisons with and without explicit feedback blocks

### 7.3 Performance Results

This subsection should report:

- training time
- ratings per second
- peak memory
- scalability trends from 100K to 20M

## 8. Discussion

This section should interpret the results rather than repeat them.

Relevant themes include:

- when clustering helps
- when `asymmetric_svd` or `asvdpp` are preferable
- where performance gains or losses come from
- how dataset scale changes the picture

## 9. Limitations

This section should explicitly acknowledge:

- under-specified parts of the poster
- limits of rating-prediction metrics
- hardware constraints
- remaining implementation and evaluation boundaries

## 10. Conclusion

This section should summarize:

- what was built
- what was reproduced
- what was learned
- what remains open

## 11. References

This section should use a consistent academic citation style.

Initial core references:

- Mirbakhsh, N., and Ling, C. X. Clustering-Based Factorized Collaborative Filtering.
- Koren, Y. Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model.
- MovieLens dataset reference.

## 12. Appendix

Optional appendix material may include:

- compact supplementary tables
- additional implementation clarifications
- additional plots that support but do not belong in the main narrative
