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

As of April 13, 2026, the first benchmark-eligible official dataset path is
also active: `MovieLens 100K` is now ingested from the original GroupLens
archive under `data/raw/ml100k/` and converted into the canonical processed
Parquet contract under `data/processed/ml100k/`.

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

For `MovieLens 100K`, the repository now supports the legacy raw layout
(`u.data`, `u.item`, `u.genre`) explicitly. The official package also contains
the provided benchmark split files `u1` to `u5`, `ua`, and `ub`. These are now
available in the raw zone and should be used to implement the canonical
`paper_faithful` split family in the next benchmark-correction step.

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

The first real tuning step now exists for `ml100k` as a leakagesafe inner
selection path. The repository uses `paper_faithful_ml100k_inner_v1` to derive
inner `train/validation` splits only from the official outer `uX.base`
partitions, while keeping `uX.test` untouched for final reporting. The first
implemented search is explicitly labeled `stage1` and uses only outer folds
`u1/u2` plus one model seed. This is sufficient for controlled correction of
weak defaults, but not yet sufficient for final benchmark claims.

## 7. Results

This section should contain only stable and evidence-backed results.

### 7.0 Initial Local POC Baseline

The first stable end-to-end results are local proof-of-concept baselines for
`biased_mf`, `svdpp`, `asymmetric_svd`, `asvdpp`, and `cb_svdpp` on
`ml_latest_small`. These runs are evidence-backed and manifest-valid, but they
are not part of the final benchmark ladder.

| Dataset | Model | Split | Train RMSE | Validation RMSE | Test RMSE | Auxiliary Fit Time (s) | Main Train Time (s) | End-to-End Fit Time (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ml_latest_small` | `biased_mf` | `benchmark_random_v1`, seed 1 | 0.4922 | 0.8706 | 0.8909 | 0.00 | 112.45 | 112.45 |
| `ml_latest_small` | `svdpp` | `benchmark_random_v1`, seed 1 | 0.4609 | 0.8670 | 0.8859 | 0.00 | 1192.53 | 1192.53 |
| `ml_latest_small` | `asymmetric_svd` | `benchmark_random_v1`, seed 1 | 0.7729 | 0.8533 | 0.8730 | 0.00 | 460.52 | 460.52 |
| `ml_latest_small` | `asvdpp` | `benchmark_random_v1`, seed 1 | 0.4386 | 0.8708 | 0.8881 | 0.00 | 1478.63 | 1478.63 |
| `ml_latest_small` | `cb_svdpp` | `benchmark_random_v1`, seed 1 | 0.5548 | 0.8549 | 0.8724 | 123.23 | 478.51 | 601.74 |

Associated evidence note:

- `docs/evidence/models/biased_mf/2026-04-12_ml_latest_small_biased_mf_local_poc_baseline.md`
- `docs/evidence/models/svdpp/2026-04-12_ml_latest_small_svdpp_local_poc_baseline.md`
- `docs/evidence/models/asymmetric_svd/2026-04-12_ml_latest_small_asymmetric_svd_local_poc_baseline.md`
- `docs/evidence/models/asvdpp/2026-04-13_ml_latest_small_asvdpp_local_poc_baseline.md`
- `docs/evidence/models/cb_svdpp/2026-04-13_ml_latest_small_cb_svdpp_local_poc_baseline.md`

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

The first local `cb_svdpp` baseline materially improves over the current
`biased_mf`, `svdpp`, and `asvdpp` runs and achieves the best test RMSE seen so
far on the local POC dataset. It does not, however, produce the best
validation RMSE; that remains with `asymmetric_svd`. The CB result is therefore
strong but not conclusive, and it must be reported together with its two-stage
fit cost: `123.23` seconds of cluster induction plus `478.51` seconds of main
training.

### 7.0.1 First Official `ml100k` Fold Readout

The repository now also has its first official `MovieLens 100K` readouts on the
implemented `paper_faithful_ml100k_v1` split family. This path uses the raw
package split files directly and therefore does not report a synthetic
validation metric.

| Dataset | Model | Split | Train RMSE | Test RMSE | Train Time (s) |
| --- | --- | --- | ---: | ---: | ---: |
| `ml100k` | `biased_mf` | `paper_faithful_ml100k_v1`, `u1` | 0.5589 | 0.9600 | 276.50 |
| `ml100k` | `svdpp` | `paper_faithful_ml100k_v1`, `u1` | 0.5461 | 0.9524 | 850.94 |

Associated evidence note:

- `docs/evidence/models/biased_mf/2026-04-13_ml100k_biased_mf_paper_faithful_u1_baseline.md`
- `docs/evidence/models/svdpp/2026-04-13_ml100k_svdpp_paper_faithful_u1_baseline.md`

The first official fold confirms the expected local direction: `svdpp`
outperforms `biased_mf` on identical official `ml100k` train/test splits.
However, this is still not a final benchmark conclusion because the result is
currently only based on fold `u1`, one seed, and draft hyperparameters.

### 7.0.2 First Official `ml100k` Five-Fold Benchmark

The repository now also has its first full `u1` to `u5` official
`MovieLens 100K` benchmark for both `biased_mf` and `svdpp` on the canonical
`paper_faithful_ml100k_v1` split family. Because the workspace was dirty during
benchmark execution, fold reuse was disabled deliberately and all five folds
were recomputed in homogeneous benchmark sessions.

| Dataset | Model | Splits | Train RMSE Mean | Train RMSE Std | Test RMSE Mean | Test RMSE Std | Train Time Mean (s) | Train Time Std (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ml100k` | `biased_mf` | `u1` to `u5` | 0.5567 | 0.0024 | 0.9524 | 0.0051 | 105.93 | 5.73 |
| `ml100k` | `svdpp` | `u1` to `u5` | 0.5446 | 0.0024 | 0.9445 | 0.0053 | 606.91 | 101.90 |

Associated evidence note:

- `docs/evidence/models/biased_mf/2026-04-13_ml100k_biased_mf_paper_faithful_u1_u5_benchmark.md`
- `docs/evidence/models/svdpp/2026-04-13_ml100k_svdpp_paper_faithful_u1_u5_benchmark.md`

These five-fold results matter more than the earlier single-fold readout for
two reasons. First, they show that the `svdpp` gain is not an artifact of one
favorable split: the mean test RMSE improves from `0.952430` to `0.944529`,
an absolute gain of `0.007901`. Second, the cost side is now quantified on the
same target hardware: `svdpp` takes about `5.73x` the mean training time of
`biased_mf` under the present draft settings.

The result closes the earlier concern that all implemented models were behaving
like near-baselines. On the official `ml100k` split family, `svdpp` now shows
the expected directional improvement over `biased_mf`. What remains open is not
whether the direction exists, but how much of the remaining gap to stronger
published numbers is due to untuned hyperparameters versus deeper implementation
or methodological debt.

### 7.0.3 First Leakagesafe `ml100k` Inner Tuning Studies

The repository now also has its first explicit tuning layer on top of the
official `ml100k` benchmark path. These studies do not use the official outer
test folds for model selection. Instead, they derive inner validation splits
only from `u1.base` and `u2.base` and rank candidates by mean validation RMSE.

| Model | Selection Stage | Outer Folds | Winning Candidate | Validation RMSE Mean | Validation RMSE Std |
| --- | --- | --- | --- | ---: | ---: |
| `biased_mf` | `stage1` | `u1`, `u2` | `rank064_lr0075_reg0030_e025` | 0.9334 | 0.0047 |
| `svdpp` | `stage1` | `u1`, `u2` | `rank080_lr0050_reg0030_e020` | 0.9207 | 0.0029 |

Associated evidence note:

- `docs/evidence/models/biased_mf/2026-04-13_ml100k_biased_mf_inner_tuning_stage1.md`
- `docs/evidence/models/svdpp/2026-04-13_ml100k_svdpp_inner_tuning_stage1.md`

These tuning studies establish two important points. First, the original draft
defaults were in fact weak enough to suppress the model family's potential.
Second, the no-leakage tuning path is now operational, so future improvements
can be made without compromising the benchmark contract.

### 7.0.4 First `stage1_tuned` Official `ml100k` Benchmarks

The winning `stage1` candidates were promoted into versioned repo configs and
re-benchmarked on the official outer `u1` to `u5` test folds. These are still
not final benchmark claims because the tuning stage used only `u1/u2` plus one
seed, but they are now the strongest current `ml100k` anchors in the repo.

| Dataset | Model | Config | Test RMSE Mean | Test RMSE Std | Absolute Gain vs. Draft | Train Time Mean (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `ml100k` | `biased_mf` | `stage1_tuned` | 0.9375 | 0.0077 | 0.0150 | 115.91 |
| `ml100k` | `svdpp` | `stage1_tuned` | 0.9235 | 0.0081 | 0.0210 | 658.87 |

Associated evidence note:

- `docs/evidence/models/biased_mf/2026-04-13_ml100k_biased_mf_stage1_tuned_benchmark.md`
- `docs/evidence/models/svdpp/2026-04-13_ml100k_svdpp_stage1_tuned_benchmark.md`

The tuned results materially change the current interpretation of the repo. The
earlier weak numbers were not just a matter of model family choice; they were
also a matter of under-tuned defaults. After the first leakagesafe tuning
correction:

- tuned `biased_mf` improves from `0.952430` to `0.937461`
- tuned `svdpp` improves from `0.944529` to `0.923483`
- tuned `svdpp` remains better than tuned `biased_mf` by `0.013978` RMSE on the
  current official five-fold mean

The price side remains important. The tuned `svdpp` profile is still much more
expensive than tuned `biased_mf`, with about `5.68x` the mean training time on
the default CPU target. So the current repo state now supports a much stronger
and more realistic statement: the model family does improve after tuning, but
the quality gain must be read together with a significant compute premium.

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
