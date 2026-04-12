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

### 3.1 Dataset Suitability

This section should document:

- rating scale compatibility
- user and item counts
- interaction counts
- sparsity characteristics
- availability of timestamps
- practical suitability for explicit-feedback factor models

### 3.2 Data Preparation

This section should document:

- ingestion format
- validation checks
- ID remapping
- split family
- conversion to Parquet or other processed forms
- dtype choices and their rationale

### 3.3 Data Engineering Decisions

This section should explain decisions such as:

- why Parquet is used
- how processed datasets are versioned
- why a given split family is appropriate

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

### 4.1 Mathematical Foundation

This section should summarize:

- the canonical notation
- the predictor functions
- the objective functions
- the update rules

Full mathematical detail belongs in `docs/math/`, while the report should
present the concise final narrative.

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

This section should briefly summarize:

- `src/` for active code
- `tests/` for verification
- `configs/` for canonical configuration
- `docs/` for governance and specifications
- `artifacts/` for generated outputs

### 5.2 Portability And Environment

This section should document:

- the canonical setup path
- device profiles
- portability goals
- dependency management

### 5.3 Performance Orientation

This section should describe:

- CPU-first optimization strategy
- data layout decisions
- sparse and vectorized processing choices
- why these choices are appropriate for the target hardware

## 6. Experimental Setup

This section should record the final experimental protocol.

### 6.1 Hardware And Software

Document:

- hardware profile
- Python version
- dependency stack
- thread configuration
- dtype profile

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
