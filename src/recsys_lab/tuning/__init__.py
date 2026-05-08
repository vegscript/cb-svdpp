"""Tuning framework contracts.

This package intentionally contains schema contracts only. It does not execute
studies, run candidates, or depend on optimizer libraries.
"""

from recsys_lab.tuning.schemas import (
    DEFAULT_CLUSTER_ARTIFACT_INVALIDATE_ON,
    DEFAULT_CLUSTER_ARTIFACT_REUSE_ACROSS,
    ArtifactReuseSpec,
    BudgetSpec,
    ClusterArtifactReuseSpec,
    DimensionSpec,
    GeneratorSpec,
    ObjectiveMetricSpec,
    ObjectiveSpec,
    SearchSpaceSpec,
    StudySpec,
    default_cluster_artifact_reuse_spec,
)

__all__ = [
    "DEFAULT_CLUSTER_ARTIFACT_INVALIDATE_ON",
    "DEFAULT_CLUSTER_ARTIFACT_REUSE_ACROSS",
    "ArtifactReuseSpec",
    "BudgetSpec",
    "ClusterArtifactReuseSpec",
    "DimensionSpec",
    "GeneratorSpec",
    "ObjectiveMetricSpec",
    "ObjectiveSpec",
    "SearchSpaceSpec",
    "StudySpec",
    "default_cluster_artifact_reuse_spec",
]
