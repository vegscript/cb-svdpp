"""Tuning framework contracts and dry-run planning helpers.

This package intentionally does not execute studies, run candidates, or depend
on optimizer libraries.
"""

from recsys_lab.tuning.candidates import CandidateSpec, generate_candidates
from recsys_lab.tuning.execution import (
    CandidateExecutionResult,
    ExecutionStatus,
    execute_candidate,
    load_candidate_manifest,
    resolve_candidate_config_path,
    skipped_candidate_result,
)
from recsys_lab.tuning.manifests import (
    CandidateManifest,
    ReuseSummary,
    StudyManifest,
    build_candidate_manifests,
    build_reuse_summary,
    build_study_manifest,
)
from recsys_lab.tuning.planner import ArtifactReuseGroup, StudyPlan, build_study_plan
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
from recsys_lab.tuning.writers import (
    update_candidate_manifest_with_execution_result,
    write_artifact_reuse_summary_csv,
    write_candidate_summary_csv,
    write_execution_summary_csv,
    write_study_execution_artifacts,
    write_study_plan,
    write_tuning_json,
    write_tuning_yaml,
)

__all__ = [
    "DEFAULT_CLUSTER_ARTIFACT_INVALIDATE_ON",
    "DEFAULT_CLUSTER_ARTIFACT_REUSE_ACROSS",
    "ArtifactReuseGroup",
    "ArtifactReuseSpec",
    "BudgetSpec",
    "CandidateExecutionResult",
    "CandidateManifest",
    "CandidateSpec",
    "ClusterArtifactReuseSpec",
    "DimensionSpec",
    "ExecutionStatus",
    "GeneratorSpec",
    "ObjectiveMetricSpec",
    "ObjectiveSpec",
    "ReuseSummary",
    "SearchSpaceSpec",
    "StudyManifest",
    "StudyPlan",
    "StudySpec",
    "build_candidate_manifests",
    "build_reuse_summary",
    "build_study_manifest",
    "build_study_plan",
    "default_cluster_artifact_reuse_spec",
    "execute_candidate",
    "generate_candidates",
    "load_candidate_manifest",
    "resolve_candidate_config_path",
    "skipped_candidate_result",
    "update_candidate_manifest_with_execution_result",
    "write_artifact_reuse_summary_csv",
    "write_candidate_summary_csv",
    "write_execution_summary_csv",
    "write_study_execution_artifacts",
    "write_study_plan",
    "write_tuning_json",
    "write_tuning_yaml",
]
