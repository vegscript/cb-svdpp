from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from recsys_lab.tuning.planner import ArtifactReuseGroup, StudyPlan
from recsys_lab.tuning.schemas import (
    ArtifactReuseSpec,
    BudgetSpec,
    GeneratorSpec,
    ObjectiveSpec,
    StrictSchema,
    StudySpec,
)

CLAIM_BOUNDARY = "Dry-run tuning planner only; no performance or quality claim."
ExecutionStatus = Literal["not_executed", "running", "succeeded", "failed", "skipped"]


class CandidateManifest(StrictSchema):
    schema_version: Literal["candidate_manifest_v1"] = "candidate_manifest_v1"
    kind: Literal["candidate_manifest"] = "candidate_manifest"
    study_id: str
    candidate_id: str
    candidate_index: int = Field(ge=0)
    status: Literal["planned"] = "planned"
    objective_status: Literal["planned"] = "planned"
    execution_status: ExecutionStatus = "not_executed"
    study: StudySpec
    base_model_config: str
    parameter_values: dict[str, Any]
    overrides: dict[str, Any]
    materialized_config_payload: dict[str, Any]
    candidate_config_path: str | None = None
    run_id: str | None = None
    run_dir: str | None = None
    metrics_path: str | None = None
    performance_profile_path: str | None = None
    kernel_profile_path: str | None = None
    run_manifest_path: str | None = None
    error_message: str | None = None
    artifact_reuse_group_ids: dict[str, str] = Field(default_factory=dict)
    claim_boundary: str = CLAIM_BOUNDARY


class StudyManifest(StrictSchema):
    schema_version: Literal["study_manifest_v1"] = "study_manifest_v1"
    kind: Literal["study_manifest"] = "study_manifest"
    study_id: str
    study_name: str
    status: Literal["planned"] = "planned"
    search_space_version: str
    dataset: str
    split_family: str
    model: str
    seed: int
    study: StudySpec
    base_model_config: str
    budget: BudgetSpec
    generator: GeneratorSpec
    objective: ObjectiveSpec
    candidate_count: int = Field(ge=0)
    candidate_ids: list[str]
    artifact_reuse_contract: ArtifactReuseSpec | None = None
    artifact_reuse_group_count: int = Field(ge=0)
    created_at_utc: str
    claim_boundary: str = CLAIM_BOUNDARY


class ReuseSummary(StrictSchema):
    schema_version: Literal["artifact_reuse_summary_v1"] = "artifact_reuse_summary_v1"
    kind: Literal["artifact_reuse_summary"] = "artifact_reuse_summary"
    study_id: str
    groups: list[dict[str, Any]]


def build_study_manifest(plan: StudyPlan, *, created_at_utc: str) -> StudyManifest:
    return StudyManifest(
        study_id=plan.study_id,
        study_name=plan.search_space.study.name,
        search_space_version=plan.search_space.search_space_version,
        dataset=plan.search_space.study.dataset,
        split_family=plan.search_space.study.split_family,
        model=plan.search_space.study.model,
        seed=plan.search_space.study.seed,
        study=plan.search_space.study,
        base_model_config=plan.search_space.base_model_config,
        budget=plan.search_space.budget,
        generator=plan.search_space.generator,
        objective=plan.search_space.objective,
        candidate_count=len(plan.candidates),
        candidate_ids=[candidate.candidate_id for candidate in plan.candidates],
        artifact_reuse_contract=plan.search_space.artifact_reuse,
        artifact_reuse_group_count=len(plan.artifact_reuse_groups),
        created_at_utc=created_at_utc,
    )


def build_candidate_manifests(plan: StudyPlan, *, output_dir: str | None = None) -> list[CandidateManifest]:
    group_ids_by_candidate = _group_ids_by_candidate(plan.artifact_reuse_groups)
    return [
        CandidateManifest(
            study_id=plan.study_id,
            candidate_id=candidate.candidate_id,
            candidate_index=candidate.index,
            objective_status="planned",
            study=plan.search_space.study,
            base_model_config=plan.search_space.base_model_config,
            parameter_values=candidate.parameter_values,
            overrides=candidate.overrides,
            materialized_config_payload=candidate.materialized_config_payload,
            candidate_config_path=(
                f"{output_dir}/candidates/{candidate.candidate_id}/candidate_config.yaml"
                if output_dir is not None
                else candidate.materialized_config_path
            ),
            artifact_reuse_group_ids=group_ids_by_candidate.get(candidate.candidate_id, {}),
        )
        for candidate in plan.candidates
    ]


def build_reuse_summary(plan: StudyPlan) -> ReuseSummary:
    return ReuseSummary(
        study_id=plan.study_id,
        groups=[
            {
                "artifact_type": group.artifact_type,
                "group_id": group.group_id,
                "candidate_ids": group.candidate_ids,
                "reuse_key": group.reuse_key,
                "reuse_across": group.reuse_across,
                "invalidate_on": group.invalidate_on,
                "notes": group.notes,
            }
            for group in plan.artifact_reuse_groups
        ],
    )


def _group_ids_by_candidate(groups: list[ArtifactReuseGroup]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for group in groups:
        for candidate_id in group.candidate_ids:
            result.setdefault(candidate_id, {})[group.artifact_type] = group.group_id
    return result
