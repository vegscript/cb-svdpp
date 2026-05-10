from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from recsys_lab.tuning.candidates import CandidateSpec, generate_candidates
from recsys_lab.tuning.schemas import SearchSpaceSpec
from recsys_lab.tuning.search_roles import is_inner_target_coordinate


@dataclass(frozen=True)
class ArtifactReuseGroup:
    artifact_type: str
    group_id: str
    candidate_ids: list[str]
    reuse_key: dict[str, Any]
    reuse_across: list[str]
    invalidate_on: list[str]
    notes: str


@dataclass(frozen=True)
class StudyPlan:
    study_id: str
    search_space: SearchSpaceSpec
    candidates: list[CandidateSpec]
    artifact_reuse_groups: list[ArtifactReuseGroup]


def build_study_plan(
    search_space: SearchSpaceSpec,
    *,
    stage_name: str | None = None,
    max_candidates: int | None = None,
) -> StudyPlan:
    study_id = _study_id(search_space)
    candidates = generate_candidates(search_space, study_id=study_id, stage_name=stage_name)
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    return StudyPlan(
        study_id=study_id,
        search_space=search_space,
        candidates=candidates,
        artifact_reuse_groups=_build_artifact_reuse_groups(search_space=search_space, candidates=candidates),
    )


def _build_artifact_reuse_groups(
    *, search_space: SearchSpaceSpec, candidates: list[CandidateSpec]
) -> list[ArtifactReuseGroup]:
    if search_space.artifact_reuse is None or search_space.artifact_reuse.cluster_artifacts is None:
        return []

    cluster_contract = search_space.artifact_reuse.cluster_artifacts
    reuse_across = set(cluster_contract.reuse_across)
    groups: dict[str, list[CandidateSpec]] = {}
    reuse_keys: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        reuse_key = _candidate_reuse_key(
            candidate=candidate,
            search_space=search_space,
            reuse_across=reuse_across,
        )
        encoded = _stable_json(reuse_key)
        groups.setdefault(encoded, []).append(candidate)
        reuse_keys[encoded] = reuse_key

    reuse_groups: list[ArtifactReuseGroup] = []
    for encoded_key, group_candidates in sorted(groups.items()):
        group_id = f"cluster_rg_{hashlib.sha256(encoded_key.encode('utf-8')).hexdigest()[:12]}"
        reuse_groups.append(
            ArtifactReuseGroup(
                artifact_type="cluster_artifacts",
                group_id=group_id,
                candidate_ids=[candidate.candidate_id for candidate in group_candidates],
                reuse_key=reuse_keys[encoded_key],
                reuse_across=list(cluster_contract.reuse_across),
                invalidate_on=list(cluster_contract.invalidate_on),
                notes=(
                    "Contract-level dry-run group; hash excludes declared reuse_across "
                    "target-model coordinates such as alpha, lambdas, learning rate, "
                    "and epochs, while preserving induction identity coordinates."
                ),
            )
        )
    return reuse_groups


def _candidate_reuse_key(
    *,
    candidate: CandidateSpec,
    search_space: SearchSpaceSpec,
    reuse_across: set[str],
) -> dict[str, Any]:
    retained_coordinates: dict[str, Any] = {}
    for dimension_name, value in candidate.parameter_values.items():
        dimension = search_space.search_space[dimension_name]
        target_path = dimension.target_path or dimension_name
        if _is_cluster_reuse_across_coordinate(
            dimension_name=dimension_name,
            target_path=target_path,
            reuse_across=reuse_across,
        ):
            continue
        retained_coordinates[dimension_name] = value
    return {
        "dataset": search_space.study.dataset,
        "split_family": search_space.study.split_family,
        "model": search_space.study.model,
        "base_model_config": search_space.base_model_config,
        "artifact_identity_fields": list(search_space.artifact_reuse.cluster_artifacts.invalidate_on)
        if search_space.artifact_reuse is not None and search_space.artifact_reuse.cluster_artifacts is not None
        else [],
        "retained_coordinates": retained_coordinates,
        "retained_stage_overrides": _stage_reuse_overrides(search_space),
    }


def _is_cluster_reuse_across_coordinate(
    *, dimension_name: str, target_path: str, reuse_across: set[str]
) -> bool:
    if is_inner_target_coordinate(target_path):
        return True
    if target_path.startswith("clustering.induction."):
        return False

    target_leaf = target_path.split(".")[-1]
    return dimension_name in reuse_across or target_path in reuse_across or target_leaf in reuse_across


def _stage_reuse_overrides(search_space: SearchSpaceSpec) -> dict[str, Any]:
    if search_space.schedule is None:
        return {}
    retained: dict[str, Any] = {}
    for stage in search_space.schedule.stages:
        retained_stage_overrides = {
            target_path: value
            for target_path, value in stage.overrides.items()
            if not is_inner_target_coordinate(target_path)
        }
        if retained_stage_overrides:
            retained[stage.name] = retained_stage_overrides
    return retained


def _study_id(search_space: SearchSpaceSpec) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", search_space.study.name).strip("_").lower()
    digest = hashlib.sha256(_stable_json(search_space.model_dump(mode="json")).encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}"


def _stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
