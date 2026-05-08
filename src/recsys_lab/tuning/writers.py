from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from recsys_lab.config.loader import dump_yaml_file, load_yaml_file
from recsys_lab.experiments.common import utc_timestamp
from recsys_lab.tuning.manifests import (
    CandidateManifest,
    build_candidate_manifests,
    build_reuse_summary,
    build_study_manifest,
)
from recsys_lab.tuning.planner import StudyPlan


def write_tuning_json(payload: BaseModel | dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    return path


def write_tuning_yaml(payload: BaseModel | dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_yaml_file(path, _jsonable(payload))
    return path


def write_candidate_summary_csv(plan: StudyPlan, path: Path, *, output_dir: Path | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_dir = output_dir if output_dir is not None else path.parent.parent
    group_ids_by_candidate = _cluster_group_ids_by_candidate(plan)
    rows = [
        {
            "candidate_id": candidate.candidate_id,
            "candidate_index": candidate.index,
            "study_id": plan.study_id,
            "model": plan.search_space.study.model,
            "dataset": plan.search_space.study.dataset,
            "split_family": plan.search_space.study.split_family,
            "alpha": _candidate_value(candidate.parameter_values, "alpha"),
            "learning_rate": _candidate_value(candidate.parameter_values, "learning_rate"),
            "latent_dim": _candidate_value(candidate.parameter_values, "latent_dim"),
            "epochs": _candidate_value(candidate.parameter_values, "epochs"),
            "cluster_reuse_group_id": group_ids_by_candidate.get(candidate.candidate_id, ""),
            "candidate_config_path": str(_candidate_config_path(resolved_output_dir, candidate.candidate_id)),
            "candidate_manifest_path": str(_candidate_manifest_path(resolved_output_dir, candidate.candidate_id)),
            "status": candidate.objective_status,
        }
        for candidate in plan.candidates
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "candidate_index",
                "study_id",
                "model",
                "dataset",
                "split_family",
                "alpha",
                "learning_rate",
                "latent_dim",
                "epochs",
                "cluster_reuse_group_id",
                "candidate_config_path",
                "candidate_manifest_path",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_artifact_reuse_summary_csv(plan: StudyPlan, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "study_id": plan.study_id,
            "artifact_type": group.artifact_type,
            "reuse_group_id": group.group_id,
            "candidate_count": len(group.candidate_ids),
            "candidate_ids": "|".join(group.candidate_ids),
            "reuse_across": "|".join(group.reuse_across),
            "invalidate_on": "|".join(group.invalidate_on),
            "notes": group.notes,
        }
        for group in plan.artifact_reuse_groups
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "study_id",
                "artifact_type",
                "reuse_group_id",
                "candidate_count",
                "candidate_ids",
                "reuse_across",
                "invalidate_on",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_study_plan(plan: StudyPlan, output_dir: Path, *, repo_root: Path | None = None) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    base_model_config_payload = _load_base_model_config(plan, repo_root=repo_root)
    created_at_utc = utc_timestamp()

    paths["search_space"] = write_tuning_yaml(plan.search_space, output_dir / "search_space.yaml")
    paths["study_manifest"] = write_tuning_json(
        build_study_manifest(plan, created_at_utc=created_at_utc),
        output_dir / "study_manifest.json",
    )
    paths["artifact_reuse_summary"] = write_tuning_json(
        build_reuse_summary(plan),
        output_dir / "reports" / "artifact_reuse_summary.json",
    )
    paths["artifact_reuse_summary_csv"] = write_artifact_reuse_summary_csv(
        plan,
        output_dir / "reports" / "artifact_reuse_summary.csv",
    )
    paths["candidate_summary"] = write_candidate_summary_csv(
        plan,
        output_dir / "reports" / "candidate_summary.csv",
        output_dir=output_dir,
    )

    candidates_dir = output_dir / "candidates"
    for candidate_manifest in build_candidate_manifests(plan, output_dir=str(output_dir)):
        candidate_dir = candidates_dir / candidate_manifest.candidate_id
        materialized_config = materialize_candidate_config(
            base_model_config_payload=base_model_config_payload,
            candidate_manifest=candidate_manifest,
        )
        candidate_manifest = candidate_manifest.model_copy(
            update={
                "materialized_config_payload": materialized_config,
                "candidate_config_path": str(candidate_dir / "candidate_config.yaml"),
            }
        )
        paths[f"candidate_manifest:{candidate_manifest.candidate_id}"] = write_tuning_json(
            candidate_manifest,
            candidate_dir / "candidate_manifest.json",
        )
        paths[f"candidate_config:{candidate_manifest.candidate_id}"] = write_tuning_yaml(
            materialized_config,
            candidate_dir / "candidate_config.yaml",
        )
    return paths


def materialize_candidate_config(
    *,
    base_model_config_payload: dict[str, Any],
    candidate_manifest: CandidateManifest,
) -> dict[str, Any]:
    return _strict_deep_merge(base_model_config_payload, candidate_manifest.overrides)


def _load_base_model_config(plan: StudyPlan, *, repo_root: Path | None) -> dict[str, Any]:
    base_path = Path(plan.search_space.base_model_config)
    if not base_path.is_absolute() and repo_root is not None:
        base_path = repo_root / base_path
    return load_yaml_file(base_path)


def _strict_deep_merge(base: dict[str, Any], overrides: dict[str, Any], *, path: str = "") -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        current_path = f"{path}.{key}" if path else str(key)
        if key not in merged:
            raise ValueError(f"unknown candidate override field: {current_path}")
        if isinstance(value, dict):
            if not isinstance(merged[key], dict):
                raise ValueError(f"candidate override field is not a mapping in base config: {current_path}")
            merged[key] = _strict_deep_merge(merged[key], value, path=current_path)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _jsonable(payload: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    return payload


def _candidate_value(parameter_values: dict[str, Any], field_name: str) -> Any:
    if field_name in parameter_values:
        return parameter_values[field_name]
    suffix = f".{field_name}"
    for key, value in parameter_values.items():
        if key.endswith(suffix):
            return value
    return ""


def _cluster_group_ids_by_candidate(plan: StudyPlan) -> dict[str, str]:
    result: dict[str, str] = {}
    for group in plan.artifact_reuse_groups:
        if group.artifact_type != "cluster_artifacts":
            continue
        for candidate_id in group.candidate_ids:
            result[candidate_id] = group.group_id
    return result


def _candidate_config_path(output_dir: Path, candidate_id: str) -> Path:
    return output_dir / "candidates" / candidate_id / "candidate_config.yaml"


def _candidate_manifest_path(output_dir: Path, candidate_id: str) -> Path:
    return output_dir / "candidates" / candidate_id / "candidate_manifest.json"
