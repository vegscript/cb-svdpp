from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recsys_lab.config.loader import load_yaml_file
from recsys_lab.experiments.common import utc_timestamp
from recsys_lab.tuning.manifests import build_candidate_manifests, build_reuse_summary, build_study_manifest
from recsys_lab.tuning.planner import StudyPlan, build_study_plan
from recsys_lab.tuning.schemas import FidelityStageSpec, SearchSpaceSpec
from recsys_lab.tuning.writers import (
    write_artifact_reuse_summary_csv,
    write_candidate_summary_csv,
    write_tuning_json,
    write_tuning_yaml,
)

REQUIRED_PROMOTION_RESULT_FIELDS = {
    "candidate_id",
    "execution_status",
    "validation_rmse",
    "validation_mae",
    "fit_model_seconds",
    "candidate_config_path",
}


@dataclass(frozen=True)
class PromotionCandidate:
    promoted_candidate_id: str
    source_candidate_id: str
    rank: int
    validation_rmse: float
    validation_mae: float
    fit_model_seconds: float
    source_candidate_config_path: str
    promoted_candidate_config_path: str | None = None


@dataclass(frozen=True)
class PromotionPlan:
    to_stage: str
    stage_overrides: dict[str, Any]
    promoted_candidates: list[PromotionCandidate]
    objective_metric: str
    tie_breakers: list[str]
    claim_boundary: str = "Dry-run staged tuning promotion plan only; no performance or quality claim."


def plan_stage_1_candidates(search_space_spec: SearchSpaceSpec) -> StudyPlan:
    stage = _first_stage(search_space_spec)
    if stage is None:
        return build_study_plan(search_space_spec)
    return build_study_plan(
        search_space_spec,
        stage_name=stage.name,
        max_candidates=stage.max_candidates,
    )


def materialize_stage_candidates(
    plan: StudyPlan,
    output_dir: Path,
    *,
    stage: FidelityStageSpec | None = None,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model_config_payload = _load_base_model_config(plan, repo_root=repo_root)
    stage_overrides = _dotted_overrides_to_nested(stage.overrides if stage is not None else {})
    paths: dict[str, Path] = {}

    paths["search_space"] = write_tuning_yaml(plan.search_space, output_dir / "search_space.yaml")
    paths["study_manifest"] = write_tuning_json(
        build_study_manifest(plan, created_at_utc=utc_timestamp()),
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

    for candidate_manifest in build_candidate_manifests(plan, output_dir=str(output_dir)):
        candidate_dir = output_dir / "candidates" / candidate_manifest.candidate_id
        combined_overrides = _deep_merge_dicts(candidate_manifest.overrides, stage_overrides)
        materialized_config = _strict_deep_merge(base_model_config_payload, combined_overrides)
        candidate_manifest = candidate_manifest.model_copy(
            update={
                "overrides": combined_overrides,
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


def build_promotion_plan(
    previous_stage_results: Path | list[dict[str, Any]],
    next_stage_spec: FidelityStageSpec,
) -> PromotionPlan:
    rows = _load_result_rows(previous_stage_results)
    _validate_result_rows(rows)
    succeeded = [row for row in rows if row.get("execution_status") == "succeeded"]
    ranked = sorted(
        succeeded,
        key=lambda row: (
            _required_float(row, "validation_rmse"),
            _required_float(row, "validation_mae"),
            _required_float(row, "fit_model_seconds"),
        ),
    )
    selected_rows = ranked[: next_stage_spec.max_candidates]
    promoted_candidates = [
        PromotionCandidate(
            promoted_candidate_id=_promoted_candidate_id(row=row, rank=rank, next_stage=next_stage_spec),
            source_candidate_id=str(row["candidate_id"]),
            rank=rank,
            validation_rmse=_required_float(row, "validation_rmse"),
            validation_mae=_required_float(row, "validation_mae"),
            fit_model_seconds=_required_float(row, "fit_model_seconds"),
            source_candidate_config_path=str(row["candidate_config_path"]),
        )
        for rank, row in enumerate(selected_rows, start=1)
    ]
    return PromotionPlan(
        to_stage=next_stage_spec.name,
        stage_overrides=dict(next_stage_spec.overrides),
        promoted_candidates=promoted_candidates,
        objective_metric=next_stage_spec.objective_metric,
        tie_breakers=list(next_stage_spec.tie_breakers),
    )


def materialize_promoted_candidates(promotion_plan: PromotionPlan, output_dir: Path) -> dict[str, Path]:
    promotion_dir = output_dir / "promotions" / promotion_plan.to_stage
    paths: dict[str, Path] = {}
    stage_overrides = _dotted_overrides_to_nested(promotion_plan.stage_overrides)
    updated_candidates: list[PromotionCandidate] = []
    for promoted_candidate in promotion_plan.promoted_candidates:
        source_payload = load_yaml_file(Path(promoted_candidate.source_candidate_config_path))
        materialized_payload = _strict_deep_merge(source_payload, stage_overrides)
        candidate_dir = promotion_dir / "candidates" / promoted_candidate.promoted_candidate_id
        config_path = candidate_dir / "candidate_config.yaml"
        paths[f"promoted_config:{promoted_candidate.promoted_candidate_id}"] = write_tuning_yaml(
            materialized_payload,
            config_path,
        )
        updated_candidates.append(
            PromotionCandidate(
                promoted_candidate_id=promoted_candidate.promoted_candidate_id,
                source_candidate_id=promoted_candidate.source_candidate_id,
                rank=promoted_candidate.rank,
                validation_rmse=promoted_candidate.validation_rmse,
                validation_mae=promoted_candidate.validation_mae,
                fit_model_seconds=promoted_candidate.fit_model_seconds,
                source_candidate_config_path=promoted_candidate.source_candidate_config_path,
                promoted_candidate_config_path=str(config_path),
            )
        )
    updated_plan = PromotionPlan(
        to_stage=promotion_plan.to_stage,
        stage_overrides=promotion_plan.stage_overrides,
        promoted_candidates=updated_candidates,
        objective_metric=promotion_plan.objective_metric,
        tie_breakers=promotion_plan.tie_breakers,
        claim_boundary=promotion_plan.claim_boundary,
    )
    paths["promotion_plan"] = write_tuning_json(
        _promotion_plan_payload(updated_plan),
        promotion_dir / "promotion_plan.json",
    )
    return paths


def _load_result_rows(previous_stage_results: Path | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(previous_stage_results, list):
        return previous_stage_results
    if previous_stage_results.suffix.lower() == ".json":
        payload = json.loads(previous_stage_results.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return payload["rows"]
        raise ValueError("promotion JSON results must be a list or contain a rows list")
    with previous_stage_results.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_result_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        missing = sorted(field for field in REQUIRED_PROMOTION_RESULT_FIELDS if row.get(field) in {None, ""})
        if missing:
            raise ValueError(f"promotion result row missing required fields: {missing}")
        if row.get("test_rmse") not in {None, ""} or row.get("test_mae") not in {None, ""}:
            raise ValueError("test metrics must not be present in promotion result inputs")


def _promotion_plan_payload(plan: PromotionPlan) -> dict[str, Any]:
    return {
        "to_stage": plan.to_stage,
        "stage_overrides": plan.stage_overrides,
        "objective_metric": plan.objective_metric,
        "tie_breakers": plan.tie_breakers,
        "promoted_candidates": [
            {
                "promoted_candidate_id": candidate.promoted_candidate_id,
                "source_candidate_id": candidate.source_candidate_id,
                "rank": candidate.rank,
                "validation_rmse": candidate.validation_rmse,
                "validation_mae": candidate.validation_mae,
                "fit_model_seconds": candidate.fit_model_seconds,
                "source_candidate_config_path": candidate.source_candidate_config_path,
                "promoted_candidate_config_path": candidate.promoted_candidate_config_path,
            }
            for candidate in plan.promoted_candidates
        ],
        "claim_boundary": plan.claim_boundary,
    }


def _promoted_candidate_id(*, row: dict[str, Any], rank: int, next_stage: FidelityStageSpec) -> str:
    encoded = json.dumps(
        {
            "candidate_id": row["candidate_id"],
            "candidate_config_path": row["candidate_config_path"],
            "rank": rank,
            "stage": next_stage.name,
            "stage_overrides": next_stage.overrides,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"prom_{rank:04d}_{hashlib.sha256(encoded).hexdigest()[:12]}"


def _first_stage(search_space_spec: SearchSpaceSpec) -> FidelityStageSpec | None:
    if search_space_spec.schedule is None:
        return None
    return search_space_spec.schedule.stages[0]


def _load_base_model_config(plan: StudyPlan, *, repo_root: Path | None) -> dict[str, Any]:
    base_path = Path(plan.search_space.base_model_config)
    if not base_path.is_absolute() and repo_root is not None:
        base_path = repo_root / base_path
    return load_yaml_file(base_path)


def _dotted_overrides_to_nested(overrides: dict[str, Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for key, value in overrides.items():
        target = nested
        parts = key.split(".")
        for part in parts[:-1]:
            child = target.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"override path conflicts at '{part}'")
            target = child
        target[parts[-1]] = value
    return nested


def _deep_merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _strict_deep_merge(base: dict[str, Any], overrides: dict[str, Any], *, path: str = "") -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in overrides.items():
        current_path = f"{path}.{key}" if path else str(key)
        if key not in merged:
            raise ValueError(f"unknown override field: {current_path}")
        if isinstance(value, dict):
            if not isinstance(merged[key], dict):
                raise ValueError(f"override field is not a mapping in base config: {current_path}")
            merged[key] = _strict_deep_merge(merged[key], value, path=current_path)
        else:
            merged[key] = value
    return merged


def _required_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"missing required numeric field {key!r}")
    return float(value)
