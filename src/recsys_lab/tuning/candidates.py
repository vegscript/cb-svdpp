from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any

from recsys_lab.tuning.schemas import DimensionSpec, SearchSpaceSpec


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    study_id: str
    index: int
    parameter_values: dict[str, Any]
    base_model_config: str
    overrides: dict[str, Any]
    materialized_config_payload: dict[str, Any]
    materialized_config_path: str | None = None
    objective_status: str = "planned"


def generate_candidates(search_space: SearchSpaceSpec, *, study_id: str | None = None) -> list[CandidateSpec]:
    """Materialize deterministic dry-run candidates from a search-space contract."""
    resolved_study_id = study_id or search_space.study.name
    if search_space.generator.type == "manual":
        return _manual_candidates(search_space=search_space, study_id=resolved_study_id)
    return _grid_candidates(search_space=search_space, study_id=resolved_study_id)


def _grid_candidates(search_space: SearchSpaceSpec, *, study_id: str) -> list[CandidateSpec]:
    dimension_names = (
        sorted(search_space.search_space)
        if search_space.generator.deterministic_order
        else list(search_space.search_space)
    )
    dimension_values = [
        _dimension_values(name=name, dimension=search_space.search_space[name]) for name in dimension_names
    ]
    candidates: list[CandidateSpec] = []
    for index, value_tuple in enumerate(itertools.product(*dimension_values)):
        if len(candidates) >= search_space.budget.max_candidates:
            break
        parameter_values = dict(zip(dimension_names, value_tuple, strict=True))
        candidates.append(
            _build_candidate(
                search_space=search_space,
                study_id=study_id,
                index=index,
                parameter_values=parameter_values,
            )
        )
    return candidates


def _manual_candidates(search_space: SearchSpaceSpec, *, study_id: str) -> list[CandidateSpec]:
    candidates: list[CandidateSpec] = []
    manual_candidates = search_space.manual_candidates or []
    dimension_names = set(search_space.search_space)
    for index, parameter_values in enumerate(manual_candidates[: search_space.budget.max_candidates]):
        unknown_dimensions = sorted(set(parameter_values) - dimension_names)
        if unknown_dimensions:
            raise ValueError(f"manual candidate contains unknown dimensions: {unknown_dimensions}")
        missing_dimensions = sorted(dimension_names - set(parameter_values))
        if missing_dimensions:
            raise ValueError(f"manual candidate is missing dimensions: {missing_dimensions}")
        candidates.append(
            _build_candidate(
                search_space=search_space,
                study_id=study_id,
                index=index,
                parameter_values=dict(parameter_values),
            )
        )
    return candidates


def _build_candidate(
    *,
    search_space: SearchSpaceSpec,
    study_id: str,
    index: int,
    parameter_values: dict[str, Any],
) -> CandidateSpec:
    overrides = _coordinates_to_overrides(parameter_values, search_space.search_space)
    materialized_config_payload = {
        "base_model_config": search_space.base_model_config,
        "overrides": overrides,
    }
    return CandidateSpec(
        candidate_id=_candidate_id(
            index=index,
            study_name=search_space.study.name,
            base_model_config=search_space.base_model_config,
            parameter_values=parameter_values,
        ),
        study_id=study_id,
        index=index,
        parameter_values=parameter_values,
        base_model_config=search_space.base_model_config,
        overrides=overrides,
        materialized_config_payload=materialized_config_payload,
    )


def _dimension_values(*, name: str, dimension: DimensionSpec) -> list[Any]:
    if dimension.values is not None:
        return list(dimension.values or [])
    if dimension.type == "categorical":
        return list(dimension.values or [])
    raise ValueError(
        f"numeric grid dimension '{name}' requires explicit values; continuous linspace/logspace sampling is deferred"
    )


def _coordinates_to_overrides(
    coordinates: dict[str, Any], dimensions: dict[str, DimensionSpec]
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for dimension_name, value in coordinates.items():
        target_path = dimensions[dimension_name].target_path or dimension_name
        _set_nested_override(overrides, target_path.split("."), value)
    return overrides


def _set_nested_override(payload: dict[str, Any], path_parts: list[str], value: Any) -> None:
    target = payload
    for part in path_parts[:-1]:
        child = target.get(part)
        if child is None:
            child = {}
            target[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"override path conflicts at '{part}'")
        target = child
    target[path_parts[-1]] = value


def _candidate_id(
    *,
    index: int,
    study_name: str,
    base_model_config: str,
    parameter_values: dict[str, Any],
) -> str:
    encoded = json.dumps(
        {
            "base_model_config": base_model_config,
            "parameter_values": parameter_values,
            "study_name": study_name,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    return f"cand_{index:04d}_{digest}"
