from __future__ import annotations

import math
import random
from typing import Any

from recsys_lab.tuning.candidates import CandidateSpec, _build_candidate
from recsys_lab.tuning.schemas import DimensionSpec, SearchSpaceSpec


def generate_random_candidates(
    search_space: SearchSpaceSpec,
    *,
    study_id: str | None = None,
    stage_name: str | None = None,
) -> list[CandidateSpec]:
    """Generate deterministic seeded random candidates without touching global RNG state."""
    n_candidates = _sampling_candidate_count(search_space)
    rng = random.Random(search_space.generator.seed)
    dimension_names = _dimension_names(search_space)
    candidates: list[CandidateSpec] = []
    for index in range(n_candidates):
        parameter_values = {
            name: _sample_random_dimension(search_space.search_space[name], rng)
            for name in dimension_names
        }
        candidates.append(
            _build_candidate(
                search_space=search_space,
                study_id=study_id or search_space.study.name,
                index=index,
                parameter_values=parameter_values,
                stage_name=stage_name,
            )
        )
    return candidates


def generate_latin_hypercube_candidates(
    search_space: SearchSpaceSpec,
    *,
    study_id: str | None = None,
    stage_name: str | None = None,
) -> list[CandidateSpec]:
    """Generate deterministic Latin-hypercube candidates for numeric dimensions."""
    n_candidates = _sampling_candidate_count(search_space)
    rng = random.Random(search_space.generator.seed)
    dimension_names = _dimension_names(search_space)
    values_by_dimension = {
        name: _latin_hypercube_values(search_space.search_space[name], n_candidates=n_candidates, rng=rng)
        for name in dimension_names
    }
    candidates: list[CandidateSpec] = []
    for index in range(n_candidates):
        parameter_values = {name: values_by_dimension[name][index] for name in dimension_names}
        candidates.append(
            _build_candidate(
                search_space=search_space,
                study_id=study_id or search_space.study.name,
                index=index,
                parameter_values=parameter_values,
                stage_name=stage_name,
            )
        )
    return candidates


def _sampling_candidate_count(search_space: SearchSpaceSpec) -> int:
    if search_space.generator.n_candidates is None:
        raise ValueError("sampling generators require n_candidates")
    return min(search_space.generator.n_candidates, search_space.budget.max_candidates)


def _dimension_names(search_space: SearchSpaceSpec) -> list[str]:
    if search_space.generator.deterministic_order:
        return sorted(search_space.search_space)
    return list(search_space.search_space)


def _sample_random_dimension(dimension: DimensionSpec, rng: random.Random) -> Any:
    if dimension.values is not None:
        return rng.choice(list(dimension.values))
    if dimension.type == "categorical":
        return rng.choice(list(dimension.values or []))
    unit = rng.random()
    return _map_unit_interval(dimension, unit)


def _latin_hypercube_values(
    dimension: DimensionSpec,
    *,
    n_candidates: int,
    rng: random.Random,
) -> list[Any]:
    if dimension.values is not None or dimension.type == "categorical":
        values = list(dimension.values or [])
        offset = rng.randrange(len(values))
        repeated = [values[(index + offset) % len(values)] for index in range(n_candidates)]
        rng.shuffle(repeated)
        return repeated

    unit_values = [(stratum + rng.random()) / n_candidates for stratum in range(n_candidates)]
    rng.shuffle(unit_values)
    return [_map_unit_interval(dimension, unit) for unit in unit_values]


def _map_unit_interval(dimension: DimensionSpec, unit_value: float) -> Any:
    if dimension.low is None or dimension.high is None:
        raise ValueError("numeric sampling dimensions require low and high")
    low = float(dimension.low)
    high = float(dimension.high)
    if dimension.distribution == "loguniform":
        value = math.exp(math.log(low) + unit_value * (math.log(high) - math.log(low)))
    elif dimension.distribution == "uniform":
        value = low + unit_value * (high - low)
    else:
        raise ValueError("numeric sampling dimensions require uniform or loguniform distribution")

    if dimension.type == "int":
        return int(min(max(round(value), math.ceil(low)), math.floor(high)))
    return float(value)
