from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from recsys_lab.config.loader import load_yaml_file
from recsys_lab.models.registry import validate_model_config_payload
from recsys_lab.tuning import SearchSpaceSpec, build_study_plan

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_TUNING_DIR = REPO_ROOT / "configs" / "experiments" / "tuning" / "active"


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def test_active_tuning_candidate_overrides_validate_as_model_profiles() -> None:
    active_configs = sorted(ACTIVE_TUNING_DIR.glob("*.yaml"))
    assert active_configs, "expected active tuning configs"

    for tuning_config_path in active_configs:
        tuning_config = load_yaml_file(tuning_config_path)
        base_model_config_ref = tuning_config["base_model_config"]
        assert isinstance(base_model_config_ref, str)

        base_model_config_path = REPO_ROOT / base_model_config_ref
        assert base_model_config_path.exists(), f"{tuning_config_path} references missing base_model_config"
        base_model_config = load_yaml_file(base_model_config_path)
        base_model_name = str(base_model_config["model"]["name"])

        if tuning_config.get("search_space_version") == "tuning_search_space_v1":
            search_space = SearchSpaceSpec.model_validate(tuning_config)
            plan = build_study_plan(search_space)
            candidates = [
                {
                    "candidate_id": candidate.candidate_id,
                    "overrides": candidate.overrides,
                }
                for candidate in plan.candidates
            ]
        else:
            candidates = tuning_config["candidates"]
        assert isinstance(candidates, list)
        assert candidates, f"{tuning_config_path} must define at least one candidate"

        for candidate in candidates:
            assert isinstance(candidate, dict)
            candidate_id = candidate["candidate_id"]
            overrides = candidate.get("overrides", {})
            assert isinstance(overrides, dict)

            candidate_payload = _deep_merge(base_model_config, overrides)
            validate_model_config_payload(candidate_payload, expected_model_name=base_model_name)

            assert candidate_payload["model"]["name"] == base_model_name, (
                f"{tuning_config_path} candidate {candidate_id} changed the model identity"
            )
