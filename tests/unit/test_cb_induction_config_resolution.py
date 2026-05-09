from __future__ import annotations

import copy
from pathlib import Path

from recsys_lab.config.loader import load_yaml_file
from recsys_lab.models.registry import validate_model_config_payload


def _build_induction_config(payload: dict[str, object], *, model_name: str):
    adapter, profile = validate_model_config_payload(payload, expected_model_name=model_name)
    model_config = adapter.build_model_config(profile, model_seed=99, runtime_dtype="float32")
    return adapter.build_induction_config(model_config, model_seed=99, model_profile=profile)


def test_cb_svdpp_induction_config_is_separate_from_target_learning_rate() -> None:
    payload = load_yaml_file(Path("configs/models/cb_svdpp.yaml"))
    payload["training"]["learning_rate"] = 0.123  # type: ignore[index]

    induction_config = _build_induction_config(payload, model_name="cb_svdpp")

    assert induction_config is not None
    assert induction_config.learning_rate == payload["clustering"]["induction"]["learning_rate"]  # type: ignore[index]
    assert induction_config.learning_rate != payload["training"]["learning_rate"]  # type: ignore[index]


def test_cb_svdpp_induction_config_is_separate_from_target_lambda_q() -> None:
    payload = load_yaml_file(Path("configs/models/cb_svdpp.yaml"))
    payload["training"]["lambda_q"] = 0.456  # type: ignore[index]

    induction_config = _build_induction_config(payload, model_name="cb_svdpp")

    assert induction_config is not None
    assert induction_config.lambda_q == payload["clustering"]["induction"]["lambda_q"]  # type: ignore[index]
    assert induction_config.lambda_q != payload["training"]["lambda_q"]  # type: ignore[index]


def test_cb_asvdpp_uses_same_induction_config_resolution_contract() -> None:
    payload = load_yaml_file(Path("configs/models/cb_asvdpp.yaml"))
    modified = copy.deepcopy(payload)
    modified["training"]["learning_rate"] = 0.123  # type: ignore[index]
    modified["training"]["lambda_q"] = 0.456  # type: ignore[index]

    induction_config = _build_induction_config(modified, model_name="cb_asvdpp")

    assert induction_config is not None
    assert induction_config.learning_rate == payload["clustering"]["induction"]["learning_rate"]  # type: ignore[index]
    assert induction_config.lambda_q == payload["clustering"]["induction"]["lambda_q"]  # type: ignore[index]
    assert induction_config.learning_rate != modified["training"]["learning_rate"]  # type: ignore[index]
    assert induction_config.lambda_q != modified["training"]["lambda_q"]  # type: ignore[index]
