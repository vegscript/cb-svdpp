from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from recsys_lab.config.loader import load_yaml_file
from recsys_lab.experiments.runtime import (
    assess_device_profile_contract,
    resolve_runtime_threading_config,
    runtime_execution_context,
    validate_claim_eligible_device_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_resolve_runtime_threading_config_rejects_missing_values() -> None:
    with pytest.raises(ValueError, match="omp_num_threads"):
        resolve_runtime_threading_config(
            device_config_payload={
                "threading": {
                    "omp_num_threads": None,
                    "blas_threads": 2,
                }
            }
        )


def test_claim_eligible_device_profile_rejects_hpc_template_placeholders() -> None:
    payload = load_yaml_file(REPO_ROOT / "configs" / "runtime" / "devices" / "hpc_cpu.yaml")

    assessment = assess_device_profile_contract(device_config_payload=payload)

    assert assessment["claim_eligible"] is False
    assert "metadata.status must not be draft/template/placeholder for claim-eligible runs" in assessment[
        "blocking_reasons"
    ]
    assert "device_profile.cpu_model must be set to a concrete value" in assessment["blocking_reasons"]
    assert "device_profile.logical_threads must be set to a positive integer" in assessment["blocking_reasons"]
    assert "device_profile.physical_cores must be set to a positive integer" in assessment["blocking_reasons"]
    assert "device_profile.ram_gb must be set to a positive number" in assessment["blocking_reasons"]
    assert "threading.omp_num_threads must be set to a positive integer" in assessment["blocking_reasons"]
    assert "threading.blas_threads must be set to a positive integer" in assessment["blocking_reasons"]

    with pytest.raises(ValueError, match="device profile is not claim-eligible"):
        validate_claim_eligible_device_profile(device_config_payload=payload)


def test_local_device_profile_is_claim_eligible_reference_profile() -> None:
    payload = load_yaml_file(REPO_ROOT / "configs" / "runtime" / "devices" / "local_i5_2500k_24gb.yaml")

    assessment = validate_claim_eligible_device_profile(device_config_payload=payload)

    assert assessment["claim_eligible"] is True
    assert assessment["profile_name"] == "local_i5_2500k_24gb"
    assert assessment["compute_class"] == "local_cpu"
    assert assessment["metadata_status"] == "validated_local_reference"
    assert assessment["blocking_reasons"] == []
    assert assessment["ram_guardrail_fraction"] == 0.8


def test_local_laptop_device_profile_is_claim_eligible_reference_profile() -> None:
    payload = load_yaml_file(REPO_ROOT / "configs" / "runtime" / "devices" / "local_u300_24gb.yaml")

    assessment = validate_claim_eligible_device_profile(device_config_payload=payload)
    threading_config = resolve_runtime_threading_config(device_config_payload=payload)

    assert assessment["claim_eligible"] is True
    assert assessment["profile_name"] == "local_u300_24gb"
    assert assessment["compute_class"] == "local_cpu"
    assert assessment["metadata_status"] == "validated_local_laptop"
    assert assessment["blocking_reasons"] == []
    assert assessment["ram_guardrail_fraction"] == 0.8
    assert threading_config.omp_num_threads == 6
    assert threading_config.blas_threads == 6


def test_runtime_execution_context_sets_and_restores_env(monkeypatch) -> None:
    calls: list[tuple[int, str | None, str, str]] = []

    @contextmanager
    def _fake_threadpool_limits(*, limits: int, user_api: str | None = None):
        calls.append(
            (
                limits,
                user_api,
                os.environ.get("OMP_NUM_THREADS", ""),
                os.environ.get("MKL_NUM_THREADS", ""),
            )
        )
        yield

    monkeypatch.setenv("OMP_NUM_THREADS", "9")
    monkeypatch.setenv("MKL_NUM_THREADS", "7")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "6")
    monkeypatch.delenv("NUMEXPR_NUM_THREADS", raising=False)
    monkeypatch.setattr("recsys_lab.experiments.runtime.threadpool_limits", _fake_threadpool_limits)

    threading_config = resolve_runtime_threading_config(
        device_config_payload={
            "threading": {
                "omp_num_threads": 4,
                "blas_threads": 2,
            }
        }
    )

    with runtime_execution_context(threading_config=threading_config):
        assert os.environ.get("OMP_NUM_THREADS") == "4"
        assert os.environ.get("MKL_NUM_THREADS") == "2"
        assert os.environ.get("OPENBLAS_NUM_THREADS") == "2"
        assert os.environ.get("NUMEXPR_NUM_THREADS") == "2"

    assert os.environ.get("OMP_NUM_THREADS") == "9"
    assert os.environ.get("MKL_NUM_THREADS") == "7"
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "6"
    assert os.environ.get("NUMEXPR_NUM_THREADS") is None
    assert calls == [
        (4, "openmp", "4", "2"),
        (2, "blas", "4", "2"),
    ]
