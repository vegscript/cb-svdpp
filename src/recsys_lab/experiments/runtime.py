from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from threadpoolctl import threadpool_limits

DEVICE_PROFILE_CONTRACT_VERSION = "device_profile_contract_v1"
_DRAFT_PROFILE_STATUSES = {"draft", "template", "placeholder"}
_PLACEHOLDER_STRINGS = {"", "override_per_cluster", "override", "placeholder", "todo", "tbd", "unknown", "none"}


@dataclass(frozen=True, slots=True)
class RuntimeThreadingConfig:
    omp_num_threads: int
    blas_threads: int


def _mapping(payload: dict[str, Any], *, key: str, blocking_reasons: list[str]) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        blocking_reasons.append(f"{key} must be a mapping")
        return {}
    return value


def _is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _PLACEHOLDER_STRINGS
    return False


def _require_concrete_string(payload: dict[str, Any], *, path: str, blocking_reasons: list[str]) -> str | None:
    key = path.rsplit(".", maxsplit=1)[-1]
    value = payload.get(key)
    if _is_placeholder(value):
        blocking_reasons.append(f"{path} must be set to a concrete value")
        return None
    return str(value)


def _require_positive_int(payload: dict[str, Any], *, path: str, blocking_reasons: list[str]) -> int | None:
    key = path.rsplit(".", maxsplit=1)[-1]
    value = payload.get(key)
    if value is None or isinstance(value, bool):
        blocking_reasons.append(f"{path} must be set to a positive integer")
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        blocking_reasons.append(f"{path} must be set to a positive integer")
        return None
    if parsed < 1:
        blocking_reasons.append(f"{path} must be >= 1")
        return None
    return parsed


def _require_positive_number(payload: dict[str, Any], *, path: str, blocking_reasons: list[str]) -> float | None:
    key = path.rsplit(".", maxsplit=1)[-1]
    value = payload.get(key)
    if value is None or isinstance(value, bool):
        blocking_reasons.append(f"{path} must be set to a positive number")
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        blocking_reasons.append(f"{path} must be set to a positive number")
        return None
    if parsed <= 0.0:
        blocking_reasons.append(f"{path} must be > 0")
        return None
    return parsed


def _parse_positive_thread_count(threading_payload: dict[str, Any], *, key: str) -> int:
    value = threading_payload.get(key)
    if value is None:
        raise ValueError(f"device profile threading.{key} must be set to a positive integer")
    if isinstance(value, bool):
        raise ValueError(f"device profile threading.{key} must be an integer, not boolean")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"device profile threading.{key} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"device profile threading.{key} must be >= 1")
    return parsed


def assess_device_profile_contract(*, device_config_payload: dict[str, Any]) -> dict[str, Any]:
    blocking_reasons: list[str] = []
    metadata = _mapping(device_config_payload, key="metadata", blocking_reasons=blocking_reasons)
    profile = _mapping(device_config_payload, key="device_profile", blocking_reasons=blocking_reasons)
    storage = _mapping(device_config_payload, key="storage", blocking_reasons=blocking_reasons)
    threading = _mapping(device_config_payload, key="threading", blocking_reasons=blocking_reasons)
    precision = _mapping(device_config_payload, key="precision", blocking_reasons=blocking_reasons)
    resource_limits = _mapping(device_config_payload, key="resource_limits", blocking_reasons=blocking_reasons)

    metadata_status = str(metadata.get("status", "missing")).strip().lower()
    if metadata_status in _DRAFT_PROFILE_STATUSES:
        blocking_reasons.append("metadata.status must not be draft/template/placeholder for claim-eligible runs")

    profile_name = _require_concrete_string(profile, path="device_profile.name", blocking_reasons=blocking_reasons)
    compute_class = _require_concrete_string(
        profile,
        path="device_profile.compute_class",
        blocking_reasons=blocking_reasons,
    )
    _require_concrete_string(profile, path="device_profile.cpu_model", blocking_reasons=blocking_reasons)
    _require_positive_int(profile, path="device_profile.logical_threads", blocking_reasons=blocking_reasons)
    _require_positive_int(profile, path="device_profile.physical_cores", blocking_reasons=blocking_reasons)
    _require_positive_number(profile, path="device_profile.ram_gb", blocking_reasons=blocking_reasons)

    if not isinstance(profile.get("gpu_enabled"), bool):
        blocking_reasons.append("device_profile.gpu_enabled must be set to a boolean")

    _require_concrete_string(storage, path="storage.cache_preference", blocking_reasons=blocking_reasons)
    _require_concrete_string(storage, path="storage.archive_preference", blocking_reasons=blocking_reasons)

    _require_positive_int(threading, path="threading.omp_num_threads", blocking_reasons=blocking_reasons)
    _require_positive_int(threading, path="threading.blas_threads", blocking_reasons=blocking_reasons)

    default_dtype = _require_concrete_string(
        precision,
        path="precision.default_dtype",
        blocking_reasons=blocking_reasons,
    )
    reference_dtype = _require_concrete_string(
        precision,
        path="precision.reference_dtype",
        blocking_reasons=blocking_reasons,
    )
    if default_dtype not in {None, "float32", "float64"}:
        blocking_reasons.append("precision.default_dtype must be float32 or float64")
    if reference_dtype not in {None, "float32", "float64"}:
        blocking_reasons.append("precision.reference_dtype must be float32 or float64")

    ram_guardrail_fraction = _require_positive_number(
        resource_limits,
        path="resource_limits.ram_guardrail_fraction",
        blocking_reasons=blocking_reasons,
    )
    if ram_guardrail_fraction is not None and not 0.0 < ram_guardrail_fraction <= 1.0:
        blocking_reasons.append("resource_limits.ram_guardrail_fraction must be in (0, 1]")
        ram_guardrail_fraction = None

    return {
        "contract_version": DEVICE_PROFILE_CONTRACT_VERSION,
        "profile_name": profile_name or "unknown",
        "compute_class": compute_class or "unknown",
        "metadata_status": metadata_status or "missing",
        "claim_eligible": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "ram_guardrail_fraction": ram_guardrail_fraction,
    }


def validate_claim_eligible_device_profile(*, device_config_payload: dict[str, Any]) -> dict[str, Any]:
    assessment = assess_device_profile_contract(device_config_payload=device_config_payload)
    if not assessment["claim_eligible"]:
        reasons = "; ".join(str(reason) for reason in assessment["blocking_reasons"])
        raise ValueError(f"device profile is not claim-eligible: {reasons}")
    return assessment


def resolve_runtime_threading_config(
    *,
    device_config_payload: dict[str, Any],
) -> RuntimeThreadingConfig:
    threading_payload = device_config_payload.get("threading")
    if not isinstance(threading_payload, dict):
        raise ValueError("device profile must define a threading mapping")
    return RuntimeThreadingConfig(
        omp_num_threads=_parse_positive_thread_count(threading_payload, key="omp_num_threads"),
        blas_threads=_parse_positive_thread_count(threading_payload, key="blas_threads"),
    )


@contextmanager
def _temporary_environment(overrides: dict[str, str]) -> Iterator[None]:
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, previous_value in previous.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value


@contextmanager
def runtime_execution_context(
    *,
    threading_config: RuntimeThreadingConfig,
) -> Iterator[RuntimeThreadingConfig]:
    env_overrides = {
        "OMP_NUM_THREADS": str(threading_config.omp_num_threads),
        "MKL_NUM_THREADS": str(threading_config.blas_threads),
        "OPENBLAS_NUM_THREADS": str(threading_config.blas_threads),
        "NUMEXPR_NUM_THREADS": str(threading_config.blas_threads),
    }
    with _temporary_environment(env_overrides):
        with ExitStack() as stack:
            stack.enter_context(threadpool_limits(limits=threading_config.omp_num_threads, user_api="openmp"))
            stack.enter_context(threadpool_limits(limits=threading_config.blas_threads, user_api="blas"))
            yield threading_config
