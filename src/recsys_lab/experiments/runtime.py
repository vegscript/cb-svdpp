from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from threadpoolctl import threadpool_limits


@dataclass(frozen=True, slots=True)
class RuntimeThreadingConfig:
    omp_num_threads: int
    blas_threads: int


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
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
            stack.enter_context(
                threadpool_limits(limits=threading_config.omp_num_threads, user_api="openmp")
            )
            stack.enter_context(
                threadpool_limits(limits=threading_config.blas_threads, user_api="blas")
            )
            yield threading_config
