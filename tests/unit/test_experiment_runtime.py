from __future__ import annotations

from contextlib import contextmanager
import os

import pytest

from recsys_lab.experiments.runtime import (
    resolve_runtime_threading_config,
    runtime_execution_context,
)


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


def test_runtime_execution_context_sets_and_restores_env(monkeypatch) -> None:
    calls: list[tuple[int, str | None, str, str]] = []

    @contextmanager
    def _fake_threadpool_limits(*, limits: int, user_api: str | None = None):
        calls.append((limits, user_api, os.environ.get("OMP_NUM_THREADS", ""), os.environ.get("MKL_NUM_THREADS", "")))
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
