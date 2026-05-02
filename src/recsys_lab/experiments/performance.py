from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter, sleep
from typing import Any, Iterator

import numpy as np
import psutil

_BYTES_PER_MB = 1024.0 * 1024.0
_PRIMITIVE_TYPES = (str, bytes, int, float, bool, complex, Path, type(None))


class PeakMemoryMonitor:
    def __init__(self, *, poll_interval_seconds: float = 0.01) -> None:
        self._poll_interval_seconds = poll_interval_seconds
        self._process = psutil.Process()
        self._stop_event = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self.start_rss_bytes = 0
        self.peak_rss_bytes = 0

    def _sample(self) -> None:
        rss_bytes = int(self._process.memory_info().rss)
        with self._lock:
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._sample()
            sleep(self._poll_interval_seconds)

    def __enter__(self) -> "PeakMemoryMonitor":
        self.start_rss_bytes = int(self._process.memory_info().rss)
        self.peak_rss_bytes = self.start_rss_bytes
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="peak-memory-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()


class StageProfiler:
    def __init__(self) -> None:
        self._process = psutil.Process()
        self._stages: list[dict[str, Any]] = []

    def _rss_mb(self) -> float:
        return float(self._process.memory_info().rss) / _BYTES_PER_MB

    @contextmanager
    def stage(self, name: str, *, metadata: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        if not name:
            raise ValueError("stage name must not be empty")
        stage_metadata: dict[str, Any] = {} if metadata is None else dict(metadata)
        start = perf_counter()
        rss_start_mb = self._rss_mb()
        status = "completed"
        exception_type: str | None = None
        try:
            yield stage_metadata
        except Exception as exc:
            status = "failed"
            exception_type = type(exc).__name__
            raise
        finally:
            rss_end_mb = self._rss_mb()
            record: dict[str, Any] = {
                "name": name,
                "status": status,
                "wall_clock_seconds": float(perf_counter() - start),
                "rss_start_mb": rss_start_mb,
                "rss_end_mb": rss_end_mb,
                "rss_delta_mb": rss_end_mb - rss_start_mb,
                "metadata": stage_metadata,
            }
            if exception_type is not None:
                record["exception_type"] = exception_type
            self._stages.append(record)

    def to_payload(self) -> dict[str, Any]:
        stages = [dict(stage) for stage in self._stages]
        return {
            "profile_version": "stage_profile_v1",
            "stages": stages,
            "stage_count": len(stages),
            "total_profiled_wall_clock_seconds": float(
                sum(float(stage["wall_clock_seconds"]) for stage in stages)
            ),
        }


def estimate_numpy_payload_nbytes(root: object) -> int:
    total_bytes = 0
    visited: set[int] = set()

    def walk(obj: object) -> None:
        nonlocal total_bytes

        if isinstance(obj, _PRIMITIVE_TYPES + (np.generic,)):
            return

        object_id = id(obj)
        if object_id in visited:
            return
        visited.add(object_id)

        if isinstance(obj, np.ndarray):
            total_bytes += int(obj.nbytes)
            return

        if is_dataclass(obj) and not isinstance(obj, type):
            for field in fields(obj):
                walk(getattr(obj, field.name))
            return

        if isinstance(obj, dict):
            for key, value in obj.items():
                walk(key)
                walk(value)
            return

        if isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                walk(item)
            return

        if hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                walk(value)

    walk(root)
    return total_bytes


def build_system_metrics(
    *,
    train_rows: int,
    epochs: int,
    training_wall_clock_seconds: float,
    inference_rows: int,
    inference_wall_clock_seconds: float,
    peak_memory_bytes: int,
    baseline_memory_bytes: int,
    model: object,
    epoch_durations_seconds: list[float] | None,
    train_time_total_seconds: float | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be positive for system metrics")

    total_train_ratings = int(train_rows) * int(epochs)
    model_size_mb = estimate_numpy_payload_nbytes(model) / _BYTES_PER_MB
    peak_memory_mb = float(peak_memory_bytes) / _BYTES_PER_MB
    peak_memory_delta_mb = max(float(peak_memory_bytes - baseline_memory_bytes) / _BYTES_PER_MB, 0.0)
    train_time_total = (
        float(training_wall_clock_seconds) if train_time_total_seconds is None else float(train_time_total_seconds)
    )
    epoch_durations = [] if epoch_durations_seconds is None else [float(value) for value in epoch_durations_seconds]

    payload: dict[str, Any] = {
        "train_time_total": train_time_total,
        "train_time_per_epoch": float(training_wall_clock_seconds) / float(epochs),
        "train_ratings_processed": total_train_ratings,
        "ratings_per_second_train": (
            float(total_train_ratings) / float(training_wall_clock_seconds)
            if training_wall_clock_seconds > 0.0
            else 0.0
        ),
        "inference_wall_clock_seconds": float(inference_wall_clock_seconds),
        "inference_rows": int(inference_rows),
        "ratings_per_second_inference": (
            float(inference_rows) / float(inference_wall_clock_seconds) if inference_wall_clock_seconds > 0.0 else 0.0
        ),
        "peak_memory_mb": peak_memory_mb,
        "peak_memory_delta_mb": peak_memory_delta_mb,
        "model_size_mb": model_size_mb,
        "epoch_durations_seconds": epoch_durations,
    }
    if extra_fields:
        payload.update(extra_fields)
    return payload
