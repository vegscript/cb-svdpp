from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recsys_lab.data.splitters import RatingsSplit
from recsys_lab.experiments.common import SplitConfig
from recsys_lab.experiments.runtime import RuntimeThreadingConfig
from recsys_lab.experiments.split_cache import SplitCacheResult
from recsys_lab.models.registry import FitArtifacts, ModelAdapter


@dataclass(frozen=True, slots=True)
class UnifiedRunInputs:
    processed_manifest_path: Path
    model_config_path: Path
    runtime_config_path: Path
    device_config_path: Path
    split_config: SplitConfig
    model_seed: int
    model_name: str | None
    split_family: str | None
    inner_validation_seed: int | None
    evaluate_test: bool
    use_split_cache: bool | None
    reuse_precomputed_indices: bool
    use_training_index_cache: bool
    use_cluster_artifact_cache: bool


@dataclass(frozen=True, slots=True)
class ResolvedExperimentConfig:
    repo_root: Path
    processed_manifest_path: Path
    model_config_path: Path
    runtime_config_path: Path
    device_config_path: Path
    runtime_config_payload: dict[str, Any]
    device_config_payload: dict[str, Any]
    threading_config: RuntimeThreadingConfig
    processed_manifest: dict[str, Any]
    dataset_short_name: str
    requested_split_family: str
    raw_model_config_payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RunPaths:
    run_id: str
    run_dir: Path
    config_snapshot_path: Path
    metrics_path: Path
    performance_profile_path: Path
    kernel_profile_path: Path
    stdout_log_path: Path
    run_manifest_path: Path


@dataclass(frozen=True, slots=True)
class ResolvedModelProfile:
    adapter: type[ModelAdapter]
    model_profile: Any
    runtime_dtype: str
    model_config: object | None
    induction_config: object | None
    cb_semantics: dict[str, Any] | None
    model_requirements: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ResolvedSplitBundle:
    ratings_data: Any
    split_result: SplitCacheResult
    split: RatingsSplit
    train_data: Any
    validation_data: Any | None
    test_data: Any
    split_id_for_cache: str


@dataclass(frozen=True, slots=True)
class FitArtifactResolution:
    artifacts: FitArtifacts
    user_history_result: Any | None
    explicit_feedback_result: Any | None
    cluster_artifact_result: Any | None
    cluster_history_result: Any | None
    induction_config: object | None
    cluster_induction_wall_clock_seconds: float


@dataclass(frozen=True, slots=True)
class ModelExecutionResult:
    model: object
    kernel_profile_payload: dict[str, Any]
    train_predictions: Any
    validation_predictions: Any | None
    test_predictions: Any | None
    training_wall_clock_seconds: float
    inference_wall_clock_seconds: float
    peak_memory_bytes: int
    baseline_memory_bytes: int


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    metrics_payload: dict[str, Any]
    rating_metrics_payload: dict[str, Any]
    caches_payload: dict[str, Any]
    model_payload: dict[str, Any]
    system_metrics: dict[str, Any]
