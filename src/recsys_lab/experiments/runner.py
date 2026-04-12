from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recsys_lab.config.loader import load_yaml_file


@dataclass(frozen=True, slots=True)
class DryRunPlan:
    experiment_config: str
    dataset_config: str | None
    model_config: str | None
    runtime_config: str
    device_config: str | None
    execution_mode: str = "dry_run"


def build_dry_run_plan(
    *,
    experiment_config: Path,
    runtime_config: Path,
    dataset_config: Path | None = None,
    model_config: Path | None = None,
    device_config: Path | None = None,
) -> dict[str, Any]:
    plan = DryRunPlan(
        experiment_config=str(experiment_config),
        dataset_config=str(dataset_config) if dataset_config else None,
        model_config=str(model_config) if model_config else None,
        runtime_config=str(runtime_config),
        device_config=str(device_config) if device_config else None,
    )
    return {
        "plan": {
            "experiment_config": plan.experiment_config,
            "dataset_config": plan.dataset_config,
            "model_config": plan.model_config,
            "runtime_config": plan.runtime_config,
            "device_config": plan.device_config,
            "execution_mode": plan.execution_mode,
        },
        "loaded_configs": {
            "experiment": load_yaml_file(experiment_config),
            "runtime": load_yaml_file(runtime_config),
            "dataset": load_yaml_file(dataset_config) if dataset_config else None,
            "model": load_yaml_file(model_config) if model_config else None,
            "device": load_yaml_file(device_config) if device_config else None,
        },
    }
