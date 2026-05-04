from __future__ import annotations

from pathlib import Path
from typing import Any

from recsys_lab.experiments.common import SplitConfig, git_snapshot
from recsys_lab.experiments.unified_runner import build_experiment_services, run_unified_experiment
from recsys_lab.models.biased_mf import BiasedMFConfig
from recsys_lab.models.registry import BiasedMFAdapter, validate_model_config_payload
from recsys_lab.utils.paths import discover_repo_root


# Legacy compatibility wrapper only.
# Do not add experiment lifecycle logic here.
# All execution must delegate to run_unified_experiment.


def _build_biased_mf_config(
    *,
    model_config_payload: dict[str, Any],
    model_seed: int,
    runtime_dtype: str,
) -> BiasedMFConfig:
    _, profile = validate_model_config_payload(model_config_payload, expected_model_name="biased_mf")
    return BiasedMFAdapter.build_model_config(
        profile,
        model_seed=model_seed,
        runtime_dtype=runtime_dtype,
    )


def run_biased_mf_experiment(
    *,
    processed_manifest_path: Path,
    model_config_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    split_config: SplitConfig,
    model_seed: int,
    repo_root: Path | None = None,
    command: str | None = None,
    split_family: str | None = None,
    inner_validation_seed: int | None = None,
    evaluate_test: bool = True,
) -> dict[str, Any]:
    return run_unified_experiment(
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_config=split_config,
        model_seed=model_seed,
        repo_root=(repo_root or discover_repo_root()).resolve(),
        command=command,
        model_name="biased_mf",
        split_family=split_family,
        inner_validation_seed=inner_validation_seed,
        evaluate_test=evaluate_test,
        services=build_experiment_services(git_snapshot_fn=git_snapshot),
    )
