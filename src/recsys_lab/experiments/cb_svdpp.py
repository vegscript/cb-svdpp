from __future__ import annotations

from pathlib import Path
from typing import Any

from recsys_lab.experiments.common import SplitConfig
from recsys_lab.experiments.unified_runner import run_unified_experiment
from recsys_lab.models.biased_mf import BiasedMFConfig
from recsys_lab.models.cb_svdpp import CBSVDppConfig
from recsys_lab.models.registry import CBSVDppAdapter, validate_model_config_payload
from recsys_lab.utils.paths import discover_repo_root

# Legacy compatibility wrapper only.
# Do not add experiment lifecycle logic here.
# All execution must delegate to run_unified_experiment.


def _build_cb_svdpp_config(
    *,
    model_config_payload: dict[str, Any],
    model_seed: int,
    runtime_dtype: str,
) -> CBSVDppConfig:
    _, profile = validate_model_config_payload(model_config_payload, expected_model_name="cb_svdpp")
    return CBSVDppAdapter.build_model_config(
        profile,
        model_seed=model_seed,
        runtime_dtype=runtime_dtype,
    )


def _build_induction_config(
    *,
    cb_config: CBSVDppConfig,
    model_seed: int,
) -> BiasedMFConfig:
    return CBSVDppAdapter.build_induction_config(cb_config, model_seed=model_seed)


def run_cb_svdpp_experiment(
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
    use_split_cache: bool | None = None,
    reuse_precomputed_indices: bool = True,
    use_training_index_cache: bool = False,
    use_cluster_artifact_cache: bool = False,
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
        model_name="cb_svdpp",
        split_family=split_family,
        inner_validation_seed=inner_validation_seed,
        evaluate_test=evaluate_test,
        use_split_cache=use_split_cache,
        reuse_precomputed_indices=reuse_precomputed_indices,
        use_training_index_cache=use_training_index_cache,
        use_cluster_artifact_cache=use_cluster_artifact_cache,
    )
