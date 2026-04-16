from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from recsys_lab.config.loader import dump_yaml_file, load_yaml_file
from recsys_lab.experiments.benchmarking import (
    build_benchmark_measurement,
    summarize_scalar_samples,
)
from recsys_lab.experiments.common import (
    SplitConfig,
    build_runtime_metadata,
    git_snapshot,
    resolve_runtime_dtype,
    seed_slug,
    split_id,
    utc_timestamp,
    write_json,
    write_log,
)
from recsys_lab.experiments.runtime import (
    resolve_runtime_threading_config,
    runtime_execution_context,
)
from recsys_lab.utils.manifests import load_json_file, validate_manifest_file
from recsys_lab.utils.paths import discover_repo_root, repo_path_string


def _read_run_manifest(run_manifest_path: Path) -> dict[str, Any]:
    payload = load_json_file(run_manifest_path)
    if payload.get("kind") != "run_manifest":
        raise ValueError(f"expected run_manifest at {run_manifest_path}")
    if payload.get("status") != "completed":
        raise ValueError(f"run_manifest must be completed: {run_manifest_path}")
    return payload


def _read_run_metrics(run_manifest: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    metrics_ref = str(run_manifest["artifacts"]["metrics"])
    metrics_path = (repo_root / metrics_ref).resolve()
    return load_json_file(metrics_path)


def _read_run_config_snapshot(run_manifest: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    snapshot_ref = str(run_manifest["artifacts"]["config_snapshot"])
    snapshot_path = (repo_root / snapshot_ref).resolve()
    return load_yaml_file(snapshot_path)


def _benchmark_fit_seconds(metrics: dict[str, Any]) -> float:
    timing = metrics["timing"]
    fit_seconds = float(timing["training_wall_clock_seconds"])
    cluster_seconds = timing.get("cluster_induction_wall_clock_seconds")
    if cluster_seconds is not None:
        fit_seconds += float(cluster_seconds)
    return fit_seconds


def _processed_manifest_contract(payload: dict[str, Any]) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "dataset_name": payload.get("dataset_name"),
        "dataset_short_name": payload.get("dataset_short_name"),
        "split_family": payload.get("split_family"),
        "preprocessing_family": payload.get("preprocessing_family"),
        "dtype": payload.get("dtype"),
        "counts": dict(payload.get("counts", {})),
        "rating_range": dict(payload.get("rating_range", {})),
    }

    source = payload.get("source")
    if isinstance(source, dict):
        source_contract = {}
        if "format_family" in source:
            source_contract["format_family"] = source["format_family"]
        if source_contract:
            contract["source"] = source_contract

    validation = payload.get("validation")
    if isinstance(validation, dict):
        validation_contract = {}
        if "format_family" in validation:
            validation_contract["format_family"] = validation["format_family"]
        if "counts" in validation:
            validation_contract["counts"] = dict(validation["counts"])
        if validation_contract:
            contract["validation"] = validation_contract

    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        official_splits = artifacts.get("official_ml100k_splits")
        if isinstance(official_splits, dict) and "version" in official_splits:
            contract["official_ml100k_splits_version"] = str(official_splits["version"])

    return contract


def _discover_matching_run(
    *,
    repo_root: Path,
    dataset_short_name: str,
    processed_manifest_ref: str,
    model_name: str,
    model_config_ref: str,
    device_profile_name: str,
    runtime_dtype: str,
    split_family_name: str,
    train_ratio: float,
    validation_ratio: float,
    split_seed: int,
    model_seed: int,
) -> dict[str, Any]:
    expected_split_id = split_id(
        split_family_name,
        SplitConfig(train_ratio=train_ratio, validation_ratio=validation_ratio, seed=split_seed),
    )

    candidates: list[dict[str, Any]] = []
    for manifest_path in sorted((repo_root / "artifacts" / "runs").glob("*/run_manifest.json")):
        try:
            run_manifest = _read_run_manifest(manifest_path)
        except Exception:
            continue

        dataset = dict(run_manifest.get("dataset", {}))
        model = dict(run_manifest.get("model", {}))
        runtime = dict(run_manifest.get("runtime", {}))
        seeds = list(run_manifest.get("seeds", []))

        if str(dataset.get("short_name")) != dataset_short_name:
            continue
        if str(dataset.get("manifest_ref")) != processed_manifest_ref:
            continue
        if str(dataset.get("split_family")) != split_family_name:
            continue
        if str(dataset.get("split_id")) != expected_split_id:
            continue
        if str(model.get("name")) != model_name:
            continue
        if str(model.get("config_ref")) != model_config_ref:
            continue
        if str(runtime.get("device_profile")) != device_profile_name:
            continue
        if str(runtime.get("dtype")) != runtime_dtype:
            continue
        if seeds != [int(model_seed)]:
            continue

        candidates.append(
            {
                "manifest": run_manifest,
                "manifest_path": manifest_path.resolve(),
            }
        )

    if not candidates:
        raise FileNotFoundError(f"no matching run_manifest found for split_seed={split_seed}")
    if len(candidates) > 1:
        raise ValueError(
            "multiple matching run manifests found for "
            f"split_seed={split_seed}; use explicit run_manifest_paths"
        )
    return candidates[0]
