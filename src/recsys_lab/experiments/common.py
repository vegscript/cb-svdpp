from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recsys_lab.data.processed import RatingsData
from recsys_lab.data.splitters import RatingsSplit
from recsys_lab.utils.paths import repo_path_string


@dataclass(frozen=True, slots=True)
class SplitConfig:
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    seed: int = 1


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def seed_slug(seed: int) -> str:
    return f"s{seed:03d}"


def git_output(args: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def git_snapshot(repo_root: Path) -> dict[str, Any]:
    commit = git_output(["rev-parse", "HEAD"], cwd=repo_root)
    branch = git_output(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    dirty = bool(git_output(["status", "--porcelain"], cwd=repo_root))
    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
    }


def build_run_id(
    *,
    timestamp: str,
    dataset_short_name: str,
    model_name: str,
    device_profile_name: str,
    model_seed: int,
) -> str:
    return "_".join(
        [
            timestamp,
            dataset_short_name,
            model_name,
            device_profile_name,
            seed_slug(model_seed),
        ]
    )


def split_id(split_family: str, split_config: SplitConfig) -> str:
    train_pct = int(round(split_config.train_ratio * 100))
    validation_pct = int(round(split_config.validation_ratio * 100))
    return (
        f"{split_family}_tr{train_pct:03d}_va{validation_pct:03d}_"
        f"{seed_slug(split_config.seed)}"
    )


def paper_faithful_ml100k_split_id(fold_index: int) -> str:
    if fold_index not in {1, 2, 3, 4, 5}:
        raise ValueError("paper-faithful ml100k fold_index must be one of 1, 2, 3, 4, 5")
    return f"paper_faithful_ml100k_v1_u{fold_index}"


def paper_faithful_ml100k_inner_split_id(
    *,
    fold_index: int,
    validation_ratio: float,
    inner_seed: int,
) -> str:
    if fold_index not in {1, 2, 3, 4, 5}:
        raise ValueError("paper-faithful ml100k fold_index must be one of 1, 2, 3, 4, 5")
    validation_pct = int(round(validation_ratio * 100))
    return (
        f"paper_faithful_ml100k_inner_v1_u{fold_index}_"
        f"va{validation_pct:03d}_{seed_slug(inner_seed)}"
    )


def ratings_summary(data: RatingsData) -> dict[str, Any]:
    return {
        "rows": len(data),
        "users": data.n_users,
        "items": data.n_items,
        "rating_min": data.rating_min,
        "rating_max": data.rating_max,
    }


def split_summary(split: RatingsSplit) -> dict[str, Any]:
    validation_rows = 0 if split.validation is None else len(split.validation)
    return {
        "train_rows": len(split.train),
        "validation_rows": validation_rows,
        "test_rows": len(split.test),
        "has_validation": split.validation is not None,
    }


def resolve_runtime_dtype(
    *,
    runtime_config_payload: dict[str, Any],
    device_config_payload: dict[str, Any],
    model_config_payload: dict[str, Any],
) -> str:
    training = model_config_payload.get("training", {})
    if "dtype" in training:
        return str(training["dtype"])

    device_precision = device_config_payload.get("precision", {})
    if "default_dtype" in device_precision:
        return str(device_precision["default_dtype"])

    runtime = runtime_config_payload.get("runtime", {})
    precision_profiles = runtime_config_payload.get("precision_profiles", {})
    default_profile = str(runtime.get("default_precision_profile", "performance_float32"))
    if default_profile in precision_profiles:
        payload = precision_profiles[default_profile]
        if "dtype" in payload:
            return str(payload["dtype"])

    raise ValueError("unable to resolve runtime dtype from runtime/device/model configs")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


def write_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).strip() + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def build_base_run_manifest(
    *,
    timestamp: str,
    run_id: str,
    command: str,
    repo_root: Path,
    git: dict[str, Any],
    processed_manifest: dict[str, Any],
    processed_manifest_path: Path,
    model_name: str,
    model_scope: str,
    model_config_path: Path,
    device_profile_name: str,
    runtime_dtype: str,
    device_config_payload: dict[str, Any],
    model_seed: int,
    split_config: SplitConfig,
    config_snapshot_path: Path,
    metrics_path: Path,
    stdout_log_path: Path,
    split_family_name: str | None = None,
    split_id_value: str | None = None,
) -> dict[str, Any]:
    resolved_split_family = split_family_name or str(processed_manifest["split_family"])
    resolved_split_id = split_id_value or split_id(resolved_split_family, split_config)
    return {
        "manifest_version": "v1",
        "kind": "run_manifest",
        "generated_at_utc": timestamp,
        "run_id": run_id,
        "status": "started",
        "command": command,
        "cwd": repo_path_string(repo_root, repo_root=repo_root),
        "git": git,
        "dataset": {
            "short_name": str(processed_manifest["dataset_short_name"]),
            "source": str(processed_manifest.get("dataset_name", processed_manifest["dataset_short_name"])),
            "version": str(processed_manifest.get("preprocessing_family", "unknown")),
            "split_family": resolved_split_family,
            "split_id": resolved_split_id,
            "manifest_ref": repo_path_string(processed_manifest_path, repo_root=repo_root),
        },
        "model": {
            "name": model_name,
            "scope": model_scope,
            "config_ref": repo_path_string(model_config_path, repo_root=repo_root),
        },
        "runtime": {
            "device_profile": device_profile_name,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
            "dtype": runtime_dtype,
            "threading": {
                "omp_num_threads": int(device_config_payload["threading"]["omp_num_threads"]),
                "blas_threads": int(device_config_payload["threading"]["blas_threads"]),
            },
        },
        "seeds": [int(model_seed)],
        "artifacts": {
            "config_snapshot": repo_path_string(config_snapshot_path, repo_root=repo_root),
            "metrics": repo_path_string(metrics_path, repo_root=repo_root),
            "stdout_log": repo_path_string(stdout_log_path, repo_root=repo_root),
        },
        "timing": {
            "started_at_utc": timestamp,
        },
    }
