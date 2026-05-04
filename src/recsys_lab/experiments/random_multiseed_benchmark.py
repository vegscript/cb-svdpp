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
    reserve_timestamped_artifact_dir,
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
from recsys_lab.models.registry import validate_model_config_payload
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
            f"multiple matching run manifests found for split_seed={split_seed}; use explicit run_manifest_paths"
        )
    return candidates[0]


def _validate_selected_run(
    *,
    selected_run: dict[str, Any],
    dataset_short_name: str,
    processed_manifest_ref: str,
    processed_manifest_contract: dict[str, Any],
    model_name: str,
    model_config_ref: str,
    model_config_payload: dict[str, Any],
    runtime_config_ref: str,
    runtime_config_payload: dict[str, Any],
    device_config_ref: str,
    device_config_payload: dict[str, Any],
    device_profile_name: str,
    runtime_dtype: str,
    split_family_name: str,
    train_ratio: float,
    validation_ratio: float,
    split_seed: int,
    model_seed: int,
) -> None:
    run_manifest = dict(selected_run["manifest"])
    config_snapshot = dict(selected_run["config_snapshot"])

    expected_split_id = split_id(
        split_family_name,
        SplitConfig(train_ratio=train_ratio, validation_ratio=validation_ratio, seed=split_seed),
    )

    dataset = dict(run_manifest.get("dataset", {}))
    model = dict(run_manifest.get("model", {}))
    runtime = dict(run_manifest.get("runtime", {}))

    if str(dataset.get("short_name")) != dataset_short_name:
        raise ValueError(f"dataset short_name mismatch for split_seed={split_seed}")
    if str(dataset.get("manifest_ref")) != processed_manifest_ref:
        raise ValueError(f"processed_manifest mismatch for split_seed={split_seed}")
    if str(dataset.get("split_family")) != split_family_name:
        raise ValueError(f"split_family mismatch for split_seed={split_seed}")
    if str(dataset.get("split_id")) != expected_split_id:
        raise ValueError(f"split_id mismatch for split_seed={split_seed}")
    if str(model.get("name")) != model_name:
        raise ValueError(f"model name mismatch for split_seed={split_seed}")
    if str(model.get("config_ref")) != model_config_ref:
        raise ValueError(f"model_config mismatch for split_seed={split_seed}")
    if str(runtime.get("device_profile")) != device_profile_name:
        raise ValueError(f"device profile mismatch for split_seed={split_seed}")
    if str(runtime.get("dtype")) != runtime_dtype:
        raise ValueError(f"runtime dtype mismatch for split_seed={split_seed}")
    if list(run_manifest.get("seeds", [])) != [int(model_seed)]:
        raise ValueError(f"model_seed mismatch for split_seed={split_seed}")

    inputs = dict(config_snapshot.get("inputs", {}))
    split_payload = dict(config_snapshot.get("split", {}))
    loaded_configs = dict(config_snapshot.get("loaded_configs", {}))

    if str(inputs.get("processed_manifest")) != processed_manifest_ref:
        raise ValueError(f"config snapshot processed_manifest mismatch for split_seed={split_seed}")
    if str(inputs.get("model_config")) != model_config_ref:
        raise ValueError(f"config snapshot model_config mismatch for split_seed={split_seed}")
    if str(inputs.get("runtime_config")) != runtime_config_ref:
        raise ValueError(f"config snapshot runtime_config mismatch for split_seed={split_seed}")
    if str(inputs.get("device_config")) != device_config_ref:
        raise ValueError(f"config snapshot device_config mismatch for split_seed={split_seed}")

    if float(split_payload.get("train_ratio", -1.0)) != float(train_ratio):
        raise ValueError(f"train_ratio mismatch for split_seed={split_seed}")
    if float(split_payload.get("validation_ratio", -1.0)) != float(validation_ratio):
        raise ValueError(f"validation_ratio mismatch for split_seed={split_seed}")
    if int(split_payload.get("seed", -1)) != int(split_seed):
        raise ValueError(f"config snapshot split seed mismatch for split_seed={split_seed}")
    if int(config_snapshot.get("model_seed", -1)) != int(model_seed):
        raise ValueError(f"config snapshot model_seed mismatch for split_seed={split_seed}")

    loaded_processed_manifest = dict(loaded_configs.get("processed_manifest", {}))
    if _processed_manifest_contract(loaded_processed_manifest) != processed_manifest_contract:
        raise ValueError(
            f"processed manifest contract mismatch for split_seed={split_seed}; canonical dataset contract differs"
        )
    if dict(loaded_configs.get("model", {})) != model_config_payload:
        raise ValueError(f"loaded model config mismatch for split_seed={split_seed}")
    if dict(loaded_configs.get("runtime", {})) != runtime_config_payload:
        raise ValueError(f"loaded runtime config mismatch for split_seed={split_seed}")
    if dict(loaded_configs.get("device", {})) != device_config_payload:
        raise ValueError(f"loaded device config mismatch for split_seed={split_seed}")

    metrics = dict(selected_run["metrics"])
    metric_block = dict(metrics.get("metrics", {}))
    if metric_block.get("train_rmse") is None:
        raise ValueError(f"missing train_rmse for split_seed={split_seed}")
    if metric_block.get("validation_rmse") is None:
        raise ValueError(f"missing validation_rmse for split_seed={split_seed}")
    if metric_block.get("test_rmse") is None:
        raise ValueError(f"missing test_rmse for split_seed={split_seed}")


def _validate_run_bundle(
    *,
    selected_runs: list[dict[str, Any]],
    dataset_short_name: str,
    processed_manifest_ref: str,
    processed_manifest_contract: dict[str, Any],
    model_name: str,
    model_config_ref: str,
    model_config_payload: dict[str, Any],
    runtime_config_ref: str,
    runtime_config_payload: dict[str, Any],
    device_config_ref: str,
    device_config_payload: dict[str, Any],
    device_profile_name: str,
    runtime_dtype: str,
    split_family_name: str,
    train_ratio: float,
    validation_ratio: float,
    split_seeds: list[int],
    model_seed: int,
    current_git: dict[str, Any],
) -> None:
    if len(selected_runs) != len(split_seeds):
        raise ValueError("selected_runs and split_seeds must have the same length")

    reference_git: tuple[str, bool, str] | None = None
    for split_seed_value, selected_run in zip(split_seeds, selected_runs, strict=True):
        _validate_selected_run(
            selected_run=selected_run,
            dataset_short_name=dataset_short_name,
            processed_manifest_ref=processed_manifest_ref,
            processed_manifest_contract=processed_manifest_contract,
            model_name=model_name,
            model_config_ref=model_config_ref,
            model_config_payload=model_config_payload,
            runtime_config_ref=runtime_config_ref,
            runtime_config_payload=runtime_config_payload,
            device_config_ref=device_config_ref,
            device_config_payload=device_config_payload,
            device_profile_name=device_profile_name,
            runtime_dtype=runtime_dtype,
            split_family_name=split_family_name,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            split_seed=split_seed_value,
            model_seed=model_seed,
        )

        run_git = dict(selected_run["manifest"].get("git", {}))
        run_git_identity = (
            str(run_git.get("commit", "")),
            bool(run_git.get("dirty", False)),
            str(run_git.get("branch", "")),
        )
        if reference_git is None:
            reference_git = run_git_identity
            continue
        if run_git_identity != reference_git:
            raise ValueError("selected runs must share identical git commit, branch, and dirty state")

    if reference_git is None:
        raise ValueError("selected_runs must not be empty")

    if reference_git[0] != str(current_git.get("commit", "")) or reference_git[1] != bool(
        current_git.get("dirty", False)
    ):
        raise ValueError("selected runs must match current repo git commit and dirty state")


def _resolve_selected_runs(
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
    split_seeds: list[int],
    model_seed: int,
    run_manifest_paths: list[Path] | None,
) -> list[dict[str, Any]]:
    if run_manifest_paths is not None:
        if len(run_manifest_paths) != len(split_seeds):
            raise ValueError("run_manifest_paths must match split_seeds length")
        selected_runs: list[dict[str, Any]] = [
            {
                "manifest": _read_run_manifest(path.resolve()),
                "manifest_path": path.resolve(),
            }
            for path in run_manifest_paths
        ]
    else:
        selected_runs = [
            _discover_matching_run(
                repo_root=repo_root,
                dataset_short_name=dataset_short_name,
                processed_manifest_ref=processed_manifest_ref,
                model_name=model_name,
                model_config_ref=model_config_ref,
                device_profile_name=device_profile_name,
                runtime_dtype=runtime_dtype,
                split_family_name=split_family_name,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                split_seed=split_seed_value,
                model_seed=model_seed,
            )
            for split_seed_value in split_seeds
        ]

    for selected_run in selected_runs:
        run_manifest = dict(selected_run["manifest"])
        selected_run["config_snapshot"] = _read_run_config_snapshot(run_manifest, repo_root=repo_root)
        selected_run["metrics"] = _read_run_metrics(run_manifest, repo_root=repo_root)
    return selected_runs


def run_random_multiseed_benchmark(
    *,
    processed_manifest_path: Path,
    model_config_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    split_seeds: list[int],
    model_seed: int,
    run_manifest_paths: list[Path] | None = None,
    repo_root: Path | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    if not split_seeds:
        raise ValueError("split_seeds must contain at least one seed")

    root = (repo_root or discover_repo_root()).resolve()
    processed_manifest_path = processed_manifest_path.resolve()
    model_config_path = model_config_path.resolve()
    runtime_config_path = runtime_config_path.resolve()
    device_config_path = device_config_path.resolve()

    processed_manifest = load_json_file(processed_manifest_path)
    dataset_short_name = str(processed_manifest["dataset_short_name"])
    split_family_name = str(processed_manifest["split_family"])
    if split_family_name != "benchmark_random_v1":
        raise ValueError("random multi-seed benchmark currently requires split_family='benchmark_random_v1'")

    model_config_payload = load_yaml_file(model_config_path)
    runtime_config_payload = load_yaml_file(runtime_config_path)
    device_config_payload = load_yaml_file(device_config_path)

    adapter, model_profile = validate_model_config_payload(model_config_payload)
    model_name = adapter.name
    runtime_dtype = adapter.runtime_dtype(model_profile)
    device_profile_name = str(device_config_payload["device_profile"]["name"])
    threading_config = resolve_runtime_threading_config(device_config_payload=device_config_payload)

    processed_manifest_ref = repo_path_string(processed_manifest_path, repo_root=root)
    model_config_ref = repo_path_string(model_config_path, repo_root=root)
    runtime_config_ref = repo_path_string(runtime_config_path, repo_root=root)
    device_config_ref = repo_path_string(device_config_path, repo_root=root)
    processed_manifest_contract = _processed_manifest_contract(processed_manifest)
    split_seed_slug = "_".join(seed_slug(seed) for seed in split_seeds)
    model_seed_slug = seed_slug(model_seed)
    run_manifest_path_refs = (
        [repo_path_string(path.resolve(), repo_root=root) for path in run_manifest_paths]
        if run_manifest_paths is not None
        else None
    )

    timestamp, benchmark_id, benchmark_dir = reserve_timestamped_artifact_dir(
        artifacts_root=root / "artifacts" / "benchmarks",
        id_from_timestamp=lambda reserved_timestamp: "_".join(
            [
                reserved_timestamp,
                dataset_short_name,
                split_family_name,
                model_name,
                "multiseed",
                split_seed_slug,
                "modelseed",
                model_seed_slug,
                device_profile_name,
            ]
        ),
    )

    summary_path = benchmark_dir / "summary.json"
    summary_md_path = benchmark_dir / "summary.md"
    stdout_log_path = benchmark_dir / "stdout.log"
    benchmark_manifest_path = benchmark_dir / "benchmark_manifest.json"
    config_snapshot_path = benchmark_dir / "config_snapshot.yaml"

    benchmark_scope = "_".join(
        [
            dataset_short_name,
            split_family_name,
            model_name,
            "multiseed",
            split_seed_slug,
            "modelseed",
            model_seed_slug,
        ]
    )
    command_string = command or (
        "recsys-lab benchmark-random-multiseed "
        f"--processed-manifest {processed_manifest_ref} "
        f"--model-config {model_config_ref} "
        f"--runtime-config {runtime_config_ref} "
        f"--device-config {device_config_ref} "
        f"--split-seeds {','.join(str(seed) for seed in split_seeds)} "
        f"--model-seed {model_seed}"
        + (f" --run-manifest-paths {','.join(run_manifest_path_refs)}" if run_manifest_path_refs is not None else "")
    )

    git = git_snapshot(root)
    measurement = build_benchmark_measurement(
        time_metric="training_wall_clock_seconds",
        time_metric_semantics=(
            "Benchmark fit time equals training_wall_clock_seconds. "
            "If cluster_induction_wall_clock_seconds is present, it is added for fair cb_* comparisons."
        ),
        sample_unit="split_seed_run",
        measured_sample_count=len(split_seeds),
        warmup_policy="none",
        warmup_sample_count=0,
        notes=[
            "No additional warmup runs are executed; this artifact aggregates completed benchmark_random_v1 runs.",
            "Split-seed aggregation keeps model_seed fixed and reports dispersion across selected split seeds.",
        ],
    )

    with runtime_execution_context(threading_config=threading_config):
        benchmark_manifest: dict[str, Any] = {
            "manifest_version": "v1",
            "kind": "benchmark_manifest",
            "generated_at_utc": timestamp,
            "benchmark_id": benchmark_id,
            "status": "started",
            "benchmark_scope": benchmark_scope,
            "command": command_string,
            "cwd": repo_path_string(root, repo_root=root),
            "git": git,
            "runtime": build_runtime_metadata(
                device_profile_name=device_profile_name,
                runtime_dtype=runtime_dtype,
                device_config_payload=device_config_payload,
            ),
            "measurement": measurement,
            "inputs": {
                "run_ids": [],
                "run_manifest_paths": run_manifest_path_refs or [],
                "model_seeds": [int(model_seed)],
                "split_seeds": [int(seed) for seed in split_seeds],
            },
            "artifacts": {
                "summary": repo_path_string(summary_path, repo_root=root),
                "tables": [repo_path_string(summary_md_path, repo_root=root)],
                "stdout_log": repo_path_string(stdout_log_path, repo_root=root),
            },
            "timing": {
                "started_at_utc": timestamp,
            },
        }

        dump_yaml_file(
            config_snapshot_path,
            {
                "benchmark_id": benchmark_id,
                "benchmark_scope": benchmark_scope,
                "dataset_short_name": dataset_short_name,
                "split_family": split_family_name,
                "split_seeds": [int(seed) for seed in split_seeds],
                "model_seed": int(model_seed),
                "processed_manifest": processed_manifest_ref,
                "model_config": model_config_ref,
                "runtime_config": runtime_config_ref,
                "device_config": device_config_ref,
            },
        )
        write_log(
            stdout_log_path,
            [
                f"[{timestamp}] benchmark_id={benchmark_id}",
                f"command={command_string}",
                "allow_existing_run_reuse=false",
            ],
        )
        write_json(benchmark_manifest_path, benchmark_manifest)

        try:
            selected_runs = _resolve_selected_runs(
                repo_root=root,
                dataset_short_name=dataset_short_name,
                processed_manifest_ref=processed_manifest_ref,
                model_name=model_name,
                model_config_ref=model_config_ref,
                device_profile_name=device_profile_name,
                runtime_dtype=runtime_dtype,
                split_family_name=split_family_name,
                train_ratio=0.8,
                validation_ratio=0.1,
                split_seeds=split_seeds,
                model_seed=model_seed,
                run_manifest_paths=run_manifest_paths,
            )
            _validate_run_bundle(
                selected_runs=selected_runs,
                dataset_short_name=dataset_short_name,
                processed_manifest_ref=processed_manifest_ref,
                processed_manifest_contract=processed_manifest_contract,
                model_name=model_name,
                model_config_ref=model_config_ref,
                model_config_payload=model_config_payload,
                runtime_config_ref=runtime_config_ref,
                runtime_config_payload=runtime_config_payload,
                device_config_ref=device_config_ref,
                device_config_payload=device_config_payload,
                device_profile_name=device_profile_name,
                runtime_dtype=runtime_dtype,
                split_family_name=split_family_name,
                train_ratio=0.8,
                validation_ratio=0.1,
                split_seeds=split_seeds,
                model_seed=model_seed,
                current_git=git,
            )

            run_ids: list[str] = []
            run_manifest_path_values: list[str] = []
            train_rmse_values: list[float] = []
            validation_rmse_values: list[float] = []
            test_rmse_values: list[float] = []
            training_seconds_values: list[float] = []
            peak_memory_values: list[float] = []
            has_peak_memory = True
            per_run: list[dict[str, Any]] = []

            for split_seed_value, selected_run in zip(split_seeds, selected_runs, strict=True):
                run_manifest = dict(selected_run["manifest"])
                metrics = dict(selected_run["metrics"])
                metric_block = dict(metrics["metrics"])
                system_metrics = dict(metrics.get("system_metrics", {}))
                split_payload = dict(metrics.get("split", {}))

                run_ids.append(str(run_manifest["run_id"]))
                run_manifest_path_values.append(
                    repo_path_string(Path(str(selected_run["manifest_path"])).resolve(), repo_root=root)
                )
                train_rmse_values.append(float(metric_block["train_rmse"]))
                validation_rmse_values.append(float(metric_block["validation_rmse"]))
                test_rmse_values.append(float(metric_block["test_rmse"]))
                training_seconds_values.append(_benchmark_fit_seconds(metrics))

                peak_memory_mb = system_metrics.get("peak_memory_mb")
                if peak_memory_mb is None:
                    has_peak_memory = False
                else:
                    peak_memory_values.append(float(peak_memory_mb))

                per_run_entry: dict[str, Any] = {
                    "split_seed": int(split_seed_value),
                    "split_id": str(run_manifest["dataset"]["split_id"]),
                    "run_id": str(run_manifest["run_id"]),
                    "train_rmse": float(metric_block["train_rmse"]),
                    "validation_rmse": float(metric_block["validation_rmse"]),
                    "test_rmse": float(metric_block["test_rmse"]),
                    "training_wall_clock_seconds": _benchmark_fit_seconds(metrics),
                    "train_rows": int(split_payload.get("train_rows", 0)),
                    "validation_rows": int(split_payload.get("validation_rows", 0)),
                    "test_rows": int(split_payload.get("test_rows", 0)),
                }
                if peak_memory_mb is not None:
                    per_run_entry["peak_memory_mb"] = float(peak_memory_mb)
                per_run.append(per_run_entry)

            aggregate: dict[str, Any] = {
                "train_rmse": summarize_scalar_samples(train_rmse_values),
                "validation_rmse": summarize_scalar_samples(validation_rmse_values),
                "test_rmse": summarize_scalar_samples(test_rmse_values),
                "training_wall_clock_seconds": summarize_scalar_samples(training_seconds_values),
            }
            if has_peak_memory and len(peak_memory_values) == len(split_seeds):
                aggregate["peak_memory_mb"] = summarize_scalar_samples(peak_memory_values)

            summary_payload: dict[str, Any] = {
                "benchmark_id": benchmark_id,
                "benchmark_scope": benchmark_scope,
                "dataset": dataset_short_name,
                "split_family": split_family_name,
                "model": model_name,
                "split_seeds": [int(seed) for seed in split_seeds],
                "model_seed": int(model_seed),
                "measurement": measurement,
                "per_run": per_run,
                "aggregate": aggregate,
            }
            write_json(summary_path, summary_payload)

            markdown_lines = [
                "# Random Multi-Seed Benchmark Summary",
                "",
                f"- benchmark_id: `{benchmark_id}`",
                f"- benchmark_scope: `{benchmark_scope}`",
                f"- dataset: `{dataset_short_name}`",
                f"- split_family: `{split_family_name}`",
                f"- model: `{model_name}`",
                f"- split_seeds: `{', '.join(str(seed) for seed in split_seeds)}`",
                f"- model_seed: `{model_seed}`",
                f"- warmup_policy: `{measurement['warmup_policy']}`",
                f"- measured_sample_count: `{measurement['measured_sample_count']}`",
                "",
                "| Split Seed | Run ID | Validation RMSE | Test RMSE | Fit Time (s) |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
            for item in per_run:
                markdown_lines.append(
                    f"| `{item['split_seed']}` | `{item['run_id']}` | "
                    f"{item['validation_rmse']:.6f} | {item['test_rmse']:.6f} | "
                    f"{item['training_wall_clock_seconds']:.2f} |"
                )
            markdown_lines.extend(
                [
                    "",
                    "## Aggregate",
                    "",
                    f"- validation_rmse mean: `{aggregate['validation_rmse']['mean']:.6f}`",
                    f"- validation_rmse std: `{aggregate['validation_rmse']['std']:.6f}`",
                    f"- test_rmse mean: `{aggregate['test_rmse']['mean']:.6f}`",
                    f"- test_rmse std: `{aggregate['test_rmse']['std']:.6f}`",
                    f"- training_wall_clock_seconds mean: `{aggregate['training_wall_clock_seconds']['mean']:.2f}`",
                    (
                        "- training_wall_clock_seconds cv: "
                        f"`{aggregate['training_wall_clock_seconds']['coefficient_of_variation']:.6f}`"
                    ),
                ]
            )
            if "peak_memory_mb" in aggregate:
                markdown_lines.append(f"- peak_memory_mb mean: `{aggregate['peak_memory_mb']['mean']:.2f}`")
                markdown_lines.append(f"- peak_memory_mb std: `{aggregate['peak_memory_mb']['std']:.2f}`")
            summary_md_path.write_text("\n".join(markdown_lines).strip() + "\n", encoding="utf-8", newline="\n")

            finished_at = utc_timestamp()
            write_log(
                stdout_log_path,
                [
                    f"[{timestamp}] benchmark_id={benchmark_id}",
                    f"command={command_string}",
                    "allow_existing_run_reuse=false",
                    f"run_count={len(per_run)}",
                    f"validation_rmse_mean={aggregate['validation_rmse']['mean']:.6f}",
                    f"test_rmse_mean={aggregate['test_rmse']['mean']:.6f}",
                    f"[{finished_at}] status=completed",
                ],
            )
            completed_manifest = {
                **benchmark_manifest,
                "status": "completed",
                "generated_at_utc": finished_at,
                "inputs": {
                    "run_ids": run_ids,
                    "run_manifest_paths": run_manifest_path_values,
                    "model_seeds": [int(model_seed)],
                    "split_seeds": [int(seed) for seed in split_seeds],
                },
                "timing": {
                    **benchmark_manifest["timing"],
                    "finished_at_utc": finished_at,
                },
            }
            write_json(benchmark_manifest_path, completed_manifest)
            validate_manifest_file(benchmark_manifest_path, repo_root=root)
            return {
                "benchmark_id": benchmark_id,
                "benchmark_dir": str(benchmark_dir),
                "benchmark_manifest": str(benchmark_manifest_path),
                "summary": aggregate,
            }
        except Exception:
            finished_at = utc_timestamp()
            write_log(
                stdout_log_path,
                [
                    f"[{timestamp}] benchmark_id={benchmark_id}",
                    f"command={command_string}",
                    "allow_existing_run_reuse=false",
                    f"[{finished_at}] status=failed",
                    traceback.format_exc().strip(),
                ],
            )
            failed_manifest = {
                **benchmark_manifest,
                "status": "failed",
                "generated_at_utc": finished_at,
                "timing": {
                    **benchmark_manifest["timing"],
                    "finished_at_utc": finished_at,
                },
            }
            write_json(benchmark_manifest_path, failed_manifest)
            validate_manifest_file(benchmark_manifest_path, repo_root=root)
            raise
