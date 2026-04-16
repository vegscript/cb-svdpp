from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Callable

from recsys_lab.config.loader import dump_yaml_file, load_yaml_file
from recsys_lab.experiments.benchmarking import (
    build_benchmark_measurement,
    summarize_scalar_samples,
)
from recsys_lab.experiments.biased_mf import run_biased_mf_experiment
from recsys_lab.experiments.cb_asvdpp import run_cb_asvdpp_experiment
from recsys_lab.experiments.cb_svdpp import run_cb_svdpp_experiment
from recsys_lab.experiments.common import (
    SplitConfig,
    build_runtime_metadata,
    git_snapshot,
    resolve_runtime_dtype,
    utc_timestamp,
    write_json,
    write_log,
)
from recsys_lab.experiments.runtime import (
    resolve_runtime_threading_config,
    runtime_execution_context,
)
from recsys_lab.experiments.svdpp import run_svdpp_experiment
from recsys_lab.utils.manifests import load_json_file, validate_manifest_file
from recsys_lab.utils.paths import discover_repo_root, repo_path_string


SUPPORTED_MODELS = {"biased_mf", "svdpp", "cb_svdpp", "cb_asvdpp"}


def _read_run_manifest(run_manifest_path: Path) -> dict[str, Any]:
    payload = load_json_file(run_manifest_path)
    if payload.get("kind") != "run_manifest":
        raise ValueError(f"expected run_manifest at {run_manifest_path}")
    return payload


def _read_run_metrics(run_manifest: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    metrics_ref = str(run_manifest["artifacts"]["metrics"])
    metrics_path = (repo_root / metrics_ref).resolve()
    return load_json_file(metrics_path)


def _read_run_config_snapshot(run_manifest: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    snapshot_ref = str(run_manifest["artifacts"]["config_snapshot"])
    snapshot_path = (repo_root / snapshot_ref).resolve()
    return load_yaml_file(snapshot_path)


def _existing_matching_run(
    *,
    repo_root: Path,
    processed_manifest_ref: str,
    processed_manifest_payload: dict[str, Any],
    model_name: str,
    model_config_ref: str,
    model_config_payload: dict[str, Any],
    runtime_config_payload: dict[str, Any],
    device_config_payload: dict[str, Any],
    device_profile_name: str,
    runtime_dtype: str,
    split_family: str,
    fold_index: int,
    model_seed: int,
    git_commit: str,
) -> Path | None:
    runs_root = repo_root / "artifacts" / "runs"
    if not runs_root.exists():
        return None

    target_split_id = f"paper_faithful_ml100k_v1_u{fold_index}"
    for run_manifest_path in sorted(runs_root.glob("*/run_manifest.json")):
        try:
            manifest = _read_run_manifest(run_manifest_path)
        except Exception:
            continue
        if manifest.get("status") != "completed":
            continue
        dataset = manifest.get("dataset", {})
        model = manifest.get("model", {})
        runtime = manifest.get("runtime", {})
        seeds = manifest.get("seeds", [])
        if str(dataset.get("manifest_ref")) != processed_manifest_ref:
            continue
        if str(manifest.get("git", {}).get("commit")) != git_commit:
            continue
        if str(dataset.get("split_family")) != split_family:
            continue
        if str(dataset.get("split_id")) != target_split_id:
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
        metrics_ref = manifest.get("artifacts", {}).get("metrics")
        if not metrics_ref:
            continue
        metrics_path = (repo_root / str(metrics_ref)).resolve()
        if not metrics_path.exists():
            continue
        try:
            config_snapshot = _read_run_config_snapshot(manifest, repo_root=repo_root)
        except Exception:
            continue
        loaded_configs = config_snapshot.get("loaded_configs", {})
        if loaded_configs.get("processed_manifest") != processed_manifest_payload:
            continue
        if loaded_configs.get("model") != model_config_payload:
            continue
        if loaded_configs.get("runtime") != runtime_config_payload:
            continue
        if loaded_configs.get("device") != device_config_payload:
            continue
        return run_manifest_path.resolve()
    return None


def _runner_for_model(model_name: str) -> Callable[..., dict[str, Any]]:
    if model_name == "biased_mf":
        return run_biased_mf_experiment
    if model_name == "svdpp":
        return run_svdpp_experiment
    if model_name == "cb_svdpp":
        return run_cb_svdpp_experiment
    if model_name == "cb_asvdpp":
        return run_cb_asvdpp_experiment
    raise ValueError(f"unsupported benchmark model: {model_name}")


def _benchmark_fit_seconds(metrics: dict[str, Any]) -> float:
    timing = metrics["timing"]
    fit_seconds = float(timing["training_wall_clock_seconds"])
    cluster_seconds = timing.get("cluster_induction_wall_clock_seconds")
    if cluster_seconds is not None:
        fit_seconds += float(cluster_seconds)
    return fit_seconds


def run_ml100k_paper_benchmark(
    *,
    model_name: str,
    processed_manifest_path: Path,
    model_config_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    model_seed: int,
    use_split_cache: bool | None = None,
    repo_root: Path | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported benchmark model: {model_name}")
    if use_split_cache is not None and model_name not in {"svdpp", "cb_svdpp"}:
        raise ValueError("explicit split-cache override is only supported for svdpp and cb_svdpp benchmarks")

    root = (repo_root or discover_repo_root()).resolve()
    processed_manifest_path = processed_manifest_path.resolve()
    model_config_path = model_config_path.resolve()
    runtime_config_path = runtime_config_path.resolve()
    device_config_path = device_config_path.resolve()

    processed_manifest = load_json_file(processed_manifest_path)
    if str(processed_manifest["dataset_short_name"]) != "ml100k":
        raise ValueError("ml100k paper benchmark requires dataset_short_name='ml100k'")

    if runtime_config_path.suffix != ".yaml" or device_config_path.suffix != ".yaml" or model_config_path.suffix != ".yaml":
        raise ValueError("runtime_config_path must be YAML")
    runtime_config_payload = load_yaml_file(runtime_config_path)
    device_config_payload = load_yaml_file(device_config_path)
    model_config_payload = load_yaml_file(model_config_path)

    runtime_dtype = resolve_runtime_dtype(
        runtime_config_payload=runtime_config_payload,
        device_config_payload=device_config_payload,
        model_config_payload=model_config_payload,
    )
    threading_config = resolve_runtime_threading_config(device_config_payload=device_config_payload)
    device_profile_name = str(device_config_payload["device_profile"]["name"])

    timestamp = utc_timestamp()
    benchmark_id = "_".join(
        [
            timestamp,
            "ml100k",
            "paper_faithful",
            model_name,
            device_profile_name,
        ]
    )
    benchmark_dir = root / "artifacts" / "benchmarks" / benchmark_id
    benchmark_dir.mkdir(parents=True, exist_ok=False)

    summary_path = benchmark_dir / "summary.json"
    summary_md_path = benchmark_dir / "summary.md"
    stdout_log_path = benchmark_dir / "stdout.log"
    benchmark_manifest_path = benchmark_dir / "benchmark_manifest.json"
    config_snapshot_path = benchmark_dir / "config_snapshot.yaml"

    benchmark_scope = f"paper_faithful_ml100k_v1_{model_name}_u1_u5"
    if use_split_cache is None:
        split_cache_command_fragment = ""
    else:
        split_cache_mode = "enable" if use_split_cache else "disable"
        split_cache_command_fragment = f"--split-cache {split_cache_mode} "
    command_string = command or (
        "recsys-lab benchmark-ml100k-paper "
        f"--model {model_name} "
        f"--processed-manifest {repo_path_string(processed_manifest_path, repo_root=root)} "
        f"--model-config {repo_path_string(model_config_path, repo_root=root)} "
        f"--runtime-config {repo_path_string(runtime_config_path, repo_root=root)} "
        f"--device-config {repo_path_string(device_config_path, repo_root=root)} "
        f"{split_cache_command_fragment}"
        f"--model-seed {model_seed}"
    )

    git = git_snapshot(root)
    allow_existing_run_reuse = not bool(git["dirty"])
    measurement = build_benchmark_measurement(
        time_metric="training_wall_clock_seconds",
        time_metric_semantics=(
            "Benchmark fit time equals training_wall_clock_seconds. "
            "If cluster_induction_wall_clock_seconds is present, it is added for fair cb_* comparisons."
        ),
        sample_unit="official_fold_run",
        measured_sample_count=5,
        warmup_policy="none",
        warmup_sample_count=0,
        notes=[
            "No separate warmup benchmark runs are executed; each official fold run is measured once.",
            "Dispersion across measured samples is reported via std and coefficient_of_variation.",
        ],
    )
    with runtime_execution_context(threading_config=threading_config):
        benchmark_manifest = {
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
                "run_manifest_paths": [],
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
                "model_name": model_name,
                "model_seed": model_seed,
                "use_split_cache": use_split_cache,
                "processed_manifest": repo_path_string(processed_manifest_path, repo_root=root),
                "model_config": repo_path_string(model_config_path, repo_root=root),
                "runtime_config": repo_path_string(runtime_config_path, repo_root=root),
                "device_config": repo_path_string(device_config_path, repo_root=root),
                "folds": [1, 2, 3, 4, 5],
            },
        )
        write_log(
            stdout_log_path,
            [
                f"[{timestamp}] benchmark_id={benchmark_id}",
                f"command={command_string}",
                f"allow_existing_run_reuse={str(allow_existing_run_reuse).lower()}",
            ],
        )
        write_json(benchmark_manifest_path, benchmark_manifest)

        run_manifest_paths: list[Path] = []
        try:
            processed_manifest_ref = repo_path_string(processed_manifest_path, repo_root=root)
            model_config_ref = repo_path_string(model_config_path, repo_root=root)
            runner = _runner_for_model(model_name)

            for fold_index in range(1, 6):
                existing_manifest_path = None
                if allow_existing_run_reuse:
                    existing_manifest_path = _existing_matching_run(
                        repo_root=root,
                        processed_manifest_ref=processed_manifest_ref,
                        processed_manifest_payload=processed_manifest,
                        model_name=model_name,
                        model_config_ref=model_config_ref,
                        model_config_payload=model_config_payload,
                        runtime_config_payload=runtime_config_payload,
                        device_config_payload=device_config_payload,
                        device_profile_name=device_profile_name,
                        runtime_dtype=runtime_dtype,
                        split_family="paper_faithful_ml100k_v1",
                        fold_index=fold_index,
                        model_seed=model_seed,
                        git_commit=str(git["commit"]),
                    )
                if existing_manifest_path is not None:
                    run_manifest_paths.append(existing_manifest_path)
                    continue

                runner_kwargs = {
                    "processed_manifest_path": processed_manifest_path,
                    "model_config_path": model_config_path,
                    "runtime_config_path": runtime_config_path,
                    "device_config_path": device_config_path,
                    "split_config": SplitConfig(train_ratio=0.8, validation_ratio=0.1, seed=fold_index),
                    "model_seed": model_seed,
                    "repo_root": root,
                    "split_family": "paper_faithful_ml100k_v1",
                }
                if model_name in {"svdpp", "cb_svdpp"}:
                    runner_kwargs["use_split_cache"] = use_split_cache
                payload = runner(
                    **runner_kwargs,
                )
                run_manifest_paths.append(Path(str(payload["run_manifest"])).resolve())

            run_manifests = [_read_run_manifest(path) for path in run_manifest_paths]
            run_metrics = [_read_run_metrics(manifest, repo_root=root) for manifest in run_manifests]

            test_rmses = [float(metrics["metrics"]["test_rmse"]) for metrics in run_metrics]
            train_rmses = [float(metrics["metrics"]["train_rmse"]) for metrics in run_metrics]
            training_seconds = [_benchmark_fit_seconds(metrics) for metrics in run_metrics]

            per_fold = []
            for fold_index, manifest, metrics in zip(range(1, 6), run_manifests, run_metrics, strict=True):
                per_fold.append(
                    {
                        "fold": f"u{fold_index}",
                        "run_id": str(manifest["run_id"]),
                        "train_rmse": float(metrics["metrics"]["train_rmse"]),
                        "test_rmse": float(metrics["metrics"]["test_rmse"]),
                        "training_wall_clock_seconds": _benchmark_fit_seconds(metrics),
                    }
                )

            summary_payload = {
                "benchmark_id": benchmark_id,
                "benchmark_scope": benchmark_scope,
                "dataset": "ml100k",
                "split_family": "paper_faithful_ml100k_v1",
                "model": model_name,
                "measurement": measurement,
                "folds": per_fold,
                "aggregate": {
                    "train_rmse": summarize_scalar_samples(train_rmses),
                    "test_rmse": summarize_scalar_samples(test_rmses),
                    "training_wall_clock_seconds": summarize_scalar_samples(training_seconds),
                },
            }
            write_json(summary_path, summary_payload)

            markdown_lines = [
                "# Benchmark Summary",
                "",
                f"- benchmark_id: `{benchmark_id}`",
                f"- benchmark_scope: `{benchmark_scope}`",
                f"- dataset: `ml100k`",
                f"- split_family: `paper_faithful_ml100k_v1`",
                f"- model: `{model_name}`",
                f"- warmup_policy: `{measurement['warmup_policy']}`",
                f"- measured_sample_count: `{measurement['measured_sample_count']}`",
                "",
                "| Fold | Run ID | Train RMSE | Test RMSE | Train Time (s) |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
            for item in per_fold:
                markdown_lines.append(
                    f"| `{item['fold']}` | `{item['run_id']}` | "
                    f"{item['train_rmse']:.6f} | {item['test_rmse']:.6f} | "
                    f"{item['training_wall_clock_seconds']:.2f} |"
                )
            markdown_lines.extend(
                [
                    "",
                    "## Aggregate",
                    "",
                    f"- test_rmse count: `{summary_payload['aggregate']['test_rmse']['count']}`",
                    f"- test_rmse mean: `{summary_payload['aggregate']['test_rmse']['mean']:.6f}`",
                    f"- test_rmse std: `{summary_payload['aggregate']['test_rmse']['std']:.6f}`",
                    f"- test_rmse median: `{summary_payload['aggregate']['test_rmse']['median']:.6f}`",
                    f"- train_rmse mean: `{summary_payload['aggregate']['train_rmse']['mean']:.6f}`",
                    f"- training_wall_clock_seconds cv: `{summary_payload['aggregate']['training_wall_clock_seconds']['coefficient_of_variation']:.6f}`",
                    f"- training_wall_clock_seconds mean: `{summary_payload['aggregate']['training_wall_clock_seconds']['mean']:.2f}`",
                ]
            )
            summary_md_path.write_text("\n".join(markdown_lines).strip() + "\n", encoding="utf-8", newline="\n")

            finished_at = utc_timestamp()
            write_log(
                stdout_log_path,
                [
                    f"[{timestamp}] benchmark_id={benchmark_id}",
                    f"command={command_string}",
                    f"allow_existing_run_reuse={str(allow_existing_run_reuse).lower()}",
                    f"run_count={len(run_manifests)}",
                    f"test_rmse_mean={summary_payload['aggregate']['test_rmse']['mean']:.6f}",
                    f"test_rmse_std={summary_payload['aggregate']['test_rmse']['std']:.6f}",
                    f"[{finished_at}] status=completed",
                ],
            )
            completed_manifest = {
                **benchmark_manifest,
                "status": "completed",
                "generated_at_utc": finished_at,
                "inputs": {
                    "run_ids": [str(manifest["run_id"]) for manifest in run_manifests],
                    "run_manifest_paths": [
                        repo_path_string(path, repo_root=root) for path in run_manifest_paths
                    ],
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
                "summary": summary_payload["aggregate"],
            }
        except Exception:
            finished_at = utc_timestamp()
            write_log(
                stdout_log_path,
                [
                    f"[{timestamp}] benchmark_id={benchmark_id}",
                    f"command={command_string}",
                    f"allow_existing_run_reuse={str(allow_existing_run_reuse).lower()}",
                    f"[{finished_at}] status=failed",
                    traceback.format_exc().strip(),
                ],
            )
            failed_manifest = {
                **benchmark_manifest,
                "status": "failed",
                "generated_at_utc": finished_at,
                "inputs": {
                    "run_ids": [
                        str(_read_run_manifest(path)["run_id"])
                        for path in run_manifest_paths
                        if path.exists()
                    ],
                    "run_manifest_paths": [
                        repo_path_string(path, repo_root=root)
                        for path in run_manifest_paths
                        if path.exists()
                    ],
                },
                "timing": {
                    **benchmark_manifest["timing"],
                    "finished_at_utc": finished_at,
                },
            }
            write_json(benchmark_manifest_path, failed_manifest)
            validate_manifest_file(benchmark_manifest_path, repo_root=root)
            raise
