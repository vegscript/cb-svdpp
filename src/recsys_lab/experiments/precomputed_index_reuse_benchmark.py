from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from recsys_lab.config.loader import dump_yaml_file, load_yaml_file
from recsys_lab.data.processed import load_processed_dataset_manifest
from recsys_lab.experiments.asvdpp import run_asvdpp_experiment
from recsys_lab.experiments.benchmarking import (
    build_benchmark_measurement,
    summarize_scalar_samples,
)
from recsys_lab.experiments.cb_svdpp import run_cb_svdpp_experiment
from recsys_lab.experiments.common import (
    SplitConfig,
    build_runtime_metadata,
    git_snapshot,
    utc_timestamp,
    write_json,
    write_log,
)
from recsys_lab.experiments.runtime import (
    resolve_runtime_threading_config,
    runtime_execution_context,
)
from recsys_lab.experiments.svdpp import run_svdpp_experiment
from recsys_lab.models.registry import validated_model_config_payload_with_training_overrides
from recsys_lab.utils.manifests import validate_manifest_file
from recsys_lab.utils.paths import discover_repo_root, repo_path_string

RunnerFn = Callable[..., dict[str, Any]]

_SUPPORTED_MODELS: tuple[str, ...] = ("svdpp", "asvdpp", "cb_svdpp")


def _resolve_runner(model_name: str) -> RunnerFn:
    if model_name == "svdpp":
        return run_svdpp_experiment
    if model_name == "asvdpp":
        return run_asvdpp_experiment
    if model_name == "cb_svdpp":
        return run_cb_svdpp_experiment
    raise ValueError(f"unsupported model for precomputed index reuse benchmark: {model_name}")


def _prepare_model_config(
    *,
    source_path: Path,
    output_path: Path,
    override_epochs: int,
) -> dict[str, Any]:
    payload = load_yaml_file(source_path)
    payload = validated_model_config_payload_with_training_overrides(
        payload,
        expected_model_name=source_path.stem,
        training_overrides={"epochs": int(override_epochs)},
    )
    dump_yaml_file(output_path, payload)
    return payload


def _load_run_outputs(run_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    run_manifest_path = Path(run_payload["run_manifest"])
    metrics_path = Path(run_payload["run_dir"]) / "metrics.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return run_manifest, metrics


def _run_variant_once(
    *,
    model_name: str,
    runner: RunnerFn,
    processed_manifest_path: Path,
    model_config_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    split_config: SplitConfig,
    model_seed: int,
    repo_root: Path,
    benchmark_command: str,
    reuse_precomputed_indices: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    command = (
        f"{benchmark_command} --model {model_name} "
        f"{'--reuse-precomputed-indices' if reuse_precomputed_indices else '--rebuild-indices-in-fit'}"
    )
    kwargs: dict[str, Any] = {
        "processed_manifest_path": processed_manifest_path,
        "model_config_path": model_config_path,
        "runtime_config_path": runtime_config_path,
        "device_config_path": device_config_path,
        "split_config": split_config,
        "model_seed": model_seed,
        "repo_root": repo_root,
        "command": command,
        "reuse_precomputed_indices": reuse_precomputed_indices,
    }
    if model_name in {"svdpp", "cb_svdpp"}:
        kwargs["evaluate_test"] = False
    run_payload = runner(**kwargs)
    run_manifest, metrics = _load_run_outputs(run_payload)
    return run_payload, run_manifest, metrics


def _summarize_variant(
    *,
    train_rows: int,
    train_time_total_samples: list[float],
    train_throughput_samples: list[float],
) -> dict[str, Any]:
    del train_rows
    return {
        "aggregate": {
            "train_time_total": summarize_scalar_samples(train_time_total_samples),
            "ratings_per_second_train": summarize_scalar_samples(train_throughput_samples),
        }
    }


def run_precomputed_index_reuse_benchmark(
    *,
    processed_manifest_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    measured_repeats: int = 2,
    override_epochs: int = 1,
    model_names: tuple[str, ...] = _SUPPORTED_MODELS,
    split_config: SplitConfig | None = None,
    model_seed: int = 1,
    repo_root: Path | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    if measured_repeats <= 0:
        raise ValueError("measured_repeats must be positive")
    if override_epochs <= 0:
        raise ValueError("override_epochs must be positive")
    if not model_names:
        raise ValueError("model_names must not be empty")
    unsupported = sorted(set(model_names) - set(_SUPPORTED_MODELS))
    if unsupported:
        raise ValueError(f"unsupported models requested: {', '.join(unsupported)}")

    root = (repo_root or discover_repo_root()).resolve()
    processed_manifest_path = processed_manifest_path.resolve()
    runtime_config_path = runtime_config_path.resolve()
    device_config_path = device_config_path.resolve()
    effective_split_config = split_config or SplitConfig(train_ratio=0.8, validation_ratio=0.1, seed=1)

    processed_manifest = load_processed_dataset_manifest(processed_manifest_path)
    dataset_short_name = str(processed_manifest["dataset_short_name"])
    runtime_config_payload = load_yaml_file(runtime_config_path)
    device_config_payload = load_yaml_file(device_config_path)
    threading_config = resolve_runtime_threading_config(device_config_payload=device_config_payload)
    device_profile_name = str(device_config_payload["device_profile"]["name"])
    git = git_snapshot(root)

    timestamp = utc_timestamp()
    benchmark_id = "_".join(
        [
            timestamp,
            dataset_short_name,
            "precomputed_index_reuse_compare",
            device_profile_name,
        ]
    )
    benchmark_scope = f"development_{dataset_short_name}_precomputed_index_reuse_compare"
    benchmark_dir = root / "artifacts" / "benchmarks" / benchmark_id
    benchmark_dir.mkdir(parents=True, exist_ok=False)
    config_dir = benchmark_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)

    summary_path = benchmark_dir / "summary.json"
    summary_md_path = benchmark_dir / "summary.md"
    stdout_log_path = benchmark_dir / "stdout.log"
    benchmark_manifest_path = benchmark_dir / "benchmark_manifest.json"
    config_snapshot_path = benchmark_dir / "config_snapshot.yaml"

    command_string = command or (
        "development benchmark precomputed index reuse compare "
        f"--processed-manifest {repo_path_string(processed_manifest_path, repo_root=root)} "
        f"--runtime-config {repo_path_string(runtime_config_path, repo_root=root)} "
        f"--device-config {repo_path_string(device_config_path, repo_root=root)} "
        f"--models {','.join(model_names)} "
        f"--epochs {override_epochs} "
        f"--measured-repeats {measured_repeats}"
    )
    measurement = build_benchmark_measurement(
        time_metric="train_time_total",
        time_metric_semantics=(
            "Per-variant training time reported by the underlying run metrics. "
            "The reference variant rebuilds training indices inside fit(); "
            "the optimized variant reuses the experiment-layer precomputed indices. "
            "Each model and variant receives one unmeasured warmup run followed by measured repeats."
        ),
        sample_unit="model_variant_train_run",
        measured_sample_count=measured_repeats,
        warmup_policy="separate_unmeasured",
        warmup_sample_count=1,
        notes=[
            "This is a development benchmark, not a benchmark-final claim.",
            "Benchmark uses reduced epochs to keep the proof focused on index construction overhead.",
            "Dirty workspaces invalidate benchmark-final reuse claims even when manifests are present.",
        ],
    )

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
            runtime_dtype=str(processed_manifest.get("dtype", "unknown")),
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

    config_snapshot_payload: dict[str, Any] = {
        "benchmark_id": benchmark_id,
        "benchmark_scope": benchmark_scope,
        "processed_manifest": repo_path_string(processed_manifest_path, repo_root=root),
        "runtime_config": repo_path_string(runtime_config_path, repo_root=root),
        "device_config": repo_path_string(device_config_path, repo_root=root),
        "measured_repeats": measured_repeats,
        "override_epochs": override_epochs,
        "model_names": list(model_names),
        "split_config": asdict(effective_split_config),
        "model_seed": model_seed,
        "loaded_configs": {
            "processed_manifest": processed_manifest,
            "runtime": runtime_config_payload,
            "device": device_config_payload,
            "models": {},
        },
    }

    benchmark_model_config_paths: dict[str, Path] = {}
    for model_name in model_names:
        source_path = root / "configs" / "models" / f"{model_name}.yaml"
        output_path = config_dir / f"{model_name}_benchmark.yaml"
        config_snapshot_payload["loaded_configs"]["models"][model_name] = _prepare_model_config(
            source_path=source_path,
            output_path=output_path,
            override_epochs=override_epochs,
        )
        benchmark_model_config_paths[model_name] = output_path

    dump_yaml_file(config_snapshot_path, config_snapshot_payload)
    write_log(
        stdout_log_path,
        [
            f"[{timestamp}] benchmark_id={benchmark_id}",
            f"command={command_string}",
        ],
    )
    write_json(benchmark_manifest_path, benchmark_manifest)

    try:
        with runtime_execution_context(threading_config=threading_config):
            summary_payload: dict[str, Any] = {
                "benchmark_id": benchmark_id,
                "benchmark_scope": benchmark_scope,
                "dataset": dataset_short_name,
                "measurement": measurement,
                "override_epochs": override_epochs,
                "split_config": asdict(effective_split_config),
                "model_seed": model_seed,
                "models": {},
            }

            for model_name in model_names:
                runner = _resolve_runner(model_name)
                model_config_path = benchmark_model_config_paths[model_name]
                model_summary: dict[str, Any] = {
                    "config_path": repo_path_string(model_config_path, repo_root=root),
                    "variants": {},
                }
                train_rows_value: int | None = None
                variant_measurements: dict[str, dict[str, Any]] = {}
                for variant_name, reuse_precomputed_indices in (
                    ("reference_rebuild_indices", False),
                    ("optimized_reuse_precomputed_indices", True),
                ):
                    warmup_payload, warmup_manifest, warmup_metrics = _run_variant_once(
                        model_name=model_name,
                        runner=runner,
                        processed_manifest_path=processed_manifest_path,
                        model_config_path=model_config_path,
                        runtime_config_path=runtime_config_path,
                        device_config_path=device_config_path,
                        split_config=effective_split_config,
                        model_seed=model_seed,
                        repo_root=root,
                        benchmark_command=command_string,
                        reuse_precomputed_indices=reuse_precomputed_indices,
                    )
                    benchmark_manifest["inputs"]["run_ids"].append(str(warmup_manifest["run_id"]))
                    benchmark_manifest["inputs"]["run_manifest_paths"].append(
                        repo_path_string(Path(warmup_payload["run_manifest"]), repo_root=root)
                    )

                    measured_runs: list[dict[str, Any]] = []
                    train_time_total_samples: list[float] = []
                    train_throughput_samples: list[float] = []
                    for _ in range(measured_repeats):
                        run_payload, run_manifest, metrics = _run_variant_once(
                            model_name=model_name,
                            runner=runner,
                            processed_manifest_path=processed_manifest_path,
                            model_config_path=model_config_path,
                            runtime_config_path=runtime_config_path,
                            device_config_path=device_config_path,
                            split_config=effective_split_config,
                            model_seed=model_seed,
                            repo_root=root,
                            benchmark_command=command_string,
                            reuse_precomputed_indices=reuse_precomputed_indices,
                        )
                        benchmark_manifest["inputs"]["run_ids"].append(str(run_manifest["run_id"]))
                        benchmark_manifest["inputs"]["run_manifest_paths"].append(
                            repo_path_string(Path(run_payload["run_manifest"]), repo_root=root)
                        )
                        train_time_total_samples.append(float(metrics["system_metrics"]["train_time_total"]))
                        train_throughput_samples.append(float(metrics["system_metrics"]["ratings_per_second_train"]))
                        measured_runs.append(
                            {
                                "run_id": str(run_manifest["run_id"]),
                                "run_manifest": repo_path_string(
                                    Path(run_payload["run_manifest"]),
                                    repo_root=root,
                                ),
                                "train_time_total": float(metrics["system_metrics"]["train_time_total"]),
                                "ratings_per_second_train": float(
                                    metrics["system_metrics"]["ratings_per_second_train"]
                                ),
                                "precomputed_index_reuse": bool(metrics["model"]["precomputed_index_reuse"]),
                            }
                        )
                        if train_rows_value is None:
                            train_rows_value = int(metrics["split"]["train_rows"])

                    variant_measurements[variant_name] = {
                        "warmup": {
                            "run_id": str(warmup_manifest["run_id"]),
                            "run_manifest": repo_path_string(
                                Path(warmup_payload["run_manifest"]),
                                repo_root=root,
                            ),
                            "train_time_total": float(warmup_metrics["system_metrics"]["train_time_total"]),
                            "ratings_per_second_train": float(
                                warmup_metrics["system_metrics"]["ratings_per_second_train"]
                            ),
                            "precomputed_index_reuse": bool(warmup_metrics["model"]["precomputed_index_reuse"]),
                        },
                        "measured_runs": measured_runs,
                        "summary": _summarize_variant(
                            train_rows=int(train_rows_value or 0),
                            train_time_total_samples=train_time_total_samples,
                            train_throughput_samples=train_throughput_samples,
                        ),
                    }

                reference_time = variant_measurements["reference_rebuild_indices"]["summary"]["aggregate"][
                    "train_time_total"
                ]["mean"]
                optimized_time = variant_measurements["optimized_reuse_precomputed_indices"]["summary"]["aggregate"][
                    "train_time_total"
                ]["mean"]
                reference_throughput = variant_measurements["reference_rebuild_indices"]["summary"]["aggregate"][
                    "ratings_per_second_train"
                ]["mean"]
                optimized_throughput = variant_measurements["optimized_reuse_precomputed_indices"]["summary"][
                    "aggregate"
                ]["ratings_per_second_train"]["mean"]
                model_summary["train_rows"] = int(train_rows_value or 0)
                model_summary["variants"] = variant_measurements
                model_summary["comparison"] = {
                    "train_time_total_speedup_reference_over_optimized": (
                        float(reference_time) / float(optimized_time) if optimized_time > 0.0 else 0.0
                    ),
                    "ratings_per_second_train_speedup_optimized_over_reference": (
                        float(optimized_throughput) / float(reference_throughput) if reference_throughput > 0.0 else 0.0
                    ),
                }
                summary_payload["models"][model_name] = model_summary

        write_json(summary_path, summary_payload)
        markdown_lines = [
            "# Precomputed Index Reuse Compare",
            "",
            f"- benchmark_id: `{benchmark_id}`",
            f"- benchmark_scope: `{benchmark_scope}`",
            f"- dataset: `{dataset_short_name}`",
            f"- models: `{', '.join(model_names)}`",
            f"- measured_repeats: `{measured_repeats}`",
            f"- override_epochs: `{override_epochs}`",
            f"- warmup_policy: `{measurement['warmup_policy']}`",
            "",
        ]
        for model_name, model_summary in summary_payload["models"].items():
            reference = model_summary["variants"]["reference_rebuild_indices"]["summary"]["aggregate"]
            optimized = model_summary["variants"]["optimized_reuse_precomputed_indices"]["summary"]["aggregate"]
            reference_time = reference["train_time_total"]
            reference_throughput = reference["ratings_per_second_train"]
            optimized_time = optimized["train_time_total"]
            optimized_throughput = optimized["ratings_per_second_train"]
            comparison = model_summary["comparison"]
            markdown_lines.extend(
                [
                    f"## {model_name}",
                    "",
                    f"- train_rows: `{model_summary['train_rows']}`",
                    "",
                    "| Variant | Train Time Mean (s) | Train Time CV | Train Ratings/s Mean |",
                    "| --- | ---: | ---: | ---: |",
                    (
                        f"| `reference_rebuild_indices` | {reference_time['mean']:.6f} | "
                        f"{reference_time['coefficient_of_variation']:.6f} | "
                        f"{reference_throughput['mean']:.6f} |"
                    ),
                    (
                        f"| `optimized_reuse_precomputed_indices` | {optimized_time['mean']:.6f} | "
                        f"{optimized_time['coefficient_of_variation']:.6f} | "
                        f"{optimized_throughput['mean']:.6f} |"
                    ),
                    "",
                    (
                        "- train_time_speedup_reference_over_optimized: "
                        f"`{comparison['train_time_total_speedup_reference_over_optimized']:.6f}`"
                    ),
                    (
                        "- throughput_speedup_optimized_over_reference: "
                        f"`{comparison['ratings_per_second_train_speedup_optimized_over_reference']:.6f}`"
                    ),
                    "",
                ]
            )
        summary_md_path.write_text("\n".join(markdown_lines).strip() + "\n", encoding="utf-8", newline="\n")

        finished_at = utc_timestamp()
        write_log(
            stdout_log_path,
            [
                f"[{timestamp}] benchmark_id={benchmark_id}",
                f"command={command_string}",
                f"[{finished_at}] status=completed",
            ],
        )
        completed_manifest = {
            **benchmark_manifest,
            "status": "completed",
            "generated_at_utc": finished_at,
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
            "summary_path": str(summary_path),
        }
    except Exception:
        finished_at = utc_timestamp()
        write_log(
            stdout_log_path,
            [
                f"[{timestamp}] benchmark_id={benchmark_id}",
                f"command={command_string}",
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
