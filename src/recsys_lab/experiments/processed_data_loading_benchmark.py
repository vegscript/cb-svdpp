from __future__ import annotations

import json
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from recsys_lab.config.loader import dump_yaml_file, load_yaml_file
from recsys_lab.data.processed import (
    build_interaction_array_manifest_payload,
    load_processed_dataset_manifest,
    load_ratings_data_from_manifest,
    materialize_interaction_array_artifacts_from_manifest,
)
from recsys_lab.experiments.benchmarking import (
    build_benchmark_measurement,
    summarize_scalar_samples,
)
from recsys_lab.experiments.common import (
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
from recsys_lab.utils.manifests import validate_manifest_file
from recsys_lab.utils.paths import discover_repo_root, repo_path_string


def _construct_only_readout(path: Path) -> tuple[int, float]:
    data = load_ratings_data_from_manifest(path, mmap_mode="r", prefer_interaction_arrays=True)
    row_count = len(data)
    sentinel = float(data.ratings[0]) if row_count > 0 else 0.0
    return row_count, sentinel


def _full_scan_checksum_readout(path: Path) -> tuple[int, float]:
    data = load_ratings_data_from_manifest(path, mmap_mode="r", prefer_interaction_arrays=True)
    checksum = float(
        np.sum(np.asarray(data.user_ids, dtype=np.int64))
        + np.sum(np.asarray(data.item_ids, dtype=np.int64))
        + np.sum(np.asarray(data.ratings, dtype=np.float64))
    )
    return len(data), checksum


def _time_loading_variant(
    *,
    manifest_path: Path,
    repeats: int,
    readout_fn: Any,
) -> tuple[list[float], int, float]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    warmup_rows, warmup_readout = readout_fn(manifest_path)
    timings: list[float] = []
    final_rows = warmup_rows
    final_readout = warmup_readout

    for _ in range(repeats):
        started = perf_counter()
        final_rows, final_readout = readout_fn(manifest_path)
        timings.append(perf_counter() - started)
    return timings, final_rows, final_readout


def _summarize_loading_variant(row_count: int, timings: list[float]) -> dict[str, Any]:
    return {
        "aggregate": {
            "data_load_wall_clock_seconds": summarize_scalar_samples(timings),
            "rows_per_second": summarize_scalar_samples([row_count / value for value in timings]),
        }
    }


def run_processed_data_loading_benchmark(
    *,
    processed_manifest_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    measured_repeats: int = 3,
    repo_root: Path | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    if measured_repeats <= 0:
        raise ValueError("measured_repeats must be positive")

    root = (repo_root or discover_repo_root()).resolve()
    processed_manifest_path = processed_manifest_path.resolve()
    runtime_config_path = runtime_config_path.resolve()
    device_config_path = device_config_path.resolve()

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
            "processed_data_loading_compare",
            device_profile_name,
        ]
    )
    benchmark_scope = f"development_{dataset_short_name}_processed_data_loading_compare"
    benchmark_dir = root / "artifacts" / "benchmarks" / benchmark_id
    benchmark_dir.mkdir(parents=True, exist_ok=False)
    config_dir = benchmark_dir / "configs"
    input_cache_dir = benchmark_dir / "input_cache"
    config_dir.mkdir(parents=True, exist_ok=False)
    input_cache_dir.mkdir(parents=True, exist_ok=False)

    summary_path = benchmark_dir / "summary.json"
    summary_md_path = benchmark_dir / "summary.md"
    stdout_log_path = benchmark_dir / "stdout.log"
    benchmark_manifest_path = benchmark_dir / "benchmark_manifest.json"
    config_snapshot_path = benchmark_dir / "config_snapshot.yaml"

    command_string = command or (
        "development benchmark processed data loading compare "
        f"--processed-manifest {repo_path_string(processed_manifest_path, repo_root=root)} "
        f"--runtime-config {repo_path_string(runtime_config_path, repo_root=root)} "
        f"--device-config {repo_path_string(device_config_path, repo_root=root)}"
    )
    measurement = build_benchmark_measurement(
        time_metric="data_load_wall_clock_seconds",
        time_metric_semantics=(
            "Per-variant processed-data load time. "
            "Two scenarios are measured: construct_only and full_scan_checksum. "
            "Each scenario executes separate unmeasured warmups for both variants and then measured repeats."
        ),
        sample_unit="scenario_variant_data_load_run",
        measured_sample_count=measured_repeats,
        warmup_policy="separate_unmeasured",
        warmup_sample_count=1,
        notes=[
            "This is a development data-loading benchmark, not a benchmark-final claim.",
            (
                "The memmap variant compares processed manifest loading with interaction array sidecars "
                "against parquet-only loading."
            ),
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

    reference_manifest_payload = json.loads(json.dumps(processed_manifest))
    reference_manifest_payload["artifacts"].pop("interaction_arrays", None)
    reference_manifest_path = config_dir / "reference_processed_manifest.json"
    reference_manifest_path.write_text(
        json.dumps(reference_manifest_payload, indent=2),
        encoding="utf-8",
        newline="\n",
    )

    interaction_arrays = materialize_interaction_array_artifacts_from_manifest(
        processed_manifest_path,
        output_dir=input_cache_dir,
        prefix=f"{dataset_short_name}_loading_compare",
    )
    optimized_manifest_payload = json.loads(json.dumps(reference_manifest_payload))
    optimized_manifest_payload["artifacts"]["interaction_arrays"] = build_interaction_array_manifest_payload(
        interaction_arrays
    )
    optimized_manifest_path = config_dir / "optimized_processed_manifest.json"
    optimized_manifest_path.write_text(
        json.dumps(optimized_manifest_payload, indent=2),
        encoding="utf-8",
        newline="\n",
    )

    dump_yaml_file(
        config_snapshot_path,
        {
            "benchmark_id": benchmark_id,
            "benchmark_scope": benchmark_scope,
            "processed_manifest": repo_path_string(processed_manifest_path, repo_root=root),
            "runtime_config": repo_path_string(runtime_config_path, repo_root=root),
            "device_config": repo_path_string(device_config_path, repo_root=root),
            "reference_manifest": repo_path_string(reference_manifest_path, repo_root=root),
            "optimized_manifest": repo_path_string(optimized_manifest_path, repo_root=root),
            "measured_repeats": measured_repeats,
            "loaded_configs": {
                "processed_manifest": processed_manifest,
                "runtime": runtime_config_payload,
                "device": device_config_payload,
            },
        },
    )
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
            scenarios = [
                ("construct_only", _construct_only_readout),
                ("full_scan_checksum", _full_scan_checksum_readout),
            ]

            summary_payload: dict[str, Any] = {
                "benchmark_id": benchmark_id,
                "benchmark_scope": benchmark_scope,
                "dataset": dataset_short_name,
                "measurement": measurement,
                "scenarios": {},
            }

            for scenario_name, readout_fn in scenarios:
                reference_timings, reference_rows, reference_readout = _time_loading_variant(
                    manifest_path=reference_manifest_path,
                    repeats=measured_repeats,
                    readout_fn=readout_fn,
                )
                optimized_timings, optimized_rows, optimized_readout = _time_loading_variant(
                    manifest_path=optimized_manifest_path,
                    repeats=measured_repeats,
                    readout_fn=readout_fn,
                )
                if reference_rows != optimized_rows:
                    raise RuntimeError("reference and optimized loading variants produced different row counts")

                summary_payload["scenarios"][scenario_name] = {
                    "row_count": reference_rows,
                    "variants": {
                        "reference_parquet": _summarize_loading_variant(reference_rows, reference_timings),
                        "optimized_memmap_arrays": _summarize_loading_variant(optimized_rows, optimized_timings),
                    },
                    "readout": {
                        "reference_value": float(reference_readout),
                        "optimized_value": float(optimized_readout),
                        "delta": float(optimized_readout - reference_readout),
                    },
                    "comparison": {
                        "data_load_wall_clock_seconds_speedup_reference_over_optimized": float(
                            summarize_scalar_samples(reference_timings)["mean"]
                            / summarize_scalar_samples(optimized_timings)["mean"]
                        ),
                        "rows_per_second_speedup_optimized_over_reference": float(
                            summarize_scalar_samples([optimized_rows / value for value in optimized_timings])["mean"]
                            / summarize_scalar_samples([reference_rows / value for value in reference_timings])["mean"]
                        ),
                    },
                }

        write_json(summary_path, summary_payload)
        markdown_lines = [
            "# Processed Data Loading Compare",
            "",
            f"- benchmark_id: `{benchmark_id}`",
            f"- benchmark_scope: `{benchmark_scope}`",
            f"- dataset: `{dataset_short_name}`",
            f"- measured_repeats: `{measured_repeats}`",
            f"- warmup_policy: `{measurement['warmup_policy']}`",
            "",
        ]
        for scenario_name, scenario_payload in summary_payload["scenarios"].items():
            reference = scenario_payload["variants"]["reference_parquet"]["aggregate"]
            optimized = scenario_payload["variants"]["optimized_memmap_arrays"]["aggregate"]
            reference_time = reference["data_load_wall_clock_seconds"]
            reference_throughput = reference["rows_per_second"]
            optimized_time = optimized["data_load_wall_clock_seconds"]
            optimized_throughput = optimized["rows_per_second"]
            comparison = scenario_payload["comparison"]
            markdown_lines.extend(
                [
                    f"## {scenario_name}",
                    "",
                    f"- row_count: `{scenario_payload['row_count']}`",
                    f"- readout_delta: `{scenario_payload['readout']['delta']:.12f}`",
                    "",
                    "| Variant | Time Mean (s) | Time CV | Rows/s Mean |",
                    "| --- | ---: | ---: | ---: |",
                    (
                        f"| `reference_parquet` | {reference_time['mean']:.6f} | "
                        f"{reference_time['coefficient_of_variation']:.6f} | "
                        f"{reference_throughput['mean']:.6f} |"
                    ),
                    (
                        f"| `optimized_memmap_arrays` | {optimized_time['mean']:.6f} | "
                        f"{optimized_time['coefficient_of_variation']:.6f} | "
                        f"{optimized_throughput['mean']:.6f} |"
                    ),
                    "",
                    (
                        "- time_speedup_reference_over_optimized: "
                        f"`{comparison['data_load_wall_clock_seconds_speedup_reference_over_optimized']:.6f}`"
                    ),
                    (
                        "- throughput_speedup_optimized_over_reference: "
                        f"`{comparison['rows_per_second_speedup_optimized_over_reference']:.6f}`"
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
