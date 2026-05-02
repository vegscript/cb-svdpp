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
    build_runtime_metadata,
    git_snapshot,
    reserve_timestamped_artifact_dir,
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


def _seed_slug(seed: int) -> str:
    return f"s{seed:03d}"


def _read_seed_benchmark(
    *,
    benchmark_manifest_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    benchmark_manifest = load_json_file(benchmark_manifest_path)
    if benchmark_manifest.get("kind") != "benchmark_manifest":
        raise ValueError(f"expected benchmark_manifest at {benchmark_manifest_path}")
    if benchmark_manifest.get("status") != "completed":
        raise ValueError(f"seed benchmark must be completed: {benchmark_manifest_path}")

    benchmark_dir = benchmark_manifest_path.parent
    summary_path = benchmark_dir / "summary.json"
    config_snapshot_path = benchmark_dir / "config_snapshot.yaml"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json next to benchmark manifest: {benchmark_manifest_path}")
    if not config_snapshot_path.exists():
        raise FileNotFoundError(f"missing config_snapshot.yaml next to benchmark manifest: {benchmark_manifest_path}")

    summary_payload = load_json_file(summary_path)
    config_snapshot = load_yaml_file(config_snapshot_path)
    if str(summary_payload.get("dataset")) != "ml100k":
        raise ValueError("multi-seed benchmark only supports dataset='ml100k'")
    if str(summary_payload.get("split_family")) != "paper_faithful_ml100k_v1":
        raise ValueError("seed benchmark must use split_family='paper_faithful_ml100k_v1'")
    return {
        "manifest": benchmark_manifest,
        "summary": summary_payload,
        "config_snapshot": config_snapshot,
        "manifest_path": benchmark_manifest_path.resolve(),
        "benchmark_dir": benchmark_dir.resolve(),
    }


def _discover_seed_benchmark(
    *,
    repo_root: Path,
    model_name: str,
    processed_manifest_ref: str,
    model_config_ref: str,
    runtime_config_ref: str,
    device_config_ref: str,
    model_seed: int,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for manifest_path in sorted((repo_root / "artifacts" / "benchmarks").glob("*/benchmark_manifest.json")):
        try:
            seed_benchmark = _read_seed_benchmark(benchmark_manifest_path=manifest_path, repo_root=repo_root)
        except Exception:
            continue
        summary_payload = seed_benchmark["summary"]
        config_snapshot = seed_benchmark["config_snapshot"]
        if str(summary_payload.get("model")) != model_name:
            continue
        if list(summary_payload.get("folds", [])) and len(summary_payload["folds"]) != 5:
            continue
        if str(config_snapshot.get("processed_manifest")) != processed_manifest_ref:
            continue
        if str(config_snapshot.get("model_config")) != model_config_ref:
            continue
        if str(config_snapshot.get("runtime_config")) != runtime_config_ref:
            continue
        if str(config_snapshot.get("device_config")) != device_config_ref:
            continue
        if int(config_snapshot.get("model_seed", -1)) != int(model_seed):
            continue
        candidates.append(seed_benchmark)

    if not candidates:
        raise FileNotFoundError(f"no matching seed benchmark found for model_seed={model_seed}")
    if len(candidates) > 1:
        raise ValueError(
            "multiple matching seed benchmarks found for "
            f"model_seed={model_seed}; use explicit benchmark_manifest_paths"
        )
    return candidates[0]


def _validate_selected_seed_benchmark(
    *,
    seed_benchmark: dict[str, Any],
    model_name: str,
    processed_manifest_ref: str,
    model_config_ref: str,
    runtime_config_ref: str,
    device_config_ref: str,
    model_seed: int,
) -> None:
    summary_payload = seed_benchmark["summary"]
    config_snapshot = seed_benchmark["config_snapshot"]

    if str(summary_payload.get("model")) != model_name:
        raise ValueError(f"seed benchmark model mismatch for model_seed={model_seed}")
    if list(summary_payload.get("folds", [])) and len(summary_payload["folds"]) != 5:
        raise ValueError(f"seed benchmark must contain 5 folds for model_seed={model_seed}")
    if str(config_snapshot.get("processed_manifest")) != processed_manifest_ref:
        raise ValueError(f"processed_manifest mismatch for model_seed={model_seed}")
    if str(config_snapshot.get("model_config")) != model_config_ref:
        raise ValueError(f"model_config mismatch for model_seed={model_seed}")
    if str(config_snapshot.get("runtime_config")) != runtime_config_ref:
        raise ValueError(f"runtime_config mismatch for model_seed={model_seed}")
    if str(config_snapshot.get("device_config")) != device_config_ref:
        raise ValueError(f"device_config mismatch for model_seed={model_seed}")
    if int(config_snapshot.get("model_seed", -1)) != int(model_seed):
        raise ValueError(f"model_seed mismatch for selected benchmark: expected {model_seed}")


def _validate_seed_benchmark_bundle(
    *,
    seed_benchmarks: list[dict[str, Any]],
    model_name: str,
    processed_manifest_ref: str,
    model_config_ref: str,
    runtime_config_ref: str,
    device_config_ref: str,
    model_seeds: list[int],
) -> None:
    if len(seed_benchmarks) != len(model_seeds):
        raise ValueError("seed_benchmarks and model_seeds must have the same length")

    reference_git: dict[str, Any] | None = None
    for model_seed, seed_benchmark in zip(model_seeds, seed_benchmarks, strict=True):
        _validate_selected_seed_benchmark(
            seed_benchmark=seed_benchmark,
            model_name=model_name,
            processed_manifest_ref=processed_manifest_ref,
            model_config_ref=model_config_ref,
            runtime_config_ref=runtime_config_ref,
            device_config_ref=device_config_ref,
            model_seed=model_seed,
        )

        benchmark_git = dict(seed_benchmark["manifest"].get("git", {}))
        benchmark_identity = (
            str(benchmark_git.get("commit", "")),
            bool(benchmark_git.get("dirty", False)),
            str(benchmark_git.get("branch", "")),
        )
        if reference_git is None:
            reference_git = {
                "commit": benchmark_identity[0],
                "dirty": benchmark_identity[1],
                "branch": benchmark_identity[2],
            }
            continue

        reference_identity = (
            str(reference_git.get("commit", "")),
            bool(reference_git.get("dirty", False)),
            str(reference_git.get("branch", "")),
        )
        if benchmark_identity != reference_identity:
            raise ValueError("selected seed benchmarks must share identical git commit, branch, and dirty state")


def _resolve_seed_benchmarks(
    *,
    repo_root: Path,
    model_name: str,
    processed_manifest_ref: str,
    model_config_ref: str,
    runtime_config_ref: str,
    device_config_ref: str,
    model_seeds: list[int],
    benchmark_manifest_paths: list[Path] | None,
) -> list[dict[str, Any]]:
    if benchmark_manifest_paths is not None:
        if len(benchmark_manifest_paths) != len(model_seeds):
            raise ValueError("benchmark_manifest_paths must match model_seeds length")
        seed_benchmarks = [
            _read_seed_benchmark(benchmark_manifest_path=manifest_path.resolve(), repo_root=repo_root)
            for manifest_path in benchmark_manifest_paths
        ]
    else:
        seed_benchmarks = [
            _discover_seed_benchmark(
                repo_root=repo_root,
                model_name=model_name,
                processed_manifest_ref=processed_manifest_ref,
                model_config_ref=model_config_ref,
                runtime_config_ref=runtime_config_ref,
                device_config_ref=device_config_ref,
                model_seed=seed,
            )
            for seed in model_seeds
        ]

    _validate_seed_benchmark_bundle(
        seed_benchmarks=seed_benchmarks,
        model_name=model_name,
        processed_manifest_ref=processed_manifest_ref,
        model_config_ref=model_config_ref,
        runtime_config_ref=runtime_config_ref,
        device_config_ref=device_config_ref,
        model_seeds=model_seeds,
    )
    return seed_benchmarks


def run_ml100k_paper_multiseed_benchmark(
    *,
    model_name: str,
    processed_manifest_path: Path,
    model_config_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    model_seeds: list[int],
    benchmark_manifest_paths: list[Path] | None = None,
    repo_root: Path | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    if not model_seeds:
        raise ValueError("model_seeds must contain at least one seed")

    root = (repo_root or discover_repo_root()).resolve()
    processed_manifest_path = processed_manifest_path.resolve()
    model_config_path = model_config_path.resolve()
    runtime_config_path = runtime_config_path.resolve()
    device_config_path = device_config_path.resolve()

    processed_manifest_ref = repo_path_string(processed_manifest_path, repo_root=root)
    model_config_ref = repo_path_string(model_config_path, repo_root=root)
    runtime_config_ref = repo_path_string(runtime_config_path, repo_root=root)
    device_config_ref = repo_path_string(device_config_path, repo_root=root)
    benchmark_manifest_path_refs = (
        [repo_path_string(path.resolve(), repo_root=root) for path in benchmark_manifest_paths]
        if benchmark_manifest_paths is not None
        else None
    )
    device_config_payload = load_yaml_file(device_config_path)
    threading_config = resolve_runtime_threading_config(device_config_payload=device_config_payload)
    device_profile_name = str(device_config_payload["device_profile"]["name"])

    seed_set_slug = "_".join(_seed_slug(seed) for seed in model_seeds)
    timestamp, benchmark_id, benchmark_dir = reserve_timestamped_artifact_dir(
        artifacts_root=root / "artifacts" / "benchmarks",
        id_from_timestamp=lambda reserved_timestamp: "_".join(
            [
                reserved_timestamp,
                "ml100k",
                "paper_faithful",
                model_name,
                "multiseed",
                seed_set_slug,
                device_profile_name,
            ]
        ),
    )

    summary_path = benchmark_dir / "summary.json"
    summary_md_path = benchmark_dir / "summary.md"
    stdout_log_path = benchmark_dir / "stdout.log"
    benchmark_manifest_path = benchmark_dir / "benchmark_manifest.json"
    config_snapshot_path = benchmark_dir / "config_snapshot.yaml"

    benchmark_scope = f"paper_faithful_ml100k_v1_{model_name}_u1_u5_multiseed_{seed_set_slug}"
    command_string = command or (
        "recsys-lab benchmark-ml100k-paper-multiseed "
        f"--model {model_name} "
        f"--processed-manifest {processed_manifest_ref} "
        f"--model-config {model_config_ref} "
        f"--runtime-config {runtime_config_ref} "
        f"--device-config {device_config_ref} "
        f"--model-seeds {','.join(str(seed) for seed in model_seeds)}"
        + (
            f" --benchmark-manifest-paths {','.join(benchmark_manifest_path_refs)}"
            if benchmark_manifest_path_refs is not None
            else ""
        )
    )

    git = git_snapshot(root)
    measurement = build_benchmark_measurement(
        time_metric="training_wall_clock_seconds",
        time_metric_semantics=(
            "The primary time aggregate summarizes per-seed benchmark means in seconds. "
            "Contributing seed benchmarks use the paper benchmark fit-time semantics, "
            "including cluster induction when present."
        ),
        sample_unit="seed_benchmark",
        measured_sample_count=len(model_seeds),
        warmup_policy="none",
        warmup_sample_count=0,
        notes=[
            "No additional multiseed warmup runs are executed; this artifact aggregates completed seed benchmarks.",
            "fold_run_level aggregates flatten all contributing fold runs to expose cross-seed timing dispersion.",
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
                runtime_dtype=str(device_config_payload["precision"]["default_dtype"]),
                device_config_payload=device_config_payload,
            ),
            "measurement": measurement,
            "inputs": {
                "run_ids": [],
                "run_manifest_paths": [],
                "benchmark_ids": [],
                "benchmark_manifest_paths": benchmark_manifest_path_refs or [],
                "model_seeds": [int(seed) for seed in model_seeds],
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
                "model_seeds": [int(seed) for seed in model_seeds],
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
            seed_benchmarks = _resolve_seed_benchmarks(
                repo_root=root,
                model_name=model_name,
                processed_manifest_ref=processed_manifest_ref,
                model_config_ref=model_config_ref,
                runtime_config_ref=runtime_config_ref,
                device_config_ref=device_config_ref,
                model_seeds=model_seeds,
                benchmark_manifest_paths=benchmark_manifest_paths,
            )

            run_ids: list[str] = []
            run_manifest_paths: list[str] = []
            benchmark_ids: list[str] = []
            contributing_benchmark_manifest_paths: list[str] = []
            per_seed: list[dict[str, Any]] = []
            all_fold_train: list[float] = []
            all_fold_test: list[float] = []
            all_fold_time: list[float] = []
            seed_mean_train: list[float] = []
            seed_mean_test: list[float] = []
            seed_mean_time: list[float] = []

            for seed, seed_benchmark in zip(model_seeds, seed_benchmarks, strict=True):
                seed_summary_payload = seed_benchmark["summary"]
                manifest_payload = seed_benchmark["manifest"]
                benchmark_ids.append(str(manifest_payload["benchmark_id"]))
                contributing_benchmark_manifest_paths.append(
                    repo_path_string(seed_benchmark["manifest_path"], repo_root=root)
                )
                run_ids.extend(str(run_id) for run_id in manifest_payload.get("inputs", {}).get("run_ids", []))
                run_manifest_paths.extend(
                    str(path_ref) for path_ref in manifest_payload.get("inputs", {}).get("run_manifest_paths", [])
                )

                folds = list(seed_summary_payload["folds"])
                fold_train = [float(item["train_rmse"]) for item in folds]
                fold_test = [float(item["test_rmse"]) for item in folds]
                fold_time = [float(item["training_wall_clock_seconds"]) for item in folds]
                all_fold_train.extend(fold_train)
                all_fold_test.extend(fold_test)
                all_fold_time.extend(fold_time)
                seed_mean_train.append(float(seed_summary_payload["aggregate"]["train_rmse"]["mean"]))
                seed_mean_test.append(float(seed_summary_payload["aggregate"]["test_rmse"]["mean"]))
                seed_mean_time.append(float(seed_summary_payload["aggregate"]["training_wall_clock_seconds"]["mean"]))

                per_seed.append(
                    {
                        "model_seed": int(seed),
                        "benchmark_id": str(manifest_payload["benchmark_id"]),
                        "benchmark_manifest_path": repo_path_string(seed_benchmark["manifest_path"], repo_root=root),
                        "test_rmse": dict(seed_summary_payload["aggregate"]["test_rmse"]),
                        "train_rmse": dict(seed_summary_payload["aggregate"]["train_rmse"]),
                        "training_wall_clock_seconds": dict(
                            seed_summary_payload["aggregate"]["training_wall_clock_seconds"]
                        ),
                    }
                )

            summary_payload: dict[str, Any] = {
                "benchmark_id": benchmark_id,
                "benchmark_scope": benchmark_scope,
                "dataset": "ml100k",
                "split_family": "paper_faithful_ml100k_v1",
                "model": model_name,
                "model_seeds": [int(seed) for seed in model_seeds],
                "measurement": measurement,
                "per_seed": per_seed,
                "aggregate": {
                    "seed_level": {
                        "train_rmse": summarize_scalar_samples(seed_mean_train),
                        "test_rmse": summarize_scalar_samples(seed_mean_test),
                        "training_wall_clock_seconds": summarize_scalar_samples(seed_mean_time),
                    },
                    "fold_run_level": {
                        "train_rmse": summarize_scalar_samples(all_fold_train),
                        "test_rmse": summarize_scalar_samples(all_fold_test),
                        "training_wall_clock_seconds": summarize_scalar_samples(all_fold_time),
                    },
                },
            }
            write_json(summary_path, summary_payload)

            markdown_lines = [
                "# Multi-Seed Benchmark Summary",
                "",
                f"- benchmark_id: `{benchmark_id}`",
                f"- benchmark_scope: `{benchmark_scope}`",
                "- dataset: `ml100k`",
                "- split_family: `paper_faithful_ml100k_v1`",
                f"- model: `{model_name}`",
                f"- model_seeds: `{', '.join(str(seed) for seed in model_seeds)}`",
                f"- warmup_policy: `{measurement['warmup_policy']}`",
                f"- measured_sample_count: `{measurement['measured_sample_count']}`",
                "",
                "## Per Seed",
                "",
                "| Seed | Benchmark ID | Test RMSE Mean | Test RMSE Std | Train Time Mean (s) |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
            for item in per_seed:
                markdown_lines.append(
                    f"| `{item['model_seed']}` | `{item['benchmark_id']}` | "
                    f"{item['test_rmse']['mean']:.6f} | {item['test_rmse']['std']:.6f} | "
                    f"{item['training_wall_clock_seconds']['mean']:.2f} |"
                )
            seed_level = summary_payload["aggregate"]["seed_level"]
            fold_run_level = summary_payload["aggregate"]["fold_run_level"]
            seed_level_time = seed_level["training_wall_clock_seconds"]
            markdown_lines.extend(
                [
                    "",
                    "## Aggregate",
                    "",
                    f"- seed_level test_rmse count: `{seed_level['test_rmse']['count']}`",
                    f"- seed_level test_rmse mean: `{seed_level['test_rmse']['mean']:.6f}`",
                    f"- seed_level test_rmse std: `{seed_level['test_rmse']['std']:.6f}`",
                    (
                        "- seed_level training_wall_clock_seconds cv: "
                        f"`{seed_level_time['coefficient_of_variation']:.6f}`"
                    ),
                    f"- fold_run_level test_rmse mean: `{fold_run_level['test_rmse']['mean']:.6f}`",
                    f"- fold_run_level test_rmse std: `{fold_run_level['test_rmse']['std']:.6f}`",
                ]
            )
            summary_md_path.write_text("\n".join(markdown_lines).strip() + "\n", encoding="utf-8", newline="\n")

            finished_at = utc_timestamp()
            write_log(
                stdout_log_path,
                [
                    f"[{timestamp}] benchmark_id={benchmark_id}",
                    f"command={command_string}",
                    "allow_existing_run_reuse=false",
                    f"seed_count={len(model_seeds)}",
                    f"test_rmse_seed_level_mean={summary_payload['aggregate']['seed_level']['test_rmse']['mean']:.6f}",
                    f"[{finished_at}] status=completed",
                ],
            )
            completed_manifest = {
                **benchmark_manifest,
                "status": "completed",
                "generated_at_utc": finished_at,
                "inputs": {
                    "run_ids": run_ids,
                    "run_manifest_paths": run_manifest_paths,
                    "benchmark_ids": benchmark_ids,
                    "benchmark_manifest_paths": contributing_benchmark_manifest_paths,
                    "model_seeds": [int(seed) for seed in model_seeds],
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
                "seed_level_test_rmse_mean": summary_payload["aggregate"]["seed_level"]["test_rmse"]["mean"],
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
