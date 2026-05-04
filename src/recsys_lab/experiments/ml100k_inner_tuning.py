from __future__ import annotations

import copy
import hashlib
import traceback
from pathlib import Path
from typing import Any, Callable

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
    utc_timestamp,
    write_json,
    write_log,
)
from recsys_lab.experiments.runtime import (
    resolve_runtime_threading_config,
    runtime_execution_context,
)
from recsys_lab.experiments.unified_runner import run_unified_experiment
from recsys_lab.models.registry import MODEL_REGISTRY, validate_model_config_payload
from recsys_lab.utils.manifests import load_json_file, validate_manifest_file
from recsys_lab.utils.paths import discover_repo_root, repo_path_string

SUPPORTED_MODELS = set(MODEL_REGISTRY)


def _runner_for_model(model_name: str) -> Callable[..., dict[str, Any]]:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"unsupported tuning model: {model_name}")

    def _run(**kwargs: Any) -> dict[str, Any]:
        return run_unified_experiment(model_name=model_name, **kwargs)

    return _run


def _merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in merged and isinstance(value, dict) and isinstance(merged[key], dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _candidate_config_filename(candidate_id: str, *, candidate_index: int) -> str:
    safe_candidate_id = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in candidate_id
    )
    if len(safe_candidate_id) <= 24:
        return f"{safe_candidate_id}.yaml"
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:12]
    return f"c{candidate_index:03d}_{digest}.yaml"


def _fit_seconds(metrics: dict[str, Any]) -> float:
    timing = metrics["timing"]
    fit_seconds = float(timing["training_wall_clock_seconds"])
    cluster_seconds = timing.get("cluster_induction_wall_clock_seconds")
    if cluster_seconds is not None:
        fit_seconds += float(cluster_seconds)
    return fit_seconds


def _peak_memory_mb(metrics: dict[str, Any]) -> float | None:
    peak_memory_mb = metrics.get("system_metrics", {}).get("peak_memory_mb")
    if peak_memory_mb is None:
        return None
    return float(peak_memory_mb)


def _candidate_sort_key(candidate_summary: dict[str, Any]) -> tuple[float, float, float]:
    aggregate = candidate_summary["aggregate"]
    return (
        float(aggregate["validation_rmse"]["mean"]),
        float(aggregate["validation_rmse"]["std"]),
        float(aggregate["training_wall_clock_seconds"]["mean"]),
    )


def _read_run_metrics(run_manifest_path: Path, *, repo_root: Path) -> dict[str, Any]:
    run_manifest = load_json_file(run_manifest_path)
    metrics_ref = str(run_manifest["artifacts"]["metrics"])
    metrics_path = (repo_root / metrics_ref).resolve()
    return load_json_file(metrics_path)


def _resolve_selection_units(
    *,
    tuning: dict[str, Any],
    dataset_short_name: str,
    split_family: str,
) -> tuple[list[int], list[str], str, int | None]:
    if split_family == "paper_faithful_ml100k_inner_v1":
        if dataset_short_name != "ml100k":
            raise ValueError("paper_faithful_ml100k_inner_v1 requires dataset_short_name='ml100k'")
        if "folds" not in tuning:
            raise ValueError("tuning.folds is required for paper_faithful_ml100k_inner_v1")
        units = [int(fold) for fold in tuning["folds"]]
        if not units:
            raise ValueError("tuning.folds must contain at least one fold")
        inner_seed = int(tuning["inner_seed"])
        return units, [f"u{fold}" for fold in units], "official_fold", inner_seed

    if split_family == "benchmark_random_v1":
        if "split_seeds" in tuning:
            raw_values = tuning["split_seeds"]
        elif "folds" in tuning:
            raw_values = tuning["folds"]
        else:
            raise ValueError("tuning.split_seeds is required for benchmark_random_v1")
        units = [int(seed) for seed in raw_values]
        if not units:
            raise ValueError("tuning.split_seeds must contain at least one split seed")
        return units, [seed_slug(seed) for seed in units], "split_seed", None

    raise ValueError(f"unsupported tuning split_family: {split_family}")


def run_inner_tuning(
    *,
    tuning_config_path: Path,
    processed_manifest_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    use_split_cache: bool | None = None,
    use_training_index_cache: bool = False,
    use_cluster_artifact_cache: bool = False,
    repo_root: Path | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    root = (repo_root or discover_repo_root()).resolve()
    tuning_config_path = tuning_config_path.resolve()
    processed_manifest_path = processed_manifest_path.resolve()
    runtime_config_path = runtime_config_path.resolve()
    device_config_path = device_config_path.resolve()

    tuning_payload = load_yaml_file(tuning_config_path)
    tuning = tuning_payload["tuning"]
    if not isinstance(tuning, dict):
        raise TypeError("tuning config field 'tuning' must be a mapping")
    processed_manifest = load_json_file(processed_manifest_path)
    runtime_config_payload = load_yaml_file(runtime_config_path)
    device_config_payload = load_yaml_file(device_config_path)
    threading_config = resolve_runtime_threading_config(device_config_payload=device_config_payload)

    dataset_short_name = str(processed_manifest["dataset_short_name"])
    if str(tuning["dataset_short_name"]) != dataset_short_name:
        raise ValueError("tuning.dataset_short_name must match processed manifest dataset_short_name")
    requested_split_family = str(tuning["split_family"])
    selection_units, selection_labels, selection_unit_kind, inner_seed = _resolve_selection_units(
        tuning=tuning,
        dataset_short_name=dataset_short_name,
        split_family=requested_split_family,
    )

    base_model_config_ref = str(tuning_payload["base_model_config"])
    base_model_config_path = (root / base_model_config_ref).resolve()
    base_model_config_payload = load_yaml_file(base_model_config_path)
    adapter, _ = validate_model_config_payload(base_model_config_payload)
    model_name = adapter.name
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported tuning model: {model_name}")
    if use_training_index_cache and not (
        adapter.requirements.needs_implicit_history or adapter.requirements.needs_explicit_feedback
    ):
        raise ValueError(f"training-index cache is not applicable to model '{model_name}'")
    if use_cluster_artifact_cache and not adapter.requirements.needs_cluster_artifacts:
        raise ValueError(f"cluster-artifact cache is not applicable to model '{model_name}'")

    candidate_payloads = list(tuning_payload["candidates"])
    if not candidate_payloads:
        raise ValueError("tuning config must contain at least one candidate")
    validation_ratio = float(tuning["validation_ratio"])
    if requested_split_family == "paper_faithful_ml100k_inner_v1":
        train_ratio = 1.0 - validation_ratio
    elif requested_split_family == "benchmark_random_v1":
        train_ratio = float(tuning["train_ratio"])
    else:
        raise ValueError(f"unsupported tuning split_family: {requested_split_family}")
    model_seed = int(tuning["model_seed"])
    selection_stage = str(tuning["selection_stage"])
    benchmark_scope = "_".join(["tuning", dataset_short_name, requested_split_family, model_name, selection_stage])

    device_profile_name = str(device_config_payload["device_profile"]["name"])
    raw_resource_gate = tuning_payload["resource_gate"] if "resource_gate" in tuning_payload else None
    resource_gate_payload: dict[str, Any] | None = None
    max_peak_memory_mb: float | None = None
    reject_candidate_on_guardrail_breach = False
    if raw_resource_gate is not None:
        resource_gate_payload = dict(raw_resource_gate)
        expected_device_profile = (
            resource_gate_payload["device_profile"] if "device_profile" in resource_gate_payload else None
        )
        if expected_device_profile is not None and str(expected_device_profile) != device_profile_name:
            raise ValueError("resource_gate.device_profile must match the selected device config")
        max_peak_memory_mb = float(resource_gate_payload["max_peak_memory_mb"])
        reject_candidate_on_guardrail_breach = (
            bool(resource_gate_payload["reject_candidate_on_any_guardrail_breach"])
            if "reject_candidate_on_any_guardrail_breach" in resource_gate_payload
            else False
        )
    timestamp, benchmark_id, benchmark_dir = reserve_timestamped_artifact_dir(
        artifacts_root=root / "artifacts" / "benchmarks",
        id_from_timestamp=lambda reserved_timestamp: "_".join(
            [
                reserved_timestamp,
                dataset_short_name,
                "inner_tuning",
                model_name,
                selection_stage,
                device_profile_name,
            ]
        ),
    )
    candidate_dir = benchmark_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=False)

    summary_path = benchmark_dir / "summary.json"
    summary_md_path = benchmark_dir / "summary.md"
    stdout_log_path = benchmark_dir / "stdout.log"
    benchmark_manifest_path = benchmark_dir / "benchmark_manifest.json"
    config_snapshot_path = benchmark_dir / "config_snapshot.yaml"

    git = git_snapshot(root)
    split_cache_requested = "auto" if use_split_cache is None else "enable" if use_split_cache else "disable"
    cache_policy_payload: dict[str, Any] = {
        "split_cache": {
            "requested": split_cache_requested,
            "override": use_split_cache,
        },
        "training_index_cache": {
            "enabled": use_training_index_cache,
            "supported_for_model": adapter.requirements.needs_implicit_history
            or adapter.requirements.needs_explicit_feedback,
        },
        "cluster_artifact_cache": {
            "enabled": use_cluster_artifact_cache,
            "supported_for_model": adapter.requirements.needs_cluster_artifacts,
        },
    }
    if use_split_cache is None:
        split_cache_command_fragment = ""
    else:
        split_cache_mode = "enable" if use_split_cache else "disable"
        split_cache_command_fragment = f"--split-cache {split_cache_mode} "
    training_index_cache_command_fragment = (
        "--training-index-cache " if use_training_index_cache else "--disable-training-index-cache "
    )
    cluster_artifact_cache_command_fragment = (
        "--cluster-artifact-cache " if use_cluster_artifact_cache else "--disable-cluster-artifact-cache "
    )
    command_string = command or (
        "recsys-lab tune-inner "
        f"{repo_path_string(tuning_config_path, repo_root=root)} "
        f"{repo_path_string(processed_manifest_path, repo_root=root)} "
        f"--runtime-config {repo_path_string(runtime_config_path, repo_root=root)} "
        f"--device-config {repo_path_string(device_config_path, repo_root=root)} "
        f"{split_cache_command_fragment}"
        f"{training_index_cache_command_fragment}"
        f"{cluster_artifact_cache_command_fragment}"
    )
    measurement = build_benchmark_measurement(
        time_metric="training_wall_clock_seconds",
        time_metric_semantics=(
            "Per-candidate fit-time aggregates summarize validation-selection runs. "
            "If cluster_induction_wall_clock_seconds is present, it is added for fair cb_* candidate comparison."
        ),
        sample_unit="inner_tuning_run",
        measured_sample_count=len(selection_units),
        warmup_policy="none",
        warmup_sample_count=0,
        notes=[
            "No separate warmup runs are executed during inner tuning; each candidate selection run is measured once.",
            "Dispersion across candidate timings is reported via std and coefficient_of_variation.",
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
                "tuning_config": repo_path_string(tuning_config_path, repo_root=root),
                "processed_manifest": repo_path_string(processed_manifest_path, repo_root=root),
                "runtime_config": repo_path_string(runtime_config_path, repo_root=root),
                "device_config": repo_path_string(device_config_path, repo_root=root),
                "cache_policy": cache_policy_payload,
                "resource_gate": resource_gate_payload,
                "loaded_configs": {
                    "tuning": tuning_payload,
                    "processed_manifest": processed_manifest,
                    "runtime": runtime_config_payload,
                    "device": device_config_payload,
                    "base_model": base_model_config_payload,
                },
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

        run_manifest_paths: list[Path] = []
        try:
            runner = _runner_for_model(model_name)
            candidate_summaries: list[dict[str, Any]] = []

            for candidate_index, candidate in enumerate(candidate_payloads, start=1):
                candidate_id = str(candidate["candidate_id"])
                candidate_overrides = dict(candidate["overrides"]) if "overrides" in candidate else {}
                effective_model_config = _merge_dicts(base_model_config_payload, candidate_overrides)
                metadata = dict(effective_model_config["metadata"])
                metadata["status"] = "stage1_tuning_candidate"
                metadata["owner"] = "repo"
                metadata["purpose"] = "tuning_candidate_profile"
                provenance = dict(metadata.get("provenance") or {})
                provenance["candidate_id"] = candidate_id
                metadata["provenance"] = provenance
                effective_model_config["metadata"] = metadata

                candidate_config_path = candidate_dir / _candidate_config_filename(
                    candidate_id,
                    candidate_index=candidate_index,
                )
                dump_yaml_file(candidate_config_path, effective_model_config)

                fold_results: list[dict[str, Any]] = []
                resource_gate_breaches: list[dict[str, Any]] = []
                for selection_unit, selection_label in zip(selection_units, selection_labels, strict=True):
                    runner_kwargs = {
                        "processed_manifest_path": processed_manifest_path,
                        "model_config_path": candidate_config_path,
                        "runtime_config_path": runtime_config_path,
                        "device_config_path": device_config_path,
                        "split_config": SplitConfig(
                            train_ratio=train_ratio,
                            validation_ratio=validation_ratio,
                            seed=selection_unit,
                        ),
                        "model_seed": model_seed,
                        "repo_root": root,
                        "split_family": requested_split_family,
                        "inner_validation_seed": inner_seed,
                        "evaluate_test": False,
                        "use_split_cache": use_split_cache,
                        "command": (
                            "recsys-lab tune-inner "
                            f"--tuning-config {repo_path_string(tuning_config_path, repo_root=root)} "
                            f"--candidate-id {candidate_id} --selection-unit {selection_label}"
                        ),
                    }
                    candidate_adapter, _ = validate_model_config_payload(
                        effective_model_config,
                        expected_model_name=model_name,
                    )
                    if (
                        candidate_adapter.requirements.needs_implicit_history
                        or candidate_adapter.requirements.needs_explicit_feedback
                    ):
                        runner_kwargs["use_training_index_cache"] = use_training_index_cache
                    if candidate_adapter.requirements.needs_cluster_artifacts:
                        runner_kwargs["use_cluster_artifact_cache"] = use_cluster_artifact_cache
                    payload = runner(**runner_kwargs)
                    run_manifest_path = Path(str(payload["run_manifest"])).resolve()
                    run_manifest_paths.append(run_manifest_path)
                    metrics = _read_run_metrics(run_manifest_path, repo_root=root)
                    peak_memory_mb = _peak_memory_mb(metrics)
                    resource_gate_passed = (
                        max_peak_memory_mb is None
                        or (peak_memory_mb is not None and peak_memory_mb <= max_peak_memory_mb)
                    )
                    fold_results.append(
                        {
                            "selection_unit": selection_label,
                            "fold": selection_label,
                            "run_id": str(metrics["run_id"]),
                            "train_rmse": float(metrics["metrics"]["train_rmse"]),
                            "validation_rmse": float(metrics["metrics"]["validation_rmse"]),
                            "training_wall_clock_seconds": _fit_seconds(metrics),
                            "resource_gate": {
                                "enabled": resource_gate_payload is not None,
                                "passed": resource_gate_passed,
                                "peak_memory_mb": peak_memory_mb,
                                "max_peak_memory_mb": max_peak_memory_mb,
                            },
                        }
                    )
                    if not resource_gate_passed:
                        resource_gate_breaches.append(
                            {
                                "selection_unit": selection_label,
                                "run_id": str(metrics["run_id"]),
                                "peak_memory_mb": peak_memory_mb,
                                "max_peak_memory_mb": max_peak_memory_mb,
                            }
                        )
                        if reject_candidate_on_guardrail_breach:
                            break

                validation_values = [item["validation_rmse"] for item in fold_results]
                train_values = [item["train_rmse"] for item in fold_results]
                training_values = [item["training_wall_clock_seconds"] for item in fold_results]
                resource_gate_passed = not resource_gate_breaches and len(fold_results) == len(selection_units)
                candidate_summaries.append(
                    {
                        "candidate_id": candidate_id,
                        "candidate_config": repo_path_string(candidate_config_path, repo_root=root),
                        "measurement": measurement,
                        "folds": fold_results,
                        "selection_unit_kind": selection_unit_kind,
                        "resource_gate": {
                            "enabled": resource_gate_payload is not None,
                            "passed": resource_gate_passed,
                            "breaches": resource_gate_breaches,
                        },
                        "aggregate": {
                            "validation_rmse": summarize_scalar_samples(validation_values),
                            "train_rmse": summarize_scalar_samples(train_values),
                            "training_wall_clock_seconds": summarize_scalar_samples(training_values),
                        },
                    }
                )

            eligible_candidates = [
                candidate_summary
                for candidate_summary in candidate_summaries
                if candidate_summary["resource_gate"]["passed"]
            ]
            if resource_gate_payload is not None and not eligible_candidates:
                raise ValueError("no tuning candidate satisfied the configured resource gate")
            ordered_candidates = sorted(eligible_candidates, key=_candidate_sort_key)
            best_candidate = ordered_candidates[0]

            summary_payload: dict[str, Any] = {
                "benchmark_id": benchmark_id,
                "benchmark_scope": benchmark_scope,
                "dataset": dataset_short_name,
                "split_family": requested_split_family,
                "selection_stage": selection_stage,
                "selection_unit_kind": selection_unit_kind,
                "selection_units": selection_labels,
                "model": model_name,
                "folds": selection_labels,
                "validation_ratio": validation_ratio,
                "train_ratio": train_ratio,
                "inner_seed": inner_seed,
                "model_seed": model_seed,
                "cache_policy": cache_policy_payload,
                "resource_gate": resource_gate_payload,
                "measurement": measurement,
                "objective": "validation_rmse_mean",
                "best_candidate": best_candidate,
                "candidates": ordered_candidates,
                "all_candidates": candidate_summaries,
            }
            write_json(summary_path, summary_payload)

            markdown_lines = [
                "# Inner Tuning Summary",
                "",
                f"- benchmark_id: `{benchmark_id}`",
                f"- benchmark_scope: `{benchmark_scope}`",
                f"- dataset: `{dataset_short_name}`",
                f"- split_family: `{requested_split_family}`",
                f"- selection_stage: `{selection_stage}`",
                f"- selection_unit_kind: `{selection_unit_kind}`",
                f"- selection_units: `{', '.join(selection_labels)}`",
                f"- validation_ratio: `{validation_ratio:.3f}`",
                f"- train_ratio: `{train_ratio:.3f}`",
                f"- inner_seed: `{'NA' if inner_seed is None else inner_seed}`",
                f"- model_seed: `{model_seed}`",
                f"- split_cache: `{split_cache_requested}`",
                f"- training_index_cache: `{use_training_index_cache}`",
                f"- cluster_artifact_cache: `{use_cluster_artifact_cache}`",
                f"- warmup_policy: `{measurement['warmup_policy']}`",
                f"- measured_sample_count: `{measurement['measured_sample_count']}`",
            ]
            if resource_gate_payload is not None:
                markdown_lines.extend(
                    [
                        "- resource_gate_enabled: `true`",
                        f"- max_peak_memory_mb: `{max_peak_memory_mb}`",
                    ]
                )
            markdown_lines.extend(
                [
                    "",
                    "## Candidates",
                    "",
                    "| Rank | Candidate | Validation RMSE Mean | Validation RMSE Std | Train Time Mean (s) |",
                    "| --- | --- | ---: | ---: | ---: |",
                ]
            )
            for rank, candidate_summary in enumerate(ordered_candidates, start=1):
                aggregate = candidate_summary["aggregate"]
                markdown_lines.append(
                    f"| {rank} | `{candidate_summary['candidate_id']}` | "
                    f"{aggregate['validation_rmse']['mean']:.6f} | "
                    f"{aggregate['validation_rmse']['std']:.6f} | "
                    f"{aggregate['training_wall_clock_seconds']['mean']:.2f} |"
                )
            best_validation = best_candidate["aggregate"]["validation_rmse"]
            best_training_time = best_candidate["aggregate"]["training_wall_clock_seconds"]
            markdown_lines.extend(
                [
                    "",
                    "## Winner",
                    "",
                    f"- candidate_id: `{best_candidate['candidate_id']}`",
                    f"- validation_rmse_mean: `{best_validation['mean']:.6f}`",
                    f"- validation_rmse_std: `{best_validation['std']:.6f}`",
                    f"- validation_rmse_count: `{best_validation['count']}`",
                    (f"- training_wall_clock_seconds_cv: `{best_training_time['coefficient_of_variation']:.6f}`"),
                    f"- candidate_config: `{best_candidate['candidate_config']}`",
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
                    f"candidate_count={len(ordered_candidates)}",
                    f"best_candidate={best_candidate['candidate_id']}",
                    f"best_validation_rmse_mean={best_candidate['aggregate']['validation_rmse']['mean']:.6f}",
                    f"[{finished_at}] status=completed",
                ],
            )
            completed_manifest = {
                **benchmark_manifest,
                "status": "completed",
                "generated_at_utc": finished_at,
                "inputs": {
                    "run_ids": [load_json_file(path)["run_id"] for path in run_manifest_paths],
                    "run_manifest_paths": [repo_path_string(path, repo_root=root) for path in run_manifest_paths],
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
                "best_candidate": best_candidate["candidate_id"],
                "best_validation_rmse_mean": best_candidate["aggregate"]["validation_rmse"]["mean"],
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
                "inputs": {
                    "run_ids": [load_json_file(path)["run_id"] for path in run_manifest_paths if path.exists()],
                    "run_manifest_paths": [
                        repo_path_string(path, repo_root=root) for path in run_manifest_paths if path.exists()
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


def run_ml100k_inner_tuning(
    *,
    tuning_config_path: Path,
    processed_manifest_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    use_split_cache: bool | None = None,
    use_training_index_cache: bool = False,
    use_cluster_artifact_cache: bool = False,
    repo_root: Path | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    return run_inner_tuning(
        tuning_config_path=tuning_config_path,
        processed_manifest_path=processed_manifest_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        use_split_cache=use_split_cache,
        use_training_index_cache=use_training_index_cache,
        use_cluster_artifact_cache=use_cluster_artifact_cache,
        repo_root=repo_root,
        command=command,
    )
