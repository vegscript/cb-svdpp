from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from recsys_lab.config.loader import dump_yaml_file, load_yaml_file  # noqa: E402
from recsys_lab.data.processed import load_processed_dataset_manifest  # noqa: E402
from recsys_lab.experiments.common import SplitConfig  # noqa: E402
from recsys_lab.experiments.unified_runner import run_unified_experiment  # noqa: E402
from recsys_lab.models.registry import validate_model_config_payload  # noqa: E402

MAX_CONFIGS = 3
CLAIM_BOUNDARY = "ML1M selected-candidate bake-off only; no performance or quality general claim."
SUMMARY_FIELDS = [
    "label",
    "model_config_path",
    "execution_status",
    "run_id",
    "run_dir",
    "validation_rmse",
    "validation_mae",
    "test_rmse",
    "test_mae",
    "fit_model_seconds",
    "total_wall_seconds",
    "cluster_total_seconds",
    "cluster_cache_status",
    "user_cluster_history_cache_status",
    "notes",
]
RESULT_FIELDS = [
    "config_label",
    "config_path",
    "validation_rmse",
    "validation_mae",
    "fit_model_seconds",
    "total_wall_seconds",
    "cluster_total_seconds",
    "cluster_cache_status",
    "user_cluster_history_cache_status",
    "run_dir",
    "run_manifest_path",
    "decision_role",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a strict ML1M selected-candidate bake-off.")
    parser.add_argument("--model-config", type=Path, action="append", default=[])
    parser.add_argument("--baseline-config", type=Path, default=None)
    parser.add_argument("--candidate-config", type=Path, default=None)
    parser.add_argument("--label", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/ml1m_selected_candidate_bakeoff_v1"))
    parser.add_argument(
        "--processed-manifest",
        type=Path,
        default=Path("data/processed/ml1m/ml1m_benchmark_random_v1_explicit_v1_float32_manifest.json"),
    )
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime/base.yaml"))
    parser.add_argument("--device-config", type=Path, default=Path("configs/runtime/devices/local_u300_24gb.yaml"))
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--model", default="cb_svdpp")
    parser.add_argument("--split-family", default="benchmark_random_v1")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--model-seed", type=int, default=1)
    parser.add_argument("--evaluate-test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--split-cache", choices=["auto", "enable", "disable"], default="auto")
    parser.add_argument("--training-index-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cluster-artifact-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run_bakeoff(
    *,
    model_config_paths: list[Path],
    labels: list[str] | None,
    output_dir: Path,
    processed_manifest_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    cache_root: Path | None,
    repo_root: Path,
    model_name: str,
    split_family: str,
    train_ratio: float,
    validation_ratio: float,
    split_seed: int,
    model_seed: int,
    evaluate_test: bool,
    split_cache: str,
    use_training_index_cache: bool,
    use_cluster_artifact_cache: bool,
    overwrite: bool,
) -> dict[str, Any]:
    root = repo_root.resolve()
    _validate_config_list(model_config_paths=model_config_paths, labels=labels)
    output_dir = _resolve_path(output_dir, repo_root=root)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace report outputs")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_manifest_path = _resolve_path(processed_manifest_path, repo_root=root)
    _validate_ml1m_manifest(processed_manifest_path, expected_split_family=split_family)
    runtime_config_path = _resolve_path(runtime_config_path, repo_root=root)
    device_config_path = _resolve_path(device_config_path, repo_root=root)
    effective_cache_root = _resolve_path(cache_root, repo_root=root) if cache_root is not None else output_dir / "cache"
    effective_runtime_config = _runtime_config_with_cache_root(
        runtime_config_path,
        cache_root=effective_cache_root,
        output_dir=output_dir,
    )

    rows: list[dict[str, str]] = []
    result_payloads: list[dict[str, Any]] = []
    resolved_labels = labels or _default_labels(model_config_paths)
    for label, model_config_path in zip(resolved_labels, model_config_paths, strict=True):
        resolved_config_path = _resolve_path(model_config_path, repo_root=root)
        _validate_model_config(resolved_config_path, expected_model_name=model_name)
        command = _command_string(
            label=label,
            model_config_path=model_config_path,
            processed_manifest_path=processed_manifest_path,
            runtime_config_path=effective_runtime_config,
            device_config_path=device_config_path,
        )
        result = run_unified_experiment(
            processed_manifest_path=processed_manifest_path,
            model_config_path=resolved_config_path,
            runtime_config_path=effective_runtime_config,
            device_config_path=device_config_path,
            split_config=SplitConfig(train_ratio=train_ratio, validation_ratio=validation_ratio, seed=split_seed),
            model_seed=model_seed,
            repo_root=root,
            command=command,
            model_name=model_name,
            split_family=split_family,
            evaluate_test=evaluate_test,
            use_split_cache=_split_cache_value(split_cache),
            use_training_index_cache=use_training_index_cache,
            use_cluster_artifact_cache=use_cluster_artifact_cache,
        )
        row = _summary_row(label=label, model_config_path=resolved_config_path, result=result)
        rows.append(row)
        result_payloads.append(result)
        _write_reports(
            output_dir=output_dir,
            rows=rows,
            result_payloads=result_payloads,
            processed_manifest_path=processed_manifest_path,
            runtime_config_path=effective_runtime_config,
            device_config_path=device_config_path,
            cache_root=effective_cache_root,
        )

    if len(rows) >= 2:
        write_bakeoff_decision_artifacts(
            output_dir=output_dir,
            rows=rows,
            baseline_label=resolved_labels[0],
            candidate_label=resolved_labels[1],
        )

    return {
        "output_dir": str(output_dir),
        "summary_csv": str(output_dir / "bakeoff_summary.csv"),
        "summary_json": str(output_dir / "bakeoff_summary.json"),
        "results_csv": str(output_dir / "bakeoff_results.csv"),
        "decision_json": str(output_dir / "bakeoff_decision.json"),
        "run_count": len(rows),
        "labels": resolved_labels,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_config_paths, labels = _resolve_cli_config_list(
        model_config_paths=args.model_config,
        labels=args.label,
        baseline_config=args.baseline_config,
        candidate_config=args.candidate_config,
    )
    payload = run_bakeoff(
        model_config_paths=model_config_paths,
        labels=labels,
        output_dir=args.output_dir,
        processed_manifest_path=args.processed_manifest,
        runtime_config_path=args.runtime_config,
        device_config_path=args.device_config,
        cache_root=args.cache_root,
        repo_root=args.repo_root,
        model_name=args.model,
        split_family=args.split_family,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        evaluate_test=args.evaluate_test,
        split_cache=args.split_cache,
        use_training_index_cache=args.training_index_cache,
        use_cluster_artifact_cache=args.cluster_artifact_cache,
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _resolve_cli_config_list(
    *,
    model_config_paths: list[Path],
    labels: list[str] | None,
    baseline_config: Path | None,
    candidate_config: Path | None,
) -> tuple[list[Path], list[str] | None]:
    if baseline_config is None and candidate_config is None:
        return model_config_paths, labels
    if baseline_config is None or candidate_config is None:
        raise ValueError("--baseline-config and --candidate-config must be provided together")

    resolved_paths = [baseline_config, candidate_config, *model_config_paths]
    if labels is None:
        return resolved_paths, ["baseline_stage0_transfer", "small_study_v2_candidate"] + [
            path.stem for path in model_config_paths
        ]
    return resolved_paths, labels


def _validate_config_list(*, model_config_paths: list[Path], labels: list[str] | None) -> None:
    if len(model_config_paths) < 2:
        raise ValueError("bake-off requires at least two explicit model configs")
    if len(model_config_paths) > MAX_CONFIGS:
        raise ValueError(f"bake-off supports at most {MAX_CONFIGS} explicit model configs")
    if labels is not None and len(labels) != len(model_config_paths):
        raise ValueError("--label must be provided once for each --model-config")


def _validate_ml1m_manifest(processed_manifest_path: Path, *, expected_split_family: str) -> None:
    manifest = load_processed_dataset_manifest(processed_manifest_path)
    if manifest.get("dataset_short_name") != "ml1m":
        raise ValueError("selected-candidate bake-off requires an ML1M processed manifest")
    if manifest.get("split_family") != expected_split_family:
        raise ValueError(
            "selected-candidate bake-off requires processed manifest split_family "
            f"{expected_split_family!r}, got {manifest.get('split_family')!r}"
        )


def _validate_model_config(model_config_path: Path, *, expected_model_name: str) -> None:
    payload = load_yaml_file(model_config_path)
    _, profile = validate_model_config_payload(payload, expected_model_name=expected_model_name)
    if str(profile.training.dtype) != "float32":  # type: ignore[attr-defined]
        raise ValueError("selected-candidate bake-off requires float32 model configs")
    clustering = getattr(profile, "clustering", None)
    if clustering is not None and getattr(clustering, "induction", None) is None:
        raise ValueError("selected-candidate bake-off requires explicit clustering.induction config")


def _summary_row(*, label: str, model_config_path: Path, result: dict[str, Any]) -> dict[str, str]:
    run_dir = Path(str(result["run_dir"]))
    metrics_path = run_dir / "metrics.json"
    performance_path = run_dir / "performance_profile.json"
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    performance_payload = json.loads(performance_path.read_text(encoding="utf-8"))
    metrics = metrics_payload.get("metrics", {})
    caches = metrics_payload.get("caches", {})
    return {
        "label": label,
        "model_config_path": str(model_config_path),
        "execution_status": "succeeded",
        "run_id": str(result["run_id"]),
        "run_dir": str(run_dir),
        "validation_rmse": _metric(metrics, "validation", "rmse"),
        "validation_mae": _metric(metrics, "validation", "mae"),
        "test_rmse": _metric(metrics, "test", "rmse"),
        "test_mae": _metric(metrics, "test", "mae"),
        "fit_model_seconds": _stage_seconds(performance_payload, "fit_model"),
        "total_wall_seconds": _total_wall_seconds(performance_payload),
        "cluster_total_seconds": _stage_seconds(performance_payload, "build_cluster_artifacts"),
        "cluster_cache_status": _cache_status(caches, "cluster_artifacts"),
        "user_cluster_history_cache_status": _cache_status(caches, "user_cluster_history"),
        "notes": "",
    }


def _write_reports(
    *,
    output_dir: Path,
    rows: list[dict[str, str]],
    result_payloads: list[dict[str, Any]],
    processed_manifest_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    cache_root: Path,
) -> None:
    csv_path = output_dir / "bakeoff_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_payload = {
        "summary_version": "ml1m_selected_candidate_bakeoff_v1",
        "processed_manifest": str(processed_manifest_path),
        "runtime_config": str(runtime_config_path),
        "device_config": str(device_config_path),
        "cache_root": str(cache_root),
        "run_count": len(rows),
        "rows": rows,
        "run_payloads": result_payloads,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / "bakeoff_summary.json").write_text(
        json.dumps(json_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_bakeoff_decision_artifacts(
    *,
    output_dir: Path,
    rows: list[dict[str, str]],
    baseline_label: str,
    candidate_label: str,
    selected_config_source: Path | None = None,
    final_selected_config_path: Path | None = None,
) -> dict[str, Any]:
    result_rows = build_bakeoff_result_rows(
        rows=rows,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
    )
    with (output_dir / "bakeoff_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(result_rows)

    decision_payload = build_bakeoff_decision(
        rows=rows,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
    )
    adopted_config_path = None
    if decision_payload["decision"] == "ADOPT_SELECTED_CONFIG" and selected_config_source is not None:
        if final_selected_config_path is None:
            raise ValueError("final_selected_config_path is required when writing an adopted selected config")
        final_selected_config_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(selected_config_source, final_selected_config_path)
        adopted_config_path = str(final_selected_config_path)
    decision_payload["adopted_config_path"] = adopted_config_path
    (output_dir / "bakeoff_decision.json").write_text(
        json.dumps(decision_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return decision_payload


def build_bakeoff_result_rows(
    *,
    rows: list[dict[str, str]],
    baseline_label: str,
    candidate_label: str,
) -> list[dict[str, str]]:
    result_rows = []
    for row in rows:
        if row["label"] == baseline_label:
            role = "baseline"
        elif row["label"] == candidate_label:
            role = "candidate"
        else:
            role = "context"
        result_rows.append(
            {
                "config_label": row["label"],
                "config_path": row["model_config_path"],
                "validation_rmse": row["validation_rmse"],
                "validation_mae": row["validation_mae"],
                "fit_model_seconds": row["fit_model_seconds"],
                "total_wall_seconds": row["total_wall_seconds"],
                "cluster_total_seconds": row["cluster_total_seconds"],
                "cluster_cache_status": row["cluster_cache_status"],
                "user_cluster_history_cache_status": row["user_cluster_history_cache_status"],
                "run_dir": row["run_dir"],
                "run_manifest_path": str(Path(row["run_dir"]) / "run_manifest.json") if row["run_dir"] else "",
                "decision_role": role,
            }
        )
    return result_rows


def build_bakeoff_decision(
    *,
    rows: list[dict[str, str]],
    baseline_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    rows_by_label = {row["label"]: row for row in rows}
    baseline = rows_by_label[baseline_label]
    candidate = rows_by_label[candidate_label]
    baseline_rmse = _required_float(baseline, "validation_rmse")
    candidate_rmse = _required_float(candidate, "validation_rmse")
    rmse_delta = candidate_rmse - baseline_rmse
    mae_delta = _required_float(candidate, "validation_mae") - _required_float(baseline, "validation_mae")
    fit_delta = _required_float(candidate, "fit_model_seconds") - _required_float(baseline, "fit_model_seconds")
    decision = "ADOPT_SELECTED_CONFIG" if candidate_rmse < baseline_rmse else "REJECT_SELECTED_CONFIG"
    winner = candidate_label if decision == "ADOPT_SELECTED_CONFIG" else baseline_label
    reason = (
        "Candidate validation RMSE is lower than baseline on the local ML1M bake-off."
        if decision == "ADOPT_SELECTED_CONFIG"
        else "Candidate validation RMSE is higher than or equal to baseline on the local ML1M bake-off; "
        "selection uses validation metrics only."
    )
    return {
        "decision": decision,
        "baseline_config": baseline["model_config_path"],
        "candidate_config": candidate["model_config_path"],
        "winner": winner,
        "validation_rmse_delta_vs_baseline": rmse_delta,
        "validation_mae_delta_vs_baseline": mae_delta,
        "fit_model_seconds_delta_vs_baseline": fit_delta,
        "reason": reason,
        "claim_boundary": "local ML1M bake-off only",
    }


def _runtime_config_with_cache_root(runtime_config_path: Path, *, cache_root: Path, output_dir: Path) -> Path:
    payload = load_yaml_file(runtime_config_path)
    runtime = dict(payload.get("runtime", {}))
    runtime["cache_root"] = str(cache_root)
    payload["runtime"] = runtime
    target_path = output_dir / "runtime_config.bakeoff_cache.yaml"
    dump_yaml_file(target_path, payload)
    return target_path.resolve()


def _default_labels(model_config_paths: list[Path]) -> list[str]:
    return [path.stem for path in model_config_paths]


def _command_string(
    *,
    label: str,
    model_config_path: Path,
    processed_manifest_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
) -> str:
    return (
        "scripts/run_ml1m_selected_candidate_bakeoff.py "
        f"--label {label} "
        f"--model-config {model_config_path} "
        f"--processed-manifest {processed_manifest_path} "
        f"--runtime-config {runtime_config_path} "
        f"--device-config {device_config_path}"
    )


def _resolve_path(path: Path | None, *, repo_root: Path) -> Path:
    if path is None:
        raise ValueError("path is required")
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _split_cache_value(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "enable"


def _metric(metrics: dict[str, Any], split: str, name: str) -> str:
    split_metrics = metrics.get(split)
    if not isinstance(split_metrics, dict):
        return ""
    value = split_metrics.get(name)
    return "" if value is None else str(value)


def _stage_seconds(performance_payload: dict[str, Any], stage_name: str) -> str:
    for stage in performance_payload.get("stages", []):
        if stage.get("name") == stage_name:
            return str(stage.get("wall_clock_seconds", ""))
    return ""


def _total_wall_seconds(performance_payload: dict[str, Any]) -> str:
    value = performance_payload.get("total_profiled_wall_clock_seconds")
    return "" if value is None else str(value)


def _cache_status(caches: dict[str, Any], name: str) -> str:
    return str(caches.get(name, {}).get("status", ""))


def _required_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"missing required numeric field {key!r}")
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
