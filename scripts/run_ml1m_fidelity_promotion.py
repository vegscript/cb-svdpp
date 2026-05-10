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
from recsys_lab.experiments.common import SplitConfig  # noqa: E402
from recsys_lab.experiments.unified_runner import run_unified_experiment  # noqa: E402
from scripts.run_ml1m_selected_candidate_bakeoff import (  # noqa: E402
    _resolve_path,
    _split_cache_value,
    _summary_row,
    _validate_ml1m_manifest,
    _validate_model_config,
)

MAX_PROMOTED_CONFIGS = 3
CLAIM_BOUNDARY = "local ML1M fidelity promotion only; no performance or quality general claim."
SUMMARY_FIELDS = [
    "label",
    "role",
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
    "label",
    "config_path",
    "source_candidate_id",
    "alpha",
    "learning_rate",
    "lambda_q",
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
    parser = argparse.ArgumentParser(description="Run a strict ML1M fidelity promotion for promoted CB-SVD++ configs.")
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument(
        "--promoted-config",
        "--candidate-config",
        dest="promoted_config",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--label", action="append", default=None)
    parser.add_argument("--baseline-label", default="baseline_stage0_transfer")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/ml1m_fidelity_promotion_v1"))
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


def run_promotion(
    *,
    baseline_config_path: Path,
    promoted_config_paths: list[Path],
    labels: list[str] | None,
    baseline_label: str,
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
    _validate_promotion_config_list(promoted_config_paths=promoted_config_paths, labels=labels)
    output_dir = _resolve_path(output_dir, repo_root=root)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace promotion outputs")
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

    resolved_promoted_labels = labels or [f"promotion_p{index}" for index in range(1, len(promoted_config_paths) + 1)]
    config_entries = [(baseline_label, baseline_config_path, "baseline")] + [
        (label, path, "promoted") for label, path in zip(resolved_promoted_labels, promoted_config_paths, strict=True)
    ]

    rows: list[dict[str, str]] = []
    result_payloads: list[dict[str, Any]] = []
    for label, model_config_path, role in config_entries:
        resolved_config_path = _resolve_path(model_config_path, repo_root=root)
        _validate_model_config(resolved_config_path, expected_model_name=model_name)
        command = _command_string(
            label=label,
            role=role,
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
        row = _promotion_summary_row(label=label, role=role, model_config_path=resolved_config_path, result=result)
        rows.append(row)
        result_payloads.append(result)
        _write_summary_reports(
            output_dir=output_dir,
            rows=rows,
            result_payloads=result_payloads,
            processed_manifest_path=processed_manifest_path,
            runtime_config_path=effective_runtime_config,
            device_config_path=device_config_path,
            cache_root=effective_cache_root,
        )

    result_rows = build_promotion_result_rows(rows)
    decision_payload = build_promotion_decision(rows)
    _write_promotion_decision_artifacts(
        output_dir=output_dir,
        result_rows=result_rows,
        decision_payload=decision_payload,
    )
    return {
        "output_dir": str(output_dir),
        "summary_csv": str(output_dir / "promotion_summary.csv"),
        "summary_json": str(output_dir / "promotion_summary.json"),
        "results_csv": str(output_dir / "promotion_results.csv"),
        "decision_json": str(output_dir / "promotion_decision.json"),
        "run_count": len(rows),
        "winner": decision_payload["winner"],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_promotion(
        baseline_config_path=args.baseline_config,
        promoted_config_paths=args.promoted_config,
        labels=args.label,
        baseline_label=args.baseline_label,
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


def _validate_promotion_config_list(*, promoted_config_paths: list[Path], labels: list[str] | None) -> None:
    if len(promoted_config_paths) < 1:
        raise ValueError("fidelity promotion requires at least one promoted config")
    if len(promoted_config_paths) > MAX_PROMOTED_CONFIGS:
        raise ValueError(f"fidelity promotion supports at most {MAX_PROMOTED_CONFIGS} promoted configs")
    if labels is not None and len(labels) != len(promoted_config_paths):
        raise ValueError("--label must be provided once for each --promoted-config")


def _promotion_summary_row(
    *,
    label: str,
    role: str,
    model_config_path: Path,
    result: dict[str, Any],
) -> dict[str, str]:
    row = _summary_row(label=label, model_config_path=model_config_path, result=result)
    row["role"] = role
    return {field: row.get(field, "") for field in SUMMARY_FIELDS}


def build_promotion_result_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    sorted_rows = sorted(
        rows,
        key=lambda row: (_required_float(row, "validation_rmse"), _required_float(row, "validation_mae")),
    )
    winner_label = sorted_rows[0]["label"]
    result_rows = []
    for row in sorted_rows:
        config_payload = load_yaml_file(Path(row["model_config_path"]))
        result_rows.append(
            {
                "label": row["label"],
                "config_path": row["model_config_path"],
                "source_candidate_id": _source_candidate_id(config_payload),
                "alpha": _config_value(config_payload, ("clustering", "alpha")),
                "learning_rate": _config_value(config_payload, ("training", "learning_rate")),
                "lambda_q": _config_value(config_payload, ("training", "lambda_q")),
                "validation_rmse": row["validation_rmse"],
                "validation_mae": row["validation_mae"],
                "fit_model_seconds": row["fit_model_seconds"],
                "total_wall_seconds": row["total_wall_seconds"],
                "cluster_total_seconds": row["cluster_total_seconds"],
                "cluster_cache_status": row["cluster_cache_status"],
                "user_cluster_history_cache_status": row["user_cluster_history_cache_status"],
                "run_dir": row["run_dir"],
                "run_manifest_path": str(Path(row["run_dir"]) / "run_manifest.json") if row["run_dir"] else "",
                "decision_role": "winner" if row["label"] == winner_label else row["role"],
            }
        )
    return result_rows


def build_promotion_decision(rows: list[dict[str, str]]) -> dict[str, Any]:
    sorted_rows = sorted(
        rows,
        key=lambda row: (_required_float(row, "validation_rmse"), _required_float(row, "validation_mae")),
    )
    winner = sorted_rows[0]
    baseline = next(row for row in rows if row["role"] == "baseline")
    promoted_rows = [row for row in rows if row["role"] == "promoted"]
    best_promoted = min(
        promoted_rows,
        key=lambda row: (_required_float(row, "validation_rmse"), _required_float(row, "validation_mae")),
    )
    baseline_rmse = _required_float(baseline, "validation_rmse")
    baseline_mae = _required_float(baseline, "validation_mae")
    best_promoted_rmse = _required_float(best_promoted, "validation_rmse")
    best_promoted_mae = _required_float(best_promoted, "validation_mae")
    if best_promoted_rmse < baseline_rmse:
        decision = "PROMOTED_CANDIDATE_READY_FOR_FINAL_BAKEOFF"
        reason = (
            "Best promoted candidate has lower validation RMSE than the baseline in this local ML1M "
            "fidelity-promotion run; final confirmatory bake-off is the next decision step."
        )
    elif best_promoted_rmse == baseline_rmse:
        decision = "REFINE_SEARCH_AROUND_BASELINE"
        reason = (
            "Best promoted candidate matches baseline validation RMSE in this local ML1M fidelity-promotion run; "
            "the observed parameter region is close enough to justify a focused refinement if needed."
        )
    else:
        decision = "REJECT_PROMOTED_CANDIDATES"
        reason = (
            "All promoted candidates have higher validation RMSE than the baseline in this local ML1M "
            "fidelity-promotion run."
    )
    return {
        "decision": decision,
        "baseline_config": baseline["model_config_path"],
        "winner": winner["label"],
        "winner_label": winner["label"],
        "winner_config": winner["model_config_path"],
        "winner_role": winner["role"],
        "baseline_label": baseline["label"],
        "best_promoted_label": best_promoted["label"],
        "baseline_validation_rmse": baseline_rmse,
        "best_promoted_validation_rmse": best_promoted_rmse,
        "validation_rmse_delta_vs_baseline": best_promoted_rmse - baseline_rmse,
        "validation_mae_delta_vs_baseline": best_promoted_mae - baseline_mae,
        "best_promoted_rmse_delta_vs_baseline": best_promoted_rmse - baseline_rmse,
        "reason": reason,
        "selection_basis": "validation_rmse_then_validation_mae",
        "claim_boundary": "local ML1M fidelity promotion only",
    }


def _write_summary_reports(
    *,
    output_dir: Path,
    rows: list[dict[str, str]],
    result_payloads: list[dict[str, Any]],
    processed_manifest_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
    cache_root: Path,
) -> None:
    with (output_dir / "promotion_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "summary_version": "ml1m_fidelity_promotion_v1",
        "processed_manifest": str(processed_manifest_path),
        "runtime_config": str(runtime_config_path),
        "device_config": str(device_config_path),
        "cache_root": str(cache_root),
        "run_count": len(rows),
        "rows": rows,
        "run_payloads": result_payloads,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / "promotion_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_promotion_decision_artifacts(
    *,
    output_dir: Path,
    result_rows: list[dict[str, str]],
    decision_payload: dict[str, Any],
) -> None:
    with (output_dir / "promotion_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(result_rows)
    (output_dir / "promotion_decision.json").write_text(
        json.dumps(decision_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _runtime_config_with_cache_root(runtime_config_path: Path, *, cache_root: Path, output_dir: Path) -> Path:
    payload = load_yaml_file(runtime_config_path)
    runtime = dict(payload.get("runtime", {}))
    runtime["cache_root"] = str(cache_root)
    payload["runtime"] = runtime
    target_path = output_dir / "runtime_config.fidelity_promotion_cache.yaml"
    dump_yaml_file(target_path, payload)
    return target_path.resolve()


def _command_string(
    *,
    label: str,
    role: str,
    model_config_path: Path,
    processed_manifest_path: Path,
    runtime_config_path: Path,
    device_config_path: Path,
) -> str:
    config_option = "--baseline-config" if role == "baseline" else "--promoted-config"
    return (
        "scripts/run_ml1m_fidelity_promotion.py "
        f"--label {label} "
        f"{config_option} {model_config_path} "
        f"--processed-manifest {processed_manifest_path} "
        f"--runtime-config {runtime_config_path} "
        f"--device-config {device_config_path}"
    )


def _required_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"missing required numeric field {key!r}")
    return float(value)


def _source_candidate_id(config_payload: dict[str, Any]) -> str:
    metadata = config_payload.get("metadata", {})
    provenance = metadata.get("provenance", {}) if isinstance(metadata, dict) else {}
    value = provenance.get("source_candidate_id") if isinstance(provenance, dict) else None
    return "" if value is None else str(value)


def _config_value(config_payload: dict[str, Any], path: tuple[str, ...]) -> str:
    current: Any = config_payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return ""
        current = current[key]
    return "" if current is None else str(current)


__all__ = [
    "MAX_PROMOTED_CONFIGS",
    "build_promotion_decision",
    "build_promotion_result_rows",
    "main",
    "parse_args",
    "run_promotion",
]


if __name__ == "__main__":
    raise SystemExit(main())
