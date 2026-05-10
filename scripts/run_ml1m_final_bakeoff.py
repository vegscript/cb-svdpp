from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_ml1m_selected_candidate_bakeoff import run_bakeoff  # noqa: E402

CLAIM_BOUNDARY = "local ML1M final bake-off only; no performance or quality general claim."
FINAL_RESULT_FIELDS = [
    "label",
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
    parser = argparse.ArgumentParser(description="Run a strict two-config ML1M final bake-off.")
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final_bakeoff/ml1m_final_bakeoff_v1"))
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


def run_final_bakeoff(
    *,
    baseline_config_path: Path,
    selected_config_path: Path,
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
    payload = run_bakeoff(
        model_config_paths=[baseline_config_path, selected_config_path],
        labels=["baseline_stage0_transfer", "fidelity_promotion_selected"],
        output_dir=output_dir,
        processed_manifest_path=processed_manifest_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        cache_root=cache_root,
        repo_root=repo_root,
        model_name=model_name,
        split_family=split_family,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        split_seed=split_seed,
        model_seed=model_seed,
        evaluate_test=evaluate_test,
        split_cache=split_cache,
        use_training_index_cache=use_training_index_cache,
        use_cluster_artifact_cache=use_cluster_artifact_cache,
        overwrite=overwrite,
    )
    payload["claim_boundary"] = CLAIM_BOUNDARY
    payload["runtime_interpretation"] = (
        "Selection uses validation RMSE/MAE only. fit_model_seconds is the primary runtime readout; "
        "total_wall_seconds must not be interpreted as a speedup claim when cache status differs."
    )
    if (output_dir / "bakeoff_summary.csv").exists():
        write_final_bakeoff_artifacts(output_dir=output_dir)
    return payload


def write_final_bakeoff_artifacts(*, output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "bakeoff_summary.csv"
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    result_rows = build_final_bakeoff_result_rows(rows)
    with (output_dir / "final_bakeoff_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINAL_RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(result_rows)
    decision = build_final_bakeoff_decision(rows)
    (output_dir / "final_bakeoff_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return decision


def build_final_bakeoff_result_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows_by_label = {row["label"]: row for row in rows}
    decision = build_final_bakeoff_decision(rows)
    winner = decision["winner"]
    result_rows = []
    for label in ("baseline_stage0_transfer", "fidelity_promotion_selected"):
        row = rows_by_label[label]
        result_rows.append(
            {
                "label": row["label"],
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
                "decision_role": "winner" if row["label"] == winner else "baseline",
            }
        )
    return result_rows


def build_final_bakeoff_decision(rows: list[dict[str, str]]) -> dict[str, Any]:
    rows_by_label = {row["label"]: row for row in rows}
    baseline = rows_by_label["baseline_stage0_transfer"]
    selected = rows_by_label["fidelity_promotion_selected"]
    required_fields = ("validation_rmse", "validation_mae", "fit_model_seconds")
    missing = [
        field
        for field in required_fields
        if baseline.get(field, "") == "" or selected.get(field, "") == ""
    ]
    if missing:
        return _decision_payload(
            decision="INCONCLUSIVE_RERUN_ONCE",
            baseline=baseline,
            selected=selected,
            winner="",
            reason=f"Missing required metric fields: {missing}.",
        )

    baseline_rmse = _required_float(baseline, "validation_rmse")
    selected_rmse = _required_float(selected, "validation_rmse")
    baseline_mae = _required_float(baseline, "validation_mae")
    selected_mae = _required_float(selected, "validation_mae")
    if selected_rmse < baseline_rmse and selected_mae <= baseline_mae:
        return _decision_payload(
            decision="ADOPT_PROMOTED_CONFIG",
            baseline=baseline,
            selected=selected,
            winner="fidelity_promotion_selected",
            reason=(
                "Selected config has lower validation RMSE and non-worse validation MAE than baseline in this "
                "local ML1M final bakeoff; no test metric was used."
            ),
        )
    if selected_rmse >= baseline_rmse:
        return _decision_payload(
            decision="REJECT_PROMOTED_CONFIG",
            baseline=baseline,
            selected=selected,
            winner="baseline_stage0_transfer",
            reason=(
                "Selected config validation RMSE is higher than or equal to baseline in this local ML1M final bakeoff."
            ),
        )
    return _decision_payload(
        decision="INCONCLUSIVE_RERUN_ONCE",
        baseline=baseline,
        selected=selected,
        winner="",
        reason="Validation RMSE and MAE signals are inconsistent for the final bakeoff decision.",
    )


def _decision_payload(
    *,
    decision: str,
    baseline: dict[str, str],
    selected: dict[str, str],
    winner: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "decision": decision,
        "baseline_config": baseline["model_config_path"],
        "selected_config": selected["model_config_path"],
        "winner": winner,
        "validation_rmse_delta_vs_baseline": _optional_delta(selected, baseline, "validation_rmse"),
        "validation_mae_delta_vs_baseline": _optional_delta(selected, baseline, "validation_mae"),
        "fit_model_seconds_delta_vs_baseline": _optional_delta(selected, baseline, "fit_model_seconds"),
        "reason": reason,
        "claim_boundary": "local ML1M final bakeoff only",
    }


def _optional_delta(selected: dict[str, str], baseline: dict[str, str], key: str) -> float | None:
    if selected.get(key, "") == "" or baseline.get(key, "") == "":
        return None
    return float(selected[key]) - float(baseline[key])


def _required_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"missing required numeric field {key!r}")
    return float(value)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_final_bakeoff(
        baseline_config_path=args.baseline_config,
        selected_config_path=args.selected_config,
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


if __name__ == "__main__":
    raise SystemExit(main())
