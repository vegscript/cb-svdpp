from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from recsys_lab.config.loader import dump_yaml_file, load_yaml_file  # noqa: E402
from recsys_lab.experiments.common import SplitConfig  # noqa: E402
from recsys_lab.tuning import SearchSpaceSpec, build_study_plan  # noqa: E402
from recsys_lab.tuning.execution import CandidateExecutionResult, execute_candidate  # noqa: E402
from recsys_lab.tuning.manifests import CandidateManifest  # noqa: E402
from recsys_lab.tuning.writers import (  # noqa: E402
    update_candidate_manifest_with_execution_result,
    write_study_execution_artifacts,
    write_tuning_json,
)
from scripts.plan_sota_tuning_study import plan_sota_tuning_study  # noqa: E402

CLAIM_BOUNDARY = "Local ML1M staged SOTA tuning pilot only; no broad performance or quality claim."
INCUMBENT_CONFIG = "configs/models/selected/ml1m/ml1m_cb_svdpp_fidelity_promotion_selected.yaml"
INCUMBENT_VALIDATION_RMSE = 0.853703596758597
INCUMBENT_VALIDATION_MAE = 0.6695381079447467
STAGE1_MAX_CANDIDATES = 16
STAGE2_MAX_CANDIDATES = 4
STAGE3_MAX_CANDIDATES = 1
TOTAL_EXECUTION_HARD_CAP = 21
ExecuteCandidateFn = Callable[..., CandidateExecutionResult]

STAGE_RESULT_FIELDS = [
    "study_id",
    "stage_name",
    "candidate_id",
    "execution_status",
    "validation_rmse",
    "validation_mae",
    "fit_model_seconds",
    "total_wall_seconds",
    "cluster_total_seconds",
    "cluster_cache_status",
    "user_cluster_history_cache_status",
    "candidate_config_path",
    "candidate_manifest_path",
    "cluster_reuse_group_id",
    "run_dir",
    "run_manifest_path",
    "metrics_path",
    "performance_profile_path",
    "kernel_profile_path",
    "error_message",
]
PILOT_SUMMARY_FIELDS = [
    "stage_name",
    "rank",
    "candidate_id",
    "source_candidate_id",
    "execution_status",
    "alpha",
    "learning_rate",
    "lambda_p",
    "lambda_q",
    "lambda_y",
    "lambda_pC",
    "lambda_qC",
    "lambda_yC",
    "training_epochs",
    "validation_rmse",
    "validation_mae",
    "fit_model_seconds",
    "total_wall_seconds",
    "cluster_total_seconds",
    "cluster_cache_status",
    "user_cluster_history_cache_status",
    "candidate_config_path",
    "run_dir",
]


def run_sota_tuning_pilot(
    *,
    search_space: Path,
    output_dir: Path,
    study_id: str | None,
    processed_manifest: Path,
    runtime_config: Path,
    device_config: Path,
    overwrite: bool,
    repo_root: Path = REPO_ROOT,
    cache_root: Path | None = None,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    split_seed: int = 1,
    model_seed: int = 1,
    execute_candidate_fn: ExecuteCandidateFn | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    search_space_path = _resolve_path(search_space, repo_root=repo_root)
    spec = SearchSpaceSpec.model_validate(load_yaml_file(search_space_path))
    _validate_pilot_scope(spec)
    if "ml1m" not in str(processed_manifest).lower():
        raise ValueError("20b pilot requires an ML1M processed manifest")

    planning_result = plan_sota_tuning_study(
        search_space_path=search_space_path,
        output_dir=output_dir,
        study_id=study_id,
        stage_name="stage1_low_fidelity",
        overwrite=overwrite,
        repo_root=repo_root,
    )
    study_dir = Path(str(planning_result["study_dir"]))
    effective_cache_root = _resolve_path(cache_root, repo_root=repo_root) if cache_root else study_dir / "local_cache"
    effective_runtime_config = _runtime_config_with_cache_root(
        runtime_config,
        cache_root=effective_cache_root,
        study_dir=study_dir,
        repo_root=repo_root,
    )

    all_results: list[CandidateExecutionResult] = []
    stage1_plan = _stage_plan(spec, study_id=study_id, stage_name="stage1_low_fidelity")
    stage1_results = _execute_stage(
        plan=stage1_plan,
        stage_dir=study_dir,
        stage_name="stage1_low_fidelity",
        candidate_manifest_paths=[
            study_dir / "candidates" / candidate.candidate_id / "candidate_manifest.json"
            for candidate in stage1_plan.candidates
        ],
        runner_kwargs=_runner_kwargs(
            processed_manifest=processed_manifest,
            runtime_config=effective_runtime_config,
            device_config=device_config,
            repo_root=repo_root,
            split_family=spec.study.split_family,
            model_name=spec.study.model,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            split_seed=split_seed,
            model_seed=model_seed,
            study_id=study_id or stage1_plan.study_id,
        ),
        repo_root=repo_root,
        execute_candidate_fn=execute_candidate_fn,
    )
    all_results.extend(stage1_results)
    _write_stage_results(study_dir / "reports" / "stage1_results.csv", stage1_results, stage_name="stage1_low_fidelity")
    if _has_failure(stage1_results):
        return _write_final_summary(
            study_dir,
            spec=spec,
            results=all_results,
            decision="SOTA_PILOT_EXECUTION_UNSTABLE",
        )

    try:
        plan_sota_tuning_study(
            search_space_path=search_space_path,
            output_dir=output_dir,
            study_id=study_id,
            promote_from_results=study_dir / "reports" / "stage1_results.csv",
            from_stage="stage1_low_fidelity",
            to_stage="stage2_mid_fidelity",
            overwrite=overwrite,
            repo_root=repo_root,
        )
    except ValueError:
        return _write_final_summary(
            study_dir,
            spec=spec,
            results=all_results,
            decision="SOTA_PILOT_PROMOTION_CONTRACT_BROKEN",
        )
    stage2_manifest_paths = _write_promoted_manifests(
        study_dir=study_dir,
        source_stage_dir=study_dir,
        source_stage_name="stage1_low_fidelity",
        target_stage_name="stage2_mid_fidelity",
        search_space=spec,
    )
    stage2_results = _execute_stage(
        plan=None,
        stage_dir=study_dir / "promotions" / "stage2_mid_fidelity",
        stage_name="stage2_mid_fidelity",
        candidate_manifest_paths=stage2_manifest_paths,
        runner_kwargs=_runner_kwargs(
            processed_manifest=processed_manifest,
            runtime_config=effective_runtime_config,
            device_config=device_config,
            repo_root=repo_root,
            split_family=spec.study.split_family,
            model_name=spec.study.model,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            split_seed=split_seed,
            model_seed=model_seed,
            study_id=study_id or stage1_plan.study_id,
        ),
        repo_root=repo_root,
        execute_candidate_fn=execute_candidate_fn,
    )
    all_results.extend(stage2_results)
    _write_stage_results(study_dir / "reports" / "stage2_results.csv", stage2_results, stage_name="stage2_mid_fidelity")
    if _has_failure(stage2_results):
        return _write_final_summary(
            study_dir,
            spec=spec,
            results=all_results,
            decision="SOTA_PILOT_EXECUTION_UNSTABLE",
        )

    try:
        plan_sota_tuning_study(
            search_space_path=search_space_path,
            output_dir=output_dir,
            study_id=study_id,
            promote_from_results=study_dir / "reports" / "stage2_results.csv",
            from_stage="stage2_mid_fidelity",
            to_stage="stage3_full_fidelity",
            overwrite=overwrite,
            repo_root=repo_root,
        )
    except ValueError:
        return _write_final_summary(
            study_dir,
            spec=spec,
            results=all_results,
            decision="SOTA_PILOT_PROMOTION_CONTRACT_BROKEN",
        )
    stage3_manifest_paths = _write_promoted_manifests(
        study_dir=study_dir,
        source_stage_dir=study_dir / "promotions" / "stage2_mid_fidelity",
        source_stage_name="stage2_mid_fidelity",
        target_stage_name="stage3_full_fidelity",
        search_space=spec,
    )
    stage3_results = _execute_stage(
        plan=None,
        stage_dir=study_dir / "promotions" / "stage3_full_fidelity",
        stage_name="stage3_full_fidelity",
        candidate_manifest_paths=stage3_manifest_paths,
        runner_kwargs=_runner_kwargs(
            processed_manifest=processed_manifest,
            runtime_config=effective_runtime_config,
            device_config=device_config,
            repo_root=repo_root,
            split_family=spec.study.split_family,
            model_name=spec.study.model,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            split_seed=split_seed,
            model_seed=model_seed,
            study_id=study_id or stage1_plan.study_id,
        ),
        repo_root=repo_root,
        execute_candidate_fn=execute_candidate_fn,
    )
    all_results.extend(stage3_results)
    _write_stage_results(
        study_dir / "reports" / "stage3_results.csv",
        stage3_results,
        stage_name="stage3_full_fidelity",
    )
    decision = (
        "SOTA_PILOT_COMPLETED_FINAL_CANDIDATE_READY_FOR_BAKEOFF"
        if not _has_failure(stage3_results)
        else "SOTA_PILOT_EXECUTION_UNSTABLE"
    )
    return _write_final_summary(study_dir, spec=spec, results=all_results, decision=decision)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sequential ML1M staged SOTA tuning pilot.")
    parser.add_argument("--search-space", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tuning"))
    parser.add_argument("--study-id", default=None)
    parser.add_argument("--processed-manifest", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime/base.yaml"))
    parser.add_argument("--device-config", type=Path, default=Path("configs/runtime/devices/local_u300_24gb.yaml"))
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--model-seed", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_sota_tuning_pilot(
        search_space=args.search_space,
        output_dir=args.output_dir,
        study_id=args.study_id,
        processed_manifest=args.processed_manifest,
        runtime_config=args.runtime_config,
        device_config=args.device_config,
        cache_root=args.cache_root,
        repo_root=args.repo_root,
        overwrite=args.overwrite,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"].startswith("SOTA_PILOT_COMPLETED") else 1


def _validate_pilot_scope(spec: SearchSpaceSpec) -> None:
    if spec.study.dataset != "ml1m":
        raise ValueError("20b pilot requires ML1M; ML100K/synthetic fallback is not allowed")
    if spec.study.model != "cb_svdpp":
        raise ValueError("20b pilot requires cb_svdpp")
    if spec.budget.max_parallel != 1:
        raise ValueError("20b pilot requires max_parallel=1")
    if spec.schedule is None or len(spec.schedule.stages) != 3:
        raise ValueError("20b pilot requires exactly three fidelity stages")
    stage1, stage2, stage3 = spec.schedule.stages
    if stage1.max_candidates > STAGE1_MAX_CANDIDATES:
        raise ValueError("stage1 max_candidates must be <= 16")
    if stage2.max_candidates > STAGE2_MAX_CANDIDATES:
        raise ValueError("stage2 max_candidates must be <= 4")
    if stage3.max_candidates > STAGE3_MAX_CANDIDATES:
        raise ValueError("stage3 max_candidates must be <= 1")
    if stage1.promote_top_k != stage2.max_candidates:
        raise ValueError("stage1 promote_top_k must match stage2 max_candidates")
    if stage2.promote_top_k != stage3.max_candidates:
        raise ValueError("stage2 promote_top_k must match stage3 max_candidates")
    if sum(stage.max_candidates for stage in spec.schedule.stages) > TOTAL_EXECUTION_HARD_CAP:
        raise ValueError("20b pilot total executed candidates must be <= 21")
    forbidden_prefixes = ("clustering.induction.",)
    forbidden_paths = {
        "training.latent_dim",
        "clustering.n_user_clusters",
        "clustering.n_item_clusters",
        "clustering.kmeans_n_init",
    }
    for dimension_name, dimension in spec.search_space.items():
        target_path = dimension.target_path or dimension_name
        if target_path in forbidden_paths or target_path.startswith(forbidden_prefixes):
            raise ValueError(f"20b pilot must not vary outer/fidelity parameter: {target_path}")


def _stage_plan(spec: SearchSpaceSpec, *, study_id: str | None, stage_name: str) -> Any:
    stage = next(stage for stage in spec.schedule.stages if stage.name == stage_name)
    plan = build_study_plan(
        spec,
        stage_name=stage.name,
        max_candidates=stage.max_candidates,
        stage_overrides=stage.overrides,
    )
    if study_id is None or study_id == plan.study_id:
        return plan
    from recsys_lab.tuning.planner import StudyPlan

    return StudyPlan(
        study_id=study_id,
        search_space=plan.search_space,
        candidates=plan.candidates,
        artifact_reuse_groups=plan.artifact_reuse_groups,
        stage_name=plan.stage_name,
        stage_overrides=plan.stage_overrides,
    )


def _execute_stage(
    *,
    plan: Any | None,
    stage_dir: Path,
    stage_name: str,
    candidate_manifest_paths: list[Path],
    runner_kwargs: dict[str, Any],
    repo_root: Path,
    execute_candidate_fn: ExecuteCandidateFn | None,
) -> list[CandidateExecutionResult]:
    results: list[CandidateExecutionResult] = []
    executor = execute_candidate_fn or execute_candidate
    for manifest_path in candidate_manifest_paths:
        result = executor(manifest_path, runner_kwargs=runner_kwargs, repo_root=repo_root)
        results.append(result)
        update_candidate_manifest_with_execution_result(manifest_path, result)
        if plan is not None:
            write_study_execution_artifacts(plan, stage_dir, results)
        if result.execution_status != "succeeded":
            break
    return results


def _write_promoted_manifests(
    *,
    study_dir: Path,
    source_stage_dir: Path,
    source_stage_name: str,
    target_stage_name: str,
    search_space: SearchSpaceSpec,
) -> list[Path]:
    promotion_dir = study_dir / "promotions" / target_stage_name
    promotion_plan = json.loads((promotion_dir / "promotion_plan.json").read_text(encoding="utf-8"))
    manifest_paths: list[Path] = []
    for index, promoted in enumerate(promotion_plan["promoted_candidates"]):
        source_manifest_path = _source_manifest_path(
            source_stage_dir=source_stage_dir,
            stage_name=source_stage_name,
            source_candidate_id=promoted["source_candidate_id"],
        )
        source_manifest = CandidateManifest.model_validate(
            json.loads(source_manifest_path.read_text(encoding="utf-8"))
        )
        promoted_config_path = Path(promoted["promoted_candidate_config_path"])
        materialized_config = load_yaml_file(promoted_config_path)
        materialized_config = _runner_compatible_promoted_config(materialized_config)
        dump_yaml_file(promoted_config_path, materialized_config)
        manifest = CandidateManifest(
            study_id=source_manifest.study_id,
            candidate_id=promoted["promoted_candidate_id"],
            candidate_index=index,
            objective_status="planned",
            execution_status="not_executed",
            stage_name=target_stage_name,
            study=search_space.study,
            base_model_config=source_manifest.base_model_config,
            parameter_values=dict(source_manifest.parameter_values),
            overrides=dict(source_manifest.overrides),
            materialized_config_payload=materialized_config,
            candidate_config_path=str(promoted_config_path),
            artifact_reuse_group_ids=dict(source_manifest.artifact_reuse_group_ids),
        )
        manifest_path = promoted_config_path.parent / "candidate_manifest.json"
        write_tuning_json(manifest, manifest_path)
        manifest_paths.append(manifest_path)
    return manifest_paths


def _source_manifest_path(*, source_stage_dir: Path, stage_name: str, source_candidate_id: str) -> Path:
    if stage_name == "stage1_low_fidelity":
        return source_stage_dir / "candidates" / source_candidate_id / "candidate_manifest.json"
    return source_stage_dir / "candidates" / source_candidate_id / "candidate_manifest.json"


def _write_stage_results(path: Path, results: list[CandidateExecutionResult], *, stage_name: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STAGE_RESULT_FIELDS)
        writer.writeheader()
        for result in results:
            manifest_path = _manifest_path_for_result(path.parent.parent, stage_name, result.candidate_id)
            row = _stage_result_row(result, manifest_path=manifest_path, stage_name=stage_name)
            writer.writerow(row)
    return path


def _manifest_path_for_result(study_dir: Path, stage_name: str, candidate_id: str) -> Path:
    if stage_name == "stage1_low_fidelity":
        return study_dir / "candidates" / candidate_id / "candidate_manifest.json"
    return study_dir / "promotions" / stage_name / "candidates" / candidate_id / "candidate_manifest.json"


def _stage_result_row(
    result: CandidateExecutionResult,
    *,
    manifest_path: Path,
    stage_name: str,
) -> dict[str, Any]:
    manifest = CandidateManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    metrics_payload = _load_json(result.metrics_path)
    performance_payload = _load_json(result.performance_profile_path)
    metrics = metrics_payload.get("metrics", {}) if isinstance(metrics_payload.get("metrics"), dict) else {}
    caches = metrics_payload.get("caches", {}) if isinstance(metrics_payload.get("caches"), dict) else {}
    timing = metrics_payload.get("timing", {}) if isinstance(metrics_payload.get("timing"), dict) else {}
    cluster_cache = caches.get("cluster_artifacts", {}) if isinstance(caches.get("cluster_artifacts"), dict) else {}
    history_cache = (
        caches.get("user_cluster_history", {})
        if isinstance(caches.get("user_cluster_history"), dict)
        else {}
    )
    return {
        "study_id": result.study_id,
        "stage_name": stage_name,
        "candidate_id": result.candidate_id,
        "execution_status": result.execution_status,
        "validation_rmse": _csv_value(metrics.get("validation_rmse")),
        "validation_mae": _csv_value(metrics.get("validation_mae")),
        "fit_model_seconds": _csv_value(_stage_seconds(performance_payload, "fit_model")),
        "total_wall_seconds": _csv_value(performance_payload.get("total_profiled_wall_clock_seconds")),
        "cluster_total_seconds": _csv_value(
            _stage_seconds(performance_payload, "build_cluster_artifacts")
            or timing.get("cluster_induction_wall_clock_seconds")
        ),
        "cluster_cache_status": _csv_value(cluster_cache.get("status")),
        "user_cluster_history_cache_status": _csv_value(history_cache.get("status")),
        "candidate_config_path": manifest.candidate_config_path or "",
        "candidate_manifest_path": str(manifest_path),
        "cluster_reuse_group_id": manifest.artifact_reuse_group_ids.get("cluster_artifacts", ""),
        "run_dir": result.run_dir or "",
        "run_manifest_path": result.run_manifest_path or "",
        "metrics_path": result.metrics_path or "",
        "performance_profile_path": result.performance_profile_path or "",
        "kernel_profile_path": result.kernel_profile_path or "",
        "error_message": result.error_message or "",
    }


def _write_final_summary(
    study_dir: Path,
    *,
    spec: SearchSpaceSpec,
    results: list[CandidateExecutionResult],
    decision: str,
) -> dict[str, Any]:
    reports_dir = study_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for stage_results_path in (
        reports_dir / "stage1_results.csv",
        reports_dir / "stage2_results.csv",
        reports_dir / "stage3_results.csv",
    ):
        if not stage_results_path.exists():
            continue
        with stage_results_path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    summary_rows = [_summary_row(row) for row in rows]
    ranked_summary_rows = _rank_summary_rows(summary_rows)
    final_row = _final_stage_row(ranked_summary_rows)
    stage_counts = _planned_stage_counts(spec)
    success_counts = _stage_success_counts(rows)
    final_payload = {
        "decision": _decision_with_quality_context(decision, final_row),
        "study_id": study_dir.name,
        "stage1_candidates": stage_counts["stage1_low_fidelity"],
        "stage2_candidates": stage_counts["stage2_mid_fidelity"],
        "stage3_candidates": stage_counts["stage3_full_fidelity"],
        "stage1_successes": success_counts["stage1_low_fidelity"],
        "stage2_successes": success_counts["stage2_mid_fidelity"],
        "stage3_successes": success_counts["stage3_full_fidelity"],
        "final_candidate_id": None if final_row is None else final_row["candidate_id"],
        "final_candidate_config_path": None if final_row is None else final_row["candidate_config_path"],
        "final_validation_rmse": None if final_row is None else _float_or_none(final_row["validation_rmse"]),
        "final_validation_mae": None if final_row is None else _float_or_none(final_row["validation_mae"]),
        "incumbent_config": INCUMBENT_CONFIG,
        "incumbent_validation_rmse": INCUMBENT_VALIDATION_RMSE,
        "incumbent_validation_mae": INCUMBENT_VALIDATION_MAE,
        "final_validation_rmse_delta_vs_incumbent": _metric_delta(
            None if final_row is None else final_row["validation_rmse"],
            INCUMBENT_VALIDATION_RMSE,
        ),
        "final_validation_mae_delta_vs_incumbent": _metric_delta(
            None if final_row is None else final_row["validation_mae"],
            INCUMBENT_VALIDATION_MAE,
        ),
        "claim_boundary": "local ML1M staged SOTA tuning pilot only",
    }
    write_tuning_json(final_payload, reports_dir / "sota_pilot_decision.json")
    with (reports_dir / "sota_pilot_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PILOT_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(ranked_summary_rows)
    return {
        "study_id": study_dir.name,
        "study_dir": str(study_dir),
        "executed_candidate_count": len(results),
        "decision": final_payload["decision"],
        "decision_json": str(reports_dir / "sota_pilot_decision.json"),
        "summary_csv": str(reports_dir / "sota_pilot_summary.csv"),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _runner_kwargs(
    *,
    processed_manifest: Path,
    runtime_config: Path,
    device_config: Path,
    repo_root: Path,
    split_family: str,
    model_name: str,
    train_ratio: float,
    validation_ratio: float,
    split_seed: int,
    model_seed: int,
    study_id: str,
) -> dict[str, Any]:
    return {
        "processed_manifest_path": _resolve_path(processed_manifest, repo_root=repo_root),
        "runtime_config_path": runtime_config,
        "device_config_path": _resolve_path(device_config, repo_root=repo_root),
        "split_config": SplitConfig(train_ratio=train_ratio, validation_ratio=validation_ratio, seed=split_seed),
        "model_seed": model_seed,
        "repo_root": repo_root,
        "model_name": model_name,
        "split_family": split_family,
        "evaluate_test": False,
        "use_split_cache": None,
        "use_training_index_cache": True,
        "use_cluster_artifact_cache": True,
        "command": f"scripts/run_sota_tuning_pilot.py --study-id {study_id}",
    }


def _summary_row(row: dict[str, Any]) -> dict[str, Any]:
    config_payload = load_yaml_file(Path(str(row["candidate_config_path"])))
    training = config_payload.get("training", {})
    clustering = config_payload.get("clustering", {})
    return {
        "stage_name": row.get("stage_name", ""),
        "rank": "",
        "candidate_id": row.get("candidate_id", ""),
        "source_candidate_id": _source_candidate_id(config_payload),
        "execution_status": row.get("execution_status", ""),
        "alpha": _csv_value(clustering.get("alpha")),
        "learning_rate": _csv_value(training.get("learning_rate")),
        "lambda_p": _csv_value(training.get("lambda_p")),
        "lambda_q": _csv_value(training.get("lambda_q")),
        "lambda_y": _csv_value(training.get("lambda_y")),
        "lambda_pC": _csv_value(training.get("lambda_pC")),
        "lambda_qC": _csv_value(training.get("lambda_qC")),
        "lambda_yC": _csv_value(training.get("lambda_yC")),
        "training_epochs": _csv_value(training.get("epochs")),
        "validation_rmse": row.get("validation_rmse", ""),
        "validation_mae": row.get("validation_mae", ""),
        "fit_model_seconds": row.get("fit_model_seconds", ""),
        "total_wall_seconds": row.get("total_wall_seconds", ""),
        "cluster_total_seconds": row.get("cluster_total_seconds", ""),
        "cluster_cache_status": row.get("cluster_cache_status", ""),
        "user_cluster_history_cache_status": row.get("user_cluster_history_cache_status", ""),
        "candidate_config_path": row.get("candidate_config_path", ""),
        "run_dir": row.get("run_dir", ""),
    }


def _rank_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked_rows: list[dict[str, Any]] = []
    for stage_name in ("stage1_low_fidelity", "stage2_mid_fidelity", "stage3_full_fidelity"):
        stage_rows = [row for row in rows if row["stage_name"] == stage_name]
        succeeded = [row for row in stage_rows if row["execution_status"] == "succeeded"]
        succeeded.sort(
            key=lambda row: (
                _float_or_inf(row["validation_rmse"]),
                _float_or_inf(row["validation_mae"]),
                _float_or_inf(row["fit_model_seconds"]),
            )
        )
        ranks = {row["candidate_id"]: rank for rank, row in enumerate(succeeded, start=1)}
        for row in stage_rows:
            row["rank"] = ranks.get(row["candidate_id"], "")
        ranked_rows.extend(sorted(stage_rows, key=lambda row: int(row["rank"]) if row["rank"] else 10**9))
    return ranked_rows


def _final_stage_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    final_rows = [
        row
        for row in rows
        if row["stage_name"] == "stage3_full_fidelity" and row["execution_status"] == "succeeded"
    ]
    if not final_rows:
        return None
    return sorted(
        final_rows,
        key=lambda row: (
            _float_or_inf(row["validation_rmse"]),
            _float_or_inf(row["validation_mae"]),
            _float_or_inf(row["fit_model_seconds"]),
        ),
    )[0]


def _stage_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        stage_name: sum(row.get("stage_name") == stage_name for row in rows)
        for stage_name in ("stage1_low_fidelity", "stage2_mid_fidelity", "stage3_full_fidelity")
    }


def _planned_stage_counts(spec: SearchSpaceSpec) -> dict[str, int]:
    configured = {
        stage.name: stage.max_candidates
        for stage in ([] if spec.schedule is None else spec.schedule.stages)
    }
    return {
        stage_name: configured.get(stage_name, 0)
        for stage_name in ("stage1_low_fidelity", "stage2_mid_fidelity", "stage3_full_fidelity")
    }


def _stage_success_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        stage_name: sum(
            row.get("stage_name") == stage_name and row.get("execution_status") == "succeeded"
            for row in rows
        )
        for stage_name in ("stage1_low_fidelity", "stage2_mid_fidelity", "stage3_full_fidelity")
    }


def _decision_with_quality_context(decision: str, final_row: dict[str, Any] | None) -> str:
    if decision != "SOTA_PILOT_COMPLETED_FINAL_CANDIDATE_READY_FOR_BAKEOFF":
        return decision
    final_rmse = None if final_row is None else _float_or_none(final_row["validation_rmse"])
    if final_rmse is None:
        return "SOTA_PILOT_EXECUTION_UNSTABLE"
    if final_rmse > INCUMBENT_VALIDATION_RMSE:
        return "SOTA_PILOT_COMPLETED_INCUMBENT_STILL_REFERENCE"
    return decision


def _source_candidate_id(config_payload: dict[str, Any]) -> str:
    metadata = config_payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return ""
    promotion = metadata.get("promotion", {})
    if not isinstance(promotion, dict):
        return ""
    return str(promotion.get("source_candidate_id", ""))


def _runner_compatible_promoted_config(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("promotion", None)
    return payload


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_inf(value: Any) -> float:
    parsed = _float_or_none(value)
    return parsed if parsed is not None else float("inf")


def _metric_delta(candidate_value: Any, incumbent_value: float) -> float | None:
    parsed = _float_or_none(candidate_value)
    return None if parsed is None else parsed - incumbent_value


def _runtime_config_with_cache_root(
    runtime_config: Path,
    *,
    cache_root: Path,
    study_dir: Path,
    repo_root: Path,
) -> Path:
    source_path = _resolve_path(runtime_config, repo_root=repo_root)
    payload = load_yaml_file(source_path)
    runtime = dict(payload.get("runtime", {}))
    runtime["cache_root"] = str(cache_root)
    payload["runtime"] = runtime
    target_path = study_dir / "runtime_config.isolated_cache.yaml"
    dump_yaml_file(target_path, payload)
    return target_path.resolve()


def _has_failure(results: list[CandidateExecutionResult]) -> bool:
    return any(result.execution_status != "succeeded" for result in results)


def _load_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _stage_seconds(profile: dict[str, Any], stage_name: str) -> float | None:
    stages = profile.get("stages")
    if not isinstance(stages, list):
        return None
    for stage in stages:
        if isinstance(stage, dict) and stage.get("name") == stage_name and stage.get("wall_clock_seconds") is not None:
            return float(stage["wall_clock_seconds"])
    return None


def _csv_value(value: Any) -> Any:
    return "" if value is None else value


def _resolve_path(path: Path, *, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
