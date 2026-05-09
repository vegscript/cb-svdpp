from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

CLAIM_BOUNDARY = "ML1M cache-aware tuning mini/small study only; no performance or quality claim."
DEFAULT_MAX_CANDIDATES = 8
MAX_CANDIDATE_HARD_CAP = 16
MIN_SUCCEEDED_CANDIDATES_FOR_SELECTION = 6
MIN_SUCCEEDED_CANDIDATES_FOR_SMALL_STUDY_SELECTION = 10
MIN_FOLLOWUP_CACHE_HITS_FOR_SMALL_STUDY = 10
ExecuteCandidateFn = Callable[..., Any]
WriteExecutionArtifactsFn = Callable[..., dict[str, Path]]
RANKING_FIELDS = [
    "rank",
    "candidate_id",
    "execution_status",
    "alpha",
    "learning_rate",
    "validation_rmse",
    "validation_mae",
    "fit_model_seconds",
    "cluster_total_seconds",
    "cluster_cache_status",
    "user_cluster_history_cache_status",
    "cluster_reuse_group_id",
    "selected",
    "selection_reason",
    "run_dir",
]


def run_tuning_mini_study(
    *,
    search_space: Path,
    output_dir: Path,
    study_id: str | None,
    processed_manifest: Path,
    runtime_config: Path,
    device_config: Path,
    cache_root: Path | None,
    repo_root: Path,
    max_candidates: int,
    overwrite: bool,
    train_ratio: float,
    validation_ratio: float,
    split_seed: int,
    model_seed: int,
    evaluate_test: bool,
    use_split_cache: bool | None,
    use_training_index_cache: bool,
    use_cluster_artifact_cache: bool,
    require_cache_reuse_evidence: bool,
    benchmark_mode: bool,
    runner_kwargs: dict[str, Any] | None = None,
    execute_candidate_fn: ExecuteCandidateFn | None = None,
    write_execution_artifacts_fn: WriteExecutionArtifactsFn | None = None,
) -> dict[str, Any]:
    if max_candidates < 1:
        raise ValueError("--max-candidates must be at least 1")
    if max_candidates > MAX_CANDIDATE_HARD_CAP:
        raise ValueError(f"--max-candidates must be <= {MAX_CANDIDATE_HARD_CAP}")

    from scripts.plan_tuning_study import plan_tuning_study

    planning_result = plan_tuning_study(
        search_space_path=search_space,
        output_dir=output_dir,
        study_id=study_id,
        overwrite=overwrite,
        repo_root=repo_root,
    )
    study_dir = Path(str(planning_result["study_dir"]))
    plan = _load_plan(study_dir)
    if benchmark_mode:
        _validate_benchmark_mode_scope(plan)
    candidate_ids = [candidate.candidate_id for candidate in plan.candidates[:max_candidates]]
    if not candidate_ids:
        raise ValueError("planned study has no candidates to execute")
    effective_cache_root = (
        _resolve_path(cache_root, repo_root=repo_root) if cache_root is not None else study_dir / "local_cache"
    )
    effective_runtime_config = _runtime_config_with_cache_root(
        runtime_config,
        cache_root=effective_cache_root,
        study_dir=study_dir,
        repo_root=repo_root,
    )

    results = []
    selection_paths = _write_ranking_and_selection(plan, study_dir, repo_root=repo_root)
    for candidate_id in candidate_ids:
        candidate_manifest_path = study_dir / "candidates" / candidate_id / "candidate_manifest.json"
        call_kwargs = {
            "processed_manifest_path": _resolve_path(processed_manifest, repo_root=repo_root),
            "runtime_config_path": effective_runtime_config,
            "device_config_path": _resolve_path(device_config, repo_root=repo_root),
            "split_config": _build_split_config(
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                split_seed=split_seed,
            ),
            "model_seed": model_seed,
            "repo_root": repo_root,
            "model_name": plan.search_space.study.model,
            "split_family": plan.search_space.study.split_family,
            "evaluate_test": evaluate_test,
            "use_split_cache": use_split_cache,
            "use_training_index_cache": use_training_index_cache,
            "use_cluster_artifact_cache": use_cluster_artifact_cache,
            "command": (
                "scripts/run_tuning_mini_study.py "
                f"--study-id {plan.study_id} --max-candidates {max_candidates} "
                f"[candidate_id={candidate_id}]"
            ),
        }
        if runner_kwargs is not None:
            call_kwargs.update(runner_kwargs)

        executor = execute_candidate_fn or _execute_candidate
        result = executor(
            candidate_manifest_path,
            runner_kwargs=call_kwargs,
            repo_root=repo_root,
        )
        results.append(result)
        writer = write_execution_artifacts_fn or _write_study_execution_artifacts
        writer(plan, study_dir, results)
        selection_paths = _write_ranking_and_selection(plan, study_dir, repo_root=repo_root)
        if result.execution_status != "succeeded":
            break

    cache_reuse_evidence = None
    if require_cache_reuse_evidence and len(results) >= 2 and all(
        result.execution_status == "succeeded" for result in results[:2]
    ):
        cache_reuse_evidence = _validate_cache_reuse_evidence(study_dir, candidate_ids[:2])

    return {
        "study_id": plan.study_id,
        "study_dir": str(study_dir),
        "cache_root": str(effective_cache_root),
        "runtime_config_path": str(effective_runtime_config),
        "planned_candidate_count": len(plan.candidates),
        "executed_candidate_count": len(results),
        "candidate_ids": candidate_ids,
        "execution_statuses": {result.candidate_id: result.execution_status for result in results},
        "cache_reuse_evidence": cache_reuse_evidence,
        "mini_study_summary_csv": str(study_dir / "reports" / "mini_study_summary.csv"),
        "mini_study_summary_json": str(study_dir / "reports" / "mini_study_summary.json"),
        "candidate_ranking_csv": str(selection_paths["candidate_ranking_csv"]),
        "selected_candidate_json": str(selection_paths["selected_candidate_json"]),
        "selected_candidate_config": (
            str(selection_paths["selected_candidate_config"])
            if selection_paths.get("selected_candidate_config") is not None
            else None
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sequential cache-aware ML1M tuning mini/small study.")
    parser.add_argument("--search-space", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tuning"))
    parser.add_argument("--study-id", default=None)
    parser.add_argument("--processed-manifest", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime/base.yaml"))
    parser.add_argument("--device-config", type=Path, default=Path("configs/runtime/devices/local_u300_24gb.yaml"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Optional cache root. Defaults to an isolated cache under the study directory.",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--model-seed", type=int, default=1)
    parser.add_argument("--evaluate-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split-cache", choices=["auto", "enable", "disable"], default="auto")
    parser.add_argument("--training-index-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cluster-artifact-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-reuse-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--benchmark-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the ML1M benchmark contract; tiny/ML100K are only allowed with --no-benchmark-mode.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_tuning_mini_study(
        search_space=args.search_space,
        output_dir=args.output_dir,
        study_id=args.study_id,
        processed_manifest=args.processed_manifest,
        runtime_config=args.runtime_config,
        device_config=args.device_config,
        cache_root=args.cache_root,
        repo_root=args.repo_root.resolve(),
        max_candidates=args.max_candidates,
        overwrite=args.overwrite,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        evaluate_test=args.evaluate_test,
        use_split_cache=_split_cache_value(args.split_cache),
        use_training_index_cache=args.training_index_cache,
        use_cluster_artifact_cache=args.cluster_artifact_cache,
        require_cache_reuse_evidence=args.cache_reuse_check,
        benchmark_mode=args.benchmark_mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if any(status != "succeeded" for status in result["execution_statuses"].values()) else 0


def _load_plan(study_dir: Path) -> Any:
    from recsys_lab.config.loader import load_yaml_file
    from recsys_lab.tuning import SearchSpaceSpec, build_study_plan
    from recsys_lab.tuning.planner import StudyPlan

    search_space = SearchSpaceSpec.model_validate(load_yaml_file(study_dir / "search_space.yaml"))
    plan = build_study_plan(search_space)
    study_manifest_path = study_dir / "study_manifest.json"
    if not study_manifest_path.exists():
        return plan
    study_manifest = json.loads(study_manifest_path.read_text(encoding="utf-8"))
    study_id = str(study_manifest["study_id"])
    if plan.study_id == study_id:
        return plan
    return StudyPlan(
        study_id=study_id,
        search_space=plan.search_space,
        candidates=plan.candidates,
        artifact_reuse_groups=plan.artifact_reuse_groups,
    )


def _execute_candidate(candidate_manifest_path: Path, *, runner_kwargs: dict[str, Any], repo_root: Path) -> Any:
    from recsys_lab.tuning.execution import execute_candidate

    return execute_candidate(candidate_manifest_path, runner_kwargs=runner_kwargs, repo_root=repo_root)


def _write_study_execution_artifacts(plan: Any, study_dir: Path, results: list[Any]) -> None:
    from recsys_lab.tuning.writers import write_study_execution_artifacts

    write_study_execution_artifacts(plan, study_dir, results)


def _validate_benchmark_mode_scope(plan: Any) -> None:
    dataset = str(plan.search_space.study.dataset)
    if dataset != "ml1m":
        raise ValueError(
            "ML1M benchmark mode requires dataset ml1m; tiny/ML100K may only be used with --no-benchmark-mode"
        )


def _write_ranking_and_selection(plan: Any, study_dir: Path, *, repo_root: Path) -> dict[str, Path | None]:
    reports_dir = study_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = reports_dir / "candidate_ranking.csv"
    selected_dir = study_dir / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_path = selected_dir / "selected_candidate.json"
    selected_config_path = selected_dir / "selected_candidate_config.yaml"

    summary_path = reports_dir / "candidate_summary.csv"
    summary_rows = _candidate_summary_rows(summary_path) if summary_path.exists() else []
    rows_by_candidate = {row["candidate_id"]: row for row in summary_rows}

    selectable_rows = []
    ranking_rows_by_candidate: dict[str, dict[str, str]] = {}
    for candidate in plan.candidates:
        summary_row = rows_by_candidate.get(candidate.candidate_id, {})
        execution_status = summary_row.get("execution_status", "not_executed")
        validation_rmse = _parse_float(summary_row.get("validation_rmse"))
        ranking_row = {
            "rank": "",
            "candidate_id": candidate.candidate_id,
            "candidate_index": str(candidate.index),
            "execution_status": execution_status,
            "alpha": _parameter_value(candidate, "alpha"),
            "learning_rate": _parameter_value(candidate, "learning_rate"),
            "validation_rmse": summary_row.get("validation_rmse", ""),
            "validation_mae": summary_row.get("validation_mae", ""),
            "fit_model_seconds": summary_row.get("fit_model_seconds", ""),
            "cluster_total_seconds": summary_row.get("cluster_total_seconds", ""),
            "cluster_cache_status": summary_row.get("cluster_cache_status", ""),
            "user_cluster_history_cache_status": summary_row.get("user_cluster_history_cache_status", ""),
            "cluster_reuse_group_id": summary_row.get("cluster_reuse_group_id", ""),
            "selected": "false",
            "selection_reason": _non_selection_reason(execution_status, validation_rmse),
            "run_dir": summary_row.get("run_dir", ""),
            "candidate_config_path": summary_row.get("candidate_config_path", ""),
        }
        ranking_rows_by_candidate[candidate.candidate_id] = ranking_row
        if execution_status == "succeeded" and validation_rmse is not None:
            selectable_rows.append(ranking_row)

    selectable_rows.sort(
        key=lambda row: (
            _parse_float(row["validation_rmse"]) if _parse_float(row["validation_rmse"]) is not None else float("inf"),
            _parse_float(row["validation_mae"]) if _parse_float(row["validation_mae"]) is not None else float("inf"),
            _parse_float(row["fit_model_seconds"])
            if _parse_float(row["fit_model_seconds"]) is not None
            else float("inf"),
            int(row["candidate_index"]),
        )
    )
    for rank, row in enumerate(selectable_rows, start=1):
        row["rank"] = str(rank)

    min_succeeded_candidates = _min_succeeded_candidates_for_selection(plan)
    decision = "EXECUTION_UNSTABLE_FIX_BEFORE_TUNING"
    selected_row = None
    if len(selectable_rows) >= min_succeeded_candidates:
        reuse_valid, reuse_reason = _selection_cache_reuse_status(plan, ranking_rows_by_candidate)
        if not reuse_valid:
            decision = "REUSE_CONTRACT_STILL_BROKEN"
            for row in selectable_rows:
                row["selection_reason"] = reuse_reason
        else:
            decision = "SELECTED_CANDIDATE_READY_FOR_BAKEOFF"
            selected_row = selectable_rows[0]
            selected_row["selected"] = "true"
            selected_row["selection_reason"] = (
                "lowest validation_rmse among succeeded candidates; "
                "ties break by validation_mae then fit_model_seconds"
            )
            for row in selectable_rows[1:]:
                row["selection_reason"] = "not selected by validation_rmse ranking"
    else:
        for row in selectable_rows:
            row["selection_reason"] = (
                f"selection requires at least {min_succeeded_candidates} succeeded candidates"
            )

    ranked_rows = sorted(
        ranking_rows_by_candidate.values(),
        key=lambda row: (int(row["rank"]) if row["rank"] else 10**9, int(row["candidate_index"])),
    )
    with ranking_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANKING_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked_rows)

    selected_candidate_config: Path | None = None
    if selected_row is not None:
        source_config = _resolve_path(Path(selected_row["candidate_config_path"]), repo_root=repo_root)
        shutil.copyfile(source_config, selected_config_path)
        selected_candidate_config = selected_config_path
    elif selected_config_path.exists():
        selected_config_path.unlink()

    selected_payload = {
        "study_id": plan.study_id,
        "selected_candidate_id": selected_row["candidate_id"] if selected_row is not None else None,
        "selected_rank": int(selected_row["rank"]) if selected_row is not None else None,
        "rank": int(selected_row["rank"]) if selected_row is not None else None,
        "alpha": selected_row["alpha"] if selected_row is not None else None,
        "learning_rate": selected_row["learning_rate"] if selected_row is not None else None,
        "lambda_q": _selected_parameter_value(plan, selected_row, "lambda_q") if selected_row is not None else None,
        "validation_rmse": selected_row["validation_rmse"] if selected_row is not None else None,
        "validation_mae": selected_row["validation_mae"] if selected_row is not None else None,
        "fit_model_seconds": selected_row["fit_model_seconds"] if selected_row is not None else None,
        "cluster_total_seconds": selected_row["cluster_total_seconds"] if selected_row is not None else None,
        "cluster_cache_status": selected_row["cluster_cache_status"] if selected_row is not None else None,
        "user_cluster_history_cache_status": (
            selected_row["user_cluster_history_cache_status"] if selected_row is not None else None
        ),
        "cluster_reuse_group_id": selected_row["cluster_reuse_group_id"] if selected_row is not None else None,
        "selection_reason": (
            selected_row["selection_reason"]
            if selected_row is not None
            else f"fewer than {min_succeeded_candidates} succeeded candidates or cache reuse not validated"
        ),
        "candidate_config_path": selected_row["candidate_config_path"] if selected_row is not None else None,
        "selected_candidate_config_path": (
            str(selected_candidate_config) if selected_candidate_config is not None else None
        ),
        "selected_config_path": str(selected_candidate_config) if selected_candidate_config is not None else None,
        "candidate_ranking_path": str(ranking_path),
        "decision": decision,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    selected_path.write_text(json.dumps(selected_payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "candidate_ranking_csv": ranking_path,
        "selected_candidate_json": selected_path,
        "selected_candidate_config": selected_candidate_config,
    }


def _parameter_value(candidate: Any, name: str) -> str:
    value = candidate.parameter_values.get(name)
    return "" if value is None else str(value)


def _selected_parameter_value(plan: Any, selected_row: dict[str, str], name: str) -> str:
    selected_candidate_id = selected_row["candidate_id"]
    for candidate in plan.candidates:
        if candidate.candidate_id == selected_candidate_id:
            return _parameter_value(candidate, name)
    return ""


def _min_succeeded_candidates_for_selection(plan: Any) -> int:
    if len(plan.candidates) >= MIN_SUCCEEDED_CANDIDATES_FOR_SMALL_STUDY_SELECTION:
        return MIN_SUCCEEDED_CANDIDATES_FOR_SMALL_STUDY_SELECTION
    return min(MIN_SUCCEEDED_CANDIDATES_FOR_SELECTION, len(plan.candidates))


def _selection_cache_reuse_status(
    plan: Any,
    rows_by_candidate: dict[str, dict[str, str]],
) -> tuple[bool, str]:
    succeeded_rows = [
        rows_by_candidate[candidate.candidate_id]
        for candidate in plan.candidates
        if rows_by_candidate.get(candidate.candidate_id, {}).get("execution_status") == "succeeded"
    ]
    if len(succeeded_rows) < 2:
        return False, "cache reuse validation requires at least two succeeded candidates"

    first = succeeded_rows[0]
    first_cluster = first.get("cluster_cache_status", "")
    first_history = first.get("user_cluster_history_cache_status", "")
    if first_cluster not in {"miss", "build"}:
        return False, f"first succeeded candidate must prove cold cluster cache status, got {first_cluster!r}"
    if first_history not in {"miss", "build"}:
        return (
            False,
            f"first succeeded candidate must prove cold user-cluster-history cache status, got {first_history!r}",
        )

    followup_rows = succeeded_rows[1:]
    required_hits = _required_followup_cache_hits(plan, len(followup_rows))
    cluster_hit_count = sum(row.get("cluster_cache_status") == "hit" for row in followup_rows)
    history_hit_count = sum(row.get("user_cluster_history_cache_status") == "hit" for row in followup_rows)
    if cluster_hit_count < required_hits:
        return False, f"cluster cache reuse requires at least {required_hits} follow-up hits, got {cluster_hit_count}"
    if history_hit_count < required_hits:
        return False, (
            f"user-cluster-history cache reuse requires at least {required_hits} follow-up hits, "
            f"got {history_hit_count}"
        )

    reuse_group_ids = {
        row.get("cluster_reuse_group_id", "")
        for row in succeeded_rows[: required_hits + 1]
    }
    if len(reuse_group_ids) != 1 or "" in reuse_group_ids:
        return False, "succeeded candidates must share one non-empty cluster_reuse_group_id"
    return True, "cache reuse contract validated"


def _required_followup_cache_hits(plan: Any, followup_count: int) -> int:
    if len(plan.candidates) >= 12:
        return MIN_FOLLOWUP_CACHE_HITS_FOR_SMALL_STUDY
    return followup_count


def _non_selection_reason(execution_status: str, validation_rmse: float | None) -> str:
    if execution_status != "succeeded":
        return f"not selectable: execution_status={execution_status}"
    if validation_rmse is None:
        return "not selectable: missing validation_rmse"
    return "pending selection"


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _runtime_config_with_cache_root(
    runtime_config: Path,
    *,
    cache_root: Path,
    study_dir: Path,
    repo_root: Path,
) -> Path:
    from recsys_lab.config.loader import dump_yaml_file, load_yaml_file

    source_path = _resolve_path(runtime_config, repo_root=repo_root)
    payload = load_yaml_file(source_path)
    runtime = dict(payload.get("runtime", {}))
    runtime["cache_root"] = str(cache_root)
    payload["runtime"] = runtime
    target_path = study_dir / "runtime_config.isolated_cache.yaml"
    dump_yaml_file(target_path, payload)
    return target_path.resolve()


def _validate_cache_reuse_evidence(study_dir: Path, candidate_ids: list[str]) -> dict[str, Any]:
    if len(candidate_ids) < 2:
        raise ValueError("cache reuse evidence requires at least two executed candidates")

    rows = _candidate_summary_rows(study_dir / "reports" / "candidate_summary.csv")
    rows_by_candidate = {row["candidate_id"]: row for row in rows}
    first = rows_by_candidate.get(candidate_ids[0])
    second = rows_by_candidate.get(candidate_ids[1])
    if first is None or second is None:
        raise ValueError("candidate_summary.csv is missing executed candidates for cache reuse evidence")

    first_cluster = first.get("cluster_cache_status", "")
    first_history = first.get("user_cluster_history_cache_status", "")
    second_cluster = second.get("cluster_cache_status", "")
    second_history = second.get("user_cluster_history_cache_status", "")
    if first_cluster not in {"miss", "build"}:
        raise ValueError(f"first candidate must prove cold cluster cache status, got {first_cluster!r}")
    if first_history not in {"miss", "build"}:
        raise ValueError(f"first candidate must prove cold user-cluster-history cache status, got {first_history!r}")
    if second_cluster != "hit":
        raise ValueError(f"second candidate must prove warm cluster cache hit, got {second_cluster!r}")
    if second_history != "hit":
        raise ValueError(f"second candidate must prove warm user-cluster-history cache hit, got {second_history!r}")

    first_group = first.get("cluster_reuse_group_id", "")
    second_group = second.get("cluster_reuse_group_id", "")
    if not first_group or first_group != second_group:
        raise ValueError("executed candidates must share one cluster_reuse_group_id")
    for row in (first, second):
        for field_name in ("run_dir", "run_manifest_path", "metrics_path", "performance_profile_path"):
            if not row.get(field_name):
                raise ValueError(f"cache reuse evidence row is missing {field_name}")

    return {
        "candidate_1": {
            "candidate_id": candidate_ids[0],
            "cluster_cache_status": first_cluster,
            "user_cluster_history_cache_status": first_history,
        },
        "candidate_2": {
            "candidate_id": candidate_ids[1],
            "cluster_cache_status": second_cluster,
            "user_cluster_history_cache_status": second_history,
        },
        "cluster_reuse_group_id": first_group,
        "status": "validated",
    }


def _candidate_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_split_config(*, train_ratio: float, validation_ratio: float, split_seed: int) -> Any:
    from recsys_lab.experiments.common import SplitConfig

    return SplitConfig(train_ratio=train_ratio, validation_ratio=validation_ratio, seed=split_seed)


def _resolve_path(path: Path, *, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _split_cache_value(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "enable"


if __name__ == "__main__":
    raise SystemExit(main())
