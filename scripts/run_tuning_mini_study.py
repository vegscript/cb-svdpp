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

CLAIM_BOUNDARY = "ML1M cache-aware tuning mini study only; no performance or quality claim."
MAX_CANDIDATE_HARD_CAP = 3
ExecuteCandidateFn = Callable[..., Any]
WriteExecutionArtifactsFn = Callable[..., dict[str, Path]]


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
        if result.execution_status != "succeeded":
            break

    cache_reuse_evidence = None
    if require_cache_reuse_evidence and len(results) >= 2 and all(
        result.execution_status == "succeeded" for result in results
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
        "claim_boundary": CLAIM_BOUNDARY,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny sequential cache-aware tuning mini study.")
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
    parser.add_argument("--max-candidates", type=int, default=2)
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
        help="Require the 19d benchmark contract; currently this means dataset ml1m.",
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
            "19d benchmark mode requires dataset ml1m; tiny/ML100K may only be used with --no-benchmark-mode"
        )


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
