from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

CLAIM_BOUNDARY = "Single-candidate tuning execution smoke only; no performance or quality claim."
ExecuteCandidateFn = Callable[..., Any]
WriteExecutionArtifactsFn = Callable[..., dict[str, Path]]


def run_tuning_candidate_smoke(
    *,
    study_dir: Path,
    candidate_id: str | None,
    repo_root: Path,
    dry_run: bool,
    processed_manifest: Path | None,
    runtime_config: Path,
    device_config: Path,
    train_ratio: float,
    validation_ratio: float,
    split_seed: int,
    model_seed: int,
    evaluate_test: bool,
    use_split_cache: bool | None,
    use_training_index_cache: bool,
    use_cluster_artifact_cache: bool,
    runner_kwargs: dict[str, Any] | None = None,
    execute_candidate_fn: ExecuteCandidateFn | None = None,
    write_execution_artifacts_fn: WriteExecutionArtifactsFn | None = None,
) -> dict[str, Any]:
    study_dir = _resolve_path(study_dir, repo_root=repo_root)
    selected_manifest_path = _select_candidate_manifest(study_dir=study_dir, candidate_id=candidate_id)
    selected_candidate_id = selected_manifest_path.parent.name
    plan = _load_plan(study_dir)

    if dry_run:
        return {
            "study_dir": str(study_dir),
            "candidate_id": selected_candidate_id,
            "candidate_manifest": str(selected_manifest_path),
            "execution_status": "not_executed",
            "dry_run": True,
            "claim_boundary": CLAIM_BOUNDARY,
        }

    if processed_manifest is None:
        raise ValueError("--processed-manifest is required unless --dry-run is set")

    call_kwargs = {
        "processed_manifest_path": _resolve_path(processed_manifest, repo_root=repo_root),
        "runtime_config_path": _resolve_path(runtime_config, repo_root=repo_root),
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
            "scripts/run_tuning_candidate_smoke.py "
            f"--study-dir {study_dir} --candidate-id {selected_candidate_id}"
        ),
    }
    if runner_kwargs is not None:
        call_kwargs.update(runner_kwargs)

    if execute_candidate_fn is None:
        from recsys_lab.tuning.execution import execute_candidate as execute_candidate_fn
    if write_execution_artifacts_fn is None:
        from recsys_lab.tuning.writers import write_study_execution_artifacts as write_execution_artifacts_fn

    result = execute_candidate_fn(
        selected_manifest_path,
        runner_kwargs=call_kwargs,
        repo_root=repo_root,
    )
    write_execution_artifacts_fn(plan, study_dir, [result])
    return _result_payload(result, study_dir=study_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exactly one planned tuning candidate as an execution smoke.")
    parser.add_argument("--study-dir", type=Path, required=True, help="Planned study artifact directory.")
    parser.add_argument(
        "--candidate-id",
        default=None,
        help="Candidate id to execute; optional only for one-candidate studies.",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root.")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--processed-manifest", type=Path, default=None, help="Processed dataset manifest path.")
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime/base.yaml"))
    parser.add_argument("--device-config", type=Path, default=Path("configs/runtime/devices/local_u300_24gb.yaml"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--model-seed", type=int, default=1)
    parser.add_argument("--evaluate-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split-cache", choices=["auto", "enable", "disable"], default="auto")
    parser.add_argument("--training-index-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cluster-artifact-cache", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_tuning_candidate_smoke(
        study_dir=args.study_dir,
        candidate_id=args.candidate_id,
        repo_root=args.repo_root.resolve(),
        dry_run=args.dry_run,
        processed_manifest=args.processed_manifest,
        runtime_config=args.runtime_config,
        device_config=args.device_config,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        evaluate_test=args.evaluate_test,
        use_split_cache=_split_cache_value(args.split_cache),
        use_training_index_cache=args.training_index_cache,
        use_cluster_artifact_cache=args.cluster_artifact_cache,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _select_candidate_manifest(*, study_dir: Path, candidate_id: str | None) -> Path:
    candidates_dir = study_dir / "candidates"
    if candidate_id is not None:
        manifest_path = candidates_dir / candidate_id / "candidate_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"candidate manifest does not exist: {manifest_path}")
        return manifest_path

    manifests = sorted(candidates_dir.glob("*/candidate_manifest.json"))
    if len(manifests) == 1:
        return manifests[0]
    if not manifests:
        raise FileNotFoundError(f"no candidate manifests found in {candidates_dir}")
    raise ValueError("--candidate-id is required when a study contains more than one candidate")


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


def _resolve_path(path: Path, *, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _split_cache_value(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "enable"


def _build_split_config(*, train_ratio: float, validation_ratio: float, split_seed: int):
    from recsys_lab.experiments.common import SplitConfig

    return SplitConfig(train_ratio=train_ratio, validation_ratio=validation_ratio, seed=split_seed)


def _result_payload(result: Any, *, study_dir: Path) -> dict[str, Any]:
    return {
        "study_dir": str(study_dir),
        "candidate_id": result.candidate_id,
        "study_id": result.study_id,
        "execution_status": result.execution_status,
        "run_id": result.run_id,
        "run_dir": result.run_dir,
        "run_manifest_path": result.run_manifest_path,
        "error_message": result.error_message,
        "claim_boundary": CLAIM_BOUNDARY,
    }


if __name__ == "__main__":
    raise SystemExit(main())
