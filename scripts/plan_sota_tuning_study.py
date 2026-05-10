from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from recsys_lab.config.loader import load_yaml_file  # noqa: E402
from recsys_lab.tuning import (  # noqa: E402
    FidelityStageSpec,
    SearchSpaceSpec,
    StudyPlan,
    build_promotion_plan,
    build_study_plan,
    materialize_promoted_candidates,
    materialize_stage_candidates,
)

CLAIM_BOUNDARY = "Dry-run SOTA tuning planner only; no execution, performance, or quality claim."


def plan_sota_tuning_study(
    *,
    search_space_path: Path,
    output_dir: Path,
    study_id: str | None = None,
    stage_name: str | None = None,
    overwrite: bool = False,
    promote_from_results: Path | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    search_space = SearchSpaceSpec.model_validate(load_yaml_file(_resolve_path(search_space_path, repo_root)))
    if promote_from_results is not None:
        return _plan_promotion(
            search_space=search_space,
            output_dir=output_dir,
            study_id=study_id,
            promote_from_results=promote_from_results,
            from_stage=from_stage,
            to_stage=to_stage,
            overwrite=overwrite,
            repo_root=repo_root,
        )
    return _plan_stage(
        search_space=search_space,
        output_dir=output_dir,
        study_id=study_id,
        stage_name=stage_name,
        overwrite=overwrite,
        repo_root=repo_root,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a staged SOTA tuning study without executing candidates.")
    parser.add_argument("--search-space", type=Path, required=True, help="Search-space YAML contract.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tuning"), help="Tuning artifact root.")
    parser.add_argument("--study-id", default=None, help="Optional study id override.")
    parser.add_argument("--stage", default=None, help="Fidelity stage to materialize. Defaults to first stage.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output for this planning action.")
    parser.add_argument(
        "--promote-from-results",
        type=Path,
        default=None,
        help="CSV/JSON results from a completed stage. Enables promotion planning mode.",
    )
    parser.add_argument("--from-stage", default=None, help="Source stage label for promotion-plan metadata.")
    parser.add_argument("--to-stage", default=None, help="Target stage to promote into.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = plan_sota_tuning_study(
        search_space_path=args.search_space,
        output_dir=args.output_dir,
        study_id=args.study_id,
        stage_name=args.stage,
        overwrite=args.overwrite,
        promote_from_results=args.promote_from_results,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        repo_root=args.repo_root.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _plan_stage(
    *,
    search_space: SearchSpaceSpec,
    output_dir: Path,
    study_id: str | None,
    stage_name: str | None,
    overwrite: bool,
    repo_root: Path,
) -> dict[str, Any]:
    stage = _resolve_stage(search_space, stage_name)
    plan = build_study_plan(
        search_space,
        stage_name=None if stage is None else stage.name,
        max_candidates=None if stage is None else stage.max_candidates,
    )
    plan = _with_study_id(plan, study_id)
    study_dir = output_dir / plan.study_id
    if study_dir.exists():
        if not overwrite:
            raise FileExistsError(f"study output directory already exists: {study_dir}")
        shutil.rmtree(study_dir)

    paths = materialize_stage_candidates(plan, study_dir, stage=stage, repo_root=repo_root)
    return {
        "mode": "stage_planning",
        "study_id": plan.study_id,
        "study_dir": str(study_dir),
        "stage": None if stage is None else stage.name,
        "candidate_count": len(plan.candidates),
        "artifact_reuse_group_count": len(plan.artifact_reuse_groups),
        "paths": {name: str(path) for name, path in paths.items()},
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _plan_promotion(
    *,
    search_space: SearchSpaceSpec,
    output_dir: Path,
    study_id: str | None,
    promote_from_results: Path,
    from_stage: str | None,
    to_stage: str | None,
    overwrite: bool,
    repo_root: Path,
) -> dict[str, Any]:
    if to_stage is None:
        raise ValueError("--to-stage is required when --promote-from-results is set")
    next_stage = _resolve_stage(search_space, to_stage)
    if next_stage is None:
        raise ValueError("promotion planning requires a scheduled target stage")
    plan = _with_study_id(build_study_plan(search_space), study_id)
    study_dir = output_dir / plan.study_id
    promotion_dir = study_dir / "promotions" / next_stage.name
    if promotion_dir.exists():
        if not overwrite:
            raise FileExistsError(f"promotion output directory already exists: {promotion_dir}")
        shutil.rmtree(promotion_dir)

    promotion_plan = build_promotion_plan(_resolve_path(promote_from_results, repo_root), next_stage)
    paths = materialize_promoted_candidates(promotion_plan, study_dir)
    return {
        "mode": "promotion_planning",
        "study_id": plan.study_id,
        "study_dir": str(study_dir),
        "from_stage": from_stage,
        "to_stage": next_stage.name,
        "promoted_candidate_count": len(promotion_plan.promoted_candidates),
        "paths": {name: str(path) for name, path in paths.items()},
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _resolve_stage(search_space: SearchSpaceSpec, stage_name: str | None) -> FidelityStageSpec | None:
    if search_space.schedule is None:
        if stage_name is not None:
            raise ValueError("--stage requires a search space schedule")
        return None
    if stage_name is None:
        return search_space.schedule.stages[0]
    for stage in search_space.schedule.stages:
        if stage.name == stage_name:
            return stage
    raise ValueError(f"unknown fidelity stage: {stage_name}")


def _with_study_id(plan: StudyPlan, study_id: str | None) -> StudyPlan:
    if study_id is None or study_id == plan.study_id:
        return plan
    return StudyPlan(
        study_id=study_id,
        search_space=plan.search_space,
        candidates=plan.candidates,
        artifact_reuse_groups=plan.artifact_reuse_groups,
    )


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


if __name__ == "__main__":
    raise SystemExit(main())
