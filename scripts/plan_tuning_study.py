from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from recsys_lab.config.loader import dump_yaml_file, load_yaml_file  # noqa: E402
from recsys_lab.tuning import (  # noqa: E402
    SearchSpaceSpec,
    build_study_plan,
    default_cluster_artifact_reuse_spec,
    write_study_plan,
)


def example_search_space_payload(base_model_config: str = "configs/models/cb_svdpp.yaml") -> dict[str, Any]:
    return {
        "search_space_version": "tuning_search_space_v1",
        "study": {
            "name": "synthetic_cb_svdpp_cache_aware_mvp_v1",
            "dataset": "ml100k",
            "split_family": "benchmark_random_v1",
            "model": "cb_svdpp",
            "seed": 1,
        },
        "base_model_config": base_model_config,
        "budget": {"max_candidates": 4, "max_parallel": 1, "max_wall_seconds": None},
        "generator": {"type": "grid", "deterministic_order": True},
        "search_space": {
            "alpha": {
                "type": "float",
                "values": [0.2, 0.8],
                "target_path": "clustering.alpha",
            },
            "lambda_q": {
                "type": "float",
                "values": [0.01, 0.02],
                "target_path": "training.lambda_q",
            },
        },
        "artifact_reuse": {
            "cluster_artifacts": default_cluster_artifact_reuse_spec().model_dump(mode="json")
        },
        "objective": {
            "primary": {"metric": "validation_rmse", "direction": "minimize", "aggregation": "mean"},
            "secondary": [
                {"metric": "validation_mae", "direction": "minimize", "aggregation": "mean"},
                {"metric": "fit_model_seconds", "direction": "minimize", "aggregation": "mean"},
            ],
            "required_guards": ["cluster_cache_status", "cluster_total_seconds"],
        },
    }


def plan_tuning_study(
    *,
    search_space_path: Path | None,
    output_dir: Path,
    study_id: str | None = None,
    overwrite: bool = False,
    example_synthetic: bool = False,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    if example_synthetic:
        search_space_payload = example_search_space_payload()
    elif search_space_path is not None:
        search_space_payload = load_yaml_file(search_space_path)
    else:
        raise ValueError("--search-space is required unless --example-synthetic is set")

    search_space = SearchSpaceSpec.model_validate(search_space_payload)
    plan = build_study_plan(search_space)
    resolved_study_id = study_id or plan.study_id
    if study_id is not None:
        plan = plan.__class__(
            study_id=study_id,
            search_space=plan.search_space,
            candidates=plan.candidates,
            artifact_reuse_groups=plan.artifact_reuse_groups,
            stage_name=plan.stage_name,
            stage_overrides=plan.stage_overrides,
        )

    study_dir = output_dir / resolved_study_id
    if study_dir.exists():
        if not overwrite:
            raise FileExistsError(f"study output directory already exists: {study_dir}")
        shutil.rmtree(study_dir)

    paths = write_study_plan(plan, study_dir, repo_root=repo_root)
    if example_synthetic:
        dump_yaml_file(study_dir / "search_space_input.yaml", search_space_payload)
    return {
        "study_id": resolved_study_id,
        "study_dir": str(study_dir),
        "candidate_count": len(plan.candidates),
        "artifact_reuse_group_count": len(plan.artifact_reuse_groups),
        "paths": {name: str(path) for name, path in paths.items()},
        "claim_boundary": "Dry-run tuning planner only; no performance or quality claim.",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a cache-aware dry-run tuning study.")
    parser.add_argument("--search-space", type=Path, default=None, help="Search-space YAML contract.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tuning"), help="Tuning artifact root.")
    parser.add_argument("--study-id", default=None, help="Optional study id override.")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing study output directory.")
    parser.add_argument("--example-synthetic", action="store_true", help="Use a built-in tiny example search-space.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run:
        raise SystemExit("Only dry-run planning is supported in Step 19b")
    result = plan_tuning_study(
        search_space_path=args.search_space,
        output_dir=args.output_dir,
        study_id=args.study_id,
        overwrite=args.overwrite,
        example_synthetic=args.example_synthetic,
    )
    print(f"planned study: {result['study_id']}")
    print(f"study_dir: {result['study_dir']}")
    print(f"candidate_count: {result['candidate_count']}")
    print(f"artifact_reuse_group_count: {result['artifact_reuse_group_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
