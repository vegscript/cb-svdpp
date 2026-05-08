from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from recsys_lab.experiments.common import utc_timestamp
from recsys_lab.experiments.unified_runner import run_unified_experiment
from recsys_lab.tuning.manifests import CandidateManifest

ExecutionStatus = Literal["not_executed", "running", "succeeded", "failed", "skipped"]
RunnerFn = Callable[..., dict[str, Any]]


@dataclass(frozen=True, slots=True)
class CandidateExecutionResult:
    candidate_id: str
    study_id: str
    execution_status: ExecutionStatus
    run_id: str | None = None
    run_dir: str | None = None
    metrics_path: str | None = None
    performance_profile_path: str | None = None
    kernel_profile_path: str | None = None
    run_manifest_path: str | None = None
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    error_message: str | None = None


def load_candidate_manifest(path: Path) -> CandidateManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CandidateManifest.model_validate(payload)


def resolve_candidate_config_path(
    candidate_manifest: CandidateManifest,
    *,
    candidate_manifest_path: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    configured_path = candidate_manifest.candidate_config_path
    if configured_path:
        path = Path(configured_path)
        if path.is_absolute():
            return path.resolve()
        if repo_root is not None:
            return (repo_root / path).resolve()
        return path.resolve()

    if candidate_manifest_path is None:
        raise ValueError("candidate_config_path is missing and no candidate manifest path was provided")
    return (candidate_manifest_path.parent / "candidate_config.yaml").resolve()


def execute_candidate(
    candidate_manifest_path: Path,
    *,
    runner_kwargs: Mapping[str, Any],
    runner: RunnerFn = run_unified_experiment,
    repo_root: Path | None = None,
    raise_on_error: bool = False,
) -> CandidateExecutionResult:
    candidate_manifest_path = candidate_manifest_path.resolve()
    candidate_manifest = load_candidate_manifest(candidate_manifest_path)
    started_at = utc_timestamp()

    try:
        if "model_config_path" in runner_kwargs:
            raise ValueError("model_config_path is derived from the candidate manifest")

        candidate_config_path = resolve_candidate_config_path(
            candidate_manifest,
            candidate_manifest_path=candidate_manifest_path,
            repo_root=repo_root,
        )
        if not candidate_config_path.exists():
            raise FileNotFoundError(f"candidate config does not exist: {candidate_config_path}")

        call_kwargs = dict(runner_kwargs)
        call_kwargs["model_config_path"] = candidate_config_path
        call_kwargs.setdefault("model_name", candidate_manifest.study.model)
        call_kwargs.setdefault("split_family", candidate_manifest.study.split_family)
        if repo_root is not None:
            call_kwargs.setdefault("repo_root", repo_root)

        payload = runner(**call_kwargs)
        run_dir = Path(str(payload["run_dir"])).resolve()
        run_manifest_path = Path(str(payload["run_manifest"])).resolve()
        finished_at = utc_timestamp()

        return CandidateExecutionResult(
            candidate_id=candidate_manifest.candidate_id,
            study_id=candidate_manifest.study_id,
            execution_status="succeeded",
            run_id=str(payload.get("run_id")) if payload.get("run_id") is not None else None,
            run_dir=str(run_dir),
            metrics_path=str(run_dir / "metrics.json"),
            performance_profile_path=str(run_dir / "performance_profile.json"),
            kernel_profile_path=str(run_dir / "kernel_profile.json"),
            run_manifest_path=str(run_manifest_path),
            started_at_utc=started_at,
            finished_at_utc=finished_at,
        )
    except Exception as exc:
        finished_at = utc_timestamp()
        if raise_on_error:
            raise
        return CandidateExecutionResult(
            candidate_id=candidate_manifest.candidate_id,
            study_id=candidate_manifest.study_id,
            execution_status="failed",
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            error_message=f"{type(exc).__name__}: {exc}",
        )


def skipped_candidate_result(
    candidate_manifest_path: Path,
    *,
    reason: str,
) -> CandidateExecutionResult:
    candidate_manifest = load_candidate_manifest(candidate_manifest_path)
    timestamp = utc_timestamp()
    return CandidateExecutionResult(
        candidate_id=candidate_manifest.candidate_id,
        study_id=candidate_manifest.study_id,
        execution_status="skipped",
        started_at_utc=timestamp,
        finished_at_utc=timestamp,
        error_message=reason,
    )
