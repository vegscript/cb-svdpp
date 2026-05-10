from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from recsys_lab.config.loader import dump_yaml_file
from recsys_lab.tuning.execution import CandidateExecutionResult, load_candidate_manifest
from scripts.run_sota_tuning_pilot import (
    PILOT_SUMMARY_FIELDS,
    STAGE_RESULT_FIELDS,
    run_sota_tuning_pilot,
)


def _base_config() -> dict[str, Any]:
    return {
        "metadata": {"status": "unit", "owner": "tests", "purpose": "model_profile"},
        "model": {"name": "cb_svdpp", "family": "clustering_based_factorization", "scope": "paper_inspired"},
        "training": {
            "latent_dim": 64,
            "epochs": 20,
            "learning_rate": 0.01,
            "lambda_b": 0.025,
            "lambda_p": 0.025,
            "lambda_q": 0.025,
            "lambda_y": 0.025,
            "lambda_pC": 0.025,
            "lambda_qC": 0.025,
            "lambda_yC": 0.025,
            "init_std": 0.1,
            "dtype": "float32",
            "implicit_policy": "ratings_as_implicit",
        },
        "clustering": {
            "n_user_clusters": 80,
            "n_item_clusters": 80,
            "alpha": 0.2,
            "algorithm": "kmeans",
            "kmeans_n_init": 10,
            "induction": {
                "latent_dim": 64,
                "epochs": 20,
                "learning_rate": 0.0075,
                "lambda_b": 0.025,
                "lambda_p": 0.025,
                "lambda_q": 0.025,
                "seed": 1,
                "init_std": 0.1,
                "dtype": "float32",
                "training_backend": "auto",
            },
        },
        "notes": [],
    }


def _search_space_payload(base_config_path: Path) -> dict[str, Any]:
    return {
        "search_space_version": "tuning_search_space_v1",
        "study": {
            "name": "ml1m_sota_pilot_unit",
            "dataset": "ml1m",
            "split_family": "benchmark_random_v1",
            "model": "cb_svdpp",
            "seed": 1,
        },
        "base_model_config": str(base_config_path),
        "budget": {"max_candidates": 4, "max_parallel": 1, "max_wall_seconds": None},
        "generator": {"type": "latin_hypercube", "deterministic_order": True, "seed": 11, "n_candidates": 4},
        "schedule": {
            "stages": [
                {
                    "name": "stage1_low_fidelity",
                    "max_candidates": 4,
                    "promote_top_k": 2,
                    "overrides": {"training.epochs": 3},
                },
                {
                    "name": "stage2_mid_fidelity",
                    "max_candidates": 2,
                    "promote_top_k": 1,
                    "overrides": {"training.epochs": 10},
                },
                {
                    "name": "stage3_full_fidelity",
                    "max_candidates": 1,
                    "promote_top_k": None,
                    "overrides": {"training.epochs": 20},
                },
            ]
        },
        "search_space": {
            "alpha": {
                "type": "float",
                "distribution": "uniform",
                "low": 0.08,
                "high": 0.32,
                "target_path": "clustering.alpha",
            },
            "learning_rate": {
                "type": "float",
                "distribution": "loguniform",
                "low": 0.004,
                "high": 0.02,
                "target_path": "training.learning_rate",
            },
        },
        "artifact_reuse": {
            "cluster_artifacts": {
                "reuse_across": [
                    "alpha",
                    "learning_rate",
                    "lambda_p",
                    "lambda_q",
                    "lambda_y",
                    "lambda_pC",
                    "lambda_qC",
                    "lambda_yC",
                    "epochs",
                ],
                "invalidate_on": [
                    "n_user_clusters",
                    "n_item_clusters",
                    "induction_config",
                    "kmeans_n_init",
                    "clustering_algorithm",
                    "dataset",
                    "split",
                    "train_fingerprint",
                ],
            }
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


def _write_search_space(tmp_path: Path, *, dataset: str = "ml1m") -> Path:
    base_config_path = tmp_path / "base.yaml"
    search_space_path = tmp_path / "search_space.yaml"
    dump_yaml_file(base_config_path, _base_config())
    payload = _search_space_payload(base_config_path)
    payload["study"]["dataset"] = dataset
    dump_yaml_file(search_space_path, payload)
    return search_space_path


def _fake_execute_success(tmp_path: Path):
    counter = {"value": 0}

    def _execute(
        candidate_manifest_path: Path,
        *,
        runner_kwargs: dict[str, Any],
        repo_root: Path,
    ) -> CandidateExecutionResult:
        counter["value"] += 1
        manifest = load_candidate_manifest(candidate_manifest_path)
        run_dir = tmp_path / "runs" / f"run_{counter['value']:02d}"
        run_dir.mkdir(parents=True)
        metrics_path = run_dir / "metrics.json"
        performance_path = run_dir / "performance_profile.json"
        kernel_path = run_dir / "kernel_profile.json"
        run_manifest_path = run_dir / "run_manifest.json"
        cache_status = "miss" if counter["value"] == 1 else "hit"
        metrics_path.write_text(
            json.dumps(
                {
                    "metrics": {
                        "validation_rmse": 0.9 + counter["value"] * 0.001,
                        "validation_mae": 0.7 + counter["value"] * 0.001,
                    },
                    "caches": {
                        "cluster_artifacts": {"status": cache_status},
                        "user_cluster_history": {"status": cache_status},
                    },
                    "timing": {"cluster_induction_wall_clock_seconds": 0.3},
                }
            ),
            encoding="utf-8",
        )
        performance_path.write_text(
            json.dumps(
                {
                    "total_profiled_wall_clock_seconds": 2.0,
                    "stages": [
                        {"name": "fit_model", "wall_clock_seconds": 1.0},
                        {"name": "build_cluster_artifacts", "wall_clock_seconds": 0.3},
                    ],
                }
            ),
            encoding="utf-8",
        )
        kernel_path.write_text("{}", encoding="utf-8")
        run_manifest_path.write_text("{}", encoding="utf-8")
        return CandidateExecutionResult(
            candidate_id=manifest.candidate_id,
            study_id=manifest.study_id,
            execution_status="succeeded",
            run_id=run_dir.name,
            run_dir=str(run_dir),
            metrics_path=str(metrics_path),
            performance_profile_path=str(performance_path),
            kernel_profile_path=str(kernel_path),
            run_manifest_path=str(run_manifest_path),
        )

    return _execute


def _fake_execute_fail_after(success_count: int, tmp_path: Path):
    success_execute = _fake_execute_success(tmp_path)
    counter = {"value": 0}

    def _execute(
        candidate_manifest_path: Path,
        *,
        runner_kwargs: dict[str, Any],
        repo_root: Path,
    ) -> CandidateExecutionResult:
        counter["value"] += 1
        if counter["value"] <= success_count:
            return success_execute(candidate_manifest_path, runner_kwargs=runner_kwargs, repo_root=repo_root)
        manifest = load_candidate_manifest(candidate_manifest_path)
        return CandidateExecutionResult(
            candidate_id=manifest.candidate_id,
            study_id=manifest.study_id,
            execution_status="failed",
            error_message="RuntimeError: synthetic failure",
        )

    return _execute


def _fake_execute_missing_validation_rmse(tmp_path: Path):
    counter = {"value": 0}

    def _execute(
        candidate_manifest_path: Path,
        *,
        runner_kwargs: dict[str, Any],
        repo_root: Path,
    ) -> CandidateExecutionResult:
        counter["value"] += 1
        manifest = load_candidate_manifest(candidate_manifest_path)
        run_dir = tmp_path / "runs_missing_metric" / f"run_{counter['value']:02d}"
        run_dir.mkdir(parents=True)
        metrics_path = run_dir / "metrics.json"
        performance_path = run_dir / "performance_profile.json"
        kernel_path = run_dir / "kernel_profile.json"
        run_manifest_path = run_dir / "run_manifest.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "metrics": {"validation_mae": 0.7 + counter["value"] * 0.001},
                    "caches": {
                        "cluster_artifacts": {"status": "hit"},
                        "user_cluster_history": {"status": "hit"},
                    },
                    "timing": {"cluster_induction_wall_clock_seconds": 0.3},
                }
            ),
            encoding="utf-8",
        )
        performance_path.write_text(
            json.dumps(
                {
                    "total_profiled_wall_clock_seconds": 2.0,
                    "stages": [
                        {"name": "fit_model", "wall_clock_seconds": 1.0},
                        {"name": "build_cluster_artifacts", "wall_clock_seconds": 0.3},
                    ],
                }
            ),
            encoding="utf-8",
        )
        kernel_path.write_text("{}", encoding="utf-8")
        run_manifest_path.write_text("{}", encoding="utf-8")
        return CandidateExecutionResult(
            candidate_id=manifest.candidate_id,
            study_id=manifest.study_id,
            execution_status="succeeded",
            run_id=run_dir.name,
            run_dir=str(run_dir),
            metrics_path=str(metrics_path),
            performance_profile_path=str(performance_path),
            kernel_profile_path=str(kernel_path),
            run_manifest_path=str(run_manifest_path),
        )

    return _execute


def test_pilot_script_rejects_non_ml1m_manifest(tmp_path: Path) -> None:
    search_space = _write_search_space(tmp_path)

    with pytest.raises(ValueError, match="ML1M processed manifest"):
        run_sota_tuning_pilot(
            search_space=search_space,
            output_dir=tmp_path / "out",
            study_id="pilot_unit",
            processed_manifest=tmp_path / "ml100k_manifest.json",
            runtime_config=tmp_path / "runtime.yaml",
            device_config=tmp_path / "device.yaml",
            overwrite=True,
            repo_root=Path.cwd(),
        )


def test_pilot_script_rejects_stage1_above_16(tmp_path: Path) -> None:
    search_space = _write_search_space(tmp_path)
    payload = _search_space_payload(tmp_path / "base.yaml")
    payload["schedule"]["stages"][0]["max_candidates"] = 17
    dump_yaml_file(search_space, payload)

    with pytest.raises(ValueError, match="stage1 max_candidates"):
        run_sota_tuning_pilot(
            search_space=search_space,
            output_dir=tmp_path / "out",
            study_id="pilot_unit",
            processed_manifest=tmp_path / "ml1m_manifest.json",
            runtime_config=tmp_path / "runtime.yaml",
            device_config=tmp_path / "device.yaml",
            overwrite=True,
            repo_root=Path.cwd(),
        )


def test_pilot_script_rejects_stage2_above_4(tmp_path: Path) -> None:
    search_space = _write_search_space(tmp_path)
    payload = _search_space_payload(tmp_path / "base.yaml")
    payload["schedule"]["stages"][0]["max_candidates"] = 5
    payload["schedule"]["stages"][0]["promote_top_k"] = 5
    payload["schedule"]["stages"][1]["max_candidates"] = 5
    dump_yaml_file(search_space, payload)

    with pytest.raises(ValueError, match="stage2 max_candidates"):
        run_sota_tuning_pilot(
            search_space=search_space,
            output_dir=tmp_path / "out",
            study_id="pilot_unit",
            processed_manifest=tmp_path / "ml1m_manifest.json",
            runtime_config=tmp_path / "runtime.yaml",
            device_config=tmp_path / "device.yaml",
            overwrite=True,
            repo_root=Path.cwd(),
        )


def test_pilot_script_rejects_stage3_above_1(tmp_path: Path) -> None:
    search_space = _write_search_space(tmp_path)
    payload = _search_space_payload(tmp_path / "base.yaml")
    payload["schedule"]["stages"][1]["max_candidates"] = 2
    payload["schedule"]["stages"][1]["promote_top_k"] = 2
    payload["schedule"]["stages"][2]["max_candidates"] = 2
    dump_yaml_file(search_space, payload)

    with pytest.raises(ValueError, match="stage3 max_candidates"):
        run_sota_tuning_pilot(
            search_space=search_space,
            output_dir=tmp_path / "out",
            study_id="pilot_unit",
            processed_manifest=tmp_path / "ml1m_manifest.json",
            runtime_config=tmp_path / "runtime.yaml",
            device_config=tmp_path / "device.yaml",
            overwrite=True,
            repo_root=Path.cwd(),
        )


def test_pilot_script_rejects_parallel_execution(tmp_path: Path) -> None:
    search_space = _write_search_space(tmp_path)
    payload = _search_space_payload(tmp_path / "base.yaml")
    payload["budget"]["max_parallel"] = 2
    dump_yaml_file(search_space, payload)

    with pytest.raises(ValueError, match="max_parallel=1"):
        run_sota_tuning_pilot(
            search_space=search_space,
            output_dir=tmp_path / "out",
            study_id="pilot_unit",
            processed_manifest=tmp_path / "ml1m_manifest.json",
            runtime_config=tmp_path / "runtime.yaml",
            device_config=tmp_path / "device.yaml",
            overwrite=True,
            repo_root=Path.cwd(),
        )


def test_stage_results_csv_contains_required_promotion_fields(tmp_path: Path) -> None:
    search_space = _write_search_space(tmp_path)
    runtime = tmp_path / "runtime.yaml"
    device = tmp_path / "device.yaml"
    dump_yaml_file(runtime, {"runtime": {}})
    dump_yaml_file(device, {"device": {}})

    run_sota_tuning_pilot(
        search_space=search_space,
        output_dir=tmp_path / "out",
        study_id="pilot_unit",
        processed_manifest=tmp_path / "ml1m_manifest.json",
        runtime_config=runtime,
        device_config=device,
        overwrite=True,
        repo_root=Path.cwd(),
        execute_candidate_fn=_fake_execute_success(tmp_path),
    )

    with (tmp_path / "out" / "pilot_unit" / "reports" / "stage1_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        assert set(STAGE_RESULT_FIELDS).issubset(reader.fieldnames or [])
        assert {
            "candidate_id",
            "execution_status",
            "validation_rmse",
            "validation_mae",
            "fit_model_seconds",
            "candidate_config_path",
            "study_id",
            "stage_name",
        }.issubset(reader.fieldnames or [])


def test_stage_results_rejects_test_metrics() -> None:
    assert "test_rmse" not in STAGE_RESULT_FIELDS
    assert "test_mae" not in STAGE_RESULT_FIELDS
    assert "test_rmse" not in PILOT_SUMMARY_FIELDS
    assert "test_mae" not in PILOT_SUMMARY_FIELDS


def test_pilot_decision_marks_execution_unstable_when_stage1_successes_too_low(tmp_path: Path) -> None:
    search_space = _write_search_space(tmp_path)
    runtime = tmp_path / "runtime.yaml"
    device = tmp_path / "device.yaml"
    dump_yaml_file(runtime, {"runtime": {}})
    dump_yaml_file(device, {"device": {}})

    result = run_sota_tuning_pilot(
        search_space=search_space,
        output_dir=tmp_path / "out",
        study_id="pilot_unit",
        processed_manifest=tmp_path / "ml1m_manifest.json",
        runtime_config=runtime,
        device_config=device,
        overwrite=True,
        repo_root=Path.cwd(),
        execute_candidate_fn=_fake_execute_fail_after(2, tmp_path),
    )

    assert result["decision"] == "SOTA_PILOT_EXECUTION_UNSTABLE"


def test_pilot_decision_marks_promotion_contract_broken_on_invalid_promotion(tmp_path: Path) -> None:
    search_space = _write_search_space(tmp_path)
    runtime = tmp_path / "runtime.yaml"
    device = tmp_path / "device.yaml"
    dump_yaml_file(runtime, {"runtime": {}})
    dump_yaml_file(device, {"device": {}})

    result = run_sota_tuning_pilot(
        search_space=search_space,
        output_dir=tmp_path / "out",
        study_id="pilot_unit",
        processed_manifest=tmp_path / "ml1m_manifest.json",
        runtime_config=runtime,
        device_config=device,
        overwrite=True,
        repo_root=Path.cwd(),
        execute_candidate_fn=_fake_execute_missing_validation_rmse(tmp_path),
    )

    assert result["decision"] == "SOTA_PILOT_PROMOTION_CONTRACT_BROKEN"


def test_final_candidate_config_not_written_to_configs_selected(tmp_path: Path) -> None:
    search_space = _write_search_space(tmp_path)
    runtime = tmp_path / "runtime.yaml"
    device = tmp_path / "device.yaml"
    dump_yaml_file(runtime, {"runtime": {}})
    dump_yaml_file(device, {"device": {}})

    run_sota_tuning_pilot(
        search_space=search_space,
        output_dir=tmp_path / "out",
        study_id="pilot_unit",
        processed_manifest=tmp_path / "ml1m_manifest.json",
        runtime_config=runtime,
        device_config=device,
        overwrite=True,
        repo_root=Path.cwd(),
        execute_candidate_fn=_fake_execute_success(tmp_path),
    )

    assert not (tmp_path / "configs" / "models" / "selected").exists()
