from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from recsys_lab.config.loader import dump_yaml_file
from scripts.plan_sota_tuning_study import plan_sota_tuning_study


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _base_model_config() -> dict[str, object]:
    return {
        "metadata": {"status": "unit", "owner": "tests", "purpose": "model_profile"},
        "model": {"name": "cb_svdpp", "family": "matrix_factorization", "scope": "extended"},
        "training": {
            "latent_dim": 64,
            "epochs": 3,
            "learning_rate": 0.0075,
            "lambda_b": 0.01,
            "lambda_p": 0.01,
            "lambda_q": 0.025,
            "lambda_y": 0.01,
            "lambda_pC": 0.01,
            "lambda_qC": 0.01,
            "lambda_yC": 0.01,
            "init_std": 0.1,
            "dtype": "float32",
            "implicit_policy": "ratings_as_implicit",
        },
        "clustering": {
            "alpha": 0.2,
            "n_user_clusters": 80,
            "n_item_clusters": 80,
            "algorithm": "kmeans",
            "kmeans_n_init": 10,
            "induction": {
                "latent_dim": 64,
                "epochs": 20,
                "learning_rate": 0.0075,
                "lambda_b": 0.025,
                "lambda_p": 0.025,
                "lambda_q": 0.025,
                "init_std": 0.1,
                "dtype": "float32",
                "seed": 1,
            },
        },
        "notes": [],
    }


def _search_space_payload(base_config: str) -> dict[str, object]:
    return {
        "search_space_version": "tuning_search_space_v1",
        "study": {
            "name": "sota_planner_unit",
            "dataset": "ml1m",
            "split_family": "benchmark_random_v1",
            "model": "cb_svdpp",
            "seed": 1,
        },
        "base_model_config": base_config,
        "budget": {"max_candidates": 6, "max_parallel": 1, "max_wall_seconds": None},
        "generator": {
            "type": "latin_hypercube",
            "deterministic_order": True,
            "seed": 7,
            "n_candidates": 6,
        },
        "search_space": {
            "alpha": {
                "type": "float",
                "distribution": "uniform",
                "low": 0.1,
                "high": 0.35,
                "target_path": "clustering.alpha",
            },
            "learning_rate": {
                "type": "float",
                "distribution": "loguniform",
                "low": 0.001,
                "high": 0.02,
                "target_path": "training.learning_rate",
            },
        },
        "artifact_reuse": {
            "cluster_artifacts": {
                "reuse_across": [
                    "alpha",
                    "learning_rate",
                    "lambda_q",
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
        "objective": {"primary": {"metric": "validation_rmse"}},
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
            ]
        },
    }


def _write_search_space(tmp_path: Path) -> Path:
    base_config_path = tmp_path / "base_model.yaml"
    search_space_path = tmp_path / "search_space.yaml"
    dump_yaml_file(base_config_path, _base_model_config())
    dump_yaml_file(search_space_path, _search_space_payload(str(base_config_path)))
    return search_space_path


def test_plan_sota_tuning_study_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/plan_sota_tuning_study.py", "--help"],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--stage" in result.stdout
    assert "--promote-from-results" in result.stdout


def test_plan_sota_tuning_stage_writes_stage_outputs(tmp_path: Path) -> None:
    search_space_path = _write_search_space(tmp_path)

    result = plan_sota_tuning_study(
        search_space_path=search_space_path,
        output_dir=tmp_path / "out",
        study_id="sota_unit_study",
        stage_name="stage1_low_fidelity",
        repo_root=_repo_root(),
    )

    study_dir = tmp_path / "out" / "sota_unit_study"
    candidate_configs = list((study_dir / "candidates").glob("*/candidate_config.yaml"))

    assert result["mode"] == "stage_planning"
    assert result["candidate_count"] == 4
    assert (study_dir / "study_manifest.json").exists()
    assert (study_dir / "reports" / "candidate_summary.csv").exists()
    assert (study_dir / "reports" / "artifact_reuse_summary.csv").exists()
    assert len(candidate_configs) == 4


def test_plan_sota_tuning_study_writes_stage1_candidates(tmp_path: Path) -> None:
    test_plan_sota_tuning_stage_writes_stage_outputs(tmp_path)


def test_plan_sota_tuning_study_writes_reuse_summary(tmp_path: Path) -> None:
    search_space_path = _write_search_space(tmp_path)

    plan_sota_tuning_study(
        search_space_path=search_space_path,
        output_dir=tmp_path / "out",
        study_id="sota_unit_study",
        stage_name="stage1_low_fidelity",
        repo_root=_repo_root(),
    )

    assert (tmp_path / "out" / "sota_unit_study" / "reports" / "artifact_reuse_summary.csv").exists()


def test_plan_sota_tuning_study_refuses_execution_flags(tmp_path: Path) -> None:
    search_space_path = _write_search_space(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plan_sota_tuning_study.py",
            "--search-space",
            str(search_space_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--processed-manifest",
            "data/processed/ml1m/example.json",
        ],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr


def test_plan_sota_tuning_stage_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    search_space_path = _write_search_space(tmp_path)
    plan_sota_tuning_study(
        search_space_path=search_space_path,
        output_dir=tmp_path / "out",
        study_id="sota_unit_study",
        repo_root=_repo_root(),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        plan_sota_tuning_study(
            search_space_path=search_space_path,
            output_dir=tmp_path / "out",
            study_id="sota_unit_study",
            repo_root=_repo_root(),
        )


def test_plan_sota_tuning_stage_overwrite_replaces_existing_output(tmp_path: Path) -> None:
    search_space_path = _write_search_space(tmp_path)
    study_dir = tmp_path / "out" / "sota_unit_study"
    study_dir.mkdir(parents=True)
    (study_dir / "stale.txt").write_text("stale", encoding="utf-8")

    plan_sota_tuning_study(
        search_space_path=search_space_path,
        output_dir=tmp_path / "out",
        study_id="sota_unit_study",
        overwrite=True,
        repo_root=_repo_root(),
    )

    assert not (study_dir / "stale.txt").exists()
    assert (study_dir / "study_manifest.json").exists()


def test_plan_sota_tuning_rejects_unknown_stage(tmp_path: Path) -> None:
    search_space_path = _write_search_space(tmp_path)

    with pytest.raises(ValueError, match="unknown fidelity stage"):
        plan_sota_tuning_study(
            search_space_path=search_space_path,
            output_dir=tmp_path / "out",
            stage_name="missing_stage",
            repo_root=_repo_root(),
        )


def test_sota_tuning_template_plans_stage1_without_execution(tmp_path: Path) -> None:
    result = plan_sota_tuning_study(
        search_space_path=(
            _repo_root()
            / "configs"
            / "experiments"
            / "tuning"
            / "templates"
            / "ml1m_cb_svdpp_sota_tuning_v1.yaml"
        ),
        output_dir=tmp_path / "out",
        study_id="sota_template_stage1",
        stage_name="stage1_low_fidelity",
        repo_root=_repo_root(),
    )

    assert result["candidate_count"] == 48
    assert result["artifact_reuse_group_count"] == 1
    assert result["stage"] == "stage1_low_fidelity"
    assert (tmp_path / "out" / "sota_template_stage1" / "reports" / "candidate_summary.csv").exists()


def test_plan_sota_tuning_promotion_writes_plan_and_promoted_configs(tmp_path: Path) -> None:
    search_space_path = _write_search_space(tmp_path)
    stage_result = plan_sota_tuning_study(
        search_space_path=search_space_path,
        output_dir=tmp_path / "out",
        study_id="sota_unit_study",
        stage_name="stage1_low_fidelity",
        repo_root=_repo_root(),
    )
    study_dir = Path(stage_result["study_dir"])
    summary_path = study_dir / "reports" / "candidate_summary.csv"
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8", newline="")))
    result_path = study_dir / "reports" / "stage1_results.csv"
    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "execution_status",
                "validation_rmse",
                "validation_mae",
                "fit_model_seconds",
                "candidate_config_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "candidate_id": rows[0]["candidate_id"],
                "execution_status": "succeeded",
                "validation_rmse": "0.91",
                "validation_mae": "0.72",
                "fit_model_seconds": "12.0",
                "candidate_config_path": rows[0]["candidate_config_path"],
            }
        )
        writer.writerow(
            {
                "candidate_id": rows[1]["candidate_id"],
                "execution_status": "succeeded",
                "validation_rmse": "0.90",
                "validation_mae": "0.71",
                "fit_model_seconds": "13.0",
                "candidate_config_path": rows[1]["candidate_config_path"],
            }
        )

    result = plan_sota_tuning_study(
        search_space_path=search_space_path,
        output_dir=tmp_path / "out",
        study_id="sota_unit_study",
        promote_from_results=result_path,
        from_stage="stage1_low_fidelity",
        to_stage="stage2_mid_fidelity",
        repo_root=_repo_root(),
    )
    promotion_plan_path = study_dir / "promotions" / "stage2_mid_fidelity" / "promotion_plan.json"
    promotion_payload = json.loads(promotion_plan_path.read_text(encoding="utf-8"))
    promoted_configs = list((study_dir / "promotions" / "stage2_mid_fidelity").glob("candidates/*/*.yaml"))

    assert result["mode"] == "promotion_planning"
    assert result["promoted_candidate_count"] == 2
    assert promotion_payload["promoted_candidates"][0]["source_candidate_id"] == rows[1]["candidate_id"]
    assert len(promoted_configs) == 2


def test_plan_sota_tuning_study_writes_promotion_plan_from_results(tmp_path: Path) -> None:
    test_plan_sota_tuning_promotion_writes_plan_and_promoted_configs(tmp_path)


def test_plan_sota_tuning_promotion_requires_to_stage(tmp_path: Path) -> None:
    search_space_path = _write_search_space(tmp_path)
    result_path = tmp_path / "results.csv"
    result_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="--to-stage is required"):
        plan_sota_tuning_study(
            search_space_path=search_space_path,
            output_dir=tmp_path / "out",
            promote_from_results=result_path,
            repo_root=_repo_root(),
        )
