from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.profile_cluster_artifacts import PROFILE_VERSION, profile_rows, stage_ranking_rows, write_reports


def _payload() -> dict[str, object]:
    return {
        "profile_version": PROFILE_VERSION,
        "claim_boundary": "Diagnostic cluster artifact profiling only; no performance claim.",
        "inputs": {"model": "cb_svdpp"},
        "profiles": [
            {
                "repeat_index": 1,
                "cluster_artifacts": {
                    "dataset_short_name": "synthetic",
                    "split_family": "benchmark_random_v1",
                    "split_id": "synthetic_split",
                    "model": "cb_svdpp",
                    "n_users": 4,
                    "n_items": 5,
                    "train_rows": 16,
                    "n_user_clusters": 2,
                    "n_item_clusters": 2,
                    "algorithm": "kmeans",
                    "kmeans_n_init": 2,
                    "induction_seed": 1,
                    "induction_latent_dim": 3,
                    "induction_epochs": 1,
                    "induction_dtype": "float32",
                    "cluster_cache_status": "miss",
                    "cluster_total_seconds": 10.0,
                    "cluster_cache_read_seconds": 0.1,
                    "cluster_cache_write_seconds": 3.0,
                    "induction_fit_seconds": 2.0,
                    "induction_predict_seconds": 1.0,
                    "induction_train_rmse_seconds": 0.2,
                    "user_kmeans_seconds": 2.5,
                    "item_kmeans_seconds": 1.1,
                    "r_star_seconds": 0.05,
                    "cluster_artifact_validation_seconds": 0.01,
                },
                "user_cluster_history": {
                    "dataset_short_name": "synthetic",
                    "split_family": "benchmark_random_v1",
                    "split_id": "synthetic_split",
                    "model": "cb_svdpp",
                    "n_users": 4,
                    "train_rows": 16,
                    "n_item_clusters": 2,
                    "user_cluster_history_cache_status": "miss",
                    "user_cluster_history_total_seconds": 2.0,
                    "user_cluster_history_cache_read_seconds": 0.1,
                    "user_cluster_history_cache_write_seconds": 1.2,
                    "user_cluster_history_build_seconds": 0.3,
                    "user_cluster_history_validation_seconds": 0.01,
                },
            },
            {
                "repeat_index": 2,
                "cluster_artifacts": {
                    "dataset_short_name": "synthetic",
                    "split_family": "benchmark_random_v1",
                    "split_id": "synthetic_split",
                    "model": "cb_svdpp",
                    "n_users": 4,
                    "n_items": 5,
                    "train_rows": 16,
                    "n_user_clusters": 2,
                    "n_item_clusters": 2,
                    "algorithm": "kmeans",
                    "kmeans_n_init": 2,
                    "induction_seed": 1,
                    "induction_latent_dim": 3,
                    "induction_epochs": 1,
                    "induction_dtype": "float32",
                    "cluster_cache_status": "hit",
                    "cluster_total_seconds": 1.0,
                    "cluster_cache_read_seconds": 0.4,
                    "cluster_cache_write_seconds": 0.0,
                    "induction_fit_seconds": 0.0,
                    "induction_predict_seconds": 0.0,
                    "induction_train_rmse_seconds": 0.0,
                    "user_kmeans_seconds": 0.0,
                    "item_kmeans_seconds": 0.0,
                    "r_star_seconds": 0.0,
                    "cluster_artifact_validation_seconds": 0.01,
                },
                "user_cluster_history": {
                    "dataset_short_name": "synthetic",
                    "split_family": "benchmark_random_v1",
                    "split_id": "synthetic_split",
                    "model": "cb_svdpp",
                    "n_users": 4,
                    "train_rows": 16,
                    "n_item_clusters": 2,
                    "user_cluster_history_cache_status": "hit",
                    "user_cluster_history_total_seconds": 0.5,
                    "user_cluster_history_cache_read_seconds": 0.02,
                    "user_cluster_history_cache_write_seconds": 0.0,
                    "user_cluster_history_build_seconds": 0.0,
                    "user_cluster_history_validation_seconds": 0.01,
                },
            },
        ],
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_profile_cluster_artifacts_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/profile_cluster_artifacts.py", "--help"],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--processed-manifest" in result.stdout
    assert "--output-stem" in result.stdout


def test_profile_cluster_artifacts_rejects_invalid_config() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/profile_cluster_artifacts.py",
            "--processed-manifest",
            "missing_manifest.json",
            "--model-config",
            "missing_config.yaml",
            "--runtime-config",
            "missing_runtime.yaml",
            "--model",
            "not_a_cb_model",
        ],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_profile_rows_flattens_cluster_and_history_profiles() -> None:
    rows = profile_rows(_payload())

    assert len(rows) == 4
    assert rows[0]["profile_version"] == PROFILE_VERSION
    assert rows[0]["repeat_index"] == 1
    assert rows[0]["profile_kind"] == "cluster_artifacts"
    assert rows[0]["cluster_cache_status"] == "miss"
    assert rows[1]["profile_kind"] == "user_cluster_history"
    assert rows[1]["user_cluster_history_cache_status"] == "miss"
    assert rows[2]["cluster_cache_status"] == "hit"
    assert rows[3]["user_cluster_history_cache_status"] == "hit"


def test_write_reports_writes_json_and_csv(tmp_path: Path) -> None:
    paths = write_reports(_payload(), output_dir=tmp_path)

    assert paths["json"] == tmp_path / "cluster_artifact_profile_v1.json"
    assert paths["csv"] == tmp_path / "cluster_artifact_profile_v1.csv"
    assert paths["summary_csv"] == tmp_path / "cluster_artifact_profile_v1_summary.csv"
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["profile_version"] == PROFILE_VERSION

    with paths["csv"].open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    assert len(rows) == 4
    assert rows[0]["profile_kind"] == "cluster_artifacts"
    assert rows[0]["cluster_cache_status"] == "miss"
    assert rows[1]["profile_kind"] == "user_cluster_history"


def test_write_reports_accepts_custom_output_stem(tmp_path: Path) -> None:
    paths = write_reports(_payload(), output_dir=tmp_path, output_stem="cluster_artifact_profile_ml1m")

    assert paths["json"] == tmp_path / "cluster_artifact_profile_ml1m.json"
    assert paths["csv"] == tmp_path / "cluster_artifact_profile_ml1m.csv"
    assert paths["summary_csv"] == tmp_path / "cluster_artifact_profile_ml1m_summary.csv"


def test_profile_cluster_artifacts_writes_summary_for_synthetic_smoke(tmp_path: Path) -> None:
    paths = write_reports(_payload(), output_dir=tmp_path, output_stem="synthetic_smoke")

    with paths["summary_csv"].open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    assert len(rows) == 2
    assert rows[0]["cache_path"] == "cold_miss_build+cold_user_cluster_history_miss_build"
    assert rows[0]["cluster_cache_status"] == "miss"
    assert rows[0]["user_cluster_history_cache_status"] == "miss"
    assert rows[1]["cache_path"] == "warm_hit_load+warm_user_cluster_history_hit_load"
    assert rows[1]["cluster_cache_status"] == "hit"
    assert rows[1]["user_cluster_history_cache_status"] == "hit"


def test_stage_ranking_computation_orders_descending() -> None:
    rows = stage_ranking_rows(_payload())
    cold_cluster_rows = [row for row in rows if row["cache_path"] == "cold_miss_build"]
    warm_cluster_rows = [row for row in rows if row["cache_path"] == "warm_hit_load"]

    assert cold_cluster_rows[0]["stage"] == "cluster_cache_write"
    assert cold_cluster_rows[0]["rank"] == 1
    assert cold_cluster_rows[1]["stage"] == "user_kmeans"
    assert warm_cluster_rows[0]["stage"] == "cluster_cache_read"
    assert warm_cluster_rows[0]["rank"] == 1


def test_write_reports_rejects_output_stem_with_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_stem"):
        write_reports(_payload(), output_dir=tmp_path, output_stem="nested/profile")


def test_write_reports_rejects_output_stem_with_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_stem"):
        write_reports(_payload(), output_dir=tmp_path, output_stem="profile.csv")
