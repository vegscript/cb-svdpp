from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.plan_tuning_study import example_search_space_payload, plan_tuning_study


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_plan_tuning_study_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/plan_tuning_study.py", "--help"],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--search-space" in result.stdout
    assert "--example-synthetic" in result.stdout


def test_plan_script_help() -> None:
    test_plan_tuning_study_help()


def test_plan_tuning_study_writes_synthetic_example(tmp_path: Path) -> None:
    result = plan_tuning_study(
        search_space_path=None,
        output_dir=tmp_path,
        study_id="synthetic_study",
        example_synthetic=True,
        repo_root=_repo_root(),
    )

    study_dir = tmp_path / "synthetic_study"

    assert result["candidate_count"] == 4
    assert (study_dir / "study_manifest.json").exists()
    assert (study_dir / "search_space.yaml").exists()
    assert (study_dir / "reports" / "candidate_summary.csv").exists()
    assert (study_dir / "reports" / "artifact_reuse_summary.csv").exists()
    assert len(list((study_dir / "candidates").glob("*/candidate_config.yaml"))) == 4
    assert json.loads((study_dir / "study_manifest.json").read_text(encoding="utf-8"))["study_id"] == (
        "synthetic_study"
    )


def test_plan_writes_study_manifest(tmp_path: Path) -> None:
    plan_tuning_study(
        search_space_path=None,
        output_dir=tmp_path,
        study_id="synthetic_study",
        example_synthetic=True,
        repo_root=_repo_root(),
    )

    assert (tmp_path / "synthetic_study" / "study_manifest.json").exists()


def test_plan_writes_candidate_manifests(tmp_path: Path) -> None:
    plan_tuning_study(
        search_space_path=None,
        output_dir=tmp_path,
        study_id="synthetic_study",
        example_synthetic=True,
        repo_root=_repo_root(),
    )

    manifests = list((tmp_path / "synthetic_study" / "candidates").glob("*/candidate_manifest.json"))
    assert len(manifests) == 4


def test_plan_writes_candidate_summary_csv(tmp_path: Path) -> None:
    plan_tuning_study(
        search_space_path=None,
        output_dir=tmp_path,
        study_id="synthetic_study",
        example_synthetic=True,
        repo_root=_repo_root(),
    )

    assert (tmp_path / "synthetic_study" / "reports" / "candidate_summary.csv").exists()


def test_plan_writes_artifact_reuse_summary_csv(tmp_path: Path) -> None:
    plan_tuning_study(
        search_space_path=None,
        output_dir=tmp_path,
        study_id="synthetic_study",
        example_synthetic=True,
        repo_root=_repo_root(),
    )

    assert (tmp_path / "synthetic_study" / "reports" / "artifact_reuse_summary.csv").exists()


def test_plan_tuning_study_rejects_existing_output_without_overwrite(tmp_path: Path) -> None:
    plan_tuning_study(
        search_space_path=None,
        output_dir=tmp_path,
        study_id="synthetic_study",
        example_synthetic=True,
        repo_root=_repo_root(),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        plan_tuning_study(
            search_space_path=None,
            output_dir=tmp_path,
            study_id="synthetic_study",
            example_synthetic=True,
            repo_root=_repo_root(),
        )


def test_plan_refuses_to_overwrite_existing_study_without_flag(tmp_path: Path) -> None:
    test_plan_tuning_study_rejects_existing_output_without_overwrite(tmp_path)


def test_plan_tuning_study_overwrite_replaces_existing_output(tmp_path: Path) -> None:
    study_dir = tmp_path / "synthetic_study"
    study_dir.mkdir()
    (study_dir / "stale.txt").write_text("stale", encoding="utf-8")

    plan_tuning_study(
        search_space_path=None,
        output_dir=tmp_path,
        study_id="synthetic_study",
        example_synthetic=True,
        overwrite=True,
        repo_root=_repo_root(),
    )

    assert not (study_dir / "stale.txt").exists()
    assert (study_dir / "study_manifest.json").exists()


def test_plan_tuning_study_reads_search_space_yaml(tmp_path: Path) -> None:
    import yaml

    payload = example_search_space_payload()
    search_space_path = tmp_path / "search_space.yaml"
    search_space_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = plan_tuning_study(
        search_space_path=search_space_path,
        output_dir=tmp_path / "out",
        study_id="yaml_study",
        repo_root=_repo_root(),
    )

    assert result["study_id"] == "yaml_study"
    assert (tmp_path / "out" / "yaml_study" / "study_manifest.json").exists()


def test_plan_tuning_study_accepts_mvp_fixture(tmp_path: Path) -> None:
    result = plan_tuning_study(
        search_space_path=_repo_root() / "tests" / "fixtures" / "tuning" / "cb_svdpp_tuning_mvp.yaml",
        output_dir=tmp_path,
        study_id="fixture_study",
        repo_root=_repo_root(),
    )

    study_dir = tmp_path / "fixture_study"

    assert result["candidate_count"] == 6
    assert result["artifact_reuse_group_count"] == 1
    assert (study_dir / "study_manifest.json").exists()
    assert (study_dir / "reports" / "artifact_reuse_summary.csv").exists()


def test_plan_script_writes_dry_run_outputs_for_fixture(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plan_tuning_study.py",
            "--search-space",
            "tests/fixtures/tuning/cb_svdpp_tuning_mvp.yaml",
            "--output-dir",
            str(tmp_path),
            "--study-id",
            "fixture_cli_study",
        ],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )

    study_dir = tmp_path / "fixture_cli_study"

    assert result.returncode == 0
    assert (study_dir / "study_manifest.json").exists()
    assert (study_dir / "reports" / "candidate_summary.csv").exists()
    assert (study_dir / "reports" / "artifact_reuse_summary.csv").exists()


def test_candidate_summary_paths_respect_custom_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "custom_tuning_output"
    plan_tuning_study(
        search_space_path=_repo_root() / "tests" / "fixtures" / "tuning" / "cb_svdpp_tuning_mvp.yaml",
        output_dir=output_dir,
        study_id="custom_path_study",
        repo_root=_repo_root(),
    )

    with (output_dir / "custom_path_study" / "reports" / "candidate_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        row = next(csv.DictReader(handle))

    assert row["candidate_config_path"].startswith(str(output_dir / "custom_path_study"))
    assert row["candidate_manifest_path"].startswith(str(output_dir / "custom_path_study"))
    assert not row["candidate_config_path"].startswith("artifacts/tuning")
    assert not row["candidate_manifest_path"].startswith("artifacts/tuning")


def test_candidate_summary_paths_match_written_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "custom_tuning_output"
    plan_tuning_study(
        search_space_path=_repo_root() / "tests" / "fixtures" / "tuning" / "cb_svdpp_tuning_mvp.yaml",
        output_dir=output_dir,
        study_id="custom_path_study",
        repo_root=_repo_root(),
    )

    with (output_dir / "custom_path_study" / "reports" / "candidate_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    for row in rows:
        assert Path(row["candidate_config_path"]).exists()
        assert Path(row["candidate_manifest_path"]).exists()
