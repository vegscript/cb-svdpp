from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.profile_cluster_artifacts import PROFILE_VERSION, profile_rows, write_reports


def _payload() -> dict[str, object]:
    return {
        "profile_version": PROFILE_VERSION,
        "claim_boundary": "Diagnostic cluster artifact profiling only; no performance claim.",
        "inputs": {"model": "cb_svdpp"},
        "profiles": [
            {
                "repeat_index": 1,
                "cluster_artifacts": {
                    "cluster_cache_status": "miss",
                    "cluster_cache_read_seconds": 0.1,
                    "cluster_cache_write_seconds": 0.2,
                    "model": "cb_svdpp",
                },
                "user_cluster_history": {
                    "user_cluster_history_cache_status": "miss",
                    "user_cluster_history_build_seconds": 0.3,
                    "model": "cb_svdpp",
                },
            },
            {
                "repeat_index": 2,
                "cluster_artifacts": {
                    "cluster_cache_status": "hit",
                    "cluster_cache_read_seconds": 0.01,
                    "cluster_cache_write_seconds": 0.0,
                    "model": "cb_svdpp",
                },
                "user_cluster_history": {
                    "user_cluster_history_cache_status": "hit",
                    "user_cluster_history_cache_read_seconds": 0.02,
                    "model": "cb_svdpp",
                },
            },
        ],
    }


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
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["profile_version"] == PROFILE_VERSION

    with paths["csv"].open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    assert len(rows) == 4
    assert rows[0]["profile_kind"] == "cluster_artifacts"
    assert rows[0]["cluster_cache_status"] == "miss"
    assert rows[1]["profile_kind"] == "user_cluster_history"
