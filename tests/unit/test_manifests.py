from pathlib import Path

import pytest

from recsys_lab.utils.manifests import infer_manifest_kind, load_json_file, schema_path_for_kind
from recsys_lab.utils.paths import discover_repo_root


def test_infer_manifest_kind_accepts_supported_kinds() -> None:
    assert infer_manifest_kind({"kind": "run_manifest"}) == "run_manifest"
    assert infer_manifest_kind({"kind": "benchmark_manifest"}) == "benchmark_manifest"


def test_infer_manifest_kind_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        infer_manifest_kind({"kind": "unknown"})


def test_schema_path_for_kind_points_to_existing_schema() -> None:
    root = discover_repo_root(Path(__file__).resolve().parent)
    assert schema_path_for_kind("run_manifest", repo_root=root).exists()
    assert schema_path_for_kind("benchmark_manifest", repo_root=root).exists()


def test_load_json_file_reads_schema_object() -> None:
    root = discover_repo_root(Path(__file__).resolve().parent)
    path = schema_path_for_kind("run_manifest", repo_root=root)
    payload = load_json_file(path)
    assert isinstance(payload, dict)
    assert payload["properties"]["kind"]["const"] == "run_manifest"
