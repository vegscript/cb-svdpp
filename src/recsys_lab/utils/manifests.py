from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recsys_lab.utils.paths import discover_repo_root

ManifestKind = str


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"expected JSON object at {path}")
    return data


def infer_manifest_kind(payload: dict[str, Any]) -> ManifestKind:
    kind = payload.get("kind")
    if kind not in {"run_manifest", "benchmark_manifest"}:
        raise ValueError("manifest kind must be 'run_manifest' or 'benchmark_manifest'")
    return kind


def schema_path_for_kind(kind: ManifestKind, repo_root: Path | None = None) -> Path:
    root = repo_root or discover_repo_root()
    if kind == "run_manifest":
        return root / "schema" / "reporting" / "run_manifest.schema.json"
    if kind == "benchmark_manifest":
        return root / "schema" / "reporting" / "benchmark_manifest.schema.json"
    raise ValueError(f"unsupported manifest kind: {kind}")


def validate_manifest_file(path: Path, repo_root: Path | None = None) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("jsonschema is required to validate manifests. Run the canonical setup path first.") from exc

    payload = load_json_file(path)
    kind = infer_manifest_kind(payload)
    schema_path = schema_path_for_kind(kind, repo_root=repo_root)
    schema = load_json_file(schema_path)
    jsonschema.validate(instance=payload, schema=schema)
