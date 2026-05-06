from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_yaml_module() -> ModuleType:
    try:
        return importlib.import_module("yaml")
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML configs. Run the canonical setup path first.") from exc


def load_yaml_file(path: Path) -> dict[str, Any]:
    yaml: Any = _load_yaml_module()

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"expected YAML mapping at {path}")
    return data


def dump_yaml_file(path: Path, payload: dict[str, Any]) -> None:
    yaml: Any = _load_yaml_module()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
