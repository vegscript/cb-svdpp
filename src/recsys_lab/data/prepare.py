from __future__ import annotations

from pathlib import Path
from typing import Any

from recsys_lab.config.loader import load_yaml_file
from recsys_lab.data.movielens import prepare_movielens_explicit_dataset
from recsys_lab.utils.paths import discover_repo_root


def prepare_dataset_from_config(
    dataset_config_path: Path,
    *,
    dtype: str = "float32",
    overwrite: bool = False,
) -> dict[str, Any]:
    repo_root = discover_repo_root(dataset_config_path.parent)
    config = load_yaml_file(dataset_config_path)
    dataset = config["dataset"]

    raw_dir = (repo_root / dataset["raw_relpath"]).resolve()
    output_dir = (repo_root / dataset["processed_relpath"]).resolve()
    source_manifest_path = dataset.get("raw_manifest_relpath")
    source_manifest = (repo_root / source_manifest_path).resolve() if source_manifest_path else None

    artifacts = prepare_movielens_explicit_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        dataset_name=dataset["name"],
        dataset_short_name=dataset["short_name"],
        split_family=dataset["default_split_family"],
        dtype=dtype,
        format_family=str(dataset.get("format_family", "auto")),
        source_manifest_path=source_manifest,
        overwrite=overwrite,
    )

    return {
        "dataset_config": str(dataset_config_path.resolve()),
        "raw_dir": str(raw_dir),
        "processed_dir": str(output_dir),
        "manifest": str(artifacts.manifest_path),
        "artifacts": {
            "interactions": str(artifacts.interactions_path),
            "user_mapping": str(artifacts.user_mapping_path),
            "item_mapping": str(artifacts.item_mapping_path),
            "movies": str(artifacts.movies_path),
            "links": str(artifacts.links_path),
            "tags": str(artifacts.tags_path),
        },
    }
