from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


REQUIRED_FILES = {
    "ratings": "ratings.csv",
    "movies": "movies.csv",
    "links": "links.csv",
    "tags": "tags.csv",
}

EXPECTED_HEADERS = {
    "ratings": ["userId", "movieId", "rating", "timestamp"],
    "movies": ["movieId", "title", "genres"],
    "links": ["movieId", "imdbId", "tmdbId"],
    "tags": ["userId", "movieId", "tag", "timestamp"],
}


@dataclass(frozen=True, slots=True)
class PreparedDatasetArtifacts:
    manifest_path: Path
    interactions_path: Path
    user_mapping_path: Path
    item_mapping_path: Path
    movies_path: Path
    links_path: Path
    tags_path: Path


def _required_path(raw_dir: Path, key: str) -> Path:
    path = raw_dir / REQUIRED_FILES[key]
    if not path.exists():
        raise FileNotFoundError(f"missing required MovieLens file: {path}")
    return path


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def validate_movielens_directory(raw_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"raw_dir": str(raw_dir)}
    counts: dict[str, int] = {}

    for key in ("ratings", "movies", "links", "tags"):
        path = _required_path(raw_dir, key)
        fieldnames, rows = _read_csv_rows(path)
        expected = EXPECTED_HEADERS[key]
        if fieldnames != expected:
            raise ValueError(
                f"unexpected header in {path.name}: expected {expected}, got {fieldnames}"
            )
        counts[key] = len(rows)

    summary["counts"] = counts
    return summary


def _write_table(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def prepare_movielens_explicit_dataset(
    raw_dir: Path,
    output_dir: Path,
    *,
    dataset_name: str,
    dataset_short_name: str,
    split_family: str,
    preprocessing_family: str = "explicit_v1",
    dtype: str = "float32",
    source_manifest_path: Path | None = None,
    overwrite: bool = False,
) -> PreparedDatasetArtifacts:
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")

    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory already contains files: {output_dir}. Use overwrite=True to replace."
        )

    if output_dir.exists() and overwrite:
        for child in output_dir.iterdir():
            if child.is_dir():
                for nested in child.rglob("*"):
                    if nested.is_file():
                        nested.unlink()
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_dir():
                        nested.rmdir()
                child.rmdir()
            else:
                child.unlink()

    validate_summary = validate_movielens_directory(raw_dir)

    ratings_path = _required_path(raw_dir, "ratings")
    movies_path = _required_path(raw_dir, "movies")
    links_path = _required_path(raw_dir, "links")
    tags_path = _required_path(raw_dir, "tags")

    _, ratings_rows = _read_csv_rows(ratings_path)
    _, movies_rows = _read_csv_rows(movies_path)
    _, links_rows = _read_csv_rows(links_path)
    _, tags_rows = _read_csv_rows(tags_path)

    ratings_records = [
        {
            "raw_user_id": int(row["userId"]),
            "raw_item_id": int(row["movieId"]),
            "rating": float(row["rating"]),
            "timestamp": int(row["timestamp"]),
        }
        for row in ratings_rows
    ]

    unique_user_ids = sorted({row["raw_user_id"] for row in ratings_records})
    unique_item_ids = sorted({row["raw_item_id"] for row in ratings_records})
    user_to_idx = {raw_id: idx for idx, raw_id in enumerate(unique_user_ids)}
    item_to_idx = {raw_id: idx for idx, raw_id in enumerate(unique_item_ids)}

    rating_type = pa.float32() if dtype == "float32" else pa.float64()

    interactions_table = pa.table(
        {
            "user_idx": pa.array(
                [user_to_idx[row["raw_user_id"]] for row in ratings_records], type=pa.int32()
            ),
            "item_idx": pa.array(
                [item_to_idx[row["raw_item_id"]] for row in ratings_records], type=pa.int32()
            ),
            "rating": pa.array([row["rating"] for row in ratings_records], type=rating_type),
            "timestamp": pa.array([row["timestamp"] for row in ratings_records], type=pa.int64()),
            "raw_user_id": pa.array(
                [row["raw_user_id"] for row in ratings_records], type=pa.int32()
            ),
            "raw_item_id": pa.array(
                [row["raw_item_id"] for row in ratings_records], type=pa.int32()
            ),
        }
    )

    user_mapping_table = pa.table(
        {
            "raw_user_id": pa.array(unique_user_ids, type=pa.int32()),
            "user_idx": pa.array(list(range(len(unique_user_ids))), type=pa.int32()),
        }
    )

    item_mapping_table = pa.table(
        {
            "raw_item_id": pa.array(unique_item_ids, type=pa.int32()),
            "item_idx": pa.array(list(range(len(unique_item_ids))), type=pa.int32()),
        }
    )

    movies_table = pa.table(
        {
            "raw_item_id": pa.array([int(row["movieId"]) for row in movies_rows], type=pa.int32()),
            "item_idx": pa.array(
                [item_to_idx.get(int(row["movieId"])) for row in movies_rows], type=pa.int32()
            ),
            "title": pa.array([row["title"] for row in movies_rows], type=pa.string()),
            "genres": pa.array([row["genres"] for row in movies_rows], type=pa.string()),
        }
    )

    links_table = pa.table(
        {
            "raw_item_id": pa.array([int(row["movieId"]) for row in links_rows], type=pa.int32()),
            "item_idx": pa.array(
                [item_to_idx.get(int(row["movieId"])) for row in links_rows], type=pa.int32()
            ),
            "imdb_id": pa.array([row["imdbId"] or None for row in links_rows], type=pa.string()),
            "tmdb_id": pa.array([row["tmdbId"] or None for row in links_rows], type=pa.string()),
        }
    )

    tags_table = pa.table(
        {
            "raw_user_id": pa.array([int(row["userId"]) for row in tags_rows], type=pa.int32()),
            "user_idx": pa.array(
                [user_to_idx.get(int(row["userId"])) for row in tags_rows], type=pa.int32()
            ),
            "raw_item_id": pa.array([int(row["movieId"]) for row in tags_rows], type=pa.int32()),
            "item_idx": pa.array(
                [item_to_idx.get(int(row["movieId"])) for row in tags_rows], type=pa.int32()
            ),
            "tag": pa.array([row["tag"] for row in tags_rows], type=pa.string()),
            "timestamp": pa.array([int(row["timestamp"]) for row in tags_rows], type=pa.int64()),
        }
    )

    prefix = f"{dataset_short_name}_{split_family}_{preprocessing_family}_{dtype}"
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = PreparedDatasetArtifacts(
        manifest_path=output_dir / f"{prefix}_manifest.json",
        interactions_path=output_dir / f"{prefix}_interactions.parquet",
        user_mapping_path=output_dir / f"{prefix}_user_mapping.parquet",
        item_mapping_path=output_dir / f"{prefix}_item_mapping.parquet",
        movies_path=output_dir / f"{prefix}_movies.parquet",
        links_path=output_dir / f"{prefix}_links.parquet",
        tags_path=output_dir / f"{prefix}_tags.parquet",
    )

    _write_table(artifacts.interactions_path, interactions_table)
    _write_table(artifacts.user_mapping_path, user_mapping_table)
    _write_table(artifacts.item_mapping_path, item_mapping_table)
    _write_table(artifacts.movies_path, movies_table)
    _write_table(artifacts.links_path, links_table)
    _write_table(artifacts.tags_path, tags_table)

    manifest_payload = {
        "manifest_version": "v1",
        "kind": "processed_dataset_manifest",
        "dataset_name": dataset_name,
        "dataset_short_name": dataset_short_name,
        "split_family": split_family,
        "preprocessing_family": preprocessing_family,
        "dtype": dtype,
        "source": {
            "raw_dir": str(raw_dir),
            "raw_manifest_path": str(source_manifest_path.resolve()) if source_manifest_path else None,
        },
        "validation": validate_summary,
        "counts": {
            "interactions": len(ratings_records),
            "users": len(unique_user_ids),
            "rated_items": len(unique_item_ids),
            "catalog_items": len(movies_rows),
            "tags": len(tags_rows),
            "links": len(links_rows),
        },
        "rating_range": {
            "min": min(row["rating"] for row in ratings_records),
            "max": max(row["rating"] for row in ratings_records),
        },
        "artifacts": {
            "interactions": str(artifacts.interactions_path),
            "user_mapping": str(artifacts.user_mapping_path),
            "item_mapping": str(artifacts.item_mapping_path),
            "movies": str(artifacts.movies_path),
            "links": str(artifacts.links_path),
            "tags": str(artifacts.tags_path),
        },
    }
    artifacts.manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    return artifacts
