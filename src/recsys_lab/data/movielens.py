from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from recsys_lab.data.ml100k_official_splits import (
    official_ml100k_split_paths,
    read_legacy_ml100k_split,
)
from recsys_lab.data.processed import (
    build_interaction_array_manifest_payload,
    write_interaction_array_artifacts,
)

MODERN_REQUIRED_FILES = {
    "ratings": "ratings.csv",
    "movies": "movies.csv",
    "links": "links.csv",
    "tags": "tags.csv",
}

MODERN_EXPECTED_HEADERS = {
    "ratings": ["userId", "movieId", "rating", "timestamp"],
    "movies": ["movieId", "title", "genres"],
    "links": ["movieId", "imdbId", "tmdbId"],
    "tags": ["userId", "movieId", "tag", "timestamp"],
}

LEGACY_100K_REQUIRED_FILES = {
    "ratings": "u.data",
    "movies": "u.item",
    "genres": "u.genre",
}

LEGACY_100K_RATING_FIELDS = ["userId", "movieId", "rating", "timestamp"]

LEGACY_1M_REQUIRED_FILES = {
    "ratings": "ratings.dat",
    "movies": "movies.dat",
    "users": "users.dat",
}

LEGACY_1M_RATING_FIELDS = ["userId", "movieId", "rating", "timestamp"]
LEGACY_1M_MOVIE_FIELDS = ["movieId", "title", "genres"]
LEGACY_1M_USER_FIELDS = ["userId", "gender", "age", "occupation", "zipCode"]

LEGACY_10M_REQUIRED_FILES = {
    "ratings": "ratings.dat",
    "movies": "movies.dat",
    "tags": "tags.dat",
}

LEGACY_10M_RATING_FIELDS = ["userId", "movieId", "rating", "timestamp"]
LEGACY_10M_MOVIE_FIELDS = ["movieId", "title", "genres"]
LEGACY_10M_TAG_FIELDS = ["userId", "movieId", "tag", "timestamp"]


@dataclass(frozen=True, slots=True)
class PreparedDatasetArtifacts:
    manifest_path: Path
    interactions_path: Path
    user_ids_array_path: Path
    item_ids_array_path: Path
    ratings_array_path: Path
    user_mapping_path: Path
    item_mapping_path: Path
    movies_path: Path
    links_path: Path
    tags_path: Path


@dataclass(frozen=True, slots=True)
class RatingArrays:
    raw_user_ids: np.ndarray
    raw_item_ids: np.ndarray
    ratings: np.ndarray
    timestamps: np.ndarray


def _required_path(raw_dir: Path, relative_name: str) -> Path:
    path = raw_dir / relative_name
    if not path.exists():
        raise FileNotFoundError(f"missing required MovieLens file: {path}")
    return path


def _read_delimited_rows(
    path: Path,
    *,
    delimiter: str,
    fieldnames: list[str] | None = None,
    encoding: str = "utf-8",
) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle, fieldnames=fieldnames, delimiter=delimiter)
        rows = [dict(row) for row in reader]
        if fieldnames is None:
            actual_fieldnames = list(reader.fieldnames or [])
        else:
            actual_fieldnames = list(fieldnames)
    return actual_fieldnames, rows


def _detect_movielens_format_family(raw_dir: Path) -> str:
    modern_present = all((raw_dir / filename).exists() for filename in MODERN_REQUIRED_FILES.values())
    legacy_100k_present = all((raw_dir / filename).exists() for filename in LEGACY_100K_REQUIRED_FILES.values())
    legacy_1m_present = all((raw_dir / filename).exists() for filename in LEGACY_1M_REQUIRED_FILES.values())
    legacy_10m_present = all((raw_dir / filename).exists() for filename in LEGACY_10M_REQUIRED_FILES.values())

    if modern_present:
        return "modern_csv"
    if legacy_100k_present:
        return "legacy_100k"
    if legacy_1m_present:
        return "legacy_1m"
    if legacy_10m_present:
        return "legacy_10m"
    raise FileNotFoundError(
        f"unable to detect supported MovieLens layout in {raw_dir}; "
        "expected modern CSV files, legacy ml100k, legacy ml1m, or legacy ml10m layout"
    )


def _read_legacy_double_colon_rows(
    path: Path,
    *,
    fieldnames: list[str],
    encoding: str = "latin-1",
) -> tuple[list[str], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding=encoding, newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("::")
            if len(parts) != len(fieldnames):
                raise ValueError(
                    f"unexpected field count in {path.name} at line {line_number}: "
                    f"expected {len(fieldnames)}, got {len(parts)}"
                )
            rows.append(dict(zip(fieldnames, parts, strict=True)))
    return list(fieldnames), rows


def _count_nonempty_lines(path: Path, *, encoding: str = "utf-8") -> int:
    with path.open("r", encoding=encoding, newline="") as handle:
        return sum(1 for raw_line in handle if raw_line.rstrip("\n\r"))


def _validate_delimited_header(path: Path, *, delimiter: str, expected: list[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        actual = next(reader, [])
    if actual != expected:
        raise ValueError(f"unexpected header in {path.name}: expected {expected}, got {actual}")


def _read_legacy_100k_genres(raw_dir: Path) -> list[str]:
    genre_path = _required_path(raw_dir, LEGACY_100K_REQUIRED_FILES["genres"])
    genres: list[str] = []
    with genre_path.open("r", encoding="latin-1", newline="") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 2:
                continue
            genre_name = parts[0].strip()
            if genre_name:
                genres.append(genre_name)
    if not genres:
        raise ValueError(f"legacy ml100k genre file is empty: {genre_path}")
    return genres


def _validate_modern_directory(raw_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, filename in MODERN_REQUIRED_FILES.items():
        path = _required_path(raw_dir, filename)
        expected = MODERN_EXPECTED_HEADERS[key]
        _validate_delimited_header(path, delimiter=",", expected=expected)
        counts[key] = max(_count_nonempty_lines(path) - 1, 0)
    return counts


def _validate_legacy_100k_directory(raw_dir: Path) -> dict[str, int]:
    ratings_path = _required_path(raw_dir, LEGACY_100K_REQUIRED_FILES["ratings"])
    movies_path = _required_path(raw_dir, LEGACY_100K_REQUIRED_FILES["movies"])
    genre_count = len(_read_legacy_100k_genres(raw_dir))
    return {
        "ratings": _count_nonempty_lines(ratings_path, encoding="latin-1"),
        "movies": _count_nonempty_lines(movies_path, encoding="latin-1"),
        "genres": genre_count,
        "links": 0,
        "tags": 0,
    }


def _validate_legacy_1m_directory(raw_dir: Path) -> dict[str, int]:
    ratings_path = _required_path(raw_dir, LEGACY_1M_REQUIRED_FILES["ratings"])
    movies_path = _required_path(raw_dir, LEGACY_1M_REQUIRED_FILES["movies"])
    users_path = _required_path(raw_dir, LEGACY_1M_REQUIRED_FILES["users"])

    return {
        "ratings": _count_nonempty_lines(ratings_path, encoding="latin-1"),
        "movies": _count_nonempty_lines(movies_path, encoding="latin-1"),
        "users": _count_nonempty_lines(users_path, encoding="latin-1"),
        "links": 0,
        "tags": 0,
    }


def _validate_legacy_10m_directory(raw_dir: Path) -> dict[str, int]:
    ratings_path = _required_path(raw_dir, LEGACY_10M_REQUIRED_FILES["ratings"])
    movies_path = _required_path(raw_dir, LEGACY_10M_REQUIRED_FILES["movies"])
    tags_path = _required_path(raw_dir, LEGACY_10M_REQUIRED_FILES["tags"])

    return {
        "ratings": _count_nonempty_lines(ratings_path),
        "movies": _count_nonempty_lines(movies_path),
        "links": 0,
        "tags": _count_nonempty_lines(tags_path),
    }


def validate_movielens_directory(
    raw_dir: Path,
    *,
    format_family: str = "auto",
) -> dict[str, Any]:
    resolved_raw_dir = raw_dir.resolve()
    actual_format_family = (
        _detect_movielens_format_family(resolved_raw_dir) if format_family == "auto" else format_family
    )

    if actual_format_family == "modern_csv":
        counts = _validate_modern_directory(resolved_raw_dir)
    elif actual_format_family == "legacy_100k":
        counts = _validate_legacy_100k_directory(resolved_raw_dir)
    elif actual_format_family == "legacy_1m":
        counts = _validate_legacy_1m_directory(resolved_raw_dir)
    elif actual_format_family == "legacy_10m":
        counts = _validate_legacy_10m_directory(resolved_raw_dir)
    else:
        raise ValueError(f"unsupported MovieLens format_family: {actual_format_family}")

    return {
        "raw_dir": str(resolved_raw_dir),
        "format_family": actual_format_family,
        "counts": counts,
    }


def _write_table(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _write_npy_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)


def _arrow_column_to_numpy(table: pa.Table, column_name: str, dtype: np.dtype | type) -> np.ndarray:
    return np.asarray(table[column_name].combine_chunks().to_numpy(), dtype=dtype)


def _rating_arrow_type(dtype: str) -> pa.DataType:
    return pa.float32() if dtype == "float32" else pa.float64()


def _rating_numpy_dtype(dtype: str) -> np.dtype:
    return np.dtype(np.float32 if dtype == "float32" else np.float64)


def _read_modern_rating_arrays(path: Path, *, dtype: str) -> RatingArrays:
    table = pacsv.read_csv(
        path,
        convert_options=pacsv.ConvertOptions(
            include_columns=["userId", "movieId", "rating", "timestamp"],
            column_types={
                "userId": pa.int32(),
                "movieId": pa.int32(),
                "rating": _rating_arrow_type(dtype),
                "timestamp": pa.int64(),
            },
        ),
    )
    return RatingArrays(
        raw_user_ids=_arrow_column_to_numpy(table, "userId", np.int32),
        raw_item_ids=_arrow_column_to_numpy(table, "movieId", np.int32),
        ratings=_arrow_column_to_numpy(table, "rating", _rating_numpy_dtype(dtype)),
        timestamps=_arrow_column_to_numpy(table, "timestamp", np.int64),
    )


def _read_legacy_tab_rating_arrays(path: Path, *, dtype: str) -> RatingArrays:
    table = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(column_names=LEGACY_100K_RATING_FIELDS),
        parse_options=pacsv.ParseOptions(delimiter="\t"),
        convert_options=pacsv.ConvertOptions(
            include_columns=LEGACY_100K_RATING_FIELDS,
            column_types={
                "userId": pa.int32(),
                "movieId": pa.int32(),
                "rating": _rating_arrow_type(dtype),
                "timestamp": pa.int64(),
            },
        ),
    )
    return RatingArrays(
        raw_user_ids=_arrow_column_to_numpy(table, "userId", np.int32),
        raw_item_ids=_arrow_column_to_numpy(table, "movieId", np.int32),
        ratings=_arrow_column_to_numpy(table, "rating", _rating_numpy_dtype(dtype)),
        timestamps=_arrow_column_to_numpy(table, "timestamp", np.int64),
    )


def _read_legacy_double_colon_rating_arrays(
    path: Path,
    *,
    dtype: str,
) -> RatingArrays:
    column_names = ["c0", "c1", "c2", "c3", "c4", "c5", "c6"]
    table = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(column_names=column_names, block_size=1 << 20),
        parse_options=pacsv.ParseOptions(delimiter=":"),
        convert_options=pacsv.ConvertOptions(
            include_columns=["c0", "c2", "c4", "c6"],
            column_types={
                "c0": pa.int32(),
                "c2": pa.int32(),
                "c4": _rating_arrow_type(dtype),
                "c6": pa.int64(),
            },
        ),
    )
    return RatingArrays(
        raw_user_ids=_arrow_column_to_numpy(table, "c0", np.int32),
        raw_item_ids=_arrow_column_to_numpy(table, "c2", np.int32),
        ratings=_arrow_column_to_numpy(table, "c4", _rating_numpy_dtype(dtype)),
        timestamps=_arrow_column_to_numpy(table, "c6", np.int64),
    )


def _read_rating_arrays_for_format(
    *,
    raw_dir: Path,
    format_family: str,
    dtype: str,
) -> RatingArrays:
    if format_family == "modern_csv":
        return _read_modern_rating_arrays(
            _required_path(raw_dir, MODERN_REQUIRED_FILES["ratings"]),
            dtype=dtype,
        )
    if format_family == "legacy_100k":
        return _read_legacy_tab_rating_arrays(
            _required_path(raw_dir, LEGACY_100K_REQUIRED_FILES["ratings"]),
            dtype=dtype,
        )
    if format_family == "legacy_1m":
        return _read_legacy_double_colon_rating_arrays(
            _required_path(raw_dir, LEGACY_1M_REQUIRED_FILES["ratings"]),
            dtype=dtype,
        )
    if format_family == "legacy_10m":
        return _read_legacy_double_colon_rating_arrays(
            _required_path(raw_dir, LEGACY_10M_REQUIRED_FILES["ratings"]),
            dtype=dtype,
        )
    raise ValueError(f"unsupported MovieLens format_family: {format_family}")


def _rating_records_from_arrays(rating_arrays: RatingArrays) -> list[dict[str, int | float]]:
    return [
        {
            "raw_user_id": int(raw_user_id),
            "raw_item_id": int(raw_item_id),
            "rating": float(rating),
            "timestamp": int(timestamp),
        }
        for raw_user_id, raw_item_id, rating, timestamp in zip(
            rating_arrays.raw_user_ids,
            rating_arrays.raw_item_ids,
            rating_arrays.ratings,
            rating_arrays.timestamps,
            strict=True,
        )
    ]


def _clear_directory_contents(output_dir: Path) -> None:
    for child in output_dir.iterdir():
        if child.is_dir():
            for nested_file in child.rglob("*"):
                if nested_file.is_file():
                    nested_file.unlink()
            for nested_dir in sorted(child.rglob("*"), reverse=True):
                if nested_dir.is_dir():
                    nested_dir.rmdir()
            child.rmdir()
        else:
            child.unlink()


def _build_empty_links_table() -> pa.Table:
    return pa.table(
        {
            "raw_item_id": pa.array([], type=pa.int32()),
            "item_idx": pa.array([], type=pa.int32()),
            "imdb_id": pa.array([], type=pa.string()),
            "tmdb_id": pa.array([], type=pa.string()),
        }
    )


def _build_empty_tags_table() -> pa.Table:
    return pa.table(
        {
            "raw_user_id": pa.array([], type=pa.int32()),
            "user_idx": pa.array([], type=pa.int32()),
            "raw_item_id": pa.array([], type=pa.int32()),
            "item_idx": pa.array([], type=pa.int32()),
            "tag": pa.array([], type=pa.string()),
            "timestamp": pa.array([], type=pa.int64()),
        }
    )


def _prepare_modern_auxiliary_tables(
    *,
    raw_dir: Path,
    item_to_idx: dict[int, int],
    user_to_idx: dict[int, int],
) -> tuple[pa.Table, pa.Table, pa.Table, int, int]:
    _, movies_rows = _read_delimited_rows(
        _required_path(raw_dir, MODERN_REQUIRED_FILES["movies"]),
        delimiter=",",
    )
    _, links_rows = _read_delimited_rows(
        _required_path(raw_dir, MODERN_REQUIRED_FILES["links"]),
        delimiter=",",
    )
    _, tags_rows = _read_delimited_rows(
        _required_path(raw_dir, MODERN_REQUIRED_FILES["tags"]),
        delimiter=",",
    )

    movies_table = pa.table(
        {
            "raw_item_id": pa.array([int(row["movieId"]) for row in movies_rows], type=pa.int32()),
            "item_idx": pa.array(
                [item_to_idx.get(int(row["movieId"])) for row in movies_rows],
                type=pa.int32(),
            ),
            "title": pa.array([row["title"] for row in movies_rows], type=pa.string()),
            "genres": pa.array([row["genres"] for row in movies_rows], type=pa.string()),
        }
    )

    links_table = pa.table(
        {
            "raw_item_id": pa.array([int(row["movieId"]) for row in links_rows], type=pa.int32()),
            "item_idx": pa.array(
                [item_to_idx.get(int(row["movieId"])) for row in links_rows],
                type=pa.int32(),
            ),
            "imdb_id": pa.array([row["imdbId"] or None for row in links_rows], type=pa.string()),
            "tmdb_id": pa.array([row["tmdbId"] or None for row in links_rows], type=pa.string()),
        }
    )

    tags_table = pa.table(
        {
            "raw_user_id": pa.array([int(row["userId"]) for row in tags_rows], type=pa.int32()),
            "user_idx": pa.array(
                [user_to_idx.get(int(row["userId"])) for row in tags_rows],
                type=pa.int32(),
            ),
            "raw_item_id": pa.array([int(row["movieId"]) for row in tags_rows], type=pa.int32()),
            "item_idx": pa.array(
                [item_to_idx.get(int(row["movieId"])) for row in tags_rows],
                type=pa.int32(),
            ),
            "tag": pa.array([row["tag"] for row in tags_rows], type=pa.string()),
            "timestamp": pa.array([int(row["timestamp"]) for row in tags_rows], type=pa.int64()),
        }
    )

    return movies_table, links_table, tags_table, len(links_rows), len(tags_rows)


def _prepare_legacy_100k_auxiliary_tables(
    *,
    raw_dir: Path,
    item_to_idx: dict[int, int],
) -> tuple[pa.Table, pa.Table, pa.Table, int, int]:
    genre_names = _read_legacy_100k_genres(raw_dir)
    movie_path = _required_path(raw_dir, LEGACY_100K_REQUIRED_FILES["movies"])

    raw_item_ids: list[int] = []
    mapped_item_ids: list[int | None] = []
    titles: list[str] = []
    genres: list[str] = []

    with movie_path.open("r", encoding="latin-1", newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        for row in reader:
            if not row:
                continue
            if len(row) < 5 + len(genre_names):
                raise ValueError(f"unexpected legacy ml100k movie row width in {movie_path}")
            raw_item_id = int(row[0])
            title = row[1]
            genre_flags = row[5 : 5 + len(genre_names)]
            active_genres = [
                genre_name for genre_name, flag in zip(genre_names, genre_flags, strict=True) if flag == "1"
            ]
            raw_item_ids.append(raw_item_id)
            mapped_item_ids.append(item_to_idx.get(raw_item_id))
            titles.append(title)
            genres.append("|".join(active_genres) if active_genres else "(no genres listed)")

    movies_table = pa.table(
        {
            "raw_item_id": pa.array(raw_item_ids, type=pa.int32()),
            "item_idx": pa.array(mapped_item_ids, type=pa.int32()),
            "title": pa.array(titles, type=pa.string()),
            "genres": pa.array(genres, type=pa.string()),
        }
    )

    return movies_table, _build_empty_links_table(), _build_empty_tags_table(), 0, 0


def _prepare_legacy_1m_auxiliary_tables(
    *,
    raw_dir: Path,
    item_to_idx: dict[int, int],
) -> tuple[pa.Table, pa.Table, pa.Table, int, int]:
    _, movie_rows = _read_legacy_double_colon_rows(
        _required_path(raw_dir, LEGACY_1M_REQUIRED_FILES["movies"]),
        fieldnames=LEGACY_1M_MOVIE_FIELDS,
    )

    movies_table = pa.table(
        {
            "raw_item_id": pa.array([int(row["movieId"]) for row in movie_rows], type=pa.int32()),
            "item_idx": pa.array(
                [item_to_idx.get(int(row["movieId"])) for row in movie_rows],
                type=pa.int32(),
            ),
            "title": pa.array([row["title"] for row in movie_rows], type=pa.string()),
            "genres": pa.array([row["genres"] for row in movie_rows], type=pa.string()),
        }
    )

    return movies_table, _build_empty_links_table(), _build_empty_tags_table(), 0, 0


def _prepare_legacy_10m_auxiliary_tables(
    *,
    raw_dir: Path,
    item_to_idx: dict[int, int],
    user_to_idx: dict[int, int],
) -> tuple[pa.Table, pa.Table, pa.Table, int, int]:
    _, movie_rows = _read_legacy_double_colon_rows(
        _required_path(raw_dir, LEGACY_10M_REQUIRED_FILES["movies"]),
        fieldnames=LEGACY_10M_MOVIE_FIELDS,
    )
    _, tag_rows = _read_legacy_double_colon_rows(
        _required_path(raw_dir, LEGACY_10M_REQUIRED_FILES["tags"]),
        fieldnames=LEGACY_10M_TAG_FIELDS,
    )

    movies_table = pa.table(
        {
            "raw_item_id": pa.array([int(row["movieId"]) for row in movie_rows], type=pa.int32()),
            "item_idx": pa.array(
                [item_to_idx.get(int(row["movieId"])) for row in movie_rows],
                type=pa.int32(),
            ),
            "title": pa.array([row["title"] for row in movie_rows], type=pa.string()),
            "genres": pa.array([row["genres"] for row in movie_rows], type=pa.string()),
        }
    )

    tags_table = pa.table(
        {
            "raw_user_id": pa.array([int(row["userId"]) for row in tag_rows], type=pa.int32()),
            "user_idx": pa.array(
                [user_to_idx.get(int(row["userId"])) for row in tag_rows],
                type=pa.int32(),
            ),
            "raw_item_id": pa.array([int(row["movieId"]) for row in tag_rows], type=pa.int32()),
            "item_idx": pa.array(
                [item_to_idx.get(int(row["movieId"])) for row in tag_rows],
                type=pa.int32(),
            ),
            "tag": pa.array([row["tag"] for row in tag_rows], type=pa.string()),
            "timestamp": pa.array([int(row["timestamp"]) for row in tag_rows], type=pa.int64()),
        }
    )

    return movies_table, _build_empty_links_table(), tags_table, 0, len(tag_rows)


def _maybe_build_official_ml100k_split_artifacts(
    *,
    raw_dir: Path,
    output_dir: Path,
    prefix: str,
    dataset_short_name: str,
    format_family: str,
    ratings_records: list[dict[str, int | float]],
) -> dict[str, Any] | None:
    if dataset_short_name != "ml100k" or format_family != "legacy_100k":
        return None

    fold_paths = official_ml100k_split_paths(raw_dir)
    if fold_paths is None:
        return None

    row_lookup: dict[tuple[int, int, float, int], int] = {}
    for row_index, record in enumerate(ratings_records):
        key = (
            int(record["raw_user_id"]),
            int(record["raw_item_id"]),
            float(record["rating"]),
            int(record["timestamp"]),
        )
        if key in row_lookup:
            raise ValueError("duplicate interaction key encountered while building official ml100k split artifacts")
        row_lookup[key] = row_index

    folds_payload: dict[str, dict[str, str]] = {}
    for fold_index, paths in fold_paths.items():
        train_indices = np.asarray(
            [row_lookup[record] for record in read_legacy_ml100k_split(paths["train"])],
            dtype=np.int64,
        )
        test_indices = np.asarray(
            [row_lookup[record] for record in read_legacy_ml100k_split(paths["test"])],
            dtype=np.int64,
        )
        train_path = output_dir / f"{prefix}_paper_faithful_u{fold_index}_train_row_indices.npy"
        test_path = output_dir / f"{prefix}_paper_faithful_u{fold_index}_test_row_indices.npy"
        _write_npy_array(train_path, train_indices)
        _write_npy_array(test_path, test_indices)
        folds_payload[f"u{fold_index}"] = {
            "train_row_indices_npy": str(train_path),
            "test_row_indices_npy": str(test_path),
        }

    return {
        "version": "paper_faithful_ml100k_v1",
        "folds": folds_payload,
    }


def prepare_movielens_explicit_dataset(
    raw_dir: Path,
    output_dir: Path,
    *,
    dataset_name: str,
    dataset_short_name: str,
    split_family: str,
    preprocessing_family: str = "explicit_v1",
    dtype: str = "float32",
    format_family: str = "auto",
    source_manifest_path: Path | None = None,
    overwrite: bool = False,
) -> PreparedDatasetArtifacts:
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")

    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory already contains files: {output_dir}. Use overwrite=True to replace.")
    if output_dir.exists() and overwrite:
        _clear_directory_contents(output_dir)

    validate_summary = validate_movielens_directory(raw_dir, format_family=format_family)
    actual_format_family = str(validate_summary["format_family"])

    rating_arrays = _read_rating_arrays_for_format(
        raw_dir=raw_dir,
        format_family=actual_format_family,
        dtype=dtype,
    )
    row_count = int(rating_arrays.ratings.shape[0])
    if row_count == 0:
        raise ValueError(f"ratings file for {dataset_short_name} is empty")

    unique_user_ids, user_idx = np.unique(rating_arrays.raw_user_ids, return_inverse=True)
    unique_item_ids, item_idx = np.unique(rating_arrays.raw_item_ids, return_inverse=True)
    user_idx = user_idx.astype(np.int32, copy=False)
    item_idx = item_idx.astype(np.int32, copy=False)
    user_to_idx = {int(raw_id): int(idx) for idx, raw_id in enumerate(unique_user_ids)}
    item_to_idx = {int(raw_id): int(idx) for idx, raw_id in enumerate(unique_item_ids)}

    interactions_table = pa.table(
        {
            "user_idx": pa.array(user_idx, type=pa.int32()),
            "item_idx": pa.array(item_idx, type=pa.int32()),
            "rating": pa.array(rating_arrays.ratings, type=_rating_arrow_type(dtype)),
            "timestamp": pa.array(rating_arrays.timestamps, type=pa.int64()),
            "raw_user_id": pa.array(rating_arrays.raw_user_ids, type=pa.int32()),
            "raw_item_id": pa.array(rating_arrays.raw_item_ids, type=pa.int32()),
        }
    )

    user_mapping_table = pa.table(
        {
            "raw_user_id": pa.array(unique_user_ids.astype(np.int32, copy=False), type=pa.int32()),
            "user_idx": pa.array(np.arange(len(unique_user_ids), dtype=np.int32), type=pa.int32()),
        }
    )
    item_mapping_table = pa.table(
        {
            "raw_item_id": pa.array(unique_item_ids.astype(np.int32, copy=False), type=pa.int32()),
            "item_idx": pa.array(np.arange(len(unique_item_ids), dtype=np.int32), type=pa.int32()),
        }
    )

    if actual_format_family == "modern_csv":
        movies_table, links_table, tags_table, link_count, tag_count = _prepare_modern_auxiliary_tables(
            raw_dir=raw_dir,
            item_to_idx=item_to_idx,
            user_to_idx=user_to_idx,
        )
    elif actual_format_family == "legacy_100k":
        movies_table, links_table, tags_table, link_count, tag_count = _prepare_legacy_100k_auxiliary_tables(
            raw_dir=raw_dir,
            item_to_idx=item_to_idx,
        )
    elif actual_format_family == "legacy_1m":
        movies_table, links_table, tags_table, link_count, tag_count = _prepare_legacy_1m_auxiliary_tables(
            raw_dir=raw_dir,
            item_to_idx=item_to_idx,
        )
    elif actual_format_family == "legacy_10m":
        movies_table, links_table, tags_table, link_count, tag_count = _prepare_legacy_10m_auxiliary_tables(
            raw_dir=raw_dir,
            item_to_idx=item_to_idx,
            user_to_idx=user_to_idx,
        )
    else:
        raise ValueError(f"unsupported MovieLens format_family: {actual_format_family}")

    prefix = f"{dataset_short_name}_{split_family}_{preprocessing_family}_{dtype}"
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = PreparedDatasetArtifacts(
        manifest_path=output_dir / f"{prefix}_manifest.json",
        interactions_path=output_dir / f"{prefix}_interactions.parquet",
        user_ids_array_path=output_dir / f"{prefix}_user_ids.npy",
        item_ids_array_path=output_dir / f"{prefix}_item_ids.npy",
        ratings_array_path=output_dir / f"{prefix}_ratings.npy",
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
    official_ml100k_splits = _maybe_build_official_ml100k_split_artifacts(
        raw_dir=raw_dir,
        output_dir=output_dir,
        prefix=prefix,
        dataset_short_name=dataset_short_name,
        format_family=actual_format_family,
        ratings_records=(
            _rating_records_from_arrays(rating_arrays)
            if dataset_short_name == "ml100k" and actual_format_family == "legacy_100k"
            else []
        ),
    )
    interaction_array_artifacts = write_interaction_array_artifacts(
        user_ids=user_idx,
        item_ids=item_idx,
        ratings=rating_arrays.ratings,
        output_dir=output_dir,
        prefix=prefix,
    )

    manifest_payload: dict[str, Any] = {
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
            "format_family": actual_format_family,
        },
        "validation": validate_summary,
        "counts": {
            "interactions": row_count,
            "users": len(unique_user_ids),
            "rated_items": len(unique_item_ids),
            "catalog_items": int(movies_table.num_rows),
            "tags": tag_count,
            "links": link_count,
        },
        "rating_range": {
            "min": float(np.min(rating_arrays.ratings)),
            "max": float(np.max(rating_arrays.ratings)),
        },
        "artifacts": {
            "interactions": str(artifacts.interactions_path),
            "interaction_arrays": build_interaction_array_manifest_payload(interaction_array_artifacts),
            "user_mapping": str(artifacts.user_mapping_path),
            "item_mapping": str(artifacts.item_mapping_path),
            "movies": str(artifacts.movies_path),
            "links": str(artifacts.links_path),
            "tags": str(artifacts.tags_path),
        },
    }
    if official_ml100k_splits is not None:
        manifest_payload["artifacts"]["official_ml100k_splits"] = official_ml100k_splits
    artifacts.manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    return artifacts
