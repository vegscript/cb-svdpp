from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from recsys_lab.data.ml100k_official_splits import read_legacy_ml100k_split
from recsys_lab.data.processed import RatingsData, load_processed_dataset_manifest

OFFICIAL_ML100K_SPLIT_SOURCE_PROCESSED_MANIFEST = "processed_manifest_official_indices"
OFFICIAL_ML100K_SPLIT_SOURCE_LEGACY_RUNTIME_LOOKUP = "legacy_runtime_lookup_fallback"


@dataclass(frozen=True, slots=True)
class RatingsSplit:
    train: RatingsData
    validation: RatingsData | None
    test: RatingsData


def _paper_faithful_ml100k_split_indices(
    *,
    processed_manifest_path: Path,
    fold_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if fold_index not in {1, 2, 3, 4, 5}:
        raise ValueError("paper-faithful ml100k fold_index must be one of 1, 2, 3, 4, 5")

    processed_manifest = load_processed_dataset_manifest(processed_manifest_path)
    if str(processed_manifest["dataset_short_name"]) != "ml100k":
        raise ValueError("paper-faithful ml100k splits are only valid for dataset_short_name='ml100k'")

    source = processed_manifest.get("source", {})
    if str(source.get("format_family")) != "legacy_100k":
        raise ValueError("paper-faithful ml100k splits require source.format_family='legacy_100k'")

    artifacts = processed_manifest.get("artifacts", {})
    official_splits = artifacts.get("official_ml100k_splits")
    if isinstance(official_splits, dict):
        folds_payload = official_splits.get("folds")
        fold_payload = None if not isinstance(folds_payload, dict) else folds_payload.get(f"u{fold_index}")
        if isinstance(fold_payload, dict):
            train_indices = np.load(Path(str(fold_payload["train_row_indices_npy"])), mmap_mode="r")
            test_indices = np.load(Path(str(fold_payload["test_row_indices_npy"])), mmap_mode="r")
            return (
                np.asarray(train_indices, dtype=np.int64),
                np.asarray(test_indices, dtype=np.int64),
            )

    raw_dir = Path(str(source["raw_dir"])).resolve()
    train_split_path = raw_dir / f"u{fold_index}.base"
    test_split_path = raw_dir / f"u{fold_index}.test"
    if not train_split_path.exists() or not test_split_path.exists():
        raise FileNotFoundError(
            f"missing official ml100k split files for fold {fold_index}: {train_split_path} / {test_split_path}"
        )

    interactions_path = Path(str(processed_manifest["artifacts"]["interactions"])).resolve()
    table = pq.read_table(
        interactions_path,
        columns=["raw_user_id", "raw_item_id", "rating", "timestamp"],
    )
    raw_user_ids = table["raw_user_id"].to_numpy().astype(np.int32, copy=False)
    raw_item_ids = table["raw_item_id"].to_numpy().astype(np.int32, copy=False)
    ratings = table["rating"].to_numpy().astype(np.float64, copy=False)
    timestamps = table["timestamp"].to_numpy().astype(np.int64, copy=False)

    row_lookup: dict[tuple[int, int, float, int], int] = {}
    for idx in range(raw_user_ids.shape[0]):
        key = (
            int(raw_user_ids[idx]),
            int(raw_item_ids[idx]),
            float(ratings[idx]),
            int(timestamps[idx]),
        )
        if key in row_lookup:
            raise ValueError("duplicate interaction key encountered while building ml100k split lookup")
        row_lookup[key] = idx

    def resolve_indices(split_records: list[tuple[int, int, float, int]]) -> np.ndarray:
        indices: list[int] = []
        for record in split_records:
            if record not in row_lookup:
                raise KeyError(f"split record not found in processed interactions: {record}")
            indices.append(int(row_lookup[record]))
        return np.asarray(indices, dtype=np.int64)

    train_idx = resolve_indices(read_legacy_ml100k_split(train_split_path))
    test_idx = resolve_indices(read_legacy_ml100k_split(test_split_path))

    train_set = set(train_idx.tolist())
    test_set = set(test_idx.tolist())
    if train_set & test_set:
        raise ValueError("official ml100k train/test splits are not disjoint")
    if len(train_set) + len(test_set) != raw_user_ids.shape[0]:
        raise ValueError("official ml100k split files do not cover the full processed interaction table")

    return train_idx, test_idx


def official_ml100k_split_resolution_source(
    *,
    processed_manifest_path: Path,
    fold_index: int,
) -> str:
    if fold_index not in {1, 2, 3, 4, 5}:
        raise ValueError("paper-faithful ml100k fold_index must be one of 1, 2, 3, 4, 5")

    processed_manifest = load_processed_dataset_manifest(processed_manifest_path)
    if str(processed_manifest["dataset_short_name"]) != "ml100k":
        raise ValueError("paper-faithful ml100k split source is only valid for dataset_short_name='ml100k'")

    artifacts = processed_manifest.get("artifacts", {})
    official_splits = artifacts.get("official_ml100k_splits")
    if isinstance(official_splits, dict):
        folds_payload = official_splits.get("folds")
        fold_payload = None if not isinstance(folds_payload, dict) else folds_payload.get(f"u{fold_index}")
        if isinstance(fold_payload, dict):
            return OFFICIAL_ML100K_SPLIT_SOURCE_PROCESSED_MANIFEST

    return OFFICIAL_ML100K_SPLIT_SOURCE_LEGACY_RUNTIME_LOOKUP


def official_ml100k_paper_faithful_split(
    data: RatingsData,
    *,
    processed_manifest_path: Path,
    fold_index: int,
) -> RatingsSplit:
    train_idx, test_idx = _paper_faithful_ml100k_split_indices(
        processed_manifest_path=processed_manifest_path,
        fold_index=fold_index,
    )
    return RatingsSplit(
        train=data.subset(train_idx, name=f"{data.name}:paper_train_u{fold_index}"),
        validation=None,
        test=data.subset(test_idx, name=f"{data.name}:paper_test_u{fold_index}"),
    )


def train_validation_split_with_train_coverage(
    data: RatingsData,
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[RatingsData, RatingsData]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in (0, 1)")

    rng = np.random.default_rng(seed)
    n_rows = len(data)
    permutation = rng.permutation(n_rows)

    validation_target = int(n_rows * validation_ratio)
    if validation_target <= 0 or validation_target >= n_rows:
        raise ValueError("validation_ratio must leave at least one row in train and validation")

    train_idx = list(permutation[:-validation_target])
    validation_idx = list(permutation[-validation_target:])

    train_user_set = set(int(data.user_ids[idx]) for idx in train_idx)
    train_item_set = set(int(data.item_ids[idx]) for idx in train_idx)

    kept_validation: list[int] = []
    for idx in validation_idx:
        user_id = int(data.user_ids[idx])
        item_id = int(data.item_ids[idx])
        if user_id not in train_user_set or item_id not in train_item_set:
            train_idx.append(int(idx))
            train_user_set.add(user_id)
            train_item_set.add(item_id)
        else:
            kept_validation.append(int(idx))

    if not kept_validation:
        raise ValueError("validation split became empty after enforcing train coverage")

    train_array = np.asarray(train_idx, dtype=np.int64)
    validation_array = np.asarray(kept_validation, dtype=np.int64)
    return (
        data.subset(train_array, name=f"{data.name}:train"),
        data.subset(validation_array, name=f"{data.name}:validation"),
    )


def official_ml100k_inner_validation_split(
    data: RatingsData,
    *,
    processed_manifest_path: Path,
    fold_index: int,
    validation_ratio: float,
    inner_seed: int,
) -> RatingsSplit:
    train_idx, test_idx = _paper_faithful_ml100k_split_indices(
        processed_manifest_path=processed_manifest_path,
        fold_index=fold_index,
    )
    outer_train = data.subset(train_idx, name=f"{data.name}:paper_outer_train_u{fold_index}")
    inner_train, inner_validation = train_validation_split_with_train_coverage(
        outer_train,
        validation_ratio=validation_ratio,
        seed=inner_seed,
    )
    return RatingsSplit(
        train=inner_train,
        validation=inner_validation,
        test=data.subset(test_idx, name=f"{data.name}:paper_outer_test_u{fold_index}"),
    )


def random_split_with_train_coverage(
    data: RatingsData,
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> RatingsSplit:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0, 1)")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be < 1")

    rng = np.random.default_rng(seed)
    n_rows = len(data)
    permutation = rng.permutation(n_rows)

    train_end = int(n_rows * train_ratio)
    validation_end = train_end + int(n_rows * validation_ratio)

    train_idx = list(permutation[:train_end])
    validation_idx = list(permutation[train_end:validation_end])
    test_idx = list(permutation[validation_end:])

    train_user_set = set(int(data.user_ids[idx]) for idx in train_idx)
    train_item_set = set(int(data.item_ids[idx]) for idx in train_idx)

    remainder = validation_idx + test_idx
    kept_remainder: list[int] = []

    for idx in remainder:
        user_id = int(data.user_ids[idx])
        item_id = int(data.item_ids[idx])
        if user_id not in train_user_set or item_id not in train_item_set:
            train_idx.append(int(idx))
            train_user_set.add(user_id)
            train_item_set.add(item_id)
        else:
            kept_remainder.append(int(idx))

    original_val_target = len(validation_idx)
    new_validation_idx = np.asarray(kept_remainder[:original_val_target], dtype=np.int64)
    new_test_idx = np.asarray(kept_remainder[original_val_target:], dtype=np.int64)
    new_train_idx = np.asarray(train_idx, dtype=np.int64)

    return RatingsSplit(
        train=data.subset(new_train_idx, name=f"{data.name}:train"),
        validation=data.subset(new_validation_idx, name=f"{data.name}:validation"),
        test=data.subset(new_test_idx, name=f"{data.name}:test"),
    )
