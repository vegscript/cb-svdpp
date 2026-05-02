from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq

MMapMode = Literal["r+", "r", "w+", "c"]


class RatingsData:
    __slots__ = (
        "_base_user_ids",
        "_base_item_ids",
        "_base_ratings",
        "_row_indices",
        "_materialized_user_ids",
        "_materialized_item_ids",
        "_materialized_ratings",
        "n_users",
        "n_items",
        "name",
        "rating_min",
        "rating_max",
        "source_manifest_path",
    )

    def __init__(
        self,
        *,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        n_users: int,
        n_items: int,
        name: str,
        rating_min: float,
        rating_max: float,
        source_manifest_path: Path,
        row_indices: np.ndarray | None = None,
    ) -> None:
        if user_ids.ndim != 1 or item_ids.ndim != 1 or ratings.ndim != 1:
            raise ValueError("ratings data arrays must be 1D")
        if user_ids.shape[0] != item_ids.shape[0] or user_ids.shape[0] != ratings.shape[0]:
            raise ValueError("ratings data arrays must have identical row counts")

        self._base_user_ids = user_ids
        self._base_item_ids = item_ids
        self._base_ratings = ratings
        self._row_indices = None if row_indices is None else np.asarray(row_indices, dtype=np.int64)
        if self._row_indices is not None and self._row_indices.ndim != 1:
            raise ValueError("row_indices must be a 1D array when provided")
        self._materialized_user_ids: np.ndarray | None = None
        self._materialized_item_ids: np.ndarray | None = None
        self._materialized_ratings: np.ndarray | None = None
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.name = str(name)
        self.rating_min = float(rating_min)
        self.rating_max = float(rating_max)
        self.source_manifest_path = Path(source_manifest_path)

    @property
    def base_user_ids(self) -> np.ndarray:
        return self._base_user_ids

    @property
    def base_item_ids(self) -> np.ndarray:
        return self._base_item_ids

    @property
    def base_ratings(self) -> np.ndarray:
        return self._base_ratings

    @property
    def row_indices(self) -> np.ndarray | None:
        return self._row_indices

    @property
    def uses_row_selection(self) -> bool:
        return self._row_indices is not None

    @property
    def user_ids(self) -> np.ndarray:
        if self._row_indices is None:
            return self._base_user_ids
        if self._materialized_user_ids is None:
            self._materialized_user_ids = self._base_user_ids[self._row_indices]
        return self._materialized_user_ids

    @property
    def item_ids(self) -> np.ndarray:
        if self._row_indices is None:
            return self._base_item_ids
        if self._materialized_item_ids is None:
            self._materialized_item_ids = self._base_item_ids[self._row_indices]
        return self._materialized_item_ids

    @property
    def ratings(self) -> np.ndarray:
        if self._row_indices is None:
            return self._base_ratings
        if self._materialized_ratings is None:
            self._materialized_ratings = self._base_ratings[self._row_indices]
        return self._materialized_ratings

    def __len__(self) -> int:
        if self._row_indices is None:
            return int(self._base_ratings.shape[0])
        return int(self._row_indices.shape[0])

    def subset(self, indices: np.ndarray, *, name: str) -> "RatingsData":
        idx = np.asarray(indices, dtype=np.int64)
        composed_indices = idx if self._row_indices is None else self._row_indices[idx]
        return RatingsData(
            user_ids=self._base_user_ids,
            item_ids=self._base_item_ids,
            ratings=self._base_ratings,
            n_users=self.n_users,
            n_items=self.n_items,
            name=name,
            rating_min=self.rating_min,
            rating_max=self.rating_max,
            source_manifest_path=self.source_manifest_path,
            row_indices=composed_indices,
        )

    def training_row_indices(self) -> np.ndarray:
        if self._row_indices is None:
            return np.arange(self._base_ratings.shape[0], dtype=np.int64)
        return self._row_indices.astype(np.int64, copy=True)

    def effective_ratings_mean(self) -> float:
        if self._row_indices is None:
            return float(np.mean(self._base_ratings))
        return float(np.mean(self._base_ratings[self._row_indices]))

    def materialize(
        self,
        *,
        name: str | None = None,
        force_copy: bool = False,
    ) -> "RatingsData":
        if not force_copy and self._row_indices is None:
            return self
        return RatingsData(
            user_ids=np.array(self.user_ids, copy=True),
            item_ids=np.array(self.item_ids, copy=True),
            ratings=np.array(self.ratings, copy=True),
            n_users=self.n_users,
            n_items=self.n_items,
            name=self.name if name is None else name,
            rating_min=self.rating_min,
            rating_max=self.rating_max,
            source_manifest_path=self.source_manifest_path,
        )


@dataclass(frozen=True, slots=True)
class InteractionArrayArtifacts:
    user_ids_path: Path
    item_ids_path: Path
    ratings_path: Path


def load_processed_dataset_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object at {path}")
    return payload


def write_interaction_array_artifacts(
    *,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    ratings: np.ndarray,
    output_dir: Path,
    prefix: str,
) -> InteractionArrayArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    user_ids_path = output_dir / f"{prefix}_user_ids.npy"
    item_ids_path = output_dir / f"{prefix}_item_ids.npy"
    ratings_path = output_dir / f"{prefix}_ratings.npy"

    np.save(user_ids_path, np.asarray(user_ids, dtype=np.int32), allow_pickle=False)
    np.save(item_ids_path, np.asarray(item_ids, dtype=np.int32), allow_pickle=False)
    np.save(ratings_path, np.asarray(ratings), allow_pickle=False)

    return InteractionArrayArtifacts(
        user_ids_path=user_ids_path,
        item_ids_path=item_ids_path,
        ratings_path=ratings_path,
    )


def materialize_interaction_array_artifacts_from_manifest(
    path: Path,
    *,
    output_dir: Path,
    prefix: str,
) -> InteractionArrayArtifacts:
    manifest_path = path.resolve()
    payload = load_processed_dataset_manifest(manifest_path)
    artifacts = payload["artifacts"]
    interactions_path = Path(artifacts["interactions"])
    table = pq.read_table(interactions_path, columns=["user_idx", "item_idx", "rating"])
    return write_interaction_array_artifacts(
        user_ids=table["user_idx"].to_numpy().astype(np.int32, copy=False),
        item_ids=table["item_idx"].to_numpy().astype(np.int32, copy=False),
        ratings=table["rating"].to_numpy(),
        output_dir=output_dir,
        prefix=prefix,
    )


def build_interaction_array_manifest_payload(
    artifacts: InteractionArrayArtifacts,
) -> dict[str, str]:
    return {
        "user_ids_npy": str(artifacts.user_ids_path),
        "item_ids_npy": str(artifacts.item_ids_path),
        "ratings_npy": str(artifacts.ratings_path),
    }


def _load_ratings_data_from_array_artifacts(
    *,
    manifest_path: Path,
    payload: dict[str, Any],
    mmap_mode: MMapMode | None,
) -> RatingsData:
    artifacts = payload["artifacts"]
    interaction_arrays = artifacts.get("interaction_arrays")
    if not isinstance(interaction_arrays, dict):
        raise TypeError("artifacts.interaction_arrays must be an object when present")

    user_ids = np.load(Path(interaction_arrays["user_ids_npy"]), mmap_mode=mmap_mode)
    item_ids = np.load(Path(interaction_arrays["item_ids_npy"]), mmap_mode=mmap_mode)
    ratings = np.load(Path(interaction_arrays["ratings_npy"]), mmap_mode=mmap_mode)

    if user_ids.ndim != 1 or item_ids.ndim != 1 or ratings.ndim != 1:
        raise ValueError("interaction array artifacts must be 1D arrays")
    if user_ids.shape[0] != item_ids.shape[0] or user_ids.shape[0] != ratings.shape[0]:
        raise ValueError("interaction array artifacts must have identical row counts")
    if user_ids.dtype != np.int32:
        raise ValueError("interaction array artifact user_ids_npy must be int32")
    if item_ids.dtype != np.int32:
        raise ValueError("interaction array artifact item_ids_npy must be int32")

    return RatingsData(
        user_ids=user_ids,
        item_ids=item_ids,
        ratings=ratings,
        n_users=int(payload["counts"]["users"]),
        n_items=int(payload["counts"]["rated_items"]),
        name=str(payload["dataset_short_name"]),
        rating_min=float(payload["rating_range"]["min"]),
        rating_max=float(payload["rating_range"]["max"]),
        source_manifest_path=manifest_path,
    )


def load_ratings_data_from_manifest(
    path: Path,
    *,
    mmap_mode: MMapMode | None = "r",
    prefer_interaction_arrays: bool = False,
) -> RatingsData:
    manifest_path = path.resolve()
    payload = load_processed_dataset_manifest(manifest_path)
    artifacts = payload["artifacts"]
    if prefer_interaction_arrays and "interaction_arrays" in artifacts:
        return _load_ratings_data_from_array_artifacts(
            manifest_path=manifest_path,
            payload=payload,
            mmap_mode=mmap_mode,
        )

    artifacts = payload["artifacts"]
    interactions_path = Path(artifacts["interactions"])
    table = pq.read_table(interactions_path, columns=["user_idx", "item_idx", "rating"])

    user_ids = table["user_idx"].to_numpy().astype(np.int32, copy=False)
    item_ids = table["item_idx"].to_numpy().astype(np.int32, copy=False)
    ratings = table["rating"].to_numpy()

    return RatingsData(
        user_ids=user_ids,
        item_ids=item_ids,
        ratings=ratings,
        n_users=int(payload["counts"]["users"]),
        n_items=int(payload["counts"]["rated_items"]),
        name=str(payload["dataset_short_name"]),
        rating_min=float(payload["rating_range"]["min"]),
        rating_max=float(payload["rating_range"]["max"]),
        source_manifest_path=manifest_path,
    )
