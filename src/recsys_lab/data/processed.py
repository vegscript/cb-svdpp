from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True, slots=True)
class RatingsData:
    user_ids: np.ndarray
    item_ids: np.ndarray
    ratings: np.ndarray
    n_users: int
    n_items: int
    name: str
    rating_min: float
    rating_max: float
    source_manifest_path: Path

    def __len__(self) -> int:
        return int(self.ratings.shape[0])

    def subset(self, indices: np.ndarray, *, name: str) -> "RatingsData":
        idx = np.asarray(indices, dtype=np.int64)
        return RatingsData(
            user_ids=self.user_ids[idx],
            item_ids=self.item_ids[idx],
            ratings=self.ratings[idx],
            n_users=self.n_users,
            n_items=self.n_items,
            name=name,
            rating_min=self.rating_min,
            rating_max=self.rating_max,
            source_manifest_path=self.source_manifest_path,
        )


def load_processed_dataset_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object at {path}")
    return payload


def load_ratings_data_from_manifest(path: Path) -> RatingsData:
    manifest_path = path.resolve()
    payload = load_processed_dataset_manifest(manifest_path)
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
