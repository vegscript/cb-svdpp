import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from recsys_lab.data.processed import (
    build_interaction_array_manifest_payload,
    load_ratings_data_from_manifest,
    materialize_interaction_array_artifacts_from_manifest,
)


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")


def _base_manifest(*, interactions_path: Path) -> dict[str, object]:
    return {
        "dataset_short_name": "ml_latest_small",
        "dataset_name": "Synthetic MovieLens Small",
        "split_family": "benchmark_random_v1",
        "preprocessing_family": "explicit_v1",
        "counts": {
            "users": 3,
            "rated_items": 4,
        },
        "rating_range": {
            "min": 2.0,
            "max": 5.0,
        },
        "artifacts": {
            "interactions": str(interactions_path),
        },
    }


def test_load_ratings_data_from_manifest_prefers_memmap_arrays_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interactions_path = tmp_path / "toy_interactions.parquet"
    pq.write_table(
        pa.table(
            {
                "user_idx": pa.array([0, 0, 1], type=pa.int32()),
                "item_idx": pa.array([0, 1, 0], type=pa.int32()),
                "rating": pa.array([4.0, 3.5, 5.0], type=pa.float32()),
            }
        ),
        interactions_path,
    )

    seed_manifest_path = tmp_path / "seed_manifest.json"
    _write_manifest(seed_manifest_path, _base_manifest(interactions_path=interactions_path))
    arrays = materialize_interaction_array_artifacts_from_manifest(
        path=seed_manifest_path,
        output_dir=tmp_path / "arrays",
        prefix="toy",
    )

    manifest_path = tmp_path / "manifest.json"
    payload = _base_manifest(interactions_path=interactions_path)
    payload["artifacts"]["interaction_arrays"] = build_interaction_array_manifest_payload(arrays)
    _write_manifest(manifest_path, payload)

    def _fail_read_table(*args: object, **kwargs: object) -> None:
        raise AssertionError("parquet fallback should not be used when interaction arrays are present")

    monkeypatch.setattr("recsys_lab.data.processed.pq.read_table", _fail_read_table)
    ratings_data = load_ratings_data_from_manifest(
        manifest_path,
        prefer_interaction_arrays=True,
    )

    assert isinstance(ratings_data.user_ids, np.memmap)
    assert isinstance(ratings_data.item_ids, np.memmap)
    assert isinstance(ratings_data.ratings, np.memmap)
    np.testing.assert_array_equal(ratings_data.user_ids, np.asarray([0, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(ratings_data.item_ids, np.asarray([0, 1, 0], dtype=np.int32))
    np.testing.assert_allclose(ratings_data.ratings, np.asarray([4.0, 3.5, 5.0], dtype=np.float32))


def test_load_ratings_data_from_manifest_falls_back_to_parquet_without_array_sidecars(
    tmp_path: Path,
) -> None:
    interactions_path = tmp_path / "toy_interactions.parquet"
    pq.write_table(
        pa.table(
            {
                "user_idx": pa.array([0, 0, 1], type=pa.int32()),
                "item_idx": pa.array([0, 1, 0], type=pa.int32()),
                "rating": pa.array([4.0, 3.5, 5.0], type=pa.float32()),
            }
        ),
        interactions_path,
    )

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, _base_manifest(interactions_path=interactions_path))
    ratings_data = load_ratings_data_from_manifest(manifest_path)

    assert not isinstance(ratings_data.user_ids, np.memmap)
    np.testing.assert_array_equal(ratings_data.user_ids, np.asarray([0, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(ratings_data.item_ids, np.asarray([0, 1, 0], dtype=np.int32))
    np.testing.assert_allclose(ratings_data.ratings, np.asarray([4.0, 3.5, 5.0], dtype=np.float32))


def test_load_ratings_data_from_manifest_does_not_use_array_sidecars_without_opt_in(
    tmp_path: Path,
) -> None:
    interactions_path = tmp_path / "toy_interactions.parquet"
    pq.write_table(
        pa.table(
            {
                "user_idx": pa.array([0, 0, 1], type=pa.int32()),
                "item_idx": pa.array([0, 1, 0], type=pa.int32()),
                "rating": pa.array([4.0, 3.5, 5.0], type=pa.float32()),
            }
        ),
        interactions_path,
    )

    seed_manifest_path = tmp_path / "seed_manifest.json"
    _write_manifest(seed_manifest_path, _base_manifest(interactions_path=interactions_path))
    arrays = materialize_interaction_array_artifacts_from_manifest(
        path=seed_manifest_path,
        output_dir=tmp_path / "arrays",
        prefix="toy",
    )

    manifest_path = tmp_path / "manifest.json"
    payload = _base_manifest(interactions_path=interactions_path)
    payload["artifacts"]["interaction_arrays"] = build_interaction_array_manifest_payload(arrays)
    _write_manifest(manifest_path, payload)

    ratings_data = load_ratings_data_from_manifest(manifest_path)

    assert not isinstance(ratings_data.user_ids, np.memmap)
    np.testing.assert_array_equal(ratings_data.user_ids, np.asarray([0, 0, 1], dtype=np.int32))


def test_materialize_interaction_array_artifacts_from_manifest_writes_expected_arrays(
    tmp_path: Path,
) -> None:
    interactions_path = tmp_path / "toy_interactions.parquet"
    pq.write_table(
        pa.table(
            {
                "user_idx": pa.array([0, 0, 1], type=pa.int32()),
                "item_idx": pa.array([0, 1, 0], type=pa.int32()),
                "rating": pa.array([4.0, 3.5, 5.0], type=pa.float32()),
            }
        ),
        interactions_path,
    )

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, _base_manifest(interactions_path=interactions_path))

    arrays = materialize_interaction_array_artifacts_from_manifest(
        path=manifest_path,
        output_dir=tmp_path / "arrays",
        prefix="toy",
    )

    np.testing.assert_array_equal(np.load(arrays.user_ids_path), np.asarray([0, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.load(arrays.item_ids_path), np.asarray([0, 1, 0], dtype=np.int32))
    np.testing.assert_allclose(np.load(arrays.ratings_path), np.asarray([4.0, 3.5, 5.0], dtype=np.float32))


def test_ratings_data_subset_composes_row_selection_without_copying_base_arrays(
    tmp_path: Path,
) -> None:
    interactions_path = tmp_path / "toy_interactions.parquet"
    pq.write_table(
        pa.table(
            {
                "user_idx": pa.array([0, 0, 1, 1], type=pa.int32()),
                "item_idx": pa.array([0, 1, 0, 2], type=pa.int32()),
                "rating": pa.array([4.0, 3.5, 5.0, 2.5], type=pa.float32()),
            }
        ),
        interactions_path,
    )

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, _base_manifest(interactions_path=interactions_path))
    ratings_data = load_ratings_data_from_manifest(manifest_path)

    subset = ratings_data.subset(np.asarray([3, 1], dtype=np.int64), name="toy:subset")
    nested_subset = subset.subset(np.asarray([1], dtype=np.int64), name="toy:nested")

    assert subset.uses_row_selection is True
    assert subset.base_user_ids is ratings_data.base_user_ids
    assert subset.base_item_ids is ratings_data.base_item_ids
    assert subset.base_ratings is ratings_data.base_ratings
    np.testing.assert_array_equal(subset.user_ids, np.asarray([1, 0], dtype=np.int32))
    np.testing.assert_array_equal(subset.item_ids, np.asarray([2, 1], dtype=np.int32))
    np.testing.assert_allclose(subset.ratings, np.asarray([2.5, 3.5], dtype=np.float32))
    np.testing.assert_array_equal(nested_subset.user_ids, np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(nested_subset.item_ids, np.asarray([1], dtype=np.int32))
    np.testing.assert_allclose(nested_subset.ratings, np.asarray([3.5], dtype=np.float32))
