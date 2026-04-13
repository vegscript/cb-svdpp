import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from recsys_lab.data.processed import RatingsData
from recsys_lab.data.splitters import (
    official_ml100k_inner_validation_split,
    official_ml100k_paper_faithful_split,
    random_split_with_train_coverage,
)


def test_random_split_preserves_train_user_and_item_coverage() -> None:
    data = RatingsData(
        user_ids=np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int32),
        item_ids=np.asarray([0, 1, 2, 0, 2, 3, 1, 2, 3, 0, 1, 3], dtype=np.int32),
        ratings=np.asarray([4.0, 3.5, 5.0, 2.5, 4.0, 3.0, 4.5, 5.0, 2.0, 3.0, 4.0, 4.5]),
        n_users=4,
        n_items=4,
        name="toy",
        rating_min=2.0,
        rating_max=5.0,
        source_manifest_path=Path("toy_manifest.json"),
    )

    split = random_split_with_train_coverage(
        data,
        train_ratio=0.5,
        validation_ratio=0.25,
        seed=7,
    )

    assert len(split.train) + len(split.validation) + len(split.test) == len(data)
    assert set(split.train.user_ids.tolist()) == {0, 1, 2, 3}
    assert set(split.train.item_ids.tolist()) == {0, 1, 2, 3}


def test_official_ml100k_paper_faithful_split_uses_given_fold_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw" / "ml100k" / "ml-100k"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "u1.base").write_text(
        "1\t10\t4\t1000\n"
        "2\t10\t3\t1001\n",
        encoding="latin-1",
    )
    (raw_dir / "u1.test").write_text(
        "1\t20\t5\t1002\n"
        "2\t20\t2\t1003\n",
        encoding="latin-1",
    )

    processed_dir = tmp_path / "data" / "processed" / "ml100k"
    processed_dir.mkdir(parents=True, exist_ok=True)
    interactions_path = processed_dir / "ml100k_interactions.parquet"
    table = pa.table(
        {
            "user_idx": pa.array([0, 0, 1, 1], type=pa.int32()),
            "item_idx": pa.array([0, 1, 0, 1], type=pa.int32()),
            "rating": pa.array([4.0, 5.0, 3.0, 2.0], type=pa.float32()),
            "timestamp": pa.array([1000, 1002, 1001, 1003], type=pa.int64()),
            "raw_user_id": pa.array([1, 1, 2, 2], type=pa.int32()),
            "raw_item_id": pa.array([10, 20, 10, 20], type=pa.int32()),
        }
    )
    pq.write_table(table, interactions_path)

    manifest_path = processed_dir / "ml100k_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_short_name": "ml100k",
                "dataset_name": "MovieLens 100K",
                "split_family": "benchmark_random_v1",
                "preprocessing_family": "explicit_v1",
                "source": {
                    "raw_dir": str(raw_dir),
                    "format_family": "legacy_100k",
                },
                "counts": {"users": 2, "rated_items": 2},
                "rating_range": {"min": 2.0, "max": 5.0},
                "artifacts": {"interactions": str(interactions_path)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    data = RatingsData(
        user_ids=np.asarray([0, 0, 1, 1], dtype=np.int32),
        item_ids=np.asarray([0, 1, 0, 1], dtype=np.int32),
        ratings=np.asarray([4.0, 5.0, 3.0, 2.0], dtype=np.float32),
        n_users=2,
        n_items=2,
        name="ml100k",
        rating_min=2.0,
        rating_max=5.0,
        source_manifest_path=manifest_path,
    )

    split = official_ml100k_paper_faithful_split(
        data,
        processed_manifest_path=manifest_path,
        fold_index=1,
    )

    assert split.validation is None
    assert split.train.ratings.tolist() == [4.0, 3.0]
    assert split.test.ratings.tolist() == [5.0, 2.0]


def test_official_ml100k_inner_validation_split_uses_only_outer_train_for_validation(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "data" / "raw" / "ml100k" / "ml-100k"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "u1.base").write_text(
        "1\t10\t4\t1000\n"
        "1\t20\t5\t1002\n"
        "2\t10\t3\t1001\n"
        "2\t20\t2\t1003\n",
        encoding="latin-1",
    )
    (raw_dir / "u1.test").write_text(
        "3\t30\t4\t1004\n"
        "4\t40\t3\t1005\n",
        encoding="latin-1",
    )

    processed_dir = tmp_path / "data" / "processed" / "ml100k"
    processed_dir.mkdir(parents=True, exist_ok=True)
    interactions_path = processed_dir / "ml100k_interactions.parquet"
    table = pa.table(
        {
            "user_idx": pa.array([0, 0, 1, 1, 2, 3], type=pa.int32()),
            "item_idx": pa.array([0, 1, 0, 1, 2, 3], type=pa.int32()),
            "rating": pa.array([4.0, 5.0, 3.0, 2.0, 4.0, 3.0], type=pa.float32()),
            "timestamp": pa.array([1000, 1002, 1001, 1003, 1004, 1005], type=pa.int64()),
            "raw_user_id": pa.array([1, 1, 2, 2, 3, 4], type=pa.int32()),
            "raw_item_id": pa.array([10, 20, 10, 20, 30, 40], type=pa.int32()),
        }
    )
    pq.write_table(table, interactions_path)

    manifest_path = processed_dir / "ml100k_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_short_name": "ml100k",
                "dataset_name": "MovieLens 100K",
                "split_family": "benchmark_random_v1",
                "preprocessing_family": "explicit_v1",
                "source": {
                    "raw_dir": str(raw_dir),
                    "format_family": "legacy_100k",
                },
                "counts": {"users": 4, "rated_items": 4},
                "rating_range": {"min": 2.0, "max": 5.0},
                "artifacts": {"interactions": str(interactions_path)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    data = RatingsData(
        user_ids=np.asarray([0, 0, 1, 1, 2, 3], dtype=np.int32),
        item_ids=np.asarray([0, 1, 0, 1, 2, 3], dtype=np.int32),
        ratings=np.asarray([4.0, 5.0, 3.0, 2.0, 4.0, 3.0], dtype=np.float32),
        n_users=4,
        n_items=4,
        name="ml100k",
        rating_min=2.0,
        rating_max=5.0,
        source_manifest_path=manifest_path,
    )

    split = official_ml100k_inner_validation_split(
        data,
        processed_manifest_path=manifest_path,
        fold_index=1,
        validation_ratio=0.25,
        inner_seed=17,
    )

    assert split.validation is not None
    assert len(split.train) + len(split.validation) == 4
    assert len(split.test) == 2
    assert split.test.ratings.tolist() == [4.0, 3.0]
