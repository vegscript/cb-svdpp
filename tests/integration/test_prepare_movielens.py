import json
from pathlib import Path

import pyarrow.parquet as pq

from recsys_lab.data.movielens import prepare_movielens_explicit_dataset, validate_movielens_directory


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_prepare_movielens_explicit_dataset_writes_expected_artifacts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"

    _write_text(
        raw_dir / "ratings.csv",
        "userId,movieId,rating,timestamp\n"
        "10,2,4.0,1000\n"
        "10,5,3.5,1001\n"
        "11,2,5.0,1002\n",
    )
    _write_text(
        raw_dir / "movies.csv",
        "movieId,title,genres\n"
        "2,Movie Two,Drama\n"
        "5,Movie Five,Comedy\n"
        "8,Movie Eight,Action\n",
    )
    _write_text(
        raw_dir / "links.csv",
        "movieId,imdbId,tmdbId\n"
        "2,0002,200\n"
        "5,0005,500\n"
        "8,0008,800\n",
    )
    _write_text(
        raw_dir / "tags.csv",
        "userId,movieId,tag,timestamp\n"
        "10,2,good,1100\n"
        "12,8,unrated-user-tag,1101\n",
    )

    summary = validate_movielens_directory(raw_dir)
    assert summary["counts"]["ratings"] == 3
    assert summary["counts"]["movies"] == 3

    artifacts = prepare_movielens_explicit_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        dataset_name="Toy MovieLens",
        dataset_short_name="ml_latest_small",
        split_family="benchmark_random_v1",
        dtype="float32",
    )

    assert artifacts.manifest_path.exists()
    assert artifacts.interactions_path.exists()
    assert artifacts.user_mapping_path.exists()
    assert artifacts.item_mapping_path.exists()

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["interactions"] == 3
    assert manifest["counts"]["users"] == 2
    assert manifest["counts"]["rated_items"] == 2

    interactions = pq.read_table(artifacts.interactions_path).to_pydict()
    assert interactions["user_idx"] == [0, 0, 1]
    assert interactions["item_idx"] == [0, 1, 0]
    assert interactions["raw_user_id"] == [10, 10, 11]
    assert interactions["raw_item_id"] == [2, 5, 2]

    movies = pq.read_table(artifacts.movies_path).to_pydict()
    assert movies["item_idx"] == [0, 1, None]

    tags = pq.read_table(artifacts.tags_path).to_pydict()
    assert tags["user_idx"] == [0, None]
