import json
from pathlib import Path

import numpy as np
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
    assert summary["format_family"] == "modern_csv"
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
    assert artifacts.user_ids_array_path.exists()
    assert artifacts.item_ids_array_path.exists()
    assert artifacts.ratings_array_path.exists()
    assert artifacts.user_mapping_path.exists()
    assert artifacts.item_mapping_path.exists()

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["interactions"] == 3
    assert manifest["counts"]["users"] == 2
    assert manifest["counts"]["rated_items"] == 2
    assert "interaction_arrays" in manifest["artifacts"]

    interactions = pq.read_table(artifacts.interactions_path).to_pydict()
    assert interactions["user_idx"] == [0, 0, 1]
    assert interactions["item_idx"] == [0, 1, 0]
    assert interactions["raw_user_id"] == [10, 10, 11]
    assert interactions["raw_item_id"] == [2, 5, 2]
    np.testing.assert_array_equal(np.load(artifacts.user_ids_array_path), np.asarray([0, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.load(artifacts.item_ids_array_path), np.asarray([0, 1, 0], dtype=np.int32))
    np.testing.assert_allclose(np.load(artifacts.ratings_array_path), np.asarray([4.0, 3.5, 5.0], dtype=np.float32))

    movies = pq.read_table(artifacts.movies_path).to_pydict()
    assert movies["item_idx"] == [0, 1, None]

    tags = pq.read_table(artifacts.tags_path).to_pydict()
    assert tags["user_idx"] == [0, None]


def test_prepare_legacy_movielens_100k_layout_writes_expected_artifacts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_ml100k"
    output_dir = tmp_path / "processed_ml100k"

    _write_text(
        raw_dir / "u.data",
        "1\t10\t4\t874965758\n"
        "1\t20\t5\t874965759\n"
        "2\t10\t3\t874965760\n",
    )
    _write_text(
        raw_dir / "u.genre",
        "unknown|0\n"
        "Action|1\n"
        "Adventure|2\n"
        "Animation|3\n"
        "Children's|4\n"
        "Comedy|5\n"
        "Crime|6\n"
        "Documentary|7\n"
        "Drama|8\n"
        "Fantasy|9\n"
        "Film-Noir|10\n"
        "Horror|11\n"
        "Musical|12\n"
        "Mystery|13\n"
        "Romance|14\n"
        "Sci-Fi|15\n"
        "Thriller|16\n"
        "War|17\n"
        "Western|18\n",
    )
    _write_text(
        raw_dir / "u.item",
        "10|Toy Movie (1995)|01-Jan-1995||http://example.com/10|0|1|0|0|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0\n"
        "20|Serious Movie (1994)|01-Jan-1994||http://example.com/20|0|0|0|0|0|1|0|0|1|0|0|0|0|0|0|0|0|0|0\n"
        "30|Unrated Movie (1993)|01-Jan-1993||http://example.com/30|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0\n",
    )
    _write_text(
        raw_dir / "u1.base",
        "1\t10\t4\t874965758\n"
        "2\t10\t3\t874965760\n",
    )
    _write_text(
        raw_dir / "u1.test",
        "1\t20\t5\t874965759\n",
    )
    for fold_index in (2, 3, 4, 5):
        _write_text(raw_dir / f"u{fold_index}.base", "")
        _write_text(raw_dir / f"u{fold_index}.test", "")

    summary = validate_movielens_directory(raw_dir)
    assert summary["format_family"] == "legacy_100k"
    assert summary["counts"]["ratings"] == 3
    assert summary["counts"]["movies"] == 3
    assert summary["counts"]["tags"] == 0

    artifacts = prepare_movielens_explicit_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        dataset_name="MovieLens 100K",
        dataset_short_name="ml100k",
        split_family="benchmark_random_v1",
        format_family="legacy_100k",
        dtype="float32",
    )

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["format_family"] == "legacy_100k"
    assert manifest["counts"]["interactions"] == 3
    assert manifest["counts"]["users"] == 2
    assert manifest["counts"]["rated_items"] == 2
    assert manifest["counts"]["links"] == 0
    assert manifest["counts"]["tags"] == 0
    assert "interaction_arrays" in manifest["artifacts"]

    movies = pq.read_table(artifacts.movies_path).to_pydict()
    assert movies["item_idx"] == [0, 1, None]
    assert movies["genres"][0] == "Action|Drama"
    assert movies["genres"][1] == "Comedy|Drama"

    np.testing.assert_array_equal(np.load(artifacts.user_ids_array_path), np.asarray([0, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.load(artifacts.item_ids_array_path), np.asarray([0, 1, 0], dtype=np.int32))
    np.testing.assert_allclose(np.load(artifacts.ratings_array_path), np.asarray([4.0, 5.0, 3.0], dtype=np.float32))

    links = pq.read_table(artifacts.links_path).to_pydict()
    assert links["raw_item_id"] == []

    tags = pq.read_table(artifacts.tags_path).to_pydict()
    assert tags["raw_user_id"] == []
    assert manifest["artifacts"]["official_ml100k_splits"]["version"] == "paper_faithful_ml100k_v1"
    u1_payload = manifest["artifacts"]["official_ml100k_splits"]["folds"]["u1"]
    assert np.load(u1_payload["train_row_indices_npy"]).tolist() == [0, 2]
    assert np.load(u1_payload["test_row_indices_npy"]).tolist() == [1]

def test_prepare_legacy_movielens_1m_layout_writes_expected_artifacts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_ml1m"
    output_dir = tmp_path / "processed_ml1m"

    _write_text(
        raw_dir / "ratings.dat",
        "1::10::4::978300760\n"
        "1::20::5::978302109\n"
        "2::10::3::978301968\n",
    )
    _write_text(
        raw_dir / "movies.dat",
        "10::Toy Movie (1995)::Animation|Children's\n"
        "20::Serious Movie (1994)::Drama\n"
        "30::Unrated Movie (1993)::Comedy\n",
    )
    _write_text(
        raw_dir / "users.dat",
        "1::F::1::10::48067\n"
        "2::M::56::16::70072\n",
    )

    summary = validate_movielens_directory(raw_dir)
    assert summary["format_family"] == "legacy_1m"
    assert summary["counts"]["ratings"] == 3
    assert summary["counts"]["movies"] == 3
    assert summary["counts"]["users"] == 2
    assert summary["counts"]["tags"] == 0

    artifacts = prepare_movielens_explicit_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        dataset_name="MovieLens 1M",
        dataset_short_name="ml1m",
        split_family="benchmark_random_v1",
        format_family="legacy_1m",
        dtype="float32",
    )

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["format_family"] == "legacy_1m"
    assert manifest["counts"]["interactions"] == 3
    assert manifest["counts"]["users"] == 2
    assert manifest["counts"]["rated_items"] == 2
    assert manifest["counts"]["catalog_items"] == 3
    assert manifest["counts"]["links"] == 0
    assert manifest["counts"]["tags"] == 0
    assert "interaction_arrays" in manifest["artifacts"]
    assert "official_ml100k_splits" not in manifest["artifacts"]

    interactions = pq.read_table(artifacts.interactions_path).to_pydict()
    assert interactions["raw_user_id"] == [1, 1, 2]
    assert interactions["raw_item_id"] == [10, 20, 10]

    movies = pq.read_table(artifacts.movies_path).to_pydict()
    assert movies["item_idx"] == [0, 1, None]
    assert movies["genres"] == ["Animation|Children's", "Drama", "Comedy"]
    np.testing.assert_array_equal(np.load(artifacts.user_ids_array_path), np.asarray([0, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.load(artifacts.item_ids_array_path), np.asarray([0, 1, 0], dtype=np.int32))
    np.testing.assert_allclose(np.load(artifacts.ratings_array_path), np.asarray([4.0, 5.0, 3.0], dtype=np.float32))

    links = pq.read_table(artifacts.links_path).to_pydict()
    assert links["raw_item_id"] == []

    tags = pq.read_table(artifacts.tags_path).to_pydict()
    assert tags["raw_user_id"] == []
