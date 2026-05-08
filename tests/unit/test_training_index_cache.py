import json
from pathlib import Path

import numpy as np

from recsys_lab.data.processed import RatingsData
from recsys_lab.data.training_index_cache import (
    load_or_build_user_explicit_feedback_index,
    load_or_build_user_history_index,
)


def _toy_ratings_data(tmp_path: Path) -> tuple[RatingsData, Path]:
    manifest_path = tmp_path / "data" / "processed" / "toy_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8", newline="\n")
    return (
        RatingsData(
            user_ids=np.asarray([0, 0, 1, 1, 1], dtype=np.int32),
            item_ids=np.asarray([0, 2, 1, 2, 3], dtype=np.int32),
            ratings=np.asarray([5.0, 4.0, 3.5, 2.5, 4.5], dtype=np.float32),
            n_users=2,
            n_items=4,
            name="toy",
            rating_min=2.5,
            rating_max=5.0,
            source_manifest_path=manifest_path,
        ),
        manifest_path,
    )


def test_user_history_index_cache_hits_on_second_load(tmp_path: Path) -> None:
    data, manifest_path = _toy_ratings_data(tmp_path)
    runtime_config_payload = {"runtime": {"cache_root": "artifacts/local"}}

    miss_result = load_or_build_user_history_index(
        data=data,
        dtype="float32",
        dataset_short_name="ml_latest_small",
        split_family="benchmark_random_v1",
        split_id="benchmark_random_v1_tr080_va010_s001",
        processed_manifest_path=manifest_path,
        repo_root=tmp_path,
        runtime_config_payload=runtime_config_payload,
        use_cache=True,
    )
    hit_result = load_or_build_user_history_index(
        data=data,
        dtype="float32",
        dataset_short_name="ml_latest_small",
        split_family="benchmark_random_v1",
        split_id="benchmark_random_v1_tr080_va010_s001",
        processed_manifest_path=manifest_path,
        repo_root=tmp_path,
        runtime_config_payload=runtime_config_payload,
        use_cache=True,
    )

    assert miss_result.metadata.cache_status == "miss"
    assert hit_result.metadata.cache_status == "hit"
    assert hit_result.metadata.cache_manifest_path.exists()
    assert "ti" in hit_result.metadata.cache_manifest_path.parts
    assert "training_indices" not in hit_result.metadata.cache_manifest_path.parts
    assert np.array_equal(miss_result.index.indptr, hit_result.index.indptr)
    assert np.array_equal(miss_result.index.item_indices, hit_result.index.item_indices)
    assert np.array_equal(miss_result.index.counts, hit_result.index.counts)
    assert np.allclose(miss_result.index.norms, hit_result.index.norms)

    manifest = json.loads(hit_result.metadata.cache_manifest_path.read_text(encoding="utf-8"))
    assert manifest["layout"] == {
        "layout_version": "history_data_layout_v1",
        "index_dtype": "int32",
        "value_dtype": "float32",
    }
    assert all(not Path(path_ref).is_absolute() for path_ref in manifest["artifacts"].values())


def test_training_index_cache_manifest_includes_layout_version(tmp_path: Path) -> None:
    data, manifest_path = _toy_ratings_data(tmp_path)
    result = load_or_build_user_history_index(
        data=data,
        dtype="float32",
        dataset_short_name="ml_latest_small",
        split_family="benchmark_random_v1",
        split_id="benchmark_random_v1_tr080_va010_s001",
        processed_manifest_path=manifest_path,
        repo_root=tmp_path,
        runtime_config_payload={"runtime": {"cache_root": "artifacts/local"}},
        use_cache=True,
    )

    manifest = json.loads(result.metadata.cache_manifest_path.read_text(encoding="utf-8"))

    assert manifest["layout"]["layout_version"] == "history_data_layout_v1"
    assert manifest["layout"]["index_dtype"] == "int32"
    assert manifest["layout"]["value_dtype"] == "float32"


def _assert_user_history_index_cache_rebuilds_legacy_layout_manifest(tmp_path: Path) -> None:
    data, manifest_path = _toy_ratings_data(tmp_path)
    runtime_config_payload = {"runtime": {"cache_root": "artifacts/local"}}
    kwargs = {
        "data": data,
        "dtype": "float32",
        "dataset_short_name": "ml_latest_small",
        "split_family": "benchmark_random_v1",
        "split_id": "benchmark_random_v1_tr080_va010_s001",
        "processed_manifest_path": manifest_path,
        "repo_root": tmp_path,
        "runtime_config_payload": runtime_config_payload,
        "use_cache": True,
        "mmap_mode": None,
    }

    initial_result = load_or_build_user_history_index(**kwargs)
    assert initial_result.metadata.cache_status == "miss"

    cache_manifest = json.loads(initial_result.metadata.cache_manifest_path.read_text(encoding="utf-8"))
    cache_manifest.pop("layout")
    initial_result.metadata.cache_manifest_path.write_text(json.dumps(cache_manifest), encoding="utf-8")

    rebuilt_result = load_or_build_user_history_index(**kwargs)

    assert rebuilt_result.metadata.cache_status == "miss"
    rebuilt_manifest = json.loads(rebuilt_result.metadata.cache_manifest_path.read_text(encoding="utf-8"))
    assert rebuilt_manifest["layout"]["layout_version"] == "history_data_layout_v1"


def test_user_history_index_cache_rebuilds_legacy_layout_manifest(tmp_path: Path) -> None:
    _assert_user_history_index_cache_rebuilds_legacy_layout_manifest(tmp_path)


def test_training_index_cache_rebuilds_or_rejects_old_layout_cache(tmp_path: Path) -> None:
    _assert_user_history_index_cache_rebuilds_legacy_layout_manifest(tmp_path)


def test_explicit_feedback_index_cache_can_be_disabled(tmp_path: Path) -> None:
    data, manifest_path = _toy_ratings_data(tmp_path)
    runtime_config_payload = {"runtime": {"cache_root": "artifacts/local"}}

    result = load_or_build_user_explicit_feedback_index(
        data=data,
        dtype="float32",
        dataset_short_name="ml_latest_small",
        split_family="benchmark_random_v1",
        split_id="benchmark_random_v1_tr080_va010_s001",
        processed_manifest_path=manifest_path,
        repo_root=tmp_path,
        runtime_config_payload=runtime_config_payload,
        use_cache=False,
    )

    assert result.metadata.cache_status == "disabled"
    assert not result.metadata.cache_manifest_path.exists()
    assert result.index.indptr.shape == (3,)
    assert result.index.item_indices.shape == result.index.ratings.shape
