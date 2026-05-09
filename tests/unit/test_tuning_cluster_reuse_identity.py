from __future__ import annotations

from pathlib import Path

from recsys_lab.clustering.cache import _cluster_identity_payload, _stable_sha256
from recsys_lab.config.loader import load_yaml_file
from recsys_lab.data.training_index_cache import RatingsDataFingerprint
from recsys_lab.models.biased_mf import BiasedMFConfig
from recsys_lab.tuning import SearchSpaceSpec, build_study_plan, default_cluster_artifact_reuse_spec


def _base_search_space_payload() -> dict[str, object]:
    return {
        "search_space_version": "tuning_search_space_v1",
        "study": {
            "name": "ml1m_cb_svdpp_reuse_contract_test_v1",
            "dataset": "ml1m",
            "split_family": "benchmark_random_v1",
            "model": "cb_svdpp",
            "seed": 1,
        },
        "base_model_config": "configs/models/selected/ml1m/ml1m_cb_svdpp_mini_study_e003.yaml",
        "budget": {"max_candidates": 2},
        "generator": {"type": "grid", "deterministic_order": True},
        "artifact_reuse": {
            "cluster_artifacts": default_cluster_artifact_reuse_spec().model_dump(mode="json")
        },
        "objective": {"primary": {"metric": "validation_rmse"}},
    }


def _cluster_reuse_group_count(payload: dict[str, object]) -> int:
    plan = build_study_plan(SearchSpaceSpec.model_validate(payload))
    return len(plan.artifact_reuse_groups)


def test_cluster_reuse_group_unchanged_when_target_learning_rate_changes() -> None:
    payload = _base_search_space_payload()
    payload["search_space"] = {
        "learning_rate": {
            "type": "float",
            "values": [0.005, 0.01],
            "target_path": "training.learning_rate",
        }
    }

    assert _cluster_reuse_group_count(payload) == 1


def test_cluster_reuse_group_unchanged_when_target_lambda_q_changes() -> None:
    payload = _base_search_space_payload()
    payload["search_space"] = {
        "lambda_q": {
            "type": "float",
            "values": [0.015, 0.04],
            "target_path": "training.lambda_q",
        }
    }

    assert _cluster_reuse_group_count(payload) == 1


def test_cluster_reuse_group_changes_when_induction_learning_rate_changes() -> None:
    payload = _base_search_space_payload()
    payload["search_space"] = {
        "induction_learning_rate": {
            "type": "float",
            "values": [0.005, 0.01],
            "target_path": "clustering.induction.learning_rate",
        }
    }

    assert _cluster_reuse_group_count(payload) == 2


def test_cluster_reuse_group_changes_when_induction_lambda_q_changes() -> None:
    payload = _base_search_space_payload()
    payload["search_space"] = {
        "induction_lambda_q": {
            "type": "float",
            "values": [0.015, 0.04],
            "target_path": "clustering.induction.lambda_q",
        }
    }

    assert _cluster_reuse_group_count(payload) == 2


def test_small_study_search_space_does_not_target_induction_fields() -> None:
    payload = load_yaml_file(
        Path("configs/experiments/tuning/active/ml1m_cb_svdpp_small_study_v1.yaml")
    )

    target_paths = {
        dimension_payload.get("target_path", dimension_name)
        for dimension_name, dimension_payload in payload["search_space"].items()
    }

    assert target_paths == {"clustering.alpha", "training.learning_rate", "training.lambda_q"}
    assert all(not target_path.startswith("clustering.induction.") for target_path in target_paths)
    assert _cluster_reuse_group_count(payload) == 1


def test_cluster_cache_identity_stable_with_same_induction_config() -> None:
    induction_config = BiasedMFConfig(
        latent_dim=64,
        epochs=3,
        learning_rate=0.0075,
        lambda_b=0.025,
        lambda_p=0.025,
        lambda_q=0.025,
        seed=1,
        init_std=0.1,
        dtype="float32",
    )

    first = _cluster_identity(induction_config)
    second = _cluster_identity(induction_config)

    assert first == second
    assert _stable_sha256(first) == _stable_sha256(second)


def test_cluster_cache_identity_changes_when_induction_config_changes() -> None:
    base = BiasedMFConfig(
        latent_dim=64,
        epochs=3,
        learning_rate=0.0075,
        lambda_b=0.025,
        lambda_p=0.025,
        lambda_q=0.025,
        seed=1,
        init_std=0.1,
        dtype="float32",
    )
    changed_learning_rate = BiasedMFConfig(
        latent_dim=64,
        epochs=3,
        learning_rate=0.01,
        lambda_b=0.025,
        lambda_p=0.025,
        lambda_q=0.025,
        seed=1,
        init_std=0.1,
        dtype="float32",
    )
    changed_lambda_q = BiasedMFConfig(
        latent_dim=64,
        epochs=3,
        learning_rate=0.0075,
        lambda_b=0.025,
        lambda_p=0.025,
        lambda_q=0.04,
        seed=1,
        init_std=0.1,
        dtype="float32",
    )

    base_hash = _stable_sha256(_cluster_identity(base))

    assert _stable_sha256(_cluster_identity(changed_learning_rate)) != base_hash
    assert _stable_sha256(_cluster_identity(changed_lambda_q)) != base_hash


def _cluster_identity(induction_config: BiasedMFConfig) -> dict[str, object]:
    return _cluster_identity_payload(
        dataset_short_name="ml1m",
        split_family="benchmark_random_v1",
        split_id="explicit_v1_float32",
        processed_manifest_path=Path("data/processed/ml1m/manifest.json"),
        repo_root=Path("."),
        fingerprint=RatingsDataFingerprint(row_count=900_000, sha256="train-fingerprint"),
        induction_config=induction_config,
        n_users=6040,
        n_items=3706,
        n_user_clusters=80,
        n_item_clusters=80,
        algorithm="kmeans",
        kmeans_n_init=10,
    )
