from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

import recsys_lab.experiments.cb_asvdpp as cb_asvdpp_experiment
import recsys_lab.experiments.cb_svdpp as cb_svdpp_experiment
import recsys_lab.experiments.unified_runner as unified_runner_module
import recsys_lab.models.cb_asvdpp as cb_asvdpp_module
import recsys_lab.models.cb_svdpp as cb_svdpp_module
from recsys_lab.clustering.latent_kmeans import ClusterArtifacts
from recsys_lab.data.histories import (
    build_user_cluster_count_index,
    build_user_explicit_feedback_index,
    build_user_history_index,
)
from recsys_lab.data.processed import RatingsData
from recsys_lab.experiments.common import SplitConfig
from recsys_lab.experiments.unified_runner import build_experiment_services, run_unified_experiment
from recsys_lab.models.cb_asvdpp import CBASVDppConfig, CBASVDppRecommender
from recsys_lab.models.cb_svdpp import CBSVDppConfig, CBSVDppRecommender
from recsys_lab.models.registry import (
    MODEL_REGISTRY,
    CBASVDppAdapter,
    CBSVDppAdapter,
    FitArtifacts,
    build_cb_semantics,
    validate_model_config_payload,
    validated_model_config_payload_with_training_overrides,
)

LEGACY_EXPERIMENT_WRAPPERS = {
    "biased_mf.py": "run_biased_mf_experiment",
    "svdpp.py": "run_svdpp_experiment",
    "asymmetric_svd.py": "run_asymmetric_svd_experiment",
    "asvdpp.py": "run_asvdpp_experiment",
    "cb_svdpp.py": "run_cb_svdpp_experiment",
    "cb_asvdpp.py": "run_cb_asvdpp_experiment",
}
LEGACY_WRAPPER_COMMENT_LINES = [
    "# Legacy compatibility wrapper only.",
    "# Do not add experiment lifecycle logic here.",
    "# All execution must delegate to run_unified_experiment.",
]
FORBIDDEN_LEGACY_LIFECYCLE_IMPORT_NAMES = {
    "PeakMemoryMonitor",
    "build_experiment_services",
    "git_snapshot",
    "load_or_build_split_cache",
    "load_ratings_data_from_manifest",
    "rmse",
    "write_json",
}
FORBIDDEN_LEGACY_LIFECYCLE_IMPORT_MODULES = {
    "recsys_lab.data.splitters",
    "recsys_lab.data.processed",
    "recsys_lab.experiments.performance",
    "recsys_lab.experiments.split_cache",
    "recsys_lab.metrics",
}
PRODUCTIVE_MODEL_CONFIG_PATHS = [
    *Path("src/recsys_lab/experiments").rglob("*.py"),
    *Path("src/recsys_lab/cli").rglob("*.py"),
]
MODEL_CONFIG_CLASS_NAMES = {
    "ASVDppConfig",
    "AsymmetricSVDConfig",
    "BiasedMFConfig",
    "CBASVDppConfig",
    "CBSVDppConfig",
    "SVDppConfig",
}
MODEL_CONFIG_FALLBACK_MARKERS = (
    ".setdefault(",
    "base_model_config_payload.get(",
    "benchmark_model_config.get(",
    "effective_model_config.get(",
    "model_config_payload.get(",
)


def test_model_registry_declares_expected_artifact_requirements() -> None:
    assert set(MODEL_REGISTRY) == {
        "biased_mf",
        "svdpp",
        "asymmetric_svd",
        "asvdpp",
        "cb_svdpp",
        "cb_asvdpp",
    }
    for model_name, adapter in MODEL_REGISTRY.items():
        assert adapter.name == model_name

    assert MODEL_REGISTRY["biased_mf"].requirements.artifact_names() == []
    assert MODEL_REGISTRY["svdpp"].requirements.artifact_names() == ["user_history_index"]
    assert MODEL_REGISTRY["asymmetric_svd"].requirements.artifact_names() == [
        "user_history_index",
        "explicit_feedback_index",
    ]
    assert MODEL_REGISTRY["asvdpp"].requirements.artifact_names() == [
        "user_history_index",
        "explicit_feedback_index",
    ]

    cb_svdpp_artifacts = MODEL_REGISTRY["cb_svdpp"].requirements.artifact_names()
    cb_asvdpp_artifacts = MODEL_REGISTRY["cb_asvdpp"].requirements.artifact_names()
    assert cb_svdpp_artifacts == [
        "user_history_index",
        "cluster_artifacts",
        "user_cluster_history_index",
    ]
    assert cb_asvdpp_artifacts == [
        "user_history_index",
        "explicit_feedback_index",
        "cluster_artifacts",
        "user_cluster_history_index",
    ]


def test_unknown_model_config_key_fails_validation() -> None:
    payload = yaml.safe_load(Path("configs/models/svdpp.yaml").read_text(encoding="utf-8"))
    payload["training"]["unknown_field"] = 1

    with pytest.raises(Exception, match="unknown_field"):
        validate_model_config_payload(payload)


def test_wrong_model_config_name_fails_validation() -> None:
    payload = yaml.safe_load(Path("configs/models/svdpp.yaml").read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="expected model config for 'biased_mf', got 'svdpp'"):
        validate_model_config_payload(payload, expected_model_name="biased_mf")


def test_wrong_model_config_family_fails_validation() -> None:
    payload = yaml.safe_load(Path("configs/models/cb_svdpp.yaml").read_text(encoding="utf-8"))
    payload["model"]["family"] = "implicit_factorization"

    with pytest.raises(ValueError, match="model.family must be 'clustering_based_factorization'"):
        validate_model_config_payload(payload)


@pytest.mark.parametrize(
    ("config_path", "field_name"),
    [
        ("configs/models/biased_mf.yaml", "training_backend"),
        ("configs/models/svdpp.yaml", "training_backend"),
        ("configs/models/cb_svdpp.yaml", "init_std"),
        ("configs/models/cb_asvdpp.yaml", "lambda_xC"),
        ("configs/models/cb_asvdpp.yaml", "lambda_yC"),
    ],
)
def test_missing_required_model_config_fields_fail_validation(config_path: str, field_name: str) -> None:
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    del payload["training"][field_name]

    with pytest.raises(Exception, match=field_name):
        validate_model_config_payload(payload)


def test_misspelled_cb_regularization_field_fails_validation() -> None:
    payload = yaml.safe_load(Path("configs/models/cb_svdpp.yaml").read_text(encoding="utf-8"))
    payload["training"]["lambda_yc"] = payload["training"].pop("lambda_yC")

    with pytest.raises(Exception) as exc_info:
        validate_model_config_payload(payload)

    message = str(exc_info.value)
    assert "lambda_yC" in message
    assert "lambda_yc" in message


@pytest.mark.parametrize("alpha", [-0.01, 1.01])
def test_cb_alpha_outside_unit_interval_fails_validation(alpha: float) -> None:
    payload = yaml.safe_load(Path("configs/models/cb_svdpp.yaml").read_text(encoding="utf-8"))
    payload["clustering"]["alpha"] = alpha

    with pytest.raises(Exception) as exc_info:
        validate_model_config_payload(payload)

    assert "alpha" in str(exc_info.value)


def test_validated_training_overrides_do_not_fill_missing_source_fields() -> None:
    payload = yaml.safe_load(Path("configs/models/cb_asvdpp.yaml").read_text(encoding="utf-8"))
    del payload["training"]["lambda_xC"]

    with pytest.raises(Exception, match="lambda_xC"):
        validated_model_config_payload_with_training_overrides(
            payload,
            expected_model_name="cb_asvdpp",
            training_overrides={"epochs": 1},
        )


def test_validated_training_overrides_revalidate_added_fields() -> None:
    payload = yaml.safe_load(Path("configs/models/svdpp.yaml").read_text(encoding="utf-8"))

    with pytest.raises(Exception, match="training_backnd"):
        validated_model_config_payload_with_training_overrides(
            payload,
            expected_model_name="svdpp",
            training_overrides={"training_backnd": "numba"},
        )


def test_productive_paths_do_not_directly_construct_model_configs_from_yaml() -> None:
    violations: list[str] = []
    for source_path in PRODUCTIVE_MODEL_CONFIG_PATHS:
        if "__pycache__" in source_path.parts:
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in MODEL_CONFIG_CLASS_NAMES:
                    violations.append(f"{source_path}:{node.lineno}:{node.func.id}")

    assert violations == []


def test_productive_paths_do_not_use_model_config_default_fallbacks() -> None:
    violations: list[str] = []
    for source_path in PRODUCTIVE_MODEL_CONFIG_PATHS:
        if "__pycache__" in source_path.parts:
            continue
        source = source_path.read_text(encoding="utf-8")
        for marker in MODEL_CONFIG_FALLBACK_MARKERS:
            if marker in source:
                violations.append(f"{source_path}:{marker}")

    assert violations == []


def test_legacy_experiment_wrappers_are_static_delegates() -> None:
    experiments_dir = Path("src/recsys_lab/experiments")
    for file_name, function_name in LEGACY_EXPERIMENT_WRAPPERS.items():
        wrapper_path = experiments_dir / file_name
        source = wrapper_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(wrapper_path))

        for comment_line in LEGACY_WRAPPER_COMMENT_LINES:
            assert comment_line in source, f"{wrapper_path} is missing legacy wrapper guard comment"

        forbidden_imports = _forbidden_legacy_lifecycle_imports(tree)
        assert forbidden_imports == [], f"{wrapper_path} imports lifecycle-only dependencies: {forbidden_imports}"

        get_fallback_lines = _attribute_call_lines(tree, attr_name="get")
        assert get_fallback_lines == [], f"{wrapper_path} uses .get(...) fallback calls on lines {get_fallback_lines}"

        _assert_run_function_delegates_to_unified_runner(
            tree,
            wrapper_path=wrapper_path,
            function_name=function_name,
        )


def test_alpha_zero_semantics_disables_cb_claim_eligibility() -> None:
    semantics = build_cb_semantics(0.0)

    assert semantics == {
        "alpha": 0.0,
        "cluster_contribution_config_enabled": False,
        "cluster_contribution_measured": False,
        "cb_claim_eligible": False,
        "claim_gate_reason": (
            "alpha=0 disables cluster factor contribution; run is a CB-disabled ablation"
        ),
    }


def test_alpha_positive_semantics_does_not_make_cb_claim_eligible() -> None:
    semantics = build_cb_semantics(0.25)

    assert semantics == {
        "alpha": 0.25,
        "cluster_contribution_config_enabled": True,
        "cluster_contribution_measured": None,
        "cb_claim_eligible": False,
        "claim_gate_reason": (
            "alpha>0 enables cluster channel, but claim eligibility requires diagnostics "
            "and ablation evidence"
        ),
    }


def test_cb_diagnostics_report_missing_expected_artifacts_and_model_fields() -> None:
    data = RatingsData(
        user_ids=np.asarray([0, 1], dtype=np.int32),
        item_ids=np.asarray([0, 1], dtype=np.int32),
        ratings=np.asarray([4.0, 3.0], dtype=np.float32),
        n_users=2,
        n_items=2,
        name="toy",
        rating_min=1.0,
        rating_max=5.0,
        source_manifest_path=Path("toy_manifest.json"),
    )
    model = CBSVDppRecommender(
        CBSVDppConfig(epochs=1, latent_dim=2, alpha=0.2),
        user_clusters=np.asarray([0, 1], dtype=np.int32),
        item_clusters=np.asarray([0, 1], dtype=np.int32),
        n_user_clusters=2,
        n_item_clusters=2,
    )

    diagnostics = unified_runner_module._build_cb_diagnostics(
        model=model,
        train_data=data,
        fit_artifacts=FitArtifacts(),
        cb_semantics=build_cb_semantics(0.2),
    )

    assert diagnostics["cluster_artifacts_present"] is False
    assert diagnostics["missing_expected_artifacts"] == [
        "cluster_artifacts",
        "user_cluster_history_index",
    ]
    assert "user_factors" in diagnostics["missing_expected_model_fields"]
    assert "user_cluster_factors" in diagnostics["missing_expected_model_fields"]
    assert diagnostics["individual_factor_norm_mean"] is None
    assert diagnostics["cluster_factor_norm_mean"] is None
    assert diagnostics["cluster_to_individual_norm_ratio"] is None
    assert diagnostics["diagnostic_claim_ready"] is False


def test_cb_svdpp_wrapper_delegates_without_lifecycle_services(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    sentinel = {"delegated": "cb_svdpp"}

    def fake_run_unified_experiment(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(cb_svdpp_experiment, "run_unified_experiment", fake_run_unified_experiment)

    payload = cb_svdpp_experiment.run_cb_svdpp_experiment(**_legacy_cb_wrapper_kwargs(tmp_path))

    assert payload is sentinel
    assert len(calls) == 1
    _assert_common_cb_wrapper_delegation(
        calls[0],
        expected_model_name="cb_svdpp",
        repo_root=tmp_path.resolve(),
    )


def test_cb_asvdpp_wrapper_delegates_without_lifecycle_services(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    sentinel = {"delegated": "cb_asvdpp"}

    def fake_run_unified_experiment(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(cb_asvdpp_experiment, "run_unified_experiment", fake_run_unified_experiment)

    payload = cb_asvdpp_experiment.run_cb_asvdpp_experiment(**_legacy_cb_wrapper_kwargs(tmp_path))

    assert payload is sentinel
    assert len(calls) == 1
    _assert_common_cb_wrapper_delegation(
        calls[0],
        expected_model_name="cb_asvdpp",
        repo_root=tmp_path.resolve(),
    )


def test_cb_asvdpp_reuses_supplied_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    data = RatingsData(
        user_ids=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32),
        item_ids=np.asarray([0, 1, 1, 2, 2, 3], dtype=np.int32),
        ratings=np.asarray([5.0, 4.0, 3.0, 3.5, 4.5, 2.0], dtype=np.float32),
        n_users=3,
        n_items=4,
        name="toy",
        rating_min=1.0,
        rating_max=5.0,
        source_manifest_path=Path("toy_manifest.json"),
    )
    explicit_feedback = build_user_explicit_feedback_index(data, dtype="float64")
    implicit_history = build_user_history_index(data, dtype="float64")
    item_clusters = np.asarray([0, 0, 1, 1], dtype=np.int32)
    cluster_history = build_user_cluster_count_index(implicit_history, item_clusters, n_clusters=2)

    def fail_rebuild(*args: object, **kwargs: object) -> None:
        raise AssertionError("model rebuilt an index that the framework supplied")

    monkeypatch.setattr(cb_asvdpp_module, "build_user_explicit_feedback_index", fail_rebuild)
    monkeypatch.setattr(cb_asvdpp_module, "build_user_history_index", fail_rebuild)
    monkeypatch.setattr(cb_asvdpp_module, "build_user_cluster_count_index", fail_rebuild)

    model = CBASVDppRecommender(
        CBASVDppConfig(
            latent_dim=3,
            epochs=1,
            learning_rate=0.01,
            lambda_b=0.01,
            lambda_p=0.01,
            lambda_q=0.01,
            lambda_x=0.01,
            lambda_y=0.01,
            lambda_pC=0.01,
            lambda_qC=0.01,
            lambda_xC=0.01,
            lambda_yC=0.01,
            alpha=0.25,
            seed=7,
            init_std=0.02,
            dtype="float64",
        ),
        user_clusters=np.asarray([0, 0, 1], dtype=np.int32),
        item_clusters=item_clusters,
        n_user_clusters=2,
        n_item_clusters=2,
    )
    model.fit(
        data,
        explicit_feedback=explicit_feedback,
        implicit_history=implicit_history,
        implicit_cluster_history=cluster_history,
    )

    assert model.explicit_feedback is explicit_feedback
    assert model.implicit_history is implicit_history
    assert model.implicit_cluster_history is cluster_history


@pytest.mark.parametrize(
    ("adapter", "model_config"),
    [
        (CBSVDppAdapter, CBSVDppConfig(latent_dim=2, epochs=0, alpha=0.2, dtype="float64")),
        (CBASVDppAdapter, CBASVDppConfig(latent_dim=2, epochs=0, alpha=0.2, dtype="float64")),
    ],
)
def test_cb_adapters_instantiate_from_framework_built_cluster_artifacts(
    adapter: type[CBSVDppAdapter] | type[CBASVDppAdapter],
    model_config: CBSVDppConfig | CBASVDppConfig,
) -> None:
    artifacts = _toy_cb_fit_artifacts()

    model = adapter.instantiate(model_config, artifacts=artifacts)

    assert artifacts.cluster_artifacts is not None
    assert np.array_equal(model.user_clusters, artifacts.cluster_artifacts.user_clusters)
    assert np.array_equal(model.item_clusters, artifacts.cluster_artifacts.item_clusters)
    assert model.n_user_clusters == artifacts.cluster_artifacts.r_star_counts.shape[0]
    assert model.n_item_clusters == artifacts.cluster_artifacts.r_star_counts.shape[1]


def test_cb_svdpp_adapter_passes_framework_built_indices_when_reuse_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _toy_cb_ratings_data()
    artifacts = _toy_cb_fit_artifacts(data)

    def fail_rebuild(*args: object, **kwargs: object) -> None:
        raise AssertionError("cb_svdpp rebuilt an index that the framework supplied")

    monkeypatch.setattr(cb_svdpp_module, "build_user_history_index", fail_rebuild)
    monkeypatch.setattr(cb_svdpp_module, "build_user_cluster_count_index", fail_rebuild)

    model = CBSVDppAdapter.instantiate(_toy_cb_svdpp_config(), artifacts=artifacts)
    CBSVDppAdapter.fit(model, data, artifacts=artifacts, reuse_precomputed_indices=True)

    assert model.user_histories is artifacts.user_history_index
    assert model.user_cluster_histories is artifacts.user_cluster_history_index


def test_cb_asvdpp_adapter_passes_framework_built_indices_when_reuse_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _toy_cb_ratings_data()
    artifacts = _toy_cb_fit_artifacts(data)

    def fail_rebuild(*args: object, **kwargs: object) -> None:
        raise AssertionError("cb_asvdpp rebuilt an index that the framework supplied")

    monkeypatch.setattr(cb_asvdpp_module, "build_user_explicit_feedback_index", fail_rebuild)
    monkeypatch.setattr(cb_asvdpp_module, "build_user_history_index", fail_rebuild)
    monkeypatch.setattr(cb_asvdpp_module, "build_user_cluster_count_index", fail_rebuild)

    model = CBASVDppAdapter.instantiate(_toy_cb_asvdpp_config(), artifacts=artifacts)
    CBASVDppAdapter.fit(model, data, artifacts=artifacts, reuse_precomputed_indices=True)

    assert model.explicit_feedback is artifacts.explicit_feedback_index
    assert model.implicit_history is artifacts.user_history_index
    assert model.implicit_cluster_history is artifacts.user_cluster_history_index


def test_cb_adapters_pass_none_to_model_fallback_builders_when_reuse_disabled() -> None:
    data = _toy_cb_ratings_data()
    artifacts = _toy_cb_fit_artifacts(data)

    cb_svdpp_model = _FitSpy()
    CBSVDppAdapter.fit(cb_svdpp_model, data, artifacts=artifacts, reuse_precomputed_indices=False)
    assert cb_svdpp_model.train_data is data
    assert cb_svdpp_model.fit_kwargs == {
        "user_histories": None,
        "user_cluster_histories": None,
    }

    cb_asvdpp_model = _FitSpy()
    CBASVDppAdapter.fit(cb_asvdpp_model, data, artifacts=artifacts, reuse_precomputed_indices=False)
    assert cb_asvdpp_model.train_data is data
    assert cb_asvdpp_model.fit_kwargs == {
        "explicit_feedback": None,
        "implicit_history": None,
        "implicit_cluster_history": None,
    }


def test_cb_models_rebuild_indices_on_explicit_model_fallback_path(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _toy_cb_ratings_data()
    cb_svdpp_calls: list[str] = []
    cb_asvdpp_calls: list[str] = []
    original_cb_svdpp_history_builder = cb_svdpp_module.build_user_history_index
    original_cb_svdpp_cluster_builder = cb_svdpp_module.build_user_cluster_count_index
    original_cb_asvdpp_explicit_builder = cb_asvdpp_module.build_user_explicit_feedback_index
    original_cb_asvdpp_history_builder = cb_asvdpp_module.build_user_history_index
    original_cb_asvdpp_cluster_builder = cb_asvdpp_module.build_user_cluster_count_index

    def build_cb_svdpp_history(*args: object, **kwargs: object) -> object:
        cb_svdpp_calls.append("user_history_index")
        return original_cb_svdpp_history_builder(*args, **kwargs)

    def build_cb_svdpp_cluster(*args: object, **kwargs: object) -> object:
        cb_svdpp_calls.append("user_cluster_history_index")
        return original_cb_svdpp_cluster_builder(*args, **kwargs)

    def build_cb_asvdpp_explicit(*args: object, **kwargs: object) -> object:
        cb_asvdpp_calls.append("explicit_feedback_index")
        return original_cb_asvdpp_explicit_builder(*args, **kwargs)

    def build_cb_asvdpp_history(*args: object, **kwargs: object) -> object:
        cb_asvdpp_calls.append("user_history_index")
        return original_cb_asvdpp_history_builder(*args, **kwargs)

    def build_cb_asvdpp_cluster(*args: object, **kwargs: object) -> object:
        cb_asvdpp_calls.append("user_cluster_history_index")
        return original_cb_asvdpp_cluster_builder(*args, **kwargs)

    monkeypatch.setattr(cb_svdpp_module, "build_user_history_index", build_cb_svdpp_history)
    monkeypatch.setattr(cb_svdpp_module, "build_user_cluster_count_index", build_cb_svdpp_cluster)
    monkeypatch.setattr(cb_asvdpp_module, "build_user_explicit_feedback_index", build_cb_asvdpp_explicit)
    monkeypatch.setattr(cb_asvdpp_module, "build_user_history_index", build_cb_asvdpp_history)
    monkeypatch.setattr(cb_asvdpp_module, "build_user_cluster_count_index", build_cb_asvdpp_cluster)

    CBSVDppAdapter.instantiate(_toy_cb_svdpp_config(), artifacts=_toy_cb_fit_artifacts(data)).fit(data)
    CBASVDppAdapter.instantiate(_toy_cb_asvdpp_config(), artifacts=_toy_cb_fit_artifacts(data)).fit(data)

    assert cb_svdpp_calls == ["user_history_index", "user_cluster_history_index"]
    assert cb_asvdpp_calls == [
        "explicit_feedback_index",
        "user_history_index",
        "user_cluster_history_index",
    ]


def test_unified_runner_builds_only_required_artifacts_for_biased_mf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_toy_processed_dataset(tmp_path)
    model_config_path = _write_yaml(
        tmp_path / "biased_mf.yaml",
        {
            "metadata": {"status": "test", "owner": "repo", "purpose": "model_profile"},
            "model": {
                "name": "biased_mf",
                "family": "matrix_factorization",
                "scope": "paper_inspired",
            },
            "training": {
                "latent_dim": 2,
                "epochs": 1,
                "learning_rate": 0.01,
                "lambda_b": 0.01,
                "lambda_p": 0.01,
                "lambda_q": 0.01,
                "init_std": 0.02,
                "dtype": "float64",
                "training_backend": "python",
            },
            "notes": ["test profile"],
        },
    )
    runtime_config_path = _write_toy_runtime_config(tmp_path)
    device_config_path = _write_toy_device_config(tmp_path)
    monkeypatch.setattr(unified_runner_module, "validate_manifest_file", lambda *args, **kwargs: None)

    payload = run_unified_experiment(
        processed_manifest_path=manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_config=SplitConfig(train_ratio=0.75, validation_ratio=0.125, seed=1),
        model_seed=1,
        repo_root=tmp_path,
        model_name="biased_mf",
        split_family="benchmark_random_v1",
        evaluate_test=True,
        use_split_cache=False,
        use_training_index_cache=False,
        use_cluster_artifact_cache=False,
        services=build_experiment_services(
            git_snapshot_fn=lambda _root: {"commit": "abcdef0", "branch": "test", "dirty": False},
        ),
    )

    run_manifest = json.loads(Path(payload["run_manifest"]).read_text(encoding="utf-8"))
    metrics = json.loads((tmp_path / run_manifest["artifacts"]["metrics"]).read_text(encoding="utf-8"))
    config_snapshot = yaml.safe_load(
        (tmp_path / run_manifest["artifacts"]["config_snapshot"]).read_text(encoding="utf-8")
    )

    assert config_snapshot["model_requirements"]["required_artifacts"] == []
    assert metrics["model"]["requirements"]["required_artifacts"] == []
    assert metrics["model"]["available_fit_artifacts"] == []
    assert set(metrics["caches"]) == {"split"}
    assert "training_index_cache" not in metrics["model"]
    assert "clustering" not in metrics["model"]
    assert "cb_semantics" not in metrics
    assert "cb_semantics" not in metrics["model"]
    assert run_manifest["caches"] == metrics["caches"]
    assert run_manifest["profiling"] == metrics["profiling"]
    assert metrics["profiling"]["stage_count"] == len(metrics["profiling"]["stages"])


def test_unified_runner_writes_alpha_zero_cb_semantics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest_path = _write_toy_processed_dataset(tmp_path)
    model_config_path = _write_yaml(
        tmp_path / "cb_svdpp_alpha_zero.yaml",
        {
            "metadata": {"status": "test", "owner": "repo", "purpose": "model_profile"},
            "model": {
                "name": "cb_svdpp",
                "family": "clustering_based_factorization",
                "scope": "paper_inspired",
            },
            "training": {
                "latent_dim": 2,
                "epochs": 1,
                "learning_rate": 0.01,
                "lambda_b": 0.01,
                "lambda_p": 0.01,
                "lambda_q": 0.01,
                "lambda_y": 0.01,
                "lambda_pC": 0.01,
                "lambda_qC": 0.01,
                "lambda_yC": 0.01,
                "init_std": 0.02,
                "dtype": "float64",
                "implicit_policy": "ratings_as_implicit",
            },
            "clustering": {
                "n_user_clusters": 2,
                "n_item_clusters": 2,
                "alpha": 0.0,
                "algorithm": "kmeans",
                "kmeans_n_init": 1,
            },
            "notes": ["test profile"],
        },
    )
    runtime_config_path = _write_yaml(
        tmp_path / "runtime.yaml",
        {
            "runtime": {
                "project_slug": "test",
                "default_device_profile": "test_cpu",
                "default_precision_profile": "reference_float64",
                "cache_root": "artifacts/local",
            },
            "precision_profiles": {"reference_float64": {"dtype": "float64"}},
        },
    )
    device_config_path = _write_yaml(
        tmp_path / "device.yaml",
        {
            "metadata": {"status": "validated_test"},
            "device_profile": {
                "name": "test_cpu",
                "compute_class": "local_cpu",
                "cpu_model": "test_cpu",
                "logical_threads": 1,
                "physical_cores": 1,
                "ram_gb": 8,
                "gpu_enabled": False,
            },
            "storage": {"cache_preference": "local", "archive_preference": "local"},
            "threading": {"omp_num_threads": 1, "blas_threads": 1},
            "resource_limits": {"ram_guardrail_fraction": 0.8},
            "precision": {"default_dtype": "float64", "reference_dtype": "float64"},
        },
    )
    monkeypatch.setattr(unified_runner_module, "validate_manifest_file", lambda *args, **kwargs: None)

    payload = run_unified_experiment(
        processed_manifest_path=manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_config=SplitConfig(train_ratio=0.75, validation_ratio=0.125, seed=1),
        model_seed=1,
        repo_root=tmp_path,
        model_name="cb_svdpp",
        split_family="benchmark_random_v1",
        evaluate_test=True,
        use_split_cache=False,
        use_training_index_cache=False,
        use_cluster_artifact_cache=False,
        services=build_experiment_services(
            git_snapshot_fn=lambda _root: {"commit": "abcdef0", "branch": "test", "dirty": False},
        ),
    )

    run_manifest = json.loads(Path(payload["run_manifest"]).read_text(encoding="utf-8"))
    metrics_path = tmp_path / run_manifest["artifacts"]["metrics"]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config_snapshot = yaml.safe_load(
        (tmp_path / run_manifest["artifacts"]["config_snapshot"]).read_text(encoding="utf-8")
    )

    assert config_snapshot["model_requirements"]["required_artifacts"] == [
        "user_history_index",
        "cluster_artifacts",
        "user_cluster_history_index",
    ]
    assert metrics["model"]["requirements"]["required_artifacts"] == [
        "user_history_index",
        "cluster_artifacts",
        "user_cluster_history_index",
    ]
    assert metrics["model"]["available_fit_artifacts"] == [
        "user_history_index",
        "user_cluster_history_index",
        "cluster_artifacts",
    ]
    assert {"split", "training_user_history", "cluster_artifacts", "user_cluster_history"}.issubset(
        metrics["caches"]
    )
    assert run_manifest["caches"] == metrics["caches"]
    assert run_manifest["profiling"] == metrics["profiling"]
    assert metrics["profiling"]["stage_count"] == len(metrics["profiling"]["stages"])
    assert run_manifest["cb_semantics"]["cb_claim_eligible"] is False
    assert metrics["cb_semantics"]["cluster_contribution_config_enabled"] is False
    assert metrics["cb_semantics"]["cluster_contribution_measured"] is False
    cb_diagnostics = metrics["cb_diagnostics"]
    assert cb_diagnostics["alpha"] == 0.0
    assert cb_diagnostics["cluster_artifacts_present"] is True
    assert cb_diagnostics["cluster_contribution_config_enabled"] is False
    assert cb_diagnostics["cluster_contribution_measured"] is False
    assert cb_diagnostics["cb_claim_eligible"] is False
    assert cb_diagnostics["diagnostic_claim_ready"] is False
    assert cb_diagnostics["missing_expected_artifacts"] == []
    assert (
        metrics["model"]["cb_semantics"]["claim_gate_reason"]
        == "alpha=0 disables cluster factor contribution; run is a CB-disabled ablation"
    )


def _forbidden_legacy_lifecycle_imports(tree: ast.AST) -> list[str]:
    forbidden: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name in FORBIDDEN_LEGACY_LIFECYCLE_IMPORT_MODULES:
                forbidden.add(module_name)
            for alias in node.names:
                imported_name = alias.name
                bound_name = alias.asname or imported_name
                if imported_name in FORBIDDEN_LEGACY_LIFECYCLE_IMPORT_NAMES:
                    forbidden.add(imported_name)
                if bound_name in FORBIDDEN_LEGACY_LIFECYCLE_IMPORT_NAMES:
                    forbidden.add(bound_name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_module = alias.name
                if imported_module in FORBIDDEN_LEGACY_LIFECYCLE_IMPORT_MODULES:
                    forbidden.add(imported_module)
    return sorted(forbidden)


def _attribute_call_lines(tree: ast.AST, *, attr_name: str) -> list[int]:
    return sorted(
        {
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == attr_name
        }
    )


def _assert_run_function_delegates_to_unified_runner(
    tree: ast.Module,
    *,
    wrapper_path: Path,
    function_name: str,
) -> None:
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    ]
    assert len(functions) == 1, f"{wrapper_path} must define exactly one {function_name}"

    function = functions[0]
    assert len(function.body) == 1, f"{wrapper_path}:{function_name} must only return unified runner delegation"
    statement = function.body[0]
    assert isinstance(statement, ast.Return), f"{wrapper_path}:{function_name} must return unified runner payload"
    assert isinstance(statement.value, ast.Call), f"{wrapper_path}:{function_name} must call run_unified_experiment"
    assert isinstance(statement.value.func, ast.Name), f"{wrapper_path}:{function_name} must call a direct function"
    assert statement.value.func.id == "run_unified_experiment", (
        f"{wrapper_path}:{function_name} must delegate to run_unified_experiment"
    )


def _legacy_cb_wrapper_kwargs(tmp_path: Path) -> dict[str, Any]:
    return {
        "processed_manifest_path": tmp_path / "processed_manifest.json",
        "model_config_path": tmp_path / "model.yaml",
        "runtime_config_path": tmp_path / "runtime.yaml",
        "device_config_path": tmp_path / "device.yaml",
        "split_config": SplitConfig(train_ratio=0.7, validation_ratio=0.1, seed=13),
        "model_seed": 17,
        "repo_root": tmp_path,
        "command": "legacy-wrapper-test",
        "split_family": "paper_faithful_ml100k_v1",
        "inner_validation_seed": 19,
        "evaluate_test": False,
        "use_split_cache": False,
        "reuse_precomputed_indices": False,
        "use_training_index_cache": True,
        "use_cluster_artifact_cache": True,
    }


def _assert_common_cb_wrapper_delegation(
    call: dict[str, Any],
    *,
    expected_model_name: str,
    repo_root: Path,
) -> None:
    assert "services" not in call
    assert call["model_name"] == expected_model_name
    assert call["repo_root"] == repo_root
    assert call["split_family"] == "paper_faithful_ml100k_v1"
    assert call["inner_validation_seed"] == 19
    assert call["evaluate_test"] is False
    assert call["use_split_cache"] is False
    assert call["reuse_precomputed_indices"] is False
    assert call["use_training_index_cache"] is True
    assert call["use_cluster_artifact_cache"] is True


class _FitSpy:
    def __init__(self) -> None:
        self.train_data: RatingsData | None = None
        self.fit_kwargs: dict[str, Any] | None = None

    def fit(self, train_data: RatingsData, **kwargs: Any) -> "_FitSpy":
        self.train_data = train_data
        self.fit_kwargs = kwargs
        return self


def _toy_cb_ratings_data() -> RatingsData:
    return RatingsData(
        user_ids=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32),
        item_ids=np.asarray([0, 1, 1, 2, 2, 3], dtype=np.int32),
        ratings=np.asarray([5.0, 4.0, 3.0, 3.5, 4.5, 2.0], dtype=np.float32),
        n_users=3,
        n_items=4,
        name="toy",
        rating_min=1.0,
        rating_max=5.0,
        source_manifest_path=Path("toy_manifest.json"),
    )


def _toy_cb_cluster_artifacts() -> ClusterArtifacts:
    return ClusterArtifacts(
        user_clusters=np.asarray([0, 0, 1], dtype=np.int32),
        item_clusters=np.asarray([0, 0, 1, 1], dtype=np.int32),
        user_cluster_sizes=np.asarray([2, 1], dtype=np.int64),
        item_cluster_sizes=np.asarray([2, 2], dtype=np.int64),
        r_star_means=np.asarray([[4.5, 0.0], [0.0, 3.25]], dtype=np.float64),
        r_star_counts=np.asarray([[2, 0], [0, 4]], dtype=np.int64),
        induction_train_rmse=0.0,
        user_kmeans_inertia=0.0,
        item_kmeans_inertia=0.0,
    )


def _toy_cb_fit_artifacts(data: RatingsData | None = None) -> FitArtifacts:
    ratings_data = data if data is not None else _toy_cb_ratings_data()
    cluster_artifacts = _toy_cb_cluster_artifacts()
    user_history_index = build_user_history_index(ratings_data, dtype="float64")
    return FitArtifacts(
        user_history_index=user_history_index,
        explicit_feedback_index=build_user_explicit_feedback_index(ratings_data, dtype="float64"),
        user_cluster_history_index=build_user_cluster_count_index(
            user_history_index,
            cluster_artifacts.item_clusters,
            n_clusters=int(cluster_artifacts.r_star_counts.shape[1]),
        ),
        cluster_artifacts=cluster_artifacts,
    )


def _toy_cb_svdpp_config() -> CBSVDppConfig:
    return CBSVDppConfig(
        latent_dim=2,
        epochs=0,
        learning_rate=0.01,
        lambda_b=0.01,
        lambda_p=0.01,
        lambda_q=0.01,
        lambda_y=0.01,
        lambda_pC=0.01,
        lambda_qC=0.01,
        lambda_yC=0.01,
        alpha=0.2,
        seed=7,
        init_std=0.02,
        dtype="float64",
    )


def _toy_cb_asvdpp_config() -> CBASVDppConfig:
    return CBASVDppConfig(
        latent_dim=2,
        epochs=0,
        learning_rate=0.01,
        lambda_b=0.01,
        lambda_p=0.01,
        lambda_q=0.01,
        lambda_x=0.01,
        lambda_y=0.01,
        lambda_pC=0.01,
        lambda_qC=0.01,
        lambda_xC=0.01,
        lambda_yC=0.01,
        alpha=0.2,
        seed=7,
        init_std=0.02,
        dtype="float64",
    )


def _write_toy_processed_dataset(tmp_path: Path) -> Path:
    user_ids = np.repeat(np.arange(4, dtype=np.int32), 4)
    item_ids = np.tile(np.arange(4, dtype=np.int32), 4)
    ratings = np.asarray(
        [
            5.0,
            4.5,
            3.5,
            3.0,
            4.5,
            4.0,
            3.0,
            2.5,
            3.5,
            3.0,
            4.0,
            4.5,
            2.0,
            2.5,
            4.5,
            5.0,
        ],
        dtype=np.float64,
    )
    interactions_path = tmp_path / "interactions.parquet"
    pq.write_table(
        pa.table(
            {
                "user_idx": user_ids,
                "item_idx": item_ids,
                "rating": ratings,
            }
        ),
        interactions_path,
    )
    manifest_path = tmp_path / "processed_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_name": "toy",
                "dataset_short_name": "ml_latest_small",
                "split_family": "benchmark_random_v1",
                "preprocessing_family": "toy_v1",
                "dtype": "float64",
                "counts": {"users": 4, "rated_items": 4, "interactions": 16},
                "rating_range": {"min": 2.0, "max": 5.0},
                "artifacts": {"interactions": str(interactions_path)},
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_toy_runtime_config(tmp_path: Path) -> Path:
    return _write_yaml(
        tmp_path / "runtime.yaml",
        {
            "runtime": {
                "project_slug": "test",
                "default_device_profile": "test_cpu",
                "default_precision_profile": "reference_float64",
                "cache_root": "artifacts/local",
            },
            "precision_profiles": {"reference_float64": {"dtype": "float64"}},
        },
    )


def _write_toy_device_config(tmp_path: Path) -> Path:
    return _write_yaml(
        tmp_path / "device.yaml",
        {
            "metadata": {"status": "validated_test"},
            "device_profile": {
                "name": "test_cpu",
                "compute_class": "local_cpu",
                "cpu_model": "test_cpu",
                "logical_threads": 1,
                "physical_cores": 1,
                "ram_gb": 8,
                "gpu_enabled": False,
            },
            "storage": {"cache_preference": "local", "archive_preference": "local"},
            "threading": {"omp_num_threads": 1, "blas_threads": 1},
            "resource_limits": {"ram_guardrail_fraction": 0.8},
            "precision": {"default_dtype": "float64", "reference_dtype": "float64"},
        },
    )


def _write_yaml(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
