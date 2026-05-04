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
from recsys_lab.data.histories import (
    build_user_cluster_count_index,
    build_user_explicit_feedback_index,
    build_user_history_index,
)
from recsys_lab.data.processed import RatingsData
from recsys_lab.experiments.common import SplitConfig
from recsys_lab.experiments.unified_runner import build_experiment_services, run_unified_experiment
from recsys_lab.models.cb_asvdpp import CBASVDppConfig, CBASVDppRecommender
from recsys_lab.models.registry import MODEL_REGISTRY, build_cb_semantics, validate_model_config_payload


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
    "load_or_build_split_cache",
    "load_ratings_data_from_manifest",
    "rmse",
    "write_json",
}
FORBIDDEN_LEGACY_LIFECYCLE_IMPORT_MODULES = {
    "recsys_lab.data.processed",
    "recsys_lab.experiments.performance",
    "recsys_lab.experiments.split_cache",
    "recsys_lab.metrics",
}


def test_model_registry_declares_expected_artifact_requirements() -> None:
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

    cb_svdpp_artifacts = set(MODEL_REGISTRY["cb_svdpp"].requirements.artifact_names())
    cb_asvdpp_artifacts = set(MODEL_REGISTRY["cb_asvdpp"].requirements.artifact_names())
    assert cb_svdpp_artifacts == {
        "user_history_index",
        "cluster_artifacts",
        "user_cluster_history_index",
    }
    assert cb_asvdpp_artifacts == cb_svdpp_artifacts | {"explicit_feedback_index"}


def test_unknown_model_config_key_fails_validation() -> None:
    payload = yaml.safe_load(Path("configs/models/svdpp.yaml").read_text(encoding="utf-8"))
    payload["training"]["unknown_field"] = 1

    with pytest.raises(Exception, match="unknown_field"):
        validate_model_config_payload(payload)


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
        "cluster_contribution_enabled": False,
        "cb_claim_eligible": False,
        "reason": "alpha=0 disables cluster factor contribution",
    }


def test_cb_svdpp_wrapper_delegates_with_explicit_services(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        wrapper_module=cb_svdpp_experiment,
        expected_model_name="cb_svdpp",
        repo_root=tmp_path.resolve(),
    )


def test_cb_asvdpp_wrapper_delegates_with_explicit_services(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        wrapper_module=cb_asvdpp_experiment,
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

    assert run_manifest["cb_semantics"]["cb_claim_eligible"] is False
    assert metrics["cb_semantics"]["cluster_contribution_enabled"] is False
    assert metrics["model"]["cb_semantics"]["reason"] == "alpha=0 disables cluster factor contribution"


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
    wrapper_module: Any,
    expected_model_name: str,
    repo_root: Path,
) -> None:
    services = call["services"]
    assert isinstance(services, unified_runner_module.ExperimentServices)
    assert services.git_snapshot_fn is wrapper_module.git_snapshot
    assert services.paper_faithful_split_fn is wrapper_module.official_ml100k_paper_faithful_split
    assert services.inner_validation_split_fn is wrapper_module.official_ml100k_inner_validation_split
    assert call["model_name"] == expected_model_name
    assert call["repo_root"] == repo_root
    assert call["split_family"] == "paper_faithful_ml100k_v1"
    assert call["inner_validation_seed"] == 19
    assert call["evaluate_test"] is False
    assert call["use_split_cache"] is False
    assert call["reuse_precomputed_indices"] is False
    assert call["use_training_index_cache"] is True
    assert call["use_cluster_artifact_cache"] is True


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


def _write_yaml(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
