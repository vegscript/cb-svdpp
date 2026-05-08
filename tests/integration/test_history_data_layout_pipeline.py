from pathlib import Path
from typing import Any

import numpy as np

from recsys_lab.experiments import unified_runner as unified_runner_module
from recsys_lab.experiments.common import SplitConfig
from recsys_lab.experiments.unified_runner import build_experiment_services, run_unified_experiment
from tests.integration.test_unified_pipeline_smoke_all_models import _prepare_toy_repo


def _assert_int32_contiguous(array: np.ndarray) -> None:
    assert array.dtype == np.int32
    assert array.flags.c_contiguous


def _assert_float32_contiguous(array: np.ndarray) -> None:
    assert array.dtype == np.float32
    assert array.flags.c_contiguous


def test_history_data_layout_survives_tiny_unified_cb_svdpp_run(tmp_path: Path, monkeypatch) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    repo_root, processed_manifest_path, runtime_config_path, device_config_path, model_config_paths = _prepare_toy_repo(
        tmp_path,
        actual_repo_root,
    )
    captured: dict[str, Any] = {}
    original_build_kernel_profile_payload = unified_runner_module.build_kernel_profile_payload

    def capture_kernel_profile_payload(**kwargs: Any) -> dict[str, Any]:
        captured["fit_artifacts"] = kwargs["fit_artifacts"]
        return original_build_kernel_profile_payload(**kwargs)

    monkeypatch.setattr(unified_runner_module, "build_kernel_profile_payload", capture_kernel_profile_payload)
    services = build_experiment_services(
        git_snapshot_fn=lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": False},
    )

    payload = run_unified_experiment(
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_paths["cb_svdpp"],
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_config=SplitConfig(train_ratio=0.5, validation_ratio=0.25, seed=3),
        model_seed=4,
        repo_root=repo_root,
        model_name="cb_svdpp",
        split_family="benchmark_random_v1",
        evaluate_test=True,
        use_split_cache=False,
        use_training_index_cache=False,
        use_cluster_artifact_cache=False,
        command="recsys-lab train --model cb_svdpp --history-layout-smoke",
        services=services,
    )

    run_dir = Path(payload["run_dir"])
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "performance_profile.json").exists()
    assert (run_dir / "kernel_profile.json").exists()

    fit_artifacts = captured["fit_artifacts"]
    user_history = fit_artifacts.user_history_index
    cluster_history = fit_artifacts.user_cluster_history_index
    assert user_history is not None
    assert cluster_history is not None

    _assert_int32_contiguous(user_history.indptr)
    _assert_int32_contiguous(user_history.item_indices)
    _assert_int32_contiguous(user_history.counts)
    _assert_float32_contiguous(user_history.norms)

    _assert_int32_contiguous(cluster_history.indptr)
    _assert_int32_contiguous(cluster_history.cluster_ids)
    _assert_int32_contiguous(cluster_history.counts)
