from __future__ import annotations

import json
from pathlib import Path

from recsys_lab.experiments.common import SplitConfig
from recsys_lab.experiments.unified_runner import build_experiment_services, run_unified_experiment
from tests.integration.test_unified_pipeline_smoke_all_models import _assert_train_coverage, _prepare_toy_repo


def test_unified_run_writes_kernel_cost_anatomy_profile_for_cb_model(tmp_path: Path) -> None:
    actual_repo_root = Path(__file__).resolve().parents[2]
    repo_root, processed_manifest_path, runtime_config_path, device_config_path, model_config_paths = _prepare_toy_repo(
        tmp_path,
        actual_repo_root,
    )
    split_config = SplitConfig(train_ratio=0.5, validation_ratio=0.25, seed=3)
    _assert_train_coverage(processed_manifest_path, split_config)
    services = build_experiment_services(
        git_snapshot_fn=lambda _repo_root: {"commit": "abcdef1234567", "branch": "main", "dirty": False},
    )

    payload = run_unified_experiment(
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_paths["cb_svdpp"],
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_config=split_config,
        model_seed=4,
        repo_root=repo_root,
        model_name="cb_svdpp",
        split_family="benchmark_random_v1",
        evaluate_test=True,
        use_split_cache=False,
        use_training_index_cache=False,
        use_cluster_artifact_cache=False,
        command="recsys-lab train --model cb_svdpp --kernel-cost-anatomy-smoke",
        services=services,
    )

    run_dir = Path(payload["run_dir"])
    kernel_profile_path = run_dir / "kernel_profile.json"
    metrics_path = run_dir / "metrics.json"
    run_manifest_path = Path(payload["run_manifest"])

    assert kernel_profile_path.exists()
    kernel_profile = json.loads(kernel_profile_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))

    assert kernel_profile["profile_version"] == "kernel_cost_anatomy_v1"
    assert kernel_profile["train_rows"] > 0
    assert kernel_profile["epochs"] > 0
    assert kernel_profile["latent_dim"] > 0
    assert kernel_profile["epoch_durations_seconds"]
    assert len(kernel_profile["epoch_durations_seconds"]) == kernel_profile["epochs"]
    assert "estimated_kernel_work" in kernel_profile
    assert kernel_profile["estimated_kernel_work"]["rating_updates"] > 0
    assert kernel_profile["estimated_kernel_work"]["cluster_history_visits"] >= 0
    assert kernel_profile["history_structure"]["cluster"]
    assert kernel_profile["history_structure"]["cluster"]["total_edges"] >= 0
    assert metrics["kernel_profile"]["path"] == "kernel_profile.json"
    assert metrics["kernel_profile"]["profile_version"] == "kernel_cost_anatomy_v1"
    assert metrics["kernel_profile"]["epoch_count"] == kernel_profile["epochs"]
    assert metrics["kernel_profile"]["train_rows"] == kernel_profile["train_rows"]
    assert metrics["kernel_profile"]["estimated_factor_touches"] == kernel_profile["estimated_kernel_work"][
        "estimated_factor_touches"
    ]
    assert metrics["artifacts"]["kernel_profile"].endswith("kernel_profile.json")
    assert run_manifest["artifacts"]["kernel_profile"].endswith("kernel_profile.json")
