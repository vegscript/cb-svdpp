import json

import pytest
import typer

from recsys_lab.cli import main as cli_main
from recsys_lab.cli.main import _resolve_split_cache_override


def test_resolve_split_cache_override_supports_auto_enable_disable() -> None:
    assert _resolve_split_cache_override("auto") is None
    assert _resolve_split_cache_override("enable") is True
    assert _resolve_split_cache_override("disable") is False


def test_resolve_split_cache_override_rejects_invalid_values() -> None:
    with pytest.raises(typer.BadParameter):
        _resolve_split_cache_override("invalid")


def test_train_cb_svdpp_wires_split_and_training_index_cache(monkeypatch, tmp_path) -> None:
    observed: dict[str, object] = {}

    monkeypatch.setattr(cli_main, "discover_repo_root", lambda: tmp_path)

    def _fake_resolve_path(path_value, *, repo_root):
        return (repo_root / path_value).resolve()

    monkeypatch.setattr(cli_main, "_resolve_path", _fake_resolve_path)

    def _fake_runner(**kwargs):
        observed.update(kwargs)
        return {"run_id": "synthetic", "run_manifest": "synthetic.json", "metrics": {}}

    monkeypatch.setattr(cli_main, "run_cb_svdpp_experiment", _fake_runner)

    cli_main.train_cb_svdpp(
        "data/processed/ml10m/manifest.json",
        model_config="configs/models/archive/tuned/ml10m_cb_svdpp_stage0_probe_e001.yaml",
        runtime_config="configs/runtime/base.yaml",
        device_config="configs/runtime/devices/local_i5_2500k_24gb.yaml",
        split_cache="enable",
        training_index_cache=True,
        cluster_artifact_cache=True,
    )

    assert observed["use_split_cache"] is True
    assert observed["use_training_index_cache"] is True
    assert observed["use_cluster_artifact_cache"] is True


def test_train_model_wires_canonical_train_model_command_to_unified_runner(monkeypatch, tmp_path) -> None:
    observed: dict[str, object] = {}

    monkeypatch.setattr(cli_main, "discover_repo_root", lambda: tmp_path)

    def _fake_resolve_path(path_value, *, repo_root):
        return (repo_root / path_value).resolve()

    monkeypatch.setattr(cli_main, "_resolve_path", _fake_resolve_path)

    def _fake_runner(**kwargs):
        observed.update(kwargs)
        return {"run_id": "synthetic", "run_manifest": "synthetic.json", "metrics": {}}

    monkeypatch.setattr(cli_main, "run_unified_experiment", _fake_runner)

    cli_main.train_model(
        model="biased_mf",
        processed_manifest="data/processed/ml_latest_small/manifest.json",
        model_config="configs/models/biased_mf.yaml",
        runtime_config="configs/runtime/base.yaml",
        device_config="configs/runtime/devices/local.yaml",
        split_family="benchmark_random_v1",
        train_ratio=0.5,
        validation_ratio=0.25,
        split_seed=3,
        model_seed=4,
        split_cache="disable",
        training_index_cache=False,
        cluster_artifact_cache=False,
        evaluate_test=True,
    )

    assert observed["model_name"] == "biased_mf"
    assert observed["processed_manifest_path"] == (tmp_path / "data/processed/ml_latest_small/manifest.json").resolve()
    assert observed["model_config_path"] == (tmp_path / "configs/models/biased_mf.yaml").resolve()
    assert observed["split_family"] == "benchmark_random_v1"
    assert observed["model_seed"] == 4
    assert observed["evaluate_test"] is True
    assert observed["use_split_cache"] is False
    assert observed["use_training_index_cache"] is False
    assert observed["use_cluster_artifact_cache"] is False


def test_train_cb_asvdpp_wires_cluster_artifact_cache(monkeypatch, tmp_path) -> None:
    observed: dict[str, object] = {}

    monkeypatch.setattr(cli_main, "discover_repo_root", lambda: tmp_path)

    def _fake_resolve_path(path_value, *, repo_root):
        return (repo_root / path_value).resolve()

    monkeypatch.setattr(cli_main, "_resolve_path", _fake_resolve_path)

    def _fake_runner(**kwargs):
        observed.update(kwargs)
        return {"run_id": "synthetic", "run_manifest": "synthetic.json", "metrics": {}}

    monkeypatch.setattr(cli_main, "run_cb_asvdpp_experiment", _fake_runner)

    cli_main.train_cb_asvdpp(
        "data/processed/ml10m/manifest.json",
        model_config="configs/models/cb_asvdpp.yaml",
        runtime_config="configs/runtime/base.yaml",
        device_config="configs/runtime/devices/local_i5_2500k_24gb.yaml",
        split_cache="auto",
        training_index_cache=False,
        cluster_artifact_cache=True,
    )

    assert observed["use_cluster_artifact_cache"] is True


def test_tune_inner_wires_supported_cache_controls(monkeypatch, tmp_path) -> None:
    observed: dict[str, object] = {}

    monkeypatch.setattr(cli_main, "discover_repo_root", lambda: tmp_path)

    def _fake_resolve_path(path_value, *, repo_root):
        return (repo_root / path_value).resolve()

    monkeypatch.setattr(cli_main, "_resolve_path", _fake_resolve_path)

    def _fake_runner(**kwargs):
        observed.update(kwargs)
        return {"benchmark_id": "synthetic", "benchmark_manifest": "synthetic.json"}

    monkeypatch.setattr(cli_main, "run_inner_tuning", _fake_runner)

    cli_main.tune_inner(
        "configs/experiments/tuning/archive/ml100k_cb_svdpp_stage1.yaml",
        "data/processed/ml100k/manifest.json",
        runtime_config="configs/runtime/base.yaml",
        device_config="configs/runtime/devices/local_i5_2500k_24gb.yaml",
        split_cache="enable",
        training_index_cache=True,
        cluster_artifact_cache=True,
    )

    assert observed["use_split_cache"] is True
    assert observed["use_training_index_cache"] is True
    assert observed["use_cluster_artifact_cache"] is True


def test_validate_runtime_profile_accepts_local_claim_eligible_profile(capsys) -> None:
    cli_main.validate_runtime_profile(
        "configs/runtime/devices/local_i5_2500k_24gb.yaml",
        claim_eligible=True,
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "valid"
    assert payload["claim_eligible_requested"] is True
    assert payload["device_profile_contract"]["claim_eligible"] is True
    assert payload["device_profile_contract"]["profile_name"] == "local_i5_2500k_24gb"


def test_validate_runtime_profile_rejects_hpc_template_for_claim_eligible_run() -> None:
    with pytest.raises(typer.BadParameter, match="device profile is not claim-eligible"):
        cli_main.validate_runtime_profile(
            "configs/runtime/devices/hpc_cpu.yaml",
            claim_eligible=True,
        )
