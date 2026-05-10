from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_ml1m_final_bakeoff as final_bakeoff


def _row(
    *,
    label: str,
    validation_rmse: float,
    validation_mae: float = 0.7,
    test_rmse: float = 0.1,
    fit_model_seconds: float = 10.0,
) -> dict[str, str]:
    return {
        "label": label,
        "model_config_path": f"configs/{label}.yaml",
        "execution_status": "succeeded",
        "run_id": f"run_{label}",
        "run_dir": f"artifacts/runs/run_{label}",
        "validation_rmse": str(validation_rmse),
        "validation_mae": str(validation_mae),
        "test_rmse": str(test_rmse),
        "test_mae": "0.1",
        "fit_model_seconds": str(fit_model_seconds),
        "total_wall_seconds": "12.0",
        "cluster_total_seconds": "1.0",
        "cluster_cache_status": "hit",
        "user_cluster_history_cache_status": "hit",
        "notes": "",
    }


def test_final_bakeoff_runs_exactly_two_configs(monkeypatch) -> None:
    captured = {}

    def fake_run_bakeoff(**kwargs):
        captured.update(kwargs)
        return {"run_count": len(kwargs["model_config_paths"])}

    monkeypatch.setattr(final_bakeoff, "run_bakeoff", fake_run_bakeoff)

    payload = final_bakeoff.run_final_bakeoff(
        baseline_config_path=Path("baseline.yaml"),
        selected_config_path=Path("selected.yaml"),
        output_dir=Path("artifacts/final_bakeoff/test"),
        processed_manifest_path=Path("manifest.json"),
        runtime_config_path=Path("runtime.yaml"),
        device_config_path=Path("device.yaml"),
        cache_root=None,
        repo_root=Path("."),
        model_name="cb_svdpp",
        split_family="benchmark_random_v1",
        train_ratio=0.8,
        validation_ratio=0.1,
        split_seed=1,
        model_seed=1,
        evaluate_test=False,
        split_cache="auto",
        use_training_index_cache=True,
        use_cluster_artifact_cache=True,
        overwrite=True,
    )

    assert payload["run_count"] == 2
    assert captured["model_config_paths"] == [Path("baseline.yaml"), Path("selected.yaml")]
    assert captured["labels"] == ["baseline_stage0_transfer", "fidelity_promotion_selected"]
    assert "total_wall_seconds must not be interpreted" in payload["runtime_interpretation"]


def test_final_bakeoff_decision_adopts_lower_validation_rmse() -> None:
    rows = [
        _row(label="baseline_stage0_transfer", validation_rmse=0.9, validation_mae=0.8),
        _row(label="fidelity_promotion_selected", validation_rmse=0.8, validation_mae=0.7),
    ]

    decision = final_bakeoff.build_final_bakeoff_decision(rows)

    assert decision["decision"] == "ADOPT_PROMOTED_CONFIG"
    assert decision["winner"] == "fidelity_promotion_selected"
    assert decision["validation_rmse_delta_vs_baseline"] == pytest.approx(-0.1)


def test_final_bakeoff_decision_rejects_equal_or_worse_validation_rmse() -> None:
    equal_rows = [
        _row(label="baseline_stage0_transfer", validation_rmse=0.9),
        _row(label="fidelity_promotion_selected", validation_rmse=0.9),
    ]
    worse_rows = [
        _row(label="baseline_stage0_transfer", validation_rmse=0.9),
        _row(label="fidelity_promotion_selected", validation_rmse=1.0),
    ]

    equal_decision = final_bakeoff.build_final_bakeoff_decision(equal_rows)
    worse_decision = final_bakeoff.build_final_bakeoff_decision(worse_rows)

    assert equal_decision["decision"] == "REJECT_PROMOTED_CONFIG"
    assert worse_decision["decision"] == "REJECT_PROMOTED_CONFIG"


def test_final_bakeoff_decision_does_not_use_test_rmse() -> None:
    rows = [
        _row(label="baseline_stage0_transfer", validation_rmse=0.8, test_rmse=0.99),
        _row(label="fidelity_promotion_selected", validation_rmse=0.9, test_rmse=0.1),
    ]

    decision = final_bakeoff.build_final_bakeoff_decision(rows)

    assert decision["decision"] == "REJECT_PROMOTED_CONFIG"
    assert decision["winner"] == "baseline_stage0_transfer"


def test_final_bakeoff_script_rejects_non_ml1m_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.run_ml1m_selected_candidate_bakeoff.load_processed_dataset_manifest",
        lambda _path: {"dataset_short_name": "ml100k", "split_family": "benchmark_random_v1"},
    )

    from scripts.run_ml1m_selected_candidate_bakeoff import _validate_ml1m_manifest

    with pytest.raises(ValueError, match="requires an ML1M processed manifest"):
        _validate_ml1m_manifest(manifest_path, expected_split_family="benchmark_random_v1")


def test_final_bakeoff_script_caps_configs_at_two(monkeypatch) -> None:
    captured = {}

    def fake_run_bakeoff(**kwargs):
        captured.update(kwargs)
        return {"run_count": len(kwargs["model_config_paths"])}

    monkeypatch.setattr(final_bakeoff, "run_bakeoff", fake_run_bakeoff)

    final_bakeoff.run_final_bakeoff(
        baseline_config_path=Path("baseline.yaml"),
        selected_config_path=Path("selected.yaml"),
        output_dir=Path("artifacts/final_bakeoff/test"),
        processed_manifest_path=Path("manifest.json"),
        runtime_config_path=Path("runtime.yaml"),
        device_config_path=Path("device.yaml"),
        cache_root=None,
        repo_root=Path("."),
        model_name="cb_svdpp",
        split_family="benchmark_random_v1",
        train_ratio=0.8,
        validation_ratio=0.1,
        split_seed=1,
        model_seed=1,
        evaluate_test=False,
        split_cache="auto",
        use_training_index_cache=True,
        use_cluster_artifact_cache=True,
        overwrite=True,
    )

    assert len(captured["model_config_paths"]) == 2


def test_config_status_updated_only_after_adopt() -> None:
    reject_rows = [
        _row(label="baseline_stage0_transfer", validation_rmse=0.8),
        _row(label="fidelity_promotion_selected", validation_rmse=0.9),
    ]
    adopt_rows = [
        _row(label="baseline_stage0_transfer", validation_rmse=0.9),
        _row(label="fidelity_promotion_selected", validation_rmse=0.8),
    ]

    assert _status_after_decision(reject_rows) != "selected"
    assert _status_after_decision(adopt_rows) == "selected"


def _status_after_decision(rows: list[dict[str, str]]) -> str:
    decision = final_bakeoff.build_final_bakeoff_decision(rows)
    if decision["decision"] == "ADOPT_PROMOTED_CONFIG":
        return "selected"
    return "not_adopted"
