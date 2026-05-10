from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_ml1m_selected_candidate_bakeoff as bakeoff


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


def test_bakeoff_decision_adopts_lower_validation_rmse() -> None:
    rows = [
        _row(label="baseline", validation_rmse=0.9, validation_mae=0.8),
        _row(label="candidate", validation_rmse=0.8, validation_mae=0.7),
    ]

    decision = bakeoff.build_bakeoff_decision(
        rows=rows,
        baseline_label="baseline",
        candidate_label="candidate",
    )

    assert decision["decision"] == "ADOPT_SELECTED_CONFIG"
    assert decision["winner"] == "candidate"
    assert decision["validation_rmse_delta_vs_baseline"] == pytest.approx(-0.1)


def test_bakeoff_decision_rejects_equal_or_worse_validation_rmse() -> None:
    equal_rows = [
        _row(label="baseline", validation_rmse=0.9),
        _row(label="candidate", validation_rmse=0.9),
    ]
    worse_rows = [
        _row(label="baseline", validation_rmse=0.9),
        _row(label="candidate", validation_rmse=1.0),
    ]

    equal_decision = bakeoff.build_bakeoff_decision(
        rows=equal_rows,
        baseline_label="baseline",
        candidate_label="candidate",
    )
    worse_decision = bakeoff.build_bakeoff_decision(
        rows=worse_rows,
        baseline_label="baseline",
        candidate_label="candidate",
    )

    assert equal_decision["decision"] == "REJECT_SELECTED_CONFIG"
    assert worse_decision["decision"] == "REJECT_SELECTED_CONFIG"


def test_bakeoff_decision_does_not_use_test_rmse() -> None:
    rows = [
        _row(label="baseline", validation_rmse=0.8, test_rmse=0.99),
        _row(label="candidate", validation_rmse=0.9, test_rmse=0.1),
    ]

    decision = bakeoff.build_bakeoff_decision(
        rows=rows,
        baseline_label="baseline",
        candidate_label="candidate",
    )

    assert decision["decision"] == "REJECT_SELECTED_CONFIG"
    assert decision["winner"] == "baseline"


def test_bakeoff_script_rejects_non_ml1m_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        bakeoff,
        "load_processed_dataset_manifest",
        lambda _path: {"dataset_short_name": "ml100k", "split_family": "benchmark_random_v1"},
    )

    with pytest.raises(ValueError, match="requires an ML1M processed manifest"):
        bakeoff._validate_ml1m_manifest(manifest_path, expected_split_family="benchmark_random_v1")


def test_bakeoff_script_caps_configs_at_three() -> None:
    with pytest.raises(ValueError, match="at most 3"):
        bakeoff._validate_config_list(
            model_config_paths=[
                Path("a.yaml"),
                Path("b.yaml"),
                Path("c.yaml"),
                Path("d.yaml"),
            ],
            labels=None,
        )


def test_bakeoff_cli_accepts_baseline_and_candidate_config() -> None:
    paths, labels = bakeoff._resolve_cli_config_list(
        model_config_paths=[],
        labels=None,
        baseline_config=Path("baseline.yaml"),
        candidate_config=Path("candidate.yaml"),
    )

    assert paths == [Path("baseline.yaml"), Path("candidate.yaml")]
    assert labels == ["baseline_stage0_transfer", "small_study_v2_candidate"]


def test_selected_config_written_only_on_adopt(tmp_path: Path) -> None:
    candidate_source = tmp_path / "candidate.yaml"
    selected_target = tmp_path / "selected.yaml"
    candidate_source.write_text("model: candidate\n", encoding="utf-8")
    reject_rows = [
        _row(label="baseline", validation_rmse=0.8),
        _row(label="candidate", validation_rmse=0.9),
    ]
    adopt_rows = [
        _row(label="baseline", validation_rmse=0.9),
        _row(label="candidate", validation_rmse=0.8),
    ]

    reject_decision = bakeoff.write_bakeoff_decision_artifacts(
        output_dir=tmp_path,
        rows=reject_rows,
        baseline_label="baseline",
        candidate_label="candidate",
        selected_config_source=candidate_source,
        final_selected_config_path=selected_target,
    )
    assert reject_decision["decision"] == "REJECT_SELECTED_CONFIG"
    assert reject_decision["adopted_config_path"] is None
    assert not selected_target.exists()

    adopt_decision = bakeoff.write_bakeoff_decision_artifacts(
        output_dir=tmp_path,
        rows=adopt_rows,
        baseline_label="baseline",
        candidate_label="candidate",
        selected_config_source=candidate_source,
        final_selected_config_path=selected_target,
    )
    assert adopt_decision["decision"] == "ADOPT_SELECTED_CONFIG"
    assert adopt_decision["adopted_config_path"] == str(selected_target)
    assert selected_target.read_text(encoding="utf-8") == "model: candidate\n"
