from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from scripts import run_ml1m_fidelity_promotion as promotion

BASELINE_CONFIG = Path("configs/models/selected/ml1m/ml1m_cb_svdpp_stage0_transfer.yaml")
PROMOTION_CONFIGS = [
    Path("configs/models/candidates/ml1m/ml1m_cb_svdpp_fidelity_promotion_p1.yaml"),
    Path("configs/models/candidates/ml1m/ml1m_cb_svdpp_fidelity_promotion_p2.yaml"),
    Path("configs/models/candidates/ml1m/ml1m_cb_svdpp_fidelity_promotion_p3.yaml"),
]
SELECTED_CONFIG = Path("configs/models/selected/ml1m/ml1m_cb_svdpp_fidelity_promotion_selected.yaml")


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _row(
    *,
    label: str,
    role: str,
    validation_rmse: float,
    validation_mae: float = 0.7,
    test_rmse: float = 0.1,
    fit_model_seconds: float = 10.0,
    model_config_path: str = "configs/model.yaml",
) -> dict[str, str]:
    return {
        "label": label,
        "role": role,
        "model_config_path": model_config_path,
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


def test_promotion_configs_keep_baseline_epochs_and_induction_epochs() -> None:
    baseline = _load_yaml(BASELINE_CONFIG)

    for config_path in PROMOTION_CONFIGS:
        promoted = _load_yaml(config_path)

        assert promoted["training"]["epochs"] == baseline["training"]["epochs"] == 20
        assert promoted["training"]["latent_dim"] == baseline["training"]["latent_dim"] == 64
        assert promoted["clustering"]["induction"]["epochs"] == baseline["clustering"]["induction"]["epochs"] == 20
        assert promoted["clustering"]["induction"] == baseline["clustering"]["induction"]


def test_promotion_configs_change_only_alpha_learning_rate_lambda_q() -> None:
    baseline = _load_yaml(BASELINE_CONFIG)
    allowed_paths = {
        ("metadata",),
        ("notes",),
        ("clustering", "alpha"),
        ("training", "learning_rate"),
        ("training", "lambda_q"),
    }

    for config_path in PROMOTION_CONFIGS:
        promoted = _load_yaml(config_path)
        bad_diffs = [
            path
            for path, _before, _after in _diff_nested(baseline, promoted)
            if not any(path[: len(allowed)] == allowed for allowed in allowed_paths)
        ]

        assert bad_diffs == []


def test_promotion_decision_prefers_lower_validation_rmse() -> None:
    rows = [
        _row(label="baseline", role="baseline", validation_rmse=0.9),
        _row(label="promotion_p1", role="promoted", validation_rmse=0.8),
    ]

    decision = promotion.build_promotion_decision(rows)

    assert decision["decision"] == "PROMOTED_CANDIDATE_READY_FOR_FINAL_BAKEOFF"
    assert decision["winner_label"] == "promotion_p1"
    assert decision["validation_rmse_delta_vs_baseline"] == pytest.approx(-0.1)


def test_promotion_decision_does_not_use_test_rmse() -> None:
    rows = [
        _row(label="baseline", role="baseline", validation_rmse=0.8, test_rmse=0.99),
        _row(label="promotion_p1", role="promoted", validation_rmse=0.9, test_rmse=0.1),
    ]

    decision = promotion.build_promotion_decision(rows)

    assert decision["decision"] == "REJECT_PROMOTED_CANDIDATES"
    assert decision["winner_label"] == "baseline"


def test_promotion_script_rejects_non_ml1m_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.run_ml1m_selected_candidate_bakeoff.load_processed_dataset_manifest",
        lambda _path: {"dataset_short_name": "ml100k", "split_family": "benchmark_random_v1"},
    )

    with pytest.raises(ValueError, match="requires an ML1M processed manifest"):
        promotion._validate_ml1m_manifest(manifest_path, expected_split_family="benchmark_random_v1")


def test_promotion_script_caps_promoted_configs_at_three() -> None:
    with pytest.raises(ValueError, match="at most 3"):
        promotion._validate_promotion_config_list(
            promoted_config_paths=[
                Path("p1.yaml"),
                Path("p2.yaml"),
                Path("p3.yaml"),
                Path("p4.yaml"),
            ],
            labels=None,
        )


def test_selected_config_written_only_when_promoted_candidate_wins(tmp_path: Path) -> None:
    selected_target = tmp_path / "selected.yaml"
    reject_rows = [
        _row(label="baseline", role="baseline", validation_rmse=0.8),
        _row(label="promotion_p1", role="promoted", validation_rmse=0.9),
    ]
    adopt_rows = [
        _row(label="baseline", role="baseline", validation_rmse=0.9),
        _row(label="promotion_p1", role="promoted", validation_rmse=0.8),
    ]

    _write_selected_config_if_promoted_wins(rows=reject_rows, source=PROMOTION_CONFIGS[0], target=selected_target)
    assert not selected_target.exists()

    _write_selected_config_if_promoted_wins(rows=adopt_rows, source=PROMOTION_CONFIGS[0], target=selected_target)
    assert selected_target.exists()
    assert _load_yaml(selected_target)["metadata"]["status"] == "fidelity_promotion_candidate"


def _diff_nested(before, after, path: tuple[str, ...] = ()):
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            yield from _diff_nested(before.get(key), after.get(key), (*path, str(key)))
    elif before != after:
        yield path, before, after


def _write_selected_config_if_promoted_wins(*, rows: list[dict[str, str]], source: Path, target: Path) -> None:
    decision = promotion.build_promotion_decision(rows)
    if decision["decision"] == "PROMOTED_CANDIDATE_READY_FOR_FINAL_BAKEOFF":
        shutil.copyfile(source, target)
