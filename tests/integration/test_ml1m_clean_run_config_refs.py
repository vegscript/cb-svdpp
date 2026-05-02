import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_completed_clean_ml1m_runs_reference_existing_model_configs() -> None:
    run_manifest_paths = sorted((REPO_ROOT / "artifacts" / "runs").glob("*ml1m*/run_manifest.json"))
    assert run_manifest_paths

    checked_manifest_count = 0
    for run_manifest_path in run_manifest_paths:
        manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            continue
        if bool(manifest.get("git", {}).get("dirty", False)):
            continue

        model_config_ref = str(manifest["model"]["config_ref"])
        model_config_path = REPO_ROOT / model_config_ref
        assert model_config_path.exists(), (
            f"missing config_ref target for clean ml1m run: {run_manifest_path} -> {model_config_ref}"
        )
        checked_manifest_count += 1

    assert checked_manifest_count >= 1
