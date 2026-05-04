from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_model_config_structure_is_frozen() -> None:
    model_config_dir = REPO_ROOT / "configs" / "models"

    top_level_dirs = {path.name for path in model_config_dir.iterdir() if path.is_dir()}

    assert top_level_dirs == {"selected", "archive"}
    assert (model_config_dir / "archive" / "development").is_dir()
    assert (model_config_dir / "archive" / "tuned").is_dir()
    assert not (model_config_dir / "tuned").exists()
    assert not (model_config_dir / "development").exists()


def test_tuning_config_structure_is_frozen() -> None:
    tuning_config_dir = REPO_ROOT / "configs" / "experiments" / "tuning"

    top_level_dirs = {path.name for path in tuning_config_dir.iterdir() if path.is_dir()}

    assert top_level_dirs == {"active", "templates", "archive"}
    assert any((tuning_config_dir / "active").glob("*.yaml"))


def test_evidence_reproduction_structure_is_frozen() -> None:
    evidence_dir = REPO_ROOT / "docs" / "evidence"
    reproduction_dir = evidence_dir / "reproduction"

    top_level_dirs = {path.name for path in reproduction_dir.iterdir() if path.is_dir()}

    assert (evidence_dir / "current_evidence_index.md").is_file()
    assert top_level_dirs == {"current", "archive"}
    assert any((reproduction_dir / "current").glob("*.md"))
    assert any((reproduction_dir / "archive").glob("*.md"))
