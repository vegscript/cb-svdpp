from pathlib import Path

from recsys_lab.utils.paths import discover_repo_root, repo_health_snapshot


def test_discover_repo_root_finds_project_root() -> None:
    start = Path(__file__).resolve().parent
    root = discover_repo_root(start)
    assert (root / "pyproject.toml").exists()
    assert (root / "AGENTS.md").exists()


def test_repo_health_snapshot_contains_required_paths() -> None:
    root = discover_repo_root(Path(__file__).resolve().parent)
    snapshot = repo_health_snapshot(root)
    assert snapshot["configs"] is True
    assert snapshot["docs"] is True
    assert snapshot["schema"] is True
