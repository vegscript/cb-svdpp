from __future__ import annotations

from pathlib import Path


def discover_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "AGENTS.md").exists():
            return candidate
    raise FileNotFoundError("could not discover repo root from current path")


def required_repo_paths(repo_root: Path) -> dict[str, Path]:
    return {
        "pyproject": repo_root / "pyproject.toml",
        "agents": repo_root / "AGENTS.md",
        "docs": repo_root / "docs",
        "configs": repo_root / "configs",
        "data": repo_root / "data",
        "artifacts": repo_root / "artifacts",
        "schema": repo_root / "schema",
        "src": repo_root / "src",
        "tests": repo_root / "tests",
        "scripts": repo_root / "scripts",
    }


def repo_health_snapshot(repo_root: Path) -> dict[str, bool]:
    return {name: path.exists() for name, path in required_repo_paths(repo_root).items()}


def repo_path_string(path: Path, *, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)
