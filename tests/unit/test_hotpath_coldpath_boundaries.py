from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

STRICT_HOTPATH = REPO_ROOT / "src" / "recsys_lab" / "models" / "kernels.py"

MODEL_HOTPATH_FILES = [
    REPO_ROOT / "src" / "recsys_lab" / "models" / "biased_mf.py",
    REPO_ROOT / "src" / "recsys_lab" / "models" / "svdpp.py",
    REPO_ROOT / "src" / "recsys_lab" / "models" / "asymmetric_svd.py",
    REPO_ROOT / "src" / "recsys_lab" / "models" / "asvdpp.py",
    REPO_ROOT / "src" / "recsys_lab" / "models" / "cb_svdpp.py",
    REPO_ROOT / "src" / "recsys_lab" / "models" / "cb_asvdpp.py",
    REPO_ROOT / "src" / "recsys_lab" / "models" / "inference.py",
]

STRICT_FORBIDDEN = [
    "yaml",
    "json",
    "pydantic",
    "Path",
    "open(",
    "write_json",
    "dump_yaml_file",
    "load_yaml_file",
    "validate_manifest_file",
    "discover_repo_root",
    "repo_path_string",
    "markdown",
    "evidence",
    "report",
    "argparse",
    "typer",
    "click",
]

MODEL_FORBIDDEN = [
    "load_yaml_file",
    "dump_yaml_file",
    "write_json",
    "validate_manifest_file",
    "discover_repo_root",
    "repo_path_string",
    "docs/",
    "evidence",
    "reporting",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assert_no_forbidden_terms(path: Path, forbidden_terms: list[str]) -> None:
    text = _read(path)
    hits = [term for term in forbidden_terms if term in text]
    assert hits == [], f"{path.relative_to(REPO_ROOT)} contains forbidden hotpath terms: {hits}"


def test_strict_hotpath_kernels_has_no_coldpath_imports_or_terms() -> None:
    _assert_no_forbidden_terms(STRICT_HOTPATH, STRICT_FORBIDDEN)


def test_strict_hotpath_kernels_has_no_file_io_patterns() -> None:
    file_io_terms = ["Path", "open(", "write_json", "dump_yaml_file", "load_yaml_file"]

    _assert_no_forbidden_terms(STRICT_HOTPATH, file_io_terms)


def test_strict_hotpath_kernels_has_no_config_manifest_reporting_terms() -> None:
    coldpath_terms = [
        "yaml",
        "json",
        "pydantic",
        "validate_manifest_file",
        "discover_repo_root",
        "repo_path_string",
        "markdown",
        "evidence",
        "report",
    ]

    _assert_no_forbidden_terms(STRICT_HOTPATH, coldpath_terms)


def test_model_hotpath_files_do_not_import_config_loaders_or_manifest_writers() -> None:
    for path in MODEL_HOTPATH_FILES:
        _assert_no_forbidden_terms(path, MODEL_FORBIDDEN)


def test_experiment_runner_is_allowed_to_contain_coldpath_terms() -> None:
    runner_path = REPO_ROOT / "src" / "recsys_lab" / "experiments" / "unified_runner.py"
    text = _read(runner_path)

    assert "write_json" in text
    assert "validate_manifest_file" in text
    assert "repo_path_string" in text


def test_reporting_scripts_and_docs_are_not_part_of_hotpath_checks() -> None:
    checked_paths = {STRICT_HOTPATH, *MODEL_HOTPATH_FILES}
    checked_relative_paths = [path.relative_to(REPO_ROOT) for path in checked_paths]

    assert all("reporting" not in path.parts for path in checked_relative_paths)
    assert all("scripts" not in path.parts for path in checked_relative_paths)
    assert all("docs" not in path.parts for path in checked_relative_paths)
