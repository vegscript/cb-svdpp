import ast
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

HOTPATH_PREPARATION_FILES = [
    REPO_ROOT / "src" / "recsys_lab" / "data" / "histories.py",
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

STRICT_FORBIDDEN_IMPORTS = [
    "recsys_lab.config",
    "recsys_lab.benchmarks",
    "recsys_lab.cli",
    "recsys_lab.experiments",
    "recsys_lab.reporting",
    "recsys_lab.utils.atomic_io",
    "recsys_lab.utils.manifests",
    "recsys_lab.utils.paths",
    "pydantic",
    "yaml",
    "json",
    "pathlib",
    "argparse",
    "typer",
    "click",
    "logging",
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

MODEL_FORBIDDEN_IMPORTS = [
    "recsys_lab.config",
    "recsys_lab.benchmarks",
    "recsys_lab.cli",
    "recsys_lab.experiments",
    "recsys_lab.reporting",
    "recsys_lab.utils.atomic_io",
    "recsys_lab.utils.manifests",
    "recsys_lab.utils.paths",
    "pydantic",
    "yaml",
    "json",
    "pathlib",
    "argparse",
    "typer",
    "click",
    "logging",
]

HOTPATH_PREPARATION_FORBIDDEN = [
    "yaml",
    "json",
    "Path",
    "open(",
    "atomic_io",
    "write_json",
    "dump_yaml_file",
    "load_yaml_file",
    "validate_manifest_file",
    "discover_repo_root",
    "repo_path_string",
    "evidence",
    "experiments",
    "reporting",
    "cli",
]

HOTPATH_PREPARATION_FORBIDDEN_IMPORTS = [
    "recsys_lab.benchmarks",
    "recsys_lab.config",
    "recsys_lab.cli",
    "recsys_lab.experiments",
    "recsys_lab.reporting",
    "recsys_lab.utils.atomic_io",
    "recsys_lab.utils.manifests",
    "recsys_lab.utils.paths",
    "yaml",
    "json",
    "pathlib",
    "argparse",
    "typer",
    "click",
    "logging",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(_read(path), filename=str(path))
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            imports.add(node.module)
            imports.update(f"{node.module}.{alias.name}" for alias in node.names if alias.name != "*")

    return imports


def _forbidden_import_hits(imports: set[str], forbidden_roots: list[str]) -> list[str]:
    hits = []
    for imported in imports:
        for forbidden in forbidden_roots:
            if imported == forbidden or imported.startswith(f"{forbidden}."):
                hits.append(imported)
                break
    return sorted(hits)


def _assert_no_forbidden_terms(path: Path, forbidden_terms: list[str]) -> None:
    text = _read(path)
    hits = [term for term in forbidden_terms if term in text]
    assert hits == [], f"{path.relative_to(REPO_ROOT)} contains forbidden hotpath terms: {hits}"


def _assert_no_forbidden_imports(path: Path, forbidden_roots: list[str]) -> None:
    hits = _forbidden_import_hits(_imported_modules(path), forbidden_roots)
    assert hits == [], f"{path.relative_to(REPO_ROOT)} imports forbidden hotpath modules: {hits}"


def test_strict_hotpath_kernels_has_no_coldpath_imports_or_terms() -> None:
    _assert_no_forbidden_terms(STRICT_HOTPATH, STRICT_FORBIDDEN)


def test_strict_hotpath_kernels_has_no_forbidden_ast_imports() -> None:
    _assert_no_forbidden_imports(STRICT_HOTPATH, STRICT_FORBIDDEN_IMPORTS)


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


def test_model_hotpath_files_have_no_forbidden_ast_imports() -> None:
    for path in MODEL_HOTPATH_FILES:
        _assert_no_forbidden_imports(path, MODEL_FORBIDDEN_IMPORTS)


def test_hotpath_preparation_histories_has_no_coldpath_imports_or_terms() -> None:
    for path in HOTPATH_PREPARATION_FILES:
        _assert_no_forbidden_terms(path, HOTPATH_PREPARATION_FORBIDDEN)
        _assert_no_forbidden_imports(path, HOTPATH_PREPARATION_FORBIDDEN_IMPORTS)


def test_import_extractor_detects_forbidden_from_import(tmp_path: Path) -> None:
    path = tmp_path / "bad_hotpath.py"
    path.write_text("from recsys_lab.experiments.performance import StageProfiler\n", encoding="utf-8")

    imports = _imported_modules(path)

    assert "recsys_lab.experiments.performance" in imports
    assert "recsys_lab.experiments.performance.StageProfiler" in imports


def test_import_guard_flags_forbidden_from_import(tmp_path: Path) -> None:
    path = tmp_path / "bad_hotpath.py"
    path.write_text("from recsys_lab.experiments.performance import StageProfiler\n", encoding="utf-8")

    hits = _forbidden_import_hits(_imported_modules(path), MODEL_FORBIDDEN_IMPORTS)

    assert hits == [
        "recsys_lab.experiments.performance",
        "recsys_lab.experiments.performance.StageProfiler",
    ]


def test_import_guard_flags_benchmark_back_import(tmp_path: Path) -> None:
    path = tmp_path / "bad_hotpath.py"
    path.write_text("from recsys_lab.benchmarks.kernel_harness import run_kernel_benchmark\n", encoding="utf-8")

    hits = _forbidden_import_hits(_imported_modules(path), MODEL_FORBIDDEN_IMPORTS)

    assert hits == [
        "recsys_lab.benchmarks.kernel_harness",
        "recsys_lab.benchmarks.kernel_harness.run_kernel_benchmark",
    ]


def test_experiment_runner_is_allowed_to_contain_coldpath_terms() -> None:
    runner_path = REPO_ROOT / "src" / "recsys_lab" / "experiments" / "unified_runner.py"
    text = _read(runner_path)

    assert "write_json" in text
    assert "validate_manifest_file" in text
    assert "repo_path_string" in text


def test_reporting_scripts_and_docs_are_not_part_of_hotpath_checks() -> None:
    checked_paths = {STRICT_HOTPATH, *MODEL_HOTPATH_FILES, *HOTPATH_PREPARATION_FILES}
    checked_relative_paths = [path.relative_to(REPO_ROOT) for path in checked_paths]

    assert all("reporting" not in path.parts for path in checked_relative_paths)
    assert all("scripts" not in path.parts for path in checked_relative_paths)
    assert all("docs" not in path.parts for path in checked_relative_paths)


def test_cache_modules_are_not_checked_as_pure_hotpath_preparation() -> None:
    checked_paths = {STRICT_HOTPATH, *MODEL_HOTPATH_FILES, *HOTPATH_PREPARATION_FILES}

    assert REPO_ROOT / "src" / "recsys_lab" / "data" / "training_index_cache.py" not in checked_paths
    assert REPO_ROOT / "src" / "recsys_lab" / "clustering" / "cache.py" not in checked_paths
