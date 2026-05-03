import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_FACING_DOCS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "publish_readiness_matrix.md",
    REPO_ROOT / "docs" / "report" / "project_report.md",
    REPO_ROOT / "docs" / "evidence" / "release" / "2026-05-02_release_hygiene.md",
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-02_public_clean_import.md",
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-03_public_path_hygiene.md",
)
PUBLIC_MARKDOWN_LINK_DOCS = (
    *RELEASE_FACING_DOCS,
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "data" / "README.md",
    REPO_ROOT / "docs" / "architecture.md",
    REPO_ROOT / "docs" / "evidence" / "README.md",
    REPO_ROOT / "docs" / "manifest_contract.md",
    REPO_ROOT / "docs" / "math" / "notation.md",
    REPO_ROOT / "docs" / "report" / "report_contract.md",
    REPO_ROOT / "docs" / "research_integrity.md",
)
REPRODUCTION_EVIDENCE_FILES = (
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-02_public_clean_import.md",
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-03_public_path_hygiene.md",
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-03_cb_svdpp_g6_validation_grid_contract.md",
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-03_cb_svdpp_g6_validation_grid_run.md",
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-03_cb_svdpp_g6_outer_benchmark_contract.md",
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-03_cb_svdpp_g6_outer_benchmark_run.md",
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-03_cb_asvdpp_hotpath_decision_g7.md",
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-03_cb_asvdpp_hotpath_remediation_contract_g8.md",
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-03_cb_asvdpp_hotpath_prechange_baseline_g9.md",
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-03_cb_asvdpp_hotpath_postchange_benchmark_g10.md",
)
EVIDENCE_PATH_RE = re.compile(r"docs/evidence/[A-Za-z0-9_./-]+\.md")
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
PASSED_COUNT_RE = re.compile(r"`(\d+) passed")
URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
GENERATED_ZONE_PREFIXES = (
    "data/raw/",
    "data/interim/",
    "data/processed/",
    "artifacts/runs/",
    "artifacts/benchmarks/",
    "artifacts/figures/",
    "artifacts/debug/",
    "artifacts/local/",
)
ALLOWED_GENERATED_ZONE_FILENAMES = {".gitkeep", "README.md"}
LOCAL_PATH_PATTERNS = (
    "G:" + "/" + "Meine" + " Ablage",
    "G:" + "\\" + "Meine" + " Ablage",
    "C:" + "/" + "Users" + "/",
    "C:" + "\\" + "Users" + "\\",
)


def test_release_facing_evidence_references_resolve() -> None:
    references: set[str] = set()
    for path in RELEASE_FACING_DOCS:
        references.update(EVIDENCE_PATH_RE.findall(path.read_text(encoding="utf-8")))

    assert references
    missing = sorted(reference for reference in references if not (REPO_ROOT / reference).is_file())
    assert not missing


def test_public_markdown_links_resolve() -> None:
    missing: list[str] = []
    for path in PUBLIC_MARKDOWN_LINK_DOCS:
        content = path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK_RE.finditer(content):
            raw_target = match.group(1).strip()
            target_without_anchor = raw_target.split("#", 1)[0]
            if (
                not target_without_anchor
                or target_without_anchor.startswith("//")
                or URI_SCHEME_RE.match(target_without_anchor)
            ):
                continue
            target_path = path.parent / unquote(target_without_anchor.strip("<>"))
            if not target_path.exists():
                source = path.relative_to(REPO_ROOT)
                missing.append(f"{source}: {raw_target}")

    assert not sorted(missing)


def test_generated_data_and_artifact_zones_track_only_placeholders() -> None:
    result = subprocess.run(
        ["git", "ls-files", "data", "artifacts"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked_paths = result.stdout.splitlines()

    offenders = [
        path
        for path in tracked_paths
        if path.startswith(GENERATED_ZONE_PREFIXES)
        and Path(path).name not in ALLOWED_GENERATED_ZONE_FILENAMES
    ]
    assert not offenders


def test_public_tree_does_not_contain_local_absolute_paths() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    offenders: list[str] = []
    for relative_path in result.stdout.splitlines():
        path = REPO_ROOT / relative_path
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in LOCAL_PATH_PATTERNS:
            if pattern in content:
                offenders.append(relative_path)
                break

    assert not sorted(offenders)


def test_readme_full_suite_readout_matches_reproduction_evidence() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    readme_match = re.search(r"full test suite from the `uv` environment: `(\d+) passed`", readme)
    assert readme_match is not None

    reproduction_counts = [
        int(count)
        for path in REPRODUCTION_EVIDENCE_FILES
        for count in PASSED_COUNT_RE.findall(path.read_text(encoding="utf-8"))
    ]
    assert reproduction_counts
    assert int(readme_match.group(1)) == max(reproduction_counts)
