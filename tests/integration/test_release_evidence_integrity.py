import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_FACING_DOCS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "publish_readiness_matrix.md",
    REPO_ROOT / "docs" / "report" / "project_report.md",
    REPO_ROOT / "docs" / "evidence" / "release" / "2026-05-02_release_hygiene.md",
    REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-02_public_clean_import.md",
)
REPRODUCTION_EVIDENCE = REPO_ROOT / "docs" / "evidence" / "reproduction" / "2026-05-02_public_clean_import.md"
EVIDENCE_PATH_RE = re.compile(r"docs/evidence/[A-Za-z0-9_./-]+\.md")
PASSED_COUNT_RE = re.compile(r"`(\d+) passed")
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


def test_release_facing_evidence_references_resolve() -> None:
    references: set[str] = set()
    for path in RELEASE_FACING_DOCS:
        references.update(EVIDENCE_PATH_RE.findall(path.read_text(encoding="utf-8")))

    assert references
    missing = sorted(reference for reference in references if not (REPO_ROOT / reference).is_file())
    assert not missing


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


def test_readme_full_suite_readout_matches_reproduction_evidence() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    reproduction = REPRODUCTION_EVIDENCE.read_text(encoding="utf-8")

    readme_match = re.search(r"full test suite from the `uv` environment: `(\d+) passed`", readme)
    assert readme_match is not None

    reproduction_counts = [int(count) for count in PASSED_COUNT_RE.findall(reproduction)]
    assert reproduction_counts
    assert int(readme_match.group(1)) == max(reproduction_counts)
