from __future__ import annotations

import csv
from pathlib import Path


OFFICIAL_ML100K_FOLDS = (1, 2, 3, 4, 5)


def official_ml100k_split_paths(raw_dir: Path) -> dict[int, dict[str, Path]] | None:
    fold_paths: dict[int, dict[str, Path]] = {}
    for fold_index in OFFICIAL_ML100K_FOLDS:
        train_path = raw_dir / f"u{fold_index}.base"
        test_path = raw_dir / f"u{fold_index}.test"
        if not train_path.exists() or not test_path.exists():
            return None
        fold_paths[fold_index] = {
            "train": train_path,
            "test": test_path,
        }
    return fold_paths


def read_legacy_ml100k_split(path: Path) -> list[tuple[int, int, float, int]]:
    records: list[tuple[int, int, float, int]] = []
    with path.open("r", encoding="latin-1", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) != 4:
                raise ValueError(f"unexpected row width in ml100k split file: {path}")
            records.append((int(row[0]), int(row[1]), float(row[2]), int(row[3])))
    return records
