"""Compare paired real run artifacts without adding benchmark claims."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

FIELDS = [
    "model",
    "before_run_id",
    "after_run_id",
    "dataset",
    "same_contract_status",
    "test_rmse_before",
    "test_rmse_after",
    "test_rmse_delta",
    "test_mae_before",
    "test_mae_after",
    "test_mae_delta",
    "fit_model_seconds_before",
    "fit_model_seconds_after",
    "fit_model_seconds_delta",
    "fit_model_seconds_ratio_after_before",
    "train_time_total_before",
    "train_time_total_after",
    "train_time_total_ratio_after_before",
    "ratings_per_second_before",
    "ratings_per_second_after",
    "peak_memory_mb_before",
    "peak_memory_mb_after",
    "estimated_factor_touches_before",
    "estimated_factor_touches_after",
    "fit_seconds_per_million_estimated_touches_before",
    "fit_seconds_per_million_estimated_touches_after",
    "target_or_control",
    "notes",
]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = [_compare_pair(_parse_pair(pair)) for pair in args.pair]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare before/after real run metrics from run artifact directories."
    )
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help="Pair as model,before_run_dir,after_run_dir,target|control.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    return parser.parse_args(argv)


def _parse_pair(raw: str) -> tuple[str, Path, Path, str]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("--pair must have format model,before_run_dir,after_run_dir,target|control")
    model, before_dir, after_dir, role = parts
    if role not in {"target", "control"}:
        raise ValueError("--pair role must be target or control")
    return model, Path(before_dir), Path(after_dir), role


def _compare_pair(pair: tuple[str, Path, Path, str]) -> dict[str, Any]:
    model, before_dir, after_dir, role = pair
    before = _load_run(before_dir)
    after = _load_run(after_dir)
    notes: list[str] = []
    status = _same_contract_status(model, before, after, notes)

    test_rmse_before = _metric(before["metrics"], "test_rmse")
    test_rmse_after = _metric(after["metrics"], "test_rmse")
    test_mae_before = _metric(before["metrics"], "test_mae")
    test_mae_after = _metric(after["metrics"], "test_mae")
    fit_before = _fit_model_seconds(before["performance"])
    fit_after = _fit_model_seconds(after["performance"])
    train_before = _train_time_total(before["kernel"])
    train_after = _train_time_total(after["kernel"])

    row = {
        "model": model,
        "before_run_id": before["run_id"],
        "after_run_id": after["run_id"],
        "dataset": _dataset(before["manifest"]) or _dataset(after["manifest"]) or "",
        "same_contract_status": status,
        "test_rmse_before": _format(test_rmse_before),
        "test_rmse_after": _format(test_rmse_after),
        "test_rmse_delta": _format(_delta(test_rmse_before, test_rmse_after)),
        "test_mae_before": _format(test_mae_before),
        "test_mae_after": _format(test_mae_after),
        "test_mae_delta": _format(_delta(test_mae_before, test_mae_after)),
        "fit_model_seconds_before": _format(fit_before),
        "fit_model_seconds_after": _format(fit_after),
        "fit_model_seconds_delta": _format(_delta(fit_before, fit_after)),
        "fit_model_seconds_ratio_after_before": _format(_ratio(fit_before, fit_after)),
        "train_time_total_before": _format(train_before),
        "train_time_total_after": _format(train_after),
        "train_time_total_ratio_after_before": _format(_ratio(train_before, train_after)),
        "ratings_per_second_before": _format(_ratings_per_second(before["kernel"])),
        "ratings_per_second_after": _format(_ratings_per_second(after["kernel"])),
        "peak_memory_mb_before": _format(_peak_memory_mb(before["performance"])),
        "peak_memory_mb_after": _format(_peak_memory_mb(after["performance"])),
        "estimated_factor_touches_before": _format(_estimated_touches(before["kernel"])),
        "estimated_factor_touches_after": _format(_estimated_touches(after["kernel"])),
        "fit_seconds_per_million_estimated_touches_before": _format(
            _fit_seconds_per_million_touches(before["kernel"])
        ),
        "fit_seconds_per_million_estimated_touches_after": _format(
            _fit_seconds_per_million_touches(after["kernel"])
        ),
        "target_or_control": role,
        "notes": "; ".join(notes),
    }
    return row


def _load_run(run_dir: Path) -> dict[str, Any]:
    return {
        "run_id": run_dir.name,
        "manifest": _read_json(run_dir / "run_manifest.json"),
        "metrics": _read_json(run_dir / "metrics.json"),
        "performance": _read_json(run_dir / "performance_profile.json"),
        "kernel": _read_json(run_dir / "kernel_profile.json"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _same_contract_status(
    expected_model: str, before: dict[str, Any], after: dict[str, Any], notes: list[str]
) -> str:
    failures: list[str] = []
    warnings: list[str] = []

    checks = {
        "dataset": (_dataset(before["manifest"]), _dataset(after["manifest"])),
        "model": (_model(before["manifest"]), _model(after["manifest"])),
        "split_family": (_split_family(before["manifest"]), _split_family(after["manifest"])),
        "split_seed": (_seed(before["manifest"], "split_seed"), _seed(after["manifest"], "split_seed")),
        "model_seed": (_seed(before["manifest"], "model_seed"), _seed(after["manifest"], "model_seed")),
        "dtype": (_runtime(before["manifest"], "dtype"), _runtime(after["manifest"], "dtype")),
        "epochs": (_kernel_value(before["kernel"], "epochs"), _kernel_value(after["kernel"], "epochs")),
        "latent_dim": (
            _kernel_value(before["kernel"], "latent_dim"),
            _kernel_value(after["kernel"], "latent_dim"),
        ),
        "model_config": (_model_config(before["manifest"]), _model_config(after["manifest"])),
        "device_profile": (
            _runtime(before["manifest"], "device_profile"),
            _runtime(after["manifest"], "device_profile"),
        ),
    }
    for name, (before_value, after_value) in checks.items():
        if before_value in {None, ""} or after_value in {None, ""}:
            warnings.append(f"{name}=not_available")
        elif before_value != after_value:
            failures.append(f"{name}:before={before_value}|after={after_value}")

    if _model(before["manifest"]) != expected_model or _model(after["manifest"]) != expected_model:
        failures.append(f"expected_model={expected_model}")

    if _runtime(before["manifest"], "device_profile") != "local_u300_24gb":
        failures.append("before_device_profile_not_local_u300_24gb")
    if _runtime(after["manifest"], "device_profile") != "local_u300_24gb":
        failures.append("after_device_profile_not_local_u300_24gb")

    before_policy = _cache_policy(before["manifest"])
    after_policy = _cache_policy(after["manifest"])
    if before_policy != after_policy:
        failures.append(f"cache_policy:before={before_policy}|after={after_policy}")

    before_status = _cache_status(before["manifest"])
    after_status = _cache_status(after["manifest"])
    if before_status != after_status:
        warnings.append(f"cache_status:before={before_status}|after={after_status}")

    before_dirty = before["manifest"].get("git", {}).get("dirty")
    after_dirty = after["manifest"].get("git", {}).get("dirty")
    if before_dirty is not False:
        warnings.append(f"before_git_dirty={before_dirty}")
    if after_dirty is not False:
        warnings.append(f"after_git_dirty={after_dirty}")

    if failures:
        notes.extend(failures + warnings)
        return "fail"
    if warnings:
        notes.extend(warnings)
        return "pass_with_notes"
    return "pass"


def _dataset(manifest: dict[str, Any]) -> Any:
    dataset = manifest.get("dataset", {})
    return dataset.get("short_name") or dataset.get("name")


def _model(manifest: dict[str, Any]) -> Any:
    return manifest.get("model", {}).get("name")


def _split_family(manifest: dict[str, Any]) -> Any:
    return manifest.get("dataset", {}).get("split_family")


def _seed(manifest: dict[str, Any], name: str) -> Any:
    command = str(manifest.get("command", ""))
    flag = "--split-seed" if name == "split_seed" else "--model-seed"
    match = re.search(rf"{re.escape(flag)}\s+(\d+)", command)
    if match:
        return int(match.group(1))
    if name == "split_seed":
        split_id = str(manifest.get("dataset", {}).get("split_id", ""))
        marker = "_s"
        if marker in split_id:
            try:
                return int(split_id.rsplit(marker, 1)[1])
            except ValueError:
                return None
    if name == "model_seed":
        for seed in manifest.get("seeds", []):
            if isinstance(seed, dict) and seed.get("name") == "model_seed":
                return seed.get("value")
    return None


def _runtime(manifest: dict[str, Any], name: str) -> Any:
    return manifest.get("runtime", {}).get(name)


def _kernel_value(kernel: dict[str, Any], name: str) -> Any:
    return kernel.get(name)


def _model_config(manifest: dict[str, Any]) -> Any:
    return manifest.get("model", {}).get("config_ref")


def _cache_policy(manifest: dict[str, Any]) -> str:
    command = str(manifest.get("command", ""))
    flags = [
        "--split-cache enable" in command,
        "--training-index-cache" in command,
        "--disable-training-index-cache" in command,
        "--cluster-artifact-cache" in command,
        "--disable-cluster-artifact-cache" in command,
    ]
    return "|".join(str(flag).lower() for flag in flags)


def _cache_status(manifest: dict[str, Any]) -> str:
    caches = manifest.get("caches", {})
    parts = []
    for name in sorted(caches):
        status = caches.get(name, {}).get("status")
        if status is not None:
            parts.append(f"{name}={status}")
    return "|".join(parts)


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get("metrics", {}).get(key)
    return _float(value)


def _fit_model_seconds(performance: dict[str, Any]) -> float | None:
    for hotspot in performance.get("hotspots", []):
        if hotspot.get("name") == "fit_model":
            return _float(hotspot.get("wall_clock_seconds"))
    for stage in performance.get("stages", []):
        if stage.get("name") == "fit_model":
            return _float(stage.get("wall_clock_seconds"))
    return None


def _train_time_total(kernel: dict[str, Any]) -> float | None:
    durations = kernel.get("epoch_durations_seconds")
    if not isinstance(durations, list):
        return None
    values = [_float(value) for value in durations]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _ratings_per_second(kernel: dict[str, Any]) -> float | None:
    values = kernel.get("ratings_per_second_by_epoch")
    if not isinstance(values, list) or not values:
        return None
    floats = [_float(value) for value in values]
    if any(value is None for value in floats):
        return None
    return sum(value for value in floats if value is not None) / len(floats)


def _peak_memory_mb(performance: dict[str, Any]) -> float | None:
    candidates = []
    for stage in performance.get("stages", []):
        for key in ("rss_start_mb", "rss_end_mb"):
            value = _float(stage.get(key))
            if value is not None:
                candidates.append(value)
    if not candidates:
        return None
    return max(candidates)


def _estimated_touches(kernel: dict[str, Any]) -> float | None:
    return _float(kernel.get("estimated_kernel_work", {}).get("estimated_factor_touches"))


def _fit_seconds_per_million_touches(kernel: dict[str, Any]) -> float | None:
    return _float(kernel.get("cost_ratios", {}).get("fit_seconds_per_million_estimated_factor_touches"))


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def _ratio(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before <= 0:
        return None
    return after / before


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
