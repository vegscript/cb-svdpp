from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

PROFILE_VERSION = "cluster_artifact_profile_v1"
DEFAULT_OUTPUT_DIR = Path("artifacts") / "reports"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from recsys_lab.utils.paths import discover_repo_root

    repo_root = discover_repo_root()
    payload = run_profile(
        processed_manifest_path=_resolve_path(args.processed_manifest, repo_root=repo_root),
        model_config_path=_resolve_path(args.model_config, repo_root=repo_root),
        runtime_config_path=_resolve_path(args.runtime_config, repo_root=repo_root),
        model_name=args.model,
        split_family=args.split_family,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        repeats=args.repeats,
        repo_root=repo_root,
    )
    paths = write_reports(
        payload,
        output_dir=_resolve_path(args.output_dir, repo_root=repo_root),
        output_stem=args.output_stem,
    )
    print(paths["json"])
    print(paths["csv"])
    print(paths["summary_csv"])
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile cluster artifact cache stages for CB model configs.")
    parser.add_argument("--processed-manifest", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--model", choices=("cb_svdpp", "cb_asvdpp"), required=True)
    parser.add_argument("--split-family", choices=("benchmark_random_v1",), default="benchmark_random_v1")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--model-seed", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=PROFILE_VERSION)
    return parser.parse_args(argv)


def run_profile(
    *,
    processed_manifest_path: Path,
    model_config_path: Path,
    runtime_config_path: Path,
    model_name: str,
    split_family: str,
    train_ratio: float,
    validation_ratio: float,
    split_seed: int,
    model_seed: int,
    repeats: int,
    repo_root: Path,
) -> dict[str, Any]:
    from recsys_lab.clustering import load_or_build_cluster_artifacts, load_or_build_user_cluster_history_index
    from recsys_lab.config.loader import load_yaml_file
    from recsys_lab.data.histories import build_user_history_index
    from recsys_lab.data.processed import load_processed_dataset_manifest, load_ratings_data_from_manifest
    from recsys_lab.data.splitters import random_split_with_train_coverage
    from recsys_lab.models.registry import validate_model_config_payload
    from recsys_lab.utils.paths import repo_path_string

    if repeats < 1:
        raise ValueError("repeats must be positive")
    if split_family != "benchmark_random_v1":
        raise ValueError("cluster artifact profiling script currently supports benchmark_random_v1 only")

    processed_manifest = load_processed_dataset_manifest(processed_manifest_path)
    dataset_short_name = str(processed_manifest["dataset_short_name"])
    ratings_data = load_ratings_data_from_manifest(processed_manifest_path)
    split = random_split_with_train_coverage(
        ratings_data,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        seed=split_seed,
    )
    train_data = split.train

    model_config_payload = load_yaml_file(model_config_path)
    runtime_config_payload = load_yaml_file(runtime_config_path)
    adapter, model_profile = validate_model_config_payload(model_config_payload, expected_model_name=model_name)
    if not adapter.requirements.needs_cluster_artifacts:
        raise ValueError(f"model '{model_name}' does not require cluster artifacts")

    runtime_dtype = adapter.runtime_dtype(model_profile)
    model_config = adapter.build_model_config(
        model_profile,
        model_seed=model_seed,
        runtime_dtype=runtime_dtype,
    )
    induction_config = adapter.build_induction_config(model_config, model_seed=model_seed)
    clustering = model_profile.clustering
    split_id = f"{split_family}_tr{train_ratio:.3f}_va{validation_ratio:.3f}_s{split_seed:03d}"

    entries: list[dict[str, Any]] = []
    for repeat_index in range(1, repeats + 1):
        cluster_result = load_or_build_cluster_artifacts(
            data=train_data,
            induction_config=induction_config,
            n_user_clusters=int(clustering.n_user_clusters),
            n_item_clusters=int(clustering.n_item_clusters),
            algorithm=str(clustering.algorithm),
            kmeans_n_init=int(clustering.kmeans_n_init),
            dataset_short_name=dataset_short_name,
            split_family=split_family,
            split_id=split_id,
            processed_manifest_path=processed_manifest_path,
            repo_root=repo_root,
            runtime_config_payload=runtime_config_payload,
            use_cache=True,
            mmap_mode=None,
            model=model_name,
        )
        history_index = build_user_history_index(train_data, dtype=runtime_dtype)
        history_result = load_or_build_user_cluster_history_index(
            history_index=history_index,
            item_clusters=cluster_result.artifacts.item_clusters,
            n_clusters=int(cluster_result.artifacts.r_star_counts.shape[1]),
            dataset_short_name=dataset_short_name,
            split_family=split_family,
            split_id=split_id,
            processed_manifest_path=processed_manifest_path,
            repo_root=repo_root,
            runtime_config_payload=runtime_config_payload,
            train_fingerprint=cluster_result.metadata.train_fingerprint,
            cluster_cache_key=cluster_result.metadata.cache_key,
            cluster_cache_fingerprint_sha256=cluster_result.metadata.cache_fingerprint_sha256,
            use_cache=True,
            mmap_mode=None,
            model=model_name,
        )
        entries.append(
            {
                "repeat_index": repeat_index,
                "cluster_artifacts": _profile_payload(cluster_result.profile),
                "user_cluster_history": _profile_payload(history_result.profile),
            }
        )

    return {
        "profile_version": PROFILE_VERSION,
        "claim_boundary": "Diagnostic cluster artifact profiling only; no performance claim.",
        "inputs": {
            "processed_manifest": repo_path_string(processed_manifest_path, repo_root=repo_root),
            "model_config": repo_path_string(model_config_path, repo_root=repo_root),
            "runtime_config": repo_path_string(runtime_config_path, repo_root=repo_root),
            "model": model_name,
            "split_family": split_family,
            "train_ratio": float(train_ratio),
            "validation_ratio": float(validation_ratio),
            "split_seed": int(split_seed),
            "model_seed": int(model_seed),
            "repeats": int(repeats),
        },
        "profiles": entries,
    }


def _profile_payload(profile: Any) -> dict[str, Any]:
    if profile is None:
        return {}
    return profile.to_payload()


def write_reports(payload: dict[str, Any], *, output_dir: Path, output_stem: str = PROFILE_VERSION) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_output_stem = _validate_output_stem(output_stem)
    json_path = output_dir / f"{safe_output_stem}.json"
    csv_path = output_dir / f"{safe_output_stem}.csv"
    summary_csv_path = output_dir / f"{safe_output_stem}_summary.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    rows = profile_rows(payload)
    fieldnames = _csv_fieldnames(rows)
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = summary_rows(payload)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=_csv_fieldnames(summary))
        writer.writeheader()
        writer.writerows(summary)
    return {"json": json_path, "csv": csv_path, "summary_csv": summary_csv_path}


def profile_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in payload.get("profiles", []):
        repeat_index = int(entry["repeat_index"])
        for profile_kind in ("cluster_artifacts", "user_cluster_history"):
            profile_payload = entry.get(profile_kind, {})
            row = {
                "profile_version": payload.get("profile_version", PROFILE_VERSION),
                "repeat_index": repeat_index,
                "profile_kind": profile_kind,
            }
            row.update(profile_payload)
            rows.append(row)
    return rows


def summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in payload.get("profiles", []):
        repeat_index = int(entry["repeat_index"])
        cluster = entry.get("cluster_artifacts", {})
        history = entry.get("user_cluster_history", {})
        rows.append(
            {
                "profile_version": payload.get("profile_version", PROFILE_VERSION),
                "repeat_index": repeat_index,
                "cache_path": _cache_path_label(cluster, history),
                "dataset_short_name": cluster.get("dataset_short_name") or history.get("dataset_short_name", ""),
                "split_family": cluster.get("split_family") or history.get("split_family", ""),
                "split_id": cluster.get("split_id") or history.get("split_id", ""),
                "model": cluster.get("model") or history.get("model", payload.get("inputs", {}).get("model", "")),
                "n_users": cluster.get("n_users") or history.get("n_users", ""),
                "n_items": cluster.get("n_items", ""),
                "train_rows": cluster.get("train_rows") or history.get("train_rows", ""),
                "n_user_clusters": cluster.get("n_user_clusters", ""),
                "n_item_clusters": cluster.get("n_item_clusters") or history.get("n_item_clusters", ""),
                "algorithm": cluster.get("algorithm", ""),
                "kmeans_n_init": cluster.get("kmeans_n_init", ""),
                "induction_seed": cluster.get("induction_seed", ""),
                "induction_latent_dim": cluster.get("induction_latent_dim", ""),
                "induction_epochs": cluster.get("induction_epochs", ""),
                "induction_dtype": cluster.get("induction_dtype", ""),
                "cluster_cache_status": cluster.get("cluster_cache_status", ""),
                "user_cluster_history_cache_status": history.get("user_cluster_history_cache_status", ""),
                "cluster_total_seconds": cluster.get("cluster_total_seconds", ""),
                "user_cluster_history_total_seconds": history.get("user_cluster_history_total_seconds", ""),
            }
        )
    return rows


def stage_ranking_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in payload.get("profiles", []):
        cluster = entry.get("cluster_artifacts", {})
        history = entry.get("user_cluster_history", {})
        rows.extend(_rank_stage_group(cluster, _cluster_stage_specs(cluster), _cluster_cache_path(cluster)))
        rows.extend(_rank_stage_group(history, _history_stage_specs(history), _history_cache_path(history), cluster))
    return rows


def write_stage_ranking_csv(payload: dict[str, Any], output_path: Path) -> Path:
    rows = stage_ranking_rows(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "dataset",
                "model",
                "cache_path",
                "stage",
                "seconds",
                "share_of_cluster_total",
                "rank",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    prefix = ["profile_version", "repeat_index", "profile_kind"]
    keys = sorted({key for row in rows for key in row if key not in prefix})
    return [*prefix, *keys]


def _cache_path_label(cluster: dict[str, Any], history: dict[str, Any]) -> str:
    return f"{_cluster_cache_path(cluster)}+{_history_cache_path(history)}"


def _cluster_cache_path(cluster: dict[str, Any]) -> str:
    status = cluster.get("cluster_cache_status")
    if status == "miss":
        return "cold_miss_build"
    if status == "hit":
        return "warm_hit_load"
    return "cluster_unknown"


def _history_cache_path(history: dict[str, Any]) -> str:
    status = history.get("user_cluster_history_cache_status")
    if status == "miss":
        return "cold_user_cluster_history_miss_build"
    if status == "hit":
        return "warm_user_cluster_history_hit_load"
    return "user_cluster_history_unknown"


def _cluster_stage_specs(profile: dict[str, Any]) -> list[tuple[str, float, str]]:
    is_hit = profile.get("cluster_cache_status") == "hit"
    return [
        (
            "cluster_cache_read",
            _float_value(profile, "cluster_cache_read_seconds"),
            _cluster_stage_note(is_hit, "read"),
        ),
        (
            "cluster_cache_write",
            _float_value(profile, "cluster_cache_write_seconds"),
            _cluster_stage_note(is_hit, "write"),
        ),
        ("induction_fit", _float_value(profile, "induction_fit_seconds"), _cluster_stage_note(is_hit, "induction_fit")),
        (
            "induction_predict",
            _float_value(profile, "induction_predict_seconds"),
            _cluster_stage_note(is_hit, "induction_predict"),
        ),
        (
            "induction_train_rmse",
            _float_value(profile, "induction_train_rmse_seconds"),
            _cluster_stage_note(is_hit, "induction_train_rmse"),
        ),
        ("user_kmeans", _float_value(profile, "user_kmeans_seconds"), _cluster_stage_note(is_hit, "user_kmeans")),
        ("item_kmeans", _float_value(profile, "item_kmeans_seconds"), _cluster_stage_note(is_hit, "item_kmeans")),
        ("r_star", _float_value(profile, "r_star_seconds"), _cluster_stage_note(is_hit, "r_star")),
        (
            "cluster_artifact_validation",
            _float_value(profile, "cluster_artifact_validation_seconds"),
            _cluster_stage_note(is_hit, "validation"),
        ),
    ]


def _history_stage_specs(profile: dict[str, Any]) -> list[tuple[str, float, str]]:
    is_hit = profile.get("user_cluster_history_cache_status") == "hit"
    return [
        (
            "user_cluster_history_cache_read",
            _float_value(profile, "user_cluster_history_cache_read_seconds"),
            _history_stage_note(is_hit, "read"),
        ),
        (
            "user_cluster_history_cache_write",
            _float_value(profile, "user_cluster_history_cache_write_seconds"),
            _history_stage_note(is_hit, "write"),
        ),
        (
            "user_cluster_history_build",
            _float_value(profile, "user_cluster_history_build_seconds"),
            _history_stage_note(is_hit, "build"),
        ),
        (
            "user_cluster_history_validation",
            _float_value(profile, "user_cluster_history_validation_seconds"),
            _history_stage_note(is_hit, "validation"),
        ),
    ]


def _rank_stage_group(
    profile: dict[str, Any],
    specs: list[tuple[str, float, str]],
    cache_path: str,
    fallback_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    fallback = fallback_profile or {}
    total = _ranking_total(profile, cache_path)
    ranked = sorted(specs, key=lambda item: item[1], reverse=True)
    rows = []
    for rank, (stage, seconds, notes) in enumerate(ranked, start=1):
        rows.append(
            {
                "dataset": profile.get("dataset_short_name") or fallback.get("dataset_short_name", ""),
                "model": profile.get("model") or fallback.get("model", ""),
                "cache_path": cache_path,
                "stage": stage,
                "seconds": f"{seconds:.9f}",
                "share_of_cluster_total": "" if total <= 0 else f"{seconds / total:.9f}",
                "rank": rank,
                "notes": notes,
            }
        )
    return rows


def _ranking_total(profile: dict[str, Any], cache_path: str) -> float:
    if cache_path.startswith("cold_user_cluster_history") or cache_path.startswith("warm_user_cluster_history"):
        return _float_value(profile, "user_cluster_history_total_seconds")
    return _float_value(profile, "cluster_total_seconds")


def _cluster_stage_note(is_hit: bool, stage_kind: str) -> str:
    if is_hit:
        return "Warm Cluster-Artifact path; non-read build stages should remain zero." if stage_kind != "read" else (
            "Warm Cluster-Artifact path; reads cached cluster artifacts."
        )
    notes = {
        "read": "Cold Cluster-Artifact path; cache lookup/read attempt before miss.",
        "write": "Cold Cluster-Artifact path; writes cluster artifact arrays and manifest.",
        "induction_fit": "Cold Cluster-Artifact path; Biased-MF induction fit.",
        "induction_predict": "Cold Cluster-Artifact path; induction predictions over train rows.",
        "induction_train_rmse": "Cold Cluster-Artifact path; train RMSE diagnostic.",
        "user_kmeans": "Cold Cluster-Artifact path; KMeans fit_predict on user induction factors.",
        "item_kmeans": "Cold Cluster-Artifact path; KMeans fit_predict on item induction factors.",
        "r_star": "Cold Cluster-Artifact path; r_star means/counts aggregation.",
        "validation": "Cold Cluster-Artifact path; artifact invariant validation.",
    }
    return notes[stage_kind]


def _history_stage_note(is_hit: bool, stage_kind: str) -> str:
    if is_hit:
        if stage_kind != "read":
            return "Warm user-cluster-history path; non-read build stages should remain zero."
        return "Warm user-cluster-history path; reads cached history arrays."
    notes = {
        "read": "Cold user-cluster-history path; cache lookup/read attempt before miss.",
        "write": "Cold user-cluster-history path; writes history arrays and manifest.",
        "build": "Cold user-cluster-history path; build_user_cluster_count_index.",
        "validation": "Cold user-cluster-history path; invariant validation.",
    }
    return notes[stage_kind]


def _float_value(profile: dict[str, Any], key: str) -> float:
    value = profile.get(key, 0.0)
    if value in ("", None):
        return 0.0
    return float(value)


def _validate_output_stem(output_stem: str) -> str:
    if not output_stem:
        raise ValueError("output_stem must be non-empty")
    output_stem_path = Path(output_stem)
    if output_stem_path.name != output_stem or output_stem_path.suffix:
        raise ValueError("output_stem must be a filename stem without directories or suffix")
    return output_stem


def _resolve_path(path: Path, *, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
