from __future__ import annotations

import json
from pathlib import Path

import typer

from recsys_lab.config.loader import load_yaml_file
from recsys_lab.data.prepare import prepare_dataset_from_config
from recsys_lab.experiments.asvdpp import run_asvdpp_experiment
from recsys_lab.experiments.asymmetric_svd import run_asymmetric_svd_experiment
from recsys_lab.experiments.biased_mf import run_biased_mf_experiment
from recsys_lab.experiments.cb_asvdpp import run_cb_asvdpp_experiment
from recsys_lab.experiments.cb_svdpp import run_cb_svdpp_experiment
from recsys_lab.experiments.common import SplitConfig
from recsys_lab.experiments.ml100k_inner_tuning import run_inner_tuning, run_ml100k_inner_tuning
from recsys_lab.experiments.ml100k_paper_benchmark import run_ml100k_paper_benchmark
from recsys_lab.experiments.ml100k_paper_multiseed_benchmark import (
    run_ml100k_paper_multiseed_benchmark,
)
from recsys_lab.experiments.random_multiseed_benchmark import run_random_multiseed_benchmark
from recsys_lab.experiments.runner import build_dry_run_plan
from recsys_lab.experiments.runtime import assess_device_profile_contract, validate_claim_eligible_device_profile
from recsys_lab.experiments.svdpp import run_svdpp_experiment
from recsys_lab.experiments.unified_runner import run_unified_experiment
from recsys_lab.utils.manifests import validate_manifest_file
from recsys_lab.utils.paths import discover_repo_root, repo_health_snapshot

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Canonical CLI for repo-safe bootstrap, dry runs, and manifest validation.",
)


def _resolve_path(path_value: str | None, *, repo_root: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _resolve_required_path(path_value: str | None, *, repo_root: Path, parameter_name: str) -> Path:
    resolved = _resolve_path(path_value, repo_root=repo_root)
    if resolved is None:
        raise typer.BadParameter(f"{parameter_name} is required")
    return resolved


def _resolve_path_list(path_values: str | None, *, repo_root: Path, parameter_name: str) -> list[Path] | None:
    if path_values is None:
        return None
    resolved_paths: list[Path] = []
    for raw_value in path_values.split(","):
        value = raw_value.strip()
        if not value:
            continue
        resolved_paths.append(_resolve_required_path(value, repo_root=repo_root, parameter_name=parameter_name))
    if not resolved_paths:
        raise typer.BadParameter(f"{parameter_name} must contain at least one comma-separated path")
    return resolved_paths


def _resolve_split_cache_override(split_cache: str) -> bool | None:
    normalized = split_cache.strip().lower()
    if normalized == "auto":
        return None
    if normalized == "enable":
        return True
    if normalized == "disable":
        return False
    raise typer.BadParameter("split_cache must be one of: auto, enable, disable")


@app.command("bootstrap-check")
def bootstrap_check(repo_root: str | None = None) -> None:
    root = (
        _resolve_required_path(repo_root, repo_root=discover_repo_root(), parameter_name="repo_root")
        if repo_root
        else discover_repo_root()
    )
    snapshot = repo_health_snapshot(root)
    typer.echo(
        json.dumps(
            {
                "repo_root": str(root),
                "required_paths": snapshot,
                "healthy": all(snapshot.values()),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("validate-manifest")
def validate_manifest(path: str, repo_root: str | None = None) -> None:
    root = (
        _resolve_required_path(repo_root, repo_root=discover_repo_root(), parameter_name="repo_root")
        if repo_root
        else discover_repo_root()
    )
    manifest_path = _resolve_required_path(path, repo_root=root, parameter_name="path")
    validate_manifest_file(manifest_path, repo_root=root)
    typer.echo(
        json.dumps(
            {
                "status": "valid",
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("validate-runtime-profile")
def validate_runtime_profile(
    device_config: str,
    claim_eligible: bool = typer.Option(
        False,
        "--claim-eligible/--allow-non-claim-eligible",
        help="Fail if the device profile cannot support claim-eligible runs.",
    ),
) -> None:
    root = discover_repo_root()
    device_config_path = _resolve_required_path(device_config, repo_root=root, parameter_name="device_config")
    device_config_payload = load_yaml_file(device_config_path)
    assessment = assess_device_profile_contract(device_config_payload=device_config_payload)
    if claim_eligible:
        try:
            assessment = validate_claim_eligible_device_profile(device_config_payload=device_config_payload)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    typer.echo(
        json.dumps(
            {
                "status": "valid",
                "device_config": str(device_config_path),
                "claim_eligible_requested": claim_eligible,
                "device_profile_contract": assessment,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("run-experiment")
def run_experiment(
    experiment_config: str,
    runtime_config: str = "configs/runtime/base.yaml",
    dataset_config: str | None = None,
    model_config: str | None = None,
    device_config: str | None = None,
    dry_run: bool = True,
) -> None:
    if not dry_run:
        raise typer.BadParameter("Only --dry-run is currently supported until the training pipeline is implemented.")

    root = discover_repo_root()
    experiment_config_path = _resolve_required_path(
        experiment_config, repo_root=root, parameter_name="experiment_config"
    )
    runtime_config_path = _resolve_required_path(runtime_config, repo_root=root, parameter_name="runtime_config")
    payload = build_dry_run_plan(
        experiment_config=experiment_config_path,
        runtime_config=runtime_config_path,
        dataset_config=_resolve_path(dataset_config, repo_root=root),
        model_config=_resolve_path(model_config, repo_root=root),
        device_config=_resolve_path(device_config, repo_root=root),
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("prepare-dataset")
def prepare_dataset(
    dataset_config: str,
    dtype: str = "float32",
    overwrite: bool = False,
) -> None:
    root = discover_repo_root()
    dataset_config_path = _resolve_path(dataset_config, repo_root=root)
    if dataset_config_path is None:
        raise typer.BadParameter("dataset_config is required")
    payload = prepare_dataset_from_config(
        dataset_config_path=dataset_config_path,
        dtype=dtype,
        overwrite=overwrite,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("train")
def train_model(
    model: str = typer.Option(..., "--model", help="Canonical model name from the unified model registry."),
    processed_manifest: str = typer.Option(..., "--processed-manifest", help="Processed dataset manifest path."),
    model_config: str = typer.Option(..., "--model-config", help="Strict model config YAML path."),
    runtime_config: str = "configs/runtime/base.yaml",
    device_config: str = "configs/runtime/devices/local_i5_2500k_24gb.yaml",
    split_family: str = "benchmark_random_v1",
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    split_seed: int = 1,
    model_seed: int = 1,
    split_cache: str = typer.Option("auto", help="Split-cache policy: auto, enable, or disable."),
    training_index_cache: bool = typer.Option(
        False,
        "--training-index-cache/--disable-training-index-cache",
        help="Persist and reuse training indices required by the selected model.",
    ),
    cluster_artifact_cache: bool = typer.Option(
        False,
        "--cluster-artifact-cache/--disable-cluster-artifact-cache",
        help="Persist and reuse train-only CB cluster artifacts where required.",
    ),
    evaluate_test: bool = typer.Option(
        True,
        "--test-eval/--skip-test-eval",
        help="Evaluate the test split after training.",
    ),
) -> None:
    root = discover_repo_root()
    processed_manifest_path = _resolve_path(processed_manifest, repo_root=root)
    model_config_path = _resolve_path(model_config, repo_root=root)
    runtime_config_path = _resolve_path(runtime_config, repo_root=root)
    device_config_path = _resolve_path(device_config, repo_root=root)

    if processed_manifest_path is None or model_config_path is None:
        raise typer.BadParameter("processed_manifest and model_config are required")
    if runtime_config_path is None or device_config_path is None:
        raise typer.BadParameter("runtime_config and device_config are required")

    payload = run_unified_experiment(
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_config=SplitConfig(
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            seed=split_seed,
        ),
        model_seed=model_seed,
        repo_root=root,
        model_name=model,
        split_family=split_family,
        evaluate_test=evaluate_test,
        use_split_cache=_resolve_split_cache_override(split_cache),
        use_training_index_cache=training_index_cache,
        use_cluster_artifact_cache=cluster_artifact_cache,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("train-biased-mf")
# Legacy compatibility wrapper only.
# Do not add experiment lifecycle logic here.
# All execution must delegate to run_unified_experiment.
def train_biased_mf(
    processed_manifest: str,
    model_config: str = "configs/models/biased_mf.yaml",
    runtime_config: str = "configs/runtime/base.yaml",
    device_config: str = "configs/runtime/devices/local_i5_2500k_24gb.yaml",
    split_family: str = "benchmark_random_v1",
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    split_seed: int = 1,
    model_seed: int = 1,
) -> None:
    root = discover_repo_root()
    processed_manifest_path = _resolve_path(processed_manifest, repo_root=root)
    model_config_path = _resolve_path(model_config, repo_root=root)
    runtime_config_path = _resolve_path(runtime_config, repo_root=root)
    device_config_path = _resolve_path(device_config, repo_root=root)

    if processed_manifest_path is None:
        raise typer.BadParameter("processed_manifest is required")
    if model_config_path is None or runtime_config_path is None or device_config_path is None:
        raise typer.BadParameter("model_config, runtime_config, and device_config are required")

    payload = run_biased_mf_experiment(
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_config=SplitConfig(
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            seed=split_seed,
        ),
        model_seed=model_seed,
        repo_root=root,
        split_family=split_family,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("train-svdpp")
# Legacy compatibility wrapper only.
# Do not add experiment lifecycle logic here.
# All execution must delegate to run_unified_experiment.
def train_svdpp(
    processed_manifest: str,
    model_config: str = "configs/models/svdpp.yaml",
    runtime_config: str = "configs/runtime/base.yaml",
    device_config: str = "configs/runtime/devices/local_i5_2500k_24gb.yaml",
    split_family: str = "benchmark_random_v1",
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    split_seed: int = 1,
    model_seed: int = 1,
    split_cache: str = typer.Option("auto", help="Split-cache policy: auto, enable, or disable."),
    training_index_cache: bool = typer.Option(
        False,
        "--training-index-cache/--disable-training-index-cache",
        help="Persist and reuse training history indices.",
    ),
) -> None:
    root = discover_repo_root()
    processed_manifest_path = _resolve_path(processed_manifest, repo_root=root)
    model_config_path = _resolve_path(model_config, repo_root=root)
    runtime_config_path = _resolve_path(runtime_config, repo_root=root)
    device_config_path = _resolve_path(device_config, repo_root=root)

    if processed_manifest_path is None:
        raise typer.BadParameter("processed_manifest is required")
    if model_config_path is None or runtime_config_path is None or device_config_path is None:
        raise typer.BadParameter("model_config, runtime_config, and device_config are required")

    payload = run_svdpp_experiment(
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_config=SplitConfig(
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            seed=split_seed,
        ),
        model_seed=model_seed,
        repo_root=root,
        split_family=split_family,
        use_split_cache=_resolve_split_cache_override(split_cache),
        use_training_index_cache=training_index_cache,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("train-asymmetric-svd")
# Legacy compatibility wrapper only.
# Do not add experiment lifecycle logic here.
# All execution must delegate to run_unified_experiment.
def train_asymmetric_svd(
    processed_manifest: str,
    model_config: str = "configs/models/asymmetric_svd.yaml",
    runtime_config: str = "configs/runtime/base.yaml",
    device_config: str = "configs/runtime/devices/local_i5_2500k_24gb.yaml",
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    split_seed: int = 1,
    model_seed: int = 1,
) -> None:
    root = discover_repo_root()
    processed_manifest_path = _resolve_path(processed_manifest, repo_root=root)
    model_config_path = _resolve_path(model_config, repo_root=root)
    runtime_config_path = _resolve_path(runtime_config, repo_root=root)
    device_config_path = _resolve_path(device_config, repo_root=root)

    if processed_manifest_path is None:
        raise typer.BadParameter("processed_manifest is required")
    if model_config_path is None or runtime_config_path is None or device_config_path is None:
        raise typer.BadParameter("model_config, runtime_config, and device_config are required")

    payload = run_asymmetric_svd_experiment(
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_config=SplitConfig(
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            seed=split_seed,
        ),
        model_seed=model_seed,
        repo_root=root,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("train-asvdpp")
# Legacy compatibility wrapper only.
# Do not add experiment lifecycle logic here.
# All execution must delegate to run_unified_experiment.
def train_asvdpp(
    processed_manifest: str,
    model_config: str = "configs/models/asvdpp.yaml",
    runtime_config: str = "configs/runtime/base.yaml",
    device_config: str = "configs/runtime/devices/local_i5_2500k_24gb.yaml",
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    split_seed: int = 1,
    model_seed: int = 1,
    split_cache: str = typer.Option("auto", help="Split-cache policy: auto, enable, or disable."),
    training_index_cache: bool = typer.Option(
        False,
        "--training-index-cache/--disable-training-index-cache",
        help="Persist and reuse training history indices.",
    ),
) -> None:
    root = discover_repo_root()
    processed_manifest_path = _resolve_path(processed_manifest, repo_root=root)
    model_config_path = _resolve_path(model_config, repo_root=root)
    runtime_config_path = _resolve_path(runtime_config, repo_root=root)
    device_config_path = _resolve_path(device_config, repo_root=root)

    if processed_manifest_path is None:
        raise typer.BadParameter("processed_manifest is required")
    if model_config_path is None or runtime_config_path is None or device_config_path is None:
        raise typer.BadParameter("model_config, runtime_config, and device_config are required")

    payload = run_asvdpp_experiment(
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_config=SplitConfig(
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            seed=split_seed,
        ),
        model_seed=model_seed,
        repo_root=root,
        use_split_cache=_resolve_split_cache_override(split_cache),
        use_training_index_cache=training_index_cache,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("train-cb-svdpp")
# Legacy compatibility wrapper only.
# Do not add experiment lifecycle logic here.
# All execution must delegate to run_unified_experiment.
def train_cb_svdpp(
    processed_manifest: str,
    model_config: str = "configs/models/cb_svdpp.yaml",
    runtime_config: str = "configs/runtime/base.yaml",
    device_config: str = "configs/runtime/devices/local_i5_2500k_24gb.yaml",
    split_family: str = "benchmark_random_v1",
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    split_seed: int = 1,
    model_seed: int = 1,
    split_cache: str = typer.Option("auto", help="Split-cache policy: auto, enable, or disable."),
    training_index_cache: bool = typer.Option(
        False,
        "--training-index-cache/--disable-training-index-cache",
        help="Persist and reuse training history indices.",
    ),
    cluster_artifact_cache: bool = typer.Option(
        False,
        "--cluster-artifact-cache/--disable-cluster-artifact-cache",
        help="Persist and reuse train-only CB cluster artifacts and cluster-history indices.",
    ),
) -> None:
    root = discover_repo_root()
    processed_manifest_path = _resolve_path(processed_manifest, repo_root=root)
    model_config_path = _resolve_path(model_config, repo_root=root)
    runtime_config_path = _resolve_path(runtime_config, repo_root=root)
    device_config_path = _resolve_path(device_config, repo_root=root)

    if processed_manifest_path is None:
        raise typer.BadParameter("processed_manifest is required")
    if model_config_path is None or runtime_config_path is None or device_config_path is None:
        raise typer.BadParameter("model_config, runtime_config, and device_config are required")

    payload = run_cb_svdpp_experiment(
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_config=SplitConfig(
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            seed=split_seed,
        ),
        model_seed=model_seed,
        repo_root=root,
        split_family=split_family,
        use_split_cache=_resolve_split_cache_override(split_cache),
        use_training_index_cache=training_index_cache,
        use_cluster_artifact_cache=cluster_artifact_cache,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("train-cb-asvdpp")
# Legacy compatibility wrapper only.
# Do not add experiment lifecycle logic here.
# All execution must delegate to run_unified_experiment.
def train_cb_asvdpp(
    processed_manifest: str,
    model_config: str = "configs/models/cb_asvdpp.yaml",
    runtime_config: str = "configs/runtime/base.yaml",
    device_config: str = "configs/runtime/devices/local_i5_2500k_24gb.yaml",
    split_family: str = "benchmark_random_v1",
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    split_seed: int = 1,
    model_seed: int = 1,
    split_cache: str = typer.Option("auto", help="Split-cache policy: auto, enable, or disable."),
    training_index_cache: bool = typer.Option(
        False,
        "--training-index-cache/--disable-training-index-cache",
        help="Persist and reuse training history and explicit-feedback indices.",
    ),
    cluster_artifact_cache: bool = typer.Option(
        False,
        "--cluster-artifact-cache/--disable-cluster-artifact-cache",
        help="Persist and reuse train-only CB cluster artifacts and cluster-history indices.",
    ),
) -> None:
    root = discover_repo_root()
    processed_manifest_path = _resolve_path(processed_manifest, repo_root=root)
    model_config_path = _resolve_path(model_config, repo_root=root)
    runtime_config_path = _resolve_path(runtime_config, repo_root=root)
    device_config_path = _resolve_path(device_config, repo_root=root)

    if processed_manifest_path is None:
        raise typer.BadParameter("processed_manifest is required")
    if model_config_path is None or runtime_config_path is None or device_config_path is None:
        raise typer.BadParameter("model_config, runtime_config, and device_config are required")

    payload = run_cb_asvdpp_experiment(
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_config=SplitConfig(
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            seed=split_seed,
        ),
        model_seed=model_seed,
        repo_root=root,
        split_family=split_family,
        use_split_cache=_resolve_split_cache_override(split_cache),
        use_training_index_cache=training_index_cache,
        use_cluster_artifact_cache=cluster_artifact_cache,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("benchmark-ml100k-paper")
def benchmark_ml100k_paper(
    model: str,
    processed_manifest: str,
    model_config: str,
    runtime_config: str = "configs/runtime/base.yaml",
    device_config: str = "configs/runtime/devices/local_i5_2500k_24gb.yaml",
    model_seed: int = 1,
    split_cache: str = typer.Option("auto", help="Split-cache policy: auto, enable, or disable."),
) -> None:
    root = discover_repo_root()
    processed_manifest_path = _resolve_path(processed_manifest, repo_root=root)
    model_config_path = _resolve_path(model_config, repo_root=root)
    runtime_config_path = _resolve_path(runtime_config, repo_root=root)
    device_config_path = _resolve_path(device_config, repo_root=root)

    if processed_manifest_path is None or model_config_path is None:
        raise typer.BadParameter("processed_manifest and model_config are required")
    if runtime_config_path is None or device_config_path is None:
        raise typer.BadParameter("runtime_config and device_config are required")

    payload = run_ml100k_paper_benchmark(
        model_name=model,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        model_seed=model_seed,
        use_split_cache=_resolve_split_cache_override(split_cache),
        repo_root=root,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("benchmark-ml100k-paper-multiseed")
def benchmark_ml100k_paper_multiseed(
    model: str,
    processed_manifest: str,
    model_config: str,
    runtime_config: str = "configs/runtime/base.yaml",
    device_config: str = "configs/runtime/devices/local_i5_2500k_24gb.yaml",
    model_seeds: str = "1,2,3",
    benchmark_manifest_paths: str | None = None,
) -> None:
    root = discover_repo_root()
    processed_manifest_path = _resolve_path(processed_manifest, repo_root=root)
    model_config_path = _resolve_path(model_config, repo_root=root)
    runtime_config_path = _resolve_path(runtime_config, repo_root=root)
    device_config_path = _resolve_path(device_config, repo_root=root)

    if processed_manifest_path is None or model_config_path is None:
        raise typer.BadParameter("processed_manifest and model_config are required")
    if runtime_config_path is None or device_config_path is None:
        raise typer.BadParameter("runtime_config and device_config are required")

    try:
        seed_values = [int(value.strip()) for value in model_seeds.split(",") if value.strip()]
    except ValueError as exc:
        raise typer.BadParameter("model_seeds must be a comma-separated list of integers") from exc
    if not seed_values:
        raise typer.BadParameter("model_seeds must contain at least one integer")

    benchmark_manifest_path_values = _resolve_path_list(
        benchmark_manifest_paths,
        repo_root=root,
        parameter_name="benchmark_manifest_paths",
    )

    payload = run_ml100k_paper_multiseed_benchmark(
        model_name=model,
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        model_seeds=seed_values,
        benchmark_manifest_paths=benchmark_manifest_path_values,
        repo_root=root,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("benchmark-random-multiseed")
def benchmark_random_multiseed(
    processed_manifest: str,
    model_config: str,
    runtime_config: str = "configs/runtime/base.yaml",
    device_config: str = "configs/runtime/devices/local_i5_2500k_24gb.yaml",
    split_seeds: str = "1,2,3",
    model_seed: int = 1,
    run_manifest_paths: str | None = None,
) -> None:
    root = discover_repo_root()
    processed_manifest_path = _resolve_path(processed_manifest, repo_root=root)
    model_config_path = _resolve_path(model_config, repo_root=root)
    runtime_config_path = _resolve_path(runtime_config, repo_root=root)
    device_config_path = _resolve_path(device_config, repo_root=root)

    if processed_manifest_path is None or model_config_path is None:
        raise typer.BadParameter("processed_manifest and model_config are required")
    if runtime_config_path is None or device_config_path is None:
        raise typer.BadParameter("runtime_config and device_config are required")

    try:
        split_seed_values = [int(value.strip()) for value in split_seeds.split(",") if value.strip()]
    except ValueError as exc:
        raise typer.BadParameter("split_seeds must be a comma-separated list of integers") from exc
    if not split_seed_values:
        raise typer.BadParameter("split_seeds must contain at least one integer")

    run_manifest_path_values = _resolve_path_list(
        run_manifest_paths,
        repo_root=root,
        parameter_name="run_manifest_paths",
    )

    payload = run_random_multiseed_benchmark(
        processed_manifest_path=processed_manifest_path,
        model_config_path=model_config_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        split_seeds=split_seed_values,
        model_seed=model_seed,
        run_manifest_paths=run_manifest_path_values,
        repo_root=root,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("tune-ml100k-inner")
def tune_ml100k_inner(
    tuning_config: str,
    processed_manifest: str,
    runtime_config: str = "configs/runtime/base.yaml",
    device_config: str = "configs/runtime/devices/local_i5_2500k_24gb.yaml",
    split_cache: str = typer.Option("auto", help="Split-cache policy: auto, enable, or disable."),
    training_index_cache: bool = typer.Option(
        False,
        "--training-index-cache/--disable-training-index-cache",
        help="Persist and reuse supported training history indices during tuning.",
    ),
    cluster_artifact_cache: bool = typer.Option(
        False,
        "--cluster-artifact-cache/--disable-cluster-artifact-cache",
        help="Persist and reuse supported train-only CB cluster artifacts during tuning.",
    ),
) -> None:
    root = discover_repo_root()
    tuning_config_path = _resolve_path(tuning_config, repo_root=root)
    processed_manifest_path = _resolve_path(processed_manifest, repo_root=root)
    runtime_config_path = _resolve_path(runtime_config, repo_root=root)
    device_config_path = _resolve_path(device_config, repo_root=root)

    if tuning_config_path is None or processed_manifest_path is None:
        raise typer.BadParameter("tuning_config and processed_manifest are required")
    if runtime_config_path is None or device_config_path is None:
        raise typer.BadParameter("runtime_config and device_config are required")

    payload = run_ml100k_inner_tuning(
        tuning_config_path=tuning_config_path,
        processed_manifest_path=processed_manifest_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        use_split_cache=_resolve_split_cache_override(split_cache),
        use_training_index_cache=training_index_cache,
        use_cluster_artifact_cache=cluster_artifact_cache,
        repo_root=root,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("tune-inner")
def tune_inner(
    tuning_config: str,
    processed_manifest: str,
    runtime_config: str = "configs/runtime/base.yaml",
    device_config: str = "configs/runtime/devices/local_i5_2500k_24gb.yaml",
    split_cache: str = typer.Option("auto", help="Split-cache policy: auto, enable, or disable."),
    training_index_cache: bool = typer.Option(
        False,
        "--training-index-cache/--disable-training-index-cache",
        help="Persist and reuse supported training history indices during tuning.",
    ),
    cluster_artifact_cache: bool = typer.Option(
        False,
        "--cluster-artifact-cache/--disable-cluster-artifact-cache",
        help="Persist and reuse supported train-only CB cluster artifacts during tuning.",
    ),
) -> None:
    root = discover_repo_root()
    tuning_config_path = _resolve_path(tuning_config, repo_root=root)
    processed_manifest_path = _resolve_path(processed_manifest, repo_root=root)
    runtime_config_path = _resolve_path(runtime_config, repo_root=root)
    device_config_path = _resolve_path(device_config, repo_root=root)

    if tuning_config_path is None or processed_manifest_path is None:
        raise typer.BadParameter("tuning_config and processed_manifest are required")
    if runtime_config_path is None or device_config_path is None:
        raise typer.BadParameter("runtime_config and device_config are required")

    payload = run_inner_tuning(
        tuning_config_path=tuning_config_path,
        processed_manifest_path=processed_manifest_path,
        runtime_config_path=runtime_config_path,
        device_config_path=device_config_path,
        use_split_cache=_resolve_split_cache_override(split_cache),
        use_training_index_cache=training_index_cache,
        use_cluster_artifact_cache=cluster_artifact_cache,
        repo_root=root,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
