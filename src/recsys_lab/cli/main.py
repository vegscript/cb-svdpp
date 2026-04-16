from __future__ import annotations

import json
from pathlib import Path

import typer

from recsys_lab.data.prepare import prepare_dataset_from_config
from recsys_lab.experiments.asymmetric_svd import run_asymmetric_svd_experiment
from recsys_lab.experiments.asvdpp import run_asvdpp_experiment
from recsys_lab.experiments.biased_mf import run_biased_mf_experiment
from recsys_lab.experiments.cb_svdpp import run_cb_svdpp_experiment
from recsys_lab.experiments.common import SplitConfig
from recsys_lab.experiments.ml100k_inner_tuning import run_ml100k_inner_tuning
from recsys_lab.experiments.ml100k_paper_benchmark import run_ml100k_paper_benchmark
from recsys_lab.experiments.ml100k_paper_multiseed_benchmark import run_ml100k_paper_multiseed_benchmark
from recsys_lab.experiments.svdpp import run_svdpp_experiment
from recsys_lab.experiments.runner import build_dry_run_plan
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
    root = _resolve_path(repo_root, repo_root=discover_repo_root()) if repo_root else discover_repo_root()
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
    root = _resolve_path(repo_root, repo_root=discover_repo_root()) if repo_root else discover_repo_root()
    manifest_path = _resolve_path(path, repo_root=root)
    if manifest_path is None:
        raise typer.BadParameter("path is required")
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
        raise typer.BadParameter(
            "Only --dry-run is currently supported until the training pipeline is implemented."
        )

    root = discover_repo_root()
    payload = build_dry_run_plan(
        experiment_config=_resolve_path(experiment_config, repo_root=root),
        runtime_config=_resolve_path(runtime_config, repo_root=root),
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


@app.command("train-biased-mf")
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
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("train-asymmetric-svd")
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
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("train-cb-svdpp")
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

    benchmark_manifest_path_values = None
    if benchmark_manifest_paths is not None:
        benchmark_manifest_path_values = [
            _resolve_path(value.strip(), repo_root=root)
            for value in benchmark_manifest_paths.split(",")
            if value.strip()
        ]
        if not benchmark_manifest_path_values:
            raise typer.BadParameter(
                "benchmark_manifest_paths must contain at least one comma-separated path"
            )
        if any(path is None for path in benchmark_manifest_path_values):
            raise typer.BadParameter("benchmark_manifest_paths contains an invalid path")

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


@app.command("tune-ml100k-inner")
def tune_ml100k_inner(
    tuning_config: str,
    processed_manifest: str,
    runtime_config: str = "configs/runtime/base.yaml",
    device_config: str = "configs/runtime/devices/local_i5_2500k_24gb.yaml",
    split_cache: str = typer.Option("auto", help="Split-cache policy: auto, enable, or disable."),
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
        repo_root=root,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
