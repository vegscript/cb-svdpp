from __future__ import annotations

import json
from pathlib import Path

import typer

from recsys_lab.data.prepare import prepare_dataset_from_config
from recsys_lab.experiments.asymmetric_svd import run_asymmetric_svd_experiment
from recsys_lab.experiments.asvdpp import run_asvdpp_experiment
from recsys_lab.experiments.biased_mf import run_biased_mf_experiment
from recsys_lab.experiments.common import SplitConfig
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
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("train-svdpp")
def train_svdpp(
    processed_manifest: str,
    model_config: str = "configs/models/svdpp.yaml",
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
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
