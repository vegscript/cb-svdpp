from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml

_MODEL_FAMILIES = {
    "biased_mf": "matrix_factorization",
    "svdpp": "implicit_factorization",
    "asymmetric_svd": "asymmetric_factorization",
    "asvdpp": "poster_extended_factorization",
    "cb_svdpp": "clustering_based_factorization",
    "cb_asvdpp": "clustering_based_factorization",
}

_BASE_METADATA = {
    "status": "draft",
    "owner": "tests",
    "purpose": "synthetic integration fixture",
}

_BASE_TRAINING: dict[str, dict[str, Any]] = {
    "biased_mf": {
        "latent_dim": 8,
        "epochs": 8,
        "learning_rate": 0.01,
        "lambda_b": 0.02,
        "lambda_p": 0.02,
        "lambda_q": 0.02,
        "init_std": 0.05,
        "dtype": "float32",
        "training_backend": "auto",
    },
    "svdpp": {
        "latent_dim": 8,
        "epochs": 8,
        "learning_rate": 0.01,
        "lambda_b": 0.02,
        "lambda_p": 0.02,
        "lambda_q": 0.02,
        "lambda_y": 0.02,
        "init_std": 0.05,
        "dtype": "float32",
        "training_backend": "auto",
        "implicit_policy": "ratings_as_implicit",
    },
    "asymmetric_svd": {
        "latent_dim": 8,
        "epochs": 8,
        "learning_rate": 0.01,
        "lambda_b": 0.02,
        "lambda_q": 0.02,
        "lambda_x": 0.02,
        "lambda_y": 0.02,
        "init_std": 0.05,
        "dtype": "float32",
        "implicit_policy": "ratings_as_implicit",
        "residual_weight_contract": "detached",
    },
    "asvdpp": {
        "latent_dim": 8,
        "epochs": 8,
        "learning_rate": 0.01,
        "lambda_b": 0.02,
        "lambda_p": 0.02,
        "lambda_q": 0.02,
        "lambda_x": 0.02,
        "lambda_y": 0.02,
        "init_std": 0.05,
        "dtype": "float32",
        "implicit_policy": "ratings_as_implicit",
        "residual_weight_contract": "detached",
    },
    "cb_svdpp": {
        "latent_dim": 8,
        "epochs": 8,
        "learning_rate": 0.02,
        "lambda_b": 0.01,
        "lambda_p": 0.01,
        "lambda_q": 0.01,
        "lambda_y": 0.01,
        "lambda_pC": 0.01,
        "lambda_qC": 0.01,
        "lambda_yC": 0.01,
        "init_std": 0.05,
        "dtype": "float32",
        "implicit_policy": "ratings_as_implicit",
    },
    "cb_asvdpp": {
        "latent_dim": 8,
        "epochs": 8,
        "learning_rate": 0.02,
        "lambda_b": 0.01,
        "lambda_p": 0.01,
        "lambda_q": 0.01,
        "lambda_x": 0.01,
        "lambda_y": 0.01,
        "lambda_pC": 0.01,
        "lambda_qC": 0.01,
        "lambda_xC": 0.01,
        "lambda_yC": 0.01,
        "init_std": 0.05,
        "dtype": "float32",
        "implicit_policy": "ratings_as_implicit",
        "residual_weight_contract": "detached",
    },
}

_BASE_CLUSTERING = {
    "n_user_clusters": 2,
    "n_item_clusters": 2,
    "alpha": 0.2,
    "algorithm": "kmeans",
    "kmeans_n_init": 5,
}


def model_config_payload(
    model_name: str,
    *,
    training: dict[str, Any] | None = None,
    clustering: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    scope: str = "paper_inspired",
    notes: list[str] | None = None,
) -> dict[str, Any]:
    if model_name not in _BASE_TRAINING:
        raise ValueError(f"unsupported synthetic model config: {model_name}")

    payload: dict[str, Any] = {
        "metadata": {**_BASE_METADATA, **(metadata or {})},
        "model": {
            "name": model_name,
            "family": _MODEL_FAMILIES[model_name],
            "scope": scope,
        },
        "training": {**deepcopy(_BASE_TRAINING[model_name]), **(training or {})},
        "notes": notes or ["Synthetic strict model config fixture."],
    }
    if model_name.startswith("cb_"):
        payload["clustering"] = {**deepcopy(_BASE_CLUSTERING), **(clustering or {})}
    return payload


def model_config_yaml(
    model_name: str,
    *,
    training: dict[str, Any] | None = None,
    clustering: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    scope: str = "paper_inspired",
    notes: list[str] | None = None,
) -> str:
    return yaml.safe_dump(
        model_config_payload(
            model_name,
            training=training,
            clustering=clustering,
            metadata=metadata,
            scope=scope,
            notes=notes,
        ),
        sort_keys=False,
    )
