from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DUPLICATION_PROFILE_VERSION = "residual_history_duplication_audit_v1"

SUPPORTED_MODELS = frozenset(
    {
        "biased_mf",
        "svdpp",
        "asymmetric_svd",
        "asvdpp",
        "cb_svdpp",
        "cb_asvdpp",
    }
)

_EXPLICIT_RESIDUAL_DUPLICATION_MODELS = frozenset(
    {"asymmetric_svd", "asvdpp", "cb_asvdpp"}
)
_EXPLICIT_DOUBLE_TRAVERSAL_MODELS = _EXPLICIT_RESIDUAL_DUPLICATION_MODELS
_IMPLICIT_DOUBLE_TRAVERSAL_MODELS = frozenset(
    {"svdpp", "asymmetric_svd", "asvdpp", "cb_svdpp", "cb_asvdpp"}
)
_CLUSTER_DOUBLE_TRAVERSAL_MODELS = frozenset({"cb_svdpp", "cb_asvdpp"})
_EXPLICIT_ITEM_CLUSTER_LOOKUP_MODELS = frozenset({"cb_asvdpp"})
_CLUSTER_HISTORY_CLUSTER_LOOKUP_MODELS = frozenset({"cb_svdpp", "cb_asvdpp"})

_POTENTIAL_EXACT_REUSE_SCOPE: dict[str, list[str]] = {
    "biased_mf": [],
    "svdpp": [],
    "asymmetric_svd": ["raw_explicit_residual_weights"],
    "asvdpp": ["raw_explicit_residual_weights"],
    "cb_svdpp": ["cluster_history_ids_counts"],
    "cb_asvdpp": [
        "raw_explicit_residual_weights",
        "explicit_history_cluster_ids",
        "cluster_history_ids_counts",
    ],
}


def build_duplication_cost_estimate(
    *,
    model: str,
    kernel_work: Mapping[str, Any],
) -> dict[str, Any]:
    """Build aggregate duplication counts from the kernel-profile work contract."""
    model_name = str(model)
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"unknown model for duplication profile: {model_name}")

    explicit_visits = _non_negative_int(kernel_work, "explicit_history_visits")
    implicit_visits = _non_negative_int(kernel_work, "implicit_history_visits")
    cluster_visits = _non_negative_int(kernel_work, "cluster_history_visits")

    duplicated_explicit_residual = (
        explicit_visits
        if model_name in _EXPLICIT_RESIDUAL_DUPLICATION_MODELS
        else 0
    )

    explicit_traversal = (
        explicit_visits if model_name in _EXPLICIT_DOUBLE_TRAVERSAL_MODELS else 0
    )
    implicit_traversal = (
        implicit_visits if model_name in _IMPLICIT_DOUBLE_TRAVERSAL_MODELS else 0
    )
    cluster_traversal = (
        cluster_visits if model_name in _CLUSTER_DOUBLE_TRAVERSAL_MODELS else 0
    )

    explicit_item_cluster_lookup = (
        explicit_visits
        if model_name in _EXPLICIT_ITEM_CLUSTER_LOOKUP_MODELS
        else 0
    )
    cluster_history_cluster_lookup = (
        cluster_visits
        if model_name in _CLUSTER_HISTORY_CLUSTER_LOOKUP_MODELS
        else 0
    )

    return {
        "profile_version": DUPLICATION_PROFILE_VERSION,
        "model": model_name,
        "explicit_history_visits": explicit_visits,
        "implicit_history_visits": implicit_visits,
        "cluster_history_visits": cluster_visits,
        "duplicated_explicit_residual_computations": duplicated_explicit_residual,
        "duplicated_history_cluster_lookups": (
            explicit_item_cluster_lookup + cluster_history_cluster_lookup
        ),
        "duplicated_history_cluster_lookup_breakdown": {
            "explicit_item_cluster_lookup": explicit_item_cluster_lookup,
            "cluster_history_cluster_id_lookup": cluster_history_cluster_lookup,
        },
        "duplicated_history_traversals": (
            explicit_traversal + implicit_traversal + cluster_traversal
        ),
        "duplicated_history_traversal_breakdown": {
            "explicit_context_and_update": explicit_traversal,
            "implicit_context_and_update": implicit_traversal,
            "cluster_context_and_update": cluster_traversal,
        },
        "potential_exact_reuse_scope": list(_POTENTIAL_EXACT_REUSE_SCOPE[model_name]),
    }


def build_duplication_cost_estimate_from_kernel_profile(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Build duplication counts from a kernel_profile.json-like payload."""
    model = payload.get("model")
    if model is None:
        raise ValueError("kernel profile payload must include model")

    kernel_work = payload.get("estimated_kernel_work")
    if not isinstance(kernel_work, Mapping):
        raise ValueError("kernel profile payload must include estimated_kernel_work")

    return build_duplication_cost_estimate(
        model=str(model),
        kernel_work=kernel_work,
    )


def _non_negative_int(mapping: Mapping[str, Any], field: str) -> int:
    value = mapping.get(field, 0)
    result = int(value)
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result
