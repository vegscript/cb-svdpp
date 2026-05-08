from __future__ import annotations

import pytest

from recsys_lab.experiments.duplication_profile import (
    DUPLICATION_PROFILE_VERSION,
    build_duplication_cost_estimate,
    build_duplication_cost_estimate_from_kernel_profile,
)


def test_duplication_profile_has_no_explicit_duplication_for_biased_mf() -> None:
    estimate = build_duplication_cost_estimate(
        model="biased_mf",
        kernel_work={
            "implicit_history_visits": 10,
            "explicit_history_visits": 20,
            "cluster_history_visits": 30,
        },
    )

    assert estimate["profile_version"] == DUPLICATION_PROFILE_VERSION
    assert estimate["duplicated_explicit_residual_computations"] == 0
    assert estimate["duplicated_history_cluster_lookups"] == 0
    assert estimate["duplicated_history_traversals"] == 0
    assert estimate["potential_exact_reuse_scope"] == []


def test_duplication_profile_has_no_explicit_duplication_for_svdpp() -> None:
    estimate = build_duplication_cost_estimate(
        model="svdpp",
        kernel_work={
            "implicit_history_visits": 10,
            "explicit_history_visits": 0,
            "cluster_history_visits": 0,
        },
    )

    assert estimate["duplicated_explicit_residual_computations"] == 0
    assert estimate["duplicated_history_cluster_lookups"] == 0
    assert estimate["duplicated_history_traversals"] == 10
    assert estimate["potential_exact_reuse_scope"] == []


def test_duplication_profile_counts_explicit_residual_duplication() -> None:
    estimate = build_duplication_cost_estimate(
        model="asvdpp",
        kernel_work={
            "implicit_history_visits": 11,
            "explicit_history_visits": 7,
            "cluster_history_visits": 0,
        },
    )

    assert estimate["duplicated_explicit_residual_computations"] == 7
    assert estimate["duplicated_history_cluster_lookups"] == 0
    assert estimate["duplicated_history_traversals"] == 18
    assert estimate["duplicated_history_traversal_breakdown"] == {
        "explicit_context_and_update": 7,
        "implicit_context_and_update": 11,
        "cluster_context_and_update": 0,
    }
    assert estimate["potential_exact_reuse_scope"] == [
        "raw_explicit_residual_weights"
    ]


def test_duplication_profile_counts_cb_asvdpp_cluster_lookup_duplication() -> None:
    estimate = build_duplication_cost_estimate(
        model="cb_asvdpp",
        kernel_work={
            "implicit_history_visits": 13,
            "explicit_history_visits": 17,
            "cluster_history_visits": 19,
        },
    )

    assert estimate["duplicated_explicit_residual_computations"] == 17
    assert estimate["duplicated_history_cluster_lookups"] == 36
    assert estimate["duplicated_history_cluster_lookup_breakdown"] == {
        "explicit_item_cluster_lookup": 17,
        "cluster_history_cluster_id_lookup": 19,
    }
    assert estimate["duplicated_history_traversals"] == 49
    assert estimate["potential_exact_reuse_scope"] == [
        "raw_explicit_residual_weights",
        "explicit_history_cluster_ids",
        "cluster_history_ids_counts",
    ]


def test_duplication_profile_from_kernel_profile_payload() -> None:
    estimate = build_duplication_cost_estimate_from_kernel_profile(
        {
            "model": "cb_svdpp",
            "estimated_kernel_work": {
                "implicit_history_visits": "5",
                "explicit_history_visits": "0",
                "cluster_history_visits": "3",
            },
        }
    )

    assert estimate["model"] == "cb_svdpp"
    assert estimate["duplicated_explicit_residual_computations"] == 0
    assert estimate["duplicated_history_cluster_lookups"] == 3
    assert estimate["duplicated_history_traversals"] == 8
    assert estimate["potential_exact_reuse_scope"] == ["cluster_history_ids_counts"]


def test_duplication_profile_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_duplication_cost_estimate(
            model="unknown",
            kernel_work={},
        )


def test_duplication_profile_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="implicit_history_visits"):
        build_duplication_cost_estimate(
            model="svdpp",
            kernel_work={"implicit_history_visits": -1},
        )
