from __future__ import annotations

from enum import StrEnum


class SearchRole(StrEnum):
    INNER_TARGET_PARAM = "INNER_TARGET_PARAM"
    OUTER_CLUSTER_PARAM = "OUTER_CLUSTER_PARAM"
    OTHER = "OTHER"


INNER_TARGET_PATHS = {
    "clustering.alpha",
    "training.learning_rate",
    "training.lambda_b",
    "training.lambda_p",
    "training.lambda_q",
    "training.lambda_y",
    "training.lambda_pC",
    "training.lambda_qC",
    "training.lambda_yC",
    "training.epochs",
}
OUTER_CLUSTER_PATHS = {
    "clustering.n_user_clusters",
    "clustering.n_item_clusters",
    "clustering.algorithm",
    "clustering.kmeans_n_init",
}


def classify_search_coordinate(target_path: str) -> SearchRole:
    if target_path in INNER_TARGET_PATHS:
        return SearchRole.INNER_TARGET_PARAM
    if target_path in OUTER_CLUSTER_PATHS or target_path.startswith("clustering.induction."):
        return SearchRole.OUTER_CLUSTER_PARAM
    return SearchRole.OTHER


def is_inner_target_coordinate(target_path: str) -> bool:
    return classify_search_coordinate(target_path) == SearchRole.INNER_TARGET_PARAM


def is_outer_cluster_coordinate(target_path: str) -> bool:
    return classify_search_coordinate(target_path) == SearchRole.OUTER_CLUSTER_PARAM
