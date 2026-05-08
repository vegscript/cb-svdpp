from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ProductiveModelName = Literal[
    "biased_mf",
    "svdpp",
    "asymmetric_svd",
    "asvdpp",
    "cb_svdpp",
    "cb_asvdpp",
]
SearchSpaceVersion = Literal["tuning_search_space_v1"]
DimensionType = Literal["categorical", "float", "int"]
DistributionName = Literal["uniform", "loguniform"]
GeneratorName = Literal["grid", "manual"]
ObjectiveDirection = Literal["minimize", "maximize"]
MetricAggregation = Literal["mean", "std", "min", "max"]

PRODUCTIVE_CB_MODELS = {"cb_svdpp", "cb_asvdpp"}
FORBIDDEN_PRIMARY_TUNING_METRICS = {"test_rmse", "test_mae"}
DEFAULT_CLUSTER_ARTIFACT_REUSE_ACROSS = (
    "alpha",
    "learning_rate",
    "lambda_p",
    "lambda_q",
    "lambda_y",
    "lambda_pC",
    "lambda_qC",
    "lambda_yC",
    "epochs",
)
DEFAULT_CLUSTER_ARTIFACT_INVALIDATE_ON = (
    "n_user_clusters",
    "n_item_clusters",
    "induction_config",
    "kmeans_n_init",
    "clustering_algorithm",
    "dataset",
    "split",
    "train_fingerprint",
)


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StudySpec(StrictSchema):
    name: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    split_family: str = Field(min_length=1)
    model: ProductiveModelName
    seed: int


class BudgetSpec(StrictSchema):
    max_candidates: int = Field(gt=0)
    max_parallel: int = Field(default=1, gt=0)
    max_wall_seconds: float | None = Field(default=None, gt=0)


class GeneratorSpec(StrictSchema):
    type: GeneratorName = "grid"
    deterministic_order: bool = True


class DimensionSpec(StrictSchema):
    type: DimensionType
    distribution: DistributionName | None = None
    low: float | int | None = None
    high: float | int | None = None
    values: list[Any] | None = None
    target_path: str | None = None

    @model_validator(mode="after")
    def _validate_dimension_shape(self) -> "DimensionSpec":
        if self.type == "categorical":
            if self.values is None or len(self.values) == 0:
                raise ValueError("categorical dimensions require non-empty values")
            if self.distribution is not None or self.low is not None or self.high is not None:
                raise ValueError("categorical dimensions must not define distribution, low, or high")
            return self

        if self.values is not None:
            if len(self.values) == 0:
                raise ValueError("numeric dimensions with values require non-empty values")
            if self.distribution is not None or self.low is not None or self.high is not None:
                raise ValueError("numeric dimensions must use either values or low/high distribution")
            return self
        if self.low is None or self.high is None:
            raise ValueError("numeric dimensions require low and high")
        if self.low >= self.high:
            raise ValueError("numeric dimension low must be smaller than high")
        if self.distribution is None:
            raise ValueError("numeric dimensions require a distribution")
        if self.distribution == "loguniform" and self.low <= 0:
            raise ValueError("loguniform dimensions require low > 0")
        return self


class ClusterArtifactReuseSpec(StrictSchema):
    reuse_across: list[str] = Field(min_length=1)
    invalidate_on: list[str] = Field(min_length=1)


class ArtifactReuseSpec(StrictSchema):
    cluster_artifacts: ClusterArtifactReuseSpec | None = None


class ObjectiveMetricSpec(StrictSchema):
    metric: str = Field(min_length=1)
    direction: ObjectiveDirection = "minimize"
    aggregation: MetricAggregation = "mean"


class ObjectiveSpec(StrictSchema):
    primary: ObjectiveMetricSpec
    secondary: list[ObjectiveMetricSpec] = Field(default_factory=list)
    required_guards: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _reject_test_metric_primary_objective(self) -> "ObjectiveSpec":
        metric = self.primary.metric.lower()
        if metric in FORBIDDEN_PRIMARY_TUNING_METRICS or metric.startswith("test_"):
            raise ValueError("test metrics must not be primary tuning objectives")
        return self


class SearchSpaceSpec(StrictSchema):
    search_space_version: SearchSpaceVersion
    study: StudySpec
    base_model_config: str = Field(min_length=1)
    budget: BudgetSpec
    generator: GeneratorSpec = Field(default_factory=GeneratorSpec)
    search_space: dict[str, DimensionSpec] = Field(min_length=1)
    manual_candidates: list[dict[str, Any]] | None = None
    artifact_reuse: ArtifactReuseSpec | None = None
    objective: ObjectiveSpec

    @model_validator(mode="after")
    def _validate_generator_contract(self) -> "SearchSpaceSpec":
        if self.generator.type == "manual":
            if not self.manual_candidates:
                raise ValueError("manual generator requires manual_candidates")
            return self
        if self.manual_candidates is not None:
            raise ValueError("manual_candidates are only valid for manual generator")
        return self

    @model_validator(mode="after")
    def _validate_productive_cb_alpha_policy(self) -> "SearchSpaceSpec":
        if self.study.model not in PRODUCTIVE_CB_MODELS:
            return self

        alpha_dimension_names = [
            name for name, dimension in self.search_space.items() if _is_alpha_dimension(name, dimension)
        ]
        for dimension_name in alpha_dimension_names:
            alpha_dimension = self.search_space[dimension_name]
            if alpha_dimension.type == "categorical":
                values = alpha_dimension.values or []
                if any(float(value) <= 0.0 or float(value) >= 1.0 for value in values):
                    raise ValueError("productive CB search spaces require alpha values strictly between 0 and 1")
            elif alpha_dimension.values is not None:
                if any(float(value) <= 0.0 or float(value) >= 1.0 for value in alpha_dimension.values):
                    raise ValueError("productive CB search spaces require alpha values strictly between 0 and 1")
            elif alpha_dimension.low is not None and alpha_dimension.high is not None and (
                float(alpha_dimension.low) <= 0.0 or float(alpha_dimension.high) >= 1.0
            ):
                raise ValueError("productive CB search spaces require alpha bounds strictly between 0 and 1")
        if self.manual_candidates is not None:
            for candidate in self.manual_candidates:
                for dimension_name in alpha_dimension_names:
                    if dimension_name in candidate and (
                        float(candidate[dimension_name]) <= 0.0 or float(candidate[dimension_name]) >= 1.0
                    ):
                        raise ValueError("productive CB manual candidates require alpha strictly between 0 and 1")
        return self


def default_cluster_artifact_reuse_spec() -> ClusterArtifactReuseSpec:
    return ClusterArtifactReuseSpec(
        reuse_across=list(DEFAULT_CLUSTER_ARTIFACT_REUSE_ACROSS),
        invalidate_on=list(DEFAULT_CLUSTER_ARTIFACT_INVALIDATE_ON),
    )


def _is_alpha_dimension(name: str, dimension: DimensionSpec) -> bool:
    target_path = dimension.target_path or name
    return name == "alpha" or target_path in {"alpha", "clustering.alpha"} or target_path.split(".")[-1] == "alpha"
