from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ModelScope = Literal["paper_faithful", "paper_inspired", "extended"]
RuntimeDType = Literal["float32", "float64"]
TrainingBackend = Literal["auto", "python", "numba"]
ImplicitPolicy = Literal["ratings_as_implicit"]
ResidualWeightContract = Literal["detached"]
ClusteringAlgorithm = Literal["kmeans"]


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetadataSchema(StrictSchema):
    status: str
    owner: str
    purpose: str
    provenance: dict[str, Any] | None = None


class ModelIdentitySchema(StrictSchema):
    name: str
    family: str
    scope: ModelScope


class BiasedMFTrainingSchema(StrictSchema):
    latent_dim: int = Field(gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    lambda_b: float = Field(ge=0)
    lambda_p: float = Field(ge=0)
    lambda_q: float = Field(ge=0)
    init_std: float = Field(gt=0)
    dtype: RuntimeDType
    training_backend: TrainingBackend


class ClusterInductionSchema(StrictSchema):
    latent_dim: int = Field(gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    lambda_b: float = Field(ge=0)
    lambda_p: float = Field(ge=0)
    lambda_q: float = Field(ge=0)
    seed: int
    init_std: float = Field(gt=0)
    dtype: RuntimeDType
    training_backend: TrainingBackend = "auto"


class SVDppTrainingSchema(BiasedMFTrainingSchema):
    lambda_y: float = Field(ge=0)
    implicit_policy: ImplicitPolicy


class AsymmetricSVDTrainingSchema(StrictSchema):
    latent_dim: int = Field(gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    lambda_b: float = Field(ge=0)
    lambda_q: float = Field(ge=0)
    lambda_x: float = Field(ge=0)
    lambda_y: float = Field(ge=0)
    init_std: float = Field(gt=0)
    dtype: RuntimeDType
    implicit_policy: ImplicitPolicy
    residual_weight_contract: ResidualWeightContract


class ASVDppTrainingSchema(AsymmetricSVDTrainingSchema):
    lambda_p: float = Field(ge=0)


class CBSVDppTrainingSchema(StrictSchema):
    latent_dim: int = Field(gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    lambda_b: float = Field(ge=0)
    lambda_p: float = Field(ge=0)
    lambda_q: float = Field(ge=0)
    lambda_y: float = Field(ge=0)
    lambda_pC: float = Field(ge=0)
    lambda_qC: float = Field(ge=0)
    lambda_yC: float = Field(ge=0)
    init_std: float = Field(gt=0)
    dtype: RuntimeDType
    implicit_policy: ImplicitPolicy


class CBASVDppTrainingSchema(CBSVDppTrainingSchema):
    lambda_x: float = Field(ge=0)
    lambda_xC: float = Field(ge=0)
    residual_weight_contract: ResidualWeightContract


class ClusteringSchema(StrictSchema):
    n_user_clusters: int = Field(gt=0)
    n_item_clusters: int = Field(gt=0)
    alpha: float = Field(ge=0.0, le=1.0)
    algorithm: ClusteringAlgorithm
    kmeans_n_init: int = Field(gt=0)
    induction: ClusterInductionSchema | None = None


class ModelProfileSchema(StrictSchema):
    metadata: MetadataSchema
    model: ModelIdentitySchema
    notes: list[str]

    expected_model_name: ClassVar[str] = ""

    @model_validator(mode="after")
    def _validate_model_name(self) -> "ModelProfileSchema":
        if self.expected_model_name and self.model.name != self.expected_model_name:
            raise ValueError(
                f"model.name must be '{self.expected_model_name}' for this schema, got '{self.model.name}'"
            )
        return self


class BiasedMFModelProfile(ModelProfileSchema):
    expected_model_name: ClassVar[str] = "biased_mf"
    training: BiasedMFTrainingSchema


class SVDppModelProfile(ModelProfileSchema):
    expected_model_name: ClassVar[str] = "svdpp"
    training: SVDppTrainingSchema


class AsymmetricSVDModelProfile(ModelProfileSchema):
    expected_model_name: ClassVar[str] = "asymmetric_svd"
    training: AsymmetricSVDTrainingSchema


class ASVDppModelProfile(ModelProfileSchema):
    expected_model_name: ClassVar[str] = "asvdpp"
    training: ASVDppTrainingSchema


class CBSVDppModelProfile(ModelProfileSchema):
    expected_model_name: ClassVar[str] = "cb_svdpp"
    training: CBSVDppTrainingSchema
    clustering: ClusteringSchema


class CBASVDppModelProfile(ModelProfileSchema):
    expected_model_name: ClassVar[str] = "cb_asvdpp"
    training: CBASVDppTrainingSchema
    clustering: ClusteringSchema
