from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel

from recsys_lab.clustering import ClusterArtifacts
from recsys_lab.data.histories import UserClusterCountIndex, UserExplicitFeedbackIndex, UserHistoryIndex
from recsys_lab.data.processed import RatingsData
from recsys_lab.models.asvdpp import ASVDppConfig, ASVDppRecommender
from recsys_lab.models.asymmetric_svd import AsymmetricSVDConfig, AsymmetricSVDRecommender
from recsys_lab.models.biased_mf import BiasedMFConfig, BiasedMFRecommender
from recsys_lab.models.cb_asvdpp import CBASVDppConfig, CBASVDppRecommender
from recsys_lab.models.cb_svdpp import CBSVDppConfig, CBSVDppRecommender
from recsys_lab.models.config_schemas import (
    ASVDppModelProfile,
    AsymmetricSVDModelProfile,
    BiasedMFModelProfile,
    CBASVDppModelProfile,
    CBSVDppModelProfile,
    ModelProfileSchema,
    SVDppModelProfile,
)
from recsys_lab.models.svdpp import SVDppConfig, SVDppRecommender


@dataclass(frozen=True, slots=True)
class ModelRequirements:
    needs_implicit_history: bool = False
    needs_explicit_feedback: bool = False
    needs_cluster_artifacts: bool = False
    needs_user_cluster_history: bool = False
    supports_precomputed_indices: bool = True
    supports_context_cache: bool = True

    def artifact_names(self) -> list[str]:
        names: list[str] = []
        if self.needs_implicit_history:
            names.append("user_history_index")
        if self.needs_explicit_feedback:
            names.append("explicit_feedback_index")
        if self.needs_cluster_artifacts:
            names.append("cluster_artifacts")
        if self.needs_user_cluster_history:
            names.append("user_cluster_history_index")
        return names


@dataclass(frozen=True, slots=True)
class FitArtifacts:
    user_history_index: UserHistoryIndex | None = None
    explicit_feedback_index: UserExplicitFeedbackIndex | None = None
    user_cluster_history_index: UserClusterCountIndex | None = None
    cluster_artifacts: ClusterArtifacts | None = None

    def available_artifact_names(self) -> list[str]:
        names: list[str] = []
        if self.user_history_index is not None:
            names.append("user_history_index")
        if self.explicit_feedback_index is not None:
            names.append("explicit_feedback_index")
        if self.user_cluster_history_index is not None:
            names.append("user_cluster_history_index")
        if self.cluster_artifacts is not None:
            names.append("cluster_artifacts")
        return names


class ModelAdapter:
    name: ClassVar[str]
    family: ClassVar[str]
    requirements: ClassVar[ModelRequirements]
    config_schema: ClassVar[type[ModelProfileSchema]]

    @classmethod
    def runtime_dtype(cls, profile: ModelProfileSchema) -> str:
        return str(profile.training.dtype)  # type: ignore[attr-defined]

    @classmethod
    def build_model_config(cls, profile: ModelProfileSchema, *, model_seed: int, runtime_dtype: str) -> object:
        raise NotImplementedError

    @classmethod
    def build_induction_config(cls, model_config: object, *, model_seed: int) -> BiasedMFConfig | None:
        return None

    @classmethod
    def instantiate(cls, model_config: object, *, artifacts: FitArtifacts) -> object:
        raise NotImplementedError

    @classmethod
    def fit(
        cls,
        model: object,
        train_data: RatingsData,
        *,
        artifacts: FitArtifacts,
        reuse_precomputed_indices: bool,
    ) -> object:
        raise NotImplementedError

    @classmethod
    def config_payload(cls, model_config: object) -> dict[str, Any]:
        return asdict(model_config)

    @classmethod
    def cb_alpha(cls, profile: ModelProfileSchema) -> float | None:
        return None


class BiasedMFAdapter(ModelAdapter):
    name = "biased_mf"
    family = "matrix_factorization"
    requirements = ModelRequirements()
    config_schema = BiasedMFModelProfile

    @classmethod
    def build_model_config(
        cls,
        profile: BiasedMFModelProfile,
        *,
        model_seed: int,
        runtime_dtype: str,
    ) -> BiasedMFConfig:
        training = profile.training
        return BiasedMFConfig(
            latent_dim=training.latent_dim,
            epochs=training.epochs,
            learning_rate=training.learning_rate,
            lambda_b=training.lambda_b,
            lambda_p=training.lambda_p,
            lambda_q=training.lambda_q,
            seed=model_seed,
            init_std=training.init_std,
            dtype=runtime_dtype,
            training_backend=training.training_backend,
        )

    @classmethod
    def instantiate(cls, model_config: BiasedMFConfig, *, artifacts: FitArtifacts) -> BiasedMFRecommender:
        return BiasedMFRecommender(model_config)

    @classmethod
    def fit(
        cls,
        model: BiasedMFRecommender,
        train_data: RatingsData,
        *,
        artifacts: FitArtifacts,
        reuse_precomputed_indices: bool,
    ) -> BiasedMFRecommender:
        return model.fit(train_data)


class SVDppAdapter(ModelAdapter):
    name = "svdpp"
    family = "implicit_factorization"
    requirements = ModelRequirements(needs_implicit_history=True)
    config_schema = SVDppModelProfile

    @classmethod
    def build_model_config(
        cls,
        profile: SVDppModelProfile,
        *,
        model_seed: int,
        runtime_dtype: str,
    ) -> SVDppConfig:
        training = profile.training
        return SVDppConfig(
            latent_dim=training.latent_dim,
            epochs=training.epochs,
            learning_rate=training.learning_rate,
            lambda_b=training.lambda_b,
            lambda_p=training.lambda_p,
            lambda_q=training.lambda_q,
            lambda_y=training.lambda_y,
            seed=model_seed,
            init_std=training.init_std,
            dtype=runtime_dtype,
            implicit_policy=training.implicit_policy,
            training_backend=training.training_backend,
        )

    @classmethod
    def instantiate(cls, model_config: SVDppConfig, *, artifacts: FitArtifacts) -> SVDppRecommender:
        return SVDppRecommender(model_config)

    @classmethod
    def fit(
        cls,
        model: SVDppRecommender,
        train_data: RatingsData,
        *,
        artifacts: FitArtifacts,
        reuse_precomputed_indices: bool,
    ) -> SVDppRecommender:
        return model.fit(
            train_data,
            user_histories=artifacts.user_history_index if reuse_precomputed_indices else None,
        )


class AsymmetricSVDAdapter(ModelAdapter):
    name = "asymmetric_svd"
    family = "asymmetric_factorization"
    requirements = ModelRequirements(needs_implicit_history=True, needs_explicit_feedback=True)
    config_schema = AsymmetricSVDModelProfile

    @classmethod
    def build_model_config(
        cls,
        profile: AsymmetricSVDModelProfile,
        *,
        model_seed: int,
        runtime_dtype: str,
    ) -> AsymmetricSVDConfig:
        training = profile.training
        return AsymmetricSVDConfig(
            latent_dim=training.latent_dim,
            epochs=training.epochs,
            learning_rate=training.learning_rate,
            lambda_b=training.lambda_b,
            lambda_q=training.lambda_q,
            lambda_x=training.lambda_x,
            lambda_y=training.lambda_y,
            seed=model_seed,
            init_std=training.init_std,
            dtype=runtime_dtype,
            implicit_policy=training.implicit_policy,
            residual_weight_contract=training.residual_weight_contract,
        )

    @classmethod
    def instantiate(cls, model_config: AsymmetricSVDConfig, *, artifacts: FitArtifacts) -> AsymmetricSVDRecommender:
        return AsymmetricSVDRecommender(model_config)

    @classmethod
    def fit(
        cls,
        model: AsymmetricSVDRecommender,
        train_data: RatingsData,
        *,
        artifacts: FitArtifacts,
        reuse_precomputed_indices: bool,
    ) -> AsymmetricSVDRecommender:
        return model.fit(
            train_data,
            explicit_feedback=artifacts.explicit_feedback_index if reuse_precomputed_indices else None,
            implicit_history=artifacts.user_history_index if reuse_precomputed_indices else None,
        )


class ASVDppAdapter(ModelAdapter):
    name = "asvdpp"
    family = "poster_extended_factorization"
    requirements = ModelRequirements(needs_implicit_history=True, needs_explicit_feedback=True)
    config_schema = ASVDppModelProfile

    @classmethod
    def build_model_config(
        cls,
        profile: ASVDppModelProfile,
        *,
        model_seed: int,
        runtime_dtype: str,
    ) -> ASVDppConfig:
        training = profile.training
        return ASVDppConfig(
            latent_dim=training.latent_dim,
            epochs=training.epochs,
            learning_rate=training.learning_rate,
            lambda_b=training.lambda_b,
            lambda_p=training.lambda_p,
            lambda_q=training.lambda_q,
            lambda_x=training.lambda_x,
            lambda_y=training.lambda_y,
            seed=model_seed,
            init_std=training.init_std,
            dtype=runtime_dtype,
            implicit_policy=training.implicit_policy,
            residual_weight_contract=training.residual_weight_contract,
        )

    @classmethod
    def instantiate(cls, model_config: ASVDppConfig, *, artifacts: FitArtifacts) -> ASVDppRecommender:
        return ASVDppRecommender(model_config)

    @classmethod
    def fit(
        cls,
        model: ASVDppRecommender,
        train_data: RatingsData,
        *,
        artifacts: FitArtifacts,
        reuse_precomputed_indices: bool,
    ) -> ASVDppRecommender:
        return model.fit(
            train_data,
            explicit_feedback=artifacts.explicit_feedback_index if reuse_precomputed_indices else None,
            implicit_history=artifacts.user_history_index if reuse_precomputed_indices else None,
        )


class CBSVDppAdapter(ModelAdapter):
    name = "cb_svdpp"
    family = "clustering_based_factorization"
    requirements = ModelRequirements(
        needs_implicit_history=True,
        needs_cluster_artifacts=True,
        needs_user_cluster_history=True,
    )
    config_schema = CBSVDppModelProfile

    @classmethod
    def cb_alpha(cls, profile: CBSVDppModelProfile) -> float:
        return float(profile.clustering.alpha)

    @classmethod
    def build_model_config(
        cls,
        profile: CBSVDppModelProfile,
        *,
        model_seed: int,
        runtime_dtype: str,
    ) -> CBSVDppConfig:
        training = profile.training
        return CBSVDppConfig(
            latent_dim=training.latent_dim,
            epochs=training.epochs,
            learning_rate=training.learning_rate,
            lambda_b=training.lambda_b,
            lambda_p=training.lambda_p,
            lambda_q=training.lambda_q,
            lambda_y=training.lambda_y,
            lambda_pC=training.lambda_pC,
            lambda_qC=training.lambda_qC,
            lambda_yC=training.lambda_yC,
            alpha=profile.clustering.alpha,
            seed=model_seed,
            init_std=training.init_std,
            dtype=runtime_dtype,
            implicit_policy=training.implicit_policy,
        )

    @classmethod
    def build_induction_config(cls, model_config: CBSVDppConfig, *, model_seed: int) -> BiasedMFConfig:
        return BiasedMFConfig(
            latent_dim=model_config.latent_dim,
            epochs=model_config.epochs,
            learning_rate=model_config.learning_rate,
            lambda_b=model_config.lambda_b,
            lambda_p=model_config.lambda_p,
            lambda_q=model_config.lambda_q,
            seed=model_seed,
            init_std=model_config.init_std,
            dtype=model_config.dtype,
            training_backend="auto",
        )

    @classmethod
    def instantiate(cls, model_config: CBSVDppConfig, *, artifacts: FitArtifacts) -> CBSVDppRecommender:
        if artifacts.cluster_artifacts is None:
            raise ValueError("cb_svdpp requires cluster_artifacts")
        cluster_artifacts = artifacts.cluster_artifacts
        return CBSVDppRecommender(
            model_config,
            user_clusters=np.asarray(cluster_artifacts.user_clusters, dtype=np.int32),
            item_clusters=np.asarray(cluster_artifacts.item_clusters, dtype=np.int32),
            n_user_clusters=cluster_artifacts.r_star_counts.shape[0],
            n_item_clusters=cluster_artifacts.r_star_counts.shape[1],
        )

    @classmethod
    def fit(
        cls,
        model: CBSVDppRecommender,
        train_data: RatingsData,
        *,
        artifacts: FitArtifacts,
        reuse_precomputed_indices: bool,
    ) -> CBSVDppRecommender:
        # False intentionally leaves index construction to the model fallback path.
        return model.fit(
            train_data,
            user_histories=artifacts.user_history_index if reuse_precomputed_indices else None,
            user_cluster_histories=artifacts.user_cluster_history_index if reuse_precomputed_indices else None,
        )


class CBASVDppAdapter(ModelAdapter):
    name = "cb_asvdpp"
    family = "clustering_based_factorization"
    requirements = ModelRequirements(
        needs_implicit_history=True,
        needs_explicit_feedback=True,
        needs_cluster_artifacts=True,
        needs_user_cluster_history=True,
    )
    config_schema = CBASVDppModelProfile

    @classmethod
    def cb_alpha(cls, profile: CBASVDppModelProfile) -> float:
        return float(profile.clustering.alpha)

    @classmethod
    def build_model_config(
        cls,
        profile: CBASVDppModelProfile,
        *,
        model_seed: int,
        runtime_dtype: str,
    ) -> CBASVDppConfig:
        training = profile.training
        return CBASVDppConfig(
            latent_dim=training.latent_dim,
            epochs=training.epochs,
            learning_rate=training.learning_rate,
            lambda_b=training.lambda_b,
            lambda_p=training.lambda_p,
            lambda_q=training.lambda_q,
            lambda_x=training.lambda_x,
            lambda_y=training.lambda_y,
            lambda_pC=training.lambda_pC,
            lambda_qC=training.lambda_qC,
            lambda_xC=training.lambda_xC,
            lambda_yC=training.lambda_yC,
            alpha=profile.clustering.alpha,
            seed=model_seed,
            init_std=training.init_std,
            dtype=runtime_dtype,
            implicit_policy=training.implicit_policy,
            residual_weight_contract=training.residual_weight_contract,
        )

    @classmethod
    def build_induction_config(cls, model_config: CBASVDppConfig, *, model_seed: int) -> BiasedMFConfig:
        return BiasedMFConfig(
            latent_dim=model_config.latent_dim,
            epochs=model_config.epochs,
            learning_rate=model_config.learning_rate,
            lambda_b=model_config.lambda_b,
            lambda_p=model_config.lambda_p,
            lambda_q=model_config.lambda_q,
            seed=model_seed,
            init_std=model_config.init_std,
            dtype=model_config.dtype,
            training_backend="auto",
        )

    @classmethod
    def instantiate(cls, model_config: CBASVDppConfig, *, artifacts: FitArtifacts) -> CBASVDppRecommender:
        if artifacts.cluster_artifacts is None:
            raise ValueError("cb_asvdpp requires cluster_artifacts")
        cluster_artifacts = artifacts.cluster_artifacts
        return CBASVDppRecommender(
            model_config,
            user_clusters=np.asarray(cluster_artifacts.user_clusters, dtype=np.int32),
            item_clusters=np.asarray(cluster_artifacts.item_clusters, dtype=np.int32),
            n_user_clusters=cluster_artifacts.r_star_counts.shape[0],
            n_item_clusters=cluster_artifacts.r_star_counts.shape[1],
        )

    @classmethod
    def fit(
        cls,
        model: CBASVDppRecommender,
        train_data: RatingsData,
        *,
        artifacts: FitArtifacts,
        reuse_precomputed_indices: bool,
    ) -> CBASVDppRecommender:
        # False intentionally leaves index construction to the model fallback path.
        return model.fit(
            train_data,
            explicit_feedback=artifacts.explicit_feedback_index if reuse_precomputed_indices else None,
            implicit_history=artifacts.user_history_index if reuse_precomputed_indices else None,
            implicit_cluster_history=artifacts.user_cluster_history_index if reuse_precomputed_indices else None,
        )


MODEL_REGISTRY: dict[str, type[ModelAdapter]] = {
    "biased_mf": BiasedMFAdapter,
    "svdpp": SVDppAdapter,
    "asymmetric_svd": AsymmetricSVDAdapter,
    "asvdpp": ASVDppAdapter,
    "cb_svdpp": CBSVDppAdapter,
    "cb_asvdpp": CBASVDppAdapter,
}


def get_model_adapter(model_name: str) -> type[ModelAdapter]:
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"unsupported model '{model_name}'. supported models: {supported}") from exc


def validate_model_config_payload(
    model_config_payload: dict[str, Any],
    *,
    expected_model_name: str | None = None,
) -> tuple[type[ModelAdapter], ModelProfileSchema]:
    model_payload = model_config_payload["model"]
    if not isinstance(model_payload, dict):
        raise TypeError("model config field 'model' must be a mapping")
    model_name = model_payload["name"]
    if not isinstance(model_name, str):
        raise TypeError("model.name must be a string")
    if expected_model_name is not None and model_name != expected_model_name:
        raise ValueError(f"expected model config for '{expected_model_name}', got '{model_name}'")

    adapter = get_model_adapter(model_name)
    profile = adapter.config_schema.model_validate(model_config_payload)
    if profile.model.family != adapter.family:
        raise ValueError(f"model.family must be '{adapter.family}' for model '{adapter.name}'")
    return adapter, profile


def validated_model_config_payload_with_training_overrides(
    model_config_payload: dict[str, Any],
    *,
    training_overrides: dict[str, Any],
    expected_model_name: str | None = None,
) -> dict[str, Any]:
    validate_model_config_payload(model_config_payload, expected_model_name=expected_model_name)
    payload = copy.deepcopy(model_config_payload)
    training_payload = payload["training"]
    if not isinstance(training_payload, dict):
        raise TypeError("model config field 'training' must be a mapping")
    for key, value in training_overrides.items():
        training_payload[key] = copy.deepcopy(value)
    validate_model_config_payload(payload, expected_model_name=expected_model_name)
    return payload


def build_cb_semantics(alpha: float) -> dict[str, Any]:
    alpha_value = float(alpha)
    cluster_enabled = alpha_value > 0.0
    return {
        "alpha": alpha_value,
        "cluster_contribution_config_enabled": cluster_enabled,
        "cluster_contribution_measured": None if cluster_enabled else False,
        "cb_claim_eligible": False,
        "claim_gate_reason": (
            "alpha>0 enables cluster channel, but claim eligibility requires diagnostics "
            "and ablation evidence"
            if cluster_enabled
            else "alpha=0 disables cluster factor contribution; run is a CB-disabled ablation"
        ),
    }


def model_requirements_payload(adapter: type[ModelAdapter]) -> dict[str, Any]:
    requirements = adapter.requirements
    return {
        **asdict(requirements),
        "required_artifacts": requirements.artifact_names(),
    }


def pydantic_profile_payload(profile: BaseModel) -> dict[str, Any]:
    return profile.model_dump(mode="json")
