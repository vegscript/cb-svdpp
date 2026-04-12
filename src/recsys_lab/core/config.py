from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    latent_dim: int = 50
    epochs: int = 20
    learning_rate: float = 0.01
    regularization: float = 0.02
    seed: int = 42


@dataclass(frozen=True, slots=True)
class SVDPPConfig(TrainingConfig):
    use_implicit_feedback: bool = True


@dataclass(frozen=True, slots=True)
class ClusterConfig:
    n_user_clusters: int = 100
    n_item_clusters: int = 100
    alpha: float = 0.1
    random_state: int = 42
