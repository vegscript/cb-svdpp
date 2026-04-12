from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from recsys_lab.core.types import Interaction


@dataclass(frozen=True, slots=True)
class InteractionDataset:
    interactions: tuple[Interaction, ...]
    n_users: int
    n_items: int
    name: str = "unknown"

    @classmethod
    def from_iterable(
        cls,
        interactions: Iterable[Interaction],
        *,
        n_users: int,
        n_items: int,
        name: str = "unknown",
    ) -> "InteractionDataset":
        return cls(tuple(interactions), n_users=n_users, n_items=n_items, name=name)

    def __len__(self) -> int:
        return len(self.interactions)
