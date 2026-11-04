"""The retriever interface shared by every method."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from retrievallab.chunking import Chunk


@dataclass(frozen=True, slots=True)
class Hit:
    """One retrieved chunk. ``score`` is only comparable within one retriever and one query."""

    chunk: Chunk
    score: float


class Retriever(Protocol):
    name: str

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Build the index. Calling it again replaces the previous index."""
        ...

    def search(self, query: str, k: int) -> list[Hit]:
        """Return at most ``k`` hits, best first; ties are broken by chunk order in the index."""
        ...


def top_k(scores: dict[int, float], chunks: Sequence[Chunk], k: int) -> list[Hit]:
    """Rank positive scores descending with a deterministic tie-break on index position."""
    if k <= 0:
        raise ValueError("k must be positive")
    ranked = sorted((i for i, s in scores.items() if s > 0.0), key=lambda i: (-scores[i], i))
    return [Hit(chunks[i], scores[i]) for i in ranked[:k]]
