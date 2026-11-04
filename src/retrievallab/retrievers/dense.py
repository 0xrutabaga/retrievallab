"""Optional dense retriever: embed chunks and queries, rank by cosine similarity.

The embedding model sits behind the ``Encoder`` protocol. The default encoder wraps
``sentence-transformers``, which is an optional extra (``uv sync --extra dense``); it is never
imported by the core package, the default eval or CI. Tests use a deterministic fake encoder.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from retrievallab.chunking import Chunk
from retrievallab.retrievers.base import Hit

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DenseUnavailableError(RuntimeError):
    """Raised when the dense extra is not installed."""


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEncoder:
    """Adapter for ``sentence_transformers.SentenceTransformer`` (downloads on first use)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise DenseUnavailableError(
                "dense retrieval needs the optional extra: uv sync --extra dense"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts), batch_size=self._batch_size, show_progress_bar=False
        )
        return [[float(x) for x in v] for v in vectors]


def _normalise(v: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm > 0 else [0.0 for _ in v]


class DenseRetriever:
    def __init__(self, encoder: Encoder, name: str = "dense") -> None:
        self.encoder = encoder
        self.name = name
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        vectors = self.encoder.encode([c.text for c in self._chunks]) if self._chunks else []
        if len(vectors) != len(self._chunks):
            raise ValueError("encoder returned a different number of vectors than texts")
        self._vectors = [_normalise(v) for v in vectors]

    def search(self, query: str, k: int) -> list[Hit]:
        if k <= 0:
            raise ValueError("k must be positive")
        if not self._chunks or not query.strip():
            return []
        q = _normalise(self.encoder.encode([query])[0])
        scores = [sum(a * b for a, b in zip(q, v, strict=True)) for v in self._vectors]
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [Hit(self._chunks[i], scores[i]) for i in ranked]
