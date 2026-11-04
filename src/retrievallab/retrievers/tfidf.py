"""TF-IDF vectors compared by cosine similarity.

Weights: sublinear term frequency ``1 + ln(tf)`` times smoothed inverse document frequency
``ln((1 + N) / (1 + df)) + 1`` (the scikit-learn default), applied to chunks and queries alike;
each vector is L2-normalised. Unlike BM25 there is no length normalisation beyond the cosine
and no term-frequency saturation parameter.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence

from retrievallab.chunking import Chunk
from retrievallab.retrievers.base import Hit, top_k
from retrievallab.text import tokenize


class TfidfRetriever:
    def __init__(self, tokenizer: Callable[[str], list[str]] = tokenize) -> None:
        self.tokenizer = tokenizer
        self.name = "tfidf"
        self._chunks: list[Chunk] = []
        self._postings: dict[str, list[tuple[int, float]]] = {}
        self._df: dict[str, int] = {}

    def idf(self, term: str) -> float:
        n = len(self._chunks)
        return math.log((1 + n) / (1 + self._df.get(term, 0))) + 1.0

    def _weights(self, tokens: list[str]) -> dict[str, float]:
        weights = {t: (1.0 + math.log(tf)) * self.idf(t) for t, tf in Counter(tokens).items()}
        norm = math.sqrt(sum(w * w for w in weights.values()))
        return {t: w / norm for t, w in weights.items()} if norm > 0 else {}

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        tokenised = [self.tokenizer(c.text) for c in self._chunks]
        self._df = {}
        for tokens in tokenised:
            for t in set(tokens):
                self._df[t] = self._df.get(t, 0) + 1
        self._postings = {}
        for i, tokens in enumerate(tokenised):
            for t, w in self._weights(tokens).items():
                self._postings.setdefault(t, []).append((i, w))

    def score_all(self, query: str) -> dict[int, float]:
        """Cosine similarity between the query and every chunk sharing a term with it."""
        # Query terms unseen in the index have no postings but still count in the query norm.
        qweights = self._weights(self.tokenizer(query))
        scores: dict[int, float] = {}
        for t, qw in qweights.items():
            for i, dw in self._postings.get(t, ()):
                scores[i] = scores.get(i, 0.0) + qw * dw
        return scores

    def search(self, query: str, k: int) -> list[Hit]:
        return top_k(self.score_all(query), self._chunks, k)
