"""Okapi BM25, implemented from the formula over an in-memory inverted index.

For a query Q and chunk D::

    score(D, Q) = sum over distinct terms t in Q of
        idf(t) * tf(t, D) * (k1 + 1) / (tf(t, D) + k1 * (1 - b + b * |D| / avgdl))

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

This is the non-negative IDF variant used by Lucene, so a term that appears in more than half
the chunks still adds a small positive amount instead of a penalty. |D| is the chunk length in
tokens after tokenisation (stopwords removed).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence

from retrievallab.chunking import Chunk
from retrievallab.retrievers.base import Hit, top_k
from retrievallab.text import tokenize


class BM25Retriever:
    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = tokenize,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.name = "bm25"
        self._chunks: list[Chunk] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._lengths: list[int] = []
        self._avgdl = 0.0

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._postings = {}
        self._lengths = []
        for i, chunk in enumerate(self._chunks):
            tokens = self.tokenizer(chunk.text)
            self._lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self._postings.setdefault(term, []).append((i, tf))
        n = len(self._chunks)
        self._avgdl = sum(self._lengths) / n if n else 0.0

    def idf(self, term: str) -> float:
        n = len(self._chunks)
        df = len(self._postings.get(term, ()))
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score_all(self, query: str) -> dict[int, float]:
        """BM25 score of every chunk that contains at least one query term."""
        scores: dict[int, float] = {}
        if not self._chunks or self._avgdl == 0.0:
            return scores
        for term in dict.fromkeys(self.tokenizer(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for i, tf in postings:
                norm = self.k1 * (1.0 - self.b + self.b * self._lengths[i] / self._avgdl)
                scores[i] = scores.get(i, 0.0) + idf * tf * (self.k1 + 1.0) / (tf + norm)
        return scores

    def search(self, query: str, k: int) -> list[Hit]:
        return top_k(self.score_all(query), self._chunks, k)
