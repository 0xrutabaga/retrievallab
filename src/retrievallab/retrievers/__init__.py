"""Retrievers behind one interface, built by name."""

from __future__ import annotations

from retrievallab.retrievers.base import Hit, Retriever
from retrievallab.retrievers.bm25 import BM25Retriever
from retrievallab.retrievers.dense import DenseRetriever, SentenceTransformerEncoder
from retrievallab.retrievers.hybrid import HybridRRFRetriever
from retrievallab.retrievers.tfidf import TfidfRetriever

METHODS = ("bm25", "tfidf", "hybrid", "dense", "hybrid-dense")
DEFAULT_METHODS = ("bm25", "tfidf", "hybrid")


def make_retriever(name: str) -> Retriever:
    """Build a retriever: ``bm25``, ``tfidf``, ``hybrid`` (RRF of bm25 and tfidf), ``dense`` or
    ``hybrid-dense`` (RRF of bm25 and dense). The last two need the ``dense`` extra."""
    if name == "bm25":
        return BM25Retriever()
    if name == "tfidf":
        return TfidfRetriever()
    if name == "hybrid":
        return HybridRRFRetriever([BM25Retriever(), TfidfRetriever()])
    if name == "dense":
        return DenseRetriever(SentenceTransformerEncoder())
    if name == "hybrid-dense":
        return HybridRRFRetriever([BM25Retriever(), DenseRetriever(SentenceTransformerEncoder())])
    raise ValueError(f"unknown method {name!r}; choose from {', '.join(METHODS)}")


__all__ = [
    "DEFAULT_METHODS",
    "METHODS",
    "BM25Retriever",
    "DenseRetriever",
    "Hit",
    "HybridRRFRetriever",
    "Retriever",
    "TfidfRetriever",
    "make_retriever",
]
