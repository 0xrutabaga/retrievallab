"""BM25 and TF-IDF against hand-computed scores; RRF fusion; dense with a fake encoder."""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence

import pytest

from retrievallab.chunking import Chunk
from retrievallab.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRRFRetriever,
    TfidfRetriever,
    make_retriever,
)
from retrievallab.retrievers.dense import DenseUnavailableError, SentenceTransformerEncoder
from retrievallab.retrievers.hybrid import reciprocal_rank_fusion


def chunks(*texts: str) -> list[Chunk]:
    return [Chunk(f"c{i}", "d", i, i + 1, t, "1", "") for i, t in enumerate(texts)]


def split(text: str) -> list[str]:
    return text.split()


# Three chunks with lengths 2, 3, 4 tokens, so avgdl = 3.
TOY = chunks("a b", "a c c", "b c d d")


def bm25() -> BM25Retriever:
    r = BM25Retriever(k1=1.2, b=0.75, tokenizer=split)
    r.index(TOY)
    return r


def test_bm25_idf_hand_computed() -> None:
    r = bm25()
    # idf = ln(1 + (N - df + 0.5) / (df + 0.5)) with N = 3
    assert r.idf("c") == pytest.approx(math.log(1 + 1.5 / 2.5))  # df = 2 -> ln(1.6)
    assert r.idf("d") == pytest.approx(math.log(1 + 2.5 / 1.5))  # df = 1
    assert r.idf("zzz") == pytest.approx(math.log(1 + 3.5 / 0.5))  # unseen, df = 0


def test_bm25_single_term_scores_hand_computed() -> None:
    scores = bm25().score_all("c")
    # chunk 1: tf=2, |D|=3 -> K = 1.2 * (0.25 + 0.75 * 3/3) = 1.2
    #   ln(1.6) * 2 * 2.2 / (2 + 1.2) = 0.4700036 * 1.375
    assert scores[1] == pytest.approx(math.log(1.6) * 4.4 / 3.2)
    assert scores[1] == pytest.approx(0.646255, abs=1e-6)
    # chunk 2: tf=1, |D|=4 -> K = 1.2 * (0.25 + 0.75 * 4/3) = 1.5
    assert scores[2] == pytest.approx(math.log(1.6) * 2.2 / 2.5)
    assert scores[2] == pytest.approx(0.413603, abs=1e-6)
    assert 0 not in scores  # chunk 0 has no "c"


def test_bm25_multi_term_sums_terms() -> None:
    scores = bm25().score_all("a d")
    # chunk 0: "a" tf=1, |D|=2 -> K = 1.2 * (0.25 + 0.5) = 0.9 -> ln(1.6) * 2.2 / 1.9
    assert scores[0] == pytest.approx(math.log(1.6) * 2.2 / 1.9)
    # chunk 1: "a" tf=1, |D|=3 -> K = 1.2 -> ln(1.6) * 2.2 / 2.2
    assert scores[1] == pytest.approx(math.log(1.6))
    # chunk 2: "d" tf=2, |D|=4 -> K = 1.5 -> ln(1 + 2.5/1.5) * 4.4 / 3.5
    assert scores[2] == pytest.approx(math.log(1 + 2.5 / 1.5) * 4.4 / 3.5)
    assert scores[2] == pytest.approx(1.233043, abs=1e-6)
    hits = bm25().search("a d", 3)
    assert [h.chunk.chunk_id for h in hits] == ["c2", "c0", "c1"]


def test_bm25_repeated_query_terms_count_once() -> None:
    assert bm25().score_all("c c c") == bm25().score_all("c")


def test_bm25_b_zero_disables_length_normalisation() -> None:
    r = BM25Retriever(b=0.0, tokenizer=split)
    r.index(chunks("x", "x y y y y y y y"))
    s = r.score_all("x")
    assert s[0] == pytest.approx(s[1])


def test_bm25_edge_cases() -> None:
    r = BM25Retriever(tokenizer=split)
    r.index([])
    assert r.search("anything", 5) == []
    r = bm25()
    assert r.search("", 5) == []
    assert r.search("nothing matches", 5) == []
    with pytest.raises(ValueError):
        r.search("a", 0)
    with pytest.raises(ValueError):
        BM25Retriever(k1=-1)
    with pytest.raises(ValueError):
        BM25Retriever(b=1.5)


def test_bm25_ties_break_by_index_order() -> None:
    r = BM25Retriever(tokenizer=split)
    r.index(chunks("q z", "q z", "q z"))
    assert [h.chunk.chunk_id for h in r.search("q", 3)] == ["c0", "c1", "c2"]


def test_bm25_reindex_replaces_index() -> None:
    r = bm25()
    r.index(chunks("only"))
    assert r.search("c", 5) == []
    assert [h.chunk.chunk_id for h in r.search("only", 5)] == ["c0"]


def test_tfidf_hand_computed_cosine() -> None:
    r = TfidfRetriever(tokenizer=split)
    r.index(chunks("a b", "a a c"))
    # N = 2. idf(a) = ln(3/3) + 1 = 1, idf(b) = idf(c) = ln(3/2) + 1
    ib = math.log(1.5) + 1
    # chunk 0: a -> 1*1, b -> 1*ib; chunk 1: a -> (1 + ln 2)*1, c -> 1*ib
    # query "b": unit vector on b, so cosine = weight of b in normalised chunk 0
    s = r.score_all("b")
    assert s == {0: pytest.approx(ib / math.sqrt(1 + ib * ib))}
    wa = 1 + math.log(2)
    s = r.score_all("a")
    assert s[0] == pytest.approx(1 / math.sqrt(1 + ib * ib))
    assert s[1] == pytest.approx(wa / math.sqrt(wa * wa + ib * ib))


def test_tfidf_identical_text_scores_one() -> None:
    r = TfidfRetriever(tokenizer=split)
    r.index(chunks("p q r", "s t"))
    assert r.score_all("p q r")[0] == pytest.approx(1.0)
    assert r.search("", 3) == []


def test_rrf_hand_computed_with_tie_breaks() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k_rrf=60)
    ids = [i for i, _ in fused]
    # a and b both score 1/61 + 1/62; a is seen first. c and d both 1/63; c is seen first.
    assert ids == ["a", "b", "c", "d"]
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 63)


def test_rrf_rewards_agreement() -> None:
    fused = dict(reciprocal_rank_fusion([["x", "y"], ["y", "z"]], k_rrf=1))
    # y: 1/(1+2) + 1/(1+1) = 0.8333 beats x: 1/2 and z: 1/3
    assert max(fused, key=fused.__getitem__) == "y"


def test_hybrid_fuses_components() -> None:
    h = HybridRRFRetriever([BM25Retriever(tokenizer=split), TfidfRetriever(tokenizer=split)])
    h.index(TOY)
    hits = h.search("c", 2)
    assert [x.chunk.chunk_id for x in hits] == ["c1", "c2"]
    assert h.name == "hybrid(bm25+tfidf)"
    with pytest.raises(ValueError):
        HybridRRFRetriever([])
    with pytest.raises(ValueError):
        h.search("c", 0)


class FakeEncoder:
    """Deterministic bag-of-letters embedding: counts of 'a', 'b', 'c'."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(t.count(ch)) for ch in "abc"] for t in texts]


def test_dense_ranks_by_cosine() -> None:
    enc = FakeEncoder()
    r = DenseRetriever(enc)
    r.index(chunks("aaa", "bbb", "aab"))
    hits = r.search("a", 3)
    assert [h.chunk.chunk_id for h in hits] == ["c0", "c2", "c1"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[2].score == pytest.approx(0.0)
    assert r.search("   ", 3) == []


def test_dense_rejects_encoder_returning_wrong_count() -> None:
    class Broken:
        def encode(self, texts: Sequence[str]) -> list[list[float]]:
            return []

    with pytest.raises(ValueError, match="different number"):
        DenseRetriever(Broken()).index(chunks("a"))


def test_dense_extra_missing_gives_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate the optional dependency being absent, whatever is installed locally.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(DenseUnavailableError, match="uv sync --extra dense"):
        SentenceTransformerEncoder()
    with pytest.raises(DenseUnavailableError):
        make_retriever("dense")


def test_make_retriever() -> None:
    assert make_retriever("bm25").name == "bm25"
    assert make_retriever("tfidf").name == "tfidf"
    assert make_retriever("hybrid").name == "hybrid(bm25+tfidf)"
    with pytest.raises(ValueError, match="unknown method"):
        make_retriever("splade")
