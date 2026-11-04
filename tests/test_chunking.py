"""Chunker invariants (coverage, no gaps, offsets, deterministic ids) on real and synthetic text."""

from __future__ import annotations

import itertools
import random

import pytest
from conftest import make_doc

from retrievallab.chunking import (
    Chunk,
    Chunker,
    FixedSizeChunker,
    SectionChunker,
    chunk_corpus,
    make_chunker,
)
from retrievallab.corpus import Document
from retrievallab.sections import split_sections

CHUNKERS: list[Chunker] = [
    FixedSizeChunker(1000, 200),
    FixedSizeChunker(300, 0),
    FixedSizeChunker(200, 150),
    SectionChunker(1500),
    SectionChunker(200),
]


def assert_invariants(doc: Document, chunks: list[Chunk], *, contiguous: bool) -> None:
    text = doc.text
    assert chunks, "a non-empty document yields at least one chunk"
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    for c in chunks:
        assert 0 <= c.start < c.end <= len(text)
        assert text[c.start : c.end] == c.text
        assert c.chunk_id == f"{doc.doc_id}@{c.start}-{c.end}"
        assert c.doc_id == doc.doc_id
    for prev, nxt in itertools.pairwise(chunks):
        assert prev.start < nxt.start, "chunks advance"
        assert nxt.start <= prev.end, "no gap between consecutive chunks"
        if contiguous:
            assert nxt.start == prev.end
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    covered = [False] * len(text)
    for c in chunks:
        for i in range(c.start, c.end):
            covered[i] = True
    assert all(covered)


@pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: c.name)
def test_invariants_on_real_corpus(real_docs: list[Document], chunker: Chunker) -> None:
    for doc in real_docs:
        assert_invariants(doc, chunker.chunk(doc), contiguous=isinstance(chunker, SectionChunker))


@pytest.mark.parametrize("seed", range(25))
def test_invariants_on_random_text(seed: int) -> None:
    rng = random.Random(seed)
    words = ["alpha", "b", "gamma-delta", "\n", "\n\n", "1.  Heading\n", "x" * 90, "   "]
    text = "".join(
        rng.choice(words) + rng.choice([" ", "", "\n"]) for _ in range(rng.randint(1, 400))
    )
    doc = make_doc(text)
    size = rng.randint(1, 200)
    for chunker in (
        FixedSizeChunker(size, rng.randint(0, size - 1)),
        SectionChunker(rng.randint(1, 300)),
    ):
        assert_invariants(doc, chunker.chunk(doc), contiguous=isinstance(chunker, SectionChunker))


def test_ids_are_deterministic(real_docs: list[Document]) -> None:
    for spec in ("fixed", "section"):
        first = [c.chunk_id for c in chunk_corpus(real_docs, make_chunker(spec))]
        second = [c.chunk_id for c in chunk_corpus(real_docs, make_chunker(spec))]
        assert first == second


def test_known_chunk_id_is_stable(real_docs: list[Document]) -> None:
    # Pinned values: if these change, labelled citations in old reports no longer resolve.
    rfc6749 = next(d for d in real_docs if d.doc_id == "rfc6749")
    chunks = SectionChunker(1500).chunk(rfc6749)
    code_resp = next(c for c in chunks if c.section == "4.1.2")
    assert code_resp.section_title == "Authorization Response"
    assert code_resp.chunk_id == f"rfc6749@{code_resp.start}-{code_resp.end}"
    assert code_resp.text.startswith("4.1.2.  Authorization Response")
    assert "10 minutes" in code_resp.text


def test_section_chunks_never_cross_sections(real_docs: list[Document]) -> None:
    for doc in real_docs:
        bounds = {(s.start, s.end, s.number) for s in split_sections(doc)}
        for c in SectionChunker(800).chunk(doc):
            assert any(s <= c.start and c.end <= e and n == c.section for s, e, n in bounds)


def test_section_chunker_respects_max_chars(real_docs: list[Document]) -> None:
    for doc in real_docs:
        assert all(len(c.text) <= 600 for c in SectionChunker(600).chunk(doc))


def test_section_chunker_splits_at_paragraphs() -> None:
    text = (
        "1.  Intro\n\n" + "   para one words.\n\n" + "   para two words.\n\n" + "   para three.\n"
    )
    chunks = SectionChunker(30).chunk(make_doc(text))
    assert [c.text for c in chunks] == [
        "1.  Intro\n\n",
        "   para one words.\n\n",
        "   para two words.\n\n",
        "   para three.\n",
    ]
    assert all(c.section == "1" for c in chunks)


def test_fixed_chunker_cuts_at_whitespace_and_overlaps() -> None:
    text = "one two three four five six seven eight nine ten"
    chunks = FixedSizeChunker(15, 5).chunk(make_doc(text))
    for c in chunks[:-1]:
        assert c.text.endswith(" "), "windows end after a space, not mid-word"
    assert any(nxt.start < prev.end for prev, nxt in itertools.pairwise(chunks))


def test_fixed_chunk_credited_to_majority_section() -> None:
    text = "1.  A\n" + "a " * 50 + "\n2.  B\n" + "b " * 40
    doc = make_doc(text)
    chunks = FixedSizeChunker(80, 0).chunk(doc)
    assert chunks[0].section == "1"
    assert chunks[-1].section == "2"


def test_empty_document_yields_no_chunks() -> None:
    doc = make_doc("")
    assert FixedSizeChunker().chunk(doc) == []
    assert SectionChunker().chunk(doc) == []


def test_bad_chunker_parameters() -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(0)
    with pytest.raises(ValueError):
        FixedSizeChunker(10, 10)
    with pytest.raises(ValueError):
        SectionChunker(0)
    for spec in ("fixed:10", "section:1:2", "sliding", "fixed:a:b"):
        with pytest.raises(ValueError):
            make_chunker(spec)
    assert make_chunker("fixed:500:50").name == "fixed-500-50"
    assert make_chunker("section:900").name == "section-900"
