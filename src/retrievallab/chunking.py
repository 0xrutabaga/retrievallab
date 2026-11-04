"""Chunkers that turn documents into retrievable units with stable ids and character offsets.

Invariants (tested in tests/test_chunking.py for both chunkers):

* every chunk satisfies ``doc.text[chunk.start:chunk.end] == chunk.text``;
* chunks are ordered by ``start``, the first starts at 0 and the last ends at ``len(doc.text)``;
* there are no gaps: each chunk starts at or before the previous chunk's end (``SectionChunker``
  chunks are exactly contiguous, ``FixedSizeChunker`` chunks overlap by up to ``overlap``);
* ids are ``"{doc_id}@{start}-{end}"``: unique within a document, identical across runs and
  machines, and readable as a citation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from retrievallab.corpus import Document
from retrievallab.sections import Section, section_at, split_sections


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    start: int
    end: int
    text: str
    section: str
    section_title: str

    @property
    def passage_id(self) -> str:
        """The labelled passage (RFC section) this chunk is credited to during evaluation."""
        return f"{self.doc_id}#{self.section}"


class Chunker(Protocol):
    name: str

    def chunk(self, doc: Document) -> list[Chunk]: ...


def _make_chunk(doc: Document, start: int, end: int, sec: Section) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}@{start}-{end}",
        doc_id=doc.doc_id,
        start=start,
        end=end,
        text=doc.text[start:end],
        section=sec.number,
        section_title=sec.title,
    )


def _window_end(text: str, start: int, limit: int, size: int) -> int:
    """End of a window starting at ``start``: ``start + size`` moved back to a word boundary.

    The window never ends past ``limit``. If no whitespace exists in the second half of the
    window, the cut is made mid-word rather than producing a tiny chunk.
    """
    end = min(start + size, limit)
    if end >= limit:
        return limit
    floor = start + size // 2
    for i in range(end - 1, floor - 1, -1):
        if text[i].isspace():
            return i + 1
    return end


def _fixed_spans(
    text: str, start: int, limit: int, size: int, overlap: int
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = start
    while pos < limit:
        end = _window_end(text, pos, limit, size)
        spans.append((pos, end))
        if end >= limit:
            break
        nxt = max(end - overlap, pos + 1)
        # Start the next window at a word start, never past the current end (no gaps).
        while nxt < end and not text[nxt - 1].isspace():
            nxt += 1
        pos = nxt
    return spans


class FixedSizeChunker:
    """Sliding windows of about ``size`` characters with ``overlap`` characters shared.

    Windows ignore document structure; each chunk is credited to the section it overlaps most.
    """

    def __init__(self, size: int = 1000, overlap: int = 200) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if not 0 <= overlap < size:
            raise ValueError("overlap must satisfy 0 <= overlap < size")
        self.size = size
        self.overlap = overlap
        self.name = f"fixed-{size}-{overlap}"

    def chunk(self, doc: Document) -> list[Chunk]:
        sections = split_sections(doc)
        return [
            _make_chunk(doc, s, e, section_at(sections, s, e))
            for s, e in _fixed_spans(doc.text, 0, len(doc.text), self.size, self.overlap)
        ]


def _paragraph_starts(text: str, start: int, end: int) -> list[int]:
    """Offsets in ``[start, end)`` where a paragraph begins (first line after a blank line)."""
    starts = [start]
    pos = start
    prev_blank = False
    for line in text[start:end].splitlines(keepends=True):
        blank = not line.strip()
        if prev_blank and not blank and pos != start:
            starts.append(pos)
        prev_blank = blank
        pos += len(line)
    return starts


class SectionChunker:
    """Heading- and paragraph-aware chunks that never cross a section boundary.

    Paragraphs (blocks separated by blank lines) are packed greedily into chunks of at most
    ``max_chars``; a single paragraph longer than that is split into word-aligned windows.
    """

    def __init__(self, max_chars: int = 1500) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.max_chars = max_chars
        self.name = f"section-{max_chars}"

    def chunk(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for sec in split_sections(doc):
            for s, e in self._section_spans(doc.text, sec):
                chunks.append(_make_chunk(doc, s, e, sec))
        return chunks

    def _section_spans(self, text: str, sec: Section) -> Iterable[tuple[int, int]]:
        if sec.start == sec.end:
            return []
        bounds = [*_paragraph_starts(text, sec.start, sec.end), sec.end]
        spans: list[tuple[int, int]] = []
        cur = bounds[0]
        for i in range(1, len(bounds)):
            para_start, para_end = bounds[i - 1], bounds[i]
            if para_end - cur <= self.max_chars:
                continue  # paragraph fits in the open chunk
            if para_start > cur:
                spans.append((cur, para_start))  # close the open chunk before this paragraph
                cur = para_start
            if para_end - cur > self.max_chars:
                pieces = _fixed_spans(text, cur, para_end, self.max_chars, 0)
                spans.extend(pieces[:-1])
                cur = pieces[-1][0]
        spans.append((cur, sec.end))
        return spans


def chunk_corpus(docs: Iterable[Document], chunker: Chunker) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunker.chunk(doc))
    return out


def make_chunker(spec: str) -> Chunker:
    """Build a chunker from ``fixed``, ``fixed:SIZE:OVERLAP``, ``section`` or ``section:MAX``."""
    name, *args = spec.split(":")
    try:
        nums = [int(a) for a in args]
    except ValueError as exc:
        raise ValueError(f"chunker arguments must be integers: {spec!r}") from exc
    if name == "fixed" and len(nums) in (0, 2):
        return FixedSizeChunker(*nums)
    if name == "section" and len(nums) in (0, 1):
        return SectionChunker(*nums)
    raise ValueError(f"unknown chunker spec {spec!r}; use fixed[:SIZE:OVERLAP] or section[:MAX]")
