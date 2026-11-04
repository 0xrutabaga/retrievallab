"""Split RFC plain text into sections.

RFC text puts section headings at column 0 and indents body text by three spaces, so a heading
is a column-0 line that is either a numbered heading (``4.1.2.  Access Token Response``), an
appendix heading (``Appendix A.  ...`` or ``A.1.  ...``) or one of a fixed set of unnumbered
headings (``Abstract``, ``Table of Contents``, ...). Page headers and footers also sit at
column 0 but match none of these forms.

Sections partition the document: every character belongs to exactly one section. Text before
the first heading is the ``front`` section. A section runs from the start of its heading line to
the start of the next heading line, so a parent section (``4.1``) holds only the text before its
first child (``4.1.1``).

Each section is a *passage*. Evaluation labels name passages (``rfc6749#4.1.2``), which keeps
labels independent of how documents are chunked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from retrievallab.corpus import Document

_NUMBERED = re.compile(r"^(?P<num>\d+(?:\.\d+)*)\.\s+(?P<title>\S.*?)\s*$")
_APPENDIX = re.compile(r"^Appendix\s+(?P<num>[A-Z](?:\.\d+)*)\.\s+(?P<title>\S.*?)\s*$")
_APPENDIX_SUB = re.compile(r"^(?P<num>[A-Z](?:\.\d+)+)\.\s+(?P<title>\S.*?)\s*$")

_UNNUMBERED: dict[str, str] = {
    "Abstract": "abstract",
    "Status of This Memo": "status",
    "Copyright Notice": "copyright",
    "Table of Contents": "toc",
    "Acknowledgements": "acknowledgements",
    "Acknowledgments": "acknowledgements",
    "Contributors": "contributors",
    "Authors' Addresses": "authors",
    "Author's Address": "authors",
    "Index": "index",
}

# Sections that hold no protocol content. They are excluded from the evaluation index by
# default (see evaluate.build_index) because their keyword-dense lists (a table of contents,
# reference titles) match many queries without answering any of them.
_BOILERPLATE_IDS = frozenset({"front", "status", "copyright", "toc", "acknowledgements", "authors"})
_BOILERPLATE_TITLE = re.compile(r"(?i)^(?:normative |informative )?references$|acknowledge?ments$")


@dataclass(frozen=True, slots=True)
class Section:
    doc_id: str
    number: str
    title: str
    start: int
    end: int
    boilerplate: bool

    @property
    def passage_id(self) -> str:
        return f"{self.doc_id}#{self.number}"


def _match_heading(line: str) -> tuple[str, str] | None:
    if not line or line[0].isspace():
        return None
    stripped = line.rstrip()
    if stripped in _UNNUMBERED:
        return _UNNUMBERED[stripped], stripped
    for pattern in (_NUMBERED, _APPENDIX, _APPENDIX_SUB):
        m = pattern.match(stripped)
        if m:
            return m.group("num"), m.group("title")
    return None


def split_sections(doc: Document) -> list[Section]:
    """Return the sections of ``doc`` in order. Offsets are contiguous and cover the text."""
    text = doc.text
    headings: list[tuple[int, str, str]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        found = _match_heading(line)
        if found is not None:
            headings.append((pos, found[0], found[1]))
        pos += len(line)

    sections: list[Section] = []
    first_start = headings[0][0] if headings else len(text)
    if first_start > 0 or not headings:
        # With no headings at all the whole text is one content section.
        sections.append(Section(doc.doc_id, "front", "", 0, first_start, bool(headings)))
    seen: set[str] = set()
    for i, (start, number, title) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        if number in seen:
            # A repeated number would make passage ids ambiguous; keep both but suffix the
            # second occurrence so ids stay unique and deterministic.
            k = 2
            while f"{number}~{k}" in seen:
                k += 1
            number = f"{number}~{k}"
        seen.add(number)
        boilerplate = number in _BOILERPLATE_IDS or bool(_BOILERPLATE_TITLE.search(title))
        sections.append(Section(doc.doc_id, number, title, start, end, boilerplate))
    return sections


def section_at(sections: list[Section], start: int, end: int) -> Section:
    """Return the section with the largest overlap with ``[start, end)``; ties go to the earliest.

    ``sections`` must be the ordered output of :func:`split_sections` for one document.
    """
    best: Section | None = None
    best_overlap = -1
    for sec in sections:
        if sec.end <= start:
            continue
        if sec.start >= end:
            break
        overlap = min(sec.end, end) - max(sec.start, start)
        if overlap > best_overlap:
            best, best_overlap = sec, overlap
    if best is None:
        raise ValueError(f"span [{start}, {end}) lies outside every section")
    return best
