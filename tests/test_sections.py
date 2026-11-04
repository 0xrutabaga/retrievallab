"""RFC section parsing, checked against each RFC's own table of contents."""

from __future__ import annotations

import itertools
import re

import pytest
from conftest import make_doc

from retrievallab.corpus import Document
from retrievallab.sections import section_at, split_sections

_TOC_ENTRY = re.compile(r"^\s{2,}(?:Appendix\s+)?(\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\.\s+\S")


def toc_numbers(doc: Document) -> list[str]:
    """Section numbers listed in the document's 'Table of Contents' section."""
    toc = next(s for s in split_sections(doc) if s.number == "toc")
    numbers = []
    for line in doc.text[toc.start : toc.end].splitlines():
        m = _TOC_ENTRY.match(line)
        if m:
            numbers.append(m.group(1))
    return numbers


def test_sections_partition_every_document(real_docs: list[Document]) -> None:
    for doc in real_docs:
        sections = split_sections(doc)
        assert sections[0].start == 0
        assert sections[-1].end == len(doc.text)
        for prev, nxt in itertools.pairwise(sections):
            assert prev.end == nxt.start
        assert len({s.passage_id for s in sections}) == len(sections)


def test_every_toc_entry_is_parsed_as_a_section(real_docs: list[Document]) -> None:
    for doc in real_docs:
        parsed = {s.number for s in split_sections(doc)}
        listed = toc_numbers(doc)
        assert len(listed) >= 5, doc.doc_id
        missing = [n for n in listed if n not in parsed]
        assert not missing, f"{doc.doc_id}: TOC entries not parsed: {missing}"


def test_known_sections_of_rfc6749(real_docs: list[Document]) -> None:
    doc = next(d for d in real_docs if d.doc_id == "rfc6749")
    by_number = {s.number: s for s in split_sections(doc)}
    assert by_number["4.1.2"].title == "Authorization Response"
    assert by_number["10.12"].title == "Cross-Site Request Forgery"
    assert by_number["A.1"].title == '"client_id" Syntax'
    assert by_number["B"].title.startswith("Use of application/x-www-form-urlencoded")
    assert by_number["toc"].boilerplate
    assert by_number["front"].boilerplate
    assert by_number["12"].boilerplate  # "References"
    assert not by_number["abstract"].boilerplate
    assert not by_number["4.1.2"].boilerplate


def test_page_headers_and_body_lines_are_not_headings() -> None:
    text = (
        "Abstract\n\n   body\n\n"
        "1.  Real Heading\n\n   1.  indented list item is not a heading\n"
        "Hardt                        Standards Track                    [Page 4]\n"
        "\f\nRFC 6749                        OAuth 2.0                   October 2012\n"
        "A. Smith wrote this line at column zero\n"
        "Appendix B.  Extra\n\nB.1.  Sub\n"
    )
    numbers = [s.number for s in split_sections(make_doc(text))]
    assert numbers == ["abstract", "1", "B", "B.1"]


def test_document_without_headings_is_one_content_section() -> None:
    sections = split_sections(make_doc("just some text\nwith no headings\n"))
    assert [(s.number, s.boilerplate) for s in sections] == [("front", False)]


def test_duplicate_numbers_get_unique_ids() -> None:
    text = "1.  One\n   a\n1.  One again\n   b\n"
    assert [s.number for s in split_sections(make_doc(text))] == ["1", "1~2"]


def test_section_at_picks_largest_overlap() -> None:
    text = "1.  A\n" + "x" * 20 + "\n2.  B\n" + "y" * 5 + "\n"
    sections = split_sections(make_doc(text))
    s1, s2 = sections
    assert section_at(sections, 0, len(text)).number == "1"
    assert section_at(sections, s2.start - 2, s2.end).number == "2"
    # Equal overlap: the earlier section wins.
    assert section_at(sections, s1.end - 3, s1.end + 3).number == "1"
    with pytest.raises(ValueError):
        section_at(sections, len(text) + 1, len(text) + 2)
