"""Query file parsing and label validation, including the real labelled set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FIXTURES, REAL_QUERIES

from retrievallab.corpus import Document
from retrievallab.qrels import (
    QrelsError,
    Query,
    load_queries,
    parse_queries,
    passage_index,
    validate_queries,
)


def line(**obj: object) -> str:
    base: dict[str, object] = {"qid": "q1", "query": "text", "relevant": {"d#1": 2}}
    base.update(obj)
    return json.dumps(base)


def test_real_query_set_is_valid(real_docs: list[Document]) -> None:
    queries = load_queries(REAL_QUERIES)
    assert len(queries) >= 50
    validate_queries(queries, passage_index(real_docs))
    assert all(1 <= len(q.relevant) <= 6 for q in queries)
    assert all(2 in q.relevant.values() for q in queries), "each query has a direct answer"
    assert len({q.text.lower() for q in queries}) == len(queries)


def test_fixture_queries_are_valid(fixture_docs: list[Document]) -> None:
    validate_queries(load_queries(FIXTURES / "queries.jsonl"), passage_index(fixture_docs))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{bad json", "invalid JSON"),
        ("[1, 2]", "expected a JSON object"),
        (line(qid=""), "'qid' must be"),
        (line(query="  "), "'query' must be"),
        (line(relevant={}), "'relevant' must be"),
        (line(relevant={"d#1": 3}), "must be 1 or 2"),
        (line(relevant={"d#1": True}), "must be 1 or 2"),
        (line(relevant={"no-hash": 1}), "doc#section"),
        (line(evidence={"d#1": 5}), "'evidence' must map"),
        (line(evidence={"d#9": "x"}), "unlabelled passages"),
    ],
)
def test_parse_errors_name_the_line(text: str, message: str) -> None:
    with pytest.raises(QrelsError, match=message) as info:
        parse_queries(["", text], "f.jsonl")
    assert "f.jsonl:2" in str(info.value)


def test_duplicate_qid_and_empty_file() -> None:
    with pytest.raises(QrelsError, match="duplicate qid"):
        parse_queries([line(), line()])
    with pytest.raises(QrelsError, match="no queries"):
        parse_queries(["", "  "])


def test_missing_queries_file(tmp_path: Path) -> None:
    with pytest.raises(QrelsError, match="not found"):
        load_queries(tmp_path / "nope.jsonl")


def test_validation_catches_bad_labels(fixture_docs: list[Document]) -> None:
    passages = passage_index(fixture_docs)
    cases = [
        (Query("a", "t", {"doc1#9.9": 2}, {"doc1#9.9": "x"}), "unknown passage doc1#9.9"),
        (Query("b", "t", {"doc1#1.1": 2}, {"doc1#1.1": "cactus daily"}), "evidence for doc1#1.1"),
        (Query("c", "t", {"doc1#1.1": 2}, {}), "no evidence quote"),
        (Query("d", "t", {"doc1#authors": 2}, {"doc1#authors": "Example"}), "boilerplate"),
    ]
    for query, message in cases:
        with pytest.raises(QrelsError, match=message):
            validate_queries([query], passages)
    # Evidence matching ignores case and whitespace layout.
    ok = Query("e", "t", {"doc1#1.1": 2}, {"doc1#1.1": "WATER   tomatoes\n every morning"})
    validate_queries([ok], passages)
    validate_queries([Query("f", "t", {"doc1#1.1": 2})], passages, require_evidence=False)
