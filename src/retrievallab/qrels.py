"""Labelled queries and their validation against the corpus.

``data/queries.jsonl`` holds one JSON object per line::

    {"qid": "q001",
     "query": "How long may an authorization code stay valid?",
     "relevant": {"rfc6749#4.1.2": 2, "rfc6749#10.5": 1},
     "evidence": {"rfc6749#4.1.2": "maximum authorization code lifetime of 10 minutes",
                  "rfc6749#10.5": "short lived and single-use"}}

``relevant`` maps passage ids (see sections.py) to grades: 2 = the passage answers the query,
1 = the passage is on topic and partially answers it. ``evidence`` quotes, for every labelled
passage, a phrase that must occur in that passage's text (compared case-insensitively with
whitespace collapsed). The quote is what makes a label checkable: :func:`validate_queries`
fails if a label points at a passage that does not contain its evidence, which catches typos in
passage ids and labels that drift if the corpus or the section parser changes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from retrievallab.corpus import Document
from retrievallab.sections import Section, split_sections

_WS = re.compile(r"\s+")


class QrelsError(Exception):
    """Raised for malformed or inconsistent labelled queries."""


@dataclass(frozen=True, slots=True)
class Query:
    qid: str
    text: str
    relevant: dict[str, int]
    evidence: dict[str, str] = field(default_factory=dict)


def _normalise(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def _parse(obj: Any, where: str) -> Query:
    if not isinstance(obj, dict):
        raise QrelsError(f"{where}: expected a JSON object")
    qid, text, relevant = obj.get("qid"), obj.get("query"), obj.get("relevant")
    evidence = obj.get("evidence", {})
    if not isinstance(qid, str) or not qid:
        raise QrelsError(f"{where}: 'qid' must be a non-empty string")
    if not isinstance(text, str) or not text.strip():
        raise QrelsError(f"{where} ({qid}): 'query' must be a non-empty string")
    if not isinstance(relevant, dict) or not relevant:
        raise QrelsError(f"{where} ({qid}): 'relevant' must be a non-empty object")
    grades: dict[str, int] = {}
    for pid, grade in relevant.items():
        if not isinstance(grade, int) or isinstance(grade, bool) or grade not in (1, 2):
            raise QrelsError(f"{where} ({qid}): grade for {pid!r} must be 1 or 2")
        if "#" not in pid:
            raise QrelsError(f"{where} ({qid}): passage id {pid!r} must look like 'doc#section'")
        grades[pid] = grade
    if not isinstance(evidence, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in evidence.items()
    ):
        raise QrelsError(f"{where} ({qid}): 'evidence' must map passage ids to strings")
    extra = set(evidence) - set(grades)
    if extra:
        raise QrelsError(f"{where} ({qid}): evidence for unlabelled passages {sorted(extra)}")
    return Query(qid, text, grades, dict(evidence))


def parse_queries(lines: Iterable[str], source: str = "<queries>") -> list[Query]:
    queries: list[Query] = []
    seen: set[str] = set()
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"{source}:{lineno}"
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise QrelsError(f"{where}: invalid JSON: {exc.msg}") from exc
        q = _parse(obj, where)
        if q.qid in seen:
            raise QrelsError(f"{where}: duplicate qid {q.qid!r}")
        seen.add(q.qid)
        queries.append(q)
    if not queries:
        raise QrelsError(f"{source}: no queries")
    return queries


def load_queries(path: Path) -> list[Query]:
    if not path.is_file():
        raise QrelsError(f"queries file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return parse_queries(fh, str(path))


def passage_index(docs: Iterable[Document]) -> dict[str, tuple[Section, str]]:
    """Map every passage id to its section and text."""
    out: dict[str, tuple[Section, str]] = {}
    for doc in docs:
        for sec in split_sections(doc):
            out[sec.passage_id] = (sec, doc.text[sec.start : sec.end])
    return out


def validate_queries(
    queries: Iterable[Query],
    passages: Mapping[str, tuple[Section, str]],
    *,
    require_evidence: bool = True,
) -> None:
    """Check every label names a real, non-boilerplate passage that contains its evidence."""
    problems: list[str] = []
    for q in queries:
        for pid in q.relevant:
            if pid not in passages:
                problems.append(f"{q.qid}: unknown passage {pid}")
                continue
            sec, text = passages[pid]
            if sec.boilerplate:
                problems.append(f"{q.qid}: {pid} is boilerplate and is never indexed")
            quote = q.evidence.get(pid)
            if quote is None:
                if require_evidence:
                    problems.append(f"{q.qid}: no evidence quote for {pid}")
            elif _normalise(quote) not in _normalise(text):
                problems.append(f"{q.qid}: evidence for {pid} not found: {quote!r}")
    if problems:
        raise QrelsError("invalid labels:\n  " + "\n  ".join(problems))
