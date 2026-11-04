"""Corpus loading, checksum enforcement and the fetch path (network mocked)."""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

import pytest
from conftest import REAL_CORPUS

from retrievallab import corpus as corpus_mod
from retrievallab.corpus import (
    CorpusError,
    decode_document,
    fetch_corpus,
    load_corpus,
    read_manifest,
)


def test_real_corpus_loads_and_matches_manifest(real_docs: list[corpus_mod.Document]) -> None:
    assert len(real_docs) == 18
    assert len({d.doc_id for d in real_docs}) == 18
    for doc in real_docs:
        assert not doc.text.startswith("﻿"), "BOM is stripped"
        assert f"Request for Comments: {doc.doc_id.removeprefix('rfc')}" in doc.text
        assert "IETF Trust" in doc.text, "each file keeps its copyright notice"
        # The manifest title appears in the document's front matter (line wraps ignored).
        head = re.sub(r"\s+", " ", doc.text[:3000])
        assert doc.title in head


def test_manifest_urls_are_rfc_editor_https() -> None:
    for entry in read_manifest(REAL_CORPUS):
        assert entry.url == f"https://www.rfc-editor.org/rfc/{entry.doc_id}.txt"


def test_checksum_mismatch_is_rejected(fixture_corpus_copy: Path) -> None:
    path = fixture_corpus_copy / "doc1.txt"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(CorpusError, match="checksum mismatch"):
        load_corpus(fixture_corpus_copy)


def test_missing_file_is_reported(fixture_corpus_copy: Path) -> None:
    (fixture_corpus_copy / "doc2.txt").unlink()
    with pytest.raises(CorpusError, match="missing; run `retrievallab fetch-corpus`"):
        load_corpus(fixture_corpus_copy)


def _write_manifest(dirpath: Path, payload: Any) -> None:
    (dirpath / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "'documents' list"),
        ({"documents": []}, "lists no documents"),
        ({"documents": ["x"]}, "is not an object"),
        ({"documents": [{"doc_id": "a"}]}, "missing field"),
        (
            {"documents": [{"doc_id": "a", "file": "../a.txt", "sha256": "0", "title": "t"}]},
            "bare file name",
        ),
        (
            {
                "documents": [
                    {"doc_id": "a", "file": "a.txt", "sha256": "0", "title": "t"},
                    {"doc_id": "a", "file": "b.txt", "sha256": "0", "title": "t"},
                ]
            },
            "duplicate doc_id",
        ),
    ],
)
def test_malformed_manifest(tmp_path: Path, payload: Any, message: str) -> None:
    _write_manifest(tmp_path, payload)
    with pytest.raises(CorpusError, match=message):
        read_manifest(tmp_path)


def test_manifest_not_json_or_absent(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="manifest not found"):
        read_manifest(tmp_path)
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(CorpusError, match="not valid JSON"):
        read_manifest(tmp_path)


def test_decode_strips_bom_and_rejects_invalid_utf8() -> None:
    assert decode_document("﻿hello".encode()) == "hello"
    with pytest.raises(CorpusError, match="not valid UTF-8"):
        decode_document(b"\xff\xfe\xfa")


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _fetch_dir(tmp_path: Path, payload: bytes, url: str = "https://example.invalid/a.txt") -> Path:
    _write_manifest(
        tmp_path,
        {
            "documents": [
                {
                    "doc_id": "a",
                    "file": "a.txt",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "title": "t",
                    "url": url,
                }
            ]
        },
    )
    return tmp_path


def test_fetch_downloads_and_verifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"1.  Intro\n   text\n"
    d = _fetch_dir(tmp_path, payload)
    calls: list[str] = []

    def fake_urlopen(url: str, timeout: float) -> _FakeResponse:
        calls.append(url)
        return _FakeResponse(payload)

    monkeypatch.setattr(corpus_mod.urllib.request, "urlopen", fake_urlopen)
    assert fetch_corpus(d) == ["a"]
    assert (d / "a.txt").read_bytes() == payload
    assert fetch_corpus(d) == [], "verified files are not downloaded again"
    assert len(calls) == 1
    assert load_corpus(d)[0].text == payload.decode()


def test_fetch_with_wrong_checksum_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = _fetch_dir(tmp_path, b"expected bytes")
    monkeypatch.setattr(
        corpus_mod.urllib.request, "urlopen", lambda url, timeout: _FakeResponse(b"evil bytes")
    )
    with pytest.raises(CorpusError, match="do not match pinned checksum"):
        fetch_corpus(d)
    assert not (d / "a.txt").exists()
    assert not (d / "a.txt.part").exists()


def test_fetch_network_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _fetch_dir(tmp_path, b"x")

    def down(url: str, timeout: float) -> _FakeResponse:
        raise TimeoutError("timed out")

    monkeypatch.setattr(corpus_mod.urllib.request, "urlopen", down)
    with pytest.raises(CorpusError, match=r"download failed.*timed out"):
        fetch_corpus(d)


def test_fetch_refuses_non_https(tmp_path: Path) -> None:
    d = _fetch_dir(tmp_path, b"x", url="http://example.invalid/a.txt")
    with pytest.raises(CorpusError, match="non-https"):
        fetch_corpus(d)
