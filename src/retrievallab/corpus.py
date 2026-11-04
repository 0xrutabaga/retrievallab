"""Corpus loading and integrity checks.

A corpus is a directory holding plain-text documents and a ``manifest.json`` that pins each
file's SHA-256. Loading refuses any file whose bytes do not match, so every evaluation runs on
exactly the text the labels were written against.

Document text is the file decoded as UTF-8 with a leading byte-order mark removed. All
character offsets in this package index into that string.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"


class CorpusError(Exception):
    """Raised when a corpus is missing, malformed or fails its checksum."""


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    doc_id: str
    file: str
    sha256: str
    title: str
    url: str


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    title: str
    text: str
    source_url: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_document(data: bytes) -> str:
    """Decode corpus bytes into the canonical document text (UTF-8, BOM stripped)."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CorpusError(f"document is not valid UTF-8: {exc}") from exc


def read_manifest(corpus_dir: Path) -> list[ManifestEntry]:
    path = corpus_dir / MANIFEST_NAME
    if not path.is_file():
        raise CorpusError(f"manifest not found: {path}")
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusError(f"manifest is not valid JSON: {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("documents"), list):
        raise CorpusError(f"manifest must be an object with a 'documents' list: {path}")
    entries: list[ManifestEntry] = []
    seen: set[str] = set()
    for i, item in enumerate(raw["documents"]):
        if not isinstance(item, dict):
            raise CorpusError(f"manifest entry {i} is not an object")
        try:
            entry = ManifestEntry(
                doc_id=str(item["doc_id"]),
                file=str(item["file"]),
                sha256=str(item["sha256"]).lower(),
                title=str(item["title"]),
                url=str(item.get("url", "")),
            )
        except KeyError as exc:
            raise CorpusError(f"manifest entry {i} is missing field {exc}") from exc
        if entry.doc_id in seen:
            raise CorpusError(f"duplicate doc_id in manifest: {entry.doc_id}")
        if Path(entry.file).name != entry.file:
            raise CorpusError(f"manifest file must be a bare file name: {entry.file!r}")
        seen.add(entry.doc_id)
        entries.append(entry)
    if not entries:
        raise CorpusError(f"manifest lists no documents: {path}")
    return entries


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Load every manifest document, verifying its SHA-256. Order follows the manifest."""
    docs: list[Document] = []
    for entry in read_manifest(corpus_dir):
        path = corpus_dir / entry.file
        if not path.is_file():
            raise CorpusError(
                f"{entry.doc_id}: file {path} is missing; run `retrievallab fetch-corpus`"
            )
        data = path.read_bytes()
        actual = sha256_bytes(data)
        if actual != entry.sha256:
            raise CorpusError(
                f"{entry.doc_id}: checksum mismatch for {path} "
                f"(expected {entry.sha256}, got {actual})"
            )
        docs.append(Document(entry.doc_id, entry.title, decode_document(data), entry.url))
    return docs


def fetch_corpus(corpus_dir: Path, *, timeout: float = 30.0, force: bool = False) -> list[str]:
    """Download missing manifest documents from their URLs and verify their checksums.

    A download is written only after its checksum matches, so a failed or tampered fetch never
    leaves a corrupt file behind. Returns the doc ids that were downloaded.
    """
    fetched: list[str] = []
    for entry in read_manifest(corpus_dir):
        path = corpus_dir / entry.file
        if path.is_file() and not force and sha256_bytes(path.read_bytes()) == entry.sha256:
            continue
        if not entry.url.startswith("https://"):
            raise CorpusError(f"{entry.doc_id}: refusing non-https URL {entry.url!r}")
        try:
            # URL scheme is restricted to https above.
            with urllib.request.urlopen(entry.url, timeout=timeout) as resp:  # noqa: S310
                data: bytes = resp.read()
        except OSError as exc:
            raise CorpusError(f"{entry.doc_id}: download failed from {entry.url}: {exc}") from exc
        actual = sha256_bytes(data)
        if actual != entry.sha256:
            raise CorpusError(
                f"{entry.doc_id}: downloaded bytes do not match pinned checksum "
                f"(expected {entry.sha256}, got {actual}); file not written"
            )
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        fetched.append(entry.doc_id)
    return fetched
