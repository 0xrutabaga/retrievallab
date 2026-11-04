from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from retrievallab.corpus import Document, load_corpus

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
REAL_CORPUS = ROOT / "data" / "corpus"
REAL_QUERIES = ROOT / "data" / "queries.jsonl"


@pytest.fixture(scope="session")
def real_docs() -> list[Document]:
    return load_corpus(REAL_CORPUS)


@pytest.fixture(scope="session")
def fixture_docs() -> list[Document]:
    return load_corpus(FIXTURES / "corpus")


@pytest.fixture
def fixture_corpus_copy(tmp_path: Path) -> Path:
    """A writable copy of the fixture corpus for tests that corrupt it."""
    dest = tmp_path / "corpus"
    shutil.copytree(FIXTURES / "corpus", dest)
    return dest


def make_doc(text: str, doc_id: str = "d") -> Document:
    return Document(doc_id=doc_id, title="t", text=text, source_url="")
