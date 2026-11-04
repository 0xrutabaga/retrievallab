"""End-to-end: the eval pipeline and CLI on the tiny fixture corpus, plus report reproducibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FIXTURES, REAL_CORPUS, REAL_QUERIES, ROOT

from retrievallab.cli import main
from retrievallab.evaluate import EvalConfig, ranked_passages, run_eval
from retrievallab.report import results_table, to_markdown, write_reports

FX = str(FIXTURES / "corpus")
FQ = str(FIXTURES / "queries.jsonl")
FIXTURE_CONFIG = EvalConfig(
    corpus_dir=FIXTURES / "corpus",
    queries_path=FIXTURES / "queries.jsonl",
    chunkers=("section", "fixed:60:20"),
    n_resamples=500,
)


def test_fixture_eval_expected_scores() -> None:
    result = run_eval(FIXTURE_CONFIG)
    assert [r.run_id for r in result.runs] == [
        "bm25/section",
        "tfidf/section",
        "hybrid/section",
        "bm25/fixed:60:20",
        "tfidf/fixed:60:20",
        "hybrid/fixed:60:20",
    ]
    assert result.n_documents == 2 and result.n_queries == 4 and result.n_labels == 6
    bm25 = result.runs[0]
    # f1-f3 share words with their answer, so BM25 ranks the grade-2 passage first.
    for qid in ("f1", "f2", "f3"):
        assert bm25.per_query[qid]["mrr@10"] == 1.0, qid
        assert bm25.per_query[qid]["ndcg@10"] > 0.8, qid
        assert bm25.top[qid][0].relevant_grade == 2
    # f4 asks for "quicker"; the answer says "faster". Lexical matching only finds the grade-1
    # passage through "pump" first: MRR counts it as a hit, nDCG does not give full credit.
    assert bm25.top["f4"][0].passage_id == "doc2#A"
    assert bm25.per_query["f4"]["mrr@10"] == 1.0
    assert bm25.per_query["f4"]["ndcg@10"] < 0.9
    top = bm25.top["f1"][0]
    assert (top.doc_id, top.section, top.section_title) == ("doc1", "1.1", "Watering Schedule")
    text = (FIXTURES / "corpus" / "doc1.txt").read_text(encoding="utf-8")
    assert text[top.start : top.end].startswith("1.1.  Watering Schedule")
    # Boilerplate (front matter, Authors' Addresses) is never indexed.
    for run in result.runs:
        for cites in run.top.values():
            assert all(c.section not in {"front", "authors"} for c in cites)
    assert len(result.comparisons) == 5 * 3


def test_reports_are_byte_identical_across_runs(tmp_path: Path) -> None:
    a = write_reports(run_eval(FIXTURE_CONFIG), tmp_path / "a")
    b = write_reports(run_eval(FIXTURE_CONFIG), tmp_path / "b")
    for pa, pb in zip(a, b, strict=True):
        assert pa.read_bytes() == pb.read_bytes()
        assert b"\r\n" not in pa.read_bytes()
    data = json.loads(a[1].read_text(encoding="utf-8"))
    assert data["data"]["queries"] == 4
    assert data["runs"][0]["per_query"]["f1"]["top"][0]["chunk_id"].startswith("doc1@")


def test_ranked_passages_deduplicates_in_order() -> None:
    from retrievallab.chunking import Chunk

    cs = [Chunk(str(i), "d", 0, 1, "", s, "") for i, s in enumerate(["2", "1", "2", "3", "1"])]
    assert ranked_passages(cs) == ["d#2", "d#1", "d#3"]


def test_cli_eval_writes_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "eval",
            "--corpus",
            str(FIXTURES / "corpus"),
            "--queries",
            str(FIXTURES / "queries.jsonl"),
            "--chunkers",
            "section",
            "--methods",
            "bm25,tfidf",
            "--resamples",
            "200",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "| bm25/section |" in out and "| tfidf/section |" in out
    assert (tmp_path / "eval.md").is_file() and (tmp_path / "eval.json").is_file()


def test_cli_search_prints_citations(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["search", "--corpus", str(FIXTURES / "corpus"), "chain oil", "-k", "2"])
    assert code == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == '1. doc2 section 1 "Bicycles"'
    assert lines[1].lstrip().startswith("chars ")
    assert "chunk doc2@" in lines[1]


def test_cli_verify(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verify", "--corpus", str(REAL_CORPUS), "--queries", str(REAL_QUERIES)]) == 0
    assert "corpus ok: 18 documents" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["eval", "--corpus", "does-not-exist"], "manifest not found"),
        (["eval", "--corpus", FX, "--queries", FQ, "--methods", "nope"], "unknown method"),
        (
            [
                "eval",
                "--corpus",
                str(FIXTURES / "corpus"),
                "--queries",
                str(FIXTURES / "queries.jsonl"),
                "--chunkers",
                "section",
                "--baseline",
                "bm25/fixed",
                "--resamples",
                "10",
            ],
            "baseline",
        ),
        (["eval", "--corpus", FX, "--queries", FQ, "--depth", "5"], "depth"),
        (["search", "--corpus", str(FIXTURES / "corpus"), "x", "-k", "0"], "-k must be positive"),
        (["search", "--corpus", str(FIXTURES / "corpus"), "x", "--chunker", "bad"], "chunker"),
    ],
)
def test_cli_errors_exit_2(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == 2
    assert message in capsys.readouterr().err


def test_cli_eval_bad_queries_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "q.jsonl"
    bad.write_text('{"qid": "x", "query": "q", "relevant": {"doc1#7": 2}}\n', encoding="utf-8")
    argv = ["eval", "--corpus", str(FIXTURES / "corpus"), "--queries", str(bad)]
    assert main(argv) == 2
    assert "unknown passage doc1#7" in capsys.readouterr().err


def test_committed_report_matches_regeneration(monkeypatch: pytest.MonkeyPatch) -> None:
    """The committed reports/eval.md is exactly what `retrievallab eval` produces today."""
    monkeypatch.chdir(ROOT)  # the report records the relative paths the default command uses
    result = run_eval(EvalConfig())
    committed = (ROOT / "reports" / "eval.md").read_text(encoding="utf-8")
    assert to_markdown(result) == committed
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert results_table(result) in readme, "README results table is stale; rerun the eval"
