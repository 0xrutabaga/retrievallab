"""Command-line interface: ``retrievallab {eval,search,verify,fetch-corpus}``."""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

from retrievallab.corpus import CorpusError, fetch_corpus, load_corpus
from retrievallab.evaluate import EvalConfig, build_index, run_eval
from retrievallab.qrels import QrelsError, load_queries, passage_index, validate_queries
from retrievallab.report import results_table, write_reports
from retrievallab.retrievers import DEFAULT_METHODS, METHODS
from retrievallab.retrievers.dense import DenseUnavailableError


def _csv(value: str) -> tuple[str, ...]:
    items = tuple(v.strip() for v in value.split(",") if v.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return items


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retrievallab", description="Reproducible retrieval evaluation over a pinned corpus."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def corpus_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--corpus", type=Path, default=Path("data/corpus"), help="corpus dir")

    ev = sub.add_parser("eval", help="evaluate methods and chunkers, write Markdown+JSON reports")
    corpus_arg(ev)
    ev.add_argument("--queries", type=Path, default=Path("data/queries.jsonl"))
    ev.add_argument(
        "--methods",
        type=_csv,
        default=DEFAULT_METHODS,
        help=f"comma-separated, from {', '.join(METHODS)} (default: {','.join(DEFAULT_METHODS)})",
    )
    ev.add_argument(
        "--chunkers",
        type=_csv,
        default=("section", "fixed"),
        help="comma-separated chunker specs: section[:MAX], fixed[:SIZE:OVERLAP]",
    )
    ev.add_argument("--baseline", default="bm25/section", help="run id compared against")
    ev.add_argument("--depth", type=int, default=100, help="chunks retrieved per query")
    ev.add_argument("--resamples", type=int, default=10_000, help="bootstrap resamples")
    ev.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    ev.add_argument("--include-boilerplate", action="store_true")
    ev.add_argument("--out", type=Path, default=Path("reports"), help="output directory")

    se = sub.add_parser("search", help="run one query and print cited passages")
    corpus_arg(se)
    se.add_argument("query")
    se.add_argument("--method", default="hybrid", choices=METHODS)
    se.add_argument("--chunker", default="section")
    se.add_argument("-k", type=int, default=5)

    ve = sub.add_parser("verify", help="check corpus checksums and query labels")
    corpus_arg(ve)
    ve.add_argument("--queries", type=Path, default=Path("data/queries.jsonl"))

    fe = sub.add_parser("fetch-corpus", help="download missing corpus files, verifying SHA-256")
    corpus_arg(fe)
    fe.add_argument("--force", action="store_true", help="re-download even if present")
    return parser


def _cmd_eval(args: argparse.Namespace) -> int:
    config = EvalConfig(
        corpus_dir=args.corpus,
        queries_path=args.queries,
        methods=args.methods,
        chunkers=args.chunkers,
        baseline=args.baseline,
        depth=args.depth,
        n_resamples=args.resamples,
        seed=args.seed,
        include_boilerplate=args.include_boilerplate,
    )
    result = run_eval(config)
    md_path, json_path = write_reports(result, args.out)
    print(results_table(result))
    print(f"\nwrote {md_path.as_posix()} and {json_path.as_posix()}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    if args.k <= 0:
        raise ValueError("-k must be positive")
    docs = load_corpus(args.corpus)
    retriever, _ = build_index(docs, args.method, args.chunker)
    hits = retriever.search(args.query, args.k)
    if not hits:
        print("no results")
        return 0
    for rank, hit in enumerate(hits, start=1):
        c = hit.chunk
        snippet = " ".join(c.text.split())[:240]
        print(f'{rank}. {c.doc_id} section {c.section} "{c.section_title}"')
        print(f"   chars {c.start}-{c.end}  chunk {c.chunk_id}  score {hit.score:.4f}")
        print(textwrap.indent(textwrap.fill(snippet, 96), "   "))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    docs = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    validate_queries(queries, passage_index(docs))
    labels = sum(len(q.relevant) for q in queries)
    print(f"corpus ok: {len(docs)} documents match their pinned SHA-256")
    print(f"queries ok: {len(queries)} queries, {labels} labels, all evidence quotes found")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    fetched = fetch_corpus(args.corpus, force=args.force)
    print(f"downloaded {len(fetched)} file(s)" if fetched else "all files present and verified")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handlers = {
        "eval": _cmd_eval,
        "search": _cmd_search,
        "verify": _cmd_verify,
        "fetch-corpus": _cmd_fetch,
    }
    try:
        return handlers[args.command](args)
    except (CorpusError, QrelsError, DenseUnavailableError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
