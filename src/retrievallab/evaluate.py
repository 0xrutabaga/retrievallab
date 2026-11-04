"""Run every (method, chunker) pair over the labelled queries and compare them.

Evaluation happens at the passage (RFC section) level, not the chunk level, so chunkers with
different granularity are scored on the same labels:

1. retrieve the top ``depth`` chunks for a query;
2. map each chunk to its passage (``Chunk.passage_id``) and keep the first occurrence of each
   passage, giving a ranked passage list;
3. score that list against the query's graded labels.

Every run's per-query scores are summarised with a bootstrap interval, and every run is
compared with the baseline run by a paired bootstrap on the same resamples. A difference is
reported as significant only when its interval excludes zero.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from retrievallab.chunking import Chunk, chunk_corpus, make_chunker
from retrievallab.corpus import Document, load_corpus
from retrievallab.metrics import (
    Bootstrap,
    Interval,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from retrievallab.qrels import Query, load_queries, passage_index, validate_queries
from retrievallab.retrievers import DEFAULT_METHODS, Retriever, make_retriever
from retrievallab.sections import Section

MetricFn = Callable[[Sequence[str], Mapping[str, int]], float]

METRICS: dict[str, MetricFn] = {
    "ndcg@10": lambda r, q: ndcg_at_k(r, q, 10),
    "mrr@10": lambda r, q: mrr_at_k(r, q, 10),
    "recall@5": lambda r, q: recall_at_k(r, q, 5),
    "recall@10": lambda r, q: recall_at_k(r, q, 10),
    "precision@5": lambda r, q: precision_at_k(r, q, 5),
}
COMPARED_METRICS = ("ndcg@10", "mrr@10", "recall@10")


@dataclass(frozen=True)
class EvalConfig:
    corpus_dir: Path = Path("data/corpus")
    queries_path: Path = Path("data/queries.jsonl")
    methods: tuple[str, ...] = DEFAULT_METHODS
    chunkers: tuple[str, ...] = ("section", "fixed")
    baseline: str = "bm25/section"
    depth: int = 100
    n_resamples: int = 10_000
    confidence: float = 0.95
    seed: int = 0
    include_boilerplate: bool = False


@dataclass(frozen=True)
class Citation:
    rank: int
    passage_id: str
    chunk_id: str
    doc_id: str
    section: str
    section_title: str
    start: int
    end: int
    relevant_grade: int


@dataclass
class RunResult:
    run_id: str
    method: str
    chunker: str
    n_chunks: int
    per_query: dict[str, dict[str, float]] = field(default_factory=dict)
    top: dict[str, list[Citation]] = field(default_factory=dict)
    summary: dict[str, Interval] = field(default_factory=dict)


@dataclass(frozen=True)
class Comparison:
    run_id: str
    baseline: str
    metric: str
    delta: Interval

    @property
    def verdict(self) -> str:
        if not self.delta.excludes_zero():
            return "no significant difference"
        return "better" if self.delta.mean > 0 else "worse"


@dataclass
class EvalResult:
    config: EvalConfig
    n_documents: int
    n_queries: int
    n_labels: int
    corpus_sha256: str
    queries_sha256: str
    runs: list[RunResult]
    comparisons: list[Comparison]


def ranked_passages(chunks: Sequence[Chunk]) -> list[str]:
    """Passage ids in rank order, keeping only the first chunk of each passage."""
    return list(dict.fromkeys(c.passage_id for c in chunks))


def indexable_chunks(
    chunks: Sequence[Chunk], sections: Mapping[str, tuple[Section, str]], include_boilerplate: bool
) -> list[Chunk]:
    if include_boilerplate:
        return list(chunks)
    return [c for c in chunks if not sections[c.passage_id][0].boilerplate]


def build_index(
    docs: Sequence[Document], method: str, chunker_spec: str, *, include_boilerplate: bool = False
) -> tuple[Retriever, list[Chunk]]:
    """Chunk ``docs``, drop boilerplate chunks unless asked not to, and index them."""
    passages = passage_index(docs)
    chunks = indexable_chunks(
        chunk_corpus(docs, make_chunker(chunker_spec)), passages, include_boilerplate
    )
    retriever = make_retriever(method)
    retriever.index(chunks)
    return retriever, chunks


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_run(
    retriever: Retriever,
    queries: Sequence[Query],
    *,
    depth: int,
    keep_top: int = 3,
) -> tuple[dict[str, dict[str, float]], dict[str, list[Citation]]]:
    per_query: dict[str, dict[str, float]] = {}
    top: dict[str, list[Citation]] = {}
    for q in queries:
        hits = retriever.search(q.text, depth)
        ranked = ranked_passages([h.chunk for h in hits])
        per_query[q.qid] = {name: fn(ranked, q.relevant) for name, fn in METRICS.items()}
        first_chunk: dict[str, Chunk] = {}
        for h in hits:
            first_chunk.setdefault(h.chunk.passage_id, h.chunk)
        top[q.qid] = [
            Citation(
                rank=i,
                passage_id=pid,
                chunk_id=first_chunk[pid].chunk_id,
                doc_id=first_chunk[pid].doc_id,
                section=first_chunk[pid].section,
                section_title=first_chunk[pid].section_title,
                start=first_chunk[pid].start,
                end=first_chunk[pid].end,
                relevant_grade=q.relevant.get(pid, 0),
            )
            for i, pid in enumerate(ranked[:keep_top], start=1)
        ]
    return per_query, top


def run_eval(config: EvalConfig) -> EvalResult:
    docs = load_corpus(config.corpus_dir)
    queries = load_queries(config.queries_path)
    passages = passage_index(docs)
    validate_queries(queries, passages)
    for method in config.methods:
        make_retriever(method)  # fail fast on an unknown or unavailable method
    if config.depth < 10:
        raise ValueError("depth must be at least 10 so that @10 metrics are defined")

    boot = Bootstrap(
        len(queries),
        n_resamples=config.n_resamples,
        confidence=config.confidence,
        seed=config.seed,
    )
    runs: list[RunResult] = []
    for spec in config.chunkers:
        chunker = make_chunker(spec)
        chunks = indexable_chunks(chunk_corpus(docs, chunker), passages, config.include_boilerplate)
        for method in config.methods:
            retriever = make_retriever(method)
            retriever.index(chunks)
            per_query, top = evaluate_run(retriever, queries, depth=config.depth)
            run = RunResult(f"{method}/{spec}", method, chunker.name, len(chunks), per_query, top)
            for name in METRICS:
                run.summary[name] = boot.interval([per_query[q.qid][name] for q in queries])
            runs.append(run)

    by_id = {r.run_id: r for r in runs}
    if config.baseline not in by_id:
        raise ValueError(f"baseline {config.baseline!r} is not among the runs {sorted(by_id)}")
    base = by_id[config.baseline]
    comparisons: list[Comparison] = []
    for run in runs:
        if run is base:
            continue
        for metric in COMPARED_METRICS:
            a = [run.per_query[q.qid][metric] for q in queries]
            b = [base.per_query[q.qid][metric] for q in queries]
            comparisons.append(Comparison(run.run_id, base.run_id, metric, boot.paired(a, b)))

    return EvalResult(
        config=config,
        n_documents=len(docs),
        n_queries=len(queries),
        n_labels=sum(len(q.relevant) for q in queries),
        corpus_sha256=_file_sha256(config.corpus_dir / "manifest.json"),
        queries_sha256=_file_sha256(config.queries_path),
        runs=runs,
        comparisons=comparisons,
    )
