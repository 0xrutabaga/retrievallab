# Engineering decisions

Each entry states the decision, the alternatives considered, and what it costs. File and
function names refer to `src/retrievallab/`.

## 1. Evaluate passages, not chunks

**Decision.** Labels name RFC sections (`rfc6749#4.1.2`). `evaluate.evaluate_run` expands
each retrieved chunk into the passages it is credited to (`Chunk.passage_ids`), and
`evaluate.ranked_passages` keeps the first occurrence of each passage before scoring. A
section chunk is credited to its own section. A fixed window is credited to every section it
overlaps by at least `MIN_CREDIT_CHARS` (100) characters or contains completely
(`sections.credited_sections`), largest overlap first, and each credited passage takes its own
rank.

**Why.** Chunk-level labels would have to be rewritten for every chunker and could not
compare a 1000-character window with a section-aligned chunk. Section ids are stable under any
chunking and are what a reader would cite.

**Cost.** A long section counts as found if any chunk of it ranks, even one that does not hold
the answering sentence.

**History.** The first version credited a window only to the section it overlapped most
(`sections.section_at`). An independent review showed that this made any labelled section
shorter than about half a window unreachable: 13 labels (in 13 queries, for example
`rfc9700#2.2.2`, `rfc8705#6.5`, `rfc9126#7.1`) could never be retrieved by a `fixed` run, and
the "fixed windows lose Recall@10" verdict (-0.079 [-0.134, -0.028]) was largely that
artifact. With multi-section credit all 152 labels are reachable under both default chunkers,
the report prints reachable labels per run, and
`tests/test_eval_e2e.py::test_every_labelled_passage_is_reachable` keeps it that way. BM25's
fixed-versus-section Recall@10 difference is now -0.050 [-0.112, +0.007], not significant.

**Fairness between chunkers.** Crediting a window with every section it touches would let a
large window collect several passages at one rank. Two rules limit that: each credited
passage occupies its own position in the ranked list, so a window credited to three sections
uses three of the ten slots in nDCG@10 and P@5 just as three separate chunks would; and a
section only counts if the window covers 100 characters of it or all of it, so grazing a
neighbour's heading line earns nothing. The threshold is a judgement call; scoring labelled
character spans would remove it at the cost of span-level labels.

## 2. Sections from RFC layout rules, checked against the table of contents

**Decision.** `sections.split_sections` treats a column-0 line as a heading if it is numbered
(`4.1.2.  Title`), an appendix (`Appendix A.  Title`, `A.1.  Title`), or one of a fixed list of
unnumbered headings. Everything else at column 0 (page headers, footers) is body text.

**Why.** RFC plain text indents body text by three spaces, so column 0 is a reliable signal;
a format-aware parser of the XML sources would be more work and need another download.
`tests/test_sections.py::test_every_toc_entry_is_parsed_as_a_section` checks every numbered
entry of each RFC's own table of contents is found, which catches a regex that silently
misses headings.

**Cost.** Specific to RFC layout. Page headers and footers stay inside chunks as noise.

## 3. Drop boilerplate sections from the index

**Decision.** Front matter, Status of This Memo, Copyright Notice, Table of Contents,
Acknowledgements, Authors' Addresses and References sections are marked `boilerplate` and
removed by `evaluate.indexable_chunks` unless `--include-boilerplate` is passed.

**Why.** A table of contents repeats every section title and matches many queries without
answering any. Real RAG pipelines strip it. Chunkers still cover the whole text (the coverage
invariant is tested before filtering).

**Cost.** A design choice that changes the numbers; it is recorded in every report header.

## 4. Two chunkers with one set of invariants

**Decision.** `chunking.FixedSizeChunker` (windows of `size` characters, `overlap` shared,
ends moved back to whitespace) and `chunking.SectionChunker` (paragraphs packed up to
`max_chars`, never across a section; an oversized paragraph is split into word-aligned
windows). Chunk id is `f"{doc_id}@{start}-{end}"`.

**Why.** These are the two strategies most RAG systems start from. Offset-based ids are
deterministic, unique within a document and double as a citation (open the file, slice the
range). A content hash was considered; it adds nothing that offsets do not already give when
the corpus is checksummed.

**Cost.** Ids change if the document text changes, which the checksum prevents. The tests in
`tests/test_chunking.py` assert coverage, no gaps, exact `text[start:end]` and determinism on
all 18 RFCs and on 25 random texts.

## 5. Write BM25, TF-IDF and the metrics instead of importing them

**Decision.** `retrievers/bm25.py`, `retrievers/tfidf.py`, `retrievers/hybrid.py` and
`metrics.py` are written from their formulas in the standard library. The core package has no
runtime dependencies.

**Why.** The plan asked for an own BM25. Beyond that, the exact formula matters: BM25 variants
differ in IDF (Robertson's can go negative; Lucene's `ln(1 + (N - df + 0.5)/(df + 0.5))` does
not) and in how repeated query terms count. Owning the code lets `tests/test_retrievers.py`
pin scores computed by hand. `rank_bm25` and scikit-learn were the alternatives; both would add
dependencies for about 60 lines each.

**Cost.** Pure-Python inverted indexes are slow for large corpora. Here the corpus is about
1,200 chunks and a full eval takes about 3 seconds.

## 6. RRF for hybrid fusion

**Decision.** `retrievers.hybrid.reciprocal_rank_fusion` with `k_rrf = 60` over the top 100
of each component; ties broken by first appearance.

**Why.** RRF needs no score calibration between BM25 (unbounded) and cosine (0 to 1) and has no
weight to tune on the test queries. Linear score combination was the alternative; it needs a
weight, and tuning that weight on the same 71 queries would overfit.

**Cost.** The default hybrid fuses two lexical methods, which make correlated errors; the
report shows it does not beat BM25 alone. `hybrid-dense` (BM25 plus dense) is the combination
where fusion would be expected to help, and it needs the optional extra.

## 7. Bootstrap with one fixed set of resamples

**Decision.** `metrics.Bootstrap` draws 10,000 resamples of query indices once from
`random.Random(seed)` and reuses them for every interval in a report. Comparisons use the
paired difference per query (`Bootstrap.paired`). The report gives two verdicts: uncorrected
(`evaluate.Comparison.verdict`, the 95% percentile interval excludes zero) and Holm-corrected
(`evaluate.Comparison.holm_verdict`), from two-sided bootstrap p-values
(`Bootstrap.p_value`) adjusted by `metrics.holm_adjust` over all comparisons in the report.
The README quotes the Holm column.

**Why.** Per-query nDCG is bounded and skewed, so a normal approximation is doubtful at this
size. Pairing removes query difficulty, which dominates variance: the intervals on single runs
are about +/- 0.07, while a paired difference between two similar runs (BM25 and hybrid on
section chunks) has an interval about +/- 0.016 wide on each side. Sharing resamples across
runs keeps the report deterministic and the comparisons consistent.

**Cost.** Percentile intervals undercover slightly at small n. Holm is conservative, so with
71 queries small real differences will be reported as not significant. Bootstrap p-values have
a resolution of 1/10,000, so the smallest are printed as "<0.0001". An earlier report used
only the uncorrected verdict, under which hybrid's MRR@10 (-0.017 [-0.035, -0.001]) looked
"worse"; it does not survive Holm (adjusted p = 0.36).

## 8. Reports are deterministic and CI enforces it

**Decision.** `report.write_reports` writes sorted JSON with rounded floats, no timestamps or
timings, and `\n` line endings. CI runs `retrievallab eval` and then
`git diff --exit-code -- reports/`. `tests/test_eval_e2e.py::test_committed_report_matches_regeneration`
does the same locally and also checks that the README's results table is the generated one.

**Why.** The quality bar is that every number in the README is reproducible by a command. Making
staleness a test failure is cheaper than remembering to update numbers.

**Cost.** Any change that shifts a score needs the report and README regenerated in the same
commit.

## 9. Labels carry evidence quotes

**Decision.** Each label has a quoted phrase that `qrels.validate_queries` must find in the
labelled passage.

**Why.** A typo in a passage id (`4.1.2` versus `4.2.1`) silently makes a query unanswerable
and lowers every score. The quote turns that into an error, and also documents why the label
was given.

**Cost.** Writing the quotes took longer than the labels. Quotes that cross a page break fail,
because page headers sit between the lines, and so do quotes across a hyphenated line wrap
(RFC 7523 wraps `grant-type` as `grant-` and `type` on two lines); such quotes were replaced
with shorter ones.

## 10. The corpus is committed, not fetched in CI

**Decision.** The 18 RFC files (1.2 MB) are committed unmodified with SHA-256 pins in
`data/corpus/manifest.json`; `.gitattributes` marks them `-text` so git never rewrites line
endings. `retrievallab fetch-corpus` (`corpus.fetch_corpus`) can re-download them and writes a
file only after its checksum matches.

**Why.** CI must be deterministic and work offline. 1.2 MB is small, and the IETF Trust
Legal Provisions permit redistribution in full.

**Cost.** Repository size; updating the corpus means updating checksums and re-validating
labels.

## 11. Dense retrieval as an optional extra behind a protocol

**Decision.** `retrievers.dense.DenseRetriever` depends on an `Encoder` protocol.
`SentenceTransformerEncoder` imports `sentence_transformers` lazily and raises
`DenseUnavailableError` with the install command if it is missing.

**Why.** Torch and model downloads would make `uv sync` large and CI network-dependent. The
protocol lets tests use a deterministic fake encoder.

**Cost.** The real model path is not exercised by tests or CI and has no numbers in the
committed report.
