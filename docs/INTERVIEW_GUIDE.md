# Interview guide

Five questions a senior search or ML engineer is likely to ask about this project, with answers
grounded in the code, followed by the tradeoffs worth raising unprompted and a live demo.
File and function names refer to `src/retrievallab/` unless a path says otherwise.

## 1. "Your hybrid retriever is no better than BM25. Isn't that a failed experiment?"

It is a result, and the harness exists to produce results like it. The default hybrid
(`retrievers/__init__.py::make_retriever("hybrid")`) fuses BM25 and TF-IDF with
`retrievers/hybrid.py::reciprocal_rank_fusion`. Both are bag-of-words models over the same
tokens (`text.py::tokenize`), so they fail on the same queries, for example `q001`, where the
question says "stay valid" and the answer says "lifetime of 10 minutes". RRF can only reorder
what its components retrieve; when both miss, the fusion misses. The report shows nDCG@10
-0.013 with a paired 95% interval of [-0.029, +0.002] against BM25 on section chunks: no
significant difference. A plausible reading of the slightly negative point estimate is that
TF-IDF, which is clearly worse on its own (-0.086 [-0.128, -0.046]), sometimes outvotes a
correct BM25 ranking; the harness does not test that explanation.

The combination where fusion should help is lexical plus dense, available as
`hybrid-dense` with the `dense` extra. It is not in the committed report because CI must not
download models; that is listed as future work, not claimed.

## 2. "You wrote the queries, the labels and the system. Why should I believe any of the numbers?"

The absolute numbers should be read with caution, and `docs/LABELS.md` says why: one annotator,
who is the author; questions written by reading the corpus, which favours lexical methods;
labels not exhaustive. What the design does to limit the damage:

- Labels were written before the first retrieval run (commit `8815782` precedes the eval
  code commit `623f704`), so they were not fitted to any method's output.
- Each label carries an evidence quote that `qrels.py::validate_queries` must find in the
  passage, so a label cannot silently point at the wrong section.
- The only post-hoc change (pass 2) re-judged five zero-recall queries against a pool drawn
  from all six runs, added two labels, removed none, and is recorded in the `note` field of
  each affected query. The pass-1 numbers can be regenerated with the command in
  `docs/LABELS.md`, and no conclusion changed.
- Comparisons are relative and paired (`metrics.py::Bootstrap.paired`), which is more robust to
  label incompleteness than absolute scores: a missing label hurts every method on that query.

What would make it trustworthy beyond that is a second annotator on a sample, with agreement
reported. That is not done.

## 3. "Chunks are retrieved but passages are scored. Doesn't that favour big chunks?"

It can, in both directions, and the first version got it wrong. `evaluate.py::evaluate_run`
retrieves the top 100 chunks and `evaluate.py::ranked_passages` expands each into
`Chunk.passage_ids`, keeping the first occurrence of each passage. For `SectionChunker` a
chunk lies inside one section by construction
(`tests/test_chunking.py::test_section_chunks_never_cross_sections`).

Originally a fixed window was credited only to the section it overlapped most
(`sections.py::section_at`). That stopped big windows collecting several passages, but an
independent review found the opposite bias: 13 labelled sections shorter than about half a
window could never be credited to any window, so fixed runs had a recall ceiling below 1 and the
"fixed loses Recall@10" verdict (-0.079) was mostly that artifact. Now
`sections.py::credited_sections` credits a window with every section it covers at least 100
characters of, or contains completely. To keep it fair to section chunks, each credited passage
takes its own rank, so a window credited to three sections uses three of the ten slots. The
report prints reachable labels per run (152/152 for both defaults), and
`tests/test_eval_e2e.py::test_every_labelled_passage_is_reachable` fails if a chunker setting
makes any label unreachable. BM25's fixed-versus-section Recall@10 difference is now -0.050
[-0.112, +0.007], not significant. What remains: a long section is "found" if any of its chunks
ranks, and the 100-character threshold is a judgement call. Labelling character spans and
scoring overlap would remove both, at the cost of much more labelling work.

## 4. "How do I know the table in the README is what the code produces?"

Three mechanisms:

1. `report.py` writes deterministic output: no timestamps, sorted JSON keys, rounded floats,
   `\n` endings. The bootstrap uses `random.Random(seed)`, whose sequence for a given seed is
   the same on every platform, and draws all resamples once (`metrics.py::Bootstrap.__init__`).
2. `tests/test_eval_e2e.py::test_committed_report_matches_regeneration` reruns the default
   eval and asserts that it equals the committed `reports/eval.md` and that
   `report.py::results_table` appears verbatim in `README.md`.
3. CI (`.github/workflows/ci.yml`) runs `uv run retrievallab eval` and then
   `git diff --exit-code -- reports/` on ubuntu-latest, so a Windows-only or stale result fails.

The corpus cannot drift either: `corpus.py::load_corpus` checks every file's SHA-256 against
`data/corpus/manifest.json`, and `.gitattributes` keeps git from rewriting line endings in the
RFC files.

## 5. "Walk me through your BM25. Which variant, and how did you test it?"

`retrievers/bm25.py::BM25Retriever.score_all` implements Okapi BM25 with `k1 = 1.2`,
`b = 0.75` and the IDF `ln(1 + (N - df + 0.5) / (df + 0.5))` that Lucene uses. The classic
Robertson-Sparck Jones IDF without the `1 +` goes negative for terms in more than half the
documents, which can make adding a common query word lower a matching document's score; this
variant keeps every matching term's contribution positive. Distinct query terms are summed once
each (`dict.fromkeys`), so repeating a word in the query does not change the ranking
(`tests/test_retrievers.py::test_bm25_repeated_query_terms_count_once`). Document length is
the token count after stopword removal, normalised by the average over the index.

Tests use a three-chunk corpus with lengths 2, 3 and 4 tokens so `avgdl = 3`, and assert the
IDF and per-chunk scores against the formula written out by hand in the test
(`test_bm25_idf_hand_computed`, `test_bm25_single_term_scores_hand_computed`,
`test_bm25_multi_term_sums_terms`), plus `b = 0` disabling length normalisation, deterministic
tie-breaking by index position, and re-indexing replacing the old index.

## Tradeoffs to raise

- **Pure Python versus a search library.** Every formula is visible and pinned by tests, and
  the core has no dependencies. The price is speed: fine for about 1,200 chunks (a full eval
  runs in about 3 seconds), wrong for millions of documents, where Lucene, Tantivy or a vector
  database would be the tool and this harness would call them through the `Retriever` protocol
  in `retrievers/base.py`.
- **Boilerplate filtering** (`evaluate.py::indexable_chunks`) improves scores and is what a
  production pipeline would do, but it is a choice that changes the numbers. The report header
  records it and `--include-boilerplate` turns it off.
- **No stemming.** Adding a stemmer would probably help lexical recall here, and would be a new
  run compared with a paired interval, not a silent change to `text.py`.
- **Multiple comparisons.** Fifteen comparisons at 95% means close to one false "significant"
  verdict is expected by chance, so the report adds Holm-adjusted verdicts
  (`metrics.py::holm_adjust`). The uncorrected "worse" for hybrid's MRR@10 does not survive it
  (adjusted p = 0.36).
- **Grades.** Only nDCG@10 uses the 1/2 grades; MRR, Recall and P@5 treat a partial answer as
  a full one. A grade-2-only recall would be a cheap addition.

## Live demo (about five minutes)

```bash
uv sync --locked
uv run retrievallab verify
uv run retrievallab eval
```

1. `verify` prints that 18 documents match their checksums and all 152 evidence quotes
   resolve. Edit one character of `data/corpus/rfc7009.txt`, run `verify` again and show the
   checksum error, then `git checkout data/corpus/rfc7009.txt`.
2. `eval` prints the results table and writes `reports/eval.md`; run `git diff reports/` to
   show that nothing changed.
3. Show a citation and a miss:

   ```bash
   uv run retrievallab search "binding an access token to the hash of the client certificate" --method bm25 -k 3
   uv run retrievallab search "how long can an authorization code be used" --method bm25 -k 3
   ```

   The first returns RFC 8705 sections with character offsets; the second returns related
   sections but not the answer (RFC 6749 section 4.1.2), which is the vocabulary mismatch
   discussed in question 1.
4. Run an ad hoc comparison without touching the committed report:

   ```bash
   uv run retrievallab eval --chunkers section:800,section:1500 --methods bm25 \
       --baseline bm25/section:1500 --out reports/tmp
   ```

   and read the paired interval for the chunk-size change in `reports/tmp/eval.md`.
5. Open `tests/test_retrievers.py::test_bm25_single_term_scores_hand_computed` and walk
   through one score by hand.
