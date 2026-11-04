# How the labelled queries were made

`data/queries.jsonl` holds 71 queries and 152 graded labels (84 at grade 2, 68 at grade 1)
over passages in all 18 RFCs. This page records how they were produced, so a reader can judge
how far to trust the numbers built on them.

## Unit of labelling

A label names a *passage*: one RFC section as produced by `sections.split_sections`, written
`doc_id#section` (`rfc6749#4.1.2`, `rfc8252#B.1`, `rfc7662#abstract`). A parent section such
as `4.1` covers only the text before its first child, so labels name the most specific section
that holds the answer. Passages are independent of chunking, which is why one label set can
score every chunker.

Grades:

- **2**: the passage answers the query directly (it states the rule, value or mechanism asked
  about).
- **1**: the passage is on the topic and partially answers it (background, a related
  requirement, a cross-reference that a reader would also want).

Every label carries an `evidence` quote: a phrase copied from the passage that shows why it was
labelled. `qrels.validate_queries` fails if the quote is not in the passage (compared
case-insensitively, whitespace collapsed), if the passage id does not exist, or if the passage
is boilerplate that is never indexed. This runs in the test suite and in CI
(`retrievallab verify`).

## Procedure

Pass 1 (commit `8815782`, before any retrieval run):

1. Listed every section of the corpus with its title and length (`split_sections`).
2. Wrote 71 questions an engineer implementing OAuth might ask, spread over all 18 RFCs. Most
   are paraphrased so they do not repeat the section title; a few (`q070`, `q071`) were written
   deliberately with little word overlap with the answer.
3. For each question, chose the answering sections by reading the RFC text and by searching it
   with `grep` for the key terms and their synonyms, then read each candidate section and
   assigned a grade.
4. Copied an evidence quote for each label and ran the validator until every quote resolved.

Pass 2 (commit `32737f7`, after the first evaluation run):

The first run left five queries with Recall@10 = 0 in every run (`q001`, `q018`, `q023`,
`q032`, `q053`). For those five only, the top five passages from each of the six default runs
were pooled and every unlabelled passage in the pool was judged. Two labels were added
(`rfc7662#2` at grade 2 for `q023`, `rfc9126#abstract` at grade 1 for `q053`); for the other
three queries no pooled passage was judged relevant. Each affected query has a `note` field
saying so. No label was removed or regraded in pass 2.

Pass 2 raised every run's numbers slightly and changed no conclusion. To reproduce the pass-1
numbers:

```bash
mkdir -p reports/tmp
git show 8815782:data/queries.jsonl > reports/tmp/queries-pass1.jsonl
uv run retrievallab eval --queries reports/tmp/queries-pass1.jsonl --out reports/tmp
```

which gives, for example, nDCG@10 of 0.598 for `bm25/section` and 0.584 for `hybrid/section`,
against 0.602 and 0.589 with the current labels.

## Known limits

- **Single annotator, who is also the system's author.** No second judge, so no agreement
  figure. Grade boundaries (1 versus 2) are one person's judgement.
- **Questions came from reading the corpus.** Even paraphrased, they lean on the RFCs'
  vocabulary, which favours lexical retrievers over dense ones. Questions from real users would
  be phrased differently.
- **Labels are not exhaustive.** Pass 1 relied on reading and keyword search; pass 2 pooled
  system output for five queries only. An answering passage that was never labelled counts as
  non-relevant, which lowers every method's score and can penalise a method that finds unusual
  correct passages.
- **Pooling bias in pass 2.** The pool came from lexical runs only, so any passages added there
  are ones lexical methods already find.
- **Skewed coverage.** RFC 9700 (the security best-practice document) has 31 labels and RFC
  9126 has 3, reflecting where the questions led rather than a sampling design.
- **Size.** 71 queries is enough to separate large differences (TF-IDF versus BM25 here), not
  small ones; see the confidence intervals in `reports/eval.md`.
