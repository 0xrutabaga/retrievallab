# History migration, 2026-11-09

The history of `main` was rewritten once, on 2026-11-09, so every commit id before this document
changed. This page records what changed so older references can be followed.

## What changed

One line was removed from all 13 commit messages: an AI co-author trailer naming Claude
(`Co-Authored-By: Claude Opus 5.5`). GitHub displays that trailer as a repository contributor.
The project was developed with AI assistance; removing the trailer does not change that.

Nothing else changed. Checked commit by commit against the original history: the tree of every
commit, the author and committer names and emails, the author and committer dates, the commit
order, and the rest of each message are identical. The rewrite used `git-filter-repo` 2.47.0.

## Reproducibility

The experiment inputs and outputs are byte-identical across the rewrite:

- the pass-1 labels (`data/queries.jsonl` at `d5a0d44`) have git blob id
  `299901a5cc44fac99db29a486b305385986d8f35`, the same as before the rewrite;
- the final tree before this document was added, `47567c73ca33ffc4bc40afdda4a4b452a7d92e37`, is
  unchanged, including `reports/`, `data/queries.jsonl` and the corpus manifest.

The commit ids quoted in `docs/LABELS.md` and `docs/INTERVIEW_GUIDE.md` were updated in the
commit that added this page, which is a separate commit on top of the rewritten history. Those
two files also now state how little the commit timestamps can prove, and that commit also
corrected the description of the annotator in `README.md`, `docs/LABELS.md` and
`docs/INTERVIEW_GUIDE.md`: the labels were written by an AI model, not by a human author.

## Commit id mapping

| Before | After | Author date | Subject |
| --- | --- | --- | --- |
| `7bbf611` | `ffb1e92` | 2026-11-04T07:24:45+07:00 | Add pinned OAuth RFC corpus with SHA-256 manifest and licence notice |
| `3da522e` | `70290e7` | 2026-11-04T07:24:45+07:00 | Add corpus loader, RFC section parser and chunkers with stable ids |
| `34ff8af` | `ced824e` | 2026-11-04T07:24:45+07:00 | Add BM25, TF-IDF, RRF hybrid and optional dense retrievers; IR metrics and bootstrap |
| `8815782` | `d5a0d44` | 2026-11-04T07:24:46+07:00 | Add 71 labelled queries with evidence quotes and label validation |
| `623f704` | `4ac4d25` | 2026-11-04T07:24:46+07:00 | Add eval pipeline, deterministic Markdown/JSON reports and CLI |
| `32737f7` | `95a9c05` | 2026-11-04T07:24:46+07:00 | Labels pass 2: pooled review of the five zero-recall queries |
| `d3f8d23` | `0612a91` | 2026-11-04T07:24:56+07:00 | Keep corpus bytes out of git line-ending normalisation |
| `1cc2214` | `2953abf` | 2026-11-04T07:33:23+07:00 | Add tests: hand-computed metrics and BM25, chunker invariants, failure paths, e2e |
| `434ebb2` | `bd2bb73` | 2026-11-04T07:33:23+07:00 | Add generated evaluation report |
| `00ba9e3` | `545f9f4` | 2026-11-04T07:33:23+07:00 | Add CI with pinned actions, report regeneration check and pip-audit |
| `b83a2da` | `0e7c5c4` | 2026-11-04T07:33:24+07:00 | Add README, labelling method, engineering decisions and interview guide |
| `391d65d` | `3f7d79c` | 2026-11-04T15:29:01+07:00 | Self-review fixes: validate method names without loading models; fix pass-1 repro command |
| `be06bc1` | `840572e` | 2026-11-04T16:32:51+07:00 | Credit fixed windows to every covered section; report reachability and Holm verdicts |
