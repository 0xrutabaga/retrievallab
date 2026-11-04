"""Hybrid retrieval by reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009).

    rrf(d) = sum over component rankings r that contain d of 1 / (k_rrf + rank_r(d))

with 1-based ranks and ``k_rrf = 60`` as in the paper. Only ranks are used, so components whose
raw scores live on different scales (BM25 and cosine) are fused without calibration. Each
component contributes its top ``depth`` hits.
"""

from __future__ import annotations

from collections.abc import Sequence

from retrievallab.chunking import Chunk
from retrievallab.retrievers.base import Hit, Retriever


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k_rrf: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked id lists. Ties are broken by first appearance, scanning rankings in order."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k_rrf + rank)
    first_seen = {item: i for i, item in enumerate(dict.fromkeys(x for r in rankings for x in r))}
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


class HybridRRFRetriever:
    def __init__(self, components: Sequence[Retriever], k_rrf: int = 60, depth: int = 100) -> None:
        if not components:
            raise ValueError("hybrid retrieval needs at least one component")
        if k_rrf < 1 or depth < 1:
            raise ValueError("k_rrf and depth must be positive")
        self.components = list(components)
        self.k_rrf = k_rrf
        self.depth = depth
        self.name = "hybrid(" + "+".join(c.name for c in self.components) + ")"
        self._by_id: dict[str, Chunk] = {}

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._by_id = {c.chunk_id: c for c in chunks}
        for component in self.components:
            component.index(chunks)

    def search(self, query: str, k: int) -> list[Hit]:
        if k <= 0:
            raise ValueError("k must be positive")
        depth = max(k, self.depth)
        rankings = [[h.chunk.chunk_id for h in c.search(query, depth)] for c in self.components]
        fused = reciprocal_rank_fusion(rankings, self.k_rrf)
        return [Hit(self._by_id[cid], score) for cid, score in fused[:k]]
