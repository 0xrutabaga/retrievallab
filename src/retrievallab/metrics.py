"""Ranking metrics over one query, plus bootstrap confidence intervals over a query set.

All per-query metrics take ``ranked``, the retrieved passage ids best first with duplicates
already removed, and ``qrels``, a mapping from relevant passage id to a positive integer grade
(2 = directly answers the query, 1 = partially relevant). Passages absent from ``qrels`` are
non-relevant. Hand-computed cases live in tests/test_metrics.py.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


def _check(qrels: Mapping[str, int], k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
    if not qrels or any(g <= 0 for g in qrels.values()):
        raise ValueError("qrels must be non-empty with positive grades")


def recall_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """Fraction of relevant passages that appear in the top ``k``."""
    _check(qrels, k)
    return sum(1 for p in ranked[:k] if p in qrels) / len(qrels)


def precision_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """Relevant passages in the top ``k`` divided by ``k`` (a short ranking is not rewarded)."""
    _check(qrels, k)
    return sum(1 for p in ranked[:k] if p in qrels) / k


def mrr_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float:
    """Reciprocal rank of the first relevant passage in the top ``k``, or 0."""
    _check(qrels, k)
    for rank, p in enumerate(ranked[:k], start=1):
        if p in qrels:
            return 1.0 / rank
    return 0.0


def dcg(gains: Sequence[int]) -> float:
    """Discounted cumulative gain with exponential gain ``2**g - 1`` and ``log2(rank + 1)``."""
    return sum((2.0**g - 1.0) / math.log2(rank + 1) for rank, g in enumerate(gains, start=1))


def ndcg_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float:
    """DCG of the top ``k`` divided by the DCG of the ideal ordering of all labelled passages."""
    _check(qrels, k)
    actual = dcg([qrels.get(p, 0) for p in ranked[:k]])
    ideal = dcg(sorted(qrels.values(), reverse=True)[:k])
    return actual / ideal


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (Hyndman and Fan type 7, numpy's default)."""
    if not sorted_values:
        raise ValueError("quantile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


@dataclass(frozen=True, slots=True)
class Interval:
    mean: float
    low: float
    high: float

    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0


class Bootstrap:
    """Percentile bootstrap over queries with one fixed set of resamples.

    The resample indices are drawn once from a seeded PRNG and reused for every interval, so all
    methods in a report are judged on the same resampled query sets and the output is identical
    on every run and platform (``random.Random`` is a Mersenne Twister with a documented,
    platform-independent sequence for a given seed).
    """

    def __init__(
        self, n_items: int, *, n_resamples: int = 10_000, confidence: float = 0.95, seed: int = 0
    ) -> None:
        if n_items < 1:
            raise ValueError("bootstrap needs at least one item")
        if n_resamples < 1 or not 0.0 < confidence < 1.0:
            raise ValueError("n_resamples must be >= 1 and confidence in (0, 1)")
        self.n_items = n_items
        self.n_resamples = n_resamples
        self.confidence = confidence
        self.seed = seed
        # Statistical resampling, not security: a seeded PRNG is what makes reports reproducible.
        rng = random.Random(seed)  # noqa: S311
        self._samples = [
            [rng.randrange(n_items) for _ in range(n_items)] for _ in range(n_resamples)
        ]

    def interval(self, values: Sequence[float]) -> Interval:
        """Interval for the mean of ``values`` (one value per query)."""
        if len(values) != self.n_items:
            raise ValueError(f"expected {self.n_items} values, got {len(values)}")
        n = self.n_items
        means = sorted(sum(values[i] for i in sample) / n for sample in self._samples)
        alpha = 1.0 - self.confidence
        return Interval(sum(values) / n, quantile(means, alpha / 2), quantile(means, 1 - alpha / 2))

    def paired(self, a: Sequence[float], b: Sequence[float]) -> Interval:
        """Interval for mean(a - b) where a[i] and b[i] are two methods' scores on query i.

        Resampling per-query differences keeps the pairing: both methods face the same hard and
        easy queries in every resample, which is what makes the comparison sensitive.
        """
        if len(a) != len(b):
            raise ValueError("paired samples must have the same length")
        return self.interval([x - y for x, y in zip(a, b, strict=True)])


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for the mean of per-query values (queries resampled)."""
    boot = Bootstrap(len(values), n_resamples=n_resamples, confidence=confidence, seed=seed)
    return boot.interval(values)


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Paired bootstrap interval for mean(a - b); see :meth:`Bootstrap.paired`."""
    boot = Bootstrap(len(a), n_resamples=n_resamples, confidence=confidence, seed=seed)
    return boot.paired(a, b)
