"""Metrics against cases computed by hand (the arithmetic is written out in each test)."""

from __future__ import annotations

import math

import pytest

from retrievallab.metrics import (
    Bootstrap,
    Interval,
    bootstrap_ci,
    dcg,
    holm_adjust,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap_ci,
    precision_at_k,
    quantile,
    recall_at_k,
)

RANKED = ["a", "x", "b", "y", "c"]
QRELS = {"a": 2, "b": 1, "z": 1}  # "z" is relevant but never retrieved


def test_recall_at_k() -> None:
    assert recall_at_k(RANKED, QRELS, 1) == pytest.approx(1 / 3)  # a
    assert recall_at_k(RANKED, QRELS, 2) == pytest.approx(1 / 3)  # a, x
    assert recall_at_k(RANKED, QRELS, 5) == pytest.approx(2 / 3)  # a, b of {a, b, z}
    assert recall_at_k(RANKED, QRELS, 100) == pytest.approx(2 / 3)


def test_precision_at_k_divides_by_k_even_for_short_rankings() -> None:
    assert precision_at_k(RANKED, QRELS, 1) == 1.0
    assert precision_at_k(RANKED, QRELS, 5) == pytest.approx(2 / 5)
    assert precision_at_k(["a"], QRELS, 5) == pytest.approx(1 / 5)


def test_mrr() -> None:
    assert mrr_at_k(RANKED, QRELS) == 1.0
    assert mrr_at_k(["x", "y", "b"], QRELS) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "y", "b"], QRELS, k=2) == 0.0
    assert mrr_at_k([], QRELS) == 0.0


def test_dcg_uses_exponential_gain_and_log2_discount() -> None:
    # (2^2 - 1)/log2(2) + 0 + (2^1 - 1)/log2(4) = 3 + 0.5
    assert dcg([2, 0, 1]) == pytest.approx(3.5)


def test_ndcg_hand_computed() -> None:
    # DCG of RANKED: a(2) at rank 1, b(1) at rank 3 -> 3/1 + 1/2 = 3.5
    # IDCG: grades [2, 1, 1] -> 3/1 + 1/log2(3) + 1/log2(4) = 3 + 0.63093 + 0.5
    expected = 3.5 / (3 + 1 / math.log2(3) + 0.5)
    assert ndcg_at_k(RANKED, QRELS, 10) == pytest.approx(expected)
    assert expected == pytest.approx(0.847267, abs=1e-6)


def test_ndcg_perfect_and_empty() -> None:
    assert ndcg_at_k(["a", "b", "z"], QRELS) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], QRELS) == 0.0
    assert ndcg_at_k([], QRELS) == 0.0


def test_ndcg_cutoff_ignores_lower_ranks() -> None:
    # Only rank 1 counts at k=1: DCG = 0, so 0 even though "a" is at rank 2.
    assert ndcg_at_k(["x", "a"], {"a": 2}, 1) == 0.0
    # IDCG at k=1 uses only the best grade: 3. DCG at k=1 with "b" (grade 1) = 1.
    assert ndcg_at_k(["b"], QRELS, 1) == pytest.approx(1 / 3)


@pytest.mark.parametrize("fn", [recall_at_k, precision_at_k, mrr_at_k, ndcg_at_k])
def test_metrics_reject_bad_input(fn) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        fn(RANKED, QRELS, 0)
    with pytest.raises(ValueError):
        fn(RANKED, {}, 5)
    with pytest.raises(ValueError):
        fn(RANKED, {"a": 0}, 5)


def test_quantile_type7() -> None:
    assert quantile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert quantile([1.0, 2.0, 3.0, 4.0], 0.25) == pytest.approx(1.75)  # pos 0.75
    assert quantile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert quantile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0
    assert quantile([7.0], 0.3) == 7.0
    with pytest.raises(ValueError):
        quantile([], 0.5)
    with pytest.raises(ValueError):
        quantile([1.0], 1.5)


def test_bootstrap_constant_values_collapse() -> None:
    iv = bootstrap_ci([0.4] * 10, n_resamples=200)
    assert iv.mean == pytest.approx(0.4)
    assert iv.low == pytest.approx(0.4)
    assert iv.high == pytest.approx(0.4)


def test_bootstrap_is_deterministic_and_brackets_mean() -> None:
    values = [0.0, 1.0, 0.5, 0.25, 1.0, 0.0, 0.75, 0.5]
    a = bootstrap_ci(values, n_resamples=2000, seed=7)
    b = bootstrap_ci(values, n_resamples=2000, seed=7)
    assert a == b
    assert a.low < a.mean < a.high
    assert a.low >= 0.0 and a.high <= 1.0
    assert bootstrap_ci(values, n_resamples=2000, seed=8) != a


def test_bootstrap_interval_narrows_with_more_queries() -> None:
    small = bootstrap_ci([0.0, 1.0] * 5, n_resamples=2000)
    large = bootstrap_ci([0.0, 1.0] * 50, n_resamples=2000)
    assert (large.high - large.low) < (small.high - small.low)


def test_paired_bootstrap_detects_consistent_improvement() -> None:
    base = [0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7, 0.0, 0.9]
    better = [x + 0.1 for x in base]
    iv = paired_bootstrap_ci(better, base, n_resamples=1000)
    assert iv.mean == pytest.approx(0.1)
    assert iv.excludes_zero() and iv.low > 0
    same = paired_bootstrap_ci(base, base, n_resamples=1000)
    assert same == Interval(0.0, 0.0, 0.0)
    assert not same.excludes_zero()


def test_paired_bootstrap_mixed_differences_include_zero() -> None:
    a = [1.0, 0.0] * 10
    b = [0.0, 1.0] * 10
    assert not paired_bootstrap_ci(a, b, n_resamples=1000).excludes_zero()


def test_bootstrap_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        bootstrap_ci([1.0], n_resamples=0)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0], confidence=1.0)
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        Bootstrap(3, n_resamples=10).interval([1.0])


def test_holm_adjust_hand_computed() -> None:
    # Sorted: 0.005*4 = 0.02, 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> running max 0.06.
    assert holm_adjust([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])  # capped at 1
    assert holm_adjust([]) == []


def test_bootstrap_p_value() -> None:
    boot = Bootstrap(10, n_resamples=1000)
    assert boot.p_value([0.1] * 10) == 0.0  # every resampled mean is positive
    assert boot.p_value([0.0] * 10) == 1.0  # every mean is exactly zero
    mixed = boot.p_value([1.0, -1.0] * 5)
    assert 0.5 < mixed <= 1.0
    with pytest.raises(ValueError):
        boot.p_value([1.0])
