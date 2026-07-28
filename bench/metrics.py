"""Scoring for §6.8: ranking quality and probabilistic calibration.

Plain implementations, no dependencies, so the numbers can be checked by hand
on a small case. Every function takes ground truth that came from the corpus
scan, never from a model.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence


# ── ranking (§6.8 retrieval arm) ────────────────────────────────────────────

def recall_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold = set(gold)
    if not gold:
        return 0.0
    hits = len(gold & set(ranked[:k]))
    return hits / len(gold)


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Binary relevance: gold documents score 1, everything else 0."""
    gold = set(gold)
    if not gold:
        return 0.0
    gains = [1.0 if doc in gold else 0.0 for doc in ranked[:k]]
    ideal = [1.0] * min(len(gold), k)
    denom = dcg(ideal)
    return (dcg(gains) / denom) if denom else 0.0


def mrr(ranked: Sequence[str], gold: Iterable[str]) -> float:
    gold = set(gold)
    for i, doc in enumerate(ranked):
        if doc in gold:
            return 1.0 / (i + 1)
    return 0.0


# ── calibration (§6.8 calibration arm) ──────────────────────────────────────

def brier(pairs: Sequence[tuple[float, int]]) -> Optional[float]:
    if not pairs:
        return None
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def log_loss(pairs: Sequence[tuple[float, int]], eps: float = 1e-6) -> Optional[float]:
    if not pairs:
        return None
    total = 0.0
    for p, y in pairs:
        q = min(1.0 - eps, max(eps, p))
        total += -(math.log(q) if y else math.log(1.0 - q))
    return total / len(pairs)


def accuracy(pairs: Sequence[tuple[float, int]], threshold: float = 0.5) -> Optional[float]:
    if not pairs:
        return None
    return sum(1 for p, y in pairs if (p >= threshold) == bool(y)) / len(pairs)


def reliability_slope(pairs: Sequence[tuple[float, int]],
                      n_buckets: int = 10) -> Optional[float]:
    """Least-squares slope of observed frequency against mean predicted p.

    A perfectly calibrated predictor has slope 1. Buckets with no mass are
    skipped rather than imputed.
    """
    buckets: dict[int, list[tuple[float, int]]] = {}
    for p, y in pairs:
        b = min(n_buckets - 1, max(0, int(p * n_buckets)))
        buckets.setdefault(b, []).append((p, y))
    points = [(sum(p for p, _ in v) / len(v), sum(y for _, y in v) / len(v))
              for v in buckets.values() if v]
    if len(points) < 2:
        return None
    mx = sum(x for x, _ in points) / len(points)
    my = sum(y for _, y in points) / len(points)
    denom = sum((x - mx) ** 2 for x, _ in points)
    if denom == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in points) / denom


def ece(pairs: Sequence[tuple[float, int]], n_buckets: int = 10) -> Optional[float]:
    """Expected calibration error: mass-weighted |confidence - accuracy|."""
    if not pairs:
        return None
    buckets: dict[int, list[tuple[float, int]]] = {}
    for p, y in pairs:
        b = min(n_buckets - 1, max(0, int(p * n_buckets)))
        buckets.setdefault(b, []).append((p, y))
    total = 0.0
    for group in buckets.values():
        conf = sum(p for p, _ in group) / len(group)
        acc = sum(y for _, y in group) / len(group)
        total += (len(group) / len(pairs)) * abs(conf - acc)
    return total


def summarize(pairs: Sequence[tuple[float, int]]) -> dict[str, Optional[float]]:
    return {
        "n": len(pairs),
        "brier": brier(pairs),
        "log_loss": log_loss(pairs),
        "accuracy": accuracy(pairs),
        "ece": ece(pairs),
        "reliability_slope": reliability_slope(pairs),
        "base_rate": (sum(y for _, y in pairs) / len(pairs)) if pairs else None,
    }


# ── significance, so a margin is not read off noise ─────────────────────────

def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_resamples: int = 10_000,
                     seed: int = 20260725) -> dict[str, float]:
    """Paired bootstrap over per-item scores. Returns mean delta and a CI.

    Paired because both systems answer the same items; the pairing removes
    item difficulty from the comparison.
    """
    import random as _random
    if len(a) != len(b) or not a:
        return {"delta": 0.0, "lo": 0.0, "hi": 0.0, "p_gt_0": 0.0}
    rng = _random.Random(seed)
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    observed = sum(diffs) / n
    means = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples) - 1]
    p_gt = sum(1 for m in means if m > 0.0) / n_resamples
    return {"delta": observed, "lo": lo, "hi": hi, "p_gt_0": p_gt}
