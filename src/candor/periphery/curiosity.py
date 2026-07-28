"""Curiosity engine — untrusted proposer (spec §3.10, §4.5–4.7).

Owns dispersion testing, changepoint detection on the time axis, guard
proposal, breadth accounting and caveat propagation. Everything it finds is a
*candidate*: repairs pass through the gate like everything else.

Stage 5 of the build order (§8) is where this becomes load-bearing. The
statistics below are implemented and unit-tested; the wiring that turns them
into gate candidates is deliberately conservative until Stage 5 runs its gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

# §4.5 gate-hard defaults, because covariate search is a fishing expedition.
MIN_SUPPORT_PER_PARTITION = 8
TARONE_Z_THRESHOLD = 3.0
CUSUM_THRESHOLD = 5.0
CUSUM_DRIFT = 0.5
BH_ALPHA = 0.05


@dataclass(frozen=True)
class Group:
    n: int
    k: int


def tarone_z(groups: Sequence[Group]) -> Optional[float]:
    """Tarone's Z for binomial overdispersion. None when it is undefined."""
    usable = [g for g in groups if g.n > 0]
    if len(usable) < 2:
        return None
    total_n = sum(g.n for g in usable)
    total_k = sum(g.k for g in usable)
    p = total_k / total_n
    if p <= 0.0 or p >= 1.0:
        return None
    chi = sum((g.k - g.n * p) ** 2 for g in usable) / (p * (1.0 - p))
    denom = math.sqrt(2.0 * sum(g.n * (g.n - 1) for g in usable) * p / (1.0 - p)
                      + 1e-300)
    if denom <= 0.0:
        return None
    return (chi - total_n) / denom


def overdispersed(groups: Sequence[Group],
                  threshold: float = TARONE_Z_THRESHOLD) -> bool:
    z = tarone_z(groups)
    return z is not None and z > threshold


def partition_by_key(observations: Iterable[tuple[dict[str, str], bool]],
                     key: str) -> dict[str, Group]:
    """Group observations by one recorded obs_context component (never the hash)."""
    out: dict[str, list[int]] = {}
    for ctx, outcome in observations:
        if key not in ctx:
            continue
        bucket = out.setdefault(ctx[key], [0, 0])
        bucket[0] += 1
        bucket[1] += 1 if outcome else 0
    return {v: Group(n, k) for v, (n, k) in out.items()}


def cusum_changepoint(series: Sequence[bool], threshold: float = CUSUM_THRESHOLD,
                      drift: float = CUSUM_DRIFT) -> Optional[int]:
    """Two-sided CUSUM on a Bernoulli series. Returns the first alarm index."""
    if len(series) < 2 * MIN_SUPPORT_PER_PARTITION:
        return None
    mean = sum(1 for x in series if x) / len(series)
    sd = math.sqrt(mean * (1.0 - mean))
    if sd <= 0.0:
        return None                      # a constant series has no changepoint
    # slack and alarm level are both in units of the series' own scale, so the
    # defaults do not silently mean different things at different base rates.
    slack = drift * sd
    alarm = threshold * sd
    hi = lo = 0.0
    for i, x in enumerate(series):
        dev = (1.0 if x else 0.0) - mean
        hi = max(0.0, hi + dev - slack)
        lo = min(0.0, lo + dev + slack)
        if hi > alarm or lo < -alarm:
            return i
    return None


def benjamini_hochberg(pvalues: Sequence[float],
                       alpha: float = BH_ALPHA) -> list[bool]:
    """Multiple-comparisons correction over the covariate set tested (§4.5)."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    keep = [False] * m
    cutoff = -1
    for rank, i in enumerate(order, start=1):
        if pvalues[i] <= alpha * rank / m:
            cutoff = rank
    for rank, i in enumerate(order, start=1):
        keep[i] = rank <= cutoff
    return keep


def description_length(groups: Sequence[Group]) -> float:
    """Bits to encode the outcomes given per-group rates. Used by the MDL gate."""
    total = 0.0
    for g in groups:
        if g.n == 0:
            continue
        p = (g.k + 0.5) / (g.n + 1.0)
        total += -(g.k * math.log2(p) + (g.n - g.k) * math.log2(1.0 - p))
    return total


def mdl_gain(residual: Sequence[Group], split: Sequence[Group],
             guard_bits: float) -> dict[str, float]:
    """DL(guard) + DL(residual | guard) < DL(residual) — the §4.5 acceptance test."""
    dl_residual = description_length(residual)
    dl_given = description_length(split)
    return {"dl_guard": guard_bits, "dl_residual_given_guard": dl_given,
            "dl_residual": dl_residual}


def normalized_entropy(values: Sequence[str]) -> float:
    """§4.6 breadth_key: normalized entropy of a covariate's observed values."""
    if not values:
        return 0.0
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    if len(counts) <= 1:
        return 0.0
    total = len(values)
    h = -sum((c / total) * math.log(c / total) for c in counts.values())
    return h / math.log(len(counts))


def breadth_class(breadth: float) -> str:
    if breadth < 0.34:
        return "narrow"
    if breadth < 0.67:
        return "moderate"
    return "broad"


TRANSFERABILITY_CAP = {"narrow": 0.5, "moderate": 0.8, "broad": 1.0}


def transferability(count_confidence: float, cls: str) -> float:
    """Low diversity caps transferability independently of count (§4.6)."""
    return min(count_confidence, TRANSFERABILITY_CAP.get(cls, 1.0))


def suggested_measurement(residual_keys: Sequence[str]) -> str:
    """The residual cluster is an experimental design, not a passive wait (§4.5)."""
    if not residual_keys:
        return ("log wider: the disagreeing observations share no recorded "
                "covariate, so the missing argument was never captured")
    return ("measure the target again while varying: " + ", ".join(sorted(residual_keys)))


def breadth_report(per_key_values: dict[str, list[str]],
                   coverage_floor: int = 3) -> dict[str, Any]:
    keys = {k: normalized_entropy(v) for k, v in per_key_values.items()
            if len(v) >= coverage_floor}
    if not keys:
        return {"breadth": 0.0, "breadth_class": "narrow", "per_key": {}}
    mean = sum(keys.values()) / len(keys)
    return {"breadth": mean, "breadth_class": breadth_class(mean), "per_key": keys}
