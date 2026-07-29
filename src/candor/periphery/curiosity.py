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
import random
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

# §4.5 gate-hard defaults, because covariate search is a fishing expedition.
MIN_SUPPORT_PER_PARTITION = 8
TARONE_Z_THRESHOLD = 3.0
CUSUM_THRESHOLD = 5.0
CUSUM_DRIFT = 0.5
BH_ALPHA = 0.05
# §4.4 changepoint significance, after correcting for the searched split point.
# Deliberately tighter than BH_ALPHA: a supersede closes a fact's validity
# window, so a false one rewrites history, while a false guard is only a
# rejected candidate.
CHANGEPOINT_ALPHA = 0.01


@dataclass(frozen=True)
class Group:
    n: int
    k: int


def tarone_z(groups: Sequence[Group]) -> Optional[float]:
    """Tarone's Z for binomial overdispersion. None when it is undefined.

        Z = [ Σ (k_i - n_i p)² / (p(1-p))  -  N ] / sqrt( 2 Σ n_i(n_i - 1) )

    The denominator is the null standard deviation of the chi-square term and
    carries no p. An earlier form here multiplied it by p/(1-p), which made the
    statistic wildly base-rate-dependent in the direction that matters least:
    measured false-positive rate on stationary data was 38% at p=0.05 and 0% at
    p=0.9, against a nominal 5%. Corrected it is flat — 3-8% across p ∈ [0.05,
    0.95] — and *more* powerful on real structure, so nothing was traded for it.
    """
    usable = [g for g in groups if g.n > 0]
    if len(usable) < 2:
        return None
    total_n = sum(g.n for g in usable)
    total_k = sum(g.k for g in usable)
    p = total_k / total_n
    if p <= 0.0 or p >= 1.0:
        return None
    chi = sum((g.k - g.n * p) ** 2 for g in usable) / (p * (1.0 - p))
    denom = math.sqrt(2.0 * sum(g.n * (g.n - 1) for g in usable))
    if denom <= 0.0:
        return None
    return (chi - total_n) / denom


#: Draws for the deterministic parametric null of Tarone's Z. Enough to resolve
#: the p-value at the BH / temporal thresholds it feeds (a few ×10⁻³). The null
#: is only simulated for keys that clear the cheap normal screen below, so the
#: sweep pays for it on the handful of promising covariates, not every one.
TARONE_BOOTSTRAP = 2000

#: Below this Z the normal tail already puts the p-value far above every
#: threshold any caller compares against (≈0.16 at Z=1), so the exact null cannot
#: change a decision and is not simulated.
_TARONE_SCREEN_Z = 1.0


def _null_seed(groups: Sequence[Group]) -> int:
    """A deterministic, order-independent seed for the parametric null, so a
    replay of the same observations reproduces the p-value bit-for-bit."""
    seed = 0x9E3779B97F4A7C15
    mask = (1 << 64) - 1
    for n, k in sorted((g.n, g.k) for g in groups):
        seed = ((seed ^ n) * 0x100000001B3) & mask
        seed = ((seed ^ k) * 0x100000001B3) & mask
    return seed


def tarone_pvalue(groups: Sequence[Group]) -> Optional[float]:
    """Upper-tail p-value for Tarone's overdispersion Z, calibrated in the far
    tail. None exactly when tarone_z is None.

    The Z statistic is standardized to mean 0 / unit variance, but its null
    distribution is a right-skewed sum of squared binomial deviations, not a
    Gaussian: reading the tail off ``0.5*erfc(z/sqrt2)`` is anti-conservative —
    measured ~3-5x at p≈1e-3 and ~1.6x at alpha/6, exactly where a BH rank-1
    threshold across several covariate keys bites and inflates the per-stream
    false-guard rate. So above a cheap normal screen the tail is read instead off
    the statistic's OWN null distribution, simulated in place: draw the group
    counts from the fitted homogeneous binomial (each nᵢ, at the pooled rate) and
    count how often a null Z reaches the observed one.

    A moment-matched gamma was tried first and rejected: matching only mean and
    variance over-corrects (measured ~0.5x nominal at alpha/6) and drops a real
    but weak covariate below the detection bar, whereas this parametric null
    lands near nominal and leaves detection intact. Its RNG is seeded from the
    observed group counts, so the sweep stays replay-stable — no unseeded
    randomness — and the simulation runs only past the screen, so it does not
    slow the sweep materially.
    """
    usable = [g for g in groups if g.n > 0]
    if len(usable) < 2:
        return None
    total_n = sum(g.n for g in usable)
    total_k = sum(g.k for g in usable)
    p = total_k / total_n
    if p <= 0.0 or p >= 1.0:
        return None
    z = tarone_z(usable)
    if z is None:
        return None
    if z <= _TARONE_SCREEN_Z:
        return 0.5 * math.erfc(z / math.sqrt(2)) if z > 0.0 else 1.0
    ns = [g.n for g in usable]
    rng = random.Random(_null_seed(usable))
    at_least = 1                                     # +1: the observed table itself
    for _ in range(TARONE_BOOTSTRAP):
        zsim = tarone_z([Group(n, rng.binomialvariate(n, p)) for n in ns])
        if zsim is not None and zsim >= z:
            at_least += 1
    return at_least / (TARONE_BOOTSTRAP + 1)


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


#: The bucket name for observations that did NOT record a given context key — the
#: honest "cannot attribute this to any value of the key" mass. A dunder so it can
#: never collide with a real context value (values are plain strings).
RESIDUAL_BUCKET = "__residual__"


def outcome_breakdown(observations: Iterable[tuple[dict[str, str], bool]],
                      key: str) -> dict[str, Group]:
    """Per-value (n, k) tally for ONE recorded context key, plus a residual bucket.

    Extends `partition_by_key` with a ``__residual__`` group: the observations
    that did not record `key` at all — mass no value of the key can account for.
    Deterministic and order-independent (integer count folds), so a replay of the
    same observation set reproduces every (n, k) exactly.

    General in the tallied *outcome*: it counts n and k where k sums a truthy
    outcome, so a future categorical fact type reuses the identical shape to break
    a chosen value's incidence down by context — only the meaning of "truthy"
    changes, decided by the caller, not this helper. (No categorical support is
    built here; the helper simply does not hard-block it.)
    """
    obs = list(observations)
    groups = dict(partition_by_key(obs, key))
    res_n = res_k = 0
    for ctx, outcome in obs:
        if key not in ctx:
            res_n += 1
            res_k += 1 if outcome else 0
    if res_n:
        groups[RESIDUAL_BUCKET] = Group(res_n, res_k)
    return groups


def explained_fraction(groups: Iterable[Group]) -> float:
    """Correlation ratio η² — the share of a binary outcome's total variance a
    categorical partition accounts for: between-group variance / total variance.

    A simple, standard, deterministic decomposition over the SAME Group(n, k)
    partitions the dispersion machinery already builds — NOT a new detector. It is
    0 when a partition separates the outcome no better than chance (every group at
    the pooled rate) and approaches 1 as the groups become internally homogeneous,
    so `1 - explained_fraction` is the honest "no recorded variable accounts for
    this" residual. A degenerate outcome (all-0 or all-1, zero total variance)
    yields 0. Order-independent up to float reassociation; pass groups in a stable
    order (e.g. sorted by value) for a bit-reproducible result.
    """
    usable = [g for g in groups if g.n > 0]
    total_n = sum(g.n for g in usable)
    if total_n == 0:
        return 0.0
    p = sum(g.k for g in usable) / total_n
    total_var = p * (1.0 - p)
    if total_var <= 0.0:
        return 0.0
    between = sum(g.n * ((g.k / g.n) - p) ** 2 for g in usable) / total_n
    return min(1.0, between / total_var)


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


def _lchoose(n: int, k: int) -> float:
    if k < 0 or k > n or n < 0:
        return float("-inf")
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def fisher_exact(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p for a success/failure table over two segments.

    Exact at every base rate, which is the whole reason it is here. A CUSUM
    normalised by sqrt(p(1-p)) is a Gaussian approximation, and Bernoulli
    increments are badly skewed near 0 and 1: the old changepoint machinery
    false-alarmed on 40% of *stationary* p=0.95 segments and 43% at p=0.05,
    against 3% at p=0.5 — worst precisely in the broken-tool regime a memory
    substrate exists to notice. Computed in log space so the binomials stay
    cheap on long series.
    """
    if n1 <= 0 or n2 <= 0:
        return 1.0
    total_n, total_k = n1 + n2, k1 + k2
    denom = _lchoose(total_n, total_k)
    if denom == float("-inf"):
        return 1.0
    observed = _lchoose(n1, k1) + _lchoose(n2, total_k - k1) - denom
    acc = 0.0
    for x in range(max(0, total_k - n2), min(n1, total_k) + 1):
        lp = _lchoose(n1, x) + _lchoose(n2, total_k - x) - denom
        if lp <= observed + 1e-9:
            acc += math.exp(lp)
    return min(1.0, acc)


def locate_changepoint(series: Sequence[bool]) -> int:
    """Index of the last observation before the biggest shift in level.

    argmax of |cumulative deviation from the series mean|. This part was always
    good — median localization error is 1 observation in 120 — so it is kept
    verbatim and only the *significance* decision around it was replaced.
    """
    n = len(series)
    mean = sum(1 for x in series if x) / n
    running, peak, located = 0.0, -1.0, 0
    for i, x in enumerate(series):
        running += (1.0 if x else 0.0) - mean
        if abs(running) > peak:
            peak, located = abs(running), i
    return located


def changepoint_test(series: Sequence[bool],
                     alpha: float = CHANGEPOINT_ALPHA
                     ) -> Optional[tuple[int, float]]:
    """(index, corrected p) for a step change in level, or None.

    The split point is chosen by maximising over every admissible position, so
    the naive p-value is anti-conservative by exactly that search. Correcting
    for the number of positions searched is what holds the false-positive rate
    near alpha at any base rate.
    """
    n = len(series)
    if n < 2 * MIN_SUPPORT_PER_PARTITION:
        return None
    idx = locate_changepoint(series)
    n1 = idx + 1
    n2 = n - n1
    if min(n1, n2) < MIN_SUPPORT_PER_PARTITION:
        return None
    k1 = sum(1 for x in series[:n1] if x)
    k2 = sum(1 for x in series[n1:] if x)
    searched = max(1, n - 2 * MIN_SUPPORT_PER_PARTITION + 1)
    p = min(1.0, fisher_exact(k1, n1, k2, n2) * searched)
    return (idx, p) if p <= alpha else None


def is_recurrent(series: Sequence[bool],
                 alpha: float = CHANGEPOINT_ALPHA) -> bool:
    """Does either side of the located split change AGAIN?

    A one-way regime change leaves two internally homogeneous segments; a
    flapping service or an outage that recovered does not. §4.4 routes the
    second kind to a condition or a question, never to a supersede — so this
    check decides whether a detection is a *date* or a *symptom*, and it has to
    be right at extreme base rates or it eats the true detections.
    """
    idx = locate_changepoint(series)
    return (changepoint_test(series[:idx + 1], alpha) is not None
            or changepoint_test(series[idx + 1:], alpha) is not None)


#: Block sizes for the temporal dispersion test. A flapping service's period is
#: unknown, so several scales are tried and the search is paid for.
TIME_BLOCK_SCALES = (8, 16, 32)


def time_blocks(series: Sequence[bool], size: int) -> list[Group]:
    """Contiguous, equal-length blocks of the observation series."""
    return [Group(len(chunk), sum(1 for x in chunk if x))
            for chunk in (series[i:i + size]
                          for i in range(0, len(series) - size + 1, size))]


def temporal_dispersion(series: Sequence[bool], alpha: float = BH_ALPHA
                        ) -> Optional[tuple[float, int, list[Group]]]:
    """Instability on the TIME axis that no single date and no covariate explains.

    Returns (corrected p, block_size, blocks) for the most extreme stretch, or
    None if the series is consistent with one stable rate.

    This is the detector behind honest confusion. Without it, a stream that
    swings between 85% and 35% produces *no signal whatsoever* when the agent
    logged no context correlated with the cause — and logging nothing relevant
    is the normal case, not the exotic one. The variance is visible in the
    series itself; it needs no covariate to exist.

    An omnibus overdispersion test across the blocks, not a per-block one: with
    an alternating pattern every individual block sits close to the global rate
    while the *spread* is enormous, so pooling across blocks is what carries the
    power (0.99 vs 0.53 for a per-block Bonferroni on the same worlds). The
    period is unknown, so several scales are tried and the search is paid for.
    """
    scales = [s for s in TIME_BLOCK_SCALES if len(series) >= 3 * s]
    if not scales:
        return None
    best: Optional[tuple[float, int, list[Group]]] = None
    for size in scales:
        blocks = time_blocks(series, size)
        z = tarone_z(blocks)
        if z is None or z <= 0.0:
            continue
        p = tarone_pvalue(blocks)
        if p is None:
            continue
        pvalue = min(1.0, p * len(scales))
        if pvalue <= alpha and (best is None or pvalue < best[0]):
            best = (pvalue, size, blocks)
    return best


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
