"""Beta distribution numerics, stdlib only. Trusted (no external numeric deps).

Provides the regularized incomplete beta function and its inverse, which the
prediction engine needs to draw stratified (deterministic) epistemic samples
without pulling in scipy.
"""

from __future__ import annotations

import math

_EPS = 3.0e-16
_FPMIN = 1e-300


def log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Lentz's method, NR §6.4)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, 301):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) = P(X <= x) for X ~ Beta(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - log_beta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_pdf(a: float, b: float, x: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return math.exp((a - 1.0) * math.log(x) + (b - 1.0) * math.log1p(-x) - log_beta(a, b))


def betaincinv(a: float, b: float, p: float, guess: float | None = None) -> float:
    """Inverse of :func:`betainc`. Safeguarded Newton inside a bisection bracket.

    Deterministic and monotone in `p`, which is what makes the prediction
    engine's stratified sampling reproducible bit-for-bit.
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    x = guess if guess is not None and 0.0 < guess < 1.0 else a / (a + b)
    for _ in range(200):
        err = betainc(a, b, x) - p
        if err > 0.0:
            hi = x
        else:
            lo = x
        if abs(err) < 1e-14:
            break
        pdf = beta_pdf(a, b, x)
        nxt = x - err / pdf if pdf > 1e-300 else 0.5 * (lo + hi)
        if not (lo < nxt < hi):
            nxt = 0.5 * (lo + hi)
        if abs(nxt - x) < 1e-16:
            x = nxt
            break
        x = nxt
    return x


def beta_quantile_grid(a: float, b: float, s: int) -> list[float]:
    """Stratified quantiles at probabilities (i + 0.5) / s, ascending.

    The grid is swept upward with the previous solution as the Newton seed, so
    the whole grid costs little more than a handful of `betainc` evaluations
    per point. Ascending order is exploited by callers: the grid is monotone in
    the success count, which is what makes prediction monotone in support.
    """
    out: list[float] = []
    prev = None
    for i in range(s):
        prob = (i + 0.5) / s
        prev = betaincinv(a, b, prob, guess=prev)
        out.append(prev)
    return out


def beta_mean(a: float, b: float) -> float:
    return a / (a + b)


def beta_var(a: float, b: float) -> float:
    t = a + b
    return (a * b) / (t * t * (t + 1.0))
