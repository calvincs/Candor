"""Scoring rules and detector statistics for the claims suite.

Kept dependency-free and separate from the worlds so a new claim test can reuse
the measurement without reusing the scenario.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

Pairs = Sequence[tuple[float, bool]]


def brier(pairs: Pairs) -> float:
    return sum((p - t) ** 2 for p, t in pairs) / len(pairs)


def log_loss(pairs: Pairs, eps: float = 1e-6) -> float:
    return sum(-(math.log(max(p, eps)) if t else math.log(max(1 - p, eps)))
               for p, t in pairs) / len(pairs)


def ece(pairs: Pairs, bins: int = 10) -> float:
    """Expected calibration error over equal-width bins."""
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        chunk = [(p, t) for p, t in pairs
                 if (lo <= p < hi or (b == bins - 1 and p == 1.0))]
        if not chunk:
            continue
        mp = sum(p for p, _ in chunk) / len(chunk)
        obs = sum(t for _, t in chunk) / len(chunk)
        total += len(chunk) / len(pairs) * abs(mp - obs)
    return total


def calibration_slope(pairs: Pairs) -> float:
    """1.0 = honest. <1 = overconfident, >1 = underconfident."""
    mp = sum(p for p, _ in pairs) / len(pairs)
    mt = sum(t for _, t in pairs) / len(pairs)
    den = sum((p - mp) ** 2 for p, _ in pairs)
    if den == 0:
        return float("nan")
    return sum((p - mp) * (t - mt) for p, t in pairs) / den


def oracle_nb(votes: dict, judges) -> float:
    """Naive Bayes using the TRUE planted rates — the ceiling for any method
    that treats conditionally-independent judges independently."""
    by_name = {j.name: j for j in judges}
    lo = 0.0
    for name, v in votes.items():
        j = by_name[name]
        s = min(max(j.sens, 1e-9), 1 - 1e-9)
        f = min(max(j.fpr, 1e-9), 1 - 1e-9)
        lo += math.log(s / f) if v else math.log((1 - s) / (1 - f))
    return 1.0 / (1.0 + math.exp(-lo))


@dataclass(frozen=True)
class DetectorScore:
    """Detection outcome over many independent replications of one world."""
    n: int
    fires: int
    localization: list[int]

    @property
    def rate(self) -> float:
        return self.fires / self.n if self.n else 0.0

    @property
    def median_error(self) -> Optional[float]:
        if not self.localization:
            return None
        s = sorted(self.localization)
        return float(s[len(s) // 2])

    @property
    def p90_error(self) -> Optional[float]:
        if not self.localization:
            return None
        s = sorted(self.localization)
        return float(s[int(0.9 * (len(s) - 1))])

    def __str__(self) -> str:
        loc = ("" if self.median_error is None
               else f", median |err| {self.median_error:.0f}, "
                    f"p90 {self.p90_error:.0f}")
        return f"{self.fires}/{self.n} = {self.rate:.2f}{loc}"
