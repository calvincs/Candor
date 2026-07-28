"""H3: the reported point estimate must lie inside its own credible interval,
and the interval must be a genuine (non-degenerate) credible interval.

Two symptoms this pins:
  (a) FREQUENCY facts — the ~1% epistemic-zero worlds make the accepted-value
      distribution bimodal, so the mean can fall BELOW its own 5th percentile.
  (b) CRISP facts — the vote path thresholds every world to 0/1, so the CI is a
      quantile of a coin: (0,0), (1,1) or (0,1) in every configuration.
"""

from __future__ import annotations

import random

from candor.periphery import predict as P

BUDGET = 10_000


def _freq(n: int, k: int, epi=(99.0, 1.0)) -> P.Problem:
    st = P.FactState("fact:one", "frequency", epi, (1.0 + k, 1.0 + (n - k)))
    return P.Problem("fact:one", [frozenset({"fact:one"})], {"fact:one": st})


def _crisp_votes(votes, confusion) -> P.Problem:
    st = P.FactState("fact:one", "crisp", (1.0, 1.0), (1.0, 1.0), votes=votes)
    prob = P.Problem("fact:one", [frozenset({"fact:one"})], {"fact:one": st})
    prob.confusion = dict(confusion)
    return prob


def test_frequency_point_estimate_lies_inside_its_interval():
    """A random (n, k) scan: the mean must never fall outside [ci_lo, ci_hi]."""
    rng = random.Random(20260728)
    offenders = []
    for _ in range(200):
        n = rng.randint(1, 700)
        k = rng.randint(0, n)
        out = P.run(_freq(n, k), BUDGET)
        lo, hi = out.ci
        if not (lo <= out.p <= hi):
            offenders.append((n, k, out.p, lo, hi))
    assert not offenders, f"p outside CI for {len(offenders)} (n,k): {offenders[:5]}"


def test_frequency_all_successes_is_the_canonical_offender():
    # 600/600 is the reported witness: p had fallen below its own 5th percentile.
    out = P.run(_freq(600, 600), BUDGET)
    lo, hi = out.ci
    assert lo <= out.p <= hi, f"p={out.p} outside ci=({lo},{hi})"


def test_crisp_point_estimate_lies_inside_its_interval():
    confs = {a: (0, 0, 0, 0) for a in "abc"}
    cases = [
        ((("a", 1, 0, None), ("b", 1, 0, None)), confs),            # two fresh yes
        ((("a", 1, 0, None),), {"a": (0, 0, 0, 0)}),                 # one fresh yes
        ((("a", 1, 0, None), ("b", 0, 0, None), ("c", 1, 0, None)), confs),  # mixed
        ((("a", 1, 0, None),), {"a": (50, 0, 50, 0)}),              # learned sycophant
    ]
    offenders = []
    for votes, conf in cases:
        out = P.run(_crisp_votes(votes, conf), BUDGET)
        lo, hi = out.ci
        if not (lo <= out.p <= hi):
            offenders.append((votes, out.p, lo, hi))
    assert not offenders, f"crisp p outside CI: {offenders}"


def test_crisp_ci_is_not_degenerate():
    """Two fresh yes-votes must not report the coin interval (1.0, 1.0), and a
    mixed vote set must not report the zero-information interval (0.0, 1.0)."""
    two_yes = P.run(_crisp_votes(
        (("a", 1, 0, None), ("b", 1, 0, None)),
        {"a": (0, 0, 0, 0), "b": (0, 0, 0, 0)}), BUDGET)
    lo, hi = two_yes.ci
    assert lo < hi, f"two fresh yes-votes gave a degenerate CI ({lo}, {hi})"
    assert lo > 0.0, "a two-yes CI pinned at the top carries no information"

    mixed = P.run(_crisp_votes(
        (("a", 1, 0, None), ("b", 0, 0, None), ("c", 1, 0, None)),
        {a: (0, 0, 0, 0) for a in "abc"}), BUDGET)
    lo, hi = mixed.ci
    assert 0.0 < lo < hi < 1.0 + 1e-9, f"mixed vote CI is degenerate ({lo}, {hi})"
    assert lo > 0.0, "mixed vote CI still pinned to the Bernoulli quantile 0.0"


def test_interval_narrows_with_more_evidence():
    # Frequency: more trials at the same rate -> tighter interval.
    thin = P.run(_freq(10, 9), BUDGET).ci
    thick = P.run(_freq(400, 360), BUDGET).ci
    assert (thick[1] - thick[0]) < (thin[1] - thin[0]), \
        "more frequency trials must narrow the credible interval"
