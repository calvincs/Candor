"""Eponymous-law worlds — the scientific_invention_axioms.md test battery (§9).

Each generator renders one exemplar law (EX-*) as a planted outcome stream at
the level CANDOR operates on: attributed observations with context. The
learnability class (LC-*) each law was assigned predicts what the substrate
should and should not do with it — the C-16 differential profile:

  * LC-1/LC-2 structure that IS in the data → originated as a gated condition;
  * LC-3 content (behavior under intervention) → locatable after the fact,
    never predictable before it (T-1's boundary, rendered executable);
  * artifact patterns (C-18) → declined, which matters as much as the hits.

All randomness comes from a caller-supplied `random.Random`; every world is
reproducible from its seed, per the suite's rules.
"""

from __future__ import annotations

import random

from .worlds import Stream


def goodhart(n: int, cp: int, p_track: float, p_decoupled: float,
             rng: random.Random, marked: bool) -> Stream:
    """EX-7 (LC-3): metric M tracks goal G until M becomes the target, then the
    coupling collapses. `marked=True` records the intervention as a `do:` key;
    `marked=False` is the same world with nothing logged — the collapse is then
    only a date, not a named regime."""
    outcomes, contexts = [], []
    for i in range(n):
        targeted = i >= cp
        outcomes.append(rng.random() < (p_decoupled if targeted else p_track))
        contexts.append({"do:optimize_metric": "yes" if targeted else "no"}
                        if marked else {})
    return Stream(outcomes, cp, "goodhart", contexts)


def parkinson(n: int, rng: random.Random,
              p_short: float = 0.85, p_long: float = 0.3,
              nuisance: int = 3) -> Stream:
    """EX-1 (LC-2, LC-1 given the frame per C-17): outcome = task done within a
    fixed wall-clock window; the allotted deadline drives it — work expands.
    Nuisance keys ride along because real agents log plenty that explains
    nothing."""
    outcomes, contexts = [], []
    for i in range(n):
        deadline = "short" if i % 2 == 0 else "long"
        ctx = {"deadline": deadline}
        for j in range(nuisance):
            ctx[f"noise{j}"] = rng.choice(["a", "b"])
        outcomes.append(rng.random() < (p_short if deadline == "short" else p_long))
        contexts.append(ctx)
    return Stream(outcomes, None, "parkinson", contexts)


def peter(n: int, rng: random.Random,
          p_below: float = 0.85, p_ceiling: float = 0.35) -> Stream:
    """EX-5 (LC-2): performance is fine below the promotion ceiling and poor at
    it — the absorbing state of promote-on-current-performance."""
    outcomes, contexts = [], []
    for i in range(n):
        tier = rng.choice(["below_ceiling", "at_ceiling"])
        outcomes.append(rng.random() < (p_below if tier == "below_ceiling"
                                        else p_ceiling))
        contexts.append({"tier": tier})
    return Stream(outcomes, None, "peter", contexts)


def regression_artifact(n: int, p: float, rng: random.Random,
                        window: int = 3) -> Stream:
    """EX-8 / the EX-5 hazard (LC-1 negative control, C-18): a HOMOGENEOUS
    stream whose only structure is a label derived from its own noisy past —
    `self_assessment` buckets the trailing outcome mean, the way the canonical
    Dunning-Kruger quartiles bucket a noisy score. Under iid outcomes the label
    predicts nothing forward; every apparent association is regression to the
    mean. The correct behavior is to DECLINE the mechanism claim: no guard, no
    dispersion verdict."""
    outcomes, contexts = [], []
    for i in range(n):
        recent = outcomes[-window:]
        if len(recent) < window:
            label = "unrated"
        else:
            label = "high" if sum(recent) >= (window + 1) // 2 else "low"
        contexts.append({"self_assessment": label})
        outcomes.append(rng.random() < p)
    return Stream(outcomes, None, "regression artifact", contexts)
