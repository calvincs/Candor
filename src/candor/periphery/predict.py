"""Prediction engine: model counting over literal masses, under an epistemic loop.

Spec §3.9. Two-loop sampling settles channel composition (I7): facts are drawn
once per epistemic world, so epistemic uncertainty correlates across queries
sharing a fact and aleatoric does not — by construction, not by bookkeeping.

Sampling is *stratified and deterministic*, not pseudo-random: each fact's
draws are the S inverse-CDF quantiles at (i+0.5)/S, dealt out through a
permutation derived from the fact's own identity. Consequences that matter:

  * the same state always yields the same number, so a model snapshot can be
    re-run bit-for-bit (I8);
  * the result cannot depend on the order facts were inserted in (§6.2
    permutation);
  * adding support moves every draw in the same direction, so monotonicity is
    a property of the estimator rather than a hope about Monte Carlo noise.

Budget degrades S first, then exact → top-k proof bounds. Degrade, never hang.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from ..core.betamath import beta_mean, beta_quantile_grid
from ..core.committed.reliability import (FPR_PRIOR, GAMMA, SENS_PRIOR,
                                          log_lr, temper)
from ..core.hashing import stable_u64

DEFAULT_SAMPLES = 512
MIN_SAMPLES = 64
CI_LO, CI_HI = 0.05, 0.95
MAX_EXACT_CONJUNCTS = 12          # inclusion–exclusion budget; above this, top-k


@dataclass(frozen=True)
class FactState:
    fact_id: str
    stmt_type: str
    epi: tuple[float, float]
    alea: tuple[float, float]
    pinned_negative: bool = False
    flagged: bool = False
    narrow: bool = False
    # Δ1/Δ6: attributed votes on a crisp fact, (actor, vote, grade, context_sig).
    # When present they supersede the epi Beta — same events at finer grain.
    votes: tuple[tuple[str, int, int, Optional[str]], ...] = ()


@dataclass
class Problem:
    goal_id: Optional[str]
    dnf: list[frozenset[str]] = field(default_factory=list)
    facts: dict[str, FactState] = field(default_factory=dict)
    constraint_groups: list[list[str]] = field(default_factory=list)
    caveats: set[str] = field(default_factory=set)
    # v0.3 Δ1: two-coin confusion cells per actor appearing in any vote.
    confusion: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    # v0.4 Δ6: mean log-LR per graded response (actor, vote, grade), read-time.
    response_lr: dict[tuple[str, int, int], float] = field(default_factory=dict)
    # Operator-set trust discounts (set_reliability). Absent = 1.0 = untempered,
    # so a store with no overrides composes exactly as it did before.
    discounts: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Outcome:
    p: float
    ci: tuple[float, float]
    channels: dict[str, float]
    sensitivity: dict[str, float]
    mpe: dict[str, int]
    caveats: frozenset[str]
    rejection_rate: float
    samples_used: int
    exact: bool


# ── deterministic dealing ───────────────────────────────────────────────────

def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)


def permutation(tag: str, n: int) -> list[int]:
    """Fisher–Yates driven by splitmix64 seeded from `tag`. Version-independent."""
    order = list(range(n))
    state = stable_u64(tag)
    for i in range(n - 1, 0, -1):
        state, r = _splitmix64(state)
        j = r % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


_UNIT_GRID_CACHE: dict[int, list[float]] = {}


def _unit_grid(s: int) -> list[float]:
    grid = _UNIT_GRID_CACHE.get(s)
    if grid is None:
        grid = [(i + 0.5) / s for i in range(s)]
        _UNIT_GRID_CACHE[s] = grid
    return grid


@dataclass
class _Draws:
    truth: list[int]
    theta: list[float]


def _actor_param_grids(problem: Problem, s: int) -> dict[str, tuple[list[float],
                                                                    list[float]]]:
    """Per-world (sens, fpr) samples for every actor with a vote in the problem.

    Sampled ONCE per epistemic world and shared across every fact the actor
    observed (I7): observer-parameter uncertainty correlates predictions
    through shared observers exactly as fact uncertainty correlates them
    through shared facts. Stratified and permuted by actor identity, so the
    draw is deterministic and independent of fact insertion order.
    """
    actors = sorted({actor for st in problem.facts.values()
                     for actor, _, _, _ in st.votes})
    out: dict[str, tuple[list[float], list[float]]] = {}
    for actor in actors:
        tp, fn, fp, tn = problem.confusion.get(actor, (0, 0, 0, 0))
        sens_grid = beta_quantile_grid(SENS_PRIOR[0] + tp, SENS_PRIOR[1] + fn, s)
        fpr_grid = beta_quantile_grid(FPR_PRIOR[0] + fp, FPR_PRIOR[1] + tn, s)
        perm_s = permutation(actor + "|sens", s)
        perm_f = permutation(actor + "|fpr", s)
        out[actor] = ([sens_grid[perm_s[i]] for i in range(s)],
                      [fpr_grid[perm_f[i]] for i in range(s)])
    return out


def _draw(state: FactState, s: int,
          actor_params: Optional[dict[str, tuple[list[float], list[float]]]] = None,
          response_lr: Optional[dict] = None,
          discounts: Optional[dict[str, float]] = None) -> _Draws:
    if state.pinned_negative:
        # A '-' pin is the only hard zero in the system (I5).
        return _Draws([0] * s, [0.0] * s)
    perm_u = permutation(state.fact_id + "|bernoulli", s)
    unit = _unit_grid(s)
    if state.stmt_type == "crisp" and state.votes and actor_params:
        # v0.3 Δ1/Δ2: validity via two-coin log-LR composition under the
        # world's sampled actor parameters, sub-additive within context groups.
        lr_table = response_lr or {}
        truth = []
        for i in range(s):
            groups: dict[str, float] = {}
            sizes: dict[str, int] = {}
            singles = 0.0
            for actor, vote, grade, sig in state.votes:
                if grade > 0 and (actor, vote, grade) in lr_table:
                    contribution = lr_table[(actor, vote, grade)]   # Δ6 mean LR
                else:
                    contribution = log_lr(actor_params[actor][0][i],
                                          actor_params[actor][1][i], bool(vote))
                if discounts:
                    contribution = temper(contribution, discounts.get(actor, 1.0))
                if sig is None:
                    singles += contribution
                else:
                    groups[sig] = groups.get(sig, 0.0) + contribution
                    sizes[sig] = sizes.get(sig, 0) + 1
            logodds = singles + sum(sub / (sizes[g] ** GAMMA)
                                    for g, sub in groups.items())
            p = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logodds))))
            truth.append(1 if unit[perm_u[i]] < p else 0)
        return _Draws(truth, [1.0] * s)
    epi_grid = beta_quantile_grid(state.epi[0], state.epi[1], s)
    perm_v = permutation(state.fact_id + "|validity", s)
    truth = [1 if unit[perm_u[i]] < epi_grid[perm_v[i]] else 0 for i in range(s)]
    if state.stmt_type == "frequency":
        alea_grid = beta_quantile_grid(state.alea[0], state.alea[1], s)
        perm_t = permutation(state.fact_id + "|rate", s)
        theta = [alea_grid[perm_t[i]] for i in range(s)]
    else:
        theta = [1.0] * s
    return _Draws(truth, theta)


# ── model counting over the proof DNF, under per-literal masses ─────────────

def _wmc(dnf: Sequence[frozenset[str]], mass: dict[str, float]) -> float:
    """Proofs share facts, so probabilities cannot be summed (§3.9)."""
    if not dnf:
        return 0.0
    if len(dnf) == 1:
        out = 1.0
        for fid in dnf[0]:
            out *= mass.get(fid, 0.0)
        return out
    if len(dnf) <= MAX_EXACT_CONJUNCTS:
        total = 0.0
        for bits in range(1, 1 << len(dnf)):
            union: set[str] = set()
            popcount = 0
            for i in range(len(dnf)):
                if bits >> i & 1:
                    union |= dnf[i]
                    popcount += 1
            term = 1.0
            for fid in union:
                term *= mass.get(fid, 0.0)
            total += term if popcount % 2 else -term
        return total
    # Above budget: top-k proof bound (§3.9 degradation path).
    ranked = sorted(dnf, key=lambda c: -_conjunct_mass(c, mass))
    bound = 0.0
    for conj in ranked[:MAX_EXACT_CONJUNCTS]:
        bound = bound + _conjunct_mass(conj, mass) - bound * _conjunct_mass(conj, mass)
    return bound


def _conjunct_mass(conj: Iterable[str], mass: dict[str, float]) -> float:
    out = 1.0
    for fid in conj:
        out *= mass.get(fid, 0.0)
    return out


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(len(sorted_values) - 1, lo + 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def run(problem: Problem, budget: int) -> Outcome:
    """Two-loop sampled WMC. Deterministic given the problem."""
    s = DEFAULT_SAMPLES
    if budget < 1000:
        s = max(MIN_SAMPLES, min(DEFAULT_SAMPLES, budget))
    s = max(MIN_SAMPLES, s)

    actor_params = _actor_param_grids(problem, s)
    draws = {fid: _draw(st, s, actor_params, problem.response_lr, problem.discounts)
             for fid, st in problem.facts.items()}
    groups = [[f for f in g if f in draws] for g in problem.constraint_groups]
    groups = [g for g in groups if len(g) > 1]

    accepted: list[float] = []
    aleatoric_only: list[float] = []
    rejected = 0
    for i in range(s):
        violated = False
        for group in groups:
            if sum(draws[f].truth[i] for f in group) > 1:
                violated = True
                break
        if violated:
            rejected += 1
            continue
        mass = {fid: draws[fid].truth[i] * draws[fid].theta[i] for fid in draws}
        accepted.append(_wmc(problem.dnf, mass))
        pure_theta = {fid: draws[fid].theta[i] for fid in draws}
        aleatoric_only.append(_wmc(problem.dnf, pure_theta))

    n_acc = len(accepted)
    if n_acc == 0:
        # Every sampled world violates a constraint: the epistemic posteriors
        # are in total tension with the admitted constraints. Report it.
        return Outcome(0.0, (0.0, 0.0), {"epistemic": 0.0, "aleatoric": 0.0},
                       {}, {}, frozenset(problem.caveats | {"constraint_tension"}),
                       1.0, s, True)

    p = sum(accepted) / n_acc
    ordered = sorted(accepted)
    ci = (_quantile(ordered, CI_LO), _quantile(ordered, CI_HI))
    mean_sq = sum(x * x for x in accepted) / n_acc
    epistemic = math.sqrt(max(0.0, mean_sq - p * p))
    has_frequency = any(st.stmt_type == "frequency" for st in problem.facts.values())
    aleatoric = (sum(aleatoric_only) / n_acc) if has_frequency else 0.0

    sensitivity = _sensitivity(problem, draws, groups, s, p)
    mpe = _mpe(problem)
    return Outcome(p, ci, {"epistemic": epistemic, "aleatoric": aleatoric},
                   sensitivity, mpe, frozenset(problem.caveats),
                   rejected / s, s, len(problem.dnf) <= MAX_EXACT_CONJUNCTS)


def _sensitivity(problem: Problem, draws: dict[str, _Draws],
                 groups: list[list[str]], s: int, base_p: float) -> dict[str, float]:
    """Which fact, if flipped, changes the conclusion (§3.9)."""
    support = sorted({fid for conj in problem.dnf for fid in conj})
    out: dict[str, float] = {}
    for target in support:
        highs: list[float] = []
        lows: list[float] = []
        for i in range(s):
            if any(sum(draws[f].truth[i] for f in g) > 1 for g in groups):
                continue
            mass = {fid: draws[fid].truth[i] * draws[fid].theta[i] for fid in draws}
            mass[target] = draws[target].theta[i]
            highs.append(_wmc(problem.dnf, mass))
            mass[target] = 0.0
            lows.append(_wmc(problem.dnf, mass))
        if highs:
            out[target] = abs(sum(highs) / len(highs) - sum(lows) / len(lows))
        else:
            out[target] = 0.0
    return out


def _mpe(problem: Problem) -> dict[str, int]:
    """Most probable explanation, constraint-respecting."""
    assign = {fid: (0 if st.pinned_negative else
                    (1 if beta_mean(*st.epi) >= 0.5 else 0))
              for fid, st in problem.facts.items()}
    for group in problem.constraint_groups:
        members = [f for f in group if assign.get(f)]
        if len(members) > 1:
            best = max(members, key=lambda f: beta_mean(*problem.facts[f].epi))
            for f in members:
                if f != best:
                    assign[f] = 0
    return assign
