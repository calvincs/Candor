"""H8: crisp predict() run-collapse is a bit-identical performance refactor.

`_draw`'s crisp-vote branch used to loop over every raw vote once per sampled
world (Θ(observations)×S). It now collapses maximal runs of identical votes and
folds each run's single contribution by its count. The fold is reproduced
EXACTLY (repeated IEEE rounding, not one multiply), so predict() output is
bit-for-bit unchanged; only the cost changes — flat in duplicate count, not
linear.
"""

from __future__ import annotations

import math
import random
import struct
import time

import pytest

from candor.core.committed.reliability import GAMMA, log_lr, temper
from candor.periphery import predict as P


# ── reference: the pre-refactor per-vote crisp fold, verbatim ────────────────
def _old_crisp_draw(state, s, actor_params, response_lr, discounts):
    perm_u = P.permutation(state.fact_id + "|bernoulli", s)
    unit = P._unit_grid(s)
    lr_table = response_lr or {}
    truth = []
    cont = []
    for i in range(s):
        groups: dict = {}
        sizes: dict = {}
        singles = 0.0
        for actor, vote, grade, sig in state.votes:
            if grade > 0 and (actor, vote, grade) in lr_table:
                contribution = lr_table[(actor, vote, grade)]
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
        cont.append(p)
        truth.append(1 if unit[perm_u[i]] < p else 0)
    return truth, [1.0] * s, cont


def _bits(xs):
    return [struct.unpack("<Q", struct.pack("<d", float(x)))[0] for x in xs]


def _naive_fold(acc, x, count):
    s = acc
    for _ in range(count):
        s = s + x
    return s


# ── the exact fast fold ──────────────────────────────────────────────────────
def test_fold_add_is_bit_identical_to_the_naive_fold():
    rng = random.Random(20260728)
    xs = [0.0, 0.1, 0.2, 1.0 / 3, math.log(19), -math.log(19), 1e-9, 1e9]
    # tie-hitting values (odd multiples of a power of two) stress round-half-even
    for k in range(-56, 4):
        for odd in (1, 3, 5, 7, -1, -3):
            xs.append(math.ldexp(float(odd), k))
    for x in xs:
        for acc in (0.0, 1.0, -1.0, 3.5, -100.0, math.pi):
            for c in (1, 2, 3, 4, 7, 8, 16, 17, 63, 512, 1000, 3200, 4096):
                assert _bits([P._fold_add(acc, x, c)]) == \
                    _bits([_naive_fold(acc, x, c)]), (acc, x, c)
    for _ in range(50_000):
        x = rng.uniform(-30, 30) * (10 ** rng.randint(-6, 3))
        acc = rng.uniform(-50, 50)
        c = rng.randint(1, 4000)
        assert _bits([P._fold_add(acc, x, c)]) == _bits([_naive_fold(acc, x, c)])


def test_vote_runs_collapse_contiguous_identical_votes():
    votes = (("a", 1, 0, "c1"), ("a", 1, 0, "c1"), ("a", 1, 0, "c1"),
             ("a", 0, 0, "c1"), ("a", 1, 0, "c1"))
    runs = P._vote_runs(votes)
    # first three collapse; the (a,0) breaks the run; the trailing (a,1) is a
    # SEPARATE run (order preserved, never merged with the leading one).
    assert runs == [["a", 1, 0, "c1", 3], ["a", 0, 0, "c1", 1],
                    ["a", 1, 0, "c1", 1]]


# ── bit-identical _draw across many vote configurations ──────────────────────
def _actor_params(actors, s, rng):
    return {a: ([rng.uniform(0.02, 0.98) for _ in range(s)],
                [rng.uniform(0.02, 0.98) for _ in range(s)]) for a in actors}


def _random_votes(rng):
    actors = [f"actor:{k}" for k in range(rng.randint(1, 3))]
    sigs = [None] + [f"ctx:{k}" for k in range(rng.randint(0, 2))]
    votes = []
    for _ in range(rng.randint(1, 5)):
        a = rng.choice(actors)
        v = rng.randint(0, 1)
        g = rng.choice([0, 0, 1, 2, 3])
        sig = rng.choice(sigs)
        votes += [(a, v, g, sig)] * rng.choice([1, 2, 3, 5, 40, 512, 3200])
    # interleave so identical classes are not always contiguous
    if rng.random() < 0.5:
        rng.shuffle(votes)
    # order as _fact_state does: actor, then context_sig (None first), stable
    votes.sort(key=lambda r: (r[0], "" if r[3] is None else "z" + r[3]))
    return actors, tuple(votes)


@pytest.mark.parametrize("seed", range(40))
def test_draw_output_is_bit_identical_old_vs_new(seed):
    rng = random.Random(1000 + seed)
    s = 512
    actors, votes = _random_votes(rng)
    actor_params = _actor_params(actors, s, rng)
    lr_table = {}
    for a in actors:
        for v in (0, 1):
            for g in (1, 2, 3):
                if rng.random() < 0.5:
                    lr_table[(a, v, g)] = rng.uniform(-4.0, 4.0)
    discounts = {a: rng.uniform(0.0, 1.0) for a in actors} if rng.random() < 0.4 \
        else {}
    state = P.FactState("fact:one", "crisp", (99.0, 1.0), (1.0, 1.0), votes=votes)

    old_t, old_th, old_c = _old_crisp_draw(state, s, actor_params, lr_table,
                                           discounts)
    new = P._draw(state, s, actor_params, lr_table, discounts)

    assert list(new.truth) == old_t
    assert _bits(new.theta) == _bits(old_th)
    assert _bits(new.cont) == _bits(old_c)          # every rounded bit unchanged


# ── flat scaling in the number of duplicate observations ─────────────────────
def _dup_state(n):
    votes = tuple(("actor:x", 1, 0, "ctx:z") for _ in range(n))
    return P.FactState("fact:one", "crisp", (99.0, 1.0), (1.0, 1.0), votes=votes)


def _time_draw(state, actor_params, reps=5):
    P._draw(state, 512, actor_params, {}, {})            # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        P._draw(state, 512, actor_params, {}, {})
    return (time.perf_counter() - t0) / reps


def test_crisp_draw_latency_is_flat_in_duplicate_count():
    ap = {"actor:x": ([0.9] * 512, [0.1] * 512)}
    small = _time_draw(_dup_state(8), ap)
    big = _time_draw(_dup_state(3200), ap)
    # 3200 identical observations is 400x the data of 8; a per-vote loop would
    # cost ~400x. Collapsed to one run it is within a small constant of the
    # 8-vote case. Generous bound to stay non-flaky, but far below linear (400x).
    assert big < small * 8, f"small={small*1e3:.2f}ms big={big*1e3:.2f}ms"
