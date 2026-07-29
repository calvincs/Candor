"""M7: the isotonic fit must be a function of the input MULTISET, not its order.

`sorted()` is stable, so tied predicted-p values keep their input order and PAVA
pools them differently depending on it — and predict() emits multiples of 1/512,
so ties are the norm, not the exception. The snapshot hash and the calibrated p
must not depend on the order settled claims happened to be enumerated in.
"""

from __future__ import annotations

import random

import pytest

from candor.core import calibration as C


def _tied_pairs():
    # Ties on x (multiples of 1/512) with mixed y — exactly what settled claims
    # look like when many predictions land in the same 1/512 bucket.
    xs = [round(i / 512, 6) for i in (100, 100, 100, 200, 200, 300, 300, 300,
                                      400, 400, 400, 400)]
    ys = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    return list(zip(xs, ys))


def test_isotonic_fit_is_order_independent_on_tied_inputs():
    base = _tied_pairs()
    m0 = C.fit_isotonic(base)
    grid = [i / 40 for i in range(41)]
    ref_apply = [m0.apply(g) for g in grid]

    rng = random.Random(20260728)
    for _ in range(12):
        shuffled = base[:]
        rng.shuffle(shuffled)
        m = C.fit_isotonic(shuffled)
        assert m.hash == m0.hash, "snapshot hash depends on training-set order"
        assert [m.apply(g) for g in grid] == ref_apply, \
            "calibrated p depends on training-set order"


def test_fit_stays_monotone_after_pooling_ties():
    m = C.fit_isotonic(_tied_pairs())
    assert list(m.xs) == sorted(m.xs)
    assert list(m.ys) == sorted(m.ys), "PAVA output is not monotone after pooling"
    # Pooling ties must collapse each distinct x to a single knot.
    assert len(m.xs) == len(set(m.xs)), "tied x survived into the fitted map"


def test_pooling_ties_matches_a_pre_pooled_fit():
    """Fitting the tied set must equal fitting the already-pooled set: each
    distinct x carried in with its mean y (the multiset, order-free)."""
    base = _tied_pairs()
    pooled: dict[float, list[int]] = {}
    for x, y in base:
        pooled.setdefault(x, []).append(y)
    prepooled = [(x, sum(ys) / len(ys)) for x, ys in sorted(pooled.items())]
    # fit_isotonic takes (float, int) pairs; feed the pooled means through the
    # same PAVA by hand-checking apply equality on a grid.
    m_raw = C.fit_isotonic(base)
    # A hand PAVA over the pre-pooled block means, weighted by block size:
    blocks = [[y * len(pooled[x]), float(len(pooled[x])), x * len(pooled[x])]
              for x, y in prepooled]
    i = 1
    while i < len(blocks):
        if blocks[i - 1][0] / blocks[i - 1][1] > blocks[i][0] / blocks[i][1]:
            a, b = blocks[i - 1], blocks.pop(i)
            blocks[i - 1] = [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
            i = max(1, i - 1)
        else:
            i += 1
    xs = tuple(bx / bn for _, bn, bx in blocks)
    ys = tuple(by / bn for by, bn, _ in blocks)
    expected = C.IsotonicMap(xs, ys)
    grid = [i / 40 for i in range(41)]
    assert [m_raw.apply(g) for g in grid] == pytest.approx(
        [expected.apply(g) for g in grid])


# ── bonus: calibration is wired into predict but never exercised non-trivially ──

def test_non_identity_calibration_moves_predict_and_snapshot(seeded):
    """Every existing test runs with the identity map, so predict()'s
    calibration plumbing is wired but unverified. Install a real map and prove
    both p and the snapshot's calib_map_hash follow it."""
    from candor.core import calibration as C

    stmt = {"pred": "flaky_link", "args": ["c", "d"]}
    raw = seeded.predict(stmt, budget=10_000)
    p_raw = raw.p
    assert p_raw > 0.0

    # A genuine, monotone, non-identity recalibration: halve every probability.
    calib = C.IsotonicMap(xs=(0.0, 1.0), ys=(0.0, 0.5))
    assert calib.hash != C.IsotonicMap().hash
    seeded._calib = calib

    cal = seeded.predict(stmt, budget=10_000)
    assert cal.p == pytest.approx(calib.apply(p_raw))
    assert cal.p != pytest.approx(p_raw), "calibration did not move the estimate"
    carried = C.parse_snapshot(cal.snapshot_id)["calib_map_hash"]
    assert carried == calib.hash, "snapshot_id did not carry the new calib map hash"
