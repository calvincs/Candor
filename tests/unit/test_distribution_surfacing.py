"""Feature A — distribution surfacing: a read-time per-context outcome breakdown
plus an explicit unexplained "unknown" residual, over data the store already keeps.

The whole feature is a PURE projection: it moves no count, changes no closure_hash
and never runs predict(), so predict()'s scalar output is byte-identical whether or
not `distribution()` is ever called. These tests prove the breakdown is correct, the
unknown residual is sensible (near-total with no explaining guard, reduced when a
guard is admitted), the projection is deterministic across replay and the checkpoint
fast path, and predict()/closure_hash are untouched (additivity).
"""

from __future__ import annotations

import json

from candor.periphery import curiosity as C
from candor.system import CandorSystem

BUDGET = 10_000
CRISP = {"pred": "link_ok", "args": ["a", "b"], "stmt_type": "crisp"}


def _admit(m, stmt):
    m.assert_(stmt, source="seed", actor="human:calvin")
    m.run_gate()
    return m.fact_id_for(stmt)


def _plant_guarded(m, stmt):
    """region is genuinely predictive: us-east ~0.9 true, eu ~0.28 true, plus 17
    observations that record no region at all (the honest 'cannot attribute' mass)."""
    obs = ([(stmt, i < 36, {"region": "us-east"}, "tool:probe") for i in range(40)]
           + [(stmt, i < 7, {"region": "eu"}, "tool:probe") for i in range(25)]
           + [(stmt, i % 2 == 0, {}, "tool:probe") for i in range(17)])
    import random
    random.Random(7).shuffle(obs)
    m.observe_batch(obs)
    m.run_gate()


def _plant_unexplained(m, stmt):
    """Alternating outcomes; region is recorded but explains nothing (both ~0.5).
    Flaky, but no guard can be admitted — the whole spread is unexplained."""
    obs = ([(stmt, i % 2 == 0, {"region": "us-east"}, "tool:probe") for i in range(40)]
           + [(stmt, i % 2 == 0, {"region": "eu"}, "tool:probe") for i in range(40)])
    import random
    random.Random(11).shuffle(obs)
    m.observe_batch(obs)
    m.run_gate()


# ── pure helper unit tests (fast, no store) ─────────────────────────────────

def test_outcome_breakdown_adds_a_residual_bucket_for_the_missing_key():
    obs = ([({"region": "us-east"}, True)] * 9 + [({"region": "us-east"}, False)]
           + [({"region": "eu"}, False)] * 8 + [({}, True), ({}, False), ({}, True)])
    groups = C.outcome_breakdown(obs, "region")
    assert groups["us-east"] == C.Group(10, 9)
    assert groups["eu"] == C.Group(8, 0)
    # the three obs that recorded no region land in __residual__, never in a value
    assert groups[C.RESIDUAL_BUCKET] == C.Group(3, 2)
    # a key nothing records is ALL residual
    allres = C.outcome_breakdown(obs, "host")
    assert set(allres) == {C.RESIDUAL_BUCKET}
    assert allres[C.RESIDUAL_BUCKET] == C.Group(len(obs), sum(1 for _, o in obs if o))


def test_explained_fraction_is_zero_for_a_non_separating_split_and_high_for_a_clean_one():
    # both groups at the pooled rate → the key explains nothing
    assert C.explained_fraction([C.Group(40, 20), C.Group(40, 20)]) == 0.0
    # perfectly separated → η² == 1
    assert abs(C.explained_fraction([C.Group(40, 40), C.Group(40, 0)]) - 1.0) < 1e-12
    # partial separation sits strictly between
    frac = C.explained_fraction([C.Group(40, 36), C.Group(25, 7)])
    assert 0.0 < frac < 1.0
    # degenerate outcome (all true): zero total variance → 0, never a divide error
    assert C.explained_fraction([C.Group(10, 10), C.Group(10, 10)]) == 0.0


# ── acceptance 1: the per-context breakdown ─────────────────────────────────

def test_per_context_breakdown_reports_rates_counts_and_a_residual_bucket(sys_):
    m = sys_
    _admit(m, CRISP)
    _plant_guarded(m, CRISP)

    d = m.distribution(CRISP)
    assert d["found"] and d["stmt_type"] == "crisp"
    region = d["modes"]["region"]
    assert region["us-east"]["n"] == 40
    assert abs(region["us-east"]["p"] - 36 / 40) < 1e-12
    assert region["eu"]["n"] == 25
    assert abs(region["eu"]["p"] - 7 / 25) < 1e-12
    # the 17 region-less observations are the honest 'cannot attribute' mass
    assert region[C.RESIDUAL_BUCKET]["n"] == 17
    assert d["n_obs"] == 82


# ── acceptance 2: the unexplained "unknown" residual ────────────────────────

def test_unknown_residual_is_near_total_without_a_guard_and_reduced_with_one(sys_):
    m = sys_
    _admit(m, CRISP)
    _plant_guarded(m, CRISP)
    d = m.distribution(CRISP)
    res = d["residual"]
    # a guard on region was admitted → part of the spread is explained
    assert res["conditioning_key"] == "region"
    assert 0.30 < res["explained"] < 0.60
    assert abs(res["explained"] + res["unexplained"] - 1.0) < 1e-12
    assert res["dispersion_stat"] is not None and res["dispersion_stat"] > 3.0
    assert d["flaky"]

    # a second, independent store where region explains nothing
    m2 = CandorSystem(m.root.parent / "store2")
    try:
        _admit(m2, CRISP)
        _plant_unexplained(m2, CRISP)
        d2 = m2.distribution(CRISP)
        res2 = d2["residual"]
        assert res2["conditioning_key"] is None       # no guard could be admitted
        assert res2["explained"] == 0.0
        assert res2["unexplained"] == 1.0             # near-total: nothing explains it
        assert d2["flaky"]                            # still dispersed / unstable
    finally:
        m2.close()


# ── acceptance 3: determinism across replay and the checkpoint fast path ────

def test_breakdown_is_reproduced_exactly_after_replay_and_via_a_snapshot(sys_):
    m = sys_
    _admit(m, CRISP)
    _plant_guarded(m, CRISP)

    def canon(d):
        return json.dumps(d, sort_keys=True)

    before = canon(m.distribution(CRISP))
    m.replay()
    assert canon(m.distribution(CRISP)) == before, "replay must reproduce the breakdown"

    # checkpoint, then reopen from the snapshot fast path (pure read over counts)
    m.checkpoint()
    m.close()
    m2 = CandorSystem(m.root)
    try:
        assert canon(m2.distribution(CRISP)) == before, "snapshot restore must match"
    finally:
        m2.close()


# ── acceptance 4: additivity — predict() and closure are untouched ──────────

def test_distribution_moves_nothing_predict_and_closure_are_byte_identical(sys_):
    m = sys_
    _admit(m, CRISP)
    _plant_guarded(m, CRISP)

    def pred_tuple(p):
        return (p.p, p.ci, tuple(sorted(p.channels.items())),
                tuple(sorted(p.sensitivity.items())), tuple(sorted(p.caveats)),
                p.snapshot_id, p.rejection_rate)

    before_pred = pred_tuple(m.predict(CRISP, BUDGET))
    before_closure = m.closure_hash()
    before_head = m.ledger_head()

    # exercise the new read repeatedly — it must perturb nothing
    for _ in range(5):
        m.distribution(CRISP)

    assert pred_tuple(m.predict(CRISP, BUDGET)) == before_pred, "predict() drifted"
    assert m.closure_hash() == before_closure, "closure_hash changed"
    assert m.ledger_head() == before_head, "distribution() appended to the ledger"
    assert m.replay() == before_closure, "replay closure changed"


def test_distribution_on_an_unknown_or_undispersed_fact_is_trivial(sys_):
    m = sys_
    # a stmt that names no fact
    d = m.distribution({"pred": "nope", "args": ["x"]})
    assert d == {"found": False, "fact_id": None, "stmt_type": None, "n_obs": 0,
                 "flaky": False, "dispersion_flag": False, "modes": {},
                 "derived_modes": {},
                 "residual": {"conditioning_key": None, "explained": 0.0,
                              "unexplained": 0.0, "dispersion_stat": None}}

    # an admitted fact with a handful of consistent observations: not dispersed
    _admit(m, CRISP)
    for _ in range(6):
        m.observe(CRISP, True, {"region": "us-east"}, actor="tool:probe")
    m.run_gate()
    d2 = m.distribution(CRISP)
    assert d2["found"] and not d2["flaky"]
    assert d2["residual"]["unexplained"] == 0.0
    assert d2["modes"]["region"]["us-east"] == {"p": 1.0, "n": 6}
