"""Δ13 — `do:` intervention semantics: acting on the world is not watching it.

A `do:` context key partitions like any covariate — no new statistics — but the
semantics change where they matter: a guard conditioned on one is labeled
REGIME dependence (P(·|observe) ≠ P(·|do); the association does not transfer
across the boundary), and a scalar prediction pooling observations from mixed
do: regimes says so out loud (`regime_mixed`). Detection is post-hoc by
construction; the pre-intervention wall is asserted in the claims-suite battery
(tests/claims/test_axiom_battery.py), not here.
"""

from __future__ import annotations

import json
import random

from candor.system import CandorSystem

BASE_TS = 1_749_945_600_000
STMT = {"pred": "metric_tracks_goal", "args": ["ctr"], "stmt_type": "frequency"}


def _admit(m, stmt=STMT):
    m.assert_(stmt, source="seed", actor="human:calvin")
    m.run_gate()
    return m.fact_id_for(stmt)


def _mixed_world(tmp_path, n=200, cp=100):
    m = CandorSystem(tmp_path / "store")
    fid = _admit(m)
    rng = random.Random(29)
    for i in range(n):
        targeted = i >= cp
        ok = rng.random() < (0.25 if targeted else 0.9)
        m.observe(STMT, ok, {"do:optimize_metric": "yes" if targeted else "no"},
                  actor="tool:probe", ts=BASE_TS + i * 1000)
    m.run_gate()
    return m, fid


def test_pooling_across_the_do_boundary_is_said_out_loud(tmp_path):
    m, fid = _mixed_world(tmp_path)
    p = m.predict(STMT, budget=10_000)
    assert "regime_mixed" in p.caveats


def test_single_regime_carries_no_caveat(tmp_path):
    m = CandorSystem(tmp_path / "store")
    _admit(m)
    rng = random.Random(31)
    for i in range(60):
        m.observe(STMT, rng.random() < 0.9, {"do:optimize_metric": "no"},
                  actor="tool:probe", ts=BASE_TS + i * 1000)
    m.run_gate()
    assert "regime_mixed" not in m.predict(STMT, budget=10_000).caveats


def test_partial_do_coverage_is_mixed(tmp_path):
    """Absent IS a regime: observations logged before anyone thought to record
    the intervention key still predate it."""
    m = CandorSystem(tmp_path / "store")
    _admit(m)
    rng = random.Random(37)
    for i in range(40):
        ctx = {} if i < 20 else {"do:optimize_metric": "yes"}
        m.observe(STMT, rng.random() < 0.5, ctx, actor="tool:probe",
                  ts=BASE_TS + i * 1000)
    m.run_gate()
    assert "regime_mixed" in m.predict(STMT, budget=10_000).caveats


def test_do_guard_is_labeled_regime_dependence(tmp_path):
    m, fid = _mixed_world(tmp_path)
    guards = [json.loads(r["body_json"]) for r in m.index.query(
        "SELECT body_json FROM candidates WHERE kind='guard' "
        "AND status='admitted' ORDER BY event_seq")]
    guards = [g for g in guards if g.get("target_fact") == fid]
    assert guards and guards[0]["conditioning_key"] == "do:optimize_metric"
    assert guards[0].get("regime_dependent") is True
    q = m.index.one(
        "SELECT suggested_measurement FROM open_questions "
        "WHERE kind='dispersion' AND target_id=?", (fid,))
    assert q is not None and "regime dependence" in q["suggested_measurement"]
    assert "does not transfer" in q["suggested_measurement"]


def test_ordinary_covariate_guard_is_not_regime_labeled(tmp_path):
    m = CandorSystem(tmp_path / "store")
    stmt = {"pred": "fetch_ok", "args": ["site"], "stmt_type": "frequency"}
    m.assert_(stmt, source="seed", actor="human:calvin")
    m.run_gate()
    fid = m.fact_id_for(stmt)
    rng = random.Random(41)
    for i in range(200):
        meth = "crawl4ai" if i % 2 == 0 else "http"
        m.observe(stmt, rng.random() < (0.9 if meth == "crawl4ai" else 0.2),
                  {"method": meth}, actor="tool:probe", ts=BASE_TS + i * 1000)
    m.run_gate()
    guards = [json.loads(r["body_json"]) for r in m.index.query(
        "SELECT body_json FROM candidates WHERE kind='guard' "
        "AND status='admitted'")]
    guards = [g for g in guards if g.get("target_fact") == fid]
    assert guards and "regime_dependent" not in guards[0]
    assert "regime_mixed" not in m.predict(stmt, budget=10_000).caveats


def test_categorical_prediction_carries_the_caveat_too(tmp_path):
    m = CandorSystem(tmp_path / "store")
    stmt = {"pred": "block_reason", "args": ["site"], "stmt_type": "categorical"}
    m.assert_(stmt, source="seed", actor="human:calvin")
    m.run_gate()
    rng = random.Random(43)
    for i in range(40):
        targeted = i >= 20
        v = rng.choice(["captcha", "block"] if targeted else ["ok", "slow"])
        m.observe(stmt, value=v,
                  ctx={"do:evade": "yes" if targeted else "no"},
                  actor="tool:probe", ts=BASE_TS + i * 1000)
    m.run_gate()
    p = m.predict(stmt, budget=10_000)
    assert "regime_mixed" in p.caveats


def test_do_semantics_survive_replay(tmp_path):
    m, fid = _mixed_world(tmp_path)
    before = m.closure_hash()
    assert m.replay() == before
