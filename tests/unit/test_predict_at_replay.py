"""H1: predict_at must reproduce a snapshot's number faithfully (I8).

The private replay inside predict_at diverged from _refold in three ways:

  * it never silenced retracted actors, so a retracted source spoke again in
    the snapshot;
  * it never ran the curiosity sweep, so under_specified / narrow caveats the
    live prediction carried were dropped;
  * a snapshot whose head is genesis left cutoff=None and folded the ENTIRE
    log instead of an empty store.
"""

from __future__ import annotations

import random

import pytest

from candor.core import calibration as calibration_mod
from candor.system import CandorSystem


def _seed_crisp(root):
    m = CandorSystem(root)
    for a in ("tool:probe", "tool:bad", "human:me"):
        m.set_actor_quota(a, obs_per_epoch=10_000, cand_per_epoch=10_000)
    m.assert_({"pred": "holds", "args": ["t0"], "stmt_type": "crisp"},
              source="s", actor="human:me")
    m.run_gate()
    return m


def test_predict_at_respects_a_retraction(tmp_path):
    m = _seed_crisp(tmp_path / "store")
    stmt = {"pred": "holds", "args": ["t0"]}
    for _ in range(8):
        m.observe(stmt, True, {}, actor="tool:probe")
        m.observe(stmt, False, {}, actor="tool:bad")   # the source to retract
    m.retract_source("tool:bad", reason="hallucinating scraper")
    p_snap = m.predict(stmt, budget=1500)               # post-retraction number
    snap = p_snap.snapshot_id
    # Move the present: honest FALSE votes drag the live number down.
    for _ in range(12):
        m.observe(stmt, False, {}, actor="tool:probe")
    p_live = m.predict(stmt, budget=1500).p
    p_at = m.predict_at(stmt, snap).p
    m.close()
    assert p_live != pytest.approx(p_snap.p, abs=1e-9), \
        "history did not move the live number (vacuous)"
    assert p_at == pytest.approx(p_snap.p, abs=1e-12), \
        "predict_at let the retracted source speak again (I8)"


def test_predict_at_reproduces_under_specified_caveat(tmp_path):
    m = CandorSystem(tmp_path / "store")
    m.set_actor_quota("tool:probe", obs_per_epoch=10_000, cand_per_epoch=10_000)
    m.assert_({"pred": "flaky_link", "args": ["c", "d"], "stmt_type": "frequency"},
              source="s", actor="human:me")
    m.run_gate()
    stmt = {"pred": "flaky_link", "args": ["c", "d"]}
    rng = random.Random(11)
    for i in range(120):                                # delta=0.55 overdispersion
        elev = "high" if i % 2 else "sea"
        theta = 0.35 if elev == "high" else 0.90
        m.observe(stmt, rng.random() < theta, {"elevation": elev}, actor="tool:probe")
    m.run_gate()
    p0 = m.predict(stmt, budget=10_000)
    assert "under_specified" in p0.caveats, \
        "live prediction lacks the caveat (vacuous)"
    snap = p0.snapshot_id
    for _ in range(10):                                 # move the present
        m.observe(stmt, True, {}, actor="tool:probe")
    p_at = m.predict_at(stmt, snap)
    m.close()
    assert p_at.p == pytest.approx(p0.p, abs=1e-12), "snapshot number drifted (I8)"
    assert "under_specified" in p_at.caveats, \
        "predict_at dropped the under_specified caveat (I8)"


def test_flaky_crisp_prediction_surfaces_caveat_and_reproduces(tmp_path):
    """H8b: an alternating crisp fact is FLAKY. predict() takes the grouped-by-key
    path and must SURFACE the instability as an ``unstable`` caveat — a single-
    context alternating fact trips no covariate or temporal dispersion, so it
    would otherwise be silent. The number stays deterministic: reproduced exactly
    by predict_at at its snapshot (I8), and closure_hash reproduces under a forced
    replay."""
    root = tmp_path / "store"
    m = CandorSystem(root)
    m.set_actor_quota("tool:probe", obs_per_epoch=100_000, cand_per_epoch=100_000)
    m.assert_({"pred": "reach", "args": ["a", "b"], "stmt_type": "crisp"},
              source="s", actor="human:me")
    m.run_gate()
    stmt = {"pred": "reach", "args": ["a", "b"]}
    for i in range(240):                    # alternating outcomes, single context
        m.observe(stmt, i % 2 == 0, {"env": "prod"}, actor="tool:probe")
    p0 = m.predict(stmt, budget=10_000)
    assert "unstable" in p0.caveats, "flaky prediction was silent (no caveat)"
    # The instability is NOT overdispersion-flagged, so under_specified would not
    # have surfaced it: the dedicated caveat is doing real work.
    assert "under_specified" not in p0.caveats
    snap = p0.snapshot_id
    for _ in range(10):                     # move the present
        m.observe(stmt, True, {"env": "prod"}, actor="tool:probe")
    p_live = m.predict(stmt, budget=10_000)
    p_at = m.predict_at(stmt, snap)
    assert p_live.p != pytest.approx(p0.p, abs=1e-9), "history did not move p (vacuous)"
    assert p_at.p == pytest.approx(p0.p, abs=1e-12), "flaky predict_at drifted (I8)"
    assert p_at.ci == pytest.approx(p0.ci, abs=1e-12)
    assert "unstable" in p_at.caveats, "predict_at dropped the unstable caveat (I8)"
    m.close()

    m2 = CandorSystem(root)                 # reopen from the log alone
    p_reopen = m2.predict(stmt, budget=10_000).p
    hash_reopen = m2.closure_hash()
    hash_replay = m2.replay()               # forced full-from-genesis
    m2.close()
    assert p_reopen == pytest.approx(p_live.p, abs=1e-12), "reopen changed a flaky p"
    assert hash_reopen == hash_replay, "flaky store: reopen != forced replay (I1)"


def test_predict_at_genesis_head_folds_nothing(tmp_path):
    m = CandorSystem(tmp_path / "store")
    m.set_actor_quota("tool:probe", obs_per_epoch=10_000, cand_per_epoch=10_000)
    m.assert_({"pred": "flaky_link", "args": ["c", "d"], "stmt_type": "frequency"},
              source="s", actor="human:me")
    m.run_gate()
    stmt = {"pred": "flaky_link", "args": ["c", "d"]}
    for _ in range(30):
        m.observe(stmt, True, {}, actor="tool:probe")
    p_full = m.predict(stmt, budget=10_000).p
    genesis_snap = calibration_mod.snapshot_id("0" * 64, m._calib.hash)
    p_genesis = m.predict_at(stmt, genesis_snap).p
    m.close()

    empty = CandorSystem(tmp_path / "empty")            # nothing folded baseline
    p_empty = empty.predict(stmt, budget=10_000).p
    empty.close()

    assert p_full != pytest.approx(p_empty, abs=1e-9), \
        "history did not move p (vacuous)"
    assert p_genesis == pytest.approx(p_empty, abs=1e-12), \
        "a genesis-head snapshot must fold NOTHING, not the entire log (I8)"
