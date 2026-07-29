"""Δ10 — derived context frames: the sweep postulates variables nobody logged.

The covariate search used to condition only on keys the agent recorded — the
C-17 wall: choosing the variable frame was itself the unautomated step. Δ10
synthesizes candidate frames from data the ledger already holds (event
timestamps → hour/dow, the fact's own outcome history → prev, compositions of
recorded keys → pairwise interactions) and runs the IDENTICAL machinery over
them: Tarone → BH → MDL → held-out quarter → gate. These tests prove each frame
class is discoverable, that a null world stays quiet, that recorded keys
outrank derived ones, that breadth ignores synthesized frames, and that the
whole thing replays bit-for-bit.
"""

from __future__ import annotations

import json
import random

from candor.periphery import curiosity as C
from candor.system import CandorSystem

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
BASE_TS = 1_749_945_600_000          # 2025-06-15 00:00 UTC — midnight-aligned


def _admit(m, stmt):
    m.assert_(stmt, source="seed", actor="human:calvin")
    m.run_gate()
    return m.fact_id_for(stmt)


def _admitted_guards(m):
    return [json.loads(r["body_json"]) for r in m.index.query(
        "SELECT body_json FROM candidates WHERE kind='guard' "
        "AND status='admitted' ORDER BY event_seq")]


def _flag(m, fid):
    return bool(m.index.one(
        "SELECT dispersion_flag FROM facts WHERE id=?", (fid,))["dispersion_flag"])


# ── derived:hour ────────────────────────────────────────────────────────────

def _hour_world(tmp_path, n=160, p_night=0.15, p_day=0.9, seed=3):
    """Outcome depends on UTC hour-of-day; the agent logs NOTHING. Alternating
    observations at 03:00 and 14:00 across successive days."""
    m = CandorSystem(tmp_path / "store")
    stmt = {"pred": "backup_ok", "args": ["nas"], "stmt_type": "frequency"}
    _admit(m, stmt)
    rng = random.Random(seed)
    for i in range(n):
        hour = 3 if i % 2 == 0 else 14
        ts = BASE_TS + (i // 2) * DAY_MS + hour * HOUR_MS
        ok = rng.random() < (p_night if hour == 3 else p_day)
        m.observe(stmt, ok, {}, actor="tool:probe", ts=ts)
    m.run_gate()
    return m, stmt


def test_hour_structure_is_found_with_nothing_logged(tmp_path):
    m, stmt = _hour_world(tmp_path)
    fid = m.fact_id_for(stmt)
    guards = [g for g in _admitted_guards(m) if g.get("target_fact") == fid]
    assert guards, "hour-driven variance with empty ctx must yield a guard"
    assert guards[0]["conditioning_key"] == C.DERIVED_HOUR
    # the guard's direction: success concentrates in the daytime hour
    assert guards[0]["body"]["guards"][0]["value"] == "14"
    assert _flag(m, fid), "the fact must carry the dispersion verdict"


def test_hour_guard_is_inspectable_in_distribution(tmp_path):
    m, stmt = _hour_world(tmp_path)
    d = m.distribution(stmt)
    assert C.DERIVED_HOUR in d["derived_modes"]
    by_hour = d["derived_modes"][C.DERIVED_HOUR]
    assert by_hour["14"]["p"] > 0.7 and by_hour["03"]["p"] < 0.35
    # η² is computed on the augmented projection, not stuck at 0
    assert d["residual"]["conditioning_key"] == C.DERIVED_HOUR
    assert d["residual"]["explained"] > 0.3
    # recorded modes stay recorded-only: nothing was logged, so nothing is there
    assert d["modes"] == {}


# ── derived:prev ────────────────────────────────────────────────────────────

def test_sticky_state_dependence_is_found(tmp_path):
    """A sticky Markov chain: P(ok|prev ok)=.92, P(ok|prev fail)=.15. No context
    is logged and every observation shares one hour, so `derived:prev` is the
    only informative frame."""
    m = CandorSystem(tmp_path / "store")
    stmt = {"pred": "warm_cache_hit", "args": ["cdn"], "stmt_type": "frequency"}
    fid = _admit(m, stmt)
    rng = random.Random(11)
    ok = True
    for i in range(220):
        ok = rng.random() < (0.92 if ok else 0.15)
        m.observe(stmt, ok, {}, actor="tool:probe", ts=BASE_TS + i * 1000)
    m.run_gate()
    guards = [g for g in _admitted_guards(m) if g.get("target_fact") == fid]
    assert guards and guards[0]["conditioning_key"] == C.DERIVED_PREV
    assert guards[0]["body"]["guards"][0]["value"] == "T"


# ── derived interactions ────────────────────────────────────────────────────

def test_pure_interaction_is_found(tmp_path):
    """XOR-shaped: p=.85 iff k1==k2, else .12 — each single key carries ~zero
    marginal signal, so only the synthesized pair frame can explain it."""
    m = CandorSystem(tmp_path / "store")
    stmt = {"pred": "handshake_ok", "args": ["mesh"], "stmt_type": "frequency"}
    fid = _admit(m, stmt)
    rng = random.Random(5)
    for i in range(240):
        k1, k2 = rng.choice("ab"), rng.choice("ab")
        ok = rng.random() < (0.85 if k1 == k2 else 0.12)
        m.observe(stmt, ok, {"k1": k1, "k2": k2}, actor="tool:probe",
                  ts=BASE_TS + i * 1000)
    m.run_gate()
    guards = [g for g in _admitted_guards(m) if g.get("target_fact") == fid]
    assert guards and guards[0]["conditioning_key"] == f"{C.DERIVED_PREFIX}k1xk2"
    v = guards[0]["body"]["guards"][0]["value"]
    assert v in ("a|a", "b|b")


def test_recorded_key_outranks_equally_predictive_derived_frame(tmp_path):
    """When a RECORDED key explains the variance, a derived frame that echoes it
    (method alternates deterministically, so derived:prev correlates) must not
    displace it: the agent's own vocabulary is the primary explanation space."""
    m = CandorSystem(tmp_path / "store")
    stmt = {"pred": "fetch_ok", "args": ["site"], "stmt_type": "frequency"}
    fid = _admit(m, stmt)
    rng = random.Random(7)
    for i in range(200):
        method = "crawl4ai" if i % 2 == 0 else "http"
        ok = rng.random() < (0.9 if method == "crawl4ai" else 0.2)
        m.observe(stmt, ok, {"method": method}, actor="tool:probe",
                  ts=BASE_TS + i * 1000)
    m.run_gate()
    guards = [g for g in _admitted_guards(m) if g.get("target_fact") == fid]
    assert guards and guards[0]["conditioning_key"] == "method"


# ── prev must not shadow other structure ────────────────────────────────────

def test_one_way_step_routes_to_a_date_not_a_prev_guard(tmp_path):
    """In a step world prev≈current by construction. The located DATE is the
    honest repair; derived:prev must yield to the §4.4 supersede routing."""
    m = CandorSystem(tmp_path / "store")
    stmt = {"pred": "resolver_ok", "args": ["dns"], "stmt_type": "frequency"}
    fid = _admit(m, stmt)
    rng = random.Random(17)
    for i in range(120):
        ok = rng.random() < (0.92 if i < 60 else 0.1)
        m.observe(stmt, ok, {}, actor="tool:probe", ts=BASE_TS + i * 1000)
    runs = m.run_gate()
    kinds = {(r["candidate_kind"], r["status"]) for r in runs}
    assert ("supersede_valid_time", "admitted") in kinds
    guards = [g for g in _admitted_guards(m) if g.get("target_fact") == fid]
    assert guards == [], "a one-way step must not mint a self-lag condition"


def test_block_shadow_opens_a_question_not_a_prev_guard(tmp_path):
    """An UNLOGGED block variable makes prev a smeared proxy: conditioned on
    prev the series still swings, so prev must decline and the honest output is
    the open question saying the missing argument was never captured."""
    m = CandorSystem(tmp_path / "store")
    stmt = {"pred": "queue_ok", "args": ["jobs"], "stmt_type": "frequency"}
    fid = _admit(m, stmt)
    rng = random.Random(23)
    for i in range(240):
        hidden_high = (i // 20) % 2 == 1
        ok = rng.random() < (0.45 if hidden_high else 0.85)
        m.observe(stmt, ok, {}, actor="tool:probe", ts=BASE_TS + i * 1000)
    m.run_gate()
    guards = [g for g in _admitted_guards(m) if g.get("target_fact") == fid]
    assert guards == [], "a block shadow must not become a prev condition"
    assert _flag(m, fid)
    q = m.index.one(
        "SELECT status FROM open_questions WHERE kind='dispersion' AND target_id=?",
        (fid,))
    assert q is not None and q["status"] == "open"


# ── null control ────────────────────────────────────────────────────────────

def test_null_world_mints_no_derived_condition(tmp_path):
    """iid outcomes with timestamps sweeping hours and days: every derived frame
    is tested and every one must stay quiet. A frame factory that fabricates
    conditions is worse than none."""
    m = CandorSystem(tmp_path / "store")
    stmt = {"pred": "ping_ok", "args": ["gw"], "stmt_type": "frequency"}
    fid = _admit(m, stmt)
    rng = random.Random(13)
    for i in range(240):
        ts = BASE_TS + i * 7 * HOUR_MS          # hours and dows both cycle
        m.observe(stmt, rng.random() < 0.55, {"noise": rng.choice("ab")},
                  actor="tool:probe", ts=ts)
    m.run_gate()
    guards = [g for g in _admitted_guards(m) if g.get("target_fact") == fid]
    assert guards == [], f"null world fabricated a condition: {guards}"
    assert not _flag(m, fid)


# ── breadth stays the agent's own ───────────────────────────────────────────

def test_breadth_ignores_derived_frames(tmp_path):
    m, stmt = _hour_world(tmp_path)
    fid = m.fact_id_for(stmt)
    row = m.index.one("SELECT breadth_class FROM facts WHERE id=?", (fid,))
    # nothing was recorded, so breadth is narrow even though derived:hour and
    # derived:dow vary richly — synthesized diversity is not logging diversity
    assert row["breadth_class"] == "narrow"


# ── determinism ─────────────────────────────────────────────────────────────

def test_derived_guard_state_survives_replay(tmp_path):
    m, _ = _hour_world(tmp_path)
    before = m.closure_hash()
    assert m.replay() == before


def test_augment_derived_is_pure_and_order_stable():
    ctxs = [{"a": "1", "b": "x"}, {"a": "2"}, {}]
    tss = [BASE_TS, BASE_TS + HOUR_MS, None]
    prevs = [None, "T", "F"]
    once = C.augment_derived(ctxs, tss, prevs)
    twice = C.augment_derived(ctxs, tss, prevs)
    assert once == twice
    assert ctxs[0] == {"a": "1", "b": "x"}, "inputs must not be mutated"
    assert C.DERIVED_PREV not in once[0] and once[1][C.DERIVED_PREV] == "T"
    assert f"{C.DERIVED_PREFIX}axb" in once[0] and \
        f"{C.DERIVED_PREFIX}axb" not in once[1]
    assert C.DERIVED_HOUR not in once[2], "no ts, no clock frame"
