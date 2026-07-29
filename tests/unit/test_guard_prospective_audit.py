"""Δ11 — the prospective guard audit: admitted axioms must keep paying rent.

Entry validation is retrospective (a held-out quarter of data that existed at
proposal time). These tests prove the OTHER half of the Popperian loop: a
guard's direction keeps getting scored on observations that postdate its
admission, a reversal or a persistent null demotes it THROUGH THE LEDGER (rule
out of the closure, candidate row closed out, read paths silent), the
memoryless sweep's re-proposal is held to the re-entry bar instead of flapping,
and the whole history replays bit-for-bit.
"""

from __future__ import annotations

import json
import random

from candor.system import CandorSystem

BASE_TS = 1_749_945_600_000


def _admit(m, stmt):
    m.assert_(stmt, source="seed", actor="human:calvin")
    m.run_gate()
    return m.fact_id_for(stmt)


def _guard_rows(m, fid, status):
    out = []
    for r in m.index.query(
            "SELECT id, status, body_json FROM candidates WHERE kind='guard' "
            "AND status=? ORDER BY event_seq", (status,)):
        body = json.loads(r["body_json"])
        if body.get("target_fact") == fid:
            out.append((r["id"], body))
    return out


def _feed(m, stmt, n, p_of, seed, t0=0):
    rng = random.Random(seed)
    for i in range(n):
        method = "crawl4ai" if i % 2 == 0 else "http"
        ok = rng.random() < p_of(method)
        m.observe(stmt, ok, {"method": method}, actor="tool:probe",
                  ts=BASE_TS + (t0 + i) * 1000)


def _world(tmp_path):
    m = CandorSystem(tmp_path / "store")
    stmt = {"pred": "fetch_ok", "args": ["site"], "stmt_type": "frequency"}
    fid = _admit(m, stmt)
    # phase 1: method genuinely conditions the outcome → guard admitted
    _feed(m, stmt, 200, lambda meth: 0.9 if meth == "crawl4ai" else 0.2, seed=7)
    m.run_gate()
    assert _guard_rows(m, fid, "admitted"), "phase 1 must admit the guard"
    return m, stmt, fid


def test_reversal_demotes_through_the_ledger(tmp_path):
    m, stmt, fid = _world(tmp_path)
    # phase 2: the world flips — crawl4ai now fails where http succeeds
    _feed(m, stmt, 200, lambda meth: 0.15 if meth == "crawl4ai" else 0.9,
          seed=8, t0=200)
    runs = m.run_gate()
    demoted = [r for r in runs if r["status"] == "demoted"]
    assert demoted and "reversed" in demoted[0]["reason"]
    assert _guard_rows(m, fid, "demoted"), "candidate row must be closed out"
    assert not _guard_rows(m, fid, "admitted")
    # the rule left the committed tier: nothing structural='admitted' remains
    rid = demoted[0]["reason"]  # noqa: F841 — reason inspected above
    rules = m.index.query(
        "SELECT id, structural FROM rules WHERE structural='admitted'")
    assert all("rule" not in r["id"] or r["structural"] != "admitted"
               or not _is_guard_rule(m, r["id"], fid) for r in rules)
    # read path goes silent: the guard no longer names a conditioning key
    assert m.distribution(stmt)["residual"]["conditioning_key"] is None


def _is_guard_rule(m, rule_id, fid):
    row = m.index.one("SELECT body_json FROM rules WHERE id=?", (rule_id,))
    return row is not None and "guards" in (row["body_json"] or "")


def test_persisting_guard_is_never_demoted(tmp_path):
    m, stmt, fid = _world(tmp_path)
    # phase 2: the structure persists
    _feed(m, stmt, 200, lambda meth: 0.9 if meth == "crawl4ai" else 0.2,
          seed=9, t0=200)
    runs = m.run_gate()
    assert not [r for r in runs if r["status"] == "demoted"]
    assert _guard_rows(m, fid, "admitted")


def test_too_little_post_admission_evidence_stays_silent(tmp_path):
    m, stmt, fid = _world(tmp_path)
    # only a handful of post-admission observations — under the audit floor
    _feed(m, stmt, 10, lambda meth: 0.15 if meth == "crawl4ai" else 0.9,
          seed=10, t0=200)
    runs = m.run_gate()
    assert not [r for r in runs if r["status"] == "demoted"]
    assert _guard_rows(m, fid, "admitted"), "no verdict on thin evidence"


def test_staleness_demotes_on_twice_the_entry_evidence(tmp_path):
    m, stmt, fid = _world(tmp_path)
    # phase 2: the direction stops beating chance (mild anti-lean, far from the
    # reversal bar) — staleness carries the demotion, not reversal
    _feed(m, stmt, 200, lambda meth: 0.40 if meth == "crawl4ai" else 0.60,
          seed=11, t0=200)
    runs = m.run_gate()
    demoted = [r for r in runs if r["status"] == "demoted"]
    assert demoted and "chance" in demoted[0]["reason"]


def test_demoted_guard_cannot_flap_back_in(tmp_path):
    """The flap world is the MILD decay: the full history still supports the
    guard (the sweep re-proposes it from exactly the data that admitted it),
    while the post-admission record killed it. Without the re-entry bar this
    would admit→demote→admit forever."""
    m, stmt, fid = _world(tmp_path)
    _feed(m, stmt, 200, lambda meth: 0.40 if meth == "crawl4ai" else 0.60,
          seed=12, t0=200)
    m.run_gate()                       # audit demotes, then the sweep re-proposes
    assert _guard_rows(m, fid, "demoted")
    m.observe(stmt, True, {"method": "crawl4ai"}, actor="tool:probe",
              ts=BASE_TS + 401_000)
    m.run_gate()                       # and re-proposes again on the new obs
    readmitted = [b for _, b in _guard_rows(m, fid, "admitted")
                  if b.get("conditioning_key") == "method"
                  and b["body"]["guards"][0]["value"] == "crawl4ai"]
    assert readmitted == [], "flap: a demoted direction re-entered under the bar"
    rejected = m.index.query(
        "SELECT reason FROM candidates WHERE kind='guard' AND status='rejected'")
    assert any("re-entry" in (r["reason"] or "") for r in rejected), \
        "the re-proposal must be rejected BY the re-entry bar, not by accident"


def test_reentry_on_fresh_evidence_is_allowed(tmp_path):
    """A demotion is not a life sentence: when the world re-structures, the
    post-demotion record itself clears the bar and the guard returns."""
    m, stmt, fid = _world(tmp_path)
    _feed(m, stmt, 200, lambda meth: 0.40 if meth == "crawl4ai" else 0.60,
          seed=14, t0=200)
    m.run_gate()
    assert _guard_rows(m, fid, "demoted")
    # the structure comes back, strongly, for long enough to overcome the
    # demotion's against-evidence on post-demotion data alone
    _feed(m, stmt, 200, lambda meth: 0.95 if meth == "crawl4ai" else 0.05,
          seed=15, t0=400)
    m.run_gate()
    readmitted = [b for _, b in _guard_rows(m, fid, "admitted")
                  if b.get("conditioning_key") == "method"
                  and b["body"]["guards"][0]["value"] == "crawl4ai"]
    assert readmitted, "fresh post-demotion evidence must be able to re-enter"


def test_demotion_history_survives_replay(tmp_path):
    m, stmt, fid = _world(tmp_path)
    _feed(m, stmt, 200, lambda meth: 0.15 if meth == "crawl4ai" else 0.9,
          seed=13, t0=200)
    m.run_gate()
    assert _guard_rows(m, fid, "demoted")
    before = m.closure_hash()
    assert m.replay() == before, "demotion state must be a fold of the ledger"
