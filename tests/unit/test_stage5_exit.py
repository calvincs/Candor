"""Spec §8 Stage-5 exit extras, demonstrated end to end.

1. A guard discovered, gated, and validated on HELD-OUT observations — the
   sweep splits discovery/validation by event parity and the gate's step 5
   judges the held-out half.
2. An open question explained RETROACTIVELY by a later-arriving covariate
   (§4.5): dispersion is detected but unexplainable while only a high-
   cardinality bookkeeping key is recorded; months later someone starts
   recording `elevation`, the sweep re-tests every open question against the
   new key automatically, and the question flips to explained with a gated
   guard naming it.
"""

from __future__ import annotations

import random

BUDGET = 10_000
STMT = {"pred": "boils_ok", "args": ["water"]}


def _phase1_unexplained(sys_):
    """240 obs whose variance is driven by an UNRECORDED covariate. The only
    recorded key is `batch` (24 values — a lookup table, not a condition)."""
    sys_.assert_({"pred": "boils_ok", "args": ["water"], "stmt_type": "frequency"},
                 source="seed", actor="human:calvin")
    sys_.run_gate()
    rng = random.Random(20260727)
    for i in range(240):
        hidden_high = (i // 20) % 2 == 1          # blocky, but never recorded
        theta = 0.45 if hidden_high else 0.85
        sys_.observe(STMT, rng.random() < theta, {"batch": str(i // 10)},
                     actor="tool:probe")
    return sys_.run_gate()


def _phase2_elevation_arrives(sys_):
    """Later observations finally record the explaining covariate."""
    rng = random.Random(99)
    for i in range(60):
        high = i % 2 == 1
        theta = 0.35 if high else 0.90
        sys_.observe(STMT, rng.random() < theta,
                     {"elevation": "high" if high else "sea"}, actor="tool:probe")
    return sys_.run_gate()


def test_question_opens_unexplained_then_is_retroactively_explained(sys_):
    runs1 = _phase1_unexplained(sys_)
    fid = sys_.fact_id_for(STMT)
    assert not any(r["candidate_kind"] == "guard" and r["status"] == "admitted"
                   for r in runs1), \
        "a 24-ary bookkeeping key must not become a guard (lookup-table rail)"
    questions = [q for q in sys_.questions("dispersion") if q["target_id"] == fid]
    assert questions and questions[0]["status"] == "open", \
        "detected-but-unexplained variance must open a question, not vanish"
    assert sys_.why(fid)["dispersion_flag"] is True

    runs2 = _phase2_elevation_arrives(sys_)
    guards = [r for r in runs2 if r["candidate_kind"] == "guard"
              and r["status"] == "admitted"]
    assert guards, "the newly recorded covariate must yield a gated guard"
    row = sys_.index.one(
        "SELECT status, suggested_measurement FROM open_questions WHERE target_id=?",
        (fid,))
    assert row["status"] == "explained", \
        "the open question must be re-tested against the new key and explained"
    assert "elevation" in row["suggested_measurement"]


def test_discovered_guard_carries_holdout_validation(sys_):
    _phase1_unexplained(sys_)
    _phase2_elevation_arrives(sys_)
    cand = sys_.index.one(
        "SELECT body_json FROM candidates WHERE kind='guard' "
        "AND status='admitted' ORDER BY event_seq DESC LIMIT 1")
    import json
    body = json.loads(cand["body_json"])
    holdout = body.get("holdout") or {}
    assert holdout.get("hits", 0) > holdout.get("misses", 0), \
        "step 5 must judge the guard on held-out observations, and it must win"
    assert body.get("conditioning_key") == "elevation"
