"""M11: resource-stability cleanups in CandorSystem.

(a) predict_at must release its temp Index's fds on every exception path.
(b) the in-memory health-event log must be bounded.
(c) an idle run_gate (no new observations since the last sweep) must skip the
    curiosity sweep entirely — appending nothing and changing no outcome.
"""

from __future__ import annotations

import os

import pytest

from candor.core import calibration as calibration_mod
from candor.system import CandorSystem, QuotaExceeded, _HEALTH_EVENTS_CAP


def _fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


# ── (a) predict_at fd leak ───────────────────────────────────────────────────
@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"),
                    reason="fd census needs /proc")
def test_failing_predict_at_does_not_leak_fds(seeded):
    stmt = {"pred": "flaky_link", "args": ["c", "d"]}
    # A well-formed snapshot whose ledger head is not in this chain: predict_at
    # opens a temp Index, then raises KeyError while locating the cutoff — the
    # path that used to skip tmp_index.close() and leak db/-wal/-shm.
    bogus = calibration_mod.snapshot_id("f" * 64, "0" * 16)

    for _ in range(5):                              # warm caches / one-time fds
        with pytest.raises(KeyError):
            seeded.predict_at(stmt, bogus)
    before = _fd_count()
    for _ in range(60):
        with pytest.raises(KeyError):
            seeded.predict_at(stmt, bogus)
    after = _fd_count()
    assert after <= before + 2, f"fd leak across predict_at: {before} -> {after}"


# ── (b) bounded health-event log ─────────────────────────────────────────────
def test_health_event_log_is_bounded(tmp_path):
    m = CandorSystem(tmp_path / "store")
    try:
        m.set_actor_quota("tool:spammer", obs_per_epoch=0)   # every observe rejected
        stmt = {"pred": "p", "args": ["a"]}
        overflow = _HEALTH_EVENTS_CAP + 250
        for _ in range(overflow):
            with pytest.raises(QuotaExceeded):
                m.observe(stmt, True, {}, actor="tool:spammer")

        assert len(m._health_events) == _HEALTH_EVENTS_CAP, \
            "health-event log grew without bound"
        events = m.health()["events"]
        quota = [e for e in events if e.get("kind") == "quota_exhausted"]
        assert quota, "health() dropped the recent quota rejections"
        assert all(e["actor"] == "tool:spammer" for e in quota)
    finally:
        m.close()


# ── (c) idle run_gate skips the sweep ────────────────────────────────────────
def _plant_dispersion(m):
    """A frequency fact with a clean covariate split, so the sweep produces a
    STANDING guard proposal — the exact input that made the pre-fix run_gate
    re-append ledger events on every call."""
    m.set_actor_quota("tool:probe", obs_per_epoch=100_000, cand_per_epoch=100_000)
    m.assert_({"pred": "scrape_ok", "args": ["s"], "stmt_type": "frequency"},
              source="seed", actor="human:calvin")
    m.run_gate()
    stmt = {"pred": "scrape_ok", "args": ["s"]}
    for i in range(48):
        site = "lab" if i % 2 == 0 else "sea"
        m.observe(stmt, site == "lab", {"site": site}, actor="tool:probe")
    return stmt


def test_idle_run_gate_appends_nothing_and_keeps_outcomes(tmp_path):
    m = CandorSystem(tmp_path / "store")
    try:
        stmt = _plant_dispersion(m)
        runs = m.run_gate()                          # sweep runs on the new data
        assert any(r["candidate_kind"] == "guard" for r in runs), \
            "expected the planted dispersion to produce a standing guard proposal"
        fid = m.fact_id_for(stmt)
        assert m.index.one("SELECT dispersion_flag FROM facts WHERE id=?",
                           (fid,))["dispersion_flag"] == 1

        head, seq = m.ledger_head(), m.ledger.seq()
        chash = m.closure_hash()
        # Idle: no new observations => no sweep, nothing appended, no re-decision.
        for _ in range(4):
            assert m.run_gate() == []
            assert m.ledger_head() == head
            assert m.ledger.seq() == seq
        assert m.closure_hash() == chash

        # Demonstrate the churn the watermark prevents: forcing a re-sweep (as
        # the pre-fix code did unconditionally) re-appends the standing proposal.
        m._sweep_obs_watermark = None
        m.run_gate()
        assert m.ledger.seq() > seq, "a forced sweep re-appends the same proposal"

        # A genuine new observation re-arms the sweep — never permanently off.
        grown = m.ledger.seq()
        m.observe(stmt, True, {"site": "lab"}, actor="tool:probe")
        m.run_gate()
        assert m.ledger.seq() > grown
    finally:
        m.close()


def test_idle_skip_does_not_change_replayed_state(tmp_path):
    """The skip drops only churn: a full replay from the log reproduces the same
    closure whether or not idle run_gates ran (I1)."""
    m = CandorSystem(tmp_path / "store")
    try:
        _plant_dispersion(m)
        m.run_gate()
        for _ in range(6):                           # idle churn that now no-ops
            m.run_gate()
        live = m.closure_hash()
        assert m.replay() == live
    finally:
        m.close()
