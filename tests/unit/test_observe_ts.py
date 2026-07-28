"""H4: observe() accepts an explicit event timestamp.

ledger.append already accepts ts; observe() never threaded it through, so a
historical replay stamped the ingest wall clock instead of the real event
time — and a located changepoint's valid_to became today's date, not the
event's. Default (ts=None) must still use time.time() exactly as before.
"""

from __future__ import annotations

import time

from candor.system import CandorSystem


def _seed(root):
    m = CandorSystem(root)
    m.set_actor_quota("tool:probe", obs_per_epoch=10_000, cand_per_epoch=10_000)
    m.assert_({"pred": "flaky_link", "args": ["c", "d"], "stmt_type": "frequency"},
              source="s", actor="human:me")
    m.run_gate()
    return m


def _event(m, seq):
    return next(e for e in m.events_since(0) if e["seq"] == seq)


def test_observe_records_supplied_ts(tmp_path):
    m = _seed(tmp_path / "store")
    stmt = {"pred": "flaky_link", "args": ["c", "d"]}
    seq = m.observe(stmt, True, {}, actor="tool:probe", ts=1234567)
    ts = _event(m, seq)["ts"]
    m.close()
    assert ts == 1234567, "observe() did not stamp the supplied event time"


def test_observe_default_ts_is_wallclock(tmp_path):
    m = _seed(tmp_path / "store")
    stmt = {"pred": "flaky_link", "args": ["c", "d"]}
    before = int(time.time() * 1000)
    seq = m.observe(stmt, True, {}, actor="tool:probe")
    after = int(time.time() * 1000)
    ts = _event(m, seq)["ts"]
    m.close()
    assert before <= ts <= after, "default ts=None must remain the wall clock"


def test_observe_batch_carries_optional_ts(tmp_path):
    m = _seed(tmp_path / "store")
    stmt = {"pred": "flaky_link", "args": ["c", "d"]}
    seqs = m.observe_batch([
        (stmt, True, {}, "tool:probe"),            # 4-tuple: backward compatible
        (stmt, False, {}, "tool:probe", 555),      # 5-tuple: explicit ts
    ])
    evs = {e["seq"]: e for e in m.events_since(0)}
    m.close()
    assert evs[seqs[1]]["ts"] == 555, "observe_batch dropped the per-tuple ts"
    assert evs[seqs[0]]["ts"] != 555, "the 4-tuple form must default to wall clock"
