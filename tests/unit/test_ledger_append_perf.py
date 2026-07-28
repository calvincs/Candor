"""H9: append() must not glob the segments directory on every call.

_open_tail() used to call _segments() (glob *.jsonl + sort) on every append, so
append throughput collapsed as segments accumulated (a cumulative O(n^2/4096) in
the write path). The fix caches the tail segment path on the Ledger instance and
recomputes it only on rollover / open / destroy. These tests pin the mechanism
(no per-append glob) and the invariants that must survive it.
"""

from __future__ import annotations

import candor.core.ledger as mod
from candor.core.ledger import Ledger


def test_steady_state_append_does_not_glob_segments(tmp_path, monkeypatch):
    lg = Ledger(tmp_path / "ledger")
    lg.open()
    lg.append("observation", "tool:probe", {"i": 0})       # opens the tail

    calls = {"n": 0}
    real = Ledger._segments

    def counting(self):
        calls["n"] += 1
        return real(self)

    monkeypatch.setattr(Ledger, "_segments", counting)
    for i in range(1, 500):
        lg.append("observation", "tool:probe", {"i": i})
    lg.close()

    assert calls["n"] == 0, (
        f"append globbed the segments dir {calls['n']} times in steady state; "
        "the tail path must be cached")


def test_append_still_rolls_over_and_recovers_with_cached_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "SEGMENT_LINES", 4)
    lg = Ledger(tmp_path / "ledger")
    lg.open()
    for i in range(9):
        lg.append("observation", "tool:probe", {"i": i})
    lg.close()

    segs = sorted((tmp_path / "ledger" / "segments").glob("*.jsonl"))
    assert len(segs) == 3, "rollover must still create a new segment every 4 lines"

    again = Ledger(tmp_path / "ledger")
    again.open()                                           # rebuilds the cache
    assert again.seq() == 9
    assert again.verify_chain()
    # A further append lands in the last segment without a fresh glob path bug.
    again.append("observation", "tool:probe", {"i": 9})
    assert again.seq() == 10
    assert again.verify_chain()
    again.close()
