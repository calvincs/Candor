"""Reopening a store must be a no-op on every derived number (I1, I3).

C1 regression: `open()` folded the whole log onto a non-empty index, so every
increment writer (fact_counts, quota_usage, calibration, confusion) re-added a
full copy of history on each process reopen. Counts inflated, predictions
drifted, and quotas could falsely lock an actor out.
"""

from __future__ import annotations

from candor.system import CandorSystem


def _build_store(root):
    m = CandorSystem(root)
    for stmt in ({"pred": "reachable", "args": ["a", "b"], "stmt_type": "crisp"},
                 {"pred": "flaky_link", "args": ["c", "d"], "stmt_type": "frequency"}):
        m.assert_(stmt, source="seed", actor="human:calvin")
    m.run_gate()
    for _ in range(10):
        m.observe({"pred": "flaky_link", "args": ["c", "d"]}, True, {},
                  actor="tool:probe")
    return m


def _quota_used(m, actor, kind="observation"):
    row = m.index.one(
        "SELECT used FROM quota_usage WHERE actor=? AND epoch=0 AND kind=?",
        (actor, kind))
    return int(row["used"]) if row else 0


def test_reopen_does_not_double_count(tmp_path):
    root = tmp_path / "store"
    m = _build_store(root)
    fid = m.fact_id_for({"pred": "flaky_link", "args": ["c", "d"]})
    raw_before = m.raw_counts(fid)
    hash_before = m.closure_hash()
    quota_before = _quota_used(m, "tool:probe")
    m.close()

    m2 = CandorSystem(root)
    fid2 = m2.fact_id_for({"pred": "flaky_link", "args": ["c", "d"]})
    raw_after = m2.raw_counts(fid2)
    hash_after = m2.closure_hash()
    quota_after = _quota_used(m2, "tool:probe")
    m2.close()

    assert raw_after == raw_before, "raw counts doubled across a reopen"
    assert quota_after == quota_before, "quota usage doubled across a reopen"
    assert hash_after == hash_before, "closure hash drifted across a reopen"


def test_reopening_three_times_keeps_counts_stable(tmp_path):
    root = tmp_path / "store"
    m = _build_store(root)
    fid = m.fact_id_for({"pred": "flaky_link", "args": ["c", "d"]})
    raw_before = m.raw_counts(fid)
    hash_before = m.closure_hash()
    quota_before = _quota_used(m, "tool:probe")
    m.close()

    for _ in range(3):
        m = CandorSystem(root)
        fid = m.fact_id_for({"pred": "flaky_link", "args": ["c", "d"]})
        assert m.raw_counts(fid) == raw_before
        assert _quota_used(m, "tool:probe") == quota_before
        assert m.closure_hash() == hash_before
        m.close()
