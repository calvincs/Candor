"""M2: quotas roll over per epoch, and cover every actor-content write path.

Before this, `epoch` was pinned to 0 everywhere, so `obs_quota_per_epoch` was
really a *lifetime* cap that never reset — an actor that hit it was locked out
forever, contradicting the documented "per-epoch" bound (docs/api.md,
docs/getting-started.md). Quota is derived state outside the closure hash (like
the fsync policy), so this changes no replay number; it only makes the
per-epoch semantics real. The epoch is a deterministic function of the event
timestamp — which is in the ledger — so replay reproduces every bucket exactly.
"""

from __future__ import annotations

import pytest

from candor.core import apply as apply_mod
from candor.system import CandorSystem, QuotaExceeded

EPOCH = apply_mod.QUOTA_EPOCH_MS
STMT = {"pred": "flaky_link", "args": ["c", "d"]}


def _seed(root):
    m = CandorSystem(root)
    m.assert_({"pred": "flaky_link", "args": ["c", "d"], "stmt_type": "frequency"},
              source="s", actor="human:me")
    m.run_gate()
    return m


def test_epoch_of_is_ts_floor_division():
    assert apply_mod.epoch_of(0) == 0
    assert apply_mod.epoch_of(EPOCH - 1) == 0
    assert apply_mod.epoch_of(EPOCH) == 1
    assert apply_mod.epoch_of(2 * EPOCH + 5) == 2


def test_observation_quota_rolls_over_next_epoch(tmp_path):
    m = _seed(tmp_path / "store")
    m.set_actor_quota("tool:probe", obs_per_epoch=2)
    m.observe(STMT, True, {}, actor="tool:probe", ts=1_000)   # epoch 0
    m.observe(STMT, False, {}, actor="tool:probe", ts=2_000)  # epoch 0 — full
    with pytest.raises(QuotaExceeded):
        m.observe(STMT, True, {}, actor="tool:probe", ts=3_000)  # epoch 0 — over
    # A later epoch starts fresh — the actor is not locked out forever.
    seq = m.observe(STMT, True, {}, actor="tool:probe", ts=EPOCH + 1_000)
    m.close()
    assert seq > 0, "quota did not roll over into the next epoch"


def test_usage_is_bucketed_by_epoch(tmp_path):
    m = _seed(tmp_path / "store")
    m.set_actor_quota("tool:probe", obs_per_epoch=100)
    m.observe(STMT, True, {}, actor="tool:probe", ts=1_000)          # epoch 0
    m.observe(STMT, True, {}, actor="tool:probe", ts=EPOCH + 1_000)  # epoch 1
    m.observe(STMT, True, {}, actor="tool:probe", ts=EPOCH + 2_000)  # epoch 1
    rows = {(r["epoch"], r["kind"]): r["used"] for r in m.index.query(
        "SELECT epoch, kind, used FROM quota_usage WHERE actor='tool:probe'")}
    m.close()
    assert rows[(0, "observation")] == 1
    assert rows[(1, "observation")] == 2


def test_same_epoch_burst_still_locks_out(tmp_path):
    # Default (wall-clock ts) keeps a live burst inside one epoch, so the
    # historical flooding bound is unchanged when no explicit ts is supplied.
    m = _seed(tmp_path / "store")
    m.set_actor_quota("tool:probe", obs_per_epoch=3)
    for _ in range(3):
        m.observe(STMT, True, {}, actor="tool:probe")
    with pytest.raises(QuotaExceeded):
        m.observe(STMT, True, {}, actor="tool:probe")
    m.close()


def test_rollover_is_replay_stable(tmp_path):
    # The epoch is a pure function of the ledger's own timestamps, so a
    # ledger-only rebuild reproduces the same per-epoch buckets (I1/I3).
    m = _seed(tmp_path / "store")
    m.set_actor_quota("tool:probe", obs_per_epoch=100)
    m.observe(STMT, True, {}, actor="tool:probe", ts=1_000)
    m.observe(STMT, True, {}, actor="tool:probe", ts=EPOCH + 1_000)
    before = {(r["epoch"], r["kind"]): r["used"] for r in m.index.query(
        "SELECT epoch, kind, used FROM quota_usage WHERE actor='tool:probe'")}
    m.replay()
    after = {(r["epoch"], r["kind"]): r["used"] for r in m.index.query(
        "SELECT epoch, kind, used FROM quota_usage WHERE actor='tool:probe'")}
    m.close()
    assert before == after, "per-epoch quota buckets did not survive replay"
