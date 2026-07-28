"""Phase 2 FIX M1 — writer lock (spec §7 single writer).

Two Ledger instances on the same root each hold their own in-memory _seq/_head
and silently fork the on-disk chain; the next open truncates one, losing
committed events with no signal. open() must take an exclusive advisory lock and
release it on close()/destroy().
"""

from __future__ import annotations

import pytest

from candor.core.ledger import Ledger, LedgerError

try:
    import fcntl as _fcntl  # noqa: F401  platform capability probe
    _HAVE_FCNTL = True
except ImportError:          # pragma: no cover - non-POSIX
    _HAVE_FCNTL = False


def _requires_fcntl():
    if not _HAVE_FCNTL:
        pytest.skip("no fcntl on this platform: writer lock degrades to a no-op")


def test_m1_second_open_on_same_root_raises(tmp_path):
    _requires_fcntl()
    root = tmp_path / "ledger"
    a = Ledger(root)
    a.open()
    try:
        b = Ledger(root)
        with pytest.raises(LedgerError):
            b.open()
    finally:
        a.close()

    # After the first writer closes, a fresh writer opens fine.
    c = Ledger(root)
    c.open()
    assert c.verify_chain() is True
    c.close()


def test_m1_build_close_reopen_still_works(tmp_path):
    """A normal sequential close -> reopen must keep working (flock released on
    close)."""
    root = tmp_path / "ledger"
    lg = Ledger(root)
    lg.open()
    lg.append("assertion", "human:calvin", {"x": 1})
    lg.append("observation", "tool:probe", {"x": 2})
    lg.close()

    again = Ledger(root)
    again.open()
    assert again.seq() == 2
    assert again.verify_chain() is True
    again.close()


def test_m1_reopen_after_destroy(tmp_path):
    """destroy() releases the lock so the same root can be reopened."""
    _requires_fcntl()
    root = tmp_path / "ledger"
    lg = Ledger(root)
    lg.open()
    lg.append("assertion", "human:calvin", {"x": 1})
    lg.destroy()

    again = Ledger(root)
    again.open()          # must not raise: lock was released by destroy()
    assert again.seq() == 0
    again.close()
