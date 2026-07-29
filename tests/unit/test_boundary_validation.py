"""Invalid enum arguments must be rejected at the API boundary, before the
ledger append (C3).

The index carries CHECK constraints the API never checked: pins.polarity IN
('+','-') and claims.frame IN ('internal','external'). A bad value used to be
appended to the ledger first and only rejected when apply_event hit the CHECK —
so the event was permanently in the chain and every future open()->_refold()
re-raised. The store was bricked with no recovery. The fix validates at the
boundary so the bad event is never recorded.
"""

from __future__ import annotations

import pytest

from candor.system import CandorSystem


def _healthy_store(root):
    m = CandorSystem(root)
    for stmt in ({"pred": "reachable", "args": ["a", "b"], "stmt_type": "crisp"},
                 {"pred": "flaky_link", "args": ["c", "d"], "stmt_type": "frequency"}):
        m.assert_(stmt, source="seed", actor="human:calvin")
    m.run_gate()
    return m


def test_bad_pin_polarity_raises_and_does_not_brick(tmp_path):
    root = tmp_path / "store"
    m = _healthy_store(root)
    fid = m.fact_id_for({"pred": "reachable", "args": ["a", "b"]})
    head_before = m.ledger_head()

    with pytest.raises(ValueError):
        m.pin(fid, "?", "typo", "human:me")

    assert m.ledger_head() == head_before, "the bad pin event was still recorded"
    assert m.ledger.verify_chain()
    m.close()

    # The store must still open — the poison event must never have been written.
    m2 = CandorSystem(root)
    assert m2.ledger.verify_chain()
    m2.close()


def test_bad_claim_frame_raises_and_does_not_brick(tmp_path):
    root = tmp_path / "store"
    m = _healthy_store(root)
    head_before = m.ledger_head()

    with pytest.raises(ValueError):
        m.claim({"pred": "reachable", "args": ["a", "b"]}, "Internal",
                "tool:closure", due=0)

    assert m.ledger_head() == head_before, "the bad claim event was still recorded"
    assert m.ledger.verify_chain()
    m.close()

    m2 = CandorSystem(root)
    assert m2.ledger.verify_chain()
    m2.close()


def test_a_good_pin_and_claim_still_work(tmp_path):
    """The valid path must be untouched by the new boundary guards."""
    root = tmp_path / "store"
    m = _healthy_store(root)
    fid = m.fact_id_for({"pred": "reachable", "args": ["a", "b"]})
    m.pin(fid, "-", "known-bad", "human:me")
    cid = m.claim({"pred": "reachable", "args": ["a", "b"]}, "internal",
                  "tool:closure", due=0)
    assert cid.startswith("claim:")
    m.close()
