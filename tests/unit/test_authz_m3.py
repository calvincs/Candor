"""M3: opt-in authorization for privileged writes.

`authority`/`actor`/`source` are attribution labels, not authenticated
identities — CANDOR's default posture is advisory, and its trust boundary is
the process (see SECURITY.md). An embedding application that needs access
control registers a policy with `set_authz`; from then on the privileged writes
(pin, redact, retract_source, register_oracle, set_reliability) consult it and
raise `Unauthorized` when it returns False.

The check runs at the API boundary BEFORE any ledger append, so a denied call
mutates nothing — the ledger head does not move and replay of an existing
ledger never re-checks authorization (every event already in the chain was, by
construction, admitted under whatever policy was active when it was written).
"""

from __future__ import annotations

import pytest

from candor.system import CandorSystem, Unauthorized

OPERATORS = {"human:operator", "human:calvin"}


def _policy(principal, op):
    return principal in OPERATORS


def _seed(root):
    m = CandorSystem(root)
    m.set_actor_quota("tool:probe", obs_per_epoch=10_000, cand_per_epoch=10_000)
    for stmt in ({"pred": "reachable", "args": ["a", "b"], "stmt_type": "crisp"},
                 {"pred": "flaky_link", "args": ["c", "d"], "stmt_type": "frequency"}):
        m.assert_(stmt, source="seed", actor="human:calvin")
    m.run_gate()
    return m


def _a_fact(m):
    return m.index.one("SELECT id FROM facts ORDER BY id LIMIT 1")["id"]


def _an_obs_payload(m):
    m.observe({"pred": "flaky_link", "args": ["c", "d"]}, True, {}, actor="tool:probe")
    return m.index.one(
        "SELECT payload_hash FROM events WHERE kind='observation' "
        "ORDER BY seq DESC LIMIT 1")["payload_hash"]


# ── default posture: advisory, nothing enforced ─────────────────────────────

def test_default_is_advisory_all_privileged_writes_succeed(tmp_path):
    m = _seed(tmp_path / "store")
    fid = _a_fact(m)
    # No policy registered: every label is accepted, as before.
    assert m.pin(fid, "+", "trusted", authority="anyone") > 0
    m.set_reliability("tool:probe", "external", 1.0, 1.0, authority="anyone")
    m.register_oracle("orc:x", "deterministic_total", "ref", "ch", "eh", authority="anyone")
    m.retract_source("tool:probe", "cleanup", authority="anyone")
    m.close()


# ── opt-in enforcement ──────────────────────────────────────────────────────

def test_pin_denied_for_unlisted_principal(tmp_path):
    m = _seed(tmp_path / "store")
    m.set_authz(_policy)
    fid = _a_fact(m)
    with pytest.raises(Unauthorized):
        m.pin(fid, "+", "trusted", authority="mallory")
    # An allowed principal still goes through.
    assert m.pin(fid, "+", "trusted", authority="human:operator") > 0
    m.close()


def test_all_privileged_ops_are_gated(tmp_path):
    m = _seed(tmp_path / "store")
    m.set_authz(_policy)
    fid = _a_fact(m)
    phash = _an_obs_payload(m)
    with pytest.raises(Unauthorized):
        m.pin(fid, "-", "r", authority="mallory")
    with pytest.raises(Unauthorized):
        m.set_reliability("tool:probe", "external", 1.0, 1.0, authority="mallory")
    with pytest.raises(Unauthorized):
        m.register_oracle("orc:y", "deterministic_total", "ref", "ch", "eh", authority="mallory")
    with pytest.raises(Unauthorized):
        m.retract_source("tool:probe", "r", authority="mallory")
    with pytest.raises(Unauthorized):
        m.redact(phash, authority="mallory")
    # The authorized operator can do each of them.
    assert m.pin(fid, "-", "r", authority="human:operator") > 0
    m.set_reliability("tool:probe", "external", 1.0, 1.0, authority="human:operator")
    m.register_oracle("orc:y", "deterministic_total", "ref", "ch", "eh", authority="human:operator")
    m.close()


def test_denied_write_mutates_nothing(tmp_path):
    m = _seed(tmp_path / "store")
    m.set_authz(_policy)
    fid = _a_fact(m)
    head_before = m.ledger_head()
    with pytest.raises(Unauthorized):
        m.pin(fid, "+", "r", authority="mallory")
    head_after = m.ledger_head()
    events = m.health()["events"]
    m.close()
    assert head_after == head_before, "a denied privileged write moved the ledger head"
    assert any(e.get("kind") == "unauthorized" and e.get("op") == "pin"
               for e in events), "denied write left no audit trace"


def test_denied_redact_leaves_payload_intact(tmp_path):
    m = _seed(tmp_path / "store")
    m.set_authz(_policy)
    phash = _an_obs_payload(m)
    with pytest.raises(Unauthorized):
        m.redact(phash, authority="mallory")
    still_there = m.ledger.payload(phash)
    m.close()
    assert still_there is not None, "a denied redact deleted the payload anyway"


def test_set_authz_none_restores_advisory(tmp_path):
    m = _seed(tmp_path / "store")
    fid = _a_fact(m)
    m.set_authz(_policy)
    with pytest.raises(Unauthorized):
        m.pin(fid, "+", "r", authority="mallory")
    m.set_authz(None)
    assert m.pin(fid, "+", "r", authority="mallory") > 0, "advisory mode not restored"
    m.close()
