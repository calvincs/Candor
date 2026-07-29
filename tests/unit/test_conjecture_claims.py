"""Δ12 — the conjecture loop closed: postulate → validate → implement.

`conjecture()` used to be a dead end — proposals returned to the caller and
nothing consumed them. With `commit=True` each analogy becomes a CLAIM by
`agent:conjecture` whose predicted_p is the analog's own earned probability
(the transfer is audited, similarity is never a probability), under the
distinct `conjecture/v1` predictor class (I9). Settling it scores the analogy
engine's calibration curve; settling TRUE asserts the goal as a fact candidate
through the gate (I10). These tests drive the whole loop on the birds world
and prove the read-only path and I4 stayed byte-identical.
"""

from __future__ import annotations

import json

import pytest

from candor.core import calibration as calibration_mod

GOAL = {"pred": "flies", "args": ["squirrel"]}


@pytest.fixture()
def birds(sys_):
    """The test_conjecture.py analogy world: glides ~ flies via a shared rule
    position; glides(squirrel) is known, flies(squirrel) is the conjecturable
    gap."""
    def reg(pred):
        sys_.assert_({"kind": "symbol", "pred": pred, "arity": 1,
                      "arg_types": ["any"]}, source="seed", actor="human:calvin")
    for pred in ("flies", "glides"):
        reg(pred)
    sys_.run_gate()
    for pred, arg in (("flies", "hawk"), ("flies", "eagle"),
                      ("glides", "hawk"), ("glides", "squirrel")):
        sys_.assert_({"pred": pred, "args": [arg], "stmt_type": "crisp"},
                     source="doc", actor="agent:x")
    sys_.assert_({"kind": "rule",
                  "head": {"pred": "airborne", "args": ["?a"]},
                  "body": {"literals": [{"pred": "flies", "args": ["?a"]},
                                        {"pred": "glides", "args": ["?a"]}]},
                  "holdout": {"hits": 9, "misses": 1}},
                 source="seed", actor="human:calvin")
    sys_.run_gate()
    return sys_


def _claim_row(m, claim_id):
    return m.index.one("SELECT * FROM claims WHERE id=?", (claim_id,))


def test_commit_files_a_claim_with_the_transferred_probability(birds):
    out = birds.conjecture(GOAL, 0.15, commit=True)
    assert len(out) == 1 and "claim_id" in out[0]
    row = _claim_row(birds, out[0]["claim_id"])
    assert row is not None
    assert row["predictor_class"] == calibration_mod.CONJECTURE_PREDICTOR_CLASS
    assert row["frame"] == "internal"
    assert row["resolved_ts"] is None
    # the transferred number IS the analog's own prediction, not the similarity
    via_p = birds.predict(out[0]["via"], budget=10_000).p
    assert row["predicted_p"] == pytest.approx(via_p)
    assert row["predicted_p"] != pytest.approx(out[0]["sim"])


def test_commit_is_idempotent_while_unresolved(birds):
    first = birds.conjecture(GOAL, 0.15, commit=True)
    head = birds.ledger_head()
    second = birds.conjecture(GOAL, 0.15, commit=True)
    assert second[0]["claim_id"] == first[0]["claim_id"]
    assert birds.ledger_head() == head, "re-committing must append nothing"


def test_true_settlement_implements_the_postulate_through_the_gate(birds):
    out = birds.conjecture(GOAL, 0.15, commit=True)
    claim_id = out[0]["claim_id"]
    assert birds.fact_id_for(GOAL) is None, "precondition: not yet a fact"
    birds.resolve(claim_id, outcome=True)
    # the settlement asserted a candidate; the gate has not run yet
    assert birds.fact_id_for(GOAL) is None
    birds.run_gate()
    fid = birds.fact_id_for(GOAL)
    assert fid is not None, "validated postulate must be admitted via the gate"
    # provenance: proposed by the conjecture engine, sourced to its claim
    cand = birds.index.one(
        "SELECT proposer, span_ref FROM candidates WHERE kind='fact' "
        "AND status='admitted' AND body_json LIKE ? ORDER BY event_seq DESC",
        ('%"squirrel"%',))
    assert cand["proposer"] == "agent:conjecture"
    assert cand["span_ref"] == f"conjecture:{claim_id}"
    # and the analogy engine's own curve was scored (I9: its class, not WMC's)
    cal = birds.index.query(
        "SELECT * FROM calibration WHERE predictor_class=?",
        (calibration_mod.CONJECTURE_PREDICTOR_CLASS,))
    assert len(cal) == 1 and cal[0]["n"] == 1 and cal[0]["k"] == 1


def test_false_settlement_implements_nothing_but_scores_the_curve(birds):
    out = birds.conjecture(GOAL, 0.15, commit=True)
    birds.resolve(out[0]["claim_id"], outcome=False)
    birds.run_gate()
    assert birds.fact_id_for(GOAL) is None, "a refuted analogy must not land"
    cal = birds.index.query(
        "SELECT * FROM calibration WHERE predictor_class=?",
        (calibration_mod.CONJECTURE_PREDICTOR_CLASS,))
    assert len(cal) == 1 and cal[0]["n"] == 1 and cal[0]["k"] == 0
    # a fresh commit may now re-file (the old claim is resolved)
    again = birds.conjecture(GOAL, 0.15, commit=True)
    assert again[0]["claim_id"] != out[0]["claim_id"]


def test_readonly_path_is_unchanged_and_i4_holds(birds):
    head = birds.ledger_head()
    hash_before = birds.closure_hash()
    out = birds.conjecture(GOAL, 0.15)
    assert out and "claim_id" not in out[0]
    assert birds.ledger_head() == head
    assert birds.closure_hash() == hash_before


def test_the_whole_loop_survives_replay(birds):
    out = birds.conjecture(GOAL, 0.15, commit=True)
    birds.resolve(out[0]["claim_id"], outcome=True)
    birds.run_gate()
    before = birds.closure_hash()
    assert birds.replay() == before
