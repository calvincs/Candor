"""Kernel-forgery paths: the trusted core vs. committed state it never admitted
(spec §3.6, invariant I1).

Two distinct defences guard the committed numbers, and they are NOT the same
mechanism — this file pins the boundary between them honestly:

  * The ledger is the only primary artifact (I1). Every other store is a
    materialized view, so a row poked straight into SQLite has no backing event
    and simply vanishes when `replay()` re-folds committed state from the log.
    Facts, rules and closure atoms are each covered here (the oracle case is
    already in tests/unit/test_two_coin.py and is not duplicated).

  * The proof checker (`kernel.check`) re-verifies a derivation against the
    premises and rules it is HANDED. It catches a proof that cites something it
    was not given — but provenance (did this rule pass the gate?) is emphatically
    not its job, and the hash chain is consistency, not tamper-evidence. The
    tests below assert exactly where each guard bites and, just as deliberately,
    where it does not.
"""

from __future__ import annotations

import json

import pytest

from candor.core import kernel
from candor.core.closure import Rule, atom_text
from candor.core.hashing import canon_json

TRANSITIVE = Rule.parse(
    "rule:t",
    json.dumps({"pred": "reachable", "args": ["?x", "?z"]}),
    json.dumps({"literals": [{"pred": "reachable", "args": ["?x", "?y"]},
                             {"pred": "reachable", "args": ["?y", "?z"]}]}))

GUARDED = Rule.parse(
    "rule:g",
    json.dumps({"pred": "hot", "args": ["?x"]}),
    json.dumps({"literals": [{"pred": "temp", "args": ["?x", "?t"]}],
                "guards": [{"var": "?t", "op": ">", "value": 300}]}))

REACH_AC = {"pred": "reachable", "args": ["a", "c"]}


# ── injected committed state does not survive replay (I1) ────────────────────

def test_a_fact_written_straight_to_sqlite_does_not_survive_replay(seeded):
    """A committed `facts` row forged past the gate corrupts the LIVE index — the
    derived store is not tamper-proof — yet has no backing ledger event, so a
    full replay recomputes it away. `verify_chain()` stays blind throughout: it
    attests the log, which the forgery never touched."""
    baseline = seeded.closure_hash()
    assert seeded.ledger.verify_chain()

    seeded.index.execute(
        "INSERT INTO facts(id, pred, args_json, stmt_type, kind, structural, numeric) "
        "VALUES('fact:forged','reachable',?, 'crisp','exact','admitted','accumulating')",
        (canon_json(["x", "z"]),))
    seeded.index.commit()
    seeded._closure = None                      # force the closure to rematerialize

    # The live derived index now believes a fact nobody admitted...
    assert seeded.closure().contains("reachable", ("x", "z"))
    assert seeded.closure_hash() != baseline
    # ...and the hash chain cannot tell — it never saw the index write.
    assert seeded.ledger.verify_chain(), "the chain verifies the log, not the index"

    # replay() is the I1 oracle: it re-folds committed state from the log alone,
    # so the un-backed forgery is gone and the fingerprint returns to baseline.
    assert seeded.replay() == baseline
    assert not seeded.closure().contains("reachable", ("x", "z"))


def test_a_rule_injected_into_the_committed_tier_fires_live_but_replay_erases_it(seeded):
    """A forged `rules` row is the sharpest case: it fires in the live closure AND
    the kernel ACCEPTS the resulting proof — because `check` verifies internal
    consistency against the rules it is handed, never whether a rule passed the
    gate (§3.6). Provenance is the ledger's job, not the checker's, and replay is
    what actually erases the injection."""
    baseline = seeded.closure_hash()
    # The seed world has reachable(a,b) and reachable(b,c) but no transitive rule.
    assert seeded.derive(REACH_AC, 10_000).status == "not_entailed"

    seeded.index.execute(
        "INSERT INTO rules(id, head_json, body_json, specificity, structural, "
        "numeric, admitted_at) VALUES(?,?,?,?,?,?,?)",
        ("rule:forged",
         canon_json({"pred": "reachable", "args": ["?x", "?z"]}),
         canon_json({"literals": [{"pred": "reachable", "args": ["?x", "?y"]},
                                  {"pred": "reachable", "args": ["?y", "?z"]}]}),
         0, "admitted", "accumulating", 0))
    seeded.index.commit()
    seeded._closure = None

    # The injected rule fires, and derive() (which runs the kernel on its output)
    # returns a clean proof: the kernel is satisfied because the rule is present
    # in the set it checks against, forged provenance notwithstanding.
    got = seeded.derive(REACH_AC, 10_000)
    assert got.status == "proof"
    assert seeded.closure().contains("reachable", ("a", "c"))

    # No admission event backs the rule, so replay folds it out of existence.
    assert seeded.replay() == baseline
    assert seeded.derive(REACH_AC, 10_000).status == "not_entailed"


def test_a_forged_closure_atom_is_transient_derived_state(seeded):
    """closure_atoms is a rebuilt view of the materialization. A phantom row poked
    into it changes the fingerprint only while the closure is CACHED — proving
    `closure_hash()` is a consistency check over the derived tables, not
    tamper-evidence. Rematerialization (and, definitively, replay) delete and
    recompute the table, so the phantom cannot persist."""
    baseline = seeded.closure_hash()            # caches the closure, fills the table
    phantom = atom_text("reachable", ("ghost", "town"))
    seeded.index.execute(
        "INSERT OR REPLACE INTO closure_atoms(atom, basis) VALUES(?, 'exact')",
        (phantom,))
    seeded.index.commit()

    # With the closure still cached, the hash trusts the tampered table verbatim.
    assert seeded.closure_hash() != baseline

    # rebuild_closure DELETEs closure_atoms and recomputes it from admitted
    # facts/rules; the phantom has no support and disappears.
    seeded._closure = None
    assert seeded.closure_hash() == baseline
    # and a full replay from the log agrees.
    assert seeded.replay() == baseline


# ── the kernel refuses forged proofs (§3.6) ─────────────────────────────────

def test_kernel_recursively_rejects_a_forged_premise_buried_in_a_valid_tree():
    """The check is recursive: a transitive proof of reachable(a,c) whose outer
    step and first leaf are impeccable is still rejected when the SECOND leaf,
    reachable(b,c), was never admitted. The same tree checks clean once (b,c) is
    admitted — so the rejection is the buried leaf, not a structural flaw."""
    proof = {
        "conclusion": ["reachable", ["a", "c"]], "rule": "rule:t",
        "premises": [
            {"conclusion": ["reachable", ["a", "b"]], "premises": ["fact:ab"]},
            {"conclusion": ["reachable", ["b", "c"]], "premises": ["fact:bc"]}],
        "edge_kinds": ["exact"]}
    with pytest.raises(kernel.KernelReject):
        kernel.check(proof, {("reachable", ("a", "b"))}, {"rule:t": TRANSITIVE})

    assert kernel.check(proof,
                        {("reachable", ("a", "b")), ("reachable", ("b", "c"))},
                        {"rule:t": TRANSITIVE})


def test_kernel_refuses_structurally_forged_proof_objects():
    """Three distinct forgeries of the proof SHAPE, each caught: an object that is
    not a proof at all; a rule step that cites cheap base-fact ids where
    sub-derivations are required; and a premise whose predicate does not match the
    body literal it is supposed to discharge."""
    admitted = {("reachable", ("a", "b")), ("reachable", ("b", "c"))}

    with pytest.raises(kernel.KernelReject):
        kernel.check({"premises": []}, set(), {})          # no "conclusion"
    with pytest.raises(kernel.KernelReject):
        kernel.check("not-a-proof", set(), {})             # not even a dict

    bare_ids = {"conclusion": ["reachable", ["a", "c"]], "rule": "rule:t",
                "premises": ["fact:ab", "fact:bc"], "edge_kinds": ["exact"]}
    with pytest.raises(kernel.KernelReject):
        kernel.check(bare_ids, admitted, {"rule:t": TRANSITIVE})

    wrong_pred = {
        "conclusion": ["reachable", ["a", "c"]], "rule": "rule:t",
        "premises": [
            {"conclusion": ["linked", ["a", "b"]], "premises": ["fact:ab"]},
            {"conclusion": ["reachable", ["b", "c"]], "premises": ["fact:bc"]}],
        "edge_kinds": ["exact"]}
    with pytest.raises(kernel.KernelReject):
        kernel.check(wrong_pred, admitted, {"rule:t": TRANSITIVE})


def test_kernel_rejects_a_conclusion_smuggled_past_a_failing_guard():
    """A guarded rule cannot be made to license a conclusion whose binding fails
    the guard: hot(b) 'derived' from temp(b, 280) unifies through the body, but
    the kernel re-runs the guard ?t > 300 and refuses the licensing step."""
    forged = {"conclusion": ["hot", ["b"]], "rule": "rule:g",
              "premises": [{"conclusion": ["temp", ["b", "280"]],
                            "premises": ["fact:tb"]}],
              "edge_kinds": ["exact"]}
    with pytest.raises(kernel.KernelReject):
        kernel.check(forged, {("temp", ("b", "280"))}, {"rule:g": GUARDED})
