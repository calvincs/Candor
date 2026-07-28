"""Golden fixtures for the gate (spec §6.3 "Gate steps").

One fixture per step 1–7 where exactly that step fails; admission requires all
seven; the rejection is recorded with the failing step rather than discarded.
"""

from __future__ import annotations

import pytest

from candor.core import gate


def _decide(sys_, kind, body, proposer="human:calvin"):
    return gate.evaluate(sys_.index, "cand:test", kind, body, proposer)


def test_admission_requires_all_seven_steps(seeded):
    seeded.assert_({"pred": "reachable", "args": ["c", "e"], "stmt_type": "crisp"},
                   source="doc", actor="agent:x")
    runs = seeded.run_gate()
    assert [r["status"] for r in runs] == ["admitted"]
    assert seeded.fact_id_for({"pred": "reachable", "args": ["c", "e"]})


def test_step1_unregistered_predicate(seeded):
    d = _decide(seeded, "fact", {"pred": "never_seen", "args": ["x"]})
    assert (d.status, d.failing_step) == ("rejected", 1)
    assert "registry" in d.reason


def test_step1_arity_mismatch(seeded):
    d = _decide(seeded, "fact", {"pred": "reachable", "args": ["a"]})
    assert (d.status, d.failing_step) == ("rejected", 1)
    assert "arity" in d.reason


def test_step2_canonicalization_failure(sys_):
    sys_.assert_({"kind": "symbol", "pred": "boils_at", "arity": 2,
                  "arg_types": ["substance", "temperature"],
                  "canonical_units": {"1": "K"}}, source="seed", actor="human:calvin")
    sys_.run_gate()
    d = _decide(sys_, "fact", {"pred": "boils_at", "args": ["water", "warmish"]})
    assert (d.status, d.failing_step) == ("rejected", 2)


def test_step2_success_normalizes_to_the_registry_unit(sys_):
    sys_.assert_({"kind": "symbol", "pred": "boils_at", "arity": 2,
                  "arg_types": ["substance", "temperature"],
                  "canonical_units": {"1": "K"}}, source="seed", actor="human:calvin")
    sys_.assert_({"pred": "boils_at", "args": ["water", "212F"], "stmt_type": "crisp"},
                 source="doc", actor="agent:x")
    sys_.run_gate()
    fid = sys_.fact_id_for({"pred": "boils_at", "args": ["water", "373.15K"]})
    assert fid is not None, "212F must be stored as 373.15K"
    # …and the same fact is readable under any equivalent notation.
    assert sys_.fact_id_for({"pred": "boils_at", "args": ["water", "100C"]}) == fid


def test_step3_verifier_that_fails_its_own_vectors(sys_):
    d = _decide(sys_, "verifier", {
        "entry": "check", "code": "def check(x):\n    return x + 1\n",
        "vectors": [[[1], 99]]})
    assert (d.status, d.failing_step) == ("rejected", 3)


def test_step3_verifier_that_passes_is_admitted_as_an_oracle(sys_):
    sys_.assert_({"kind": "verifier", "oracle_id": "verifier:inc", "entry": "check",
                  "code": "def check(x):\n    return x + 1\n",
                  "vectors": [[[1], 2], [[41], 42]]},
                 source="seed", actor="human:calvin")
    runs = sys_.run_gate()
    assert [r["status"] for r in runs] == ["admitted"]
    row = sys_.index.one("SELECT kind, code_hash FROM oracles WHERE id=?",
                         ("verifier:inc",))
    assert row is not None and row["kind"] == "deterministic_total"


def test_step3_sandbox_denies_the_import_machinery(sys_):
    d = _decide(sys_, "verifier", {
        "entry": "check",
        "code": "import os\ndef check(x):\n    return os.getpid()\n",
        "vectors": [[[1], 1]]})
    assert (d.status, d.failing_step) == ("rejected", 3)


def test_step4_pin_vetoes_a_candidate(seeded):
    fid = seeded.fact_id_for({"pred": "reachable", "args": ["a", "b"]})
    seeded.pin(fid, polarity="-", reason="known-bad", authority="human:calvin")
    d = _decide(seeded, "fact", {"pred": "reachable", "args": ["a", "b"]})
    assert (d.status, d.failing_step) == ("rejected", 4)


def test_step5_held_out_evidence_check_on_a_rule(seeded):
    d = _decide(seeded, "rule", {
        "head": {"pred": "reachable", "args": ["?x", "?z"]},
        "body": {"literals": [{"pred": "reachable", "args": ["?x", "?y"]}]},
        "holdout": {"hits": 1, "misses": 9}})
    assert (d.status, d.failing_step) == ("rejected", 5)


def test_step5_alias_below_the_behavioural_similarity_floor(seeded):
    d = _decide(seeded, "alias", {"canonical": "flaky_link", "alias": "lossy_link",
                                  "basis": "behavioral", "sim": 0.10})
    assert (d.status, d.failing_step) == ("rejected", 5)


def test_step5_guard_support_floor(seeded):
    d = _decide(seeded, "guard", {
        "head": {"pred": "flaky_link", "args": ["c", "d"]},
        "body": {"literals": [], "guards": [{"var": "?p", "op": ">", "value": 1}]},
        "support": {"left": 3, "right": 40},
        "mdl": {"dl_guard": 1.0, "dl_residual_given_guard": 1.0, "dl_residual": 90.0}})
    assert (d.status, d.failing_step) == ("rejected", 5)


def test_step6_mdl_rejects_a_guard_that_costs_more_than_it_compresses(seeded):
    overfit = {
        "head": {"pred": "flaky_link", "args": ["c", "d"]},
        "body": {"literals": [], "guards": [{"var": "?p", "op": ">", "value": 1}]},
        "support": {"left": 20, "right": 20},
        "mdl": {"dl_guard": 64.0, "dl_residual_given_guard": 38.0, "dl_residual": 40.0}}
    assert _decide(seeded, "guard", overfit).failing_step == 6

    genuine = dict(overfit)
    genuine["mdl"] = {"dl_guard": 4.0, "dl_residual_given_guard": 12.0,
                      "dl_residual": 40.0}
    assert _decide(seeded, "guard", genuine).status == "admitted"


def test_step7_contradiction_against_a_certain_fact(sys_):
    sys_.assert_({"kind": "constraint", "ctype": "mutex",
                  "body": {"pred": "link_state", "exclusive_values": ["up", "down"]}},
                 source="seed", actor="human:calvin")
    sys_.assert_({"pred": "link_state", "args": ["c", "up"], "stmt_type": "crisp"},
                 source="seed", actor="human:calvin")
    sys_.run_gate()
    fid = sys_.fact_id_for({"pred": "link_state", "args": ["c", "up"]})
    sys_.pin(fid, polarity="+", reason="ground truth", authority="human:calvin")

    d = _decide(sys_, "fact", {"pred": "link_state", "args": ["c", "down"],
                               "stmt_type": "crisp"})
    assert (d.status, d.failing_step) == ("rejected", 7)


def test_rejections_are_recorded_not_discarded(seeded):
    seeded.assert_({"pred": "reachable", "args": ["only-one-arg"],
                    "stmt_type": "crisp"}, source="doc", actor="agent:x")
    runs = seeded.run_gate()
    rejected = [r for r in runs if r["status"] == "rejected"]
    assert rejected and rejected[0]["failing_step"] == 1
    row = seeded.index.one(
        "SELECT status, failing_step, reason FROM candidates WHERE id=?",
        (rejected[0]["candidate_id"],))
    assert row["status"] == "rejected" and row["failing_step"] == 1
    assert any(e.get("kind") == "gate_rejection" for e in seeded.health()["events"])


# ── supersede-with-valid-time (§4.4) ────────────────────────────────────────
# Previously the only candidate kind admitted unconditionally, so a periphery
# false positive became committed history unopposed. See CLAIMS_HARDENING F4.

def _supersede_body(seeded, **over):
    fid = seeded.fact_id_for({"pred": "flaky_link", "args": ["c", "d"]})
    body = {"fact_id": fid, "changepoint_index": 20, "valid_to": 1_700_000_000,
            "support": {"before": 21, "after": 19}, "pvalue": 1e-6,
            "reason": "one-way level change"}
    body.update(over)
    return body


def test_a_located_and_significant_regime_change_is_admitted(seeded):
    d = _decide(seeded, "supersede_valid_time", _supersede_body(seeded))
    assert d.status == "admitted"


def test_supersede_without_a_located_date_is_rejected(seeded):
    d = _decide(seeded, "supersede_valid_time", _supersede_body(seeded, valid_to=None))
    assert (d.status, d.failing_step) == ("rejected", 5)
    assert "WHEN" in d.reason


def test_supersede_on_a_thin_regime_is_rejected(seeded):
    d = _decide(seeded, "supersede_valid_time",
                _supersede_body(seeded, support={"before": 40, "after": 3}))
    assert (d.status, d.failing_step) == ("rejected", 5)


def test_supersede_that_is_not_significant_is_rejected(seeded):
    d = _decide(seeded, "supersede_valid_time", _supersede_body(seeded, pvalue=0.4))
    assert (d.status, d.failing_step) == ("rejected", 6)
    d = _decide(seeded, "supersede_valid_time", _supersede_body(seeded, pvalue=None))
    assert (d.status, d.failing_step) == ("rejected", 6)


def test_supersede_of_an_unknown_fact_is_rejected(seeded):
    d = _decide(seeded, "supersede_valid_time", _supersede_body(seeded, fact_id="fact:nope"))
    assert (d.status, d.failing_step) == ("rejected", 1)
