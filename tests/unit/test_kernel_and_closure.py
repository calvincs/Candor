"""Proof checker kernel + closure engine (spec §3.5, §3.6, I4).

The kernel is the only eval in the system that catches bugs in the engine
itself, so it is tested against forged proofs, not just honest ones.
"""

from __future__ import annotations

import json

import pytest

from candor.core import kernel
from candor.core.closure import Rule, backward, materialize

TRANSITIVE = Rule.parse(
    "rule:t",
    json.dumps({"pred": "reachable", "args": ["?x", "?z"]}),
    json.dumps({"literals": [{"pred": "reachable", "args": ["?x", "?y"]},
                             {"pred": "reachable", "args": ["?y", "?z"]}]}))

BASE = [("reachable", ["a", "b"], "fact:ab"), ("reachable", ["b", "c"], "fact:bc")]


def test_materialization_computes_the_transitive_closure():
    clo = materialize(BASE, [TRANSITIVE])
    assert clo.contains("reachable", ("a", "c"))
    assert ("reachable", ("a", "c")) not in clo.base


def test_recursion_terminates_and_carries_support():
    clo = materialize(BASE + [("reachable", ["c", "d"], "fact:cd")], [TRANSITIVE])
    assert clo.contains("reachable", ("a", "d"))
    support = clo.support[("reachable", ("a", "d"))][0]
    assert "rule:t" in support


def test_stratified_negation():
    rule = Rule.parse(
        "rule:n", json.dumps({"pred": "isolated", "args": ["?x"]}),
        json.dumps({"literals": [{"pred": "node", "args": ["?x"]},
                                 {"pred": "linked", "args": ["?x"], "negated": True}]}))
    clo = materialize([("node", ["a"], "f1"), ("node", ["b"], "f2"),
                       ("linked", ["a"], "f3")], [rule])
    assert clo.contains("isolated", ("b",))
    assert not clo.contains("isolated", ("a",))


def test_guards_filter_the_join():
    rule = Rule.parse(
        "rule:g", json.dumps({"pred": "hot", "args": ["?x"]}),
        json.dumps({"literals": [{"pred": "temp", "args": ["?x", "?t"]}],
                    "guards": [{"var": "?t", "op": ">", "value": 300}]}))
    clo = materialize([("temp", ["a", "373.15"], "f1"), ("temp", ["b", "280"], "f2")],
                      [rule])
    assert clo.contains("hot", ("a",))
    assert not clo.contains("hot", ("b",))


def test_kernel_accepts_the_engine_output():
    clo = materialize(BASE, [TRANSITIVE])
    out = backward("reachable", ("a", "c"), clo, [TRANSITIVE], budget=1000)
    assert out.proved
    assert kernel.check(out.proof, set(clo.base), {"rule:t": TRANSITIVE})


def test_kernel_rejects_a_premise_that_is_not_admitted():
    forged = {"conclusion": ["reachable", ["a", "z"]], "premises": ["fact:zz"],
              "edge_kinds": ["exact"]}
    with pytest.raises(kernel.KernelReject):
        kernel.check(forged, {("reachable", ("a", "b"))}, {})


def test_kernel_rejects_a_conclusion_the_rule_does_not_license():
    clo = materialize(BASE, [TRANSITIVE])
    forged = {
        "conclusion": ["reachable", ["a", "z"]], "rule": "rule:t",
        "premises": [
            {"conclusion": ["reachable", ["a", "b"]], "premises": ["fact:ab"]},
            {"conclusion": ["reachable", ["b", "c"]], "premises": ["fact:bc"]}],
        "edge_kinds": ["exact"]}
    with pytest.raises(kernel.KernelReject):
        kernel.check(forged, set(clo.base), {"rule:t": TRANSITIVE})


def test_kernel_rejects_an_unknown_rule_citation():
    with pytest.raises(kernel.KernelReject):
        kernel.check({"conclusion": ["reachable", ["a", "c"]], "rule": "rule:ghost",
                      "premises": []}, set(), {})


def test_kernel_rejects_a_short_premise_list():
    clo = materialize(BASE, [TRANSITIVE])
    with pytest.raises(kernel.KernelReject):
        kernel.check({"conclusion": ["reachable", ["a", "c"]], "rule": "rule:t",
                      "premises": [{"conclusion": ["reachable", ["a", "b"]],
                                    "premises": ["fact:ab"]}]},
                     set(clo.base), {"rule:t": TRANSITIVE})


def test_budget_honesty_is_a_property_of_the_search():
    clo = materialize(BASE, [TRANSITIVE])
    starved = backward("reachable", ("a", "c"), clo, [TRANSITIVE], budget=1)
    assert not starved.proved
    assert not starved.exhausted, "truncation must never masquerade as absence (I4)"

    absent = backward("reachable", ("z", "q"), clo, [TRANSITIVE], budget=10_000)
    assert not absent.proved and absent.exhausted


def test_derivation_quality_downgrades_on_a_flagged_premise():
    clo = materialize(BASE, [TRANSITIVE])
    out = backward("reachable", ("a", "c"), clo, [TRANSITIVE], budget=1000)
    assert kernel.quality(out.proof, flagged=set(), narrow=set()) == "proof"
    assert kernel.quality(out.proof, flagged={"fact:ab"}, narrow=set()) == \
        "proof-modulo-unknown-context"
    soft = dict(out.proof)
    soft["edge_kinds"] = ["soft"]
    assert kernel.quality(soft, set(), set()) == "conjecture"
