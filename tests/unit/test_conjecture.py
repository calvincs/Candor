"""Soft unification / conjecture — untrusted analogy generator (spec §4.3, I4).

`conjecture(goal, sim_budget)` reads behavioural signatures off the committed
tier and proposes analogical substitutions: for a goal `P(args)`, a predicate
`Q` that fires like `P` and already holds at those same `args` is offered as
`P(args)` "by analogy to `Q(args)`". Soft edges propose; they are never typed as
a Proof and they never write a committed number (I4/I1). These tests drive the
system-level `conjecture` through a small planted world so both the neighbour
search (budget, signature floor) and the known-atom filter are exercised.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def birds(sys_):
    """A tiny analogy world. `flies` and `glides` share a rule argument
    position, so their behavioural signatures overlap and they read as
    neighbours; `lonely` gets a single fact and no rule, so it stays below the
    signature-support floor (§3.13). Everything enters through the gate.
    """
    def reg(pred: str) -> None:
        sys_.assert_({"kind": "symbol", "pred": pred, "arity": 1,
                      "arg_types": ["any"]}, source="seed", actor="human:calvin")

    for pred in ("flies", "glides", "lonely"):
        reg(pred)
    sys_.run_gate()

    def fact(pred: str, arg: str) -> None:
        sys_.assert_({"pred": pred, "args": [arg], "stmt_type": "crisp"},
                     source="doc", actor="agent:x")

    for arg in ("hawk", "eagle"):      # flies: hawk, eagle
        fact("flies", arg)
    for arg in ("hawk", "squirrel"):   # glides: hawk, squirrel
        fact("glides", arg)
    fact("lonely", "x")

    # A rule mentioning flies and glides at the SAME argument position is what
    # makes their signatures overlap (§4.3). It need not be a *good* rule — only
    # admitted — so it carries just-passing held-out evidence and no more.
    sys_.assert_({"kind": "rule",
                  "head": {"pred": "airborne", "args": ["?a"]},
                  "body": {"literals": [{"pred": "flies", "args": ["?a"]},
                                        {"pred": "glides", "args": ["?a"]}]},
                  "holdout": {"hits": 9, "misses": 1}},
                 source="seed", actor="human:calvin")
    sys_.run_gate()
    return sys_


def test_conjecture_offers_an_analogy_that_clears_the_budget(birds):
    # squirrel glides but is not known to fly; glides ~ flies (cosine 0.2),
    # so under a 0.15 budget the engine proposes flies(squirrel) via glides.
    out = birds.conjecture({"pred": "flies", "args": ["squirrel"]}, 0.15)
    assert len(out) == 1
    c = out[0]
    assert c["goal"] == {"pred": "flies", "args": ["squirrel"]}
    # The analogy substitutes the PREDICATE, keeping the same individual.
    assert c["via"] == {"pred": "glides", "args": ["squirrel"]}
    assert c["via"]["pred"] != c["goal"]["pred"]
    assert c["sim"] == pytest.approx(0.2)
    assert c["sim"] >= 0.15
    # It is labelled a conjecture over a soft edge, budget-caveated — never a proof.
    assert c["quality"] == "conjecture"
    assert c["edge_kinds"] == ["soft"]
    assert c["caveats"] == ["analogy_budget"]


def test_conjecture_is_refused_when_the_budget_outruns_the_neighbour(birds):
    # Same world, but a 0.30 budget sits above the best neighbour's 0.2
    # similarity: no neighbour survives, so nothing is proposed.
    assert birds.conjecture({"pred": "flies", "args": ["squirrel"]}, 0.30) == []


def test_conjecture_needs_the_analogue_to_be_a_known_atom(birds):
    # glides is a neighbour of flies for BOTH individuals, but the conjecture is
    # only offered where the analogue actually holds. glides(squirrel) is known;
    # glides(moon) is not — so flies(moon) yields nothing at the same budget,
    # isolating the known-atom filter from the similarity budget.
    assert birds.conjecture({"pred": "flies", "args": ["squirrel"]}, 0.15)
    assert birds.conjecture({"pred": "flies", "args": ["moon"]}, 0.15) == []


def test_conjecture_is_blind_below_the_signature_support_floor(birds):
    # §3.13 cold start: analogy-from-nothing is exactly the licence refused.
    # `lonely` has one signature component (< the floor of 2), and a wholly
    # unregistered predicate has none — both stay blind even at a zero budget.
    assert birds.conjecture({"pred": "lonely", "args": ["x"]}, 0.0) == []
    assert birds.conjecture({"pred": "never_seen", "args": ["z"]}, 0.0) == []


def test_conjecture_proposes_but_commits_no_number(birds):
    # I4/I1 periphery firewall: reading conjectures appends nothing to the
    # ledger, moves no committed number, and never turns the goal into a fact.
    head_before = birds.ledger_head()
    hash_before = birds.closure_hash()

    out = birds.conjecture({"pred": "flies", "args": ["squirrel"]}, 0.15)
    assert out, "precondition: this world does produce a conjecture"

    assert birds.ledger_head() == head_before
    assert birds.closure_hash() == hash_before
    # The conjectured goal is still just a proposal — not an admitted fact.
    assert birds.fact_id_for({"pred": "flies", "args": ["squirrel"]}) is None
