"""Credit assignment: §4.4 integer blame to the argmax-sensitivity *eligible*
component.

Exercises `CandorSystem._blame_target` / `_blame_eligible` directly — the argmax
walk and the eligibility filter — and the end-to-end path where `resolve()`
records a `blame_target` into the resolution payload. Fact and rule ids are the
deterministic content hashes the gate assigns, so every assertion pins a concrete
id or a concrete `None`.
"""

from __future__ import annotations


def _fact(sys_, pred, args, stmt_type="crisp", basis=None):
    """Assert a single fact candidate; `basis` becomes its committed `kind`."""
    stmt = {"pred": pred, "args": args, "stmt_type": stmt_type}
    if basis is not None:
        stmt["basis"] = basis
    sys_.assert_(stmt, source="seed", actor="human:calvin")


def _transitivity_rule(sys_):
    """reachable(x,z) :- reachable(x,y), reachable(y,z); passes the §6 gate."""
    sys_.assert_({"kind": "rule",
                  "head": {"pred": "reachable", "args": ["?x", "?z"]},
                  "body": {"literals": [
                      {"pred": "reachable", "args": ["?x", "?y"]},
                      {"pred": "reachable", "args": ["?y", "?z"]}]},
                  "holdout": {"hits": 9, "misses": 1}},
                 source="seed", actor="human:calvin")


def _resolution_payload(sys_, seq):
    """The committed resolution event's payload, read back off the ledger."""
    ev = next(e for e in sys_.ledger.read_all() if e.seq == seq)
    return sys_.ledger.payload(ev.payload_hash)


def test_blame_target_picks_the_most_sensitive_eligible_fact(seeded):
    """A clear argmax over eligible base facts is the blame target (§4.4)."""
    ab = seeded.fact_id_for({"pred": "reachable", "args": ["a", "b"]})
    bc = seeded.fact_id_for({"pred": "reachable", "args": ["b", "c"]})
    assert seeded._blame_eligible(ab) and seeded._blame_eligible(bc)
    assert seeded._blame_target({ab: 0.10, bc: 0.90}) == bc
    assert seeded._blame_target({ab: 0.90, bc: 0.10}) == ab


def test_blame_target_skips_a_definitional_fact_with_the_top_sensitivity(sys_):
    """kind='definitional' is blame-ineligible: the argmax skips it (§4.4)."""
    _fact(sys_, "reachable", ["a", "b"])                        # exact
    _fact(sys_, "reachable", ["b", "c"], basis="definitional")  # definitional
    sys_.run_gate()
    exact = sys_.fact_id_for({"pred": "reachable", "args": ["a", "b"]})
    defin = sys_.fact_id_for({"pred": "reachable", "args": ["b", "c"]})
    assert sys_._blame_eligible(exact) is True
    assert sys_._blame_eligible(defin) is False
    # The definitional fact dwarfs the exact one in sensitivity, yet blame lands
    # on the eligible exact fact — the filter, not the raw argmax, decides.
    assert sys_._blame_target({exact: 0.01, defin: 0.99}) == exact


def test_blame_target_returns_none_when_nothing_is_eligible(sys_):
    """Only ineligible components (definitional + unknown id) => None; {} => None."""
    _fact(sys_, "reachable", ["b", "c"], basis="definitional")
    sys_.run_gate()
    defin = sys_.fact_id_for({"pred": "reachable", "args": ["b", "c"]})
    assert sys_._blame_eligible(defin) is False
    assert sys_._blame_target({defin: 0.99, "fact:ghost": 0.80}) is None
    assert sys_._blame_target({}) is None


def test_blame_eligibility_of_fact_kinds(seeded):
    """Eligible: a live admitted 'exact' fact. Ineligible: unknown id, frozen fact."""
    ab = seeded.fact_id_for({"pred": "reachable", "args": ["a", "b"]})
    assert seeded._blame_eligible(ab) is True
    assert seeded._blame_eligible("fact:deadbeef") is False      # never admitted
    # numeric='frozen' is schema-legal but no public path sets it on a fact; write
    # it straight to the committed row to exercise the numeric=='frozen' guard.
    seeded.index.execute("UPDATE facts SET numeric='frozen' WHERE id=?", (ab,))
    assert seeded._blame_eligible(ab) is False


def test_blame_eligibility_of_rules(sys_):
    """A live admitted rule is eligible; an unknown rule id and a frozen rule are not."""
    _fact(sys_, "reachable", ["a", "b"])
    _fact(sys_, "reachable", ["b", "c"])
    sys_.run_gate()
    _transitivity_rule(sys_)
    sys_.run_gate()
    rule_id = sys_.index.one(
        "SELECT id FROM rules WHERE structural='admitted'")["id"]
    assert sys_._blame_eligible(rule_id) is True
    assert sys_._blame_eligible("rule:99999") is False
    # A frozen rule is ineligible (same white-box note as the frozen fact above).
    sys_.index.execute("UPDATE rules SET numeric='frozen' WHERE id=?", (rule_id,))
    assert sys_._blame_eligible(rule_id) is False


def test_blame_target_tie_breaks_on_the_lexically_first_id(seeded):
    """Equal sensitivities: the strict `>` keeps the first id in sorted order (§4.4)."""
    ab = seeded.fact_id_for({"pred": "reachable", "args": ["a", "b"]})
    bc = seeded.fact_id_for({"pred": "reachable", "args": ["b", "c"]})
    assert seeded._blame_target({ab: 0.5, bc: 0.5}) == min(ab, bc)


def test_resolution_blames_the_argmax_sensitivity_component(sys_):
    """End-to-end: resolve() records the argmax-sensitivity eligible fact (§4.4)."""
    _fact(sys_, "reachable", ["a", "b"])
    _fact(sys_, "reachable", ["b", "c"])
    sys_.run_gate()
    _transitivity_rule(sys_)
    sys_.run_gate()

    claim_id = sys_.claim({"pred": "reachable", "args": ["a", "c"]},
                          frame="internal", criterion="tool:x", due=10 ** 12)
    # reachable(a,c) is entailed via transitivity: a two-fact proof whose per-fact
    # sensitivities are frozen into proof_steps at claim time.
    steps = {r["fact_id"]: r["sensitivity"] for r in sys_.index.query(
        "SELECT fact_id, sensitivity FROM proof_steps WHERE claim_id=?",
        (claim_id,))}
    assert len(steps) == 2 and all(k.startswith("fact:") for k in steps)

    seq = sys_.resolve(claim_id, outcome=True)
    blame = _resolution_payload(sys_, seq)["blame_target"]

    # Reproduce _blame_target's exact selection over the recorded (all-eligible)
    # steps: sorted walk keeping the first strictly-greater sensitivity.
    best, best_s = None, -1.0
    for k in sorted(steps):
        if steps[k] > best_s:
            best, best_s = k, steps[k]
    assert blame == best
    assert sys_._blame_eligible(blame) is True
