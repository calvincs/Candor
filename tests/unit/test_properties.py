"""§6.2 property invariants the shipped harness does not itself assert.

`irrelevance` is named in §6.2 and in the Stage-4 gate but has no test in
`candor_conformance.py`. It is asserted here, at the system level, alongside
system-level restatements of the properties the harness checks only through the
prediction engine.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

BUDGET = 10_000
STMT = {"pred": "flaky_link", "args": ["c", "d"]}


def _unrelated(system, n, outcome=True):
    system.assert_({"pred": "unrelated", "args": ["x"], "stmt_type": "frequency"},
                   source="seed", actor="human:calvin")
    system.run_gate()
    for _ in range(n):
        system.observe({"pred": "unrelated", "args": ["x"]}, outcome, {},
                       actor="tool:probe")


@settings(max_examples=10, deadline=None)
@given(n=st.integers(min_value=1, max_value=30))
def test_irrelevance_a_fact_outside_the_support_cannot_move_the_result(n, tmp_path_factory):
    from candor.system import CandorSystem
    root = tmp_path_factory.mktemp("irrelevance")
    system = CandorSystem(root)
    try:
        for stmt in ({"pred": "flaky_link", "args": ["c", "d"],
                      "stmt_type": "frequency"},):
            system.assert_(stmt, source="seed", actor="human:calvin")
        system.run_gate()
        before = system.predict(STMT, BUDGET).p
        _unrelated(system, n)
        after = system.predict(STMT, BUDGET).p
        assert after == pytest.approx(before, abs=1e-12), \
            "a fact outside the proof's support moved the result"
    finally:
        system.close()


def test_irrelevance_holds_across_a_shared_predicate(seeded):
    before = seeded.predict(STMT, BUDGET).p
    for _ in range(20):
        seeded.observe({"pred": "flaky_link", "args": ["z", "z"]}, True, {},
                       actor="tool:probe")
    assert seeded.predict(STMT, BUDGET).p == pytest.approx(before, abs=1e-12)


def test_monotonicity_at_the_system_level(seeded):
    p0 = seeded.predict(STMT, BUDGET).p
    seen = [p0]
    for _ in range(8):
        seeded.observe(STMT, True, {}, actor="tool:probe")
        seen.append(seeded.predict(STMT, BUDGET).p)
    assert all(b >= a - 1e-12 for a, b in zip(seen, seen[1:]))


def test_permutation_at_the_system_level(tmp_path):
    from candor.system import CandorSystem
    outcomes = [True, True, False, True, False, False, True]
    results = []
    for order in (outcomes, list(reversed(outcomes)),
                  [outcomes[i] for i in (3, 0, 6, 1, 5, 2, 4)]):
        system = CandorSystem(tmp_path / f"store{len(results)}")
        system.assert_({"pred": "flaky_link", "args": ["c", "d"],
                        "stmt_type": "frequency"}, source="seed", actor="human:calvin")
        system.run_gate()
        for outcome in order:
            system.observe(STMT, outcome, {}, actor="tool:probe")
        results.append(system.predict(STMT, BUDGET).p)
        system.close()
    assert results[0] == pytest.approx(results[1], abs=1e-12)
    assert results[0] == pytest.approx(results[2], abs=1e-12)


def test_composition_purity_holds_after_every_observation(seeded):
    fid = seeded.fact_id_for(STMT)
    seeded.set_reliability("tool:probe", "external", 6.0, 4.0)   # E[rel] = 0.6
    for i in range(10):
        seeded.observe(STMT, i % 3 != 0, {}, actor="tool:probe")
        raw = seeded.raw_counts(fid)
        n = sum(n for (_, ch), (n, _) in raw.items() if ch == "alea")
        k = sum(k for (_, ch), (_, k) in raw.items() if ch == "alea")
        composed = seeded.composed_counts(fid)
        assert composed.alea_n == pytest.approx(0.6 * n)
        assert composed.alea_k == pytest.approx(0.6 * k)


def test_snapshot_completeness_survives_structural_change(seeded):
    p0 = seeded.predict(STMT, BUDGET)
    seeded.assert_({"pred": "reachable", "args": ["q", "r"], "stmt_type": "crisp"},
                   source="later", actor="agent:x")
    seeded.run_gate()
    for _ in range(5):
        seeded.observe(STMT, False, {}, actor="tool:probe")
    assert seeded.predict(STMT, BUDGET).p != pytest.approx(p0.p, abs=1e-6)
    assert seeded.predict_at(STMT, p0.snapshot_id).p == pytest.approx(p0.p, abs=1e-12)


def test_budget_honesty_never_reports_absence_for_truncation(seeded):
    for budget in (1, 2, 3, 5, 8, 13):
        out = seeded.derive({"pred": "reachable", "args": ["a", "zzz"]}, budget)
        if out.status == "not_entailed":
            assert out.search_exhausted
