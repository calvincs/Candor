"""Prediction engine: determinism, channel composition, constraint conditioning."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from candor.periphery import predict as P

BUDGET = 10_000


def _problem(stmt_type="frequency", epi=(99.0, 1.0), alea=(1.0, 1.0)):
    state = P.FactState("fact:one", stmt_type, epi, alea)
    return P.Problem("fact:one", [frozenset({"fact:one"})], {"fact:one": state})


def test_permutation_of_the_sample_grid_is_a_permutation():
    perm = P.permutation("fact:one|validity", 128)
    assert sorted(perm) == list(range(128))
    assert P.permutation("fact:one|validity", 128) == perm
    assert P.permutation("fact:two|validity", 128) != perm


def test_prediction_is_a_pure_function_of_the_problem():
    a = P.run(_problem(), BUDGET)
    b = P.run(_problem(), BUDGET)
    assert a.p == b.p and a.ci == b.ci


def test_frequency_fact_recovers_the_posterior_mean():
    # 30 successes out of 40 trials, uniform prior -> E[theta] = 31/42
    out = P.run(_problem(alea=(31.0, 11.0)), BUDGET)
    assert out.p == pytest.approx(0.99 * 31 / 42, abs=0.02)


def test_ci_is_an_interval_around_the_posterior():
    out = P.run(_problem(alea=(31.0, 11.0)), BUDGET)
    lo, hi = out.ci
    assert 0.0 < lo < out.p < hi < 1.0


@settings(max_examples=40, deadline=None)
@given(extra=st.integers(min_value=1, max_value=50))
def test_monotone_in_support(extra):
    base = P.run(_problem(alea=(1.0, 1.0)), BUDGET).p
    more = P.run(_problem(alea=(1.0 + extra, 1.0)), BUDGET).p
    assert more >= base - 1e-12


def test_crisp_only_proof_has_a_degenerate_aleatoric_channel():
    out = P.run(_problem(stmt_type="crisp"), BUDGET)
    assert out.channels["aleatoric"] == 0.0
    assert out.p == pytest.approx(0.99, abs=0.02)


def test_negative_pin_is_the_only_hard_zero():
    state = P.FactState("fact:one", "crisp", (99.0, 1.0), (1.0, 1.0),
                        pinned_negative=True)
    out = P.run(P.Problem("fact:one", [frozenset({"fact:one"})], {"fact:one": state}),
                BUDGET)
    assert out.p == 0.0


def test_shared_facts_are_not_summed_across_proofs():
    """Two proofs sharing a fact: inclusion–exclusion, never addition (§3.9)."""
    states = {f: P.FactState(f, "crisp", (99.0, 1.0), (1.0, 1.0))
              for f in ("fact:a", "fact:b", "fact:c")}
    problem = P.Problem("goal", [frozenset({"fact:a", "fact:b"}),
                                 frozenset({"fact:a", "fact:c"})], states)
    out = P.run(problem, BUDGET)
    assert out.p <= 1.0
    # P(ab ∪ ac) = P(a)·(1 - (1-P(b))(1-P(c))) ≈ 0.99 · (1 - 0.01²)
    assert out.p == pytest.approx(0.99 * (1 - 0.01 ** 2), abs=0.02)


def test_mutex_group_produces_rejections_and_renormalizes():
    states = {f: P.FactState(f, "crisp", (99.0, 1.0), (1.0, 1.0))
              for f in ("fact:up", "fact:down")}
    problem = P.Problem("fact:up", [frozenset({"fact:up"})], states,
                        constraint_groups=[["fact:up", "fact:down"]])
    out = P.run(problem, BUDGET)
    assert out.rejection_rate > 0.9, "two near-certain mutex facts are in tension"
    assert 0.0 < out.p < 1.0


def test_no_constraint_means_no_rejections():
    assert P.run(_problem(), BUDGET).rejection_rate == 0.0


def test_sensitivity_reports_the_fact_that_would_flip_the_conclusion():
    states = {"fact:a": P.FactState("fact:a", "crisp", (99.0, 1.0), (1.0, 1.0)),
              "fact:b": P.FactState("fact:b", "crisp", (99.0, 1.0), (1.0, 1.0))}
    out = P.run(P.Problem("goal", [frozenset({"fact:a", "fact:b"})], states), BUDGET)
    assert set(out.sensitivity) == {"fact:a", "fact:b"}
    assert all(v > 0.9 for v in out.sensitivity.values())


def test_budget_degrades_the_sample_count_first():
    cheap = P.run(_problem(), budget=64)
    rich = P.run(_problem(), budget=BUDGET)
    assert cheap.samples_used < rich.samples_used
    assert cheap.p == pytest.approx(rich.p, abs=0.05)


def test_epistemic_spread_shrinks_as_validity_sharpens():
    vague = P.run(_problem(stmt_type="crisp", epi=(2.0, 2.0)), BUDGET)
    sharp = P.run(_problem(stmt_type="crisp", epi=(999.0, 1.0)), BUDGET)
    assert vague.channels["epistemic"] > sharp.channels["epistemic"]
