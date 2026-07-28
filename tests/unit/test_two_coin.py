"""v0.3 Δ1/Δ2: two-coin actor model and context-grouped composition.

The failure these exist to prevent is on the record (bench/FINDINGS_6_8.md):
a single symmetric reliability scalar treated an always-yes observer as a
noisy coin, when its TRUE votes carry no information and its FALSE votes would
be decisive. These tests pin the asymmetry, the trusted-path-only movement,
the sub-additive grouping, and the invariants (I11 integrality, replay
determinism, I7 cross-fact correlation through shared observers).
"""

from __future__ import annotations

import math

import pytest

from candor.core.committed import reliability as R

BUDGET = 10_000
STMT = {"pred": "reachable", "args": ["a", "b"]}   # crisp in the seed world


# ── the parameter model ─────────────────────────────────────────────────────

def test_cold_start_trusts_votes_at_the_prior_operating_point():
    sens, fpr = R.rates((0, 0, 0, 0))
    assert sens == pytest.approx(0.95) and fpr == pytest.approx(0.05)
    assert R.log_lr(sens, fpr, True) == pytest.approx(math.log(19.0))
    # Newcomers are informative until scored — never LR = 1 (evidence must be
    # able to speak before it has been settled against).


def test_an_always_yes_actor_learns_an_uninformative_true_vote():
    # 50 settled-true all voted T, 50 settled-false all voted T
    conf = (50, 0, 50, 0)
    sens, fpr = R.rates(conf)
    # ~0.16 residual is the informative prior still speaking; it shrinks as
    # settlements accumulate and is nothing like a real vote's ~2.9.
    assert abs(R.log_lr(sens, fpr, True)) < 0.25
    assert R.log_lr(sens, fpr, False) < -1.0, \
        "the rare FALSE vote from an optimist must be decisive"


def test_an_exact_checker_learns_a_decisive_false_vote():
    # good on true claims, coin-flip on false ones (the v1 tool:exact shape)
    conf = (58, 2, 30, 30)
    sens, fpr = R.rates(conf)
    assert R.log_lr(sens, fpr, False) < -2.0
    assert 0.0 < R.log_lr(sens, fpr, True) < 1.5, \
        "its TRUE vote must be weak, not worthless and not decisive"


def test_symmetric_scalar_cannot_distinguish_optimist_from_coin():
    """The v1 failure, as arithmetic: identical agree-rates, and only the
    two-coin view can see that the optimist's dissent is worth ~10x more."""
    optimist = (50, 0, 50, 0)        # agrees on 50 of 100
    coin = (25, 25, 25, 25)          # also agrees on 50 of 100
    assert optimist[0] + optimist[3] == coin[0] + coin[3], \
        "one-coin sees these as the same actor"
    lr_opt_false = R.log_lr(*R.rates(optimist), False)
    lr_coin_false = R.log_lr(*R.rates(coin), False)
    assert lr_opt_false < -2.5, "optimist dissent is near-decisive"
    assert abs(lr_coin_false) < 0.5, "coin dissent is near-noise"


# ── movement: trusted path only (§3.12 unchanged) ───────────────────────────

def test_confusion_moves_only_through_settlement(seeded):
    idx = seeded.index
    for _ in range(5):
        seeded.observe(STMT, True, {}, actor="tool:probe")
    assert R.confusion(idx, "tool:probe") == (0, 0, 0, 0), \
        "observations alone must not move the confusion table"
    fid = seeded.fact_id_for(STMT)
    R.score_against_settlement(idx, fid, outcome=True)
    tp, fn, fp, tn = R.confusion(idx, "tool:probe")
    assert (tp, fn, fp, tn) == (5, 0, 0, 0)
    R.score_against_settlement(idx, fid, outcome=False)
    assert R.confusion(idx, "tool:probe") == (5, 0, 5, 0)


def test_confusion_cells_are_integers_in_storage(seeded):
    fid = seeded.fact_id_for(STMT)
    seeded.observe(STMT, True, {}, actor="tool:probe")
    R.score_against_settlement(seeded.index, fid, outcome=True)
    assert seeded.index.nonintegral_counts() == []


# ── Δ2: context-grouped sub-additive composition ────────────────────────────

def test_shared_context_votes_compose_subadditively():
    params = {"a1": (0.9, 0.1), "a2": (0.9, 0.1)}
    independent = R.grouped_logodds(
        [("a1", 1, "ctx-A"), ("a2", 1, "ctx-B")], params)
    shared = R.grouped_logodds(
        [("a1", 1, "ctx-S"), ("a2", 1, "ctx-S")], params)
    single = R.grouped_logodds([("a1", 1, "ctx-A")], params)
    assert independent == pytest.approx(2 * single)
    assert single < shared < independent, \
        "two votes from one context are more than one vote, less than two"
    assert shared == pytest.approx(independent / math.sqrt(2))


def test_uncontexted_votes_are_singleton_groups():
    params = {"a1": (0.9, 0.1), "a2": (0.9, 0.1)}
    assert R.grouped_logodds([("a1", 1, None), ("a2", 1, None)], params) == \
        pytest.approx(R.grouped_logodds([("a1", 1, "x"), ("a2", 1, "y")], params))


def test_composition_is_order_free():
    params = {"a1": (0.9, 0.1), "a2": (0.7, 0.2), "a3": (0.95, 0.6)}
    votes = [("a1", 1, "c1"), ("a2", 0, "c1"), ("a3", 1, None)]
    assert R.grouped_logodds(votes, params) == \
        pytest.approx(R.grouped_logodds(list(reversed(votes)), params))


# ── end to end through predict ──────────────────────────────────────────────

def test_votes_move_a_crisp_prediction_asymmetrically(seeded):
    idx = seeded.index
    # Teach the substrate two actors through settlements on a scratch fact.
    seeded.assert_({"pred": "reachable", "args": ["t", "u"], "stmt_type": "crisp"},
                   source="seed", actor="human:calvin")
    seeded.run_gate()
    scratch = {"pred": "reachable", "args": ["t", "u"]}
    sid = seeded.fact_id_for(scratch)
    for _ in range(30):
        seeded.observe(scratch, True, {}, actor="agent:optimist")
        seeded.observe(scratch, True, {}, actor="agent:careful")
    R.score_against_settlement(idx, sid, outcome=True)   # both look good...
    for _ in range(30):
        seeded.observe(scratch, True, {}, actor="agent:optimist")
        seeded.observe(scratch, False, {}, actor="agent:careful")
    R.score_against_settlement(idx, sid, outcome=False)  # ...only one is
    tp, fn, fp, tn = R.confusion(idx, "agent:optimist")
    assert fp > 0 and fn == 0, "the optimist has confessed its shape"

    # A fresh crisp fact: one TRUE vote from each actor must not move p equally.
    seeded.assert_({"pred": "reachable", "args": ["v", "w"], "stmt_type": "crisp"},
                   source="seed", actor="human:calvin")
    seeded.run_gate()
    fresh = {"pred": "reachable", "args": ["v", "w"]}
    base = seeded.predict(fresh, BUDGET).p
    seeded.observe(fresh, True, {}, actor="agent:optimist")
    p_optimist = seeded.predict(fresh, BUDGET).p
    seeded.observe(fresh, True, {}, actor="agent:careful")
    p_both = seeded.predict(fresh, BUDGET).p
    assert p_optimist - base < 0.15, \
        "a learned optimist's TRUE vote must barely move the posterior"
    assert p_both > p_optimist + 0.05, \
        "a learned careful actor's TRUE vote must move it more"


def test_prediction_with_votes_is_deterministic_and_order_free(seeded):
    seeded.observe(STMT, True, {"site": "lab"}, actor="tool:probe")
    seeded.observe(STMT, False, {"site": "sea"}, actor="agent:x")
    a = seeded.predict(STMT, BUDGET)
    b = seeded.predict(STMT, BUDGET)
    assert a.p == b.p and a.ci == b.ci


def test_replay_determinism_includes_the_confusion_table(seeded):
    fid = seeded.fact_id_for({"pred": "flaky_link", "args": ["c", "d"]})
    seeded.observe({"pred": "flaky_link", "args": ["c", "d"]}, True, {},
                   actor="tool:probe")
    seeded.register_oracle("verifier:t", "deterministic_total", "t", "h", "e")
    cid = seeded.claim({"pred": "flaky_link", "args": ["c", "d"]}, "external",
                       "verifier:t", due=0)
    seeded.resolve(cid, outcome=True)
    assert R.confusion(seeded.index, "tool:probe") != (0, 0, 0, 0)
    before = seeded.closure_hash()
    assert seeded.replay() == before, \
        "confusion is ledger-derived and must survive replay bit-for-bit"


def test_frequency_facts_ignore_the_confusion_machinery(seeded):
    stmt = {"pred": "flaky_link", "args": ["c", "d"]}
    seeded.index.execute(
        "INSERT OR REPLACE INTO actor_confusion(actor, frame, tp, fn, fp, tn) "
        "VALUES('tool:probe','external',0,50,50,0)")   # maximally anti-reliable
    seeded.index.commit()
    p0 = seeded.predict(stmt, BUDGET).p
    for _ in range(10):
        seeded.observe(stmt, True, {}, actor="tool:probe")
    assert seeded.predict(stmt, BUDGET).p >= p0 - 1e-9, \
        "alea trials keep the v0.2 discount semantics; two-coin is epi-only"