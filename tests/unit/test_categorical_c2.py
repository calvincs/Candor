"""Categorical facts — Stage C2: the posterior + predict() with the first-class
UNKNOWN mass (design_categorical.md §2/§2.3/§2.4/§5/§6/§8 "Stage C2").

C2 is READ-TIME ONLY: it turns the C1 per-value integer tallies into a proper
distribution over {seen values} ∪ {unknown} and returns it from predict(). The
model is LOCKED (decision 1) to the Dirichlet-process / CRP predictive with a
single pre-registered concentration alpha and the Pitman–Yor discount pinned to
d=0 — so P(unknown)=alpha/(N+alpha) is a function of N alone (NOT of the
distinct-value count; that separation is the deferred d>0 upgrade).

    P(v)       = n_v / (N + alpha)      for each seen value v
    P(unknown) = alpha / (N + alpha)    the never-seen mass, carried as 1 − ΣP(v)

Per-value credible intervals are the Beta MARGINALS of the Dirichlet, reusing
`betamath` verbatim (the same deterministic numerics the frequency path uses).

The gates enforced here:
  * SUMS TO ONE — Σ values.p + unknown.p == 1.0 exactly, composed in the
    canonical order predict emits (ORDER BY value, unknown last).
  * REDUCTION — thin data ⇒ large unknown; N→large ⇒ unknown→0 and mass
    concentrates on the seen values; a still-discovering fact keeps more unknown
    than a heavily-confirmed one.
  * INTERVALS — each ci is a deterministic Beta quantile pair with lo≤p≤hi that
    narrows as n_v grows; the unknown slice carries its own interval.
  * DETERMINISM (I3/I8) — replay() reproduces the distribution; predict_at at a
    recorded snapshot reproduces the FULL distribution bit-for-bit; alpha rides
    the snapshot id.
  * ADDITIVITY (§6) — a crisp/frequency prediction is a PredictOutcome, byte-
    identical whether or not categorical data shares the store.
"""

from __future__ import annotations

import random

import pytest

import candor.core.committed.counts as counts_mod
from candor.core.betamath import betaincinv
from candor.core.committed.counts import CAT_CI_HI, CAT_CI_LO, CATEGORICAL_ALPHA
from candor.system import CategoricalPrediction, PredictOutcome, CandorSystem

CAT_STMT = {"pred": "resolves", "args": ["login"]}


# ── fixtures / helpers ─────────────────────────────────────────────────────────

def _categorical_store(root) -> CandorSystem:
    m = CandorSystem(root)
    for a in ("tool:probe", "tool:liar"):
        m.set_actor_quota(a, obs_per_epoch=100_000, cand_per_epoch=100_000)
    m.assert_({"pred": "resolves", "args": ["login"], "stmt_type": "categorical"},
              source="seed", actor="tool:probe")
    m.run_gate()
    return m


def _observe(m, values, actor="tool:probe"):
    for v in values:
        m.observe(CAT_STMT, ctx={}, actor=actor, value=v)


def _canonical_total(pred: CategoricalPrediction) -> float:
    """Σ over the distribution in the EXACT canonical order predict emits — the
    values in ORDER BY value, then the unknown slice last. The residual
    construction (§2.1) makes this ordinary left-fold sum to 1.0 bit-exactly.
    (Python 3.12's `sum()` is Neumaier-compensated and reorders the error, so the
    'exact by construction' claim is specifically about this canonical fold.)"""
    t = 0.0
    for slice_ in pred.values.values():
        t += slice_.p
    t += pred.unknown.p
    return t


# ══ SUMS TO ONE ════════════════════════════════════════════════════════════════

def test_captcha_block_distribution_is_the_target_statement(tmp_path):
    """The worked example: captcha×8, block×2 (N=10, alpha=1, denom=11)."""
    m = _categorical_store(tmp_path / "s")
    _observe(m, ["captcha"] * 8 + ["block"] * 2)
    p = m.predict(CAT_STMT, budget=1000)

    assert isinstance(p, CategoricalPrediction)
    # Canonical order: ORDER BY value → 'block' before 'captcha'.
    assert list(p.values) == ["block", "captcha"]
    assert p.total_observations == 10

    assert p.values["captcha"].p == pytest.approx(8 / 11)
    assert p.values["block"].p == pytest.approx(2 / 11)
    assert p.unknown.p == pytest.approx(1 / 11)

    # Exact by construction, in canonical order.
    assert _canonical_total(p) == 1.0

    # Every slice's point estimate lies inside its own interval.
    for slice_ in (*p.values.values(), p.unknown):
        assert slice_.ci[0] <= slice_.p <= slice_.ci[1]
    m.close()


def test_sums_to_one_exactly_for_many_stores(tmp_path):
    """Σ values.p + unknown.p == 1.0 (exact) across many random stores, including
    the never-observed N=0 store."""
    for trial in range(120):
        rng = random.Random(trial)
        m = _categorical_store(tmp_path / f"s{trial}")
        vocab = [f"v{i}" for i in range(rng.randint(1, 7))]
        for _ in range(rng.randint(0, 60)):        # 0 exercises the N=0 branch
            m.observe(CAT_STMT, ctx={}, actor="tool:probe", value=rng.choice(vocab))
        p = m.predict(CAT_STMT, budget=1000)
        assert _canonical_total(p) == 1.0, (trial, p.total_observations)
        m.close()


def test_never_observed_fact_is_entirely_unknown(tmp_path):
    m = _categorical_store(tmp_path / "s")
    p = m.predict(CAT_STMT, budget=1000)
    assert p.values == {}
    assert p.unknown.p == 1.0
    assert p.unknown.ci == (1.0, 1.0)
    assert p.total_observations == 0
    assert "no_observations" in p.caveats
    m.close()


# ══ REDUCTION (the whole point) ════════════════════════════════════════════════

def test_thin_data_gives_large_unknown_thick_data_gives_tiny(tmp_path):
    """N=1 ⇒ unknown ≈ alpha/(1+alpha); N=1000 ⇒ unknown < 0.01 and mass
    concentrates on the seen value."""
    thin = _categorical_store(tmp_path / "thin")
    _observe(thin, ["captcha"])
    p_thin = thin.predict(CAT_STMT, budget=1000)
    assert p_thin.unknown.p == pytest.approx(CATEGORICAL_ALPHA / (1 + CATEGORICAL_ALPHA))
    assert p_thin.unknown.p == pytest.approx(0.5)
    assert p_thin.values["captcha"].p == pytest.approx(0.5)
    thin.close()

    thick = _categorical_store(tmp_path / "thick")
    _observe(thick, ["captcha"] * 1000)
    p_thick = thick.predict(CAT_STMT, budget=1000)
    assert p_thick.unknown.p < 0.01
    assert p_thick.unknown.p == pytest.approx(1 / 1001)
    assert p_thick.values["captcha"].p > 0.99             # mass concentrated
    thick.close()


def test_unknown_mass_is_strictly_monotone_decreasing_in_N(tmp_path):
    """unknown = alpha/(N+alpha) shrinks with every additional observation."""
    prev = 1.0
    for n in (1, 2, 5, 10, 50, 200, 1000):
        m = _categorical_store(tmp_path / f"n{n}")
        _observe(m, ["captcha"] * n)
        u = m.predict(CAT_STMT, budget=1000).unknown.p
        assert u == pytest.approx(1 / (n + 1))
        assert u < prev, (n, u, prev)
        prev = u
        m.close()


def test_still_discovering_keeps_more_unknown_than_heavily_confirmed(tmp_path):
    """A fact that keeps seeing NEW values (few confirmations each) retains a
    larger unknown slice than one that has settled and been confirmed many times.
    Under CRP (d=0) the driver is N: a still-discovering fact has accumulated
    fewer observations, so alpha/(N+alpha) stays larger."""
    churning = _categorical_store(tmp_path / "churn")
    _observe(churning, [f"novel-{i}" for i in range(6)])        # 6 singletons, N=6
    p_churn = churning.predict(CAT_STMT, budget=1000)

    settled = _categorical_store(tmp_path / "settled")
    _observe(settled, ["captcha"] * 40)                         # settled, N=40
    p_settled = settled.predict(CAT_STMT, budget=1000)

    assert p_churn.unknown.p > p_settled.unknown.p
    assert p_churn.unknown.p == pytest.approx(1 / 7)
    assert p_settled.unknown.p == pytest.approx(1 / 41)
    churning.close()
    settled.close()


def test_at_equal_N_settled_concentrates_mass_churning_does_not(tmp_path):
    """Honest CRP (d=0) characterization: at EQUAL N the unknown mass is identical
    (a function of N alone), but the settled fact concentrates predictive mass on
    its seen value while the churning fact spreads it thin — 'mass concentrates on
    the seen values'. Separating the unknown slice by distinct-value count is the
    deferred Pitman–Yor (d>0) upgrade."""
    settled = _categorical_store(tmp_path / "settled")
    _observe(settled, ["captcha"] * 20)
    p_settled = settled.predict(CAT_STMT, budget=1000)

    churning = _categorical_store(tmp_path / "churn")
    _observe(churning, [f"novel-{i}" for i in range(20)])
    p_churn = churning.predict(CAT_STMT, budget=1000)

    # Same N ⇒ same unknown mass under CRP.
    assert p_settled.unknown.p == pytest.approx(p_churn.unknown.p)
    # But the settled fact's top value dominates; the churning fact's does not.
    top_settled = max(s.p for s in p_settled.values.values())
    top_churn = max(s.p for s in p_churn.values.values())
    assert top_settled == pytest.approx(20 / 21)
    assert top_churn == pytest.approx(1 / 21)
    assert top_settled > top_churn
    settled.close()
    churning.close()


# ══ INTERVALS — Beta marginals via betamath ════════════════════════════════════

def test_intervals_are_the_exact_beta_quantile_pair(tmp_path):
    """Each per-value ci is Beta(n_v, (N+alpha)−n_v) at [0.05, 0.95] via betaincinv,
    and the unknown slice is Beta(alpha, N) — the same deterministic kernel the
    frequency path uses, no sampler."""
    m = _categorical_store(tmp_path / "s")
    _observe(m, ["captcha"] * 8 + ["block"] * 2)              # N=10, denom=11
    p = m.predict(CAT_STMT, budget=1000)

    assert p.values["captcha"].ci == (betaincinv(8.0, 3.0, CAT_CI_LO),
                                      betaincinv(8.0, 3.0, CAT_CI_HI))
    assert p.values["block"].ci == (betaincinv(2.0, 9.0, CAT_CI_LO),
                                    betaincinv(2.0, 9.0, CAT_CI_HI))
    assert p.unknown.ci == (betaincinv(1.0, 10.0, CAT_CI_LO),
                            betaincinv(1.0, 10.0, CAT_CI_HI))
    for slice_ in (*p.values.values(), p.unknown):
        assert slice_.ci[0] <= slice_.p <= slice_.ci[1]
        assert slice_.ci[0] < slice_.ci[1]
    m.close()


def test_intervals_narrow_as_the_value_count_grows(tmp_path):
    """A value seen 8/10 (Beta(8,3)) has a WIDER interval than the same value seen
    800/1000 (Beta(800,201)); the unknown slice also narrows as N grows."""
    small = _categorical_store(tmp_path / "small")
    _observe(small, ["captcha"] * 8 + ["block"] * 2)
    ps = small.predict(CAT_STMT, budget=1000)

    big = _categorical_store(tmp_path / "big")
    _observe(big, ["captcha"] * 800 + ["block"] * 200)
    pb = big.predict(CAT_STMT, budget=1000)

    def width(sl):
        return sl.ci[1] - sl.ci[0]

    assert width(pb.values["captcha"]) < width(ps.values["captcha"])
    assert width(pb.values["block"]) < width(ps.values["block"])
    assert width(pb.unknown) < width(ps.unknown)
    small.close()
    big.close()


# ══ DETERMINISM / REPLAY (I3 / I8) ═════════════════════════════════════════════

def test_replay_reproduces_the_categorical_prediction(tmp_path):
    m = _categorical_store(tmp_path / "s")
    rng = random.Random(3)
    vocab = ["captcha", "block", "allow", "tos"]
    for _ in range(30):
        m.observe(CAT_STMT, ctx={}, actor="tool:probe", value=rng.choice(vocab))
    m.run_gate()
    before = m.predict(CAT_STMT, budget=1000)
    assert m.closure_hash() == m.replay()
    after = m.predict(CAT_STMT, budget=1000)
    assert after.values == before.values          # bit-for-bit
    assert after.unknown == before.unknown
    assert after.snapshot_id == before.snapshot_id
    m.close()


def test_predict_at_reproduces_the_full_distribution_bit_for_bit(tmp_path):
    """I8: predict_at at a recorded snapshot (whose head is NOT the current head)
    reproduces the FULL distribution bit-for-bit, and the values iterate in the
    same canonical order."""
    m = _categorical_store(tmp_path / "s")
    rng = random.Random(9)
    vocab = ["captcha", "block", "allow", "tos"]
    for _ in range(15):
        m.observe(CAT_STMT, ctx={}, actor="tool:probe", value=rng.choice(vocab))
    m.run_gate()
    live = m.predict(CAT_STMT, budget=1000)
    snap = live.snapshot_id

    # Advance the ledger head with more observations, incl. a brand-new value.
    for _ in range(15):
        m.observe(CAT_STMT, ctx={}, actor="tool:probe", value="tail-" + str(rng.random()))
    m.run_gate()
    assert m.ledger.head() != snap                     # the tmp-index path

    at = m.predict_at(CAT_STMT, snap)
    assert list(at.values) == list(live.values)        # same canonical order
    assert at.values == live.values                    # bit-for-bit floats
    assert at.unknown == live.unknown
    assert at.total_observations == live.total_observations
    assert at.snapshot_id == live.snapshot_id
    m.close()


def test_alpha_rides_the_snapshot_id(tmp_path, monkeypatch):
    """The CRP concentration alpha is a versioned read-time constant: changing it
    changes the snapshot id (I8) AND the distribution."""
    m = _categorical_store(tmp_path / "s")
    _observe(m, ["captcha"] * 4 + ["block"] * 1)
    p1 = m.predict(CAT_STMT, budget=1000)
    assert "categorical/v1" in p1.snapshot_id
    assert "alpha=1.0" in p1.snapshot_id

    monkeypatch.setattr(counts_mod, "CATEGORICAL_ALPHA", 3.0)
    p3 = m.predict(CAT_STMT, budget=1000)
    assert p3.snapshot_id != p1.snapshot_id
    assert "alpha=3.0" in p3.snapshot_id
    # The distribution moved: a bigger alpha reserves more unknown mass.
    assert p3.unknown.p > p1.unknown.p
    assert p3.unknown.p == pytest.approx(3.0 / (5 + 3.0))
    m.close()


# ══ ADDITIVITY (§6) — the scalar path is untouched ═════════════════════════════

def _crisp_store(root) -> CandorSystem:
    m = CandorSystem(root)
    m.set_actor_quota("tool:probe", obs_per_epoch=100_000, cand_per_epoch=100_000)
    m.assert_({"pred": "reachable", "args": ["a", "b"], "stmt_type": "crisp"},
              source="seed", actor="human:me")
    m.run_gate()
    for _ in range(5):
        m.observe({"pred": "reachable", "args": ["a", "b"]}, True, {},
                  actor="tool:probe")
    m.run_gate()
    return m


def test_crisp_prediction_is_a_predictoutcome_unaffected_by_categorical_data(tmp_path):
    """A crisp fact still predicts to a scalar PredictOutcome, and it is
    byte-identical whether or not a categorical fact + its observations share the
    store — no categorical code runs for a non-categorical fact."""
    crisp_only = _crisp_store(tmp_path / "crisp_only")
    a = crisp_only.predict({"pred": "reachable", "args": ["a", "b"]}, budget=10_000)
    assert isinstance(a, PredictOutcome)

    mixed = _crisp_store(tmp_path / "mixed")
    mixed.assert_({"pred": "resolves", "args": ["login"], "stmt_type": "categorical"},
                  source="seed", actor="human:me")
    mixed.run_gate()
    for v in ("captcha", "captcha", "block", "novel"):
        mixed.observe(CAT_STMT, ctx={}, actor="tool:probe", value=v)
    mixed.run_gate()
    b = mixed.predict({"pred": "reachable", "args": ["a", "b"]}, budget=10_000)

    assert isinstance(b, PredictOutcome)
    assert (a.p, a.ci, a.channels, a.sensitivity, a.mpe, a.rejection_rate) == \
           (b.p, b.ci, b.channels, b.sensitivity, b.mpe, b.rejection_rate)
    crisp_only.close()
    mixed.close()
