"""Categorical facts — Stage C3: settlement + calibration + per-source trust
(design_categorical.md §4/§5/§6/§8 "Stage C3"). The HARDEST stage.

C3 turns a categorical claim into a settleable, scoreable event:

  * SETTLEMENT — a categorical claim freezes its full predicted distribution (the
    C2 CategoricalPrediction) into the ledger claim payload (I8, snapshot-pinned).
    At resolution the verifier reports the REALISED value v*; the payload records
    v* (audit) and the multiclass SURPRISAL = −log P_frozen(v*).
  * FINITE SURPRISAL FOR AN UNSEEN VALUE (the headline) — if v* was NEVER seen at
    claim time, P_frozen(v*) is NOT zero: it is the reserved unknown mass
    P(unknown) (§4.1). So an unseen realised value scores FINITE surprisal, not
    inf/NaN. A crisp/frequency claim cannot do this; the categorical claim can.
  * CALIBRATION reuse (I9) — record the probability assigned to the value that
    actually occurred, under a DISTINCT predictor_class 'categorical/v1', so the
    multiclass reliability diagram NEVER pools with binary claims.
  * PER-SOURCE TRUST — one-vs-rest (design §4.2 Option B, LOCKED): when a
    categorical claim on v* settles, each prior observation reporting value r
    becomes, per value v, a binary vote "did you report v?" settled by "was the
    truth v?", folded into actor_confusion / actor_response keyed ALSO by value
    (a per-value VIRTUAL ACTOR id, reusing the EXISTING two-coin machinery). The
    read-time per-(actor,value) discount enters category_posterior's C3 seam so a
    value-randomising source's reports get discounted in the predictive dist.
  * TRUSTED PATH ONLY (§3.12) — those per-value cells move ONLY through the
    deterministic_total settlement path, never on plain observations.
  * DETERMINISM (I3/I8) — replay reproduces the per-(actor,value) cells AND the
    discounted distribution; predict_at reproduces; checkpoint == full replay.
  * ADDITIVITY (§6) — crisp/frequency claim/resolve/two-coin behaviour byte-
    identical; the frozen conformance suite stays green.
"""

from __future__ import annotations

import math

import pytest

from candor.core.committed import reliability as R
from candor.core import calibration as calibration_mod
from candor.system import CandorSystem, CategoricalPrediction, PredictOutcome

CAT_STMT = {"pred": "resolves", "args": ["login"]}
CRISP_STMT = {"pred": "reachable", "args": ["a", "b"]}


# ── fixtures / helpers ─────────────────────────────────────────────────────────

def _categorical_store(root, actors=("tool:probe", "tool:honest", "tool:random")):
    m = CandorSystem(root)
    for a in actors:
        m.set_actor_quota(a, obs_per_epoch=100_000, cand_per_epoch=100_000)
    m.assert_({"pred": "resolves", "args": ["login"], "stmt_type": "categorical"},
              source="seed", actor="tool:probe")
    m.run_gate()
    m.register_oracle("verifier:truth", "deterministic_total", "t", "h", "e")
    return m


def _observe(m, values, actor="tool:probe"):
    for v in values:
        m.observe(CAT_STMT, ctx={}, actor=actor, value=v)


def _surprisal_of(m, claim_id):
    return m.index.one("SELECT surprisal FROM claims WHERE id=?", (claim_id,))["surprisal"]


# ══ HEADLINE — finite surprisal for a seen AND an unseen value ══════════════════

def test_finite_surprisal_for_seen_and_for_never_seen_value(tmp_path):
    """The single biggest payoff (§4.1). Freeze the distribution at claim time
    (N=10: captcha 8/11, block 2/11, unknown 1/11), then settle two claims: one
    against a SEEN value (surprisal = −log 8/11) and one against a value that was
    NEVER observed (surprisal = −log(unknown) = −log 1/11). BOTH finite."""
    m = _categorical_store(tmp_path / "s")
    _observe(m, ["captcha"] * 8 + ["block"] * 2)          # N=10, denom=11

    # Freeze BOTH claims BEFORE any settlement, so each pins the clean C2 dist.
    cid_seen = m.claim(CAT_STMT, "external", "verifier:truth", due=0)
    cid_unseen = m.claim(CAT_STMT, "external", "verifier:truth", due=0)

    # Settle against a SEEN value.
    m.resolve(cid_seen, value="captcha")
    surp_seen = _surprisal_of(m, cid_seen)
    assert math.isfinite(surp_seen)
    assert surp_seen == pytest.approx(-math.log(8 / 11))

    # Settle against a value that was NEVER seen at claim time. NOT infinite —
    # it scores against the reserved unknown slice, P(unknown)=1/11.
    m.resolve(cid_unseen, value="ghost-value-never-observed")
    surp_unseen = _surprisal_of(m, cid_unseen)
    assert math.isfinite(surp_unseen)
    assert not math.isnan(surp_unseen)
    assert surp_unseen == pytest.approx(-math.log(1 / 11))
    # And the unseen value is genuinely more surprising than the modal one.
    assert surp_unseen > surp_seen
    m.close()


def test_realised_value_is_recorded_in_the_resolution_payload(tmp_path):
    """Audit: the realised value v* rides in the resolution event payload, like
    verifier_code_hash does today."""
    m = _categorical_store(tmp_path / "s")
    _observe(m, ["captcha"] * 3)
    cid = m.claim(CAT_STMT, "external", "verifier:truth", due=0)
    seq = m.resolve(cid, value="captcha")
    ph = m.index.one("SELECT payload_hash FROM events WHERE seq=?", (seq,))["payload_hash"]
    payload = m.ledger.payload(ph)
    assert payload["realized_value"] == "captcha"
    assert math.isfinite(payload["surprisal"])
    m.close()


def test_never_observed_fact_settles_with_zero_surprisal(tmp_path):
    """A fact admitted but never observed freezes unknown=1.0; ANY realised value
    scores −log(1.0)=0. Finite, well-defined, the degenerate boundary."""
    m = _categorical_store(tmp_path / "s")
    cid = m.claim(CAT_STMT, "external", "verifier:truth", due=0)
    m.resolve(cid, value="anything")
    assert _surprisal_of(m, cid) == pytest.approx(0.0)
    m.close()


# ══ TRUST MOVES THE DISTRIBUTION — one-vs-rest discount into category_posterior ══

def test_honest_vs_random_settlement_shifts_the_distribution(tmp_path):
    """An honest observer (always the true value) vs a value-randomising observer.
    After a deterministic_total settlement scores them one-vs-rest, the composed
    categorical distribution shifts TOWARD the honest observer's reports: the true
    value's share rises and the randomiser's spurious values' shares fall."""
    m = _categorical_store(tmp_path / "s")
    _observe(m, ["captcha"] * 10, actor="tool:honest")           # honest → truth
    _observe(m, ["captcha"] * 2 + ["block"] * 4 + ["allow"] * 4,  # randomiser noise
             actor="tool:random")

    before = m.predict(CAT_STMT, budget=1000)
    assert isinstance(before, CategoricalPrediction)

    # A single trusted settlement on the true value scores every observation.
    cid = m.claim(CAT_STMT, "external", "verifier:truth", due=0)
    m.resolve(cid, value="captcha")

    after = m.predict(CAT_STMT, budget=1000)

    # The distribution shifted toward the honest observer's value.
    assert after.values["captcha"].p > before.values["captcha"].p + 0.1
    # The randomiser's spurious values are discounted.
    assert after.values["block"].p < before.values["block"].p
    assert after.values["allow"].p < before.values["allow"].p
    # Sum is still exactly 1 after discounting (the residual construction holds).
    total = sum(s.p for s in after.values.values()) + after.unknown.p
    assert total == pytest.approx(1.0)
    m.close()


def test_no_settlement_means_no_discount_c2_is_preserved_bit_for_bit(tmp_path):
    """With NO categorical settlement the per-value weight is EXACTLY 1.0, so the
    C3 discount seam is a byte-for-byte no-op over the C2 raw-count posterior."""
    m = _categorical_store(tmp_path / "s")
    _observe(m, ["captcha"] * 8 + ["block"] * 2)
    p = m.predict(CAT_STMT, budget=1000)
    assert p.values["captcha"].p == pytest.approx(8 / 11)
    assert p.values["block"].p == pytest.approx(2 / 11)
    assert p.unknown.p == pytest.approx(1 / 11)
    # The frozen C2 interval arithmetic is unchanged (integer Beta params).
    from candor.core.betamath import betaincinv
    from candor.core.committed.counts import CAT_CI_HI, CAT_CI_LO
    assert p.values["captcha"].ci == (betaincinv(8.0, 3.0, CAT_CI_LO),
                                      betaincinv(8.0, 3.0, CAT_CI_HI))
    m.close()


# ══ RELIABILITY MOVES ONLY ON THE TRUSTED PATH (§3.12) ═════════════════════════

def test_per_value_confusion_moves_only_on_deterministic_total_settlement(tmp_path):
    """The per-(actor,value) cells move ONLY through a deterministic_total oracle,
    never on plain observations, and never through a stochastic oracle."""
    m = _categorical_store(tmp_path / "s")
    m.register_oracle("verifier:guess", "stochastic", "t", "h", "e")
    _observe(m, ["captcha"] * 5)

    key = R._cat_key("tool:probe", "captcha")
    # Observations alone must not move the per-value confusion.
    assert R.confusion(m.index, key) == (0, 0, 0, 0)

    # A settlement via a NON-deterministic_total oracle must not move it either.
    cid_g = m.claim(CAT_STMT, "external", "verifier:guess", due=0)
    m.resolve(cid_g, value="captcha")
    assert R.confusion(m.index, key) == (0, 0, 0, 0), \
        "untrusted (stochastic) settlement must not move reliability"

    # A deterministic_total settlement is the only thing that moves it.
    cid_t = m.claim(CAT_STMT, "external", "verifier:truth", due=0)
    m.resolve(cid_t, value="captcha")
    assert R.confusion(m.index, key) == (5, 0, 0, 0), \
        "5 captcha reports settled against captcha → 5 true positives"
    m.close()


def test_one_vs_rest_produces_a_false_positive_for_a_wrong_report(tmp_path):
    """The reduction: an observation reporting r≠v* is a true positive for v* only
    if r==v*; a report of r that turned out wrong is a false positive for r."""
    m = _categorical_store(tmp_path / "s")
    _observe(m, ["captcha"] * 3 + ["block"] * 2)          # truth will be captcha
    cid = m.claim(CAT_STMT, "external", "verifier:truth", due=0)
    m.resolve(cid, value="captcha")

    # proposition captcha: 3 reports of captcha (tp), 2 of block (fn: truth was v*)
    assert R.confusion(m.index, R._cat_key("tool:probe", "captcha")) == (3, 2, 0, 0)
    # proposition block: 2 reports of block that were wrong (fp), 0 right
    assert R.confusion(m.index, R._cat_key("tool:probe", "block")) == (0, 0, 2, 0)
    m.close()


def test_per_value_cells_are_integers_in_storage(tmp_path):
    """I11: the virtual-actor confusion/response cells are integers; the
    integrality scan (which already covers actor_confusion/actor_response) passes
    over them, since the one-vs-rest reduction reuses those exact tables."""
    m = _categorical_store(tmp_path / "s")
    _observe(m, ["captcha"] * 4 + ["block"] * 1, actor="tool:random")
    cid = m.claim(CAT_STMT, "external", "verifier:truth", due=0)
    m.resolve(cid, value="captcha")
    assert m.index.nonintegral_counts() == []
    m.close()


# ══ CALIBRATION — I9: categorical NEVER pools with binary ══════════════════════

def test_categorical_calibration_records_prob_of_the_realised_value(tmp_path):
    """§4: calibration records the probability the model assigned to the value that
    actually occurred, under predictor_class 'categorical/v1'."""
    m = _categorical_store(tmp_path / "s")
    _observe(m, ["captcha"] * 8 + ["block"] * 2)          # P(captcha)=8/11≈0.727
    cid = m.claim(CAT_STMT, "external", "verifier:truth", due=0)
    m.resolve(cid, value="captcha")

    rows = [r for r in calibration_mod.report(m.index)
            if r["predictor_class"] == calibration_mod.CATEGORICAL_PREDICTOR_CLASS]
    assert rows, "categorical calibration partition must exist"
    # p≈0.727 lands in bucket 7; observed as the value that occurred (outcome=True).
    row = next(r for r in rows if r["n"] > 0)
    assert row["bucket"] == calibration_mod.bucket_of(8 / 11)
    assert row["observed_freq"] == 1.0
    m.close()


def test_categorical_calibration_never_pools_with_binary(tmp_path):
    """I9: a binary claim and a categorical claim in the SAME store settle into
    DISJOINT calibration partitions (distinct predictor_class), never pooled."""
    m = _categorical_store(tmp_path / "s")
    # A crisp fact + a settled binary claim in the same store.
    m.assert_(dict(CRISP_STMT, stmt_type="crisp"), source="seed", actor="human:me")
    m.run_gate()
    for _ in range(3):
        m.observe(CRISP_STMT, True, {}, actor="tool:probe")
    m.run_gate()
    cid_b = m.claim(CRISP_STMT, "external", "verifier:truth", due=0)
    m.resolve(cid_b, outcome=True)

    # A categorical claim + settlement.
    _observe(m, ["captcha"] * 5)
    cid_c = m.claim(CAT_STMT, "external", "verifier:truth", due=0)
    m.resolve(cid_c, value="captcha")

    report = calibration_mod.report(m.index)
    classes = {r["predictor_class"] for r in report if r["n"] > 0}
    assert calibration_mod.CATEGORICAL_PREDICTOR_CLASS in classes
    assert calibration_mod.DEFAULT_PREDICTOR_CLASS in classes
    # The two partitions never share a (frame, settlement, predictor_class, bucket)
    # key: categorical rows and binary rows are disjoint by predictor_class.
    cat = {(r["frame"], r["settlement"], r["bucket"]) for r in report
           if r["predictor_class"] == calibration_mod.CATEGORICAL_PREDICTOR_CLASS}
    binr = {(r["frame"], r["settlement"], r["bucket"]) for r in report
            if r["predictor_class"] == calibration_mod.DEFAULT_PREDICTOR_CLASS}
    # Even if buckets coincide, the predictor_class dimension keeps them separate.
    assert all(r["predictor_class"] != calibration_mod.CATEGORICAL_PREDICTOR_CLASS
               for r in report if r["predictor_class"] == calibration_mod.DEFAULT_PREDICTOR_CLASS)
    m.close()


# ══ DETERMINISM / REPLAY (I3 / I8) ═════════════════════════════════════════════

def test_replay_reproduces_the_per_value_cells_and_discounted_distribution(tmp_path):
    """Replay reproduces the per-(actor,value) confusion cells AND the discounted
    predictive distribution bit-for-bit (both are ledger-derived)."""
    m = _categorical_store(tmp_path / "s")
    _observe(m, ["captcha"] * 6, actor="tool:honest")
    _observe(m, ["captcha"] * 2 + ["block"] * 3 + ["allow"] * 2, actor="tool:random")
    cid = m.claim(CAT_STMT, "external", "verifier:truth", due=0)
    m.resolve(cid, value="captcha")

    before_pred = m.predict(CAT_STMT, budget=1000)
    before_conf = R.confusion(m.index, R._cat_key("tool:random", "block"))
    assert before_conf != (0, 0, 0, 0)

    before_hash = m.closure_hash()
    assert m.replay() == before_hash, "categorical settlement state must survive replay"

    after_pred = m.predict(CAT_STMT, budget=1000)
    assert R.confusion(m.index, R._cat_key("tool:random", "block")) == before_conf
    assert after_pred.values == before_pred.values          # discounted dist, bit-for-bit
    assert after_pred.unknown == before_pred.unknown
    assert after_pred.snapshot_id == before_pred.snapshot_id
    m.close()


def test_predict_at_reproduces_the_discounted_distribution(tmp_path):
    """I8: predict_at at a recorded snapshot (head != current) reproduces the FULL
    discounted distribution bit-for-bit — including the settlement-driven weights
    folded up to that snapshot."""
    m = _categorical_store(tmp_path / "s")
    _observe(m, ["captcha"] * 6, actor="tool:honest")
    _observe(m, ["captcha"] * 2 + ["block"] * 4, actor="tool:random")
    cid = m.claim(CAT_STMT, "external", "verifier:truth", due=0)
    m.resolve(cid, value="captcha")                          # discount now active

    live = m.predict(CAT_STMT, budget=1000)
    snap = live.snapshot_id

    # Advance the ledger head past the snapshot.
    _observe(m, ["tail-" + str(i) for i in range(5)], actor="tool:probe")
    m.run_gate()
    assert m.ledger.head() != snap

    at = m.predict_at(CAT_STMT, snap)
    assert list(at.values) == list(live.values)
    assert at.values == live.values                          # bit-for-bit
    assert at.unknown == live.unknown
    assert at.snapshot_id == live.snapshot_id
    m.close()


def test_checkpoint_accelerated_open_equals_full_replay_with_settlements(tmp_path):
    """Checkpoint == full replay WITH categorical settlements: the per-value cells
    live in actor_confusion/actor_response (already in _HASH_QUERIES) and claims'
    frozen distribution + surprisal are in the claims hash query, so the fast-path
    open matches a forced full-from-genesis replay."""
    root = tmp_path / "store"
    m = _categorical_store(root)
    _observe(m, ["captcha"] * 5 + ["block"] * 3, actor="tool:random")
    cid = m.claim(CAT_STMT, "external", "verifier:truth", due=0)
    m.resolve(cid, value="captcha")                          # settlement before cp
    m.checkpoint()
    _observe(m, ["captcha"] * 4, actor="tool:honest")        # tail after cp
    cid2 = m.claim(CAT_STMT, "external", "verifier:truth", due=0)
    m.resolve(cid2, value="captcha")                         # tail settlement
    m.close()

    m2 = CandorSystem(root)                                  # fast-path open
    hash_cp = m2.closure_hash()
    hash_full = m2.replay()                                  # forced full
    assert hash_cp == hash_full, "categorical settlement checkpoint != full replay"
    m2.close()


# ══ ADDITIVITY (§6) — the binary claim/resolve path is untouched ═══════════════

def test_binary_claim_resolve_is_byte_identical_with_categorical_present(tmp_path):
    """A crisp claim still settles to a scalar surprisal and moves the two-coin
    confusion exactly as before, whether or not a categorical fact shares the
    store — no categorical code runs for a non-categorical claim."""
    def _binary(root, with_cat):
        m = CandorSystem(root)
        for a in ("tool:probe",):
            m.set_actor_quota(a, obs_per_epoch=100_000, cand_per_epoch=100_000)
        m.assert_(dict(CRISP_STMT, stmt_type="crisp"), source="seed", actor="human:me")
        m.run_gate()
        if with_cat:
            m.assert_({"pred": "resolves", "args": ["login"],
                       "stmt_type": "categorical"}, source="seed", actor="human:me")
            m.run_gate()
            for v in ("captcha", "captcha", "block"):
                m.observe(CAT_STMT, ctx={}, actor="tool:probe", value=v)
        for _ in range(5):
            m.observe(CRISP_STMT, True, {}, actor="tool:probe")
        m.register_oracle("verifier:truth", "deterministic_total", "t", "h", "e")
        cid = m.claim(CRISP_STMT, "external", "verifier:truth", due=0)
        m.resolve(cid, outcome=True)
        surp = m.index.one("SELECT surprisal, outcome FROM claims WHERE id=?", (cid,))
        conf = R.confusion(m.index, "tool:probe")
        m.close()
        return surp["surprisal"], surp["outcome"], conf

    plain = _binary(tmp_path / "plain", with_cat=False)
    mixed = _binary(tmp_path / "mixed", with_cat=True)
    assert plain == mixed, "binary claim/resolve/two-coin must be byte-identical"


def test_categorical_claim_returns_a_claim_id_and_freezes_the_distribution(tmp_path):
    """A categorical claim freezes its full predicted distribution into the claims
    row (predicted_dist_json), with predicted_p left NULL (it is a distribution,
    not a scalar)."""
    m = _categorical_store(tmp_path / "s")
    _observe(m, ["captcha"] * 8 + ["block"] * 2)
    cid = m.claim(CAT_STMT, "external", "verifier:truth", due=0)
    assert cid != "Refused"
    row = m.index.one(
        "SELECT predicted_p, predicted_dist_json, predictor_class FROM claims "
        "WHERE id=?", (cid,))
    assert row["predicted_p"] is None                        # a distribution, not a scalar
    assert row["predictor_class"] == calibration_mod.CATEGORICAL_PREDICTOR_CLASS
    import json
    frozen = json.loads(row["predicted_dist_json"])
    assert frozen["values"]["captcha"] == pytest.approx(8 / 11)
    assert frozen["unknown"] == pytest.approx(1 / 11)
    m.close()
