"""v0.4 Δ6: graded observations (categorical Dawid–Skene at the API boundary).

`observe(..., confidence=c)` bins c into an integer grade at ingest (the raw
confidence stays in the event payload for audit; only the grade reaches the
derived index, I11). Grades feed a per-actor response ledger (actor_response:
vote, grade, n_true, n_false) that — like the two-coin confusion table — moves
ONLY when a claim settles through a deterministic_total oracle. These tests pin
the binning at its exact edges, the store/payload split, the settlement
movement by (vote, grade), the likelihood-ratio direction (a confident correct
vote is stronger evidence than a hesitant one), and replay determinism.

Spec: docs/spec-v0.4-delta.md (Δ6). House style follows tests/unit/test_two_coin.py.
"""

from __future__ import annotations

import math

import pytest

from candor.core.committed import reliability as R

BUDGET = 10_000
FACT_A = {"pred": "reachable", "args": ["a", "b"]}   # crisp seed fact, settle True
FACT_B = {"pred": "reachable", "args": ["b", "c"]}   # crisp seed fact, settle False


# ── helpers ──────────────────────────────────────────────────────────────────

def _response_cell(idx, actor, vote, grade, frame="external"):
    row = idx.one(
        "SELECT n_true, n_false FROM actor_response "
        "WHERE actor=? AND frame=? AND vote=? AND grade=?",
        (actor, frame, 1 if vote else 0, int(grade)))
    return (0, 0) if row is None else (int(row["n_true"]), int(row["n_false"]))


def _ledger_payload(sys_, seq):
    for ev in sys_.ledger.read_all():
        if ev.seq == seq:
            return sys_.ledger.payload(ev.payload_hash)
    raise AssertionError(f"no ledger event with seq={seq}")


def _settle(sys_, stmt, outcome, oracle="verifier:t"):
    """Register-if-needed a deterministic_total oracle, claim, resolve."""
    if sys_.index.one("SELECT id FROM oracles WHERE id=?", (oracle,)) is None:
        sys_.register_oracle(oracle, "deterministic_total", "t", "h", "e")
    cid = sys_.claim(stmt, "external", oracle, due=0)
    sys_.resolve(cid, outcome=outcome)


# ── the binning: grade_of, pinned at its exact edges ─────────────────────────

def test_grade_of_none_is_the_ungraded_legacy_bin():
    assert R.grade_of(None) == 0


def test_grade_of_bins_by_max_c_1minus_c_at_the_documented_cuts():
    # strength = max(c, 1-c) in [0.5, 1.0]; cuts at 0.75 and 0.9 (Δ6).
    assert R.grade_of(0.5) == 1          # strength 0.50  -> weak
    assert R.grade_of(0.6) == 1          # strength 0.60  -> weak
    assert R.grade_of(0.8) == 2          # strength 0.80  -> firm
    assert R.grade_of(0.95) == 3         # strength 0.95  -> strong


def test_grade_of_edges_are_half_open_lower_inclusive():
    # < 0.75 -> 1, [0.75, 0.9) -> 2, >= 0.9 -> 3. The cut value itself grades up.
    assert R.grade_of(0.74) == 1
    assert R.grade_of(0.75) == 2, "the 0.75 cut is inclusive into 'firm'"
    assert R.grade_of(0.8999) == 2
    assert R.grade_of(0.9) == 3, "the 0.9 cut is inclusive into 'strong'"


def test_grade_of_saturates_at_the_extremes():
    # c = 0 and c = 1 both have strength 1.0 -> the top grade; direction (the
    # vote) is carried separately, not by the grade.
    assert R.grade_of(0.0) == 3
    assert R.grade_of(1.0) == 3


def test_grade_ignores_vote_direction_only_the_strength_binned():
    # A confident NO (c near 0) and a confident YES (c near 1) share a grade.
    for c in (0.05, 0.2, 0.5, 0.7, 0.85, 0.97):
        assert R.grade_of(c) == R.grade_of(1.0 - c), f"asymmetry at c={c}"


# ── end to end: grade in the index, raw confidence in the ledger (I11) ───────

def test_observe_stores_the_grade_and_keeps_raw_confidence_in_the_payload(seeded):
    idx = seeded.index
    seq = seeded.observe(FACT_A, True, {}, actor="tool:probe", confidence=0.83)
    row = idx.one("SELECT grade, outcome FROM observations WHERE event_seq=?", (seq,))
    assert int(row["grade"]) == 2, "0.83 bins to firm; the index stores the grade"
    assert int(row["outcome"]) == 1, "the vote direction is untouched by the grade"
    payload = _ledger_payload(seeded, seq)
    assert payload["confidence"] == 0.83, "raw confidence survives in the event payload"
    assert payload["grade"] == 2, "the payload carries both raw c and its derived grade"


def test_ungraded_observe_defaults_to_grade_zero_and_null_confidence(seeded):
    idx = seeded.index
    seq = seeded.observe(FACT_A, True, {}, actor="tool:probe")   # legacy call, no c
    row = idx.one("SELECT grade FROM observations WHERE event_seq=?", (seq,))
    assert int(row["grade"]) == 0
    assert _ledger_payload(seeded, seq)["confidence"] is None


def test_full_grade_range_round_trips_through_observe(seeded):
    idx = seeded.index
    for confidence, expected in ((None, 0), (0.55, 1), (0.8, 2), (0.99, 3)):
        seq = seeded.observe(FACT_A, True, {}, actor="tool:probe", confidence=confidence)
        row = idx.one("SELECT grade FROM observations WHERE event_seq=?", (seq,))
        assert int(row["grade"]) == expected, f"c={confidence} should store grade {expected}"


# ── movement: the response ledger moves ONLY through settlement ──────────────

def test_observations_alone_do_not_move_the_response_ledger(seeded):
    idx = seeded.index
    for _ in range(5):
        seeded.observe(FACT_A, True, {}, actor="tool:probe", confidence=0.95)
    rows = idx.query("SELECT * FROM actor_response WHERE actor=?", ("tool:probe",))
    assert list(rows) == [], "the response ledger is a settled-only view (§3.12)"


def test_settlement_folds_each_observation_into_its_vote_grade_cell(seeded):
    idx = seeded.index
    # One actor, one crisp fact settled TRUE: a strong TRUE, a firm TRUE, and a
    # weak FALSE vote each land in their own (vote, grade) cell. n_true/n_false
    # index the SETTLED WORLD, not vote-agreement — so under a True settlement
    # every response, dissent included, folds into its cell's n_true column.
    seeded.observe(FACT_A, True, {}, actor="agent:g", confidence=0.95)   # (T, 3)
    seeded.observe(FACT_A, True, {}, actor="agent:g", confidence=0.80)   # (T, 2)
    seeded.observe(FACT_A, False, {}, actor="agent:g", confidence=0.60)  # (F, 1)
    _settle(seeded, FACT_A, outcome=True)

    assert _response_cell(idx, "agent:g", True, 3) == (1, 0)
    assert _response_cell(idx, "agent:g", True, 2) == (1, 0)
    # The dissenting FALSE vote still occurred under a True settlement.
    assert _response_cell(idx, "agent:g", False, 1) == (1, 0)
    # Cells nobody landed in stay empty; the counts are integers (I11).
    assert _response_cell(idx, "agent:g", True, 1) == (0, 0)
    assert _response_cell(idx, "agent:g", False, 3) == (0, 0)
    assert idx.nonintegral_counts() == []


def test_n_true_and_n_false_split_across_two_settlements(seeded):
    idx = seeded.index
    # Same (T, 1) response on two facts that settle oppositely: one right, one wrong.
    seeded.observe(FACT_A, True, {}, actor="agent:g", confidence=0.60)   # will be right
    seeded.observe(FACT_B, True, {}, actor="agent:g", confidence=0.60)   # will be wrong
    _settle(seeded, FACT_A, outcome=True)
    _settle(seeded, FACT_B, outcome=False)
    assert _response_cell(idx, "agent:g", True, 1) == (1, 1), \
        "the same vote/grade accrues n_true or n_false by the settled outcome"


# ── the likelihood ratio: confident correctness is stronger evidence ─────────

def _seed_graded_actor(seeded):
    """A learned actor: TRUE votes at grade 3 are always right; at grade 1 they
    are a coin. Needs >= RESPONSE_MIN_SCORED scored responses to leave the
    binary fallback and use the categorical response distribution."""
    idx = seeded.index
    for _ in range(5):
        seeded.observe(FACT_A, True, {}, actor="agent:g", confidence=0.95)   # (T,3) right
    for _ in range(3):
        seeded.observe(FACT_A, True, {}, actor="agent:g", confidence=0.60)   # (T,1) right
    _settle(seeded, FACT_A, outcome=True)
    for _ in range(4):
        seeded.observe(FACT_B, True, {}, actor="agent:g", confidence=0.60)   # (T,1) wrong
    _settle(seeded, FACT_B, outcome=False)
    return idx


def test_settled_response_cells_have_the_expected_counts(seeded):
    idx = _seed_graded_actor(seeded)
    assert _response_cell(idx, "agent:g", True, 3) == (5, 0)
    assert _response_cell(idx, "agent:g", True, 1) == (3, 4)


def test_response_lr_uses_the_graded_distribution_past_the_min_scored_floor(seeded):
    idx = _seed_graded_actor(seeded)
    # 8 true + 4 false = 12 scored responses > RESPONSE_MIN_SCORED (10): the
    # graded categorical path, not the binary Δ1 fallback.
    assert 12 >= R.RESPONSE_MIN_SCORED
    k = 2 * R.N_GRADES
    a = R.RESPONSE_ALPHA
    total_t, total_f = 8, 4
    # (T, 3): 5 true, 0 false — Dirichlet-smoothed over the ledger.
    lr3 = math.log(((5 + a) / (total_t + a * k)) / ((0 + a) / (total_f + a * k)))
    # (T, 1): 3 true, 4 false.
    lr1 = math.log(((3 + a) / (total_t + a * k)) / ((4 + a) / (total_f + a * k)))
    assert R.response_log_lr(idx, "agent:g", True, 3) == pytest.approx(lr3)
    assert R.response_log_lr(idx, "agent:g", True, 1) == pytest.approx(lr1)


def test_confident_correct_vote_outweighs_the_hesitant_one(seeded):
    idx = _seed_graded_actor(seeded)
    lr3 = R.response_log_lr(idx, "agent:g", True, 3)
    lr1 = R.response_log_lr(idx, "agent:g", True, 1)
    assert lr3 > 0.0, "a grade-3 TRUE that was always right is evidence FOR truth"
    assert lr1 < 0.0, "a grade-1 TRUE that was as often wrong points the other way"
    assert lr3 - lr1 > 2.0, "the confident correct vote is decisively the stronger"


# ── replay: the graded response state is ledger-derived ──────────────────────

def test_response_ledger_survives_replay_bit_for_bit(seeded):
    idx = _seed_graded_actor(seeded)
    before_cell = _response_cell(idx, "agent:g", True, 3)
    assert before_cell == (5, 0)
    before_hash = seeded.closure_hash()
    assert seeded.replay() == before_hash, \
        "actor_response is in the closure hash; it must survive a full refold"
    # And concretely: the cell is still there after the ledger-only rebuild.
    assert _response_cell(seeded.index, "agent:g", True, 3) == (5, 0)
