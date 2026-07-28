"""Counts, channels, reliability composition (spec §3.7, §3.12, §4.2, I5/I11)."""

from __future__ import annotations

import pytest

from candor.core.committed import counts, reliability


def test_channel_is_decided_by_statement_type():
    assert counts.channel_for("crisp") == "epi"
    assert counts.channel_for("frequency") == "alea"
    with pytest.raises(ValueError):
        counts.channel_for("vibes")


def test_crisp_observations_move_only_epi(seeded):
    fid = seeded.fact_id_for({"pred": "reachable", "args": ["a", "b"]})
    seeded.observe({"pred": "reachable", "args": ["a", "b"]}, True, {},
                   actor="tool:probe")
    raw = seeded.raw_counts(fid)
    assert raw[("tool:probe", "epi")] == (1, 1)
    assert not any(ch == "alea" and n for (_, ch), (n, _) in raw.items())


def test_frequency_observations_move_only_alea(seeded):
    fid = seeded.fact_id_for({"pred": "flaky_link", "args": ["c", "d"]})
    for outcome in (True, False, True):
        seeded.observe({"pred": "flaky_link", "args": ["c", "d"]}, outcome, {},
                       actor="tool:probe")
    raw = seeded.raw_counts(fid)
    assert raw[("tool:probe", "alea")] == (3, 2)
    assert not any(ch == "epi" and n for (_, ch), (n, _) in raw.items())


def test_admission_creates_an_addressable_zero_row(seeded):
    """Unobserved is ε, not a hard zero (I5): the audit trail exists from admission."""
    fid = seeded.fact_id_for({"pred": "flaky_link", "args": ["c", "d"]})
    assert seeded.raw_counts(fid)[("human:calvin", "alea")] == (0, 0)


def test_counts_are_integers_and_keyed_by_actor_and_channel(seeded):
    seeded.observe({"pred": "flaky_link", "args": ["c", "d"]}, True, {},
                   actor="tool:probe")
    assert seeded.index.nonintegral_counts() == []
    for key, (n, k) in seeded.raw_counts(
            seeded.fact_id_for({"pred": "flaky_link", "args": ["c", "d"]})).items():
        assert isinstance(key, tuple) and len(key) == 2
        assert isinstance(n, int) and isinstance(k, int)


def test_composition_is_a_read_time_view_not_storage(seeded):
    fid = seeded.fact_id_for({"pred": "flaky_link", "args": ["c", "d"]})
    for outcome in (True, True, False, True):
        seeded.observe({"pred": "flaky_link", "args": ["c", "d"]}, outcome, {},
                       actor="agent:x")
    seeded.set_reliability("agent:x", "external", 8.0, 2.0)
    composed = seeded.composed_counts(fid)
    assert composed.alea_n == pytest.approx(0.8 * 4)
    assert composed.alea_k == pytest.approx(0.8 * 3)
    assert seeded.raw_counts(fid)[("agent:x", "alea")] == (4, 3), "storage untouched"
    assert seeded.index.nonintegral_counts() == []


def test_retroactive_exclusion_is_a_pure_recompute(seeded):
    fid = seeded.fact_id_for({"pred": "flaky_link", "args": ["c", "d"]})
    for _ in range(10):
        seeded.observe({"pred": "flaky_link", "args": ["c", "d"]}, True, {},
                       actor="tool:honest")
        seeded.observe({"pred": "flaky_link", "args": ["c", "d"]}, False, {},
                       actor="agent:compromised")
    seeded.set_reliability("agent:compromised", "external", 0.0001, 100.0)
    composed = seeded.composed_counts(fid)
    assert composed.alea_k / composed.alea_n > 0.9
    assert seeded.raw_counts(fid)[("agent:compromised", "alea")] == (10, 0)


def test_frozen_targets_are_a_no_op_for_the_updater(seeded):
    fid = seeded.fact_id_for({"pred": "flaky_link", "args": ["c", "d"]})
    seeded.index.execute("UPDATE facts SET numeric='frozen' WHERE id=?", (fid,))
    seeded.index.commit()
    seeded.observe({"pred": "flaky_link", "args": ["c", "d"]}, True, {},
                   actor="tool:probe")
    assert seeded.raw_counts(fid).get(("tool:probe", "alea")) is None


def test_reliability_moves_only_through_a_trusted_settlement(seeded):
    idx = seeded.index
    before = reliability.expected(idx, "tool:probe")
    for _ in range(5):
        seeded.observe({"pred": "flaky_link", "args": ["c", "d"]}, True, {},
                       actor="tool:probe")
    assert reliability.expected(idx, "tool:probe") == before, \
        "observations alone must not move reliability (§3.12)"

    fid = seeded.fact_id_for({"pred": "flaky_link", "args": ["c", "d"]})
    reliability.score_against_settlement(idx, fid, outcome=False)
    assert reliability.expected(idx, "tool:probe") < before


def test_unscored_actors_start_from_the_prior():
    a, b = reliability.REL_PRIOR_A, reliability.REL_PRIOR_B
    assert 0.9 < a / (a + b) < 1.0, "newcomers are not taxed, only the demonstrated"


def test_priors_are_applied_at_read_time_only():
    composed = counts.Composed(epi_a=0.0, epi_b=0.0, alea_n=0.0, alea_k=0.0)
    epi, alea = counts.posterior_params(composed, "frequency")
    assert epi == counts.EPI_PRIOR_FREQUENCY
    assert alea == (counts.ALEA_PRIOR_A, counts.ALEA_PRIOR_B)


def test_the_epistemic_prior_is_not_shared_between_statement_types():
    """§4.2: epi means p(true) for crisp and p(class valid) for frequency.

    Admission is a structural check; it observes nothing about the world, so it
    must not hand a crisp fact a near-certain truth prior.
    """
    assert counts.epi_prior("crisp") == (1.0, 1.0)
    assert counts.epi_prior("frequency")[0] > 1.0

    # Twenty contradicting observations must actually move a crisp fact.
    contradicted = counts.Composed(epi_a=0.0, epi_b=20.0, alea_n=0.0, alea_k=0.0)
    (a, b), _ = counts.posterior_params(contradicted, "crisp")
    assert a / (a + b) < 0.10
