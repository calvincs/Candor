"""The §5 API contracts: pins, aliases, claims, quotas, provenance, replay."""

from __future__ import annotations

import pytest

from candor.system import QuotaExceeded, REFUSED

BUDGET = 10_000


# ── pins: the only hard zero (I5) ───────────────────────────────────────────

def test_negative_pin_removes_the_fact_from_the_closure(seeded):
    fid = seeded.fact_id_for({"pred": "reachable", "args": ["a", "b"]})
    assert seeded.derive({"pred": "reachable", "args": ["a", "b"]},
                         BUDGET).status == "proof"
    seeded.pin(fid, "-", "known-bad", "human:calvin")
    assert seeded.derive({"pred": "reachable", "args": ["a", "b"]},
                         BUDGET).status == "not_entailed"


def test_contradicting_observations_are_absorbed_counted_and_paged(seeded):
    fid = seeded.fact_id_for({"pred": "reachable", "args": ["a", "b"]})
    seeded.pin(fid, "-", "known-bad", "human:calvin")
    assert seeded.questions() == []
    for _ in range(20):
        seeded.observe({"pred": "reachable", "args": ["a", "b"]}, True, {},
                       actor="tool:probe")
    assert seeded.raw_counts(fid)[("tool:probe", "epi")] == (20, 20), "counted"
    assert seeded.derive({"pred": "reachable", "args": ["a", "b"]},
                         BUDGET).status != "proof", "the pin still wins"
    q = [x for x in seeded.questions() if x["kind"] == "pin_tension"]
    assert q and q[0]["target_id"] == fid
    assert q[0]["suggested_measurement"]


def test_pin_tension_does_not_open_on_a_single_blip(seeded):
    fid = seeded.fact_id_for({"pred": "reachable", "args": ["a", "b"]})
    seeded.pin(fid, "-", "known-bad", "human:calvin")
    seeded.observe({"pred": "reachable", "args": ["a", "b"]}, True, {},
                   actor="tool:probe")
    assert seeded.questions() == []


# ── aliases: union at read, reversible by supersede (I11) ───────────────────

def test_alias_unions_counts_at_read_without_merging_storage(seeded):
    seeded.assert_({"pred": "lossy_link", "args": ["c", "d"],
                    "stmt_type": "frequency"}, source="seed", actor="human:calvin")
    seeded.run_gate()
    for _ in range(4):
        seeded.observe({"pred": "flaky_link", "args": ["c", "d"]}, True, {},
                       actor="tool:probe")
    for _ in range(6):
        seeded.observe({"pred": "lossy_link", "args": ["c", "d"]}, False, {},
                       actor="tool:probe")
    flaky = seeded.fact_id_for({"pred": "flaky_link", "args": ["c", "d"]})
    lossy = seeded.fact_id_for({"pred": "lossy_link", "args": ["c", "d"]})
    assert seeded.raw_counts(flaky)[("tool:probe", "alea")] == (4, 4)

    seeded.assert_({"kind": "alias", "canonical": "flaky_link",
                    "alias": "lossy_link", "basis": "pinned"},
                   source="test", actor="human:calvin")
    seeded.run_gate()

    unioned = seeded.raw_counts(flaky)[("tool:probe", "alea")]
    assert unioned == (10, 4), "union happens at read"
    stored_flaky = seeded.index.query(
        "SELECT n, k FROM fact_counts WHERE fact_id=? AND actor='tool:probe'", (flaky,))
    assert [tuple(r) for r in stored_flaky] == [(4, 4)], "storage was never merged"
    stored_lossy = seeded.index.query(
        "SELECT n, k FROM fact_counts WHERE fact_id=? AND actor='tool:probe'", (lossy,))
    assert [tuple(r) for r in stored_lossy] == [(6, 0)]


def test_alias_is_reversible_by_superseding_its_event(seeded):
    flaky = seeded.fact_id_for({"pred": "flaky_link", "args": ["c", "d"]})
    before = seeded.raw_counts(flaky)
    seeded.assert_({"kind": "alias", "canonical": "flaky_link",
                    "alias": "lossy_link", "basis": "pinned"},
                   source="test", actor="human:calvin")
    seeded.run_gate()
    alias_events = seeded.events_since(0, kinds={"alias"})
    assert alias_events, "alias admission must be a ledger event"
    seeded.supersede(str(alias_events[-1]["id"]), "test unwind")
    assert seeded.raw_counts(flaky) == before


# ── claims & settlement triage (§3.8) ───────────────────────────────────────

def test_unsettleable_claims_are_refused_entry(seeded):
    assert seeded.claim({"pred": "flaky_link", "args": ["c", "d"]}, "external",
                        "", due=0) == REFUSED


def test_entailed_claims_settle_from_the_closure(seeded):
    cid = seeded.claim({"pred": "reachable", "args": ["a", "b"]}, "internal",
                       "tool:closure", due=0)
    row = seeded.index.one("SELECT settlement, certainty_class FROM claims WHERE id=?",
                           (cid,))
    assert row["settlement"] == "entailed"
    assert row["certainty_class"] == "certain"


def test_a_claim_records_its_snapshot_and_resolution_is_auditable(seeded):
    cid = seeded.claim({"pred": "flaky_link", "args": ["c", "d"]}, "external",
                       "tool:probe", due=99)
    row = seeded.index.one("SELECT model_snapshot, predicted_p FROM claims WHERE id=?",
                           (cid,))
    assert "ledger_head" in row["model_snapshot"]
    ev = seeded.resolve(cid, outcome=True, verifier_code_hash="abc", env_hash="def")
    payload = seeded.ledger.payload(
        seeded.events_since(ev - 1)[0]["payload_hash"])
    assert payload["verifier_code_hash"] == "abc" and payload["env_hash"] == "def"
    assert "sensitivity" in payload, "full sensitivity vector is always logged (§4.4)"
    assert seeded.index.one("SELECT outcome FROM claims WHERE id=?", (cid,))["outcome"] == 1


def test_resolution_feeds_the_calibration_bucketer(seeded):
    cid = seeded.claim({"pred": "flaky_link", "args": ["c", "d"]}, "external",
                       "tool:probe", due=99)
    seeded.resolve(cid, outcome=True)
    report = seeded.health()["calibration"]
    assert sum(r["n"] for r in report) == 1


# ── quotas (§3.12) ──────────────────────────────────────────────────────────

def test_observation_quota_bounds_flooding(seeded, monkeypatch):
    monkeypatch.setattr("candor.core.apply.DEFAULT_OBS_QUOTA", 5)
    seeded.index.execute("UPDATE actors SET obs_quota_per_epoch=5 "
                         "WHERE name='agent:spammer'")
    stmt = {"pred": "flaky_link", "args": ["c", "d"]}
    for _ in range(5):
        seeded.observe(stmt, True, {}, actor="agent:spammer")
    with pytest.raises(QuotaExceeded):
        seeded.observe(stmt, True, {}, actor="agent:spammer")
    assert any(e.get("kind") == "quota_exhausted" for e in seeded.health()["events"])


def test_candidate_quota_bounds_gate_flooding(seeded):
    seeded.index.execute(
        "INSERT OR IGNORE INTO actors(name, class, obs_quota_per_epoch, "
        "cand_quota_per_epoch) VALUES('agent:chatty','agent',10,6)")
    with pytest.raises(QuotaExceeded):
        for i in range(50):
            seeded.assert_({"pred": "reachable", "args": [f"x{i}", "y"],
                            "stmt_type": "crisp"}, source="spam", actor="agent:chatty")
    assert seeded.health()["queue_depth"] <= 6


# ── provenance & introspection ──────────────────────────────────────────────

def test_why_reports_raw_and_composed_counts_and_provenance(seeded):
    fid = seeded.fact_id_for({"pred": "flaky_link", "args": ["c", "d"]})
    seeded.observe({"pred": "flaky_link", "args": ["c", "d"]}, True,
                   {"site": "lab"}, actor="tool:probe")
    out = seeded.why(fid)
    assert out["found"] and out["pred"] == "flaky_link"
    assert out["raw_counts"]["tool:probe|alea"] == [1, 1]
    assert out["composed_counts"]["alea_n"] > 0
    assert out["gate_run"] and out["gate_run"].startswith("gate:")
    assert out["source_diversity"] == {"site": 1}
    assert out["derivation"]["status"] == "proof"


def test_events_since_is_the_outbox(seeded):
    head = len(seeded.events_since(0))
    seeded.observe({"pred": "flaky_link", "args": ["c", "d"]}, True, {},
                   actor="tool:probe")
    new = seeded.events_since(head)
    assert [e["kind"] for e in new] == ["observation"]
    assert seeded.events_since(0, kinds={"admission"})


def test_retrieval_is_logged_to_the_side_stream_and_moves_nothing(seeded):
    fid = seeded.fact_id_for({"pred": "flaky_link", "args": ["c", "d"]})
    before_counts = seeded.raw_counts(fid)
    before_head = seeded.ledger_head()
    for _ in range(25):
        seeded.recall("flaky link c d", budget=512)
    assert seeded.raw_counts(fid) == before_counts
    assert seeded.ledger_head() == before_head, "retrieval is outside the chain (I2)"
    assert seeded.health()["retrieval_log_size"] == 25


# ── replay (I1, I3) ─────────────────────────────────────────────────────────

def test_replay_reproduces_the_closure_after_every_kind_of_change(seeded):
    fid = seeded.fact_id_for({"pred": "flaky_link", "args": ["c", "d"]})
    seeded.observe({"pred": "flaky_link", "args": ["c", "d"]}, True,
                   {"site": "lab"}, actor="tool:probe")
    seeded.pin(fid, "+", "trusted", "human:calvin")
    seeded.assert_({"kind": "alias", "canonical": "flaky_link", "alias": "lossy_link",
                    "basis": "pinned"}, source="t", actor="human:calvin")
    seeded.run_gate()
    cid = seeded.claim({"pred": "flaky_link", "args": ["c", "d"]}, "external",
                       "tool:probe", due=1)
    seeded.resolve(cid, outcome=True)
    before = seeded.closure_hash()
    assert seeded.replay() == before


def test_dropping_the_index_loses_nothing(seeded):
    seeded.observe({"pred": "flaky_link", "args": ["c", "d"]}, True, {},
                   actor="tool:probe")
    before = seeded.closure_hash()
    seeded.corrupt("drop_index")
    assert seeded.replay() == before


def test_supersede_sets_valid_to_and_leaves_history(seeded):
    fid = seeded.fact_id_for({"pred": "reachable", "args": ["a", "b"]})
    seeded.supersede(fid, "regime change")
    row = seeded.index.one("SELECT valid_to FROM facts WHERE id=?", (fid,))
    assert row["valid_to"] is not None
    assert seeded.derive({"pred": "reachable", "args": ["a", "b"]},
                         BUDGET).status == "not_entailed"
