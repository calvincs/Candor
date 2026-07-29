"""redaction_scope() blast radius + retract_source(restore=True) round-trip (§3.13).

Two knobs on the same axis, tested against the property that gives them meaning:
payloads are content-addressed and deduplicated, so a hash carries no actor.

  * redaction_scope reports WHO would lose data if a payload were purged. On a
    payload only one actor produced it is `shared=False`; on a payload two actors
    independently produced (same stmt, same outcome, same ctx -> one hash) it is
    `shared=True` and BOTH actors are named. That collision is the whole reason
    the call exists: it is what makes redact() dangerous as a source-purge and
    why retract_source() is the right tool instead.

  * retract_source(restore=True) un-silences a previously retracted source. The
    history never leaves the chain; every downstream number recomputes as if the
    source had spoken all along. The strongest statement of "recomputes exactly"
    is a byte-identical closure_hash, and the whole thing survives a forced
    from-genesis replay (I1).
"""

from __future__ import annotations

import pytest


FLAKY = {"pred": "flaky_link", "args": ["c", "d"]}


def _payload_hash_of(system, seq):
    """The content-address of the event appended at `seq`."""
    return system.events_since(seq - 1)[0]["payload_hash"]


# ── redaction_scope: blast radius ────────────────────────────────────────────

def test_redaction_scope_on_a_unique_payload_is_not_shared(seeded):
    """One actor, one distinctive observation -> the hash names exactly them."""
    seq = seeded.observe(FLAKY, True, {"run": "solo-42"}, actor="tool:probe")
    ph = _payload_hash_of(seeded, seq)

    scope = seeded.redaction_scope(ph)
    assert scope["payload_hash"] == ph
    assert scope["events"] == 1
    assert scope["actors"] == ["tool:probe"]
    assert scope["shared"] is False


def test_redaction_scope_on_a_shared_payload_names_both_colliding_actors(seeded):
    """The key edge case: two sources reporting the IDENTICAL observation
    (same stmt, same outcome, same ctx) dedup onto ONE content-address, so
    redacting it would silently destroy the honest report that agreed."""
    seq_a = seeded.observe(FLAKY, True, {}, actor="tool:probe")
    seq_b = seeded.observe(FLAKY, True, {}, actor="tool:mirror")
    ph_a = _payload_hash_of(seeded, seq_a)
    ph_b = _payload_hash_of(seeded, seq_b)
    assert ph_a == ph_b, "two identical observations must collide on one hash"

    scope = seeded.redaction_scope(ph_a)
    assert scope["events"] == 2
    assert scope["actors"] == ["tool:mirror", "tool:probe"]  # returned sorted
    assert scope["shared"] is True


def test_redaction_scope_does_not_collide_across_a_differing_ctx(seeded):
    """Content-addressing is exact: the same actors reporting the same outcome
    in DIFFERENT contexts land on different hashes -> each scope is unshared."""
    seq_a = seeded.observe(FLAKY, True, {"elevation": "high"}, actor="tool:probe")
    seq_b = seeded.observe(FLAKY, True, {"elevation": "sea"}, actor="tool:mirror")
    ph_a = _payload_hash_of(seeded, seq_a)
    ph_b = _payload_hash_of(seeded, seq_b)
    assert ph_a != ph_b, "a differing ctx must not dedup onto one payload"
    assert seeded.redaction_scope(ph_a)["shared"] is False
    assert seeded.redaction_scope(ph_b)["shared"] is False


# ── retract_source(restore=True): reversible silencing ───────────────────────

def test_retract_then_restore_round_trips_closure_hash(seeded):
    """Silence a source, watch every number it moved drop, then restore it and
    assert the committed state is byte-identical to before it was ever touched.

    The baseline is taken after run_gate(): retract_source performs a FULL
    refold (curiosity/dispersion sweep included), so the pre-retraction state
    must be fully folded too, or the round-trip would compare against a store
    whose breadth_class was still stale from the raw observe() calls. That is a
    property of the baseline, not of restore -- the numbers below round-trip
    regardless."""
    fid = seeded.fact_id_for(FLAKY)
    for _ in range(12):
        seeded.observe(FLAKY, True, {}, actor="tool:probe")
    for _ in range(4):
        seeded.observe(FLAKY, False, {}, actor="tool:probe")
    seeded.run_gate()

    closure_before = seeded.closure_hash()
    p_before = seeded.predict(FLAKY, budget=10_000).p
    counts_before = seeded.raw_counts(fid)
    assert counts_before[("tool:probe", "alea")] == (16, 12)

    # Silence: the source's 16 observations stop counting.
    seeded.retract_source("tool:probe", reason="suspected scraper drift")
    assert seeded.closure_hash() != closure_before, "retraction moved no number (vacuous)"
    assert seeded.predict(FLAKY, budget=10_000).p != pytest.approx(p_before, abs=1e-9)
    assert seeded.raw_counts(fid).get(("tool:probe", "alea")) in (None, (0, 0)), \
        "a silenced source still carries committed counts"

    # Restore: the whole history recomputes as if it had spoken all along.
    seeded.retract_source("tool:probe", reason="cleared on review", restore=True)
    assert seeded.closure_hash() == closure_before, \
        "restore did not reproduce the pre-retraction closure (I1)"
    assert seeded.predict(FLAKY, budget=10_000).p == pytest.approx(p_before, abs=1e-12)
    assert seeded.raw_counts(fid) == counts_before


def test_restore_is_replay_stable(seeded):
    """After a retract/restore cycle the live closure must equal a forced
    from-genesis replay: the two extra ledger control events fold to nothing."""
    for _ in range(6):
        seeded.observe(FLAKY, True, {}, actor="tool:probe")
        seeded.observe(FLAKY, False, {}, actor="tool:probe")
    seeded.retract_source("tool:probe", reason="quarantine")
    seeded.retract_source("tool:probe", reason="reinstated", restore=True)

    live = seeded.closure_hash()
    replayed = seeded.replay()          # forced full-from-genesis fold (I1 oracle)
    assert live == replayed, "restored store: live closure != forced replay"


def test_double_restore_is_idempotent(seeded):
    """A restore on an already-live source is a no-op for the committed number:
    restore is defined by the final control state, not by event count."""
    for _ in range(9):
        seeded.observe(FLAKY, True, {}, actor="tool:probe")
    seeded.run_gate()                   # fully fold the baseline (see round-trip test)
    baseline = seeded.closure_hash()

    seeded.retract_source("tool:probe", reason="hold")
    seeded.retract_source("tool:probe", reason="restore-1", restore=True)
    once = seeded.closure_hash()
    seeded.retract_source("tool:probe", reason="restore-2", restore=True)
    twice = seeded.closure_hash()

    assert once == baseline
    assert twice == baseline


# ── redact() surfaces the shared-payload hazard ──────────────────────────────

def test_redact_of_a_shared_payload_emits_the_health_warning(seeded):
    """redact() consults redaction_scope and, when the payload is shared, records
    a diagnostic naming the collateral actors -- the operator-facing signal that
    a source-purge just took out an innocent bystander."""
    seq_a = seeded.observe(FLAKY, True, {}, actor="tool:probe")
    seeded.observe(FLAKY, True, {}, actor="tool:mirror")
    ph = _payload_hash_of(seeded, seq_a)
    assert seeded.redaction_scope(ph)["shared"] is True

    seeded.redact(ph)

    warnings = [e for e in seeded._health_events
                if e.get("kind") == "redaction_shared_payload"]
    assert warnings, "shared-payload redaction emitted no health warning"
    w = warnings[-1]
    assert w["shared"] is True
    assert w["events"] == 2
    assert set(w["actors"]) == {"tool:probe", "tool:mirror"}


def test_redact_of_a_unique_payload_stays_quiet(seeded):
    """The warning is not noise: a single-actor payload trips no shared-payload
    diagnostic (the branch is genuinely gated on scope['shared'])."""
    seq = seeded.observe(FLAKY, True, {"run": "lonely"}, actor="tool:probe")
    ph = _payload_hash_of(seeded, seq)
    assert seeded.redaction_scope(ph)["shared"] is False

    seeded.redact(ph)

    assert not [e for e in seeded._health_events
                if e.get("kind") == "redaction_shared_payload"], \
        "a unique-payload redaction should not warn about sharing"
