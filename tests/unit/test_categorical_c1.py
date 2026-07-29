"""Categorical facts — Stage C1 foundation (design_categorical.md §1/§5/§6/§8).

C1 is storage + fold + audit wiring ONLY: a new open-vocabulary `categorical`
statement type, per-value integer tallies in `fact_category_counts`, the
`observations.value` column, `observe(value=...)`, and the two audit-invariant
list additions (COUNT_COLUMNS for I11, _HASH_QUERIES for replay determinism).

There is NO posterior/predict here (that is C2): a categorical fact must simply
STORE correctly and fold deterministically. `predict` guards with a clear
NotImplementedError rather than returning a silent wrong scalar answer.

The gates enforced here:
  * ADDITIVITY — a crisp+frequency store's derived state is byte-for-byte what it
    was before this change (the only delta is one empty per-value-counts hash
    entry), and full closure_hash still equals a forced replay.
  * ADMIT + OPEN SET — a categorical fact admits; observe(value=...) increments
    the right rows; a brand-new value is a brand-new row.
  * I11 — the integrality scan passes over the new table and DETECTS a planted
    real (negative control).
  * DETERMINISM (I3) — replay() reproduces closure_hash INCLUDING per-value
    counts, and stored counts are fold-order independent.
  * REDACTION / RETRACTION — both exclude a categorical observer's per-value
    counts on refold (retraction reverses on restore).
  * CHECKPOINT — a categorical store's checkpoint-accelerated open is byte-
    identical to a forced full-from-genesis replay (new table + column ride the
    snapshot AND enter the replay hash).
"""

from __future__ import annotations

import random

import pytest

from candor.core.apply import _HASH_QUERIES
from candor.core.hashing import canon_json, sha256_hex
from candor.system import CandorSystem

# ── the additivity golden ─────────────────────────────────────────────────────
# Full closure_hash of the crisp+frequency store built by `_crisp_freq_store`
# below, computed on the PRE-categorical code (feat/distribution-surfacing) with
# the wall clock frozen (see FROZEN_TS). After this change the SAME store's hash
# over the SAME (pre-categorical) query set must reproduce this byte-for-byte:
# proof that no crisp/frequency/two-coin derived state moved. The only change the
# feature makes to a no-categorical store is appending one empty
# `fact_category_counts` entry to the full hash.
BASE_GOLDEN = "c0e20e8a84e4bad9770a329a6e83c652bb7a111d6077c05ff98fc30df6f344d1"
FROZEN_TS = 1_700_000.0


def _legacy_hash(index) -> str:
    """closure_hash over the PRE-categorical query set (excludes the new table)."""
    parts = []
    for name, sql in _HASH_QUERIES:
        if name == "fact_category_counts":
            continue
        parts.append([name, [[*row] for row in index.query(sql)]])
    return sha256_hex(canon_json(parts))


# ── store builders ────────────────────────────────────────────────────────────

def _crisp_freq_store(root) -> CandorSystem:
    """The additivity fixture: one crisp fact, one frequency fact, observed.
    Deterministic (fixed seed); must be built under a frozen clock to reproduce
    BASE_GOLDEN, since facts.valid_from/admitted_at carry the event timestamp."""
    m = CandorSystem(root)
    m.set_actor_quota("tool:probe", obs_per_epoch=100_000, cand_per_epoch=100_000)
    m.assert_({"pred": "reachable", "args": ["a", "b"], "stmt_type": "crisp"},
              source="seed", actor="human:calvin")
    m.assert_({"pred": "flaky", "args": ["c", "d"], "stmt_type": "frequency"},
              source="seed", actor="human:calvin")
    m.run_gate()
    cstmt = {"pred": "reachable", "args": ["a", "b"]}
    for _ in range(5):
        m.observe(cstmt, True, {}, actor="tool:probe")
    fstmt = {"pred": "flaky", "args": ["c", "d"]}
    rng = random.Random(1234)
    for i in range(40):
        m.observe(fstmt, rng.random() < 0.7, {"env": "prod" if i % 2 else "dev"},
                  actor="tool:probe")
    m.run_gate()
    return m


CAT_STMT = {"pred": "resolves", "args": ["login"]}


def _categorical_store(root) -> CandorSystem:
    """A store with one admitted categorical fact and quotas lifted."""
    m = CandorSystem(root)
    for a in ("tool:probe", "tool:liar", "human:me"):
        m.set_actor_quota(a, obs_per_epoch=100_000, cand_per_epoch=100_000)
    m.assert_({"pred": "resolves", "args": ["login"], "stmt_type": "categorical"},
              source="seed", actor="human:me")
    m.run_gate()
    return m


def _cat_counts(m, fid) -> dict[tuple[str, str], int]:
    return {(r["actor"], r["value"]): int(r["n"]) for r in m.index.query(
        "SELECT actor, value, n FROM fact_category_counts WHERE fact_id=?", (fid,))}


def _dump(index) -> dict:
    """Full logical dump of every user table, each ordered deterministically —
    so the new `fact_category_counts` table and `observations.value` column are
    both compared between two fold reconstructions."""
    tables = [r["name"] for r in index.query(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    out: dict = {}
    for t in tables:
        cols = [r["name"] for r in index.query(f"PRAGMA table_info({t})")]
        order = ", ".join(cols) if cols else "1"
        rows = index.query(f"SELECT * FROM {t} ORDER BY {order}")
        out[t] = [tuple(r[c] for c in cols) for r in rows]
    return out


# ══ ADDITIVITY ═════════════════════════════════════════════════════════════════

def test_crisp_freq_derived_state_is_byte_identical_to_before(tmp_path, monkeypatch):
    """The categorical feature is strictly additive: a store with no categorical
    facts folds NOTHING new, and its crisp/frequency derived state is byte-for-
    byte the pre-change closure_hash. The only difference the change makes to the
    full hash is one empty `fact_category_counts` entry."""
    monkeypatch.setattr("time.time", lambda: FROZEN_TS)   # reproducible timestamps
    m = _crisp_freq_store(tmp_path / "store")

    # Nothing categorical folded.
    assert m.index.query("SELECT * FROM fact_category_counts") == []
    assert m.index.one(
        "SELECT COUNT(*) c FROM observations WHERE value IS NOT NULL")["c"] == 0

    # The crisp/frequency-derived portion is byte-identical to before this change.
    # closure_hash() first, to materialize closure_atoms (which is in the set).
    full = m.closure_hash()
    assert _legacy_hash(m.index) == BASE_GOLDEN, \
        "crisp/frequency derived state changed (additivity broken)"

    # Determinism is intact through the new fold code: full hash == forced replay.
    assert full == m.replay()
    m.close()


# ══ ADMIT + OPEN VOCABULARY ════════════════════════════════════════════════════

def test_categorical_fact_admits_through_the_gate(tmp_path):
    m = _categorical_store(tmp_path / "store")
    fid = m.fact_id_for(CAT_STMT)
    row = m.index.one("SELECT stmt_type, structural FROM facts WHERE id=?", (fid,))
    assert row is not None
    assert row["stmt_type"] == "categorical"
    assert row["structural"] != "candidate", "categorical fact never left candidacy"
    m.close()


def test_observing_values_increments_the_right_open_set_rows(tmp_path):
    """observe(value=X)/Y/a brand-NEW Z each increment the right per-value row;
    the vocabulary is OPEN — a value never seen before is a new row."""
    m = _categorical_store(tmp_path / "store")
    fid = m.fact_id_for(CAT_STMT)

    for v in ("captcha", "captcha", "block"):
        m.observe(CAT_STMT, ctx={}, actor="tool:probe", value=v)
    assert _cat_counts(m, fid) == {("tool:probe", "captcha"): 2,
                                   ("tool:probe", "block"): 1}

    # A brand-new value "allow" appears as a brand-new row (open set).
    m.observe(CAT_STMT, ctx={}, actor="tool:probe", value="allow")
    assert _cat_counts(m, fid)[("tool:probe", "allow")] == 1

    # The value, channel='cat' and NULL outcome are recorded on the observation.
    obs = m.index.one(
        "SELECT value, channel, outcome FROM observations "
        "WHERE fact_id=? ORDER BY event_seq LIMIT 1", (fid,))
    assert (obs["value"], obs["channel"], obs["outcome"]) == ("captcha", "cat", None)
    m.close()


def test_a_frozen_categorical_target_is_a_no_op(tmp_path):
    """apply_category_observation respects numeric='frozen' exactly like the
    crisp/frequency updater (§1.4)."""
    m = _categorical_store(tmp_path / "store")
    fid = m.fact_id_for(CAT_STMT)
    m.index.execute("UPDATE facts SET numeric='frozen' WHERE id=?", (fid,))
    m.index.commit()
    m.observe(CAT_STMT, ctx={}, actor="tool:probe", value="captcha")
    assert _cat_counts(m, fid) == {}, "a frozen categorical target still tallied"
    m.close()


# ══ I11 INTEGRALITY (with a negative control) ══════════════════════════════════

def test_i11_passes_clean_and_detects_a_planted_real(tmp_path):
    m = _categorical_store(tmp_path / "store")
    fid = m.fact_id_for(CAT_STMT)
    for v in ("captcha", "block", "captcha"):
        m.observe(CAT_STMT, ctx={}, actor="tool:probe", value=v)

    assert m.index.nonintegral_counts() == [], "clean categorical store failed I11"

    # Negative control: plant a REAL in the per-value count column.
    m.index.execute(
        "UPDATE fact_category_counts SET n=1.5 WHERE fact_id=? AND value='block'",
        (fid,))
    m.index.commit()
    offenders = m.index.nonintegral_counts()
    assert any(o.startswith("fact_category_counts.n[") and o.endswith(":real")
               for o in offenders), \
        f"integrality scan missed a real planted in the new table: {offenders}"
    m.close()


# ══ DETERMINISM / REPLAY (I3) ══════════════════════════════════════════════════

def test_replay_reproduces_closure_hash_including_per_value_counts(tmp_path):
    m = _categorical_store(tmp_path / "store")
    fid = m.fact_id_for(CAT_STMT)
    rng = random.Random(7)
    vocab = ["captcha", "block", "allow", "novel-value"]
    for _ in range(30):
        m.observe(CAT_STMT, ctx={}, actor="tool:probe", value=rng.choice(vocab))
    m.run_gate()          # sync the live sweep so live state == a full refold
    assert _cat_counts(m, fid), "vacuous — no per-value counts to hash"

    live = m.closure_hash()
    assert live == m.replay(), "replay diverged from live (per-value counts, I3)"

    # The hash actually COVERS the per-value counts: perturb one and it must move.
    # (Without the _HASH_QUERIES entry this mutation would be invisible.)
    m.index.execute(
        "UPDATE fact_category_counts SET n=n+1 WHERE fact_id=? AND value='captcha'",
        (fid,))
    m.index.commit()
    assert m.closure_hash() != live, \
        "closure_hash does not cover fact_category_counts (replay equality vacuous)"
    m.close()


def test_per_value_counts_are_fold_order_independent(tmp_path):
    """Integer increments commute: the same multiset of values in a different
    order folds to identical stored counts (§1.5)."""
    seq_a = ["captcha", "block", "captcha", "allow", "captcha", "block"]
    seq_b = list(reversed(seq_a))

    ma = _categorical_store(tmp_path / "a")
    fa = ma.fact_id_for(CAT_STMT)
    for v in seq_a:
        ma.observe(CAT_STMT, ctx={}, actor="tool:probe", value=v)
    counts_a = _cat_counts(ma, fa)
    ma.close()

    mb = _categorical_store(tmp_path / "b")
    fb = mb.fact_id_for(CAT_STMT)
    for v in seq_b:
        mb.observe(CAT_STMT, ctx={}, actor="tool:probe", value=v)
    counts_b = _cat_counts(mb, fb)
    mb.close()

    assert counts_a == counts_b == {("tool:probe", "captcha"): 3,
                                    ("tool:probe", "block"): 2,
                                    ("tool:probe", "allow"): 1}


# ══ REDACTION / RETRACTION ═════════════════════════════════════════════════════

def test_retraction_excludes_a_categorical_observers_counts_and_restore_reverses(
        tmp_path):
    m = _categorical_store(tmp_path / "store")
    fid = m.fact_id_for(CAT_STMT)
    for v in ("captcha", "captcha"):
        m.observe(CAT_STMT, ctx={}, actor="tool:probe", value=v)
    for v in ("spam", "spam", "spam"):
        m.observe(CAT_STMT, ctx={}, actor="tool:liar", value=v)
    assert _cat_counts(m, fid)[("tool:liar", "spam")] == 3

    m.retract_source("tool:liar", reason="value-fabricating")   # refolds
    counts = _cat_counts(m, fid)
    assert not any(actor == "tool:liar" for actor, _ in counts), \
        "retracted source still has per-value counts after refold"
    assert counts[("tool:probe", "captcha")] == 2, "honest counts wrongly dropped"
    assert m.closure_hash() == m.replay()

    m.retract_source("tool:liar", reason="cleared", restore=True)   # refolds
    assert _cat_counts(m, fid)[("tool:liar", "spam")] == 3, \
        "restore did not bring the categorical counts back"
    m.close()


def test_redacting_a_categorical_observation_excludes_it(tmp_path):
    """Redaction is content-addressed; distinct ctx gives distinct payloads so
    exactly one categorical observation is purged and its per-value tally drops."""
    m = _categorical_store(tmp_path / "store")
    fid = m.fact_id_for(CAT_STMT)
    seqs = []
    for i in range(3):
        seqs.append(m.observe(CAT_STMT, ctx={"i": str(i)}, actor="tool:probe",
                              value="captcha"))
    assert _cat_counts(m, fid)[("tool:probe", "captcha")] == 3

    ph = m.index.one("SELECT payload_hash FROM events WHERE seq=?",
                     (seqs[1],))["payload_hash"]
    m.redact(ph)                                        # refolds, excludes payload
    assert _cat_counts(m, fid)[("tool:probe", "captcha")] == 2, \
        "redacted categorical observation still counted"
    assert m.closure_hash() == m.replay()
    m.close()


# ══ CHECKPOINT (Phase 4): fast-path open == forced full replay ═════════════════

def test_checkpoint_accelerated_open_equals_full_replay_with_categorical(tmp_path):
    """A categorical store with a checkpoint: the checkpoint-accelerated open must
    be byte-identical to a forced full-from-genesis replay. The new table rides
    the snapshot AND is in _HASH_QUERIES, so both the hash and a full table dump
    (which includes fact_category_counts + observations.value) must match."""
    root = tmp_path / "store"
    m = _categorical_store(root)
    rng = random.Random(11)
    vocab = ["captcha", "block", "allow"]
    for _ in range(20):                                 # before the checkpoint
        m.observe(CAT_STMT, ctx={}, actor="tool:probe", value=rng.choice(vocab))
    s_cp = m.ledger.seq()
    m.checkpoint()
    for _ in range(8):                                  # tail: incl. a NEW value
        m.observe(CAT_STMT, ctx={}, actor="tool:probe", value="tail-novel")
    n = m.ledger.seq()
    m.close()

    m2 = CandorSystem(root)                             # open() takes the fast path
    assert m2._last_checkpoint_seq == s_cp, "expected the checkpoint fast path"
    assert m2._last_tail_folded == n - s_cp
    fid = m2.fact_id_for(CAT_STMT)
    assert _cat_counts(m2, fid)[("tool:probe", "tail-novel")] == 8, \
        "tail-only categorical value missed by the fast path"

    hash_cp = m2.closure_hash()
    dump_cp = _dump(m2.index)
    hash_full = m2.replay()                             # forced full, ignores cache
    dump_full = _dump(m2.index)
    assert hash_cp == hash_full, "categorical checkpoint hash != full replay"
    assert dump_cp == dump_full, "categorical checkpoint state != full replay"
    m2.close()


# ══ predict is guarded until C2 ════════════════════════════════════════════════

def test_predict_on_a_categorical_fact_raises_until_c2(tmp_path):
    m = _categorical_store(tmp_path / "store")
    for v in ("captcha", "block"):
        m.observe(CAT_STMT, ctx={}, actor="tool:probe", value=v)
    with pytest.raises(NotImplementedError, match="C2"):
        m.predict(CAT_STMT, budget=1000)
    m.close()
