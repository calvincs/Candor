"""Ledger CHECKPOINT: replay from the latest VALID snapshot, fold only the tail.

The checkpoint is a DROPPABLE snapshot of the derived index keyed by the ledger
head hash it summarizes (<root>/checkpoints/<headhash>.sqlite3). It exists purely
to stop open()/_refold running from genesis. Correctness dominates: every test
here asserts the checkpoint-accelerated state is byte-identical to a forced
full-from-genesis replay (`replay()`, which ignores the cache).

The invariants under test:
  1. Bit-identical to full replay for every event kind (plain/alias/pin/supersede/
     gate/curiosity/reliability), incl. closure_hash AND a full table dump.
  2. Retroactive invalidation (THE CRUX): a retraction/redaction/restore appended
     AFTER a checkpoint retroactively rewrites events it froze, so the checkpoint
     must be refused and we fall back to full replay. One appearing BEFORE the
     checkpoint is baked in and the checkpoint stays valid. A `reliability` event
     after the checkpoint folds into the tail normally and does NOT invalidate.
  3. Re-derivation: the curiosity sweep re-runs after the tail folds (checked via
     closure_hash, which fingerprints flags/questions/breadth).
  4. I1: deleting the whole checkpoint cache changes nothing.
  5. Snapshot consistency: a torn/corrupt snapshot is detected and ignored.
"""

from __future__ import annotations

import random
import zlib
from pathlib import Path

from candor.system import CandorSystem


def seed_for(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


# ── comparison helpers ───────────────────────────────────────────────────────

def _dump(index) -> dict:
    """Full logical dump of every user table, each ordered deterministically.

    Compared between two FOLD reconstructions (checkpoint-open vs forced full
    replay), so even write-path-only tables (diagnostics) match: neither path
    goes through the live write API.
    """
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


def _assert_checkpoint_equals_full_replay(m, *, expect_fast: bool):
    """Open state (already computed by the caller's reopen) must equal a forced
    full replay of the same ledger. `expect_fast` asserts whether the fast path
    engaged, so a silent fallback can never masquerade as a pass."""
    if expect_fast:
        assert m._last_checkpoint_seq > 0, "expected the checkpoint fast path"
    else:
        assert m._last_checkpoint_seq == 0, "expected a full-replay fallback"
    hash_cp = m.closure_hash()
    dump_cp = _dump(m.index)
    hash_full = m.replay()          # forced full-from-genesis, ignores the cache
    dump_full = _dump(m.index)
    assert hash_cp == hash_full, "checkpoint closure_hash != full replay"
    assert dump_cp == dump_full, "checkpoint derived state != full replay"
    return hash_cp


# ── a store exercising many event kinds ──────────────────────────────────────

def _rich_burst(m, tag: str) -> None:
    """Append a varied, deterministic burst: symbols/facts, crisp+frequency
    observations (enough to drive the sweep), an alias, a pin, a supersede, a
    claim+resolution, and a reliability override."""
    rng = random.Random(seed_for(tag))
    m.assert_({"pred": f"reach_{tag}", "args": ["a", "b"], "stmt_type": "crisp"},
              source="seed", actor="human:me")
    m.assert_({"pred": f"flaky_{tag}", "args": ["c", "d"], "stmt_type": "frequency"},
              source="seed", actor="human:me")
    m.run_gate()

    fstmt = {"pred": f"flaky_{tag}", "args": ["c", "d"]}
    for i in range(40):                       # >= MIN_OBS so the sweep engages
        m.observe(fstmt, rng.random() < 0.8, {"env": "prod" if i % 2 else "dev"},
                  actor="tool:probe")
    cstmt = {"pred": f"reach_{tag}", "args": ["a", "b"]}
    for _ in range(5):
        m.observe(cstmt, True, {}, actor="tool:probe")

    m.assert_({"kind": "alias", "canonical": f"flaky_{tag}", "alias": f"lossy_{tag}",
               "basis": "pinned"}, source="test", actor="human:me")
    m.run_gate()

    fid = m.fact_id_for(cstmt)
    if fid:
        m.pin(fid, "+", reason="trusted", authority="human:me")
        m.supersede(fid, reason="rotate", actor="human:me")

    m.register_oracle(f"verifier:o_{tag}", "deterministic_total", "gt", "h", "e")
    cid = m.claim(cstmt, frame="external", criterion=f"verifier:o_{tag}", due=0)
    if cid != "Refused":
        m.resolve(cid, outcome=True)

    m.set_reliability("tool:probe", "external", 2.0, 1.0)


# ══ Requirement 1 & 3: bit-identical across every event kind, sweep re-derived ═

def test_all_kinds_checkpoint_equals_full_replay(tmp_path):
    root = tmp_path / "store"
    m = CandorSystem(root)
    _rich_burst(m, "p1")
    s_cp = m.ledger.seq()
    m.checkpoint()                    # snapshot at head after the first burst
    _rich_burst(m, "p2")             # tail: more of every kind
    n = m.ledger.seq()
    m.close()

    m2 = CandorSystem(root)          # open() must take the fast path
    assert m2._last_checkpoint_seq == s_cp
    assert m2._last_tail_folded == n - s_cp
    _assert_checkpoint_equals_full_replay(m2, expect_fast=True)
    m2.close()


def test_snapshot_is_pre_sweep_and_pre_closure(tmp_path):
    """The correctness pivot: the snapshot freezes the folded state BEFORE the
    curiosity sweep (which sets dispersion_flag monotonically and never clears
    it). The live store has swept state; the snapshot must not, so the reopened
    store re-sweeps over the complete observation set."""
    import sqlite3
    root = tmp_path / "store"
    m = CandorSystem(root)
    _rich_burst(m, "p1")
    head = m.ledger_head()
    # The live, swept store has breadth classified on its 40-obs frequency fact.
    live_breadth = m.index.query(
        "SELECT breadth_class FROM facts WHERE breadth_class IS NOT NULL")
    assert live_breadth, "expected the live sweep to classify at least one fact"
    m.checkpoint()
    m.close()

    snap = root / "checkpoints" / f"{head}.sqlite3"
    assert snap.exists()
    db = sqlite3.connect(str(snap))
    db.row_factory = sqlite3.Row
    flagged = db.execute("SELECT COUNT(*) c FROM facts WHERE dispersion_flag!=0").fetchone()["c"]
    breadth = db.execute("SELECT COUNT(*) c FROM facts WHERE breadth_class IS NOT NULL").fetchone()["c"]
    atoms = db.execute("SELECT COUNT(*) c FROM closure_atoms").fetchone()["c"]
    db.close()
    assert flagged == 0, "snapshot carries swept dispersion flags (not pre-sweep)"
    assert breadth == 0, "snapshot carries swept breadth (not pre-sweep)"
    assert atoms == 0, "snapshot carries materialized closure (not pre-closure)"


# ══ Requirement 2 (THE CRUX): retroactive invalidation ═════════════════════════

def _poisoned_store(root):
    """A crisp fact with honest and poison observations, quotas lifted."""
    m = CandorSystem(root)
    for a in ("tool:good", "tool:poison", "human:me"):
        m.set_actor_quota(a, obs_per_epoch=100_000, cand_per_epoch=100_000)
    for i in range(6):
        m.assert_({"pred": "holds", "args": [f"t{i}"], "stmt_type": "crisp"},
                  source="suite", actor="human:me")
    m.run_gate()
    rng = random.Random(7)
    for i in range(6):
        stmt = {"pred": "holds", "args": [f"t{i}"]}
        for _ in range(8):
            m.observe(stmt, rng.random() < 0.7, {}, actor="tool:good")
            m.observe(stmt, False, {}, actor="tool:poison")
    return m


def test_retraction_predating_checkpoint_stays_valid(tmp_path):
    """A retraction BEFORE the checkpoint is baked into the snapshot: the fast
    path stays valid and reproduces the source-never-spoke state."""
    root = tmp_path / "store"
    m = _poisoned_store(root)
    m.retract_source("tool:poison", reason="hallucinating")   # predates checkpoint
    for i in range(6):                                         # more, still silenced
        m.observe({"pred": "holds", "args": [f"t{i}"]}, False, {}, actor="tool:poison")
    s_cp = m.ledger.seq()
    m.checkpoint()
    for i in range(6):
        m.observe({"pred": "holds", "args": [f"t{i}"]}, True, {}, actor="tool:good")
    m.close()

    m2 = CandorSystem(root)
    assert m2._last_checkpoint_seq == s_cp, "checkpoint before a retraction must stay valid"
    _assert_checkpoint_equals_full_replay(m2, expect_fast=True)
    live = m2.index.one(
        "SELECT COUNT(*) n FROM observations WHERE actor='tool:poison'")["n"]
    m2.close()
    assert live == 0, "checkpoint let a retracted source keep speaking"


def test_retraction_after_checkpoint_invalidates(tmp_path):
    """THE CRUX. A retraction appended AFTER the checkpoint silences events the
    snapshot already counted, so the snapshot is stale and must be refused."""
    root = tmp_path / "store"
    m = _poisoned_store(root)
    s_cp = m.ledger.seq()
    m.checkpoint()                       # valid at creation (no retraction yet)
    m.retract_source("tool:poison", reason="hallucinating")   # seq > s_cp
    m.close()

    m2 = CandorSystem(root)
    assert m2._last_checkpoint_seq == 0, "stale checkpoint was trusted (CRUX failure)"
    _assert_checkpoint_equals_full_replay(m2, expect_fast=False)
    live = m2.index.one(
        "SELECT COUNT(*) n FROM observations WHERE actor='tool:poison'")["n"]
    m2.close()
    assert live == 0, "retraction did not take effect through the fallback"


def test_restore_after_checkpoint_invalidates(tmp_path):
    """A restore (retraction with restore=True) after the checkpoint un-silences
    a source the snapshot froze as silenced — equally retroactive, equally stale."""
    root = tmp_path / "store"
    m = _poisoned_store(root)
    m.retract_source("tool:poison", reason="bad")     # silence, before checkpoint
    s_cp = m.ledger.seq()
    m.checkpoint()                                     # snapshot has poison silenced
    m.retract_source("tool:poison", reason="cleared", restore=True)   # un-silence, after
    m.close()

    m2 = CandorSystem(root)
    assert m2._last_checkpoint_seq == 0, "a restore after the checkpoint was ignored"
    _assert_checkpoint_equals_full_replay(m2, expect_fast=False)
    live = m2.index.one(
        "SELECT COUNT(*) n FROM observations WHERE actor='tool:poison'")["n"]
    m2.close()
    assert live > 0, "restore did not bring the source back through the fallback"


def test_redaction_predating_checkpoint_stays_valid(tmp_path):
    root = tmp_path / "store"
    m = CandorSystem(root)
    m.assert_({"pred": "holds", "args": ["t0"], "stmt_type": "frequency"},
              source="s", actor="human:me")
    m.run_gate()
    stmt = {"pred": "holds", "args": ["t0"]}
    for i in range(20):
        m.observe(stmt, True, {"i": str(i)}, actor="tool:probe")
    sq = m.observe(stmt, False, {"marker": "unique-redactable"}, actor="tool:probe")
    ph = m.index.one("SELECT payload_hash FROM events WHERE seq=?", (sq,))["payload_hash"]
    m.redact(ph)                         # predates checkpoint
    for i in range(20, 30):
        m.observe(stmt, True, {"i": str(i)}, actor="tool:probe")
    s_cp = m.ledger.seq()
    m.checkpoint()
    for i in range(30, 36):
        m.observe(stmt, True, {"i": str(i)}, actor="tool:probe")
    m.close()

    m2 = CandorSystem(root)
    assert m2._last_checkpoint_seq == s_cp
    _assert_checkpoint_equals_full_replay(m2, expect_fast=True)
    m2.close()


def test_redaction_after_checkpoint_invalidates(tmp_path):
    """THE CRUX for content redaction: it removes a payload everywhere, including
    events at seq <= S the snapshot counted."""
    root = tmp_path / "store"
    m = CandorSystem(root)
    m.assert_({"pred": "holds", "args": ["t0"], "stmt_type": "frequency"},
              source="s", actor="human:me")
    m.run_gate()
    stmt = {"pred": "holds", "args": ["t0"]}
    for i in range(20):
        m.observe(stmt, True, {"i": str(i)}, actor="tool:probe")
    sq = m.observe(stmt, False, {"marker": "unique-redactable"}, actor="tool:probe")
    ph = m.index.one("SELECT payload_hash FROM events WHERE seq=?", (sq,))["payload_hash"]
    s_cp = m.ledger.seq()
    m.checkpoint()                       # snapshot still counts the soon-redacted event
    m.redact(ph)                         # seq > s_cp
    m.close()

    m2 = CandorSystem(root)
    assert m2._last_checkpoint_seq == 0, "stale checkpoint trusted across a redaction"
    _assert_checkpoint_equals_full_replay(m2, expect_fast=False)
    m2.close()


def test_reliability_after_checkpoint_does_not_invalidate(tmp_path):
    """A reliability override sets a table in ledger order; folded into the tail
    it does NOT retroactively alter pre-checkpoint counts, so the checkpoint
    stays valid and the fast path engages."""
    root = tmp_path / "store"
    m = CandorSystem(root)
    _rich_burst(m, "p1")
    s_cp = m.ledger.seq()
    m.checkpoint()
    m.set_reliability("tool:probe", "external", 5.0, 2.0)   # after the checkpoint
    m.set_reliability("tool:good", "internal", 1.0, 3.0)
    m.close()

    m2 = CandorSystem(root)
    assert m2._last_checkpoint_seq == s_cp, "a reliability event wrongly invalidated"
    _assert_checkpoint_equals_full_replay(m2, expect_fast=True)
    m2.close()


# ══ Requirement 4 (I1): the cache is droppable ═════════════════════════════════

def test_deleting_the_cache_changes_nothing(tmp_path):
    import shutil
    root = tmp_path / "store"
    m = CandorSystem(root)
    _rich_burst(m, "p1")
    m.checkpoint()
    _rich_burst(m, "p2")
    m.close()

    with_cache = CandorSystem(root)
    assert with_cache._last_checkpoint_seq > 0
    hash_with = with_cache.closure_hash()
    with_cache.close()

    shutil.rmtree(root / "checkpoints")          # drop the ENTIRE cache
    without = CandorSystem(root)
    assert without._last_checkpoint_seq == 0, "phantom checkpoint after cache delete"
    hash_without = without.closure_hash()
    without.close()

    assert hash_with == hash_without, "state changed when the cache was deleted (I1)"


# ══ Requirement 5: torn / corrupt snapshot detection ═══════════════════════════

def test_corrupt_snapshot_is_detected_and_ignored(tmp_path):
    root = tmp_path / "store"
    m = CandorSystem(root)
    _rich_burst(m, "p1")
    head = m.ledger_head()
    m.checkpoint()
    _rich_burst(m, "p2")
    good_hash = m.closure_hash()
    m.close()

    snap = root / "checkpoints" / f"{head}.sqlite3"
    raw = bytearray(snap.read_bytes())
    raw[0:16] = b"\x00" * 16                     # clobber the SQLite header magic
    snap.write_bytes(bytes(raw))

    m2 = CandorSystem(root)
    assert m2._last_checkpoint_seq == 0, "a corrupt snapshot was trusted"
    assert m2.closure_hash() == good_hash, "fallback replay diverged from the truth"
    m2.close()


def test_truncated_snapshot_is_detected_and_ignored(tmp_path):
    root = tmp_path / "store"
    m = CandorSystem(root)
    _rich_burst(m, "p1")
    head = m.ledger_head()
    m.checkpoint()
    _rich_burst(m, "p2")
    good_hash = m.closure_hash()
    m.close()

    snap = root / "checkpoints" / f"{head}.sqlite3"
    raw = snap.read_bytes()
    snap.write_bytes(raw[: max(1, len(raw) // 3)])   # partial write: pages missing

    m2 = CandorSystem(root)
    assert m2._last_checkpoint_seq == 0, "a truncated snapshot was trusted"
    assert m2.closure_hash() == good_hash
    m2.close()


# ══ Performance: open() folds only the post-checkpoint tail ════════════════════

def test_open_folds_only_the_tail(tmp_path):
    root = tmp_path / "store"
    m = CandorSystem(root)
    m.set_actor_quota("tool:probe", obs_per_epoch=100_000, cand_per_epoch=100_000)
    m.assert_({"pred": "holds", "args": ["t0"], "stmt_type": "frequency"},
              source="s", actor="human:me")
    m.run_gate()
    stmt = {"pred": "holds", "args": ["t0"]}
    for _ in range(500):
        m.observe(stmt, True, {}, actor="tool:probe")
    s_cp = m.ledger.seq()
    m.checkpoint()
    for _ in range(5):
        m.observe(stmt, True, {}, actor="tool:probe")
    n = m.ledger.seq()
    m.close()

    m2 = CandorSystem(root)
    assert m2._last_checkpoint_seq == s_cp
    assert m2._last_tail_folded == n - s_cp == 5, (
        f"folded {m2._last_tail_folded} tail events, expected {n - s_cp}")
    # And a forced full replay of the same store folds the whole log.
    m2.replay()
    assert m2._last_tail_folded == n, "forced replay should fold the whole log"
    m2.close()


def test_checkpoint_at_head_folds_zero_tail(tmp_path):
    root = tmp_path / "store"
    m = CandorSystem(root)
    _rich_burst(m, "p1")
    s_cp = m.ledger.seq()
    m.checkpoint()
    m.close()

    m2 = CandorSystem(root)
    assert m2._last_checkpoint_seq == s_cp
    assert m2._last_tail_folded == 0, "checkpoint at head still folded a tail"
    _assert_checkpoint_equals_full_replay(m2, expect_fast=True)
    m2.close()


def test_empty_ledger_checkpoint_is_a_noop(tmp_path):
    root = tmp_path / "store"
    m = CandorSystem(root)
    assert m.checkpoint() is None, "checkpointing an empty ledger should be a no-op"
    assert not (root / "checkpoints").exists() or not list(
        (root / "checkpoints").glob("*.sqlite3"))
    m.close()
