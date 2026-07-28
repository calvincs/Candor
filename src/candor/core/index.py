"""SQLite derived index over the ledger (spec §2, invariant I1).

Everything in this file is a materialized view. Deleting the database file
loses nothing: `Rebuilder.replay()` reconstructs it from the JSONL segments.

Storage discipline (I11): every count column is INTEGER. No real-valued number
is ever written to a count column — reliability discounts, alias unions and
isotonic maps are all read-time compositions.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

-- ── LEDGER INDEX (derived) ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS events(
  seq INTEGER PRIMARY KEY, ts INTEGER NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN
    ('assertion','observation','supersede','admission','demotion','pin','claim',
     'resolution','alias','redaction','retraction','checkpoint')),
  actor TEXT NOT NULL, payload_hash TEXT NOT NULL, source_ref TEXT,
  context_sig TEXT, prev_hash TEXT NOT NULL, hash TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS redactions(payload_hash TEXT PRIMARY KEY, event_seq INTEGER);
-- Source retraction: silences an ACTOR, where redaction purges a PAYLOAD.
-- Distinct because payloads are content-addressed and carry no actor, so
-- honest sources reporting the same outcome share a hash with a liar.
CREATE TABLE IF NOT EXISTS retractions(
  actor TEXT PRIMARY KEY, event_seq INTEGER NOT NULL, reason TEXT,
  restored INTEGER NOT NULL DEFAULT 0);

-- ── ACTORS & RELIABILITY ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS actors(
  name TEXT PRIMARY KEY,
  class TEXT NOT NULL CHECK(class IN ('human','verifier','tool','agent')),
  obs_quota_per_epoch INTEGER NOT NULL, cand_quota_per_epoch INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS actor_reliability(
  actor TEXT NOT NULL, frame TEXT NOT NULL CHECK(frame IN ('internal','external')),
  rel_a REAL NOT NULL, rel_b REAL NOT NULL, PRIMARY KEY(actor, frame));
-- v0.3 Δ1: the two-coin confusion table. Four INTEGER cells per (actor, frame),
-- moved only by the trusted settlement path; every real number (sens, fpr,
-- likelihood ratios) is a read-time composition (I11).
-- v0.4 Δ6: categorical response ledger. Integers, settlement-moved only.
CREATE TABLE IF NOT EXISTS actor_response(
  actor TEXT NOT NULL, frame TEXT NOT NULL,
  vote INTEGER NOT NULL, grade INTEGER NOT NULL,
  n_true INTEGER NOT NULL DEFAULT 0, n_false INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(actor, frame, vote, grade));
CREATE TABLE IF NOT EXISTS actor_confusion(
  actor TEXT NOT NULL, frame TEXT NOT NULL CHECK(frame IN ('internal','external')),
  tp INTEGER NOT NULL DEFAULT 0, fn INTEGER NOT NULL DEFAULT 0,
  fp INTEGER NOT NULL DEFAULT 0, tn INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(actor, frame));
CREATE TABLE IF NOT EXISTS quota_usage(
  actor TEXT NOT NULL, epoch INTEGER NOT NULL, kind TEXT NOT NULL,
  used INTEGER NOT NULL, PRIMARY KEY(actor, epoch, kind));

-- ── IDENTITY & SCHEMA ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predicates(
  pred TEXT PRIMARY KEY, arity INTEGER NOT NULL, arg_types_json TEXT NOT NULL,
  canonical_units_json TEXT, admitted_at INTEGER, admitted_by_event INTEGER);
CREATE TABLE IF NOT EXISTS aliases(
  canonical TEXT NOT NULL, alias TEXT NOT NULL,
  basis TEXT NOT NULL CHECK(basis IN ('behavioral','definitional','pinned')),
  admitted_at INTEGER, admitted_by_event INTEGER,
  PRIMARY KEY(canonical, alias));

-- ── CANDIDATES (never facts) ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS candidates(
  id TEXT PRIMARY KEY, event_seq INTEGER NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN
    ('fact','rule','guard','verifier','symbol','alias','constraint',
     'supersede_valid_time')),
  body_json TEXT NOT NULL, span_ref TEXT, proposer TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pending','admitted','rejected','superseded')),
  gate_run_id TEXT, failing_step INTEGER, reason TEXT);

-- ── COMMITTED TIER ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS facts(
  id TEXT PRIMARY KEY, pred TEXT NOT NULL, args_json TEXT NOT NULL,
  stmt_type TEXT NOT NULL CHECK(stmt_type IN ('crisp','frequency')),
  kind TEXT NOT NULL CHECK(kind IN ('exact','soft','definitional')),
  sim REAL,
  structural TEXT NOT NULL CHECK(structural IN ('candidate','admitted','pinned')),
  numeric TEXT NOT NULL CHECK(numeric IN ('accumulating','frozen')),
  breadth_class TEXT CHECK(breadth_class IN ('narrow','moderate','broad')),
  dispersion_flag INTEGER NOT NULL DEFAULT 0,
  valid_from INTEGER, valid_to INTEGER,
  admitted_at INTEGER, admitted_by_event INTEGER);

-- The RAW counts. Integers. The audit trail. (I11)
CREATE TABLE IF NOT EXISTS fact_counts(
  fact_id TEXT NOT NULL, actor TEXT NOT NULL,
  channel TEXT NOT NULL CHECK(channel IN ('epi','alea')),
  n INTEGER NOT NULL, k INTEGER NOT NULL,
  PRIMARY KEY(fact_id, actor, channel));

CREATE TABLE IF NOT EXISTS rules(
  id TEXT PRIMARY KEY, head_json TEXT NOT NULL, body_json TEXT NOT NULL,
  specificity INTEGER NOT NULL DEFAULT 0, parent_rule_id TEXT,
  structural TEXT NOT NULL, numeric TEXT NOT NULL,
  gate_run_id TEXT, admitted_at INTEGER);
CREATE TABLE IF NOT EXISTS rule_counts(
  rule_id TEXT NOT NULL, actor TEXT NOT NULL, n INTEGER NOT NULL, k INTEGER NOT NULL,
  PRIMARY KEY(rule_id, actor));

CREATE TABLE IF NOT EXISTS constraints(
  id TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('mutex','functional')),
  body_json TEXT NOT NULL, structural TEXT NOT NULL,
  admitted_at INTEGER, gate_run_id TEXT);

CREATE TABLE IF NOT EXISTS pins(
  id TEXT PRIMARY KEY, target_kind TEXT NOT NULL, target_id TEXT NOT NULL,
  polarity TEXT NOT NULL CHECK(polarity IN ('+','-')),
  reason TEXT, authority TEXT, created_at INTEGER, active INTEGER NOT NULL DEFAULT 1);

-- ── CLAIMS & CALIBRATION ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS claims(
  id TEXT PRIMARY KEY, stmt_json TEXT NOT NULL,
  frame TEXT NOT NULL CHECK(frame IN ('internal','external')),
  settlement TEXT NOT NULL CHECK(settlement IN
    ('entailed','tool_decidable','observation_pending','unsettleable')),
  verifier_id TEXT, due_ts INTEGER,
  predicted_p REAL, predicted_ci_lo REAL, predicted_ci_hi REAL,
  model_snapshot TEXT NOT NULL, predictor_class TEXT NOT NULL,
  certainty_class TEXT CHECK(certainty_class IN
    ('certain','high','estimated','unlicensed')),
  resolved_ts INTEGER, outcome INTEGER, surprisal REAL,
  CHECK (settlement = 'unsettleable' OR verifier_id IS NOT NULL));
CREATE TABLE IF NOT EXISTS proof_steps(
  claim_id TEXT, step_no INTEGER, rule_id TEXT, fact_id TEXT, edge_kind TEXT,
  sensitivity REAL, PRIMARY KEY(claim_id, step_no));
CREATE TABLE IF NOT EXISTS oracles(
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL CHECK(kind IN
    ('deterministic_total','deterministic_partial','stochastic')),
  impl_ref TEXT, code_hash TEXT, env_hash TEXT,
  n_trials INTEGER NOT NULL DEFAULT 0, n_correct INTEGER NOT NULL DEFAULT 0,
  validated_at INTEGER);
-- calibration keeps INTEGER tallies; mean_p / observed_freq are read-time ratios.
CREATE TABLE IF NOT EXISTS calibration(
  frame TEXT, settlement TEXT, predictor_class TEXT, bucket INTEGER,
  n INTEGER NOT NULL, k INTEGER NOT NULL, p_milli INTEGER NOT NULL,
  updated_at INTEGER, PRIMARY KEY(frame, settlement, predictor_class, bucket));

-- ── LEARNING & CURIOSITY ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS observations(
  event_seq INTEGER PRIMARY KEY, fact_id TEXT, actor TEXT, outcome INTEGER,
  grade INTEGER NOT NULL DEFAULT 0,   -- v0.4 Δ6: 0=ungraded, 1..3 by confidence
  channel TEXT, context_sig TEXT, ts INTEGER);
CREATE TABLE IF NOT EXISTS obs_context(
  event_seq INTEGER NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
  PRIMARY KEY(event_seq, key));
CREATE TABLE IF NOT EXISTS soft_edges(
  from_sym TEXT, to_sym TEXT, sim REAL, basis TEXT, computed_at INTEGER,
  PRIMARY KEY(from_sym, to_sym));
CREATE TABLE IF NOT EXISTS open_questions(
  id TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('dispersion','pin_tension')),
  target_kind TEXT, target_id TEXT, residual_partition TEXT, dispersion_stat REAL,
  ruled_out_json TEXT, suggested_measurement TEXT,
  status TEXT NOT NULL CHECK(status IN ('open','explained','abandoned')),
  explained_by_guard_id TEXT);
CREATE TABLE IF NOT EXISTS closure_atoms(
  atom TEXT PRIMARY KEY, basis TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS diagnostics(
  seq INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, kind TEXT, detail_json TEXT);
CREATE TABLE IF NOT EXISTS invariants(
  id TEXT PRIMARY KEY, family TEXT, target_scope TEXT,
  fail_policy TEXT CHECK(fail_policy IN ('fail_stop','alert_only')),
  last_run INTEGER, status TEXT, failure_ref TEXT);
CREATE TABLE IF NOT EXISTS eval_queue(
  target_kind TEXT, target_id TEXT, dependents INTEGER, sensitivity REAL,
  cost REAL, score REAL, PRIMARY KEY(target_kind, target_id));

CREATE INDEX IF NOT EXISTS ix_fc_fact ON fact_counts(fact_id);
CREATE INDEX IF NOT EXISTS ix_obs_fact ON observations(fact_id);
CREATE INDEX IF NOT EXISTS ix_cand_status ON candidates(status);
CREATE INDEX IF NOT EXISTS ix_pins_target ON pins(target_id);
"""

# Columns that hold counts. The integrality scan (I11) walks exactly these.
COUNT_COLUMNS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("fact_counts", ("n", "k"), ("fact_id", "actor", "channel")),
    ("rule_counts", ("n", "k"), ("rule_id", "actor")),
    ("actor_confusion", ("tp", "fn", "fp", "tn"), ("actor", "frame")),
    ("actor_response", ("n_true", "n_false"), ("actor", "frame", "vote", "grade")),
    ("calibration", ("n", "k", "p_milli"), ("frame", "settlement", "predictor_class", "bucket")),
    ("oracles", ("n_trials", "n_correct"), ("id",)),
    ("quota_usage", ("used",), ("actor", "epoch", "kind")),
)


class Index:
    """Thin typed wrapper over the derived SQLite database."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.db: Optional[sqlite3.Connection] = None

    # ── lifecycle ────────────────────────────────────────────────────────────
    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self) -> None:
        if self.db is not None:
            self.db.commit()
            self.db.close()
            self.db = None

    def drop(self) -> None:
        """Delete the index entirely — it is a view, not an artifact (I1)."""
        self.close()
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(self.path) + suffix)
            if p.exists():
                p.unlink()

    def reset(self) -> None:
        self.drop()
        self.open()

    # ── helpers ──────────────────────────────────────────────────────────────
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        assert self.db is not None, "index not open"
        return self.db.execute(sql, tuple(params))

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return list(self.execute(sql, params).fetchall())

    def one(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        rows = self.execute(sql, params).fetchmany(1)
        return rows[0] if rows else None

    def commit(self) -> None:
        if self.db is not None:
            self.db.commit()

    # ── integrality scan (I11, §6.2 count integrality) ───────────────────────
    def nonintegral_counts(self) -> list[str]:
        offenders: list[str] = []
        for table, cols, keys in COUNT_COLUMNS:
            key_expr = " || ':' || ".join(f"CAST({k} AS TEXT)" for k in keys)
            for col in cols:
                sql = (f"SELECT {key_expr} AS rid, typeof({col}) AS t FROM {table} "
                       f"WHERE typeof({col}) <> 'integer'")
                for row in self.query(sql):
                    offenders.append(f"{table}.{col}[{row['rid']}]:{row['t']}")
        return offenders
