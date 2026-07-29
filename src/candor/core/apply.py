"""Replayer: fold ledger events into the derived index (trusted, spec §3.1, I1/I3).

Every mutating API path goes through exactly this function, so the live index is
always replay-equivalent by construction rather than by hope. `replay()` drops
the index and re-folds the segments; the closure hash must come out identical.

Redaction implies exclusion (§3.1): events whose payload has been redacted are
skipped entirely, so downstream state is recomputed *without* the content.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Iterable, Optional

from . import closure as closure_mod
from . import constraints as constraints_mod
from .canonical import context_signature, fact_key
from .committed import counts as counts_mod
from .committed import facts as facts_mod
from .committed import reliability as reliability_mod
from .hashing import canon_json, sha256_hex

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .index import Index
    from .ledger import Event, Ledger

DEFAULT_OBS_QUOTA = 3000
DEFAULT_CAND_QUOTA = 500

# §3.10/§4.4: how many observations must contradict a '-' pin before the
# surprisal router opens a pin_tension question and pages a human.
PIN_TENSION_MIN_CONTRADICTIONS = 3

ACTOR_CLASSES = {"human": "human", "tool": "tool", "agent": "agent",
                 "verifier": "verifier"}


def actor_class(name: str) -> str:
    return ACTOR_CLASSES.get(name.split(":", 1)[0], "agent")


def ensure_actor(idx: "Index", name: str) -> None:
    idx.execute(
        "INSERT OR IGNORE INTO actors(name, class, obs_quota_per_epoch, "
        "cand_quota_per_epoch) VALUES(?,?,?,?)",
        (name, actor_class(name), DEFAULT_OBS_QUOTA, DEFAULT_CAND_QUOTA))


def diagnostic(idx: "Index", ts: int, kind: str, detail: dict[str, Any]) -> None:
    idx.execute("INSERT INTO diagnostics(ts, kind, detail_json) VALUES(?,?,?)",
                (ts, kind, canon_json(detail)))


# ── the fold ────────────────────────────────────────────────────────────────

def apply_event(idx: "Index", ev: "Event", payload: Optional[dict[str, Any]]) -> None:
    """Fold one ledger event into the derived index. Pure function of (state, event)."""
    idx.execute(
        "INSERT OR REPLACE INTO events(seq, ts, kind, actor, payload_hash, "
        "source_ref, context_sig, prev_hash, hash) VALUES(?,?,?,?,?,?,?,?,?)",
        (ev.seq, ev.ts, ev.kind, ev.actor, ev.payload_hash, ev.source_ref,
         ev.context_sig, ev.prev_hash, ev.hash))
    ensure_actor(idx, ev.actor)
    if payload is None:
        # Payload redacted: the skeleton stays forever, the content does not.
        return
    handler = _HANDLERS.get(ev.kind)
    if handler is not None:
        handler(idx, ev, payload)


def candidate_id_for(seq: int) -> str:
    return f"cand:{seq}"


def gate_run_id_for(seq: int) -> str:
    return f"gate:{seq}"


def _apply_assertion(idx: "Index", ev: "Event", payload: dict[str, Any]) -> None:
    idx.execute(
        "INSERT OR REPLACE INTO candidates(id, event_seq, kind, body_json, span_ref, "
        "proposer, status, gate_run_id, failing_step, reason) "
        "VALUES(?,?,?,?,?,?,'pending',NULL,NULL,NULL)",
        (candidate_id_for(ev.seq), ev.seq, payload["candidate_kind"],
         canon_json(payload["body"]), ev.source_ref, ev.actor))
    _bump_quota(idx, ev.actor, "candidate")


def _apply_admission(idx: "Index", ev: "Event", payload: dict[str, Any]) -> None:
    cid = payload["candidate_id"]
    gate_run_id = payload.get("gate_run_id") or gate_run_id_for(ev.seq)
    idx.execute(
        "UPDATE candidates SET status=?, gate_run_id=?, failing_step=?, reason=? "
        "WHERE id=?",
        (payload["status"], gate_run_id, payload.get("failing_step"),
         payload.get("reason"), cid))
    if payload["status"] != "admitted":
        diagnostic(idx, ev.ts, "gate_rejection", {
            "candidate_id": cid, "candidate_kind": payload["candidate_kind"],
            "failing_step": payload.get("failing_step"),
            "reason": payload.get("reason")})
        return

    kind = payload["candidate_kind"]
    body = payload["body"]
    if kind == "symbol":
        idx.execute(
            "INSERT OR REPLACE INTO predicates(pred, arity, arg_types_json, "
            "canonical_units_json, admitted_at, admitted_by_event) "
            "VALUES(?,?,?,?,?,?)",
            (body["pred"], body["arity"], canon_json(body["arg_types"]),
             canon_json(body["canonical_units"]), ev.ts, ev.seq))
    elif kind == "fact":
        fid = body["fact_id"]
        prior = idx.one("SELECT dispersion_flag, breadth_class FROM facts WHERE id=?",
                        (fid,))
        idx.execute(
            "INSERT OR REPLACE INTO facts(id, pred, args_json, stmt_type, kind, sim, "
            "structural, numeric, breadth_class, dispersion_flag, valid_from, "
            "valid_to, admitted_at, admitted_by_event) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (fid, body["pred"], canon_json(body["args"]), body["stmt_type"],
             body["kind"], body.get("sim"), body["structural"], body["numeric"],
             prior["breadth_class"] if prior else None,
             int(prior["dispersion_flag"]) if prior else 0,
             body.get("valid_from", ev.ts), body.get("valid_to"), ev.ts, ev.seq))
        # The admitted fact's audit trail is addressable from admission onward,
        # keyed by its proposer, at zero (I5: unobserved is ε, never a hard 0).
        # A categorical fact has an OPEN vocabulary with no value known at
        # admission, so there is no single baseline row to seed — its per-value
        # tallies (fact_category_counts) become addressable on the first
        # observation. So skip channel_for (which knows only epi/alea) for it.
        if body["stmt_type"] != "categorical":
            proposer = idx.one("SELECT proposer FROM candidates WHERE id=?", (cid,))
            counts_mod.ensure_row(
                idx, fid, proposer["proposer"] if proposer else ev.actor,
                counts_mod.channel_for(body["stmt_type"]))
    elif kind == "constraint":
        idx.execute(
            "INSERT OR REPLACE INTO constraints(id, kind, body_json, structural, "
            "admitted_at, gate_run_id) VALUES(?,?,?,'admitted',?,?)",
            (f"con:{ev.seq}", body["ctype"], canon_json(body["body"]), ev.ts,
             gate_run_id))
    elif kind in ("rule", "guard"):
        idx.execute(
            "INSERT OR REPLACE INTO rules(id, head_json, body_json, specificity, "
            "parent_rule_id, structural, numeric, gate_run_id, admitted_at) "
            "VALUES(?,?,?,?,?,'admitted','accumulating',?,?)",
            (f"rule:{ev.seq}", canon_json(body["head"]),
             canon_json(body.get("body", {})), int(body.get("specificity", 0)),
             body.get("parent_rule_id"), gate_run_id, ev.ts))
    elif kind == "verifier":
        idx.execute(
            "INSERT OR REPLACE INTO oracles(id, kind, impl_ref, code_hash, env_hash, "
            "n_trials, n_correct, validated_at) VALUES(?,?,?,?,?,0,0,?)",
            (body["oracle_id"], body["kind"], body["entry"], body["code_hash"],
             body["env_hash"], ev.ts))
    elif kind == "supersede_valid_time":
        idx.execute("UPDATE facts SET valid_to=? WHERE id=?",
                    (body.get("valid_to", ev.ts), body["fact_id"]))
    # alias admissions additionally emit a dedicated `alias` event (see §3.13)


def _apply_alias(idx: "Index", ev: "Event", payload: dict[str, Any]) -> None:
    idx.execute(
        "INSERT OR REPLACE INTO aliases(canonical, alias, basis, admitted_at, "
        "admitted_by_event) VALUES(?,?,?,?,?)",
        (payload["canonical"], payload["alias"], payload["basis"], ev.ts, ev.seq))


def _apply_observation(idx: "Index", ev: "Event", payload: dict[str, Any]) -> None:
    stmt = payload["stmt"]
    ctx = payload.get("ctx") or {}
    value = payload.get("value")
    fid = facts_mod.lookup(idx, stmt["pred"], stmt["args"])
    # Which field is authoritative is decided HERE, at fold time, by the fact's
    # stmt_type — not at the API (§1.4). An observation that arrives before its
    # fact is admitted has fid=None and simply attributes nothing, exactly as the
    # crisp/frequency path already does.
    stmt_type = None
    if fid is not None:
        row = idx.one("SELECT stmt_type FROM facts WHERE id=?", (fid,))
        stmt_type = row["stmt_type"] if row else None
    categorical = stmt_type == "categorical"
    channel: Optional[str] = None
    outcome_col: Optional[int] = None
    outcome = bool(payload.get("outcome"))
    if categorical:
        # Categorical: the realised value moves the per-value tallies; outcome is
        # ignored and stored NULL. channel='cat' partitions these rows off the
        # epi/alea observation stream.
        channel = "cat"
        counts_mod.apply_category_observation(idx, fid, ev.actor, value)
    else:
        outcome_col = 1 if outcome else 0
        if fid is not None:
            channel = counts_mod.channel_for(stmt_type)
            counts_mod.apply_observation(idx, fid, ev.actor, channel, outcome)
    idx.execute(
        "INSERT OR REPLACE INTO observations(event_seq, fact_id, actor, outcome, "
        "grade, channel, context_sig, value, ts) VALUES(?,?,?,?,?,?,?,?,?)",
        (ev.seq, fid, ev.actor, outcome_col,
         int(payload.get("grade", 0)), channel, ev.context_sig, value, ev.ts))
    for key, cval in sorted(ctx.items()):
        idx.execute(
            "INSERT OR REPLACE INTO obs_context(event_seq, key, value) VALUES(?,?,?)",
            (ev.seq, str(key), str(cval)))
    _bump_quota(idx, ev.actor, "observation")
    # Pin tension is a boolean-outcome semantics; a categorical observation has no
    # true/false outcome to contradict a '-' pin, so it is skipped for those.
    if fid is not None and not categorical:
        _check_pin_tension(idx, ev, fid, outcome)


def _apply_pin(idx: "Index", ev: "Event", payload: dict[str, Any]) -> None:
    idx.execute(
        "INSERT OR REPLACE INTO pins(id, target_kind, target_id, polarity, reason, "
        "authority, created_at, active) VALUES(?,?,?,?,?,?,?,1)",
        (f"pin:{ev.seq}", payload.get("target_kind", "fact"), payload["target_id"],
         payload["polarity"], payload.get("reason"), payload.get("authority"), ev.ts))


def _apply_supersede(idx: "Index", ev: "Event", payload: dict[str, Any]) -> None:
    kind = payload.get("target_kind")
    target = payload["target_id"]
    if kind == "alias_event":
        prior = payload.get("alias")
        if prior:
            idx.execute("DELETE FROM aliases WHERE canonical=? AND alias=?",
                        (prior["canonical"], prior["alias"]))
    elif kind == "pin":
        idx.execute("UPDATE pins SET active=0 WHERE id=?", (target,))
    elif kind == "fact":
        idx.execute("UPDATE facts SET valid_to=? WHERE id=? AND valid_to IS NULL",
                    (ev.ts, target))
    elif kind == "rule":
        idx.execute("UPDATE rules SET structural='candidate' WHERE id=?", (target,))
    elif kind == "constraint":
        idx.execute("DELETE FROM constraints WHERE id=?", (target,))
    idx.execute("UPDATE candidates SET status='superseded' WHERE id=?", (target,))


def _apply_demotion(idx: "Index", ev: "Event", payload: dict[str, Any]) -> None:
    target = payload["target_id"]
    if payload.get("target_kind") == "rule":
        idx.execute("UPDATE rules SET structural='candidate' WHERE id=?", (target,))
    else:
        idx.execute("UPDATE facts SET structural='candidate' WHERE id=?", (target,))


def _apply_claim(idx: "Index", ev: "Event", payload: dict[str, Any]) -> None:
    # A categorical claim freezes its full predicted DISTRIBUTION (design §4.1),
    # not a scalar predicted_p; it is snapshot-pinned here so resolution can score
    # the realised value's surprisal against it. NULL for crisp/frequency.
    dist = payload.get("predicted_dist")
    idx.execute(
        "INSERT OR REPLACE INTO claims(id, stmt_json, frame, settlement, verifier_id, "
        "due_ts, predicted_p, predicted_ci_lo, predicted_ci_hi, model_snapshot, "
        "predictor_class, certainty_class, resolved_ts, outcome, surprisal, "
        "predicted_dist_json) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,NULL,?)",
        (payload["claim_id"], canon_json(payload["stmt"]), payload["frame"],
         payload["settlement"], payload.get("verifier_id"), payload.get("due"),
         payload.get("predicted_p"), payload.get("ci_lo"), payload.get("ci_hi"),
         payload["model_snapshot"], payload["predictor_class"],
         payload.get("certainty_class"),
         canon_json(dist) if dist is not None else None))
    for i, (comp, sens) in enumerate(sorted((payload.get("sensitivity") or {}).items())):
        idx.execute(
            "INSERT OR REPLACE INTO proof_steps(claim_id, step_no, rule_id, fact_id, "
            "edge_kind, sensitivity) VALUES(?,?,?,?,?,?)",
            (payload["claim_id"], i, comp if comp.startswith("rule:") else None,
             comp if comp.startswith("fact:") else None, "exact", float(sens)))


def _apply_resolution(idx: "Index", ev: "Event", payload: dict[str, Any]) -> None:
    from . import calibration as calibration_mod

    claim_id = payload["claim_id"]
    outcome = bool(payload["outcome"])
    row = idx.one("SELECT * FROM claims WHERE id=?", (claim_id,))
    idx.execute(
        "UPDATE claims SET resolved_ts=?, outcome=?, surprisal=? WHERE id=?",
        (ev.ts, 1 if outcome else 0, payload.get("surprisal"), claim_id))
    if row is None:
        return
    # ── categorical settlement (design §4): the verifier reported a realised
    # value v*, and the payload carries the multiclass surprisal −log P(v*) plus
    # P(v*) itself (both frozen at resolve time against the claim's snapshot). It
    # calibrates under the DISTINCT categorical predictor_class (I9 — never pools
    # with binary) and scores per-source trust via one-vs-rest, not the binary
    # two-coin scorer. No credit-assignment blame: a categorical fact is a leaf
    # query, not a proof literal (decision 3).
    if payload.get("realized_value") is not None:
        calibration_mod.record(idx, row["frame"], row["settlement"],
                               row["predictor_class"],
                               float(payload.get("predicted_p") or 0.0),
                               True, ev.ts)
        if payload.get("oracle_kind") == "deterministic_total":
            stmt = json.loads(row["stmt_json"])
            fid = facts_mod.lookup(idx, stmt["pred"], stmt["args"])
            if fid is not None:
                reliability_mod.score_category_against_settlement(
                    idx, fid, payload["realized_value"], row["frame"])
        return
    calibration_mod.record(idx, row["frame"], row["settlement"],
                           row["predictor_class"], float(row["predicted_p"] or 0.5),
                           outcome, ev.ts)
    # §3.12: reliability moves ONLY here, and only for a deterministic_total oracle.
    if payload.get("oracle_kind") == "deterministic_total":
        stmt = json.loads(row["stmt_json"])
        fid = facts_mod.lookup(idx, stmt["pred"], stmt["args"])
        if fid is not None:
            reliability_mod.score_against_settlement(idx, fid, outcome, row["frame"])
    # §4.4 credit assignment: integer blame to the argmax-sensitivity component only.
    blame = payload.get("blame_target")
    if blame:
        if blame.startswith("rule:"):
            counts_mod.apply_rule_observation(idx, blame, ev.actor, outcome)
        else:
            frow = idx.one("SELECT stmt_type FROM facts WHERE id=?", (blame,))
            if frow is not None:
                counts_mod.apply_observation(
                    idx, blame, ev.actor,
                    counts_mod.channel_for(frow["stmt_type"]), outcome)


def _apply_redaction(idx: "Index", ev: "Event", payload: dict[str, Any]) -> None:
    idx.execute("INSERT OR REPLACE INTO redactions(payload_hash, event_seq) "
                "VALUES(?,?)", (payload["payload_hash"], ev.seq))


def _apply_retraction(idx: "Index", ev: "Event", payload: dict[str, Any]) -> None:
    """Record that a source was silenced (or restored). The *exclusion* itself
    happens at fold time in `CandorSystem._refold`; this is the audit trail."""
    idx.execute(
        "INSERT OR REPLACE INTO retractions(actor, event_seq, reason, restored) "
        "VALUES(?,?,?,?)",
        (payload["actor"], ev.seq, payload.get("reason"),
         1 if payload.get("restore") else 0))


def _apply_checkpoint(idx: "Index", ev: "Event", payload: dict[str, Any]) -> None:
    idx.execute("INSERT OR REPLACE INTO diagnostics(ts, kind, detail_json) "
                "VALUES(?,?,?)", (ev.ts, "checkpoint", canon_json(payload)))


def _apply_reliability(idx: "Index", ev: "Event", payload: dict[str, Any]) -> None:
    """Operator trust override, folded in ledger order (I1/I3).

    Writes the actor_reliability Beta — consumed by the alea (frequency) discount
    via `reliability.expected` — and records the override in a dedicated table so
    the crisp-vote path (`_actor_discounts`) tempers on operator levers ONLY.
    Learned settlement reliability moves actor_reliability too, but it already
    speaks through the two-coin LR; counting it on the crisp path as well would
    double-count the same settlements, so it must never enter this table.
    """
    actor = payload["actor"]
    frame = payload["frame"]
    a, b = float(payload["rel_a"]), float(payload["rel_b"])
    reliability_mod.set_reliability(idx, actor, frame, a, b)
    idx.execute(
        "INSERT INTO reliability_overrides(actor, frame, rel_a, rel_b) "
        "VALUES(?,?,?,?) ON CONFLICT(actor, frame) DO UPDATE SET "
        "rel_a=excluded.rel_a, rel_b=excluded.rel_b", (actor, frame, a, b))


_HANDLERS = {
    "assertion": _apply_assertion,
    "admission": _apply_admission,
    "alias": _apply_alias,
    "observation": _apply_observation,
    "pin": _apply_pin,
    "supersede": _apply_supersede,
    "demotion": _apply_demotion,
    "claim": _apply_claim,
    "resolution": _apply_resolution,
    "redaction": _apply_redaction,
    "retraction": _apply_retraction,
    "checkpoint": _apply_checkpoint,
    "reliability": _apply_reliability,
}


# ── derived bookkeeping ─────────────────────────────────────────────────────

def _bump_quota(idx: "Index", actor: str, kind: str, epoch: int = 0) -> int:
    idx.execute(
        "INSERT OR IGNORE INTO quota_usage(actor, epoch, kind, used) VALUES(?,?,?,0)",
        (actor, epoch, kind))
    idx.execute(
        "UPDATE quota_usage SET used = used + 1 WHERE actor=? AND epoch=? AND kind=?",
        (actor, epoch, kind))
    row = idx.one(
        "SELECT used FROM quota_usage WHERE actor=? AND epoch=? AND kind=?",
        (actor, epoch, kind))
    return int(row["used"])


def quota_used(idx: "Index", actor: str, kind: str, epoch: int = 0) -> int:
    row = idx.one(
        "SELECT used FROM quota_usage WHERE actor=? AND epoch=? AND kind=?",
        (actor, epoch, kind))
    return int(row["used"]) if row else 0


def quota_limit(idx: "Index", actor: str, kind: str) -> int:
    ensure_actor(idx, actor)
    row = idx.one(
        "SELECT obs_quota_per_epoch, cand_quota_per_epoch FROM actors WHERE name=?",
        (actor,))
    return int(row["obs_quota_per_epoch"] if kind == "observation"
               else row["cand_quota_per_epoch"])


def _check_pin_tension(idx: "Index", ev: "Event", fid: str, outcome: bool) -> None:
    """§3.10: the pin still wins; past threshold a human gets paged."""
    if facts_mod.pin_polarity(idx, fid) != "-" or not outcome:
        return
    row = idx.one(
        "SELECT COUNT(*) AS c FROM observations WHERE fact_id=? AND outcome=1",
        (fid,))
    contradictions = int(row["c"]) if row else 0
    if contradictions < PIN_TENSION_MIN_CONTRADICTIONS:
        return
    qid = f"q:pin_tension:{fid}"
    partition = canon_json({"contradicting_observations": contradictions})
    idx.execute(
        "INSERT OR REPLACE INTO open_questions(id, kind, target_kind, target_id, "
        "residual_partition, dispersion_stat, ruled_out_json, suggested_measurement, "
        "status, explained_by_guard_id) "
        "VALUES(?,'pin_tension','fact',?,?,?,'[]',?,'open',NULL)",
        (qid, fid, partition, float(contradictions),
         "re-examine the '-' pin authority, or measure the disputed condition "
         "under the contexts the contradicting observations share"))


# ── closure materialization & hashing ───────────────────────────────────────

def rebuild_closure(idx: "Index") -> closure_mod.Closure:
    """Materialize the exact closure. '-' pins are the only hard zero (I5)."""
    base: list[tuple[str, list[Any], str]] = []
    for row in idx.query(
            "SELECT id, pred, args_json FROM facts WHERE structural <> 'candidate' "
            "AND valid_to IS NULL ORDER BY id"):
        if facts_mod.is_negatively_pinned(idx, row["id"]):
            continue
        base.append((row["pred"], json.loads(row["args_json"]), row["id"]))
    rules = [closure_mod.Rule.parse(r["id"], r["head_json"], r["body_json"],
                                    int(r["specificity"]))
             for r in idx.query("SELECT * FROM rules WHERE structural='admitted' "
                                "ORDER BY specificity DESC, id")]
    clo = closure_mod.materialize(base, rules)
    cons = [constraints_mod.parse(r) for r in idx.query("SELECT * FROM constraints")]
    if cons:
        # Constraint enforcement during materialization (§3.5): an atom that
        # collides with a *certain* atom under an admitted constraint is dropped.
        # Collisions between two merely-admitted atoms survive here and are
        # priced at prediction time as `rejection_rate` (§3.9, DEVIATIONS D3).
        certain: set[tuple[str, tuple[Any, ...]]] = set()
        for row in idx.query("SELECT pred, args_json FROM facts "
                             "WHERE kind='definitional' OR structural='pinned'"):
            certain.add((row["pred"], tuple(json.loads(row["args_json"]))))
        certain_keys = {c.group_key(p, list(a))
                        for c in cons for p, a in certain} - {None}
        if certain_keys:
            for atom in sorted(clo.atoms - certain,
                               key=lambda a: closure_mod.atom_text(*a)):
                if any(c.group_key(atom[0], list(atom[1])) in certain_keys
                       for c in cons):
                    clo.atoms.discard(atom)
    idx.execute("DELETE FROM closure_atoms")
    for pred, args in sorted(clo.atoms, key=lambda a: closure_mod.atom_text(*a)):
        idx.execute("INSERT OR REPLACE INTO closure_atoms(atom, basis) VALUES(?,?)",
                    (closure_mod.atom_text(pred, args), clo.basis.get((pred, args), "exact")))
    return clo


_HASH_QUERIES: tuple[tuple[str, str], ...] = (
    ("predicates", "SELECT pred, arity, arg_types_json, canonical_units_json, "
                   "admitted_at FROM predicates ORDER BY pred"),
    ("aliases", "SELECT canonical, alias, basis, admitted_at FROM aliases "
                "ORDER BY canonical, alias"),
    ("facts", "SELECT id, pred, args_json, stmt_type, kind, sim, structural, numeric, "
              "breadth_class, dispersion_flag, valid_from, valid_to, admitted_at "
              "FROM facts ORDER BY id"),
    ("fact_counts", "SELECT fact_id, actor, channel, n, k FROM fact_counts "
                    "ORDER BY fact_id, actor, channel"),
    ("fact_category_counts", "SELECT fact_id, actor, value, n FROM "
                             "fact_category_counts ORDER BY fact_id, actor, value"),
    ("rules", "SELECT id, head_json, body_json, specificity, structural, numeric, "
              "admitted_at FROM rules ORDER BY id"),
    ("rule_counts", "SELECT rule_id, actor, n, k FROM rule_counts ORDER BY rule_id, actor"),
    ("constraints", "SELECT id, kind, body_json, structural, admitted_at FROM constraints "
                    "ORDER BY id"),
    ("pins", "SELECT id, target_kind, target_id, polarity, authority, created_at, active "
             "FROM pins ORDER BY id"),
    ("candidates", "SELECT id, kind, body_json, proposer, status, failing_step "
                   "FROM candidates ORDER BY id"),
    ("oracles", "SELECT id, kind, impl_ref, code_hash, env_hash, n_trials, n_correct "
                "FROM oracles ORDER BY id"),
    ("claims", "SELECT id, stmt_json, frame, settlement, predicted_p, model_snapshot, "
               "resolved_ts, outcome, surprisal, predicted_dist_json "
               "FROM claims ORDER BY id"),
    ("calibration", "SELECT frame, settlement, predictor_class, bucket, n, k, p_milli "
                    "FROM calibration ORDER BY frame, settlement, predictor_class, bucket"),
    ("open_questions", "SELECT id, kind, target_kind, target_id, residual_partition, "
                       "dispersion_stat, status FROM open_questions ORDER BY id"),
    ("closure_atoms", "SELECT atom, basis FROM closure_atoms ORDER BY atom"),
    ("redactions", "SELECT payload_hash FROM redactions ORDER BY payload_hash"),
    ("actor_reliability", "SELECT actor, frame, rel_a, rel_b FROM actor_reliability "
                          "ORDER BY actor, frame"),
    ("reliability_overrides", "SELECT actor, frame, rel_a, rel_b "
                              "FROM reliability_overrides ORDER BY actor, frame"),
    ("actor_confusion", "SELECT actor, frame, tp, fn, fp, tn FROM actor_confusion "
                        "ORDER BY actor, frame"),
    ("actor_response", "SELECT actor, frame, vote, grade, n_true, n_false "
                       "FROM actor_response ORDER BY actor, frame, vote, grade"),
)


def closure_hash(idx: "Index") -> str:
    """Bit-for-bit fingerprint of every derived structure (§6.2 replay determinism)."""
    parts: list[Any] = []
    for name, sql in _HASH_QUERIES:
        parts.append([name, [[*row] for row in idx.query(sql)]])
    return sha256_hex(canon_json(parts))
