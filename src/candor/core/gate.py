"""The gate — trusted harness over untrusted content (spec §3.4).

Single serialization point for all structural change. Seven steps; admission
requires all seven. Rejections are recorded, not discarded: they are training
signal and they prevent re-proposal churn.

The gate *decides*; it does not write. The decision is appended to the ledger
as an `admission` event and applied by the replayer, so a rebuild from the log
reproduces the same committed tier without re-running any judgement.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from . import constraints as constraints_mod
from . import sandbox
from .canonical import CanonicalizationError, canonicalize_args, fact_key
from .committed import facts as facts_mod

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .index import Index

STEP_NAMES = {
    1: "syntactic/AST validation against the predicate registry",
    2: "canonicalization (units, argument normal forms)",
    3: "sandboxed execution of synthesized verifiers/tools",
    4: "pinned regression cases",
    5: "held-out evidence check",
    6: "MDL improvement for guards",
    7: "contradiction check against admitted constraints",
}

# §3.4: demotion runs the same path backward with a strictly higher bar.
DEMOTION_HYSTERESIS = 3.0
# §3.4 alias admission: behavioral-signature similarity floor.
ALIAS_SIM_THRESHOLD = 0.85
# §4.5 guard gate: minimum support per partition.
GUARD_MIN_SUPPORT = 8


@dataclass
class Decision:
    candidate_id: str
    candidate_kind: str
    status: str                       # 'admitted' | 'rejected'
    failing_step: Optional[int] = None
    reason: Optional[str] = None
    body: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id, "candidate_kind": self.candidate_kind,
            "status": self.status, "failing_step": self.failing_step,
            "reason": self.reason, "body": self.body,
        }


def _reject(cid: str, kind: str, step: int, reason: str) -> Decision:
    return Decision(cid, kind, "rejected", step, f"step {step}: {reason}")


def evaluate(idx: "Index", cid: str, kind: str, body: dict[str, Any],
             proposer: str) -> Decision:
    """Run all seven steps for one candidate."""
    if kind == "symbol":
        return _evaluate_symbol(idx, cid, body)
    if kind == "fact":
        return _evaluate_fact(idx, cid, body)
    if kind == "alias":
        return _evaluate_alias(idx, cid, body)
    if kind == "constraint":
        return _evaluate_constraint(idx, cid, body)
    if kind == "verifier":
        return _evaluate_verifier(idx, cid, body)
    if kind in ("rule", "guard"):
        return _evaluate_rule(idx, cid, kind, body)
    if kind == "supersede_valid_time":
        return _evaluate_supersede(idx, cid, body)
    return _reject(cid, kind, 1, f"unknown candidate kind {kind!r}")


# ── per-kind evaluation ─────────────────────────────────────────────────────

def _evaluate_symbol(idx: "Index", cid: str, body: dict[str, Any]) -> Decision:
    pred = body.get("pred")
    if not isinstance(pred, str) or not pred:
        return _reject(cid, "symbol", 1, "symbol candidate has no predicate name")
    arity = body.get("arity")
    if not isinstance(arity, int) or arity < 0:
        return _reject(cid, "symbol", 1, "arity must be a non-negative integer")
    arg_types = body.get("arg_types") or ["any"] * arity
    if len(arg_types) != arity:
        return _reject(cid, "symbol", 1, "arg_types length does not match arity")
    units = body.get("canonical_units") or {}
    existing = facts_mod.predicate(idx, pred)
    if existing and existing["arity"] != arity:
        return _reject(cid, "symbol", 7,
                       f"predicate {pred!r} already registered with arity "
                       f"{existing['arity']}")
    if facts_mod.pin_polarity(idx, f"pred:{pred}") == "-":
        return _reject(cid, "symbol", 4, f"predicate {pred!r} is negatively pinned")
    return Decision(cid, "symbol", "admitted", body={
        "pred": pred, "arity": arity, "arg_types": arg_types,
        "canonical_units": units})


def _evaluate_fact(idx: "Index", cid: str, body: dict[str, Any]) -> Decision:
    pred = body.get("pred")
    args = body.get("args")
    if not isinstance(pred, str) or not isinstance(args, list):
        return _reject(cid, "fact", 1, "fact candidate needs `pred` and `args`")
    reg = facts_mod.predicate(idx, pred)
    if reg is None:
        return _reject(cid, "fact", 1,
                       f"predicate {pred!r} is not in the registry (§3.3)")
    if reg["arity"] != len(args):
        return _reject(cid, "fact", 1,
                       f"arity mismatch: registry says {reg['arity']}, "
                       f"candidate has {len(args)}")
    try:
        cargs = canonicalize_args(args, reg["canonical_units"])
    except CanonicalizationError as exc:
        return _reject(cid, "fact", 2, str(exc))

    # M5: a non-finite float (NaN/Infinity) has no canonical numeric normal form
    # and cannot be serialized into a portable payload. Reject it here so it can
    # never reach the committed tier (and never raises later inside fact_key).
    for a in cargs:
        if isinstance(a, float) and not math.isfinite(a):
            return _reject(cid, "fact", 2,
                           "argument is a non-finite float (NaN/Infinity) with "
                           "no canonical numeric normal form")

    stmt_type = body.get("stmt_type", "crisp")
    if stmt_type not in ("crisp", "frequency"):
        return _reject(cid, "fact", 1, f"unknown stmt_type {stmt_type!r}")

    fid = fact_key(pred, cargs)
    if facts_mod.pin_polarity(idx, fid) == "-":
        return _reject(cid, "fact", 4, "a '-' pin vetoes this fact")

    conflict = _certain_contradiction(idx, pred, cargs, fid)
    if conflict:
        return _reject(cid, "fact", 7, conflict)

    return Decision(cid, "fact", "admitted", body={
        "fact_id": fid, "pred": pred, "args": cargs, "stmt_type": stmt_type,
        "kind": body.get("kind", "exact"),
        "structural": body.get("structural", "admitted"),
        "numeric": body.get("numeric", "accumulating"),
        "sim": body.get("sim"),
        "valid_from": body.get("valid_from"), "valid_to": body.get("valid_to"),
    })


def _evaluate_alias(idx: "Index", cid: str, body: dict[str, Any]) -> Decision:
    canonical, alias = body.get("canonical"), body.get("alias")
    basis = body.get("basis")
    if not canonical or not alias:
        return _reject(cid, "alias", 1, "alias candidate needs canonical and alias")
    if basis not in ("behavioral", "definitional", "pinned"):
        return _reject(cid, "alias", 1, f"unknown alias basis {basis!r}")
    if basis == "behavioral":
        sim = float(body.get("sim") or 0.0)
        if sim < ALIAS_SIM_THRESHOLD:
            return _reject(cid, "alias", 5,
                           f"behavioral similarity {sim:.3f} below "
                           f"{ALIAS_SIM_THRESHOLD}")
    conflict = _alias_merge_conflict(idx, canonical, alias)
    if conflict:
        return _reject(cid, "alias", 7, conflict)
    return Decision(cid, "alias", "admitted", body={
        "canonical": canonical, "alias": alias, "basis": basis})


def _evaluate_constraint(idx: "Index", cid: str, body: dict[str, Any]) -> Decision:
    ctype = body.get("ctype") or body.get("kind")
    if ctype not in ("mutex", "functional"):
        return _reject(cid, "constraint", 1, f"unknown constraint kind {ctype!r}")
    inner = body.get("body")
    if not isinstance(inner, dict) or "pred" not in inner:
        return _reject(cid, "constraint", 1, "constraint body needs a `pred`")
    return Decision(cid, "constraint", "admitted",
                    body={"ctype": ctype, "body": inner})


def _evaluate_verifier(idx: "Index", cid: str, body: dict[str, Any]) -> Decision:
    code = body.get("code")
    entry = body.get("entry")
    if not isinstance(code, str) or not isinstance(entry, str):
        return _reject(cid, "verifier", 1, "verifier candidate needs `code` and `entry`")
    vectors = [(tuple(v[0]), v[1]) for v in (body.get("vectors") or [])]
    ok, detail = sandbox.check_vectors(code, entry, vectors)
    if not ok:
        return _reject(cid, "verifier", 3, detail)
    return Decision(cid, "verifier", "admitted", body={
        "oracle_id": body.get("oracle_id") or f"verifier:{cid}",
        "kind": body.get("oracle_kind", "deterministic_total"),
        "entry": entry, "code": code,
        "code_hash": sandbox.code_hash(code), "env_hash": sandbox.env_hash()})


def _evaluate_rule(idx: "Index", cid: str, kind: str,
                   body: dict[str, Any]) -> Decision:
    head = body.get("head")
    rule_body = body.get("body") or {}
    if not isinstance(head, dict) or "pred" not in head:
        return _reject(cid, kind, 1, "rule candidate needs a head literal")
    for lit in rule_body.get("literals", []):
        if "pred" not in lit:
            return _reject(cid, kind, 1, "body literal without a predicate")
    holdout = body.get("holdout") or {}
    hits = int(holdout.get("hits", 0))
    misses = int(holdout.get("misses", 0))
    if hits + misses and hits <= misses:
        return _reject(cid, kind, 5,
                       f"held-out evidence check failed ({hits} hits / "
                       f"{misses} misses)")
    if kind == "guard":
        support = body.get("support") or {}
        if min(int(support.get("left", 0)), int(support.get("right", 0))) < GUARD_MIN_SUPPORT:
            return _reject(cid, "guard", 5,
                           f"partition support below {GUARD_MIN_SUPPORT} per side")
        mdl = body.get("mdl") or {}
        cost = float(mdl.get("dl_guard", 0.0)) + float(mdl.get("dl_residual_given_guard", 0.0))
        base = float(mdl.get("dl_residual", 0.0))
        if not cost < base:
            return _reject(cid, "guard", 6,
                           f"MDL: {cost:.3f} !< {base:.3f}, guard costs more than "
                           f"it compresses")
    return Decision(cid, kind, "admitted", body=body)


#: A supersede closes a fact's validity window, so it is held to the same
#: evidentiary bar as any other structural change. It was previously the only
#: candidate kind admitted unconditionally, which meant a periphery
#: false-positive became committed history with nothing in the way (F4).
SUPERSEDE_ALPHA = 0.01


def _evaluate_supersede(idx: "Index", cid: str, body: dict[str, Any]) -> Decision:
    """Steps 1/5/6 for a located regime change."""
    fact_id = body.get("fact_id")
    if not fact_id:
        return _reject(cid, "supersede_valid_time", 1,
                       "supersede candidate names no fact")
    if idx.one("SELECT id FROM facts WHERE id=?", (fact_id,)) is None:
        return _reject(cid, "supersede_valid_time", 1,
                       f"supersede targets unknown fact {fact_id!r}")
    if body.get("valid_to") is None:
        return _reject(cid, "supersede_valid_time", 5,
                       "no located date: a regime change that cannot say WHEN "
                       "is not a regime change")
    support = body.get("support") or {}
    thin = min(int(support.get("before", 0)), int(support.get("after", 0)))
    if thin < GUARD_MIN_SUPPORT:
        return _reject(cid, "supersede_valid_time", 5,
                       f"regime support below {GUARD_MIN_SUPPORT} observations "
                       f"per side (thinnest side {thin})")
    pvalue = body.get("pvalue")
    if pvalue is None or float(pvalue) > SUPERSEDE_ALPHA:
        return _reject(cid, "supersede_valid_time", 6,
                       f"level change not significant after correcting for the "
                       f"searched changepoint (p={pvalue})")
    return Decision(cid, "supersede_valid_time", "admitted", body=body)


# ── shared checks ───────────────────────────────────────────────────────────

def _admitted_constraints(idx: "Index") -> list[constraints_mod.Constraint]:
    return [constraints_mod.parse(r) for r in idx.query("SELECT * FROM constraints")]


def _certain_contradiction(idx: "Index", pred: str, args: list[Any],
                           fid: str) -> Optional[str]:
    """Step 7 over the *certain* fragment of the closure.

    A candidate is rejected when it would contradict something the system holds
    with certainty: a '+'-pinned fact or a `definitional` one. Tension between
    two merely-admitted (epistemically uncertain) facts is admitted and shows up
    as `rejection_rate` in `predict` (§3.9). See DEVIATIONS.md D3.
    """
    cons = _admitted_constraints(idx)
    if not cons:
        return None
    for c in cons:
        key = c.group_key(pred, list(args))
        if key is None:
            continue
        for row in idx.query(
                "SELECT id, pred, args_json, kind, structural FROM facts"):
            other_id = row["id"]
            if other_id == fid:
                continue
            other_args = json.loads(row["args_json"])
            if c.group_key(row["pred"], other_args) != key:
                continue
            certain = (row["kind"] == "definitional"
                       or row["structural"] == "pinned"
                       or facts_mod.pin_polarity(idx, other_id) == "+")
            if certain:
                return (f"constraint {c.id} is violated against certain fact "
                        f"{other_id}")
    return None


def _alias_merge_conflict(idx: "Index", canonical: str, alias: str) -> Optional[str]:
    """§3.4: zero constraint conflicts between the merged extensions."""
    cons = _admitted_constraints(idx)
    if not cons:
        return None
    left = idx.query("SELECT pred, args_json FROM facts WHERE pred=?", (canonical,))
    right = idx.query("SELECT pred, args_json FROM facts WHERE pred=?", (alias,))
    merged = [(canonical, json.loads(r["args_json"])) for r in left] + \
             [(canonical, json.loads(r["args_json"])) for r in right]
    bad = constraints_mod.violated_by(cons, merged)
    if bad:
        return f"merged extensions violate constraint {bad}"
    return None


def demotion_warranted(admission_log_odds: float, against_log_odds: float) -> bool:
    """§3.4 hysteresis: the bar for removal is strictly higher than for entry."""
    import math
    return against_log_odds > admission_log_odds + math.log(DEMOTION_HYSTERESIS)
