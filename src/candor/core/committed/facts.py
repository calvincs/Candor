"""Committed-tier accessors: facts, aliases, pins (trusted, spec §2, §3.13).

Counts are NEVER merged on alias. The union happens here, at read time, over
the alias closure — so a bad alias is reversible by supersede and the audit
trail stays intact (I11).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

from ..canonical import canonicalize_args, fact_key

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..index import Index


# ── predicate registry ──────────────────────────────────────────────────────

def predicate(idx: "Index", pred: str) -> Optional[dict[str, Any]]:
    row = idx.one("SELECT * FROM predicates WHERE pred=?", (pred,))
    if row is None:
        return None
    return {
        "pred": row["pred"], "arity": row["arity"],
        "arg_types": json.loads(row["arg_types_json"]),
        "canonical_units": json.loads(row["canonical_units_json"] or "{}"),
    }


def canonical_args(idx: "Index", pred: str, args: list[Any]) -> list[Any]:
    reg = predicate(idx, pred)
    return canonicalize_args(args, (reg or {}).get("canonical_units"))


# ── alias closure ───────────────────────────────────────────────────────────

def alias_closure(idx: "Index", symbol: str) -> list[str]:
    """All symbols identified with `symbol` under admitted aliases, sorted."""
    seen = {symbol}
    frontier = [symbol]
    while frontier:
        cur = frontier.pop()
        for row in idx.query(
                "SELECT canonical, alias FROM aliases WHERE canonical=? OR alias=?",
                (cur, cur)):
            for other in (row["canonical"], row["alias"]):
                if other not in seen:
                    seen.add(other)
                    frontier.append(other)
    return sorted(seen)


def resolve_ids(idx: "Index", pred: str, args: list[Any]) -> list[str]:
    """Every stored fact id this statement reads through (self + alias siblings)."""
    cargs = canonical_args(idx, pred, args)
    ids: list[str] = []
    for sym in alias_closure(idx, pred):
        fid = fact_key(sym, canonical_args(idx, sym, args) if sym != pred else cargs)
        if fid not in ids:
            ids.append(fid)
    return ids


def lookup(idx: "Index", pred: str, args: list[Any]) -> Optional[str]:
    """The admitted fact id for a statement, resolving through aliases."""
    for fid in resolve_ids(idx, pred, args):
        if idx.one("SELECT id FROM facts WHERE id=?", (fid,)) is not None:
            return fid
    return None


def get(idx: "Index", fact_id: str) -> Optional[dict[str, Any]]:
    row = idx.one("SELECT * FROM facts WHERE id=?", (fact_id,))
    if row is None:
        return None
    out = dict(row)
    out["args"] = json.loads(out.pop("args_json"))
    return out


# ── pins (the only hard zero, I5) ───────────────────────────────────────────

def pin_polarity(idx: "Index", target_id: str) -> Optional[str]:
    row = idx.one(
        "SELECT polarity FROM pins WHERE target_id=? AND active=1 "
        "ORDER BY created_at DESC, id DESC", (target_id,))
    return row["polarity"] if row else None


def is_negatively_pinned(idx: "Index", target_id: str) -> bool:
    return pin_polarity(idx, target_id) == "-"
