"""Extractor / proposer — untrusted (spec §3.3, I10).

Turns incoming statements into `candidates`, each carrying a span reference.
It cannot write to `facts`, `rules`, or any committed-tier column: everything it
produces goes through the gate.

Registry linkage (§3.3): a fact naming an unregistered predicate is emitted
*alongside* a `symbol` candidate proposing the predicate, never assumed into
existence.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

CANDIDATE_KINDS = ("fact", "rule", "guard", "verifier", "symbol", "alias",
                   "constraint", "supersede_valid_time")


class ExtractionError(ValueError):
    pass


def classify(stmt: Mapping[str, Any]) -> str:
    """Which candidate kind a §5 `assert` payload describes."""
    kind = stmt.get("kind")
    if kind in CANDIDATE_KINDS:
        return kind
    if "pred" in stmt and "args" in stmt:
        return "fact"
    if "head" in stmt:
        return "rule"
    raise ExtractionError(f"cannot classify assertion payload: {dict(stmt)!r}")


def propose(stmt: Mapping[str, Any],
            known_predicates: Mapping[str, int]) -> list[tuple[str, dict[str, Any]]]:
    """Return (candidate_kind, body) pairs in the order they must reach the gate."""
    kind = classify(stmt)
    body = {k: v for k, v in stmt.items() if k != "kind"}

    if kind == "fact":
        out: list[tuple[str, dict[str, Any]]] = []
        pred, args = stmt["pred"], list(stmt["args"])
        if pred not in known_predicates:
            out.append(("symbol", {
                "pred": pred, "arity": len(args),
                "arg_types": ["any"] * len(args), "canonical_units": {},
                "proposed_for": "unregistered predicate seen in a fact assertion",
            }))
        out.append(("fact", {
            "pred": pred, "args": args,
            "stmt_type": stmt.get("stmt_type", "crisp"),
            "kind": stmt.get("basis", "exact"),
        }))
        return out

    if kind == "symbol":
        return [("symbol", {
            "pred": stmt["pred"], "arity": int(stmt.get("arity", 0)),
            "arg_types": list(stmt.get("arg_types") or []),
            "canonical_units": dict(stmt.get("canonical_units") or {}),
        })]

    return [(kind, body)]


def suspected_alias(canonical: str, alias: str, sim: float) -> dict[str, Any]:
    """Suspected co-reference is *proposed*, never assumed (§3.3)."""
    return {"canonical": canonical, "alias": alias, "basis": "behavioral",
            "sim": float(sim)}


def span_ref(entry_id: Optional[str], content_hash: Optional[str],
             offset: int = 0) -> Optional[str]:
    if not entry_id:
        return None
    return f"{entry_id}|{content_hash or ''}|{offset}"
