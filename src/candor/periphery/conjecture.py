"""Soft unification — untrusted hypothesis generator (spec §4.3, I4).

A symbol's behavioural signature is the set of rules it fires in (with argument
positions) plus its co-derivation set, computed over the *alias closure* of the
materialized exact closure. No text embeddings anywhere: "these words look
alike" is not a licence for substitution, and there is no embedding model, no
network call, and nothing uninspectable in this file.

Soft edges propose. Observation tests. The gate promotes survivors to exact.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional

# §3.13 cold start: a symbol with fewer than this many signature components is
# invisible to the conjecture engine. Analogy from nothing is exactly the
# licence this system refuses to grant.
SIGNATURE_SUPPORT_FLOOR = 2


def signature(symbol: str, rule_firings: Iterable[tuple[str, int, str]],
              co_derived: Iterable[str]) -> dict[str, float]:
    """Sparse vector over (rule_id, arg_position) plus co-derivation membership."""
    sig: dict[str, float] = {}
    for rule_id, position, sym in rule_firings:
        if sym != symbol:
            continue
        sig[f"r:{rule_id}:{position}"] = sig.get(f"r:{rule_id}:{position}", 0.0) + 1.0
    for other in co_derived:
        sig[f"c:{other}"] = sig.get(f"c:{other}", 0.0) + 1.0
    return sig


def cosine(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def neighbourhood(target: str, signatures: Mapping[str, Mapping[str, float]],
                  sim_budget: float, limit: int = 16) -> list[tuple[str, float]]:
    """Nearest behavioural neighbours above a similarity budget, best first."""
    own = signatures.get(target) or {}
    if len(own) < SIGNATURE_SUPPORT_FLOOR:
        return []          # cold start: blind here by design (§3.13)
    out: list[tuple[str, float]] = []
    for sym, sig in signatures.items():
        if sym == target or len(sig) < SIGNATURE_SUPPORT_FLOOR:
            continue
        sim = cosine(own, sig)
        if sim >= sim_budget:
            out.append((sym, sim))
    out.sort(key=lambda t: (-t[1], t[0]))
    return out[:limit]


def conjectures(goal: Mapping[str, Any], neighbours: list[tuple[str, float]],
                known: Optional[set[tuple[str, tuple]]] = None) -> list[dict[str, Any]]:
    """Analogical substitutions, each carrying its similarity budget.

    Never typed as a Proof. The caller is expected to keep it that way (I4).
    """
    known = known or set()
    out: list[dict[str, Any]] = []
    args = tuple(goal.get("args", []))
    for sym, sim in neighbours:
        if (sym, args) not in known:
            continue
        out.append({
            "goal": {"pred": goal["pred"], "args": list(args)},
            "via": {"pred": sym, "args": list(args)},
            "sim": sim,
            "edge_kinds": ["soft"],
            "quality": "conjecture",
            "caveats": ["analogy_budget"],
        })
    return out
