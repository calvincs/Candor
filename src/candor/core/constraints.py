"""Integrity constraints — first-class, gate-admitted (spec §2, §3.4 step 7).

Two kinds:
  mutex       — a predicate's value position is exclusive over a value set
  functional  — a predicate is single-valued in the given argument positions

The gate uses these to reject candidates that would make the *certain* fragment
of the closure inconsistent. Tension among merely-admitted (uncertain) facts is
not a gate rejection: it is resolved at prediction time by constraint
conditioning with a reported rejection rate (§3.9). See DEVIATIONS.md D3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class Constraint:
    id: str
    kind: str
    body: dict[str, Any]

    # ── group membership ────────────────────────────────────────────────────
    def matches(self, pred: str, args: list[Any]) -> bool:
        if self.body.get("pred") != pred:
            return False
        if self.kind == "mutex":
            values = self.body.get("exclusive_values")
            if values is None:
                return True
            pos = int(self.body.get("value_position", len(args) - 1))
            return 0 <= pos < len(args) and args[pos] in values
        if self.kind == "functional":
            return True
        return False

    def group_key(self, pred: str, args: list[Any]) -> Optional[tuple]:
        """Facts sharing a group key may not all be true at once."""
        if not self.matches(pred, args):
            return None
        if self.kind == "mutex":
            pos = int(self.body.get("value_position", len(args) - 1))
            keyed = tuple(a for i, a in enumerate(args) if i != pos)
            return (self.id, keyed)
        positions = self.body.get("key_positions") or list(range(len(args) - 1))
        return (self.id, tuple(args[i] for i in positions if i < len(args)))


def parse(row: Any) -> Constraint:
    return Constraint(row["id"], row["kind"], json.loads(row["body_json"]))


def violated_by(constraints: Iterable[Constraint],
                atoms: Iterable[tuple[str, list[Any]]]) -> Optional[str]:
    """Return the id of a constraint violated by this set of true atoms."""
    seen: dict[tuple, int] = {}
    atoms = list(atoms)
    for c in constraints:
        for pred, args in atoms:
            key = c.group_key(pred, list(args))
            if key is None:
                continue
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 1:
                return c.id
    return None


def groups_for(constraints: Iterable[Constraint],
               atoms: Iterable[tuple[str, list[Any], str]]) -> dict[tuple, list[str]]:
    """Map group key -> fact ids sharing it, for constraint-conditioned sampling."""
    out: dict[tuple, list[str]] = {}
    for c in constraints:
        for pred, args, fid in atoms:
            key = c.group_key(pred, list(args))
            if key is None:
                continue
            out.setdefault(key, [])
            if fid not in out[key]:
                out[key].append(fid)
    return {k: v for k, v in out.items() if len(v) > 1}
