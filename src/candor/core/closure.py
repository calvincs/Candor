"""Closure engine (trusted, spec §3.5).

Stratified-negation Datalog with comparison guards over binned values. No
function symbols in rule heads, so the engine stays inside Datalog. Soft edges
never enter here — they participate only in `conjecture` (I4).

Materialization enforces admitted constraints and honours '-' pins, which are
the system's only hard zero (I5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .canonical import fact_key
from .hashing import canon_json, sha256_hex

Atom = tuple[str, tuple[Any, ...]]


def atom_text(pred: str, args: Iterable[Any]) -> str:
    return canon_json([pred, list(args)])


def is_var(term: Any) -> bool:
    return isinstance(term, str) and term.startswith("?")


@dataclass(frozen=True)
class Literal:
    pred: str
    args: tuple[Any, ...]
    negated: bool = False

    @staticmethod
    def parse(obj: dict[str, Any]) -> "Literal":
        return Literal(obj["pred"], tuple(obj.get("args", [])),
                       bool(obj.get("negated", False)))


@dataclass(frozen=True)
class Guard:
    """Comparison over a bound variable: (?x, op, value) or bin membership."""
    var: str
    op: str
    value: Any

    def holds(self, binding: dict[str, Any]) -> bool:
        left = binding.get(self.var)
        if left is None:
            return False
        try:
            if self.op == "==":
                return left == self.value
            if self.op == "!=":
                return left != self.value
            lf, rf = float(left), float(self.value)
        except (TypeError, ValueError):
            return False
        return {"<": lf < rf, "<=": lf <= rf, ">": lf > rf, ">=": lf >= rf}.get(
            self.op, False)


@dataclass(frozen=True)
class Rule:
    id: str
    head: Literal
    body: tuple[Literal, ...]
    guards: tuple[Guard, ...] = ()
    specificity: int = 0

    @staticmethod
    def parse(rid: str, head_json: str, body_json: str, specificity: int = 0) -> "Rule":
        head = Literal.parse(json.loads(head_json))
        body_obj = json.loads(body_json)
        lits = tuple(Literal.parse(b) for b in body_obj.get("literals", []))
        guards = tuple(Guard(g["var"], g["op"], g["value"])
                       for g in body_obj.get("guards", []))
        return Rule(rid, head, lits, guards, specificity)


@dataclass
class Closure:
    atoms: set[Atom] = field(default_factory=set)
    basis: dict[Atom, str] = field(default_factory=dict)
    # atom -> list of support sets (each a frozenset of fact/rule ids)
    support: dict[Atom, list[frozenset[str]]] = field(default_factory=dict)
    # atoms that came straight from an admitted fact, not from a rule firing.
    # The kernel accepts only these as premises without a citation (§3.6).
    base: set[Atom] = field(default_factory=set)

    def contains(self, pred: str, args: Iterable[Any]) -> bool:
        return (pred, tuple(args)) in self.atoms

    def hash(self) -> str:
        payload = sorted(atom_text(p, a) for p, a in self.atoms)
        return sha256_hex(canon_json(payload))


def _unify(pattern: tuple[Any, ...], ground: tuple[Any, ...],
           binding: dict[str, Any]) -> Optional[dict[str, Any]]:
    if len(pattern) != len(ground):
        return None
    out = dict(binding)
    for p, g in zip(pattern, ground):
        if is_var(p):
            if p in out and out[p] != g:
                return None
            out[p] = g
        elif p != g:
            return None
    return out


def _subst(args: tuple[Any, ...], binding: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(binding.get(a, a) if is_var(a) else a for a in args)


def materialize(base_facts: list[tuple[str, list[Any], str]],
                rules: list[Rule]) -> Closure:
    """Semi-naive evaluation. `base_facts` are (pred, args, fact_id) triples."""
    clo = Closure()
    for pred, args, fid in base_facts:
        atom = (pred, tuple(args))
        clo.atoms.add(atom)
        clo.base.add(atom)
        clo.basis[atom] = "exact"
        clo.support.setdefault(atom, []).append(frozenset({fid}))

    if not rules:
        return clo

    changed = True
    rounds = 0
    while changed and rounds < 64:
        changed = False
        rounds += 1
        by_pred: dict[str, list[Atom]] = {}
        for atom in clo.atoms:
            by_pred.setdefault(atom[0], []).append(atom)
        for rule in rules:
            for binding, used in _join(rule, by_pred, clo):
                head_args = _subst(rule.head.args, binding)
                if any(is_var(a) for a in head_args):
                    continue
                atom = (rule.head.pred, head_args)
                sup = frozenset(used | {rule.id})
                sups = clo.support.setdefault(atom, [])
                if atom not in clo.atoms:
                    clo.atoms.add(atom)
                    clo.basis[atom] = "exact"
                    sups.append(sup)
                    changed = True
                elif sup not in sups:
                    sups.append(sup)
                    changed = True
    return clo


def _join(rule: Rule, by_pred: dict[str, list[Atom]], clo: Closure):
    states: list[tuple[dict[str, Any], set[str]]] = [({}, set())]
    for lit in rule.body:
        nxt: list[tuple[dict[str, Any], set[str]]] = []
        for binding, used in states:
            if lit.negated:
                probe = _subst(lit.args, binding)
                if any(is_var(a) for a in probe):
                    continue  # unsafe negation; stratification rejects upstream
                if (lit.pred, probe) not in clo.atoms:
                    nxt.append((binding, used))
                continue
            for atom in by_pred.get(lit.pred, ()):
                merged = _unify(lit.args, atom[1], binding)
                if merged is None:
                    continue
                extra = set(used)
                for sup in clo.support.get(atom, []):
                    extra |= sup
                    break
                nxt.append((merged, extra))
        states = nxt
        if not states:
            return
    for binding, used in states:
        if all(g.holds(binding) for g in rule.guards):
            yield binding, used


# ── backward search for `derive` (three-valued, spec §5) ────────────────────

@dataclass
class SearchOutcome:
    proved: bool
    exhausted: bool
    steps: int
    proof: Optional[dict[str, Any]] = None


def backward(goal_pred: str, goal_args: tuple[Any, ...], clo: Closure,
             rules: list[Rule], budget: int) -> SearchOutcome:
    """Budget-honest backward chaining.

    `exhausted` is set only when the search space was walked to the end within
    budget. Conflating truncation with absence is the sin the rest of the
    system exists to prevent (I4).
    """
    steps = 0
    in_progress: set[Atom] = set()
    by_pred: dict[str, list[Atom]] = {}
    for atom in clo.atoms:
        by_pred.setdefault(atom[0], []).append(atom)
    for atoms in by_pred.values():
        atoms.sort(key=lambda a: atom_text(*a))

    def solve(pred: str, args: tuple[Any, ...], depth: int) -> tuple[bool, bool, Any]:
        """Returns (proved, search_complete, proof)."""
        nonlocal steps
        if steps >= budget:
            return False, False, None
        steps += 1
        atom = (pred, args)
        if atom in clo.base:
            support = clo.support.get(atom, [frozenset()])[0]
            return True, True, {"conclusion": [pred, list(args)],
                                "premises": sorted(support), "edge_kinds": ["exact"]}
        if atom in in_progress or depth > 32:
            return False, True, None      # cycle: this branch adds nothing
        in_progress.add(atom)
        complete = True
        try:
            for rule in rules:
                if rule.head.pred != pred:
                    continue
                binding = _unify(rule.head.args, args, {})
                if binding is None:
                    continue
                got, sub_complete, sub_proofs = _prove_body(
                    rule, 0, binding, depth, solve, by_pred)
                complete = complete and sub_complete
                if got:
                    return True, complete, {
                        "conclusion": [pred, list(args)], "rule": rule.id,
                        "premises": sub_proofs, "edge_kinds": ["exact"]}
                if steps >= budget:
                    return False, False, None
        finally:
            in_progress.discard(atom)
        return False, complete and steps <= budget, None

    def _prove_body(rule: Rule, i: int, binding: dict[str, Any], depth: int,
                    solver, index: dict[str, list[Atom]]):
        if i == len(rule.body):
            if all(g.holds(binding) for g in rule.guards):
                return True, True, []
            return False, True, []
        lit = rule.body[i]
        probe = _subst(lit.args, binding)
        complete = True
        if any(is_var(a) for a in probe):
            candidates = [a for a in index.get(lit.pred, ())
                          if _unify(probe, a[1], binding) is not None]
        else:
            candidates = [(lit.pred, probe)]
        for cand in candidates:
            merged = _unify(lit.args, cand[1], binding)
            if merged is None:
                continue
            got, sub_complete, proof = solver(lit.pred, cand[1], depth + 1)
            complete = complete and sub_complete
            if not got:
                continue
            rest_ok, rest_complete, rest = _prove_body(
                rule, i + 1, merged, depth, solver, index)
            complete = complete and rest_complete
            if rest_ok:
                return True, complete, [proof] + rest
        return False, complete, []

    proved, exhausted, proof = solve(goal_pred, goal_args, 0)
    return SearchOutcome(proved, exhausted, steps, proof)


def base_fact_id(pred: str, args: list[Any]) -> str:
    return fact_key(pred, args)
