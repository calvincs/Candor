"""Proof checker kernel — trusted, tiny, isolated (spec §3.6).

Independent of the search that produced the derivation. It re-verifies that
each emitted conclusion follows from its cited premises. Checking is
polynomially cheap relative to finding, so it runs on every emitted proof.

This module imports nothing from the rest of CANDOR except the closure term
representation, on purpose: it is the only component whose correctness is
provable in the strict sense and the only eval that catches bugs in the engine.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .closure import Literal, Rule, _subst, _unify, is_var


class KernelReject(Exception):
    """A derivation failed independent verification. Fail-stop (§6.2)."""


def check(proof: dict[str, Any], base_atoms: set[tuple[str, tuple[Any, ...]]],
          rules_by_id: dict[str, Rule]) -> bool:
    """Verify one derivation tree. Raises KernelReject with the offending step."""
    if not isinstance(proof, dict) or "conclusion" not in proof:
        raise KernelReject("malformed proof object")
    pred, args = proof["conclusion"][0], tuple(proof["conclusion"][1])

    rule_id = proof.get("rule")
    if rule_id is None:
        if (pred, args) not in base_atoms:
            raise KernelReject(f"cited base fact is not admitted: {pred}{args}")
        return True

    rule = rules_by_id.get(rule_id)
    if rule is None:
        raise KernelReject(f"proof cites unknown rule {rule_id!r}")

    binding = _unify(rule.head.args, args, {})
    if binding is None:
        raise KernelReject(f"rule {rule_id!r} head does not unify with conclusion")

    premises: Sequence[Any] = proof.get("premises") or []
    if len(premises) != len(rule.body):
        raise KernelReject(
            f"rule {rule_id!r} needs {len(rule.body)} premises, proof cites "
            f"{len(premises)}")

    for lit, sub in zip(rule.body, premises):
        if not isinstance(sub, dict):
            raise KernelReject("premise is not a derivation object")
        sub_pred, sub_args = sub["conclusion"][0], tuple(sub["conclusion"][1])
        if sub_pred != lit.pred:
            raise KernelReject(
                f"premise predicate {sub_pred!r} does not match body literal "
                f"{lit.pred!r}")
        merged = _unify(lit.args, sub_args, binding)
        if merged is None:
            raise KernelReject(f"premise {sub_pred}{sub_args} does not unify")
        binding = merged
        check(sub, base_atoms, rules_by_id)

    for guard in rule.guards:
        if not guard.holds(binding):
            raise KernelReject(f"guard {guard.var} {guard.op} {guard.value} fails")

    expected = _subst(rule.head.args, binding)
    if any(is_var(a) for a in expected) or tuple(expected) != args:
        raise KernelReject("head instantiation does not match the conclusion")
    return True


def quality(proof: dict[str, Any], flagged: set[str], narrow: set[str]) -> str:
    """§4.7 derivation quality: proof / proof-modulo-unknown-context / conjecture."""
    kinds: set[str] = set()
    premises: set[str] = set()

    def walk(node: dict[str, Any]) -> None:
        kinds.update(node.get("edge_kinds", []))
        subs = node.get("premises") or []
        if subs and isinstance(subs[0], str):
            premises.update(subs)
            return
        for sub in subs:
            if isinstance(sub, dict):
                walk(sub)

    walk(proof)
    if "soft" in kinds:
        return "conjecture"
    if premises & flagged or premises & narrow:
        return "proof-modulo-unknown-context"
    return "proof"
