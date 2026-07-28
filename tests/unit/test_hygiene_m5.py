"""Phase 2 FIX M5 — NaN/Infinity must never enter canonical payloads.

canon_json used json.dumps with the default allow_nan=True, so NaN/Infinity got
written verbatim into payload files (non-portable JSON) and poisoned float math.
canon_json must reject non-finite floats at serialization, and the gate must
reject a fact candidate whose canonicalized args contain a non-finite float
before it can reach the committed tier.
"""

from __future__ import annotations

import math

import pytest

from candor.core import gate
from candor.core.hashing import canon_json


# ── (a) serialization rejects non-finite ─────────────────────────────────────

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_m5_canon_json_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        canon_json({"x": bad})


def test_m5_canon_json_still_serializes_finite_values():
    assert canon_json({"x": 1.5, "y": 0, "z": -2.25}) == '{"x":1.5,"y":0,"z":-2.25}'


# ── (a) an observe with NaN confidence raises, and records nothing ───────────

def test_m5_observe_with_nan_confidence_raises_and_records_nothing(seeded):
    before = seeded.ledger.seq()
    with pytest.raises(ValueError):
        seeded.observe({"pred": "flaky_link", "args": ["c", "d"]}, True, {},
                       actor="tool:probe", confidence=float("nan"))
    assert seeded.ledger.seq() == before, "a rejected observe must record nothing"
    assert seeded.ledger.verify_chain() is True


# ── (b) the gate rejects a fact candidate carrying a non-finite arg ──────────

def test_m5_gate_rejects_non_finite_fact_arg(seeded):
    d = gate.evaluate(seeded.index, "cand:nf", "fact",
                      {"pred": "reachable", "args": [float("inf"), "b"]},
                      "human:calvin")
    assert d.status == "rejected"
    assert d.failing_step in (1, 2)


def test_m5_gate_admits_finite_fact_arg(seeded):
    d = gate.evaluate(seeded.index, "cand:ok", "fact",
                      {"pred": "reachable", "args": ["x", "y"]},
                      "human:calvin")
    assert d.status == "admitted"
    # sanity: the finite path is genuinely finite.
    assert all(not (isinstance(a, float) and not math.isfinite(a))
               for a in d.body["args"])
