"""HarnessDriver adapter (spec §6.1).

Production surfaces never carry fault injection, so it lives here: the adapter
wraps `CandorSystem` and adds `reset` / `replay` / `corrupt` plus the enumerated
impl hooks. The conformance suite owns the result types (`DeriveResult`,
`Prediction`, …); they are injected rather than imported so the suite stays the
single definition of the contract and this module stays importable without it.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from . import audit
from .system import CandorSystem, REFUSED

_TEMP_ROOTS: list[Path] = []


def _cleanup() -> None:
    for path in _TEMP_ROOTS:
        shutil.rmtree(path, ignore_errors=True)


atexit.register(_cleanup)


class CandorHarnessDriver:
    """Driver + fault injection + impl hooks. Test builds only."""

    def __init__(self, root: Path, types: Mapping[str, Any]) -> None:
        self.root = Path(root)
        self.t = types
        self.system = CandorSystem(self.root)

    # ── writes ───────────────────────────────────────────────────────────────
    def assert_(self, stmt: dict, source: str, actor: str) -> str:
        return self.system.assert_(stmt, source, actor)

    def observe(self, stmt: dict, outcome: bool, ctx: dict, actor: str) -> int:
        return self.system.observe(stmt, outcome, ctx, actor)

    def observe_batch(self, obs: Sequence[tuple]) -> list[int]:
        return self.system.observe_batch(obs)

    def claim(self, stmt: dict, frame: str, criterion: str, due: int) -> str:
        return self.system.claim(stmt, frame, criterion, due)

    def resolve(self, claim_id: str, outcome: bool) -> int:
        return self.system.resolve(claim_id, outcome)

    def supersede(self, target_id: str, reason: str) -> int:
        return self.system.supersede(target_id, reason)

    def pin(self, target_id: str, polarity: str, reason: str, authority: str) -> int:
        return self.system.pin(target_id, polarity, reason, authority)

    def redact(self, payload_hash: str) -> int:
        return self.system.redact(payload_hash)

    def run_gate(self) -> list[dict]:
        return self.system.run_gate()

    # ── reads ────────────────────────────────────────────────────────────────
    def recall(self, query: str, budget: int) -> list[dict]:
        return self.system.recall(query, budget)

    def derive(self, goal: dict, budget: int):
        out = self.system.derive(goal, budget)
        status = {
            "proof": self.t["DeriveStatus"].PROOF,
            "not_entailed": self.t["DeriveStatus"].NOT_ENTAILED,
            "budget_exceeded": self.t["DeriveStatus"].BUDGET_EXCEEDED,
        }[out.status]
        return self.t["DeriveResult"](status=status, proof=out.proof,
                                      search_exhausted=out.search_exhausted)

    def conjecture(self, goal: dict, sim_budget: float) -> list[dict]:
        return self.system.conjecture(goal, sim_budget)

    def predict(self, stmt: dict, budget: int):
        return self._as_prediction(self.system.predict(stmt, budget))

    def predict_at(self, stmt: dict, snapshot_id: str):
        return self._as_prediction(self.system.predict_at(stmt, snapshot_id))

    def _as_prediction(self, out):
        return self.t["Prediction"](
            p=out.p, ci=out.ci, channels=out.channels, sensitivity=out.sensitivity,
            mpe=out.mpe, caveats=out.caveats, snapshot_id=out.snapshot_id,
            rejection_rate=out.rejection_rate)

    def why(self, obj_id: str) -> dict:
        return self.system.why(obj_id)

    def questions(self, scope: str = "*") -> list[dict]:
        return self.system.questions(scope)

    def events_since(self, cursor: int, kinds: Optional[set[str]] = None) -> list[dict]:
        return self.system.events_since(cursor, kinds)

    def health(self) -> dict:
        return self.system.health()

    # ── §6.1 hooks ───────────────────────────────────────────────────────────
    def reset(self) -> None:
        self.system.reset()

    def replay(self) -> str:
        return self.system.replay()

    def closure_hash(self) -> str:
        return self.system.closure_hash()

    def ledger_head(self) -> str:
        return self.system.ledger_head()

    def corrupt(self, what: str, arg: str = "") -> None:
        self.system.corrupt(what, arg)

    # ── impl hooks ───────────────────────────────────────────────────────────
    def raw_counts(self, fact_id: str):
        return self.t["RawCounts"](by_actor=self.system.raw_counts(fact_id))

    def composed_counts(self, fact_id: str):
        c = self.system.composed_counts(fact_id)
        return self.t["ComposedCounts"](epi_a=c.epi_a, epi_b=c.epi_b,
                                        alea_n=c.alea_n, alea_k=c.alea_k)

    def fact_id_for(self, stmt: dict) -> Optional[str]:
        return self.system.fact_id_for(stmt)

    def set_reliability(self, actor: str, frame: str, a: float, b: float) -> None:
        self.system.set_reliability(actor, frame, a, b)

    def storage_scan_nonintegral_counts(self) -> list[str]:
        return self.system.index.nonintegral_counts()

    def grep_weight_outside_committed(self) -> list[str]:
        return audit.grep_weight_outside_committed(
            evidence_dirs=[self.root / "evidence"])

    def retrieval_writer_import_paths(self) -> list[str]:
        return audit.retrieval_writer_import_paths()


def build_harness_driver(types: Mapping[str, Any],
                         root: Optional[Path] = None) -> CandorHarnessDriver:
    """Factory for `make_driver()` in the conformance suite."""
    if root is None:
        root = Path(tempfile.mkdtemp(prefix="candor-conformance-"))
        _TEMP_ROOTS.append(root)
    return CandorHarnessDriver(root, types)
