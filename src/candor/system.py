"""CANDOR system assembly — the §5 agent-facing API over core + periphery.

This is the only module that knows about both halves of the tree. It holds no
epistemic logic of its own: it sequences ledger appends, hands candidates to the
gate, and composes read-time views. Everything it writes goes through
`core.apply.apply_event`, so the live index is replay-equivalent by
construction (I1, I3).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .core import apply as apply_mod
from .core import calibration as calibration_mod
from .core import closure as closure_mod
from .core import constraints as constraints_mod
from .core import gate as gate_mod
from .core import kernel as kernel_mod
from .core.canonical import context_signature, fact_key
from .core.committed import counts as counts_mod
from .core.committed import facts as facts_mod
from .core.committed import reliability as reliability_mod
from .core.index import Index
from .core.ledger import Ledger, is_payload_hash
from .periphery import conjecture as conjecture_mod
from .periphery import curiosity_engine as curiosity_mod
from .periphery import embedder as embedder_mod
from .periphery import extractor as extractor_mod
from .periphery import predict as predict_mod
from .periphery.retrieval import RetrievalLog, Retriever

REFUSED = "Refused"


class QuotaExceeded(RuntimeError):
    """§3.12 flooding bound. Raised at the API boundary, before the ledger."""


class Refused(RuntimeError):
    """§3.8: an unsettleable claim is refused entry and stays prose."""


@dataclass(frozen=True)
class DeriveOutcome:
    status: str                       # 'proof' | 'not_entailed' | 'budget_exceeded'
    proof: Optional[dict[str, Any]]
    search_exhausted: bool
    quality: str = "proof"


@dataclass(frozen=True)
class PredictOutcome:
    p: float
    ci: tuple[float, float]
    channels: dict[str, float]
    sensitivity: dict[str, float]
    mpe: Any
    caveats: frozenset[str]
    snapshot_id: str
    rejection_rate: float


class CandorSystem:
    """One store. Single writer to the gate, single sequencer for the ledger."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.ledger = Ledger(self.root / "ledger")
        self.index = Index(self.root / "index.sqlite3")
        self.retrieval_log = RetrievalLog(self.root / "retrieval.sqlite3")
        self.retriever = Retriever(
            evidence_dir=self.root / "evidence",
            payload_dir=self.root / "ledger" / "payloads",
            log=self.retrieval_log,
            dense=embedder_mod.from_env(self.root))   # v0.3 Δ5: optional, injected
        self._closure: Optional[closure_mod.Closure] = None
        self._calib = calibration_mod.IsotonicMap()
        self._health_events: list[dict[str, Any]] = []
        self.open()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def open(self) -> None:
        (self.root / "evidence").mkdir(parents=True, exist_ok=True)
        self.ledger.open()
        self.index.open()
        self._refold()

    def reset(self) -> None:
        self.ledger.destroy()
        self.index.reset()
        self._closure = None
        self._health_events = []
        self.ledger.open()

    def close(self) -> None:
        self.ledger.close()
        self.index.close()
        self.retrieval_log.close()

    # ── replay (§6.1 hook, I1) ───────────────────────────────────────────────
    def _refold(self) -> None:
        """Fold the whole log into a fresh index. Redaction implies exclusion."""
        # Start from an empty index unconditionally (C1): the count writers are
        # increments, so folding onto a non-empty index re-adds every prior
        # count. open() used to skip this and double-count on every reopen;
        # replay/retract_source/redact already reset and now just reset twice
        # (harmless), matching this method's "fresh index" contract.
        self.index.reset()
        redacted = self._redacted_payloads()
        retracted = self._retracted_actors()
        for ev in self.ledger.read_all():
            # A retracted source keeps its event skeletons forever (nothing is
            # erased) but contributes no payload, so every downstream number
            # recomputes as if it never spoke. Retraction events themselves are
            # always applied, or retracting an operator would be irreversible.
            silenced = ev.actor in retracted and ev.kind != "retraction"
            payload = (None if silenced or ev.payload_hash in redacted
                       else self.ledger.payload(ev.payload_hash))
            apply_mod.apply_event(self.index, ev, payload)
        # Reliability overrides fold in naturally now (they are `reliability`
        # events), in true ledger order — no post-fold re-application (I3).
        # Replay determinism: flags/questions/breadth are a deterministic
        # function of the folded observations, so re-derive them here.
        curiosity_mod.sweep(self.index)
        self.index.commit()
        self._closure = None

    def _redacted_payloads(self) -> set[str]:
        out: set[str] = set()
        for ev in self.ledger.read_all():
            if ev.kind != "redaction":
                continue
            payload = self.ledger.payload(ev.payload_hash)
            if payload:
                out.add(payload["payload_hash"])
        return out

    def _retracted_actors(self) -> set[str]:
        """Actors currently silenced. Folded in sequence order, so a later
        restore reverses an earlier retraction without editing anything."""
        out: set[str] = set()
        for ev in self.ledger.read_all():
            if ev.kind != "retraction":
                continue
            payload = self.ledger.payload(ev.payload_hash)
            if not payload:
                continue
            if payload.get("restore"):
                out.discard(payload["actor"])
            else:
                out.add(payload["actor"])
        return out

    def replay(self) -> str:
        self.ledger.close()
        self.ledger.open()
        self.index.reset()
        self._refold()
        return self.closure_hash()

    # ── closure ──────────────────────────────────────────────────────────────
    def closure(self) -> closure_mod.Closure:
        if self._closure is None:
            self._closure = apply_mod.rebuild_closure(self.index)
            self.index.commit()
        return self._closure

    def closure_hash(self) -> str:
        self.closure()
        return apply_mod.closure_hash(self.index)

    def ledger_head(self) -> str:
        return self.ledger.head()

    # ── writes ───────────────────────────────────────────────────────────────
    def _known_predicates(self) -> dict[str, int]:
        known = {r["pred"]: int(r["arity"])
                 for r in self.index.query("SELECT pred, arity FROM predicates")}
        for row in self.index.query(
                "SELECT body_json FROM candidates WHERE kind='symbol' "
                "AND status='pending'"):
            body = json.loads(row["body_json"])
            known.setdefault(body["pred"], int(body.get("arity", 0)))
        return known

    def assert_(self, stmt: Mapping[str, Any], source: str, actor: str) -> str:
        """§5 assert → candidate_id. Never a fact (I10)."""
        self._check_quota(actor, "candidate")
        proposals = extractor_mod.propose(stmt, self._known_predicates())
        last = ""
        for kind, body in proposals:
            ev = self.ledger.append(
                "assertion", actor,
                {"candidate_kind": kind, "body": body},
                source_ref=source)
            apply_mod.apply_event(self.index, ev, {"candidate_kind": kind, "body": body})
            last = apply_mod.candidate_id_for(ev.seq)
        self.index.commit()
        return last

    def run_gate(self) -> list[dict[str, Any]]:
        """Curiosity sweep (§3.10), then drain candidates through the gate."""
        for kind, body in curiosity_mod.sweep(self.index):
            ev = self.ledger.append("assertion", "agent:curiosity",
                                    {"candidate_kind": kind, "body": body},
                                    source_ref="curiosity:sweep")
            apply_mod.apply_event(self.index, ev,
                                  {"candidate_kind": kind, "body": body})
        runs: list[dict[str, Any]] = []
        pending = self.index.query(
            "SELECT id, kind, body_json, proposer FROM candidates "
            "WHERE status='pending' ORDER BY event_seq")
        for row in pending:
            body = json.loads(row["body_json"])
            decision = gate_mod.evaluate(self.index, row["id"], row["kind"], body,
                                         row["proposer"])
            gate_run_id = apply_mod.gate_run_id_for(self.ledger.seq() + 1)
            payload = decision.as_payload()
            payload["gate_run_id"] = gate_run_id
            ev = self.ledger.append("admission", row["proposer"], payload)
            apply_mod.apply_event(self.index, ev, payload)
            if decision.status == "admitted" and decision.candidate_kind == "alias":
                alias_payload = dict(decision.body)
                alias_payload["gate_run_id"] = gate_run_id
                aev = self.ledger.append("alias", row["proposer"], alias_payload)
                apply_mod.apply_event(self.index, aev, alias_payload)
            runs.append({
                "gate_run_id": gate_run_id, "candidate_id": row["id"],
                "candidate_kind": decision.candidate_kind,
                "status": decision.status, "failing_step": decision.failing_step,
                "reason": decision.reason,
            })
        self._closure = None
        self.index.commit()
        return runs

    def set_actor_quota(self, actor: str, obs_per_epoch: Optional[int] = None,
                        cand_per_epoch: Optional[int] = None) -> None:
        """Provision an actor's §3.12 quotas (operational config, spec §2).

        The bound stays enforced at the API boundary; only the configured limit
        changes. Quotas are not part of the replay-determinism hash — they are
        deployment configuration, like the fsync policy.
        """
        apply_mod.ensure_actor(self.index, actor)
        if obs_per_epoch is not None:
            self.index.execute("UPDATE actors SET obs_quota_per_epoch=? WHERE name=?",
                               (int(obs_per_epoch), actor))
        if cand_per_epoch is not None:
            self.index.execute("UPDATE actors SET cand_quota_per_epoch=? WHERE name=?",
                               (int(cand_per_epoch), actor))
        self.index.commit()

    def _check_quota(self, actor: str, kind: str) -> None:
        limit = apply_mod.quota_limit(self.index, actor, kind)
        used = apply_mod.quota_used(self.index, actor, kind)
        if used >= limit:
            self._health_events.append({
                "kind": "quota_exhausted", "actor": actor, "quota_kind": kind,
                "used": used, "limit": limit})
            apply_mod.diagnostic(self.index, 0, "quota_exhausted",
                                 {"actor": actor, "quota_kind": kind, "limit": limit})
            self.index.commit()
            raise QuotaExceeded(
                f"{actor} exhausted its {kind} quota for this epoch "
                f"({used}/{limit})")

    def observe(self, stmt: Mapping[str, Any], outcome: bool,
                ctx: Mapping[str, str], actor: str,
                confidence: Optional[float] = None) -> int:
        self._check_quota(actor, "observation")
        payload = {"stmt": {"pred": stmt["pred"], "args": list(stmt["args"])},
                   "outcome": bool(outcome), "ctx": dict(ctx or {}),
                   "grade": reliability_mod.grade_of(confidence),
                   "confidence": confidence}
        ev = self.ledger.append("observation", actor, payload,
                                context_sig=context_signature(ctx))
        apply_mod.apply_event(self.index, ev, payload)
        self.index.commit()
        return ev.seq

    def observe_batch(self, obs: Sequence[tuple[Mapping[str, Any], bool,
                                                Mapping[str, str], str]]) -> list[int]:
        out = [self.observe(stmt, outcome, ctx, actor)
               for stmt, outcome, ctx, actor in obs]
        return out

    def pin(self, target_id: str, polarity: str, reason: str, authority: str) -> int:
        # Validate at the boundary, before the ledger append (C3): the index's
        # pins.polarity CHECK would otherwise reject the event only after it was
        # already in the chain, bricking every future replay.
        if polarity not in ("+", "-"):
            raise ValueError(f"pin polarity must be '+' or '-', got {polarity!r}")
        payload = {"target_kind": "fact", "target_id": target_id,
                   "polarity": polarity, "reason": reason, "authority": authority}
        ev = self.ledger.append("pin", authority, payload)
        apply_mod.apply_event(self.index, ev, payload)
        self._closure = None
        self.index.commit()
        return ev.seq

    def supersede(self, target_id: str, reason: str,
                  actor: str = "human:operator") -> int:
        payload = self._supersede_payload(str(target_id), reason)
        ev = self.ledger.append("supersede", actor, payload)
        apply_mod.apply_event(self.index, ev, payload)
        self._closure = None
        self.index.commit()
        return ev.seq

    def _supersede_payload(self, target_id: str, reason: str) -> dict[str, Any]:
        """Resolve what is being superseded: an event, a fact, a rule, a pin."""
        if target_id.isdigit():
            row = self.index.one("SELECT kind, payload_hash FROM events WHERE seq=?",
                                 (int(target_id),))
            if row is not None and row["kind"] == "alias":
                payload = self.ledger.payload(row["payload_hash"]) or {}
                return {"target_kind": "alias_event", "target_id": target_id,
                        "reason": reason,
                        "alias": {"canonical": payload.get("canonical"),
                                  "alias": payload.get("alias")}}
            return {"target_kind": "event", "target_id": target_id, "reason": reason}
        if target_id.startswith("pin:"):
            kind = "pin"
        elif target_id.startswith("rule:"):
            kind = "rule"
        elif target_id.startswith("con:"):
            kind = "constraint"
        elif target_id.startswith("cand:"):
            kind = "candidate"
        else:
            kind = "fact"
        return {"target_kind": kind, "target_id": target_id, "reason": reason}

    def retract_source(self, actor: str, reason: str, restore: bool = False,
                       authority: str = "human:operator") -> int:
        """Silence one source. Every number it ever moved recomputes without it.

        This — not `redact` — is how you recover from a bad source. Redaction
        is keyed on a *content-addressed payload*, which by construction has no
        actor in it: two sources reporting the same outcome on the same
        statement in the same context share one payload, so redacting a liar's
        hash also destroys the honest reports that agreed with it. Retraction
        is keyed on the actor, so its blast radius is exactly one source.

        Append-only and reversible: `restore=True` un-silences, and the whole
        history stays in the chain either way.
        """
        payload = {"actor": actor, "reason": reason, "restore": bool(restore)}
        ev = self.ledger.append("retraction", authority, payload)
        self.index.reset()
        self._refold()
        self.closure()
        return ev.seq

    def redaction_scope(self, payload_hash: str) -> dict[str, Any]:
        """Who would lose data if this payload were redacted (§3.13 blast radius).

        Payloads are content-addressed and deduplicated, so a hash can be
        shared by any number of actors. Call this before `redact` when the
        intent is to purge a *source* rather than a piece of content — or use
        `retract_source`, which cannot over-reach.
        """
        rows = self.index.query(
            "SELECT actor, kind FROM events WHERE payload_hash=?", (payload_hash,))
        actors = sorted({r["actor"] for r in rows})
        return {"payload_hash": payload_hash, "events": len(rows),
                "actors": actors, "shared": len(actors) > 1}

    def redact(self, payload_hash: str) -> int:
        """Purge one payload's CONTENT everywhere it appears.

        Scoped to content, not to a source: every event carrying this hash
        loses its payload, whoever wrote it. See `redaction_scope` for the
        blast radius, and `retract_source` to silence a single actor instead.
        """
        # Reject a bad hash at the boundary, before appending anything (M4):
        # the argument flows into `cas_dir / f"{payload_hash}.json"`, so a value
        # with `..` in it would unlink a file outside the store.
        if not is_payload_hash(payload_hash):
            raise ValueError(
                "payload_hash must be a 64-char lowercase sha256 hex digest, "
                f"got {payload_hash!r}")
        scope = self.redaction_scope(payload_hash)
        if scope["shared"]:
            apply_mod.diagnostic(self.index, 0, "redaction_shared_payload", scope)
            self._health_events.append({"kind": "redaction_shared_payload", **scope})
        payload = {"payload_hash": payload_hash}
        ev = self.ledger.append("redaction", "human:operator", payload)
        apply_mod.apply_event(self.index, ev, payload)
        self.ledger.delete_payload(payload_hash)
        # Redaction implies exclusion: downstream state is recomputed without it.
        self.index.reset()
        self._refold()
        self.closure()
        return ev.seq

    # ── claims & settlement (§3.8) ───────────────────────────────────────────
    def claim(self, stmt: Mapping[str, Any], frame: str, criterion: str,
              due: int) -> str:
        # Validate at the boundary, before the ledger append (C3): the index's
        # claims.frame CHECK would otherwise reject the event only after it was
        # already in the chain, bricking every future replay.
        if frame not in ("internal", "external"):
            raise ValueError(
                f"claim frame must be 'internal' or 'external', got {frame!r}")
        settlement, verifier_id = self._triage(stmt, criterion)
        if settlement == "unsettleable":
            return REFUSED
        pred = self.predict(stmt, budget=10_000)
        claim_id = f"claim:{self.ledger.seq() + 1}"
        payload = {
            "claim_id": claim_id,
            "stmt": {"pred": stmt["pred"], "args": list(stmt["args"])},
            "frame": frame, "settlement": settlement, "criterion": criterion,
            "verifier_id": verifier_id, "due": due,
            "predicted_p": pred.p, "ci_lo": pred.ci[0], "ci_hi": pred.ci[1],
            "model_snapshot": pred.snapshot_id,
            "predictor_class": calibration_mod.DEFAULT_PREDICTOR_CLASS,
            "certainty_class": self._certainty_class(settlement, verifier_id),
            "sensitivity": pred.sensitivity,
        }
        ev = self.ledger.append("claim", "agent:planner", payload)
        apply_mod.apply_event(self.index, ev, payload)
        self.index.commit()
        return claim_id

    def _triage(self, stmt: Mapping[str, Any],
                criterion: str) -> tuple[str, Optional[str]]:
        if criterion in ("", None, "none", "unsettleable"):
            return "unsettleable", None
        # An explicitly named, registered oracle outranks the closure. §3.8 says
        # plainly that self-consistency is not truth: if a caller has gone to the
        # trouble of constructing an external verifier, settling the claim
        # internally instead would inflate the entailed share of the
        # external-settled ratio that the same section tells us to watch.
        if self.index.one("SELECT id FROM oracles WHERE id=?", (criterion,)):
            return "tool_decidable", criterion
        derived = self.derive(stmt, budget=10_000)
        if derived.status == "proof":
            return "entailed", "verifier:closure"
        if criterion.startswith("tool:") or criterion.startswith("verifier:"):
            return "tool_decidable", criterion
        return "observation_pending", f"verifier:observation({criterion})"

    def _certainty_class(self, settlement: str, verifier_id: Optional[str]) -> str:
        if settlement == "entailed":
            return "certain"
        row = self.index.one("SELECT kind FROM oracles WHERE id=?", (verifier_id,))
        if row is not None and row["kind"] == "deterministic_total":
            return "certain"
        return "estimated"

    def register_oracle(self, oracle_id: str, kind: str, impl_ref: str,
                        code_hash: str, env_hash: str,
                        authority: str = "human:operator") -> None:
        """Register a pre-validated external oracle, through the ledger.

        Synthesized verifiers earn their oracle row through the gate's sandbox
        step; an externally validated oracle enters as a human-authorized
        admission event instead — same append path, same replayability. A
        direct SQLite write here would vanish on replay (the test that caught
        exactly that lives in tests/unit/test_two_coin.py).
        """
        payload = {
            "candidate_id": f"cand:oracle:{oracle_id}",
            "candidate_kind": "verifier", "status": "admitted",
            "failing_step": None, "reason": "external registration",
            "body": {"oracle_id": oracle_id, "kind": kind, "entry": impl_ref,
                     "code_hash": code_hash, "env_hash": env_hash},
        }
        ev = self.ledger.append("admission", authority, payload)
        apply_mod.apply_event(self.index, ev, payload)
        self.index.commit()

    def resolve(self, claim_id: str, outcome: bool,
                verifier_code_hash: str = "", env_hash: str = "") -> int:
        row = self.index.one("SELECT * FROM claims WHERE id=?", (claim_id,))
        if row is None:
            raise KeyError(f"unknown claim {claim_id!r}")
        predicted = float(row["predicted_p"] if row["predicted_p"] is not None else 0.5)
        sensitivity = {
            r["fact_id"] or r["rule_id"]: float(r["sensitivity"])
            for r in self.index.query(
                "SELECT rule_id, fact_id, sensitivity FROM proof_steps WHERE claim_id=?",
                (claim_id,))
            if (r["fact_id"] or r["rule_id"])
        }
        oracle_kind = None
        if row["verifier_id"]:
            orow = self.index.one("SELECT kind FROM oracles WHERE id=?",
                                  (row["verifier_id"],))
            oracle_kind = orow["kind"] if orow else None
        payload = {
            "claim_id": claim_id, "outcome": bool(outcome),
            "surprisal": calibration_mod.surprisal(predicted, outcome),
            "sensitivity": sensitivity,          # full vector, always logged (§4.4)
            "blame_target": self._blame_target(sensitivity),
            "verifier_code_hash": verifier_code_hash,
            "env_hash": env_hash,
            "oracle_kind": oracle_kind,
        }
        ev = self.ledger.append("resolution", "verifier:harness", payload)
        apply_mod.apply_event(self.index, ev, payload)
        self.index.commit()
        return ev.seq

    def _blame_target(self, sensitivity: Mapping[str, float]) -> Optional[str]:
        """§4.4: integer blame to the argmax-sensitivity *eligible* component."""
        best, best_s = None, -1.0
        for comp, s in sorted(sensitivity.items()):
            if not self._blame_eligible(comp):
                continue
            if s > best_s:
                best, best_s = comp, s
        return best

    def _blame_eligible(self, component: str) -> bool:
        if component.startswith("rule:"):
            row = self.index.one("SELECT numeric FROM rules WHERE id=?", (component,))
            return row is not None and row["numeric"] != "frozen"
        row = self.index.one("SELECT kind, numeric FROM facts WHERE id=?", (component,))
        if row is None:
            return False
        if row["kind"] == "definitional" or row["numeric"] == "frozen":
            return False
        # exact-derived facts are blame-ineligible; base admitted facts are not.
        atom = self.index.one(
            "SELECT admitted_by_event FROM facts WHERE id=?", (component,))
        return atom is not None

    # ── reads ────────────────────────────────────────────────────────────────
    def recall(self, query: str, budget: int,
               actor: str = "agent:reader") -> list[dict[str, Any]]:
        """Side-stream logged; no effect on any committed number (I2)."""
        return self.retriever.recall(query, budget, actor)

    def derive(self, goal: Mapping[str, Any], budget: int) -> DeriveOutcome:
        pred = goal["pred"]
        args = tuple(facts_mod.canonical_args(self.index, pred, list(goal["args"])))
        clo = self.closure()
        rules = self._rules()
        outcome = closure_mod.backward(pred, args, clo, rules, budget)
        if outcome.proved and outcome.proof is not None:
            # Only genuinely base atoms may be cited without a rule (§3.6): a
            # rule-derived conclusion must present its rule to the kernel.
            kernel_mod.check(outcome.proof, set(clo.base), {r.id: r for r in rules})
            quality = kernel_mod.quality(outcome.proof, self._flagged_facts(),
                                         self._narrow_facts())
            return DeriveOutcome("proof", outcome.proof, True, quality)
        if not outcome.exhausted:
            return DeriveOutcome("budget_exceeded", None, False)
        return DeriveOutcome("not_entailed", None, True)

    def _rules(self) -> list[closure_mod.Rule]:
        return [closure_mod.Rule.parse(r["id"], r["head_json"], r["body_json"],
                                       int(r["specificity"]))
                for r in self.index.query(
                    "SELECT * FROM rules WHERE structural='admitted' "
                    "ORDER BY specificity DESC, id")]

    def _flagged_facts(self) -> set[str]:
        return {r["id"] for r in self.index.query(
            "SELECT id FROM facts WHERE dispersion_flag=1")}

    def _narrow_facts(self) -> set[str]:
        return {r["id"] for r in self.index.query(
            "SELECT id FROM facts WHERE breadth_class='narrow'")}

    def conjecture(self, goal: Mapping[str, Any],
                   sim_budget: float) -> list[dict[str, Any]]:
        signatures = self._signatures()
        neighbours = conjecture_mod.neighbourhood(goal["pred"], signatures, sim_budget)
        known = {(p, a) for p, a in self.closure().atoms}
        return conjecture_mod.conjectures(goal, neighbours, known)

    def _signatures(self) -> dict[str, dict[str, float]]:
        firings: list[tuple[str, int, str]] = []
        for rule in self._rules():
            for pos, arg in enumerate(rule.head.args):
                if not closure_mod.is_var(arg):
                    firings.append((rule.id, pos, rule.head.pred))
            for lit in rule.body:
                for pos, arg in enumerate(lit.args):
                    firings.append((rule.id, pos, lit.pred))
        preds = {r["pred"] for r in self.index.query("SELECT DISTINCT pred FROM facts")}
        out: dict[str, dict[str, float]] = {}
        for pred in sorted(preds):
            symbols = facts_mod.alias_closure(self.index, pred)
            co_derived = [p for p, _ in self.closure().atoms if p in symbols]
            out[pred] = conjecture_mod.signature(pred, firings, co_derived)
        return out

    # ── prediction (§3.9) ────────────────────────────────────────────────────
    def predict(self, stmt: Mapping[str, Any], budget: int) -> PredictOutcome:
        out = self._predict_with(self.index, self.closure(), stmt, budget,
                                 self._calib, self.ledger.head())
        self._last_rejection_rate = out.rejection_rate
        return out

    def predict_at(self, stmt: Mapping[str, Any], snapshot_id: str) -> PredictOutcome:
        """I8: re-run a prediction at its recorded ledger position, pinned map."""
        snap = calibration_mod.parse_snapshot(snapshot_id)
        head = snap["ledger_head"]
        if head == self.ledger.head():
            return self.predict(stmt, budget=10_000)
        tmp_dir = Path(tempfile.mkdtemp(prefix="candor-snapshot-"))
        try:
            tmp_index = Index(tmp_dir / "index.sqlite3")
            tmp_index.open()
            events = list(self.ledger.read_all())
            cutoff = None
            for ev in events:
                if ev.hash == head:
                    cutoff = ev.seq
                    break
            if cutoff is None and head != "0" * 64:
                raise KeyError("snapshot ledger head is not in this chain")
            redacted = self._redacted_payloads()
            for ev in events:
                if cutoff is not None and ev.seq > cutoff:
                    break
                payload = (None if ev.payload_hash in redacted
                           else self.ledger.payload(ev.payload_hash))
                apply_mod.apply_event(tmp_index, ev, payload)
            clo = apply_mod.rebuild_closure(tmp_index)
            tmp_index.commit()
            calib = calibration_mod.IsotonicMap.from_json(snap.get("calib_map"))
            if calib.hash != snap["calib_map_hash"]:
                calib = self._calib if self._calib.hash == snap["calib_map_hash"] \
                    else calibration_mod.IsotonicMap()
            out = self._predict_with(tmp_index, clo, stmt, 10_000, calib, head)
            tmp_index.close()
            return out
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _predict_with(self, idx: Index, clo: closure_mod.Closure,
                      stmt: Mapping[str, Any], budget: int,
                      calib: calibration_mod.IsotonicMap,
                      ledger_head: str) -> PredictOutcome:
        pred = stmt["pred"]
        args = facts_mod.canonical_args(idx, pred, list(stmt["args"]))
        problem = self._build_problem(idx, clo, pred, args)
        outcome = predict_mod.run(problem, budget)
        p = calib.apply(outcome.p)
        ci = (calib.apply(outcome.ci[0]), calib.apply(outcome.ci[1]))
        snapshot = calibration_mod.snapshot_id(ledger_head, calib.hash)
        return PredictOutcome(p, ci, dict(outcome.channels),
                              dict(outcome.sensitivity), outcome.mpe,
                              outcome.caveats, snapshot, outcome.rejection_rate)

    def _build_problem(self, idx: Index, clo: closure_mod.Closure, pred: str,
                       args: list[Any]) -> predict_mod.Problem:
        fid = facts_mod.lookup(idx, pred, args) or fact_key(pred, args)
        atom = (pred, tuple(args))
        supports = [s for s in clo.support.get(atom, []) if s]
        dnf: list[frozenset[str]] = []
        for support in supports:
            conj = frozenset(s for s in support
                             if idx.one("SELECT id FROM facts WHERE id=?", (s,)))
            if conj:
                dnf.append(conj)
        if not dnf:
            dnf = [frozenset({fid})]

        needed = {f for conj in dnf for f in conj}
        cons = [constraints_mod.parse(r) for r in idx.query("SELECT * FROM constraints")]
        groups: list[list[str]] = []
        if cons:
            all_facts = [(r["pred"], json.loads(r["args_json"]), r["id"])
                         for r in idx.query("SELECT id, pred, args_json FROM facts")]
            for key, members in sorted(constraints_mod.groups_for(cons, all_facts).items()):
                if needed & set(members):
                    groups.append(sorted(members))
                    needed |= set(members)

        states: dict[str, predict_mod.FactState] = {}
        caveats: set[str] = set()
        sources: dict[str, int] = {}
        confusion: dict[str, tuple[int, int, int, int]] = {}
        response_lr: dict = {}
        for target in sorted(needed):
            states[target] = self._fact_state(idx, target)
            for actor, vote, grade, _ in states[target].votes:
                if actor not in confusion:
                    confusion[actor] = reliability_mod.confusion(idx, actor)
                if grade > 0 and (actor, vote, grade) not in response_lr:
                    response_lr[(actor, vote, grade)] = \
                        reliability_mod.response_log_lr(idx, actor, bool(vote), grade)
            src = idx.one(
                "SELECT source_ref FROM events WHERE seq=("
                "SELECT admitted_by_event FROM facts WHERE id=?)", (target,))
            if src is not None and src["source_ref"]:
                sources[src["source_ref"]] = sources.get(src["source_ref"], 0) + 1
            if states[target].flagged:
                caveats.add("under_specified")
            if states[target].narrow:
                caveats.add("narrow_breadth")
        if any(v >= 2 for v in sources.values()) and len(needed) >= 2:
            caveats.add("shared_provenance")
        return predict_mod.Problem(fid, dnf, states, groups, caveats, confusion,
                                   response_lr, self._actor_discounts(idx))

    def _fact_state(self, idx: Index, fact_id: str) -> predict_mod.FactState:
        row = idx.one("SELECT stmt_type, dispersion_flag, breadth_class FROM facts "
                      "WHERE id=?", (fact_id,))
        stmt_type = row["stmt_type"] if row else "crisp"
        composed = counts_mod.compose(idx, [fact_id])
        epi, alea = counts_mod.posterior_params(composed, stmt_type)
        if row is None:
            # Never admitted: no gate evidence, so no admission pseudocount either.
            epi = (1.0 + composed.epi_a, 1.0 + composed.epi_b)
        votes: tuple = ()
        if stmt_type == "crisp":
            # v0.3 Δ1: attributed votes supersede the epi Beta for crisp facts —
            # same events at finer grain. Ordered deterministically so the
            # composition is independent of insertion order.
            vrows = idx.query(
                "SELECT actor, outcome, grade, context_sig FROM observations "
                "WHERE fact_id=? AND channel='epi' "
                "ORDER BY actor, context_sig, event_seq", (fact_id,))
            votes = tuple((r["actor"], int(r["outcome"]), int(r["grade"]),
                           r["context_sig"]) for r in vrows)
        return predict_mod.FactState(
            fact_id=fact_id, stmt_type=stmt_type, epi=epi, alea=alea,
            pinned_negative=facts_mod.is_negatively_pinned(idx, fact_id),
            flagged=bool(row["dispersion_flag"]) if row else False,
            narrow=(row["breadth_class"] == "narrow") if row else False,
            votes=votes)

    # ── introspection ────────────────────────────────────────────────────────
    def raw_counts(self, fact_id: str) -> dict[tuple[str, str], tuple[int, int]]:
        return counts_mod.raw_counts(self.index, self._alias_ids(fact_id))

    def composed_counts(self, fact_id: str) -> counts_mod.Composed:
        return counts_mod.compose(self.index, self._alias_ids(fact_id))

    def _alias_ids(self, fact_id: str) -> list[str]:
        """Union-at-read over the alias closure. Counts are never merged (I11)."""
        row = self.index.one("SELECT pred, args_json FROM facts WHERE id=?", (fact_id,))
        if row is None:
            return [fact_id]
        args = json.loads(row["args_json"])
        return facts_mod.resolve_ids(self.index, row["pred"], args)

    def fact_id_for(self, stmt: Mapping[str, Any]) -> Optional[str]:
        return facts_mod.lookup(self.index, stmt["pred"], list(stmt["args"]))

    def _actor_discounts(self, idx: Optional[Index] = None) -> dict[str, float]:
        """Operator-set trust discounts for the crisp vote path.

        Only *explicit* overrides count — read from the `reliability_overrides`
        table, which is folded from `reliability` ledger events (never from
        settlements). Learned reliability lives in the confusion ledger and
        already speaks through the two-coin LR; folding E[rel] in as well would
        double-count the same settlements. An untouched store returns {} and
        composes byte-identically to before this existed.

        Reads from `idx` so a snapshot replay (predict_at) tempers with the
        overrides as of its own ledger position, not the live store's.
        """
        target = idx if idx is not None else self.index
        out: dict[str, float] = {}
        for row in target.query(
                "SELECT actor, frame, rel_a, rel_b FROM reliability_overrides"):
            a, b = float(row["rel_a"]), float(row["rel_b"])
            if row["frame"] == reliability_mod.FACT_FRAME and (a + b) > 0:
                out[row["actor"]] = a / (a + b)
        return out

    def set_reliability(self, actor: str, frame: str, a: float, b: float,
                        authority: str = "human:operator") -> None:
        """Operator override on a source's trust. Moves BOTH channels.

        On frequency facts it discounts the aleatoric trial contribution; on
        crisp facts it tempers the source's vote evidence (v0.3 Δ1 replaced the
        epi Beta with two-coin votes, and before this the lever silently missed
        them — see bench/CLAIMS_HARDENING.md F2).

        A first-class ledger event, not a side file: it folds in sequence order
        like everything else, so a ledger-only rebuild reproduces it (I1) and it
        composes with settlements at its true position (I3).
        """
        # Validate before appending (C3): actor_reliability's frame CHECK would
        # otherwise reject the event only after it is already in the chain,
        # bricking every future replay.
        if frame not in ("internal", "external"):
            raise ValueError(
                f"reliability frame must be 'internal' or 'external', got {frame!r}")
        payload = {"actor": actor, "frame": frame,
                   "rel_a": float(a), "rel_b": float(b)}
        ev = self.ledger.append("reliability", authority, payload)
        apply_mod.apply_event(self.index, ev, payload)
        self.index.commit()

    def why(self, obj_id: str) -> dict[str, Any]:
        row = self.index.one("SELECT * FROM facts WHERE id=?", (obj_id,))
        if row is None:
            return {"id": obj_id, "found": False}
        args = json.loads(row["args_json"])
        raw = self.raw_counts(obj_id)
        composed = self.composed_counts(obj_id)
        keys = self._context_keys(obj_id)
        derivation = self.derive({"pred": row["pred"], "args": args}, budget=10_000)
        return {
            "id": obj_id, "found": True, "pred": row["pred"], "args": args,
            "stmt_type": row["stmt_type"], "kind": row["kind"],
            "structural": row["structural"], "numeric": row["numeric"],
            "dispersion_flag": bool(row["dispersion_flag"]),
            "breadth_class": row["breadth_class"],
            "valid_from": row["valid_from"], "valid_to": row["valid_to"],
            "gate_run": self._gate_run_for(obj_id),
            "span_provenance": self._spans_for(obj_id),
            "raw_counts": {f"{a}|{c}": [n, k] for (a, c), (n, k) in raw.items()},
            "composed_counts": {"epi_a": composed.epi_a, "epi_b": composed.epi_b,
                                "alea_n": composed.alea_n, "alea_k": composed.alea_k},
            "source_diversity": keys,
            "derivation": {"status": derivation.status, "quality": derivation.quality,
                           "proof": derivation.proof},
        }

    def _gate_run_for(self, fact_id: str) -> Optional[str]:
        row = self.index.one(
            "SELECT gate_run_id FROM candidates WHERE status='admitted' AND kind='fact' "
            "AND body_json LIKE ? ORDER BY event_seq DESC LIMIT 1",
            (f"%{fact_id}%",))
        if row is not None:
            return row["gate_run_id"]
        row = self.index.one(
            "SELECT payload_hash FROM events WHERE seq=(SELECT admitted_by_event "
            "FROM facts WHERE id=?)", (fact_id,))
        if row is None:
            return None
        payload = self.ledger.payload(row["payload_hash"]) or {}
        return payload.get("gate_run_id")

    def _spans_for(self, fact_id: str) -> list[str]:
        rows = self.index.query(
            "SELECT DISTINCT span_ref FROM candidates WHERE span_ref IS NOT NULL "
            "AND kind='fact' AND body_json LIKE ?", (f"%{fact_id}%",))
        return [r["span_ref"] for r in rows]

    def _context_keys(self, fact_id: str) -> dict[str, int]:
        rows = self.index.query(
            "SELECT oc.key AS key, COUNT(DISTINCT oc.value) AS distinct_values "
            "FROM obs_context oc JOIN observations o ON o.event_seq = oc.event_seq "
            "WHERE o.fact_id=? GROUP BY oc.key ORDER BY oc.key", (fact_id,))
        return {r["key"]: int(r["distinct_values"]) for r in rows}

    def questions(self, scope: str = "*") -> list[dict[str, Any]]:
        sql = "SELECT * FROM open_questions WHERE status='open'"
        params: tuple = ()
        if scope not in ("*", "", None):
            sql += " AND kind=?"
            params = (scope,)
        return [{
            "id": r["id"], "kind": r["kind"], "target_kind": r["target_kind"],
            "target_id": r["target_id"],
            "residual_partition": json.loads(r["residual_partition"] or "{}"),
            "dispersion_stat": r["dispersion_stat"],
            "ruled_out": json.loads(r["ruled_out_json"] or "[]"),
            "suggested_measurement": r["suggested_measurement"],
            "status": r["status"],
        } for r in self.index.query(sql + " ORDER BY id", params)]

    def events_since(self, cursor: int,
                     kinds: Optional[set[str]] = None) -> list[dict[str, Any]]:
        rows = self.index.query(
            "SELECT seq, ts, kind, actor, payload_hash, source_ref, context_sig, hash "
            "FROM events WHERE seq > ? ORDER BY seq", (int(cursor),))
        out = []
        for r in rows:
            if kinds and r["kind"] not in kinds:
                continue
            out.append({"id": int(r["seq"]), "seq": int(r["seq"]), "ts": int(r["ts"]),
                        "kind": r["kind"], "actor": r["actor"],
                        "payload_hash": r["payload_hash"],
                        "source_ref": r["source_ref"],
                        "context_sig": r["context_sig"], "hash": r["hash"]})
        return out

    def health(self) -> dict[str, Any]:
        settled = self.index.query(
            "SELECT frame, settlement, COUNT(*) AS c FROM claims "
            "WHERE resolved_ts IS NOT NULL GROUP BY frame, settlement")
        total = sum(int(r["c"]) for r in settled) or 0
        external = sum(int(r["c"]) for r in settled if r["frame"] == "external")
        breadth = {r["breadth_class"] or "unclassified": int(r["c"])
                   for r in self.index.query(
                       "SELECT breadth_class, COUNT(*) AS c FROM facts "
                       "GROUP BY breadth_class")}
        quota_rows = self.index.query(
            "SELECT actor, kind, used FROM quota_usage ORDER BY actor, kind")
        diagnostics = [{"kind": r["kind"], **json.loads(r["detail_json"])}
                       for r in self.index.query(
                           "SELECT kind, detail_json FROM diagnostics "
                           "ORDER BY seq DESC LIMIT 100")]
        return {
            "calibration": calibration_mod.report(self.index),
            "external_settled_ratio": (external / total) if total else None,
            "invariants": [dict(r) for r in self.index.query("SELECT * FROM invariants")],
            "breadth_distribution": breadth,
            "queue_depth": int(self.index.one(
                "SELECT COUNT(*) AS c FROM candidates WHERE status='pending'")["c"]),
            "quota_usage": [{"actor": r["actor"], "kind": r["kind"],
                             "used": int(r["used"])} for r in quota_rows],
            "constraint_rejection_rate": self._last_rejection_rate,
            "events": self._health_events + diagnostics,
            "ledger_head": self.ledger.head(),
            "retrieval_log_size": self.retrieval_log.count(),
        }

    _last_rejection_rate: float = 0.0

    # ── test-only fault injection (§6.1) ─────────────────────────────────────
    def corrupt(self, what: str, arg: str = "") -> None:
        if what == "torn_tail":
            self.ledger.append_raw_line('{"seq": 999999, "ts": 0, "kind": "obser')
        elif what == "drop_index":
            self.index.drop()
        elif what == "delete_payload":
            self.ledger.delete_payload(arg)
        else:
            raise ValueError(f"unknown fault {what!r}")
