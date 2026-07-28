"""The v2 observer panel — twelve actors, sparse assignment (suite v2, T1).

Observer ids are the pre-registered contract (generate_suite_v2.OBSERVER_IDS);
this module supplies the implementations. Context discipline matters for Δ2:
`ctx` records the *evidence* an observation was made under (retriever, k, top
entries) and never the model — the model is the actor, already keyed. Two LLMs
reading the same top-k therefore share a context_sig and compose
sub-additively; the degenerate actors carry no context and compose as
singletons.

Ground truth never reaches an observer.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from . import ollama
from .baseline import EmbeddingStore
from .observers import JUDGE_PROMPT, JUDGE_SYSTEM, NUMBER, _norm

# id -> (mechanism, model, k)
CONFIG: dict[str, tuple[str, Optional[str], int]] = {
    "agent:qwen27@k8": ("llm", "qwen3.6:27b-q8_0", 8),
    "agent:qwen27@k3": ("llm", "qwen3.6:27b-q8_0", 3),
    "agent:qwen9@k8": ("llm", "qwen3.5:9b", 8),
    "agent:qwen9@k3": ("llm", "qwen3.5:9b", 3),
    "agent:qwen35moe@k8": ("llm", "qwen3.6:35b-a3b-q8_0", 8),
    "agent:mistral24@k8": ("llm", "mistral-small3.2:24b-instruct-2506-q8_0", 8),
    "tool:exact@k8": ("exact", None, 8),
    "tool:exact@k3": ("exact", None, 3),
    "tool:span@k8": ("span", None, 8),
    "agent:optimist": ("optimist", None, 0),
    "agent:pessimist": ("pessimist", None, 0),
    "agent:coin": ("coin", None, 0),
}

WORD5 = re.compile(r"[A-Za-z][A-Za-z0-9_-]{4,}")


@dataclass
class Vote:
    observer: str
    outcome: bool
    ctx: dict[str, str]
    confidence: Optional[float] = None   # v0.4 Δ6: graded when the judge can


class PanelObservers:
    def __init__(self, store: EmbeddingStore) -> None:
        self.store = store
        self._evidence_cache: dict[tuple[str, int], tuple[str, dict[str, str]]] = {}

    # ── shared evidence per (claim, k): same k -> same context_sig (Δ2) ─────
    def evidence(self, claim_text: str, k: int) -> tuple[str, dict[str, str]]:
        key = (claim_text, k)
        if key not in self._evidence_cache:
            hits = self.store.search(claim_text, k)
            blob = "\n\n".join(f"[{eid}]\n{self.store.texts.get(eid, '')[:1200]}"
                               for eid, _ in hits)
            ctx = {"retriever": "bge-m3", "k": str(k),
                   "top": ",".join(eid for eid, _ in hits[:3])}
            self._evidence_cache[key] = (blob, ctx)
        return self._evidence_cache[key]

    # ── mechanisms ──────────────────────────────────────────────────────────
    def _llm(self, claim_text: str, context: str, model: str) -> float:
        """v0.4 Δ6: elicit a probability, not a boolean — a binary vote throws
        away most of what the judge knows (FINDINGS F8)."""
        from .baseline import ELICIT_PROMPT, ELICIT_SYSTEM
        raw = ollama.generate(
            ELICIT_PROMPT.format(context=context[:24000], claim=claim_text),
            system=ELICIT_SYSTEM, num_predict=120, model=model)
        parsed = ollama.extract_json(raw)
        try:
            return min(1.0, max(0.0, float(parsed.get("p", 0.5)))) \
                if isinstance(parsed, dict) else 0.5
        except (TypeError, ValueError):
            return 0.5

    def _exact(self, claim_text: str, context: str) -> bool:
        ctx = _norm(context)
        numbers = NUMBER.findall(claim_text)
        if numbers:
            return all(n in ctx for n in numbers)
        words = WORD5.findall(claim_text)
        return any(_norm(w) in ctx for w in words) if words else True

    def _span(self, claim_text: str, context: str) -> bool:
        ctx = _norm(context)
        words = [w.lower() for w in WORD5.findall(claim_text)]
        if not words:
            return True
        present = sum(1 for w in words if w in ctx)
        return present / len(words) >= 0.6

    @staticmethod
    def _coin(claim_id: str) -> bool:
        return bool(int(hashlib.sha256(f"coin|{claim_id}".encode())
                        .hexdigest()[0], 16) & 1)

    # ── one (claim, observer) -> one attributed vote ────────────────────────
    def observe(self, claim_id: str, claim_text: str, observer: str) -> Vote:
        mechanism, model, k = CONFIG[observer]
        if mechanism in ("llm", "exact", "span"):
            context, ctx = self.evidence(claim_text, k)
            if mechanism == "llm":
                p = self._llm(claim_text, context, model)
                return Vote(observer, p >= 0.5, ctx, confidence=p)
            if mechanism == "exact":
                outcome = self._exact(claim_text, context)
            else:
                outcome = self._span(claim_text, context)
            return Vote(observer, outcome, ctx)
        if mechanism == "optimist":
            return Vote(observer, True, {})
        if mechanism == "pessimist":
            return Vote(observer, False, {})
        return Vote(observer, self._coin(claim_id), {})

    # ── batched by model so the box loads each set of tensors once ──────────
    def observe_suite(self, claims: Sequence[dict],
                      log=print) -> dict[str, list[Vote]]:
        votes: dict[str, list[Vote]] = {c["claim_id"]: [] for c in claims}
        tasks: dict[Optional[str], list[tuple[dict, str]]] = {}
        for claim in claims:
            for observer in claim["panel"]:
                model = CONFIG[observer][1]
                tasks.setdefault(model, []).append((claim, observer))
        # deterministic model order: cheap tools first, then models by name
        for model in sorted(tasks, key=lambda m: (m is not None, m or "")):
            batch = tasks[model]
            label = model or "tools/degenerate"
            for n, (claim, observer) in enumerate(batch, 1):
                votes[claim["claim_id"]].append(
                    self.observe(claim["claim_id"], claim["text"], observer))
                if n % 100 == 0 or n == len(batch):
                    log(f"  {label}: {n}/{len(batch)}")
        return votes
