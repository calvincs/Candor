"""Observers: fallible judges whose votes reach CANDOR as `observation` events.

This is the part of §6.8 that exercises what CANDOR actually is. A claim is not
answered by asking one good model how sure it is; it is answered by several
attributed, fallible observers whose reliability is *learned from settlements*
and applied as a read-time discount (§3.12, I11).

Four observers with genuinely different failure modes:

  tool:exact     deterministic lexical check — does the claim's distinguishing
                 value appear in the retrieved evidence? Strong on the numeric
                 and entity perturbations, blind to paraphrase.
  agent:llm_big  laguna-s-2.1 judging from retrieved context. Same information
                 the baseline gets, cast as a single vote.
  agent:llm_small a much smaller model, same prompt. Noisier, cheaper, and
                 wrong in different places.
  agent:optimist a deliberately biased observer that almost always says true.
                 Present so the reliability discount has something real to
                 catch; without a bad actor the machinery is untested.

Ground truth never reaches an observer. Only the trusted settlement path sees
it, and only for the training split.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from . import ollama
from .baseline import EmbeddingStore

SMALL_MODEL = "qwen3.5:9b"

JUDGE_SYSTEM = (
    "You judge whether a statement is supported by retrieved notes. "
    "Answer strictly from the notes. Reply with one JSON object and nothing else."
)

JUDGE_PROMPT = """Retrieved notes:
\"\"\"
{context}
\"\"\"

STATEMENT: {claim}

Is the STATEMENT true according to the notes? Reply as JSON:
{{"true": true or false}}"""

NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass
class Observation:
    actor: str
    outcome: bool
    ctx: dict[str, str]


class Observers:
    def __init__(self, store: EmbeddingStore, k: int = 8) -> None:
        self.store = store
        self.k = k

    # ── shared retrieval, so every observer sees the same evidence ──────────
    def evidence(self, claim_text: str) -> tuple[str, list[str]]:
        hits = self.store.search(claim_text, self.k)
        blob = "\n\n".join(f"[{eid}]\n{self.store.texts.get(eid,'')[:1200]}"
                           for eid, _ in hits)
        return blob, [eid for eid, _ in hits]

    # ── the four observers ─────────────────────────────────────────────────
    def exact(self, claim_text: str, context: str) -> bool:
        """Every number in the claim must appear in the retrieved evidence.

        Deterministic, reproducible, and blind in an interesting way: it cannot
        see paraphrase, so it fires false negatives on true claims whose wording
        drifted from the source.
        """
        ctx = _norm(context)
        numbers = NUMBER.findall(claim_text)
        if not numbers:
            # no numeric handle; fall back to the rarest long token
            words = [w for w in re.findall(r"[A-Za-z0-9_.:/-]{6,}", claim_text)]
            if not words:
                return True
            return any(_norm(w) in ctx for w in words)
        return all(n in ctx for n in numbers)

    def llm(self, claim_text: str, context: str, model: str) -> bool:
        raw = ollama.generate(
            JUDGE_PROMPT.format(context=context[:24000], claim=claim_text),
            system=JUDGE_SYSTEM, num_predict=80, model=model)
        obj = ollama.extract_json(raw)
        if isinstance(obj, dict) and "true" in obj:
            return bool(obj["true"])
        return "true" in raw.lower()[:200]

    def optimist(self, claim_text: str, context: str) -> bool:
        """Biased-to-yes observer. Wrong on roughly every false claim."""
        return not claim_text.strip().lower().startswith("no ")

    # ── one claim -> four attributed observations ──────────────────────────
    def observe_all(self, claim_text: str) -> list[Observation]:
        context, hits = self.evidence(claim_text)
        ctx_meta = {"retrieval": "bge-m3", "k": str(self.k),
                    "top_entry": hits[0] if hits else "none"}
        return [
            Observation("tool:exact", self.exact(claim_text, context), ctx_meta),
            Observation("agent:llm_big",
                        self.llm(claim_text, context, ollama.JUDGE_MODEL), ctx_meta),
            Observation("agent:llm_small",
                        self.llm(claim_text, context, SMALL_MODEL), ctx_meta),
            Observation("agent:optimist", self.optimist(claim_text, context),
                        ctx_meta),
        ]

    # ── batched by model, because only one fits in VRAM at a time ──────────
    def observe_batch(self, claim_texts: Sequence[str], workers: int = 1,
                      on_phase: Optional[Callable[[str, int, int], None]] = None
                      ) -> list[list[Observation]]:
        """Same four observations per claim, but grouped into per-model passes.

        Interleaving two large models per claim makes the server evict and
        reload weights on every call, which costs minutes each. Running one
        model to completion before touching the next costs two swaps in total.
        """
        def note(phase: str, done: int, total: int) -> None:
            if on_phase and (done % 25 == 0 or done == total):
                on_phase(phase, done, total)

        contexts: list[tuple[str, list[str]]] = []
        for i, text in enumerate(claim_texts, 1):
            contexts.append(self.evidence(text))
            note("evidence", i, len(claim_texts))

        big = ollama.parallel(
            lambda pair: self.llm(pair[0], pair[1][0], ollama.JUDGE_MODEL),
            list(zip(claim_texts, contexts)), workers=workers,
            on_progress=lambda d, t: note("llm_big", d, t))
        small = ollama.parallel(
            lambda pair: self.llm(pair[0], pair[1][0], SMALL_MODEL),
            list(zip(claim_texts, contexts)), workers=workers,
            on_progress=lambda d, t: note("llm_small", d, t))

        out: list[list[Observation]] = []
        for i, text in enumerate(claim_texts):
            context, hits = contexts[i]
            meta = {"retrieval": "bge-m3", "k": str(self.k),
                    "top_entry": hits[0] if hits else "none"}
            out.append([
                Observation("tool:exact", self.exact(text, context), meta),
                Observation("agent:llm_big", bool(big[i]) if not isinstance(
                    big[i], dict) else False, meta),
                Observation("agent:llm_small", bool(small[i]) if not isinstance(
                    small[i], dict) else False, meta),
                Observation("agent:optimist", self.optimist(text, context), meta),
            ])
        return out
