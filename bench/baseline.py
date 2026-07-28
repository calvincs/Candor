"""The fair baseline for §6.8 — the boring alternative, built to actually win.

Two components, matching the two arms of the honest test:

  EmbeddingStore   dense retrieval over the same corpus with bge-m3. This is
                   the comparator §6.8 names for nDCG@k / recall@k.
  ElicitedRag      §6.8 is explicit that an embedding store cannot emit
                   probabilities, so the calibration comparator is *RAG with
                   elicited probabilities plus isotonic fitting* — retrieve
                   top-k, ask the LLM for a probability, then fit isotonic on
                   the training split and apply it to held out.

A weak baseline would make CANDOR look good and teach us nothing, so this is
built with the same care as the system under test: a strong embedding model, a
real prompt, and the isotonic correction the spec grants it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from . import ollama
from .corpus import Entry

TOP_K = 8

ELICIT_SYSTEM = (
    "You judge whether a statement is supported by retrieved notes. "
    "You are well calibrated: you output the probability that the statement is "
    "true, and you are willing to say 0.5 when the notes do not settle it. "
    "You reply with one JSON object and nothing else."
)

ELICIT_PROMPT = """Retrieved notes:
\"\"\"
{context}
\"\"\"

STATEMENT: {claim}

Considering only the notes above, what is the probability that the STATEMENT is
true? Reply as JSON: {{"p": <number between 0 and 1>, "why": "<10 words>"}}"""


from operator import mul


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(map(mul, a, b))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return _dot(a, b) / (na * nb) if na and nb else 0.0


@dataclass
class EmbeddingStore:
    """Plain dense retrieval. No reranking, no tricks — and no handicap.

    Vectors are unit-normalized once at construction, so a search is a dot
    product rather than a full cosine. Recomputing every entry's norm on every
    query was doubling the work for no reason; the ranking is unchanged.
    """
    entry_ids: list[str]
    vectors: list[list[float]]
    texts: dict[str, str]

    def __post_init__(self) -> None:
        self._unit = [self._normalize(v) for v in self.vectors]

    @staticmethod
    def _normalize(vec: Sequence[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else list(vec)

    @classmethod
    def build(cls, entries: list[Entry], progress: bool = True) -> "EmbeddingStore":
        ids = [e.entry_id for e in entries]
        texts = [e.text for e in entries]
        vectors: list[list[float]] = []
        batch = 32
        for start in range(0, len(texts), batch):
            vectors.extend(ollama.embed(texts[start:start + batch]))
            if progress and (start // batch) % 10 == 0:
                print(f"  embedded {min(start + batch, len(texts))}/{len(texts)}",
                      flush=True)
        return cls(ids, vectors, {e.entry_id: e.text for e in entries})

    def search(self, query: str, k: int = TOP_K) -> list[tuple[str, float]]:
        qv = self._normalize(ollama.embed([query])[0])
        dot = _dot
        scored = [(eid, dot(qv, vec))
                  for eid, vec in zip(self.entry_ids, self._unit)]
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored[:k]

    def save(self, path: Path) -> None:
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for eid, vec in zip(self.entry_ids, self.vectors):
                fh.write(json.dumps({"id": eid, "v": vec}) + "\n")

    @classmethod
    def load(cls, path: Path, entries: list[Entry]) -> "EmbeddingStore":
        import json
        ids, vectors = [], []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                ids.append(obj["id"])
                vectors.append(obj["v"])
        return cls(ids, vectors, {e.entry_id: e.text for e in entries})


class ElicitedRag:
    """Retrieve with the embedding store, elicit p from the LLM, then isotonic."""

    def __init__(self, store: EmbeddingStore, k: int = TOP_K) -> None:
        self.store = store
        self.k = k

    def context_for(self, claim: str) -> tuple[str, list[str]]:
        hits = self.store.search(claim, self.k)
        chunks = []
        for eid, score in hits:
            chunks.append(f"[{eid}]\n{self.store.texts.get(eid, '')[:1200]}")
        return "\n\n".join(chunks), [eid for eid, _ in hits]

    def elicit(self, claim: str) -> tuple[float, list[str]]:
        context, hits = self.context_for(claim)
        raw = ollama.generate(
            ELICIT_PROMPT.format(context=context[:24000], claim=claim),
            system=ELICIT_SYSTEM, num_predict=120, model=ollama.JUDGE_MODEL)
        obj = ollama.extract_json(raw)
        p = 0.5
        if isinstance(obj, dict):
            try:
                p = float(obj.get("p", 0.5))
            except (TypeError, ValueError):
                p = 0.5
        else:
            m = re.search(r"0?\.\d+|[01](?:\.0+)?", raw)
            if m:
                try:
                    p = float(m.group(0))
                except ValueError:
                    p = 0.5
        return min(1.0, max(0.0, p)), hits
