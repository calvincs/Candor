"""Dense embedding client — untrusted periphery ranking input (v0.3 Δ5 = R2).

The §0 non-goal is narrowed, not abandoned: embeddings exist only here, in the
fallible periphery, as a *ranking input* with exactly the standing of
`@salience`. They never enter the trusted core, never touch the committed
tier, and retrieval still cannot move a number (I2) — this module is wired
into `Retriever` by injection precisely so `retrieval.py` keeps its empty
import list and the mechanical audit keeps meaning something.

Entirely optional: constructed from environment (`CANDOR_EMBED_URL`,
`CANDOR_EMBED_MODEL`); absent or unreachable, retrieval degrades to lexical.
Vectors are cached on disk keyed by content hash, so the store is droppable
and rebuildable like every other derived artifact (I1 in spirit).
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Optional

ENV_URL = "CANDOR_EMBED_URL"
ENV_MODEL = "CANDOR_EMBED_MODEL"
DEFAULT_MODEL = "bge-m3:latest"
BATCH = 32


class Embedder:
    def __init__(self, url: str, model: str, cache_dir: Path,
                 timeout: float = 300.0) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def _key(self, text: str) -> Path:
        digest = hashlib.sha256(f"{self.model}|{text}".encode()).hexdigest()[:40]
        return self.cache / f"{digest}.json"

    def __call__(self, texts: list[str]) -> list[list[float]]:
        out: list[Optional[list[float]]] = [None] * len(texts)
        pending: list[int] = []
        for i, text in enumerate(texts):
            path = self._key(text)
            if path.exists():
                out[i] = json.loads(path.read_text(encoding="utf-8"))
            else:
                pending.append(i)
        for start in range(0, len(pending), BATCH):
            chunk = pending[start:start + BATCH]
            req = urllib.request.Request(
                self.url + "/api/embed",
                data=json.dumps({"model": self.model,
                                 "input": [texts[i] for i in chunk],
                                 "keep_alive": "30m"}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                vectors = json.loads(resp.read().decode()).get("embeddings") or []
            if len(vectors) != len(chunk):
                raise RuntimeError("embed batch size mismatch")
            for i, vec in zip(chunk, vectors):
                out[i] = vec
                self._key(texts[i]).write_text(json.dumps(vec), encoding="utf-8")
        return [v if v is not None else [] for v in out]


def from_env(root: Path) -> Optional[Embedder]:
    url = os.environ.get(ENV_URL)
    if not url:
        return None
    return Embedder(url, os.environ.get(ENV_MODEL, DEFAULT_MODEL),
                    Path(root) / "dense_cache")
