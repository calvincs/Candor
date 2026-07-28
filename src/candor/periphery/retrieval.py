"""Retrieval: evidence tier reader + the retrieval side stream (spec §3.2, I2).

Retrieval never moves a number. That is enforced two ways:

  * by stream — this module writes only to `retrieval.sqlite3`, which is not
    hash-chained and which nothing downstream reads;
  * by import graph — this module imports the standard library and nothing
    else. It cannot obtain a reference to the count updater, because there is
    no path from here to `candor.core.committed.counts`. `candor.audit`
    verifies that mechanically.

Do not add a `candor.*` import to this file. The firewall is the point.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

TOKEN = re.compile(r"[A-Za-z0-9_.:-]+")
SUBSPLIT = re.compile(r"[._:\-]+")
CHARS_PER_TOKEN = 4          # crude budget accounting; entries are short

# v0.3 R1 — RM3 pseudo-relevance feedback (deterministic, stdlib, no model).
RM3_FEEDBACK_DOCS = 10
RM3_EXPANSION_TERMS = 8
RM3_MIX = 0.4                # expansion terms carry this fraction of a term


def tokenize(text: str) -> list[str]:
    """Tokens plus sub-tokens of dotted/hyphenated identifiers.

    The corpus is full of `infra.aibox-llm`-style names; a query saying
    "aibox" must be able to reach them (v1 lost 23 gold documents to exactly
    this class of mismatch — FINDINGS_6_8 F4).
    """
    out: list[str] = []
    for raw in TOKEN.findall(text):
        token = raw.lower()
        out.append(token)
        if any(ch in token for ch in "._:-"):
            out.extend(p for p in SUBSPLIT.split(token) if len(p) >= 2)
    return out


class EvidenceEntry:
    """A prose span with stable identity (§3.2): (entry_id, content_hash, offset)."""

    __slots__ = ("entry_id", "content_hash", "offset", "text", "meta", "origin")

    def __init__(self, entry_id: str, content_hash: str, offset: int, text: str,
                 meta: dict[str, Any], origin: str) -> None:
        self.entry_id = entry_id
        self.content_hash = content_hash
        self.offset = offset
        self.text = text
        self.meta = meta
        self.origin = origin

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id, "content_hash": self.content_hash,
            "offset": self.offset, "text": self.text, "origin": self.origin,
            "salience": self.meta.get("salience", 1.0),
            "span": [self.entry_id, self.content_hash, self.offset],
        }


class RetrievalLog:
    """Side stream. No chain, no downstream writes (§2, §3.1)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS retrieval_log("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, actor TEXT,"
            "  query TEXT, spans_json TEXT)")
        self._db.commit()

    def record(self, actor: str, query: str, spans: list[Any]) -> None:
        self._db.execute(
            "INSERT INTO retrieval_log(ts, actor, query, spans_json) VALUES(?,?,?,?)",
            (int(time.time() * 1000), actor, query, json.dumps(spans)))
        self._db.commit()

    def count(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM retrieval_log").fetchone()[0])

    def close(self) -> None:
        self._db.close()


class Retriever:
    """Hybrid retrieval over the evidence tier and the ledger's source material.

    Two corpora, both read-only:
      * `<root>/evidence/*.md` — prose entries with @salience metadata;
      * `<root>/payloads/*.json` — the content-addressed payload store. Redaction
        deletes the file, so redacted content cannot resurface here (§3.1).
    """

    def __init__(self, evidence_dir: Path, payload_dir: Path,
                 log: Optional[RetrievalLog] = None,
                 dense=None) -> None:
        self.evidence_dir = Path(evidence_dir)
        self.payload_dir = Path(payload_dir)
        self.log = log
        # v0.3 Δ5 (R2): optional dense ranker, INJECTED as a bare callable
        # (list[str] -> list[vector]) so this module keeps its empty import
        # list and the I2 audit keeps meaning something. Ranking input only,
        # same standing as @salience; absent or failing, ranking is lexical.
        self.dense = dense
        self._dense_vectors: dict[str, list[float]] = {}
        # Incremental corpus index (H7). Rebuilding the whole index on every
        # call made an appended entry cost O(corpus): a single new payload
        # changed the directory fingerprint and forced a from-scratch re-read +
        # re-tokenize + BM25 rebuild of EVERY document. Instead the index is
        # maintained per file: only new/changed files are read and tokenized,
        # and a file that disappears (redaction, §3.1/§6.6) is evicted. Corpus
        # statistics (df, doc-length sum) are updated incrementally, so scoring
        # sees exactly what a full rebuild would — this stays memoization, not a
        # ranking change. Keyed by a cheap (mtime_ns, size) stamp so unchanged
        # files are never re-read.
        self._files: dict[Path, tuple[tuple[int, int], str,
                                      list[EvidenceEntry], list[list[str]]]] = {}
        self._df: Counter = Counter()
        self._doclen_sum: int = 0
        self._flat_entries: Optional[list[EvidenceEntry]] = None
        self._flat_docs: list[list[str]] = []
        self._avgdl: float = 1.0

    def _scan(self) -> dict[Path, tuple[tuple[int, int], str]]:
        """Cheap directory census: stat only, never read or tokenize.

        Returns each present file's (mtime_ns, size) stamp and its origin. This
        is what lets an appended file be detected, and a deleted one evicted,
        without touching the O(corpus) files whose stamp is unchanged.
        """
        current: dict[Path, tuple[tuple[int, int], str]] = {}
        for directory, pattern, origin in (
                (self.evidence_dir, "**/*.md", "evidence"),
                (self.payload_dir, "*.json", "payload")):
            if not directory.exists():
                continue
            for path in directory.glob(pattern):
                try:
                    st = path.stat()
                except OSError:
                    continue
                current[path] = ((st.st_mtime_ns, st.st_size), origin)
        return current

    def _evict(self, path: Path) -> None:
        _stamp, _origin, _entries, docs = self._files.pop(path)
        for doc in docs:
            self._doclen_sum -= len(doc)
            for term in set(doc):
                remaining = self._df[term] - 1
                if remaining <= 0:
                    del self._df[term]
                else:
                    self._df[term] = remaining

    def _ingest(self, path: Path, stamp: tuple[int, int], origin: str) -> None:
        entries = (self._entries_for_evidence(path) if origin == "evidence"
                   else self._entries_for_payload(path))
        docs = [tokenize(e.text) for e in entries]
        for doc in docs:
            self._doclen_sum += len(doc)
            self._df.update(set(doc))
        self._files[path] = (stamp, origin, entries, docs)

    def _corpus(self) -> tuple[list[EvidenceEntry], list[list[str]],
                               Counter, float]:
        current = self._scan()
        changed = False
        for path in list(self._files):
            stamp = self._files[path][0]
            seen = current.get(path)
            if seen is None or seen[0] != stamp:   # gone or edited → drop it
                self._evict(path)
                changed = True
        for path, (stamp, origin) in current.items():
            if path not in self._files:            # new or just-evicted-changed
                self._ingest(path, stamp, origin)
                changed = True
        if changed or self._flat_entries is None:
            self._rebuild_flat()
        return self._flat_entries, self._flat_docs, self._df, self._avgdl

    def _rebuild_flat(self) -> None:
        """Flatten per-file records into the (entries, docs) lists the scorer
        walks. Evidence before payloads, each path-sorted — the exact order a
        `_evidence_entries() + _payload_entries()` full build produced.
        """
        entries: list[EvidenceEntry] = []
        docs: list[list[str]] = []
        for path in sorted(self._files,
                           key=lambda p: (self._files[p][1] != "evidence", str(p))):
            _stamp, _origin, file_entries, file_docs = self._files[path]
            entries.extend(file_entries)
            docs.extend(file_docs)
        self._flat_entries = entries
        self._flat_docs = docs
        self._avgdl = (self._doclen_sum / len(docs)) if docs else 1.0

    # ── corpus ──────────────────────────────────────────────────────────────
    def _entries_for_evidence(self, path: Path) -> list[EvidenceEntry]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return []
        meta: dict[str, Any] = {}
        for line in text.splitlines():
            m = re.match(r"^@(\w+):\s*(.*)$", line.strip())
            if m:
                meta[m.group(1)] = m.group(2)
        out: list[EvidenceEntry] = []
        offset = 0
        for i, chunk in enumerate(text.split("\n\n")):
            if chunk.strip():
                # hash exactly the text stored, or every re-hash misses
                out.append(EvidenceEntry(
                    f"{path.stem}#{i}", _content_hash(chunk.strip()), offset,
                    chunk.strip(), meta, "evidence"))
            offset += len(chunk) + 2
        return out

    def _entries_for_payload(self, path: Path) -> list[EvidenceEntry]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return []
        return [EvidenceEntry(path.stem, path.stem, 0, text, {}, "payload")]

    # ── ranking ─────────────────────────────────────────────────────────────
    def _score_pass(self, terms: list[tuple[str, float]], entries, docs, df,
                    avgdl) -> list[tuple[float, EvidenceEntry, int]]:
        n_docs = len(docs) or 1
        scored: list[tuple[float, EvidenceEntry, int]] = []
        for i, (entry, doc) in enumerate(zip(entries, docs)):
            score = _bm25(terms, doc, df, n_docs, avgdl)
            if score <= 0.0:
                continue
            # @salience is a retrieval-ranking input and nothing else (§3.2).
            try:
                salience = float(entry.meta.get("salience", 1.0))
            except (TypeError, ValueError):
                salience = 1.0
            scored.append((score * salience, entry, i))
        scored.sort(key=lambda t: (-t[0], t[1].entry_id))
        return scored

    def _rm3_terms(self, base: list[str],
                   feedback: list[tuple[float, EvidenceEntry, int]],
                   docs: list[list[str]], df: Counter,
                   n_docs: int) -> list[tuple[str, float]]:
        """Expansion terms from the first-pass leaders, ranked by tf-idf mass.

        Deterministic and purely lexical: the classic fix for paraphrased
        queries that never share the document's exact vocabulary.
        """
        base_set = set(base)
        gain: Counter = Counter()
        for _, _, idx in feedback[:RM3_FEEDBACK_DOCS]:
            tf = Counter(docs[idx])
            length = len(docs[idx]) or 1
            for term, count in tf.items():
                if term in base_set or len(term) < 3 or term.isdigit():
                    continue
                idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
                gain[term] += (count / length) * idf
        ranked = sorted(gain.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(t, RM3_MIX) for t, _ in ranked[:RM3_EXPANSION_TERMS]]

    def _dense_rank(self, query: str, entries: list[EvidenceEntry]) -> list[int]:
        """Cosine ranking over the injected embedder. Degrades to [] on failure."""
        try:
            candidates = [(i, e) for i, e in enumerate(entries)
                          if len(e.text) >= 100]        # headers stay lexical-only
            missing = [e for _, e in candidates
                       if e.content_hash not in self._dense_vectors]
            if missing:
                for entry, vec in zip(missing,
                                      self.dense([e.text for e in missing])):
                    if vec:
                        self._dense_vectors[entry.content_hash] = vec
            qvec = self.dense([query])[0]
            if not qvec:
                return []
            qn = math.sqrt(sum(x * x for x in qvec)) or 1.0
            sims: list[tuple[float, int]] = []
            for i, entry in candidates:
                vec = self._dense_vectors.get(entry.content_hash)
                if not vec:
                    continue
                dot = sum(a * b for a, b in zip(qvec, vec))
                norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                sims.append((dot / (qn * norm), i))
            sims.sort(key=lambda t: (-t[0], entries[t[1]].entry_id))
            return [i for _, i in sims]
        except Exception:                                 # noqa: BLE001
            return []                                     # degrade, never fail

    @staticmethod
    def _rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
        """Reciprocal rank fusion — robust, parameter-light, deterministic."""
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
        return fused

    def recall(self, query: str, budget: int,
               actor: str = "agent:reader") -> list[dict]:
        base = tokenize(query)
        entries, docs, df, avgdl = self._corpus()
        first = self._score_pass([(t, 1.0) for t in base], entries, docs, df, avgdl)
        scored = first
        if first:
            expansion = self._rm3_terms(base, first, docs, df, len(docs) or 1)
            if expansion:
                scored = self._score_pass(
                    [(t, 1.0) for t in base] + expansion, entries, docs, df, avgdl)
        if self.dense is not None:
            dense_order = self._dense_rank(query, entries)
            if dense_order:
                lex_order = [idx for _, _, idx in scored]
                fused = self._rrf([lex_order, dense_order])
                order = sorted(fused, key=lambda i: (-fused[i], entries[i].entry_id))
                scored = [(fused[i], entries[i], i) for i in order]

        # token-budgeted context packing via greedy knapsack
        out: list[dict] = []
        spent = 0
        for score, entry, _ in scored:
            cost = max(1, len(entry.text) // CHARS_PER_TOKEN)
            if spent + cost > budget:
                continue
            spent += cost
            row = entry.as_dict()
            row["score"] = score
            out.append(row)
        if self.log is not None:
            self.log.record(actor, query, [r["span"] for r in out])
        return out


def _bm25(terms: list[tuple[str, float]], doc: list[str], df: Counter,
          n_docs: int, avgdl: float, k1: float = 1.5, b: float = 0.75) -> float:
    """BM25 over (term, mix) pairs; expansion terms carry a reduced mix."""
    if not doc:
        return 0.0
    tf = Counter(doc)
    score = 0.0
    for term, mix in terms:
        f = tf.get(term, 0)
        if not f:
            continue
        idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
        score += mix * idf * (f * (k1 + 1.0)) / (
            f + k1 * (1.0 - b + b * len(doc) / avgdl))
    return score


def _content_hash(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
