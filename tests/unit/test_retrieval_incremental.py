"""H7: recall() maintains its corpus index incrementally, not by full rebuild.

The old cache was keyed on a (mtime, size) fingerprint of the whole evidence +
payload directories, so a single new payload invalidated it and forced a
from-scratch re-read + re-tokenize + BM25 rebuild of EVERY document. These tests
pin the fix: after the corpus is indexed once, appending one document ingests
ONLY that document (O(1) tokenize work, not O(corpus)), and a redacted (deleted)
payload is evicted from recall results.
"""

from __future__ import annotations

import json

import candor.periphery.retrieval as rmod
from candor.periphery.retrieval import Retriever


def _write_payload(directory, name, text):
    (directory / f"{name}.json").write_text(json.dumps({"text": text}),
                                            encoding="utf-8")


def test_appending_one_doc_does_not_retokenize_the_corpus(tmp_path, monkeypatch):
    ev = tmp_path / "evidence"
    ev.mkdir()
    payloads = tmp_path / "payloads"
    payloads.mkdir()

    n = 400
    for i in range(n):
        _write_payload(payloads, f"doc{i:04d}", f"alpha beta gamma document number {i}")

    r = Retriever(ev, payloads)
    r.recall("alpha", budget=512)          # index the corpus once

    calls = {"n": 0}
    real_tokenize = rmod.tokenize

    def counting_tokenize(text):
        calls["n"] += 1
        return real_tokenize(text)

    monkeypatch.setattr(rmod, "tokenize", counting_tokenize)

    _write_payload(payloads, "doc_new", "epsilon zeta a brand new document")
    r.recall("epsilon", budget=512)

    # Steady state: one query tokenized + one new document tokenized. The old
    # full-rebuild path would tokenize all n + 1 documents here.
    assert calls["n"] <= 5, (
        f"appending one document re-tokenized {calls['n']} texts; "
        f"incremental ingest should be O(1), not O({n})")


def test_recall_results_unchanged_after_incremental_append(tmp_path):
    """The incremental index must produce the same ranking a full rebuild would."""
    ev = tmp_path / "evidence"
    ev.mkdir()
    payloads = tmp_path / "payloads"
    payloads.mkdir()
    for i in range(20):
        _write_payload(payloads, f"doc{i:02d}", f"alpha beta term{i} shared context")

    incremental = Retriever(ev, payloads)
    incremental.recall("alpha", budget=512)                 # warm the index
    _write_payload(payloads, "late", "alpha beta term7 shared context arrival")
    incr_hits = [h["entry_id"] for h in incremental.recall("alpha term7", budget=512)]

    fresh = Retriever(ev, payloads)                         # cold, full build
    fresh_hits = [h["entry_id"] for h in fresh.recall("alpha term7", budget=512)]

    assert incr_hits == fresh_hits, \
        "incremental ingest must match a from-scratch build exactly"


def test_redacted_payload_is_evicted_from_recall(tmp_path):
    ev = tmp_path / "evidence"
    ev.mkdir()
    payloads = tmp_path / "payloads"
    payloads.mkdir()
    secret = payloads / "secret.json"
    secret.write_text(json.dumps({"text": "purge-me confidential incident detail"}),
                      encoding="utf-8")

    r = Retriever(ev, payloads)
    hits = r.recall("purge-me confidential", budget=512)
    assert "purge-me" in str(hits), "sanity: the payload is indexed before redaction"

    secret.unlink()                                        # §6.6 redaction = delete
    after = r.recall("purge-me confidential", budget=512)
    assert "purge-me" not in str(after), \
        "a redacted (deleted) payload must disappear from recall"
