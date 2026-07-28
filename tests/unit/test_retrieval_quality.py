"""v0.3 R1/R2 retrieval: sub-tokens, RM3 determinism, dense fusion + degrade."""

from __future__ import annotations

from pathlib import Path

import pytest

from candor.periphery.retrieval import Retriever, tokenize


@pytest.fixture()
def tier(tmp_path):
    ev = tmp_path / "evidence"
    ev.mkdir()
    (ev / "infra.md").write_text(
        "@salience: 0.9\n\n"
        "The ollama service on gpu-host.example.com:11434 serves the models and "
        "the crawl service answers on crawl-host.example.com:11235 for scraping.\n",
        encoding="utf-8")
    (ev / "cooking.md").write_text(
        "@salience: 0.5\n\n"
        "Slow-roasted vegetables need a hot oven and patience, nothing else. "
        "The tray should be crowded loosely to let the moisture escape fully.\n",
        encoding="utf-8")
    (ev / "gpu.md").write_text(
        "@salience: 0.5\n\n"
        "Inference throughput depends on quantization format and tensor "
        "parallel layout across the four graphics cards in the server chassis.\n",
        encoding="utf-8")
    return ev


def test_subtokens_reach_dotted_identifiers(tier, tmp_path):
    r = Retriever(tier, tmp_path / "none")
    hits = r.recall("ollama gpu host", budget=512)
    assert hits and "infra" in hits[0]["entry_id"], \
        "'gpu' and 'host' must match gpu-host.example.com via sub-token indexing"


def test_tokenize_emits_both_whole_and_parts():
    toks = tokenize("gpu-host.example.com:11434")
    assert "gpu-host.example.com:11434" in toks
    assert "gpu" in toks and "example" in toks and "11434" in toks


def test_recall_is_deterministic(tier, tmp_path):
    r = Retriever(tier, tmp_path / "none")
    a = [h["entry_id"] for h in r.recall("gpu quantization speed", budget=512)]
    b = [h["entry_id"] for h in r.recall("gpu quantization speed", budget=512)]
    assert a == b


def _fake_dense(mapping):
    """Embedder stub: bag-of-keyword axes, no network."""
    axes = ("oven", "gpu", "server")

    def embed(texts):
        out = []
        for text in texts:
            low = text.lower()
            vec = [1.0 if any(w in low for w in words) else 0.0
                   for words in mapping.get("axes", [["oven", "roast", "vegetable",
                                                      "dinner", "cook"],
                                                     ["gpu", "graphics",
                                                      "quantization", "inference"],
                                                     ["server", "host", "service"]])]
            out.append(vec if any(vec) else [0.0, 0.0, 0.001])
        return out
    return embed


def test_dense_fusion_rescues_a_paraphrase(tier, tmp_path):
    """'what should I cook for dinner' shares no useful term with the roasting
    note; lexical alone cannot rank it first, the dense axis can."""
    lexical = Retriever(tier, tmp_path / "none")
    lex_hits = [h["entry_id"] for h in
                lexical.recall("what should I cook for dinner tonight", budget=512)]
    assert not lex_hits or "cooking" not in lex_hits[0]

    fused = Retriever(tier, tmp_path / "none", dense=_fake_dense({}))
    hits = [h["entry_id"] for h in
            fused.recall("what should I cook for dinner tonight", budget=512)]
    assert hits and "cooking" in hits[0], \
        "dense fusion must surface the semantically-right entry"


def test_absent_embedder_means_pure_lexical(tier, tmp_path):
    plain = Retriever(tier, tmp_path / "none")
    assert plain.dense is None
    hits = plain.recall("ollama gpu host", budget=512)
    assert hits, "lexical path unaffected by the R2 machinery"


def test_failing_embedder_degrades_to_lexical(tier, tmp_path):
    def broken(texts):
        raise ConnectionError("embedder offline")

    r = Retriever(tier, tmp_path / "none", dense=broken)
    hits = [h["entry_id"] for h in r.recall("ollama gpu host", budget=512)]
    plain = Retriever(tier, tmp_path / "none")
    assert hits == [h["entry_id"] for h in plain.recall("ollama gpu host",
                                                        budget=512)], \
        "degrade, never fail: a dead embedder must leave lexical ranking intact"


def test_embedder_stays_out_of_the_import_graph():
    """R2's condition: retrieval gains dense ranking by injection only."""
    from candor import audit
    assert audit.direct_imports("candor.periphery.retrieval") == []
    assert audit.retrieval_writer_import_paths() == []
