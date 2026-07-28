"""Tests for the §6.8 harness itself.

The honest test is only as good as its ground truth, so the machinery that
produces the labels gets tested like anything else: the metrics against hand
computable cases, the perturbations against the property that makes them
provable, and the train/held-out split against leakage.

No network. Everything here runs offline against synthetic fixtures.
"""

from __future__ import annotations

import math

import pytest

from bench import metrics
from bench.corpus import Entry
from bench.generate_suite import (CorpusIndex, Item, anchor_class, build_claims,
                                  entity_swap, longest_shared_ngram, norm,
                                  numeric_swap, verify)
from bench.run_6_8 import split_of


# ── metrics, against cases you can check by hand ────────────────────────────

def test_recall_at_k():
    assert metrics.recall_at_k(["a", "b", "c"], ["b"], 3) == 1.0
    assert metrics.recall_at_k(["a", "b", "c"], ["z"], 3) == 0.0
    assert metrics.recall_at_k(["a", "b", "c"], ["b", "z"], 3) == 0.5
    assert metrics.recall_at_k(["a", "b", "c"], ["c"], 2) == 0.0


def test_ndcg_rewards_rank():
    top = metrics.ndcg_at_k(["gold", "x", "y"], ["gold"], 3)
    mid = metrics.ndcg_at_k(["x", "gold", "y"], ["gold"], 3)
    assert top == 1.0
    assert mid == pytest.approx(1.0 / math.log2(3))
    assert top > mid > metrics.ndcg_at_k(["x", "y", "gold"], ["gold"], 3)


def test_mrr():
    assert metrics.mrr(["x", "gold"], ["gold"]) == 0.5
    assert metrics.mrr(["x", "y"], ["gold"]) == 0.0


def test_brier_and_log_loss():
    assert metrics.brier([(1.0, 1), (0.0, 0)]) == 0.0
    assert metrics.brier([(0.0, 1)]) == 1.0
    assert metrics.log_loss([(0.5, 1)]) == pytest.approx(math.log(2))


def test_ece_is_zero_for_a_calibrated_predictor():
    pairs = [(0.5, 1)] * 50 + [(0.5, 0)] * 50
    assert metrics.ece(pairs) == pytest.approx(0.0, abs=1e-9)


def test_reliability_slope_is_one_when_frequency_matches_confidence():
    # says 0.1 and is right 10% of the time; says 0.9 and is right 90%
    calibrated = ([(0.1, 1)] * 10 + [(0.1, 0)] * 90
                  + [(0.9, 1)] * 90 + [(0.9, 0)] * 10)
    assert metrics.reliability_slope(calibrated) == pytest.approx(1.0, abs=1e-9)

    # says 0.1 and is never right, says 0.9 and is always right: under-confident
    underconfident = [(0.1, 0), (0.9, 1)] * 20
    assert metrics.reliability_slope(underconfident) > 1.0


def test_ece_catches_overconfidence():
    overconfident = [(0.99, 1)] * 50 + [(0.99, 0)] * 50   # says 99%, right half
    assert metrics.ece(overconfident) == pytest.approx(0.49, abs=0.01)


def test_paired_bootstrap_detects_a_real_gap_and_ignores_noise():
    better = metrics.paired_bootstrap([1.0] * 40, [0.0] * 40, n_resamples=500)
    assert better["delta"] == 1.0 and better["lo"] > 0

    same = metrics.paired_bootstrap([0.5] * 40, [0.5] * 40, n_resamples=500)
    assert same["delta"] == 0.0 and same["lo"] == 0.0 and same["hi"] == 0.0


# ── the split must not leak ─────────────────────────────────────────────────

def test_split_is_deterministic_and_roughly_balanced():
    ids = [f"c{i:05d}" for i in range(2000)]
    first = [split_of(i) for i in ids]
    assert first == [split_of(i) for i in ids], "split must be reproducible"
    train = sum(1 for s in first if s == "train")
    assert 0.45 < train / len(ids) < 0.55


def test_split_does_not_depend_on_the_outcome():
    """The label must not be able to influence which side an item lands on."""
    assert split_of("c00042t") == split_of("c00042t")
    # ids differing only in the true/false suffix land independently
    pairs = [(split_of(f"c{i:05d}t"), split_of(f"c{i:05d}f")) for i in range(400)]
    disagree = sum(1 for a, b in pairs if a != b)
    assert 0.3 < disagree / len(pairs) < 0.7


# ── perturbations: the property that makes falsity provable ─────────────────

def test_anchor_classes():
    assert anchor_class("105-115") == "range"
    assert anchor_class("3.11.2") == "version"
    assert anchor_class("gpu-host.example.com:11434") == "hostname"
    assert anchor_class("403") == "http_code"
    assert anchor_class("0.3") == "number"
    assert anchor_class("CUDA_ERROR") == "screaming"


def test_numeric_swap_keeps_ranges_well_ordered():
    import random
    rng = random.Random(1)
    for _ in range(50):
        out = numeric_swap("throughput was 105-115 tps", rng)
        if out is None:
            continue
        text, note = out
        lo, hi = [float(x) for x in
                  __import__("re").search(r"(\d+)-(\d+)", text).groups()]
        assert lo < hi, f"perturbation produced a malformed range: {text}"
        assert text != "throughput was 105-115 tps"


def test_numeric_swap_preserves_decimal_shape():
    import random
    out = numeric_swap("the value is 0.30 units", random.Random(3))
    assert out is not None
    assert __import__("re").search(r"\d+\.\d{2}", out[0]), out[0]


def test_entity_swap_only_within_a_class():
    assert entity_swap("host is gpu-host.example.com", "0.3",
                       "gpu-host.example.com") is None, "type-incoherent swap refused"
    ok = entity_swap("host is gpu-host.example.com", "crawl-host.example.com",
                     "gpu-host.example.com")
    assert ok is not None and "crawl-host.example.com" in ok[0]


def test_false_claims_are_provably_false_against_the_corpus():
    entries = [
        Entry("e1", "t", "f", "The proxy on gpu-host.example.com reached 105-115 tps."),
        Entry("e2", "t", "f", "An unrelated note about 42 widgets and crawl-host.example.com."),
    ]
    index = CorpusIndex(entries)
    items = [
        Item("q1", "e1", "How fast?", "reached 105-115 tps",
             "The proxy reached 105-115 tps.", "105-115", ["e1"]),
        Item("q2", "e1", "How fast again?", "reached 105-115 tps",
             "The proxy reached 105-115 tps.", "105-115", ["e1"]),
    ]
    claims, stats = build_claims(items, index)
    assert stats.get("true") == 1 and stats.get("false") == 1
    for claim in claims:
        if claim.outcome:
            continue
        # the substituted value must appear nowhere in the gold entries
        substituted = claim.perturbation.split("->")[1].strip()
        gold = " ".join(index.normed[g] for g in ["e1"])
        assert norm(substituted) not in gold
        assert index.containing(claim.text) == []


def test_true_claims_restate_something_the_corpus_actually_contains():
    entries = [Entry("e1", "t", "f", "The retry limit is 5 attempts per host."),
               Entry("e2", "t", "f", "The cache holds 900 entries before eviction.")]
    index = CorpusIndex(entries)
    items = [
        Item("q1", "e1", "What is the retry limit?", "retry limit is 5 attempts",
             "The retry limit is 5 attempts per host.", "5", ["e1"]),
        Item("q2", "e2", "How big is the cache?", "cache holds 900 entries",
             "The cache holds 900 entries before eviction.", "900", ["e2"]),
    ]
    claims, _ = build_claims(items, index)
    true_claims = [c for c in claims if c.outcome]
    assert true_claims, "a balanced set still needs true claims"
    for claim in true_claims:
        assert index.containing(claim.evidence_span), \
            "a true claim must restate something the corpus contains"


def test_claims_are_balanced_so_the_base_rate_cannot_flatter_a_lazy_predictor():
    entries, items = [], []
    for i in range(20):
        entries.append(Entry(f"e{i}", "t", "f",
                             f"Job {i} retried {i + 3} times before the timeout."))
        items.append(Item(f"q{i}", f"e{i}", f"How many retries for job {i}?",
                          f"retried {i + 3} times",
                          f"Job {i} retried {i + 3} times before the timeout.",
                          str(i + 3), [f"e{i}"]))
    claims, stats = build_claims(items, CorpusIndex(entries))
    n_true = sum(1 for c in claims if c.outcome)
    n_false = sum(1 for c in claims if not c.outcome)
    assert n_true == n_false, f"base rate skewed: {n_true} true vs {n_false} false"
    assert n_true > 0


# ── item verification rejects what it should ────────────────────────────────

def _entry(text: str) -> Entry:
    return Entry("e1", "t", "f", text)


def _index(*entries: Entry) -> CorpusIndex:
    return CorpusIndex(list(entries))


BODY = ("The crawl4ai container answers on port 11235 and the healthcheck "
        "passed on the first tick after deployment on the LAN host.")


def test_verify_rejects_a_hallucinated_span():
    entry = _entry(BODY)
    obj = {"question": "Which port does the container answer on and why",
           "answer_span": "answers on port 9999 and the healthcheck passed",
           "fact": "It answers on 9999.", "anchor": "9999"}
    item, reason = verify(obj, entry, _index(entry))
    assert item is None and "verbatim" in reason


def test_verify_rejects_a_copy_paste_question():
    entry = _entry(BODY)
    obj = {"question": BODY, "answer_span": "answers on port 11235",
           "fact": "It answers on port 11235.", "anchor": "11235"}
    item, reason = verify(obj, entry, _index(entry))
    assert item is None and "gram" in reason


def test_verify_accepts_a_good_item_and_records_the_gold_set():
    entry = _entry(BODY)
    obj = {"question": "Where does the crawler service listen for requests",
           "answer_span": "answers on port 11235",
           "fact": "The crawl4ai container listens on port 11235.",
           "anchor": "11235"}
    item, reason = verify(obj, entry, _index(entry))
    assert item is not None, reason
    assert item.gold_entries == ["e1"]


def test_verify_rejects_boilerplate_spans():
    body = "the healthcheck passed on the first tick after deployment today"
    entries = [Entry(f"e{i}", "t", "f", body + f" run {i}") for i in range(5)]
    obj = {"question": "What happened to the healthcheck on this run",
           "answer_span": "healthcheck passed on the first tick",
           "fact": "The healthcheck passed.", "anchor": "healthcheck"}
    item, reason = verify(obj, entries[0], CorpusIndex(entries))
    assert item is None and "boilerplate" in reason


def test_item_ids_are_stable_across_processes():
    """PYTHONHASHSEED must not be able to change a suite's identity."""
    entry = _entry(BODY)
    obj = {"question": "Where does the crawler service listen for requests",
           "answer_span": "answers on port 11235",
           "fact": "The crawl4ai container listens on port 11235.",
           "anchor": "11235"}
    a, _ = verify(obj, entry, _index(entry))
    b, _ = verify(dict(obj), entry, _index(entry))
    assert a.item_id == b.item_id == "q" + __import__("hashlib").sha256(
        b"e1").hexdigest()[:10]


def test_shared_ngram_measure():
    assert longest_shared_ngram("alpha beta gamma", "alpha beta gamma delta") == 3
    assert longest_shared_ngram("completely different", "alpha beta") == 0
