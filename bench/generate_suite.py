"""Synthetic §6.8 suite generation with mechanically provable ground truth.

The model proposes; arithmetic disposes. Nothing the LLM says is trusted as a
label. Every item that survives carries ground truth that a human can re-check
with `grep`:

  retrieval gold   the answer span occurs verbatim in exactly one corpus entry
                   -> that entry is the gold document, by exact string match
  claim outcome    TRUE items restate a verified span; FALSE items are
                   *programmatic* perturbations of it, and are re-scanned
                   against the whole corpus to prove they match nothing

Items failing any check are discarded with the reason recorded, so the yield
and the rejection reasons are themselves auditable.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import ollama
from .corpus import Entry, load

SUITE_PATH = Path("data/bench/suite.json")
REJECTS_PATH = Path("data/bench/rejects.json")

MIN_SPAN = 20
MAX_SPAN = 420
# A question that reuses a long verbatim run from its source entry makes the
# retrieval task degenerate for every system at once. Symmetric difficulty
# filter; it favours neither the lexical nor the dense side.
MAX_SHARED_NGRAM = 7

WORD = re.compile(r"[A-Za-z0-9_.:/-]+")
NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")

GEN_SYSTEM = (
    "You extract verifiable test items from technical notes. "
    "You copy spans EXACTLY, character for character. You never invent facts. "
    "You reply with one JSON object and nothing else."
)

GEN_PROMPT = """Below is a single technical note.

Produce ONE test item as JSON with these keys:

  "question"    A specific question a colleague might ask, answerable ONLY by
                someone who has read this note. Paraphrase — do NOT copy long
                phrases from the note. Do not mention the note or its filename.
  "answer_span" A span copied EXACTLY from the note, character for character,
                between {min_span} and {max_span} characters (aim for 60-250), that answers the
                question. It must appear in the note verbatim.
  "fact"        One self-contained declarative sentence stating the same fact,
                understandable without the note. Include the specific
                identifier, number, or name that makes it checkable.
  "anchor"      The single most distinctive token in the answer span — an
                identifier, error code, version, hostname, or number.

NOTE:
\"\"\"
{text}
\"\"\"

JSON only."""


@dataclass
class Item:
    item_id: str
    entry_id: str
    question: str
    answer_span: str
    fact: str
    anchor: str
    gold_entries: list[str] = field(default_factory=list)
    source_file: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Claim:
    claim_id: str
    item_id: str
    entry_id: str
    text: str
    outcome: bool                    # ground truth, set programmatically
    kind: str                        # 'verbatim' | 'numeric_swap' | 'entity_swap'
    evidence_span: str = ""
    perturbation: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# ── normalization used by every exact-match check ───────────────────────────

def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def tokens(text: str) -> list[str]:
    return [t.lower() for t in WORD.findall(text)]


def longest_shared_ngram(a: str, b: str) -> int:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0
    setb = set()
    best = 0
    for n in range(1, min(len(ta), 24) + 1):
        grams_b = {" ".join(tb[i:i + n]) for i in range(len(tb) - n + 1)}
        hit = any(" ".join(ta[i:i + n]) in grams_b for i in range(len(ta) - n + 1))
        if hit:
            best = n
        else:
            break
    return best


class CorpusIndex:
    """Exact-match index over the whole corpus. The arbiter of ground truth."""

    def __init__(self, entries: list[Entry]) -> None:
        self.entries = entries
        self.normed = {e.entry_id: norm(e.text) for e in entries}

    def containing(self, span: str) -> list[str]:
        needle = norm(span)
        if len(needle) < MIN_SPAN // 2:
            return []
        return [eid for eid, text in self.normed.items() if needle in text]

    def anchor_hits(self, anchor: str) -> list[str]:
        needle = norm(anchor)
        return [eid for eid, text in self.normed.items() if needle in text]


# ── stage A: propose ────────────────────────────────────────────────────────

def propose(entry: Entry) -> dict[str, Any]:
    raw = ollama.generate(
        GEN_PROMPT.format(text=entry.text[:5000], min_span=MIN_SPAN,
                          max_span=MAX_SPAN),
        system=GEN_SYSTEM, num_predict=600)
    obj = ollama.extract_json(raw)
    if not isinstance(obj, dict):
        return {"__error__": "no JSON in response"}
    obj["entry_id"] = entry.entry_id
    return obj


# ── stage B: verify mechanically ────────────────────────────────────────────

def verify(obj: dict[str, Any], entry: Entry,
           index: CorpusIndex) -> tuple[Optional[Item], str]:
    if "__error__" in obj:
        return None, obj["__error__"]
    for key in ("question", "answer_span", "fact", "anchor"):
        if not isinstance(obj.get(key), str) or not obj[key].strip():
            return None, f"missing field {key}"
    question = obj["question"].strip()
    span = obj["answer_span"].strip()
    fact = obj["fact"].strip()
    anchor = obj["anchor"].strip()

    if not (MIN_SPAN <= len(span) <= MAX_SPAN):
        return None, f"span length {len(span)} outside [{MIN_SPAN},{MAX_SPAN}]"
    # V1 — the span must really be in its source entry. Exact, not fuzzy.
    if norm(span) not in norm(entry.text):
        return None, "answer_span is not verbatim in the source entry"
    # V2 — uniqueness. The gold set is whatever the scan says it is.
    holders = index.containing(span)
    if entry.entry_id not in holders:
        return None, "index disagrees with source containment"
    if len(holders) > 3:
        return None, f"span is boilerplate: appears in {len(holders)} entries"
    # V3 — the anchor must be real and reasonably distinctive.
    if norm(anchor) not in norm(entry.text):
        return None, "anchor is not present in the source entry"
    if len(index.anchor_hits(anchor)) > 40:
        return None, "anchor is not distinctive"
    # V4 — symmetric difficulty filter against copy-paste questions.
    shared = longest_shared_ngram(question, entry.text)
    if shared > MAX_SHARED_NGRAM:
        return None, f"question shares a {shared}-gram with the entry"
    if len(tokens(question)) < 5:
        return None, "question too short to be a real query"

    # hashlib, not hash(): PYTHONHASHSEED randomizes str hashing per process,
    # which would make item ids differ between runs of a "reproducible" suite.
    item_id = "q" + hashlib.sha256(entry.entry_id.encode()).hexdigest()[:10]
    return Item(item_id=item_id, entry_id=entry.entry_id, question=question,
                answer_span=span, fact=fact, anchor=anchor,
                gold_entries=sorted(holders), source_file=entry.file), ""


# ── stage C: claims with programmatic outcomes ──────────────────────────────

# A false claim has to be *plausible*, or the experiment measures "can you spot
# garbled text" rather than "do you know what the corpus says". Perturbations
# are therefore type-preserving: a throughput becomes a different throughput, a
# hostname becomes a different hostname, a range stays a well-ordered range.
RANGE = re.compile(r"\b(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\b")
FACTORS = (0.4, 0.5, 1.6, 2.0, 2.5, 3.0)

ANCHOR_CLASSES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("range", re.compile(r"^\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?")),
    # A version needs a leading 'v' or two dots. Without that rule "0.3" reads
    # as a version and could be swapped for "3.11.2", which is not a
    # type-preserving perturbation of a decimal quantity.
    ("version", re.compile(r"^(?:v\d|\d+\.\d+\.\d+)")),
    ("path", re.compile(r"^[~/]")),
    ("http_code", re.compile(r"^[1-5]\d{2}$")),
    ("number", re.compile(r"^\d+(?:\.\d+)?$")),
    # quantity BEFORE hostname: "5.19s" is a duration, and the hostname pattern
    # would otherwise claim it (digits and letters both being label characters).
    ("quantity", re.compile(r"^\d+(?:\.\d+)?\s*[A-Za-z%/]+$")),
    # dotted names whose final label is a file extension are filenames, not
    # hosts: "news-summary-2026-05-29.md" otherwise reads as a hostname.
    ("filename", re.compile(
        r"^[\w.-]+\.(?:md|py|json|ya?ml|txt|sh|js|ts|log|csv|toml|ini|cfg|html)$",
        re.I)),
    # a hostname's last label must be alphabetic, and its first must contain a
    # letter — otherwise "5.19s" and "0.3" both read as hostnames.
    ("hostname", re.compile(
        r"^(?=[a-z0-9-]*[a-z])[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}(?::\d+)?$",
        re.I)),
    ("screaming", re.compile(r"^[A-Z][A-Z0-9_]{3,}$")),
    ("identifier", re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{2,}$")),
)

# Polarity flip: the most natural false claim of all, and type-safe by
# construction. Applied only to an unnegated auxiliary, so it cannot produce a
# double negative.
POLARITY = tuple(
    (verb, re.compile(rf"\b({verb})\b(?!\s+not\b)"), r"\1 not")
    for verb in ("is", "are", "was", "were", "can", "should", "must", "will",
                 "does", "did", "has", "have"))

# Classes whose members are interchangeable enough that transplanting one for
# another yields a sentence that still reads naturally. `identifier` and
# `phrase` are deliberately excluded: they lump together proper nouns, file
# names, regexes and product names, and swapping across those produces visible
# word salad — which would test whether a reader can spot nonsense rather than
# whether it knows what the corpus says.
TRANSPLANTABLE = frozenset({"hostname", "path", "version", "http_code",
                            "filename", "screaming", "quantity"})


def anchor_class(anchor: str) -> str:
    text = anchor.strip()
    for name, pattern in ANCHOR_CLASSES:
        if pattern.match(text):
            return name
    return "phrase"


def _shift(value: str, factor: float) -> str:
    """Scale a number, preserving its decimal shape."""
    decimals = len(value.split(".")[1]) if "." in value else 0
    scaled = float(value) * factor
    out = f"{scaled:.{decimals}f}" if decimals else str(int(round(scaled)))
    return out


def numeric_swap(text: str, rng: random.Random) -> Optional[tuple[str, str]]:
    """Move a quantity to a different plausible quantity. Ranges stay ordered."""
    factor = rng.choice(FACTORS)
    rng_match = list(RANGE.finditer(text))
    if rng_match:
        m = rng.choice(rng_match)
        lo, hi = m.group(1), m.group(2)
        new_lo, new_hi = _shift(lo, factor), _shift(hi, factor)
        if float(new_lo) >= float(new_hi) or (new_lo, new_hi) == (lo, hi):
            return None
        mutated = f"{new_lo}-{new_hi}"
        return (text[:m.start()] + mutated + text[m.end():],
                f"{m.group(0)} -> {mutated}")
    matches = [m for m in NUMBER.finditer(text) if m.group(0) not in ("0", "1")]
    if not matches:
        return None
    m = rng.choice(matches)
    original = m.group(0)
    value = float(original)
    if "." not in original and 1900 <= value <= 2100:
        # A year scaled by 2.5 gives "5062", which is false but not plausible.
        # Move it to a different nearby year instead.
        offset = rng.choice([-7, -5, -3, -2, 2, 3, 5, 7])
        mutated = str(int(value) + offset)
    else:
        mutated = _shift(original, factor)
    if mutated == original or float(mutated) == value:
        return None
    return text[:m.start()] + mutated + text[m.end():], f"{original} -> {mutated}"


def entity_swap(text: str, donor: str, own: str) -> Optional[tuple[str, str]]:
    """Transplant a same-class entity from a different entry — what a
    hallucinating agent actually does, rather than word salad."""
    own_class = anchor_class(own)
    if own_class not in TRANSPLANTABLE or anchor_class(donor) != own_class:
        return None
    if norm(own) not in norm(text) or norm(donor) == norm(own):
        return None
    pattern = re.compile(re.escape(own), re.IGNORECASE)
    mutated, n = pattern.subn(donor, text, count=1)
    if not n:
        return None
    return mutated, f"{own} -> {donor}"


def polarity_flip(text: str) -> Optional[tuple[str, str]]:
    """Negate the first unnegated auxiliary. Grammatical, plausible, and false."""
    for verb, pattern, replacement in POLARITY:
        mutated, n = pattern.subn(replacement, text, count=1)
        if n:
            return mutated, f"polarity: {verb} -> {verb} not"
    return None


def _substituted_value(note: str) -> str:
    if note.startswith("polarity:"):
        return ""          # nothing was substituted; falsity is structural
    return note.split("->", 1)[1].strip() if "->" in note else ""


def build_claims(items: list[Item], index: CorpusIndex,
                 seed: int = 20260725) -> tuple[list[Claim], dict[str, int]]:
    """Half true, half false. Every FALSE proved false by a corpus-wide scan."""
    rng = random.Random(seed)
    claims: list[Claim] = []
    stats: dict[str, int] = {}

    def bump(reason: str) -> None:
        stats[reason] = stats.get(reason, 0) + 1

    by_class: dict[str, list[str]] = {}
    for i in items:
        by_class.setdefault(anchor_class(i.anchor), []).append(i.anchor)

    def falsify(item: Item) -> Optional[Claim]:
        attempts: list[tuple[str, tuple[str, str]]] = []
        swap = numeric_swap(item.fact, rng)
        if swap is not None:
            attempts.append(("numeric_swap", swap))
        pool = [a for a in by_class.get(anchor_class(item.anchor), [])
                if norm(a) != norm(item.anchor)]
        if pool:
            swap = entity_swap(item.fact, rng.choice(pool), item.anchor)
            if swap is not None:
                attempts.append(("entity_swap", swap))
        flip = polarity_flip(item.fact)
        if flip is not None:
            attempts.append(("polarity_flip", flip))

        for kind, (mutated, note) in attempts:
            if norm(mutated) == norm(item.fact):
                continue
            # Provable falsity, two ways:
            #  1. the substituted value is absent from every gold entry, so the
            #     claim contradicts the only evidence that bears on it;
            #  2. the mutated sentence matches nothing anywhere in the corpus.
            substituted = _substituted_value(note)
            gold_text = " ".join(index.normed.get(g, "") for g in item.gold_entries)
            if substituted and norm(substituted) in gold_text:
                bump("substitution_present_in_gold")
                continue
            if index.containing(mutated):
                bump("perturbation_matched_corpus")
                continue
            return Claim(
                claim_id=f"c{item.item_id[1:]}f", item_id=item.item_id,
                entry_id=item.entry_id, text=mutated, outcome=False,
                kind=kind, evidence_span=item.answer_span, perturbation=note)
        return None

    # Which items *can* carry a false claim is a property of the item, so the
    # pool is computed first and the split assigned afterwards. Assigning
    # alternately up front silently skews the base rate whenever a perturbation
    # fails, and a skewed base rate flatters any predictor that leans one way.
    falsifiable: list[tuple[Item, Claim]] = []
    plain: list[Item] = []
    for item in items:
        made = falsify(item)
        (falsifiable.append((item, made)) if made else plain.append(item))

    n_false = min(len(falsifiable), len(items) // 2)
    for item, claim in falsifiable[:n_false]:
        claims.append(claim)
        bump("false")
    truth_pool = [i for i, _ in falsifiable[n_false:]] + plain
    for item in truth_pool[:n_false]:
        claims.append(Claim(
            claim_id=f"c{item.item_id[1:]}t", item_id=item.item_id,
            entry_id=item.entry_id, text=item.fact, outcome=True,
            kind="verbatim", evidence_span=item.answer_span))
        bump("true")
    stats["falsifiable_pool"] = len(falsifiable)
    claims.sort(key=lambda c: c.claim_id)
    return claims, stats


# ── driver ──────────────────────────────────────────────────────────────────

def main(n_items: int = 260, seed: int = 20260725, workers: int = 1) -> None:
    entries = load(Path("data/bench/corpus.jsonl"))
    index = CorpusIndex(entries)
    rng = random.Random(seed)
    pool = [e for e in entries if len(e.text) >= 400]
    rng.shuffle(pool)
    sample = pool[:n_items]
    print(f"corpus {len(entries)} entries; sampling {len(sample)}")

    def progress(done: int, total: int) -> None:
        if done % 20 == 0 or done == total:
            print(f"  proposed {done}/{total}", flush=True)

    proposals = ollama.parallel(propose, sample, workers=workers,
                                on_progress=progress)

    items: list[Item] = []
    rejects: list[dict[str, str]] = []
    for entry, obj in zip(sample, proposals):
        item, reason = verify(obj if isinstance(obj, dict) else {}, entry, index)
        if item is None:
            rejects.append({"entry_id": entry.entry_id, "reason": reason})
        else:
            items.append(item)

    claims, claim_stats = build_claims(items, index, seed=seed)

    reasons: dict[str, int] = {}
    for r in rejects:
        reasons[r["reason"][:48]] = reasons.get(r["reason"][:48], 0) + 1

    SUITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUITE_PATH.write_text(json.dumps({
        "seed": seed,
        "gen_model": ollama.GEN_MODEL,
        "corpus_entries": len(entries),
        "sampled": len(sample),
        "items": [i.to_json() for i in items],
        "claims": [c.to_json() for c in claims],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    REJECTS_PATH.write_text(json.dumps(
        {"reasons": reasons, "rejects": rejects}, indent=1), encoding="utf-8")

    print(f"\nitems kept:  {len(items)}/{len(sample)}  "
          f"({100*len(items)/max(1,len(sample)):.0f}% yield)")
    print(f"claims:      {len(claims)}  {claim_stats}")
    print(f"reject reasons: {json.dumps(reasons, indent=1)}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 260
    main(n_items=n)
