"""Suite v2 — the sparse-panel, off-ceiling test (FINDINGS_6_8 F1–F3).

Reuses the v1 authoring + mechanical verification machinery verbatim; what
changes is exactly what the post-mortem said had to change:

  T1  sparse heterogeneous panels — 12 observers, each claim assigned a random
      4-of-12 subset, seeded by claim id. Assignment is part of the suite
      artifact so it is pre-registered, not runtime randomness. Under a fixed
      panel the vote pattern is a sufficient statistic and isotonic-on-mean is
      near-optimal by construction; sparse panels are the regime where
      per-actor calibration must transfer or fail.

  T2  off the ceiling — a new `within_entry_swap` perturbation whose substituted
      value is *present in the gold entry* (defeats lexical presence checks and
      forces actual reading), plus per-kind caps so no single failure axis
      dominates the false class the way polarity flips (47/120) did in v1.

  T3  size — default 1600 sampled entries -> ~1200 items/claims, ~600 held out,
      per the power analysis (resolving 0.01 Brier needs ~600 held-out claims).

Ground truth remains mechanical: every label re-checkable with grep.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Optional

from . import ollama
from .corpus import load
from .generate_suite import (Claim, CorpusIndex, Item, NUMBER, entity_swap,
                             norm, numeric_swap, polarity_flip, propose,
                             verify)

SUITE_V2_PATH = Path("data/bench/suite_v2.json")
REJECTS_V2_PATH = Path("data/bench/rejects_v2.json")

# ── the panel (T1) ──────────────────────────────────────────────────────────
# Six LLM judges across model families and context sizes, three deterministic
# tools with different lexical biases, three deliberately degenerate actors so
# the reliability machinery has something real to learn. Implementations live
# in bench/observers_v2.py; the ids here are the pre-registered contract.
OBSERVER_IDS: tuple[str, ...] = (
    "agent:qwen27@k8", "agent:qwen27@k3",
    "agent:qwen9@k8", "agent:qwen9@k3",
    "agent:qwen35moe@k8", "agent:mistral24@k8",
    "tool:exact@k8", "tool:exact@k3", "tool:span@k8",
    "agent:optimist", "agent:pessimist", "agent:coin",
)
PANEL_SIZE = 4

KIND_CAP_FRACTION = 1 / 3          # T2: no perturbation kind may exceed this
YEAR = re.compile(r"^(19|20)\d\d$")


def panel_for(claim_id: str, seed: int) -> list[str]:
    """Deterministic 4-of-12 assignment. Part of the artifact, not the run."""
    digest = hashlib.sha256(f"{seed}|panel|{claim_id}".encode()).hexdigest()
    rng = random.Random(int(digest[:12], 16))
    return sorted(rng.sample(OBSERVER_IDS, PANEL_SIZE))


# ── T2: the within-entry swap ───────────────────────────────────────────────

def _num_class(text: str) -> str:
    if YEAR.match(text):
        return "year"
    return "decimal" if "." in text else "int"


def within_entry_swap(fact: str, entry_text: str,
                      rng: random.Random) -> Optional[tuple[str, str]]:
    """Replace a number in the fact with a *different* number from the same
    entry. The substituted value is lexically present in the gold document, so
    a presence check passes while the (subject, value) binding is wrong — the
    shape of a real misreading, and the hard case v1 never posed."""
    fact_nums = [m for m in NUMBER.finditer(fact) if m.group(0) not in ("0", "1")]
    if not fact_nums:
        return None
    entry_nums = {m.group(0) for m in NUMBER.finditer(entry_text)}
    target = rng.choice(fact_nums)
    original = target.group(0)
    donors = []
    for cand in sorted(entry_nums):
        if cand in (original, "0", "1") or _num_class(cand) != _num_class(original):
            continue
        if _num_class(cand) != "year":
            try:
                if abs(math.log10(float(cand) / float(original))) > 2:
                    continue
            except (ValueError, ZeroDivisionError):
                continue
        donors.append(cand)
    if not donors:
        return None
    donor = rng.choice(donors)
    mutated = fact[:target.start()] + donor + fact[target.end():]
    return mutated, f"within: {original} -> {donor}"


# ── claim building with kind caps and exact balance ─────────────────────────

def build_claims_v2(items: list[Item], index: CorpusIndex, entry_text: dict[str, str],
                    seed: int) -> tuple[list[Claim], dict[str, int]]:
    rng = random.Random(seed)
    stats: dict[str, int] = {}

    def bump(key: str) -> None:
        stats[key] = stats.get(key, 0) + 1

    from .generate_suite import anchor_class, _substituted_value
    by_class: dict[str, list[str]] = {}
    for i in items:
        by_class.setdefault(anchor_class(i.anchor), []).append(i.anchor)

    n_false_target = len(items) // 2
    cap = math.ceil(n_false_target * KIND_CAP_FRACTION)
    kind_counts: dict[str, int] = {}

    def attempts_for(item: Item) -> list[tuple[str, tuple[str, str]]]:
        out: list[tuple[str, tuple[str, str]]] = []
        got = within_entry_swap(item.fact, entry_text.get(item.entry_id, ""), rng)
        if got:
            out.append(("within_entry_swap", got))
        got = numeric_swap(item.fact, rng)
        if got:
            out.append(("numeric_swap", got))
        pool = [a for a in by_class.get(anchor_class(item.anchor), [])
                if norm(a) != norm(item.anchor)]
        if pool:
            got = entity_swap(item.fact, rng.choice(pool), item.anchor)
            if got:
                out.append(("entity_swap", got))
        got = polarity_flip(item.fact)
        if got:
            out.append(("polarity_flip", got))
        return out

    def admissible(item: Item, kind: str, mutated: str, note: str) -> bool:
        if norm(mutated) == norm(item.fact):
            return False
        substituted = _substituted_value(note)
        gold_text = " ".join(index.normed.get(g, "") for g in item.gold_entries)
        # Provable falsity. For within_entry_swap the substituted value IS in
        # the gold entry by design — the proof there is that the mutated
        # *sentence* matches nothing corpus-wide and differs from the fact the
        # span verifies. For every other kind the stronger absence check holds.
        if kind != "within_entry_swap" and substituted and norm(substituted) in gold_text:
            bump("substitution_present_in_gold")
            return False
        if index.containing(mutated):
            bump("perturbation_matched_corpus")
            return False
        return True

    falsifiable: list[tuple[Item, Claim]] = []
    plain: list[Item] = []
    for item in items:
        made = None
        # prefer the scarcest kind so the caps fill evenly
        ranked = sorted(attempts_for(item),
                        key=lambda kv: (kind_counts.get(kv[0], 0), rng.random()))
        for kind, (mutated, note) in ranked:
            if kind_counts.get(kind, 0) >= cap:
                continue
            if not admissible(item, kind, mutated, note):
                continue
            made = Claim(claim_id=f"c{item.item_id[1:]}f", item_id=item.item_id,
                         entry_id=item.entry_id, text=mutated, outcome=False,
                         kind=kind, evidence_span=item.answer_span,
                         perturbation=note)
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            break
        (falsifiable.append((item, made)) if made else plain.append(item))

    claims: list[Claim] = []
    n_false = min(len(falsifiable), len(items) // 2)
    for item, claim in falsifiable[:n_false]:
        claims.append(claim)
        bump("false")
    for item, claim in falsifiable[n_false:]:
        kind_counts[claim.kind] -= 1          # unused perturbation, release cap
    truth_pool = [i for i, _ in falsifiable[n_false:]] + plain
    for item in truth_pool[:n_false]:
        claims.append(Claim(claim_id=f"c{item.item_id[1:]}t", item_id=item.item_id,
                            entry_id=item.entry_id, text=item.fact, outcome=True,
                            kind="verbatim", evidence_span=item.answer_span))
        bump("true")
    stats["falsifiable_pool"] = len(falsifiable)
    stats.update({f"kind:{k}": v for k, v in kind_counts.items()})
    claims.sort(key=lambda c: c.claim_id)
    return claims, stats


# ── driver ──────────────────────────────────────────────────────────────────

def main(n_items: int = 1600, seed: int = 20260726) -> None:
    entries = load(Path("data/bench/corpus.jsonl"))
    index = CorpusIndex(entries)
    entry_text = {e.entry_id: e.text for e in entries}
    rng = random.Random(seed)
    pool = [e for e in entries if len(e.text) >= 400]
    rng.shuffle(pool)
    sample = pool[:n_items]
    print(f"corpus {len(entries)} entries; sampling {len(sample)} "
          f"(v1 cache overlap makes repeats free)", flush=True)

    proposals = ollama.parallel(
        propose, sample, workers=1,
        on_progress=lambda d, t: (d % 50 == 0 or d == t) and
        print(f"  proposed {d}/{t}", flush=True))

    items: list[Item] = []
    rejects: list[dict[str, str]] = []
    for entry, obj in zip(sample, proposals):
        item, reason = verify(obj if isinstance(obj, dict) else {}, entry, index)
        (items.append(item) if item else
         rejects.append({"entry_id": entry.entry_id, "reason": reason}))

    claims, stats = build_claims_v2(items, index, entry_text, seed)
    panels = {c.claim_id: panel_for(c.claim_id, seed) for c in claims}

    reasons: dict[str, int] = {}
    for r in rejects:
        reasons[r["reason"][:48]] = reasons.get(r["reason"][:48], 0) + 1

    SUITE_V2_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUITE_V2_PATH.write_text(json.dumps({
        "version": 2, "seed": seed,
        "gen_model": ollama.GEN_MODEL,
        "observers": list(OBSERVER_IDS), "panel_size": PANEL_SIZE,
        "corpus_entries": len(entries), "sampled": len(sample),
        "items": [i.to_json() for i in items],
        "claims": [{**c.to_json(), "panel": panels[c.claim_id]} for c in claims],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    REJECTS_V2_PATH.write_text(json.dumps(
        {"reasons": reasons, "rejects": rejects}, indent=1), encoding="utf-8")

    n_true = sum(1 for c in claims if c.outcome)
    print(f"\nitems kept:  {len(items)}/{len(sample)} "
          f"({100 * len(items) / max(1, len(sample)):.0f}%)")
    print(f"claims:      {len(claims)} (true {n_true} / false {len(claims) - n_true})")
    print(f"stats:       {json.dumps(stats)}")


if __name__ == "__main__":
    import sys
    main(n_items=int(sys.argv[1]) if len(sys.argv) > 1 else 1600)
