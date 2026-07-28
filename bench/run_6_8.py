"""Test 6.8 — the honest test. Run after PREREGISTRATION.md is frozen.

Two arms, both scored against ground truth that came from a corpus scan:

  retrieval    CANDOR `recall` vs a bge-m3 embedding store over the same
               corpus. nDCG@10, recall@10, MRR.
  calibration  CANDOR (attributed observers, reliability learned from settled
               training claims, read-time isotonic) vs RAG-with-elicited-
               probabilities + isotonic. Brier, log loss, ECE, reliability slope.

A third, unrequested arm is reported because leaving it out would flatter
CANDOR: a NAIVE ENSEMBLE control that averages the same observers with no
reliability discount. The CANDOR-minus-control gap is what the trust machinery
buys; the CANDOR-minus-baseline gap includes the ensemble itself.

Held-out claims are never resolved in the store, so no ground truth leaks into
anything the system reads.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from candor.core import calibration as calib_mod          # noqa: E402
from candor.system import CandorSystem                    # noqa: E402

from . import metrics, ollama                             # noqa: E402
from .baseline import ElicitedRag, EmbeddingStore         # noqa: E402
from .corpus import Entry, load, write_evidence_tier      # noqa: E402
from .observers import Observers                          # noqa: E402

BENCH = Path("data/bench")
RESULTS = BENCH / "results_6_8.json"
K = 10
TRAIN_FRACTION = 0.5
ORACLE_ID = "verifier:corpus_exact"


def split_of(claim_id: str, seed: str = "20260725") -> str:
    h = hashlib.sha256((seed + claim_id).encode()).hexdigest()
    return "train" if int(h[:8], 16) % 100 < TRAIN_FRACTION * 100 else "heldout"


# ── arm 1: retrieval ────────────────────────────────────────────────────────

def run_retrieval(items: list[dict], entries: list[Entry],
                  store: EmbeddingStore) -> dict[str, Any]:
    root = BENCH / "candor_retrieval"
    shutil.rmtree(root, ignore_errors=True)
    system = CandorSystem(root)
    print(f"  writing evidence tier: {len(entries)} entries", flush=True)
    write_evidence_tier(entries, root / "evidence")

    # evidence filenames are sanitized entry ids; map chunk ids back to entries
    import re
    safe_to_entry = {re.sub(r"[^A-Za-z0-9._-]", "_", e.entry_id): e.entry_id
                     for e in entries}

    def candor_rank(question: str) -> list[str]:
        hits = system.recall(question, budget=200_000)
        ranked: list[str] = []
        for hit in hits:
            stem = hit["entry_id"].rsplit("#", 1)[0]
            eid = safe_to_entry.get(stem)
            if eid and eid not in ranked:
                ranked.append(eid)
        return ranked[:K]

    rows = []
    for n, item in enumerate(items, 1):
        gold = item["gold_entries"]
        c_rank = candor_rank(item["question"])
        b_rank = [eid for eid, _ in store.search(item["question"], K)]
        rows.append({
            "item_id": item["item_id"], "gold": gold,
            "candor": {"ndcg": metrics.ndcg_at_k(c_rank, gold, K),
                       "recall": metrics.recall_at_k(c_rank, gold, K),
                       "mrr": metrics.mrr(c_rank, gold)},
            "baseline": {"ndcg": metrics.ndcg_at_k(b_rank, gold, K),
                         "recall": metrics.recall_at_k(b_rank, gold, K),
                         "mrr": metrics.mrr(b_rank, gold)},
        })
        if n % 25 == 0:
            print(f"  retrieval {n}/{len(items)}", flush=True)
    system.close()

    out: dict[str, Any] = {"n": len(rows), "k": K}
    for metric in ("ndcg", "recall", "mrr"):
        c = [r["candor"][metric] for r in rows]
        b = [r["baseline"][metric] for r in rows]
        out[metric] = {
            "candor": sum(c) / len(c) if c else None,
            "baseline": sum(b) / len(b) if b else None,
            "paired_bootstrap": metrics.paired_bootstrap(c, b),
        }
    out["rows"] = rows
    return out


# ── arm 2: calibration ──────────────────────────────────────────────────────

def gather_votes(claims: list[dict], observers: Observers,
                 workers: int = 1) -> dict[str, list[dict]]:
    batches = observers.observe_batch(
        [c["text"] for c in claims], workers=workers,
        on_phase=lambda phase, done, total:
            print(f"  {phase} {done}/{total}", flush=True))
    return {c["claim_id"]: [{"actor": o.actor, "outcome": o.outcome, "ctx": o.ctx}
                            for o in batch]
            for c, batch in zip(claims, batches)}


def judge_agreement(claims: list[dict], votes: dict[str, list[dict]],
                    observers: Observers) -> dict[str, Any]:
    """How often the substituted judge agrees with laguna, where laguna is cached.

    Amendment 3 swapped the judge for throughput reasons. That is only
    defensible if the substitute is not a weaker judge in kind, so the overlap
    is measured rather than asserted. Uses only cached laguna responses — no
    new calls to the slow model.
    """
    from .observers import JUDGE_PROMPT, JUDGE_SYSTEM

    cache = ollama.Cache()
    agree = matched = legacy_right = new_right = 0
    for claim in claims:
        context, _ = observers.evidence(claim["text"])
        prompt = JUDGE_PROMPT.format(context=context[:24000], claim=claim["text"])
        hit = cache.get(ollama._key("gen", ollama.LEGACY_JUDGE_MODEL, prompt,
                                    JUDGE_SYSTEM, 80, 0.0))
        if hit is None:
            continue
        parsed = ollama.extract_json(hit)
        legacy = bool(parsed.get("true")) if isinstance(parsed, dict) \
            else ("true" in hit.lower()[:200])
        current = next((v["outcome"] for v in votes.get(claim["claim_id"], [])
                        if v["actor"] == "agent:llm_big"), None)
        if current is None:
            continue
        matched += 1
        agree += int(legacy == bool(current))
        legacy_right += int(legacy == bool(claim["outcome"]))
        new_right += int(bool(current) == bool(claim["outcome"]))
    if not matched:
        return {"overlap": 0}
    return {
        "overlap": matched,
        "agreement": agree / matched,
        "legacy_accuracy": legacy_right / matched,
        "current_accuracy": new_right / matched,
        "legacy_model": ollama.LEGACY_JUDGE_MODEL,
        "current_model": ollama.JUDGE_MODEL,
    }


def run_calibration(claims: list[dict], votes: dict[str, list[dict]],
                    rag: ElicitedRag) -> dict[str, Any]:
    root = BENCH / "candor_calibration"
    shutil.rmtree(root, ignore_errors=True)
    system = CandorSystem(root)
    system.register_oracle(ORACLE_ID, "deterministic_total",
                           impl_ref="bench.generate_suite.CorpusIndex",
                           code_hash="corpus-exact-match", env_hash="stdlib")

    # 1. every claim becomes an admitted crisp fact. Admission says the
    #    statement is well-formed; it says nothing about whether it is true.
    for claim in claims:
        system.assert_({"pred": "holds", "args": [claim["claim_id"]],
                        "stmt_type": "crisp"},
                       source=f"suite:{claim['item_id']}", actor="human:calvin")
    system.run_gate()
    print(f"  admitted {len(claims)} claim facts", flush=True)

    # 2. the observers' votes arrive as attributed observations.
    for claim in claims:
        for vote in votes.get(claim["claim_id"], []):
            system.observe({"pred": "holds", "args": [claim["claim_id"]]},
                           bool(vote["outcome"]), vote["ctx"], actor=vote["actor"])

    # 3. TRAIN claims settle through the deterministic oracle. This is the only
    #    place ground truth enters, and held-out claims never reach it (§3.12).
    train = [c for c in claims if split_of(c["claim_id"]) == "train"]
    heldout = [c for c in claims if split_of(c["claim_id"]) == "heldout"]
    for claim in train:
        cid = system.claim({"pred": "holds", "args": [claim["claim_id"]]},
                           frame="external", criterion=ORACLE_ID, due=0)
        if cid != "Refused":
            system.resolve(cid, outcome=bool(claim["outcome"]),
                           verifier_code_hash="corpus-exact-match",
                           env_hash="stdlib")
    print(f"  settled {len(train)} training claims -> reliability scored",
          flush=True)

    learned = {row["actor"]: row["rel_a"] / (row["rel_a"] + row["rel_b"])
               for row in system.index.query(
                   "SELECT actor, rel_a, rel_b FROM actor_reliability "
                   "WHERE frame='external'")}

    # 4. read-time isotonic, fitted on the training split only (§3.9, I8).
    raw_train = [(system.predict({"pred": "holds", "args": [c["claim_id"]]},
                                 budget=10_000).p, int(c["outcome"]))
                 for c in train]
    system._calib = calib_mod.fit_isotonic(raw_train)

    candor_pairs, control_pairs, baseline_pairs = [], [], []

    def naive(claim) -> float:
        v = votes.get(claim["claim_id"], [])
        return (sum(1 for x in v if x["outcome"]) / len(v)) if v else 0.5

    # One parallel pass over every claim, so the baseline's 117B elicitor is not
    # driven one request at a time. Same calls, same cache, same numbers.
    elicited_raw = ollama.parallel(
        lambda c: rag.elicit(c["text"])[0], claims, workers=1,
        on_progress=lambda d, t: (d % 25 == 0 or d == t) and
        print(f"  elicited {d}/{t}", flush=True))
    elicited = {c["claim_id"]: (p if isinstance(p, float) else 0.5)
                for c, p in zip(claims, elicited_raw)}

    baseline_train = [(elicited[c["claim_id"]], int(c["outcome"])) for c in train]
    control_train = [(naive(c), int(c["outcome"])) for c in train]
    baseline_map = calib_mod.fit_isotonic(baseline_train)
    control_map = calib_mod.fit_isotonic(control_train)

    for n, claim in enumerate(heldout, 1):
        y = int(claim["outcome"])
        p = system.predict({"pred": "holds", "args": [claim["claim_id"]]},
                           budget=10_000).p
        candor_pairs.append((p, y))
        control_pairs.append((control_map.apply(naive(claim)), y))
        baseline_pairs.append(
            (baseline_map.apply(elicited[claim["claim_id"]]), y))
        if n % 20 == 0:
            print(f"  scored {n}/{len(heldout)}", flush=True)

    per_item = {
        "candor": [(p - y) ** 2 for p, y in candor_pairs],
        "baseline": [(p - y) ** 2 for p, y in baseline_pairs],
        "control": [(p - y) ** 2 for p, y in control_pairs],
    }
    system.close()
    return {
        "n_train": len(train), "n_heldout": len(heldout),
        "learned_reliability": learned,
        "candor": metrics.summarize(candor_pairs),
        "baseline": metrics.summarize(baseline_pairs),
        "naive_ensemble_control": metrics.summarize(control_pairs),
        "brier_bootstrap_vs_baseline": metrics.paired_bootstrap(
            per_item["baseline"], per_item["candor"]),   # positive = CANDOR better
        "brier_bootstrap_vs_control": metrics.paired_bootstrap(
            per_item["control"], per_item["candor"]),
    }


# ── driver ──────────────────────────────────────────────────────────────────

def main() -> None:
    prereg = Path("bench/PREREGISTRATION.md")
    if not prereg.exists():
        raise SystemExit("refusing to run: bench/PREREGISTRATION.md is not frozen")

    entries = load(BENCH / "corpus.jsonl")
    suite = json.loads((BENCH / "suite.json").read_text(encoding="utf-8"))
    items, claims = suite["items"], suite["claims"]
    print(f"corpus {len(entries)}  items {len(items)}  claims {len(claims)}")

    vec_path = BENCH / "embeddings.jsonl"
    if vec_path.exists():
        store = EmbeddingStore.load(vec_path, entries)
        print("  loaded cached embeddings")
    else:
        print("  embedding corpus with bge-m3 ...")
        store = EmbeddingStore.build(entries)
        store.save(vec_path)

    print("\n== arm 1: retrieval ==")
    retrieval = run_retrieval(items, entries, store)

    print("\n== arm 2: calibration ==")
    observers = Observers(store)
    votes = gather_votes(claims, observers)
    calibration = run_calibration(claims, votes, ElicitedRag(store))
    calibration["judge_substitution_check"] = judge_agreement(
        claims, votes, observers)

    results = {
        "prereg_sha256": hashlib.sha256(prereg.read_bytes()).hexdigest(),
        "gen_model": ollama.GEN_MODEL, "judge_model": ollama.JUDGE_MODEL,
        "embed_model": ollama.EMBED_MODEL,
        "corpus_entries": len(entries),
        "retrieval": retrieval, "calibration": calibration,
    }
    RESULTS.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
