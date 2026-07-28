"""Test 6.8 v2 — sparse panels, off the ceiling, v0.3 mechanism under test.

Refuses to run until PREREGISTRATION_V2.md carries `STATUS: FROZEN`. The v1
result stands on the record for the build at 5b5cd57 / suite v1; this run
supersedes, never rewrites.

Reported systems (Δ3 makes the control mandatory forever):

  candor            the substrate as built: two-coin confusion learned from
                    settled TRAIN claims only, context-grouped composition,
                    read-time isotonic fitted on train.
  uniform control   identical pipeline, discount knob off: unweighted mean of
                    the same votes + the same train-fitted isotonic. If the
                    mechanism cannot beat its own degenerate case, that is the
                    result.
  baseline          §6.8's comparator: bge-m3 RAG + elicited probabilities +
                    isotonic.

Held-out claims are never resolved anywhere; ground truth enters only through
the train-split settlements.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from candor.core import calibration as calib_mod          # noqa: E402
from candor.system import CandorSystem                    # noqa: E402

from . import metrics, ollama                             # noqa: E402
from .baseline import ElicitedRag, EmbeddingStore         # noqa: E402
from .corpus import load, write_evidence_tier             # noqa: E402
from .observers_v2 import PanelObservers                  # noqa: E402
from .run_6_8 import split_of                             # noqa: E402

import os
BENCH = Path("data/bench")
VARIANT = os.environ.get("CANDOR_68_VARIANT", "v2")
PREREG = Path(f"bench/PREREGISTRATION_{VARIANT.upper()}.md")
RESULTS = BENCH / f"results_6_8_{VARIANT}.json"
SUITE = BENCH / ("suite_v2.json" if VARIANT == "v2" else f"suite_{VARIANT}.json")
K = 10
ORACLE_ID = "verifier:corpus_exact"


def require_frozen() -> str:
    if not PREREG.exists():
        raise SystemExit("refusing to run: bench/PREREGISTRATION_V2.md missing")
    text = PREREG.read_text(encoding="utf-8")
    if "STATUS: FROZEN" not in text:
        raise SystemExit("refusing to run: PREREGISTRATION_V2.md is not FROZEN")
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


# ── arm 1: retrieval (lexical + R1 vs bge-m3, larger n) ─────────────────────

def run_retrieval(items: list[dict], entries, store: EmbeddingStore) -> dict:
    import re
    root = BENCH / f"candor_retrieval_{VARIANT}"
    shutil.rmtree(root, ignore_errors=True)
    system = CandorSystem(root)
    write_evidence_tier(entries, root / "evidence")
    safe_to_entry = {re.sub(r"[^A-Za-z0-9._-]", "_", e.entry_id): e.entry_id
                     for e in entries}

    def candor_rank(question: str) -> list[str]:
        ranked: list[str] = []
        for hit in system.recall(question, budget=200_000):
            eid = safe_to_entry.get(hit["entry_id"].rsplit("#", 1)[0])
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
        if n % 100 == 0:
            print(f"  retrieval {n}/{len(items)}", flush=True)
    system.close()
    out: dict[str, Any] = {"n": len(rows), "k": K}
    for metric in ("ndcg", "recall", "mrr"):
        c = [r["candor"][metric] for r in rows]
        b = [r["baseline"][metric] for r in rows]
        out[metric] = {"candor": sum(c) / len(c), "baseline": sum(b) / len(b),
                       "paired_bootstrap": metrics.paired_bootstrap(c, b)}
    out["rows"] = rows
    return out


# ── arm 2: calibration under sparse panels ──────────────────────────────────

def run_calibration(claims: list[dict], votes: dict[str, list],
                    rag: ElicitedRag) -> dict[str, Any]:
    root = BENCH / f"candor_calibration_{VARIANT}"
    shutil.rmtree(root, ignore_errors=True)
    system = CandorSystem(root)
    system.register_oracle(ORACLE_ID, "deterministic_total",
                           impl_ref="bench.generate_suite.CorpusIndex",
                           code_hash="corpus-exact-match", env_hash="stdlib")
    # §3.12 quotas stay enforced; the seed proposer is provisioned for the
    # workload. v2's 1,194 assertions tripped the 500 default mid-run — the
    # flood defence working exactly as specified, on the wrong target.
    system.set_actor_quota("human:calvin", cand_per_epoch=len(claims) + 100)

    for claim in claims:
        system.assert_({"pred": "holds", "args": [claim["claim_id"]],
                        "stmt_type": "crisp"},
                       source=f"suite2:{claim['item_id']}", actor="human:calvin")
    system.run_gate()
    print(f"  admitted {len(claims)} claim facts", flush=True)

    for claim in claims:
        for vote in votes.get(claim["claim_id"], []):
            system.observe({"pred": "holds", "args": [claim["claim_id"]]},
                           bool(vote.outcome), vote.ctx, actor=vote.observer,
                           confidence=getattr(vote, "confidence", None))

    train = [c for c in claims if split_of(c["claim_id"]) == "train"]
    heldout = [c for c in claims if split_of(c["claim_id"]) == "heldout"]
    for n, claim in enumerate(train, 1):
        cid = system.claim({"pred": "holds", "args": [claim["claim_id"]]},
                           frame="external", criterion=ORACLE_ID, due=0)
        if cid != "Refused":
            system.resolve(cid, outcome=bool(claim["outcome"]),
                           verifier_code_hash="corpus-exact-match",
                           env_hash="stdlib")
        if n % 100 == 0:
            print(f"  settled {n}/{len(train)}", flush=True)

    learned_confusion = {
        row["actor"]: [int(row["tp"]), int(row["fn"]), int(row["fp"]),
                       int(row["tn"])]
        for row in system.index.query(
            "SELECT actor, tp, fn, fp, tn FROM actor_confusion "
            "WHERE frame='external' ORDER BY actor")}

    # read-time isotonic, train only (I8)
    raw_train = [(system.predict({"pred": "holds", "args": [c["claim_id"]]},
                                 budget=10_000).p, int(c["outcome"]))
                 for c in train]
    system._calib = calib_mod.fit_isotonic(raw_train)

    def naive(claim) -> float:
        """Knob-off control sees the SAME graded information (Δ3 fairness):
        mean of confidences where graded, of 0/1 where not."""
        v = votes.get(claim["claim_id"], [])
        if not v:
            return 0.5
        vals = [x.confidence if getattr(x, "confidence", None) is not None
                else (1.0 if x.outcome else 0.0) for x in v]
        return sum(vals) / len(vals)

    elicited_raw = ollama.parallel(
        lambda c: rag.elicit(c["text"])[0], claims, workers=1,
        on_progress=lambda d, t: (d % 100 == 0 or d == t) and
        print(f"  elicited {d}/{t}", flush=True))
    elicited = {c["claim_id"]: (p if isinstance(p, float) else 0.5)
                for c, p in zip(claims, elicited_raw)}

    control_map = calib_mod.fit_isotonic(
        [(naive(c), int(c["outcome"])) for c in train])
    baseline_map = calib_mod.fit_isotonic(
        [(elicited[c["claim_id"]], int(c["outcome"])) for c in train])

    candor_pairs, control_pairs, baseline_pairs = [], [], []
    for n, claim in enumerate(heldout, 1):
        y = int(claim["outcome"])
        candor_pairs.append(
            (system.predict({"pred": "holds", "args": [claim["claim_id"]]},
                            budget=10_000).p, y))
        control_pairs.append((control_map.apply(naive(claim)), y))
        baseline_pairs.append((baseline_map.apply(elicited[claim["claim_id"]]), y))
        if n % 100 == 0:
            print(f"  scored {n}/{len(heldout)}", flush=True)
    system.close()

    sq = {name: [(p - y) ** 2 for p, y in pairs] for name, pairs in
          (("candor", candor_pairs), ("control", control_pairs),
           ("baseline", baseline_pairs))}
    return {
        "n_train": len(train), "n_heldout": len(heldout),
        "learned_confusion": learned_confusion,
        "candor": metrics.summarize(candor_pairs),
        "baseline": metrics.summarize(baseline_pairs),
        "uniform_control": metrics.summarize(control_pairs),
        "brier_bootstrap_vs_baseline": metrics.paired_bootstrap(
            sq["baseline"], sq["candor"]),
        "brier_bootstrap_vs_control": metrics.paired_bootstrap(
            sq["control"], sq["candor"]),
    }


def main() -> None:
    prereg_hash = require_frozen()
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    entries = load(BENCH / "corpus.jsonl")
    items, claims = suite["items"], suite["claims"]
    print(f"suite v2: {len(items)} items, {len(claims)} claims, "
          f"panel size {suite['panel_size']} of {len(suite['observers'])}")

    store = EmbeddingStore.load(BENCH / "embeddings.jsonl", entries)

    print("\n== arm 1: retrieval ==")
    retrieval = run_retrieval(items, entries, store)

    print("\n== arm 2: calibration ==")
    observers = PanelObservers(store)
    votes = observers.observe_suite(claims, log=lambda m: print(m, flush=True))
    calibration = run_calibration(claims, votes, ElicitedRag(store))

    RESULTS.write_text(json.dumps({
        "prereg_sha256": prereg_hash,
        "suite": SUITE.name, "suite_seed": suite["seed"],
        "gen_model": suite["gen_model"], "judge_model": ollama.JUDGE_MODEL,
        "embed_model": ollama.EMBED_MODEL, "corpus_entries": len(entries),
        "retrieval": retrieval, "calibration": calibration,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
