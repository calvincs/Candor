"""Offline reanalysis of the failed §6.8 calibration arm (investigation #3).

No model calls. Everything recomputes from artifacts already on disk: the
suite's ground truth and the calibration store's ledger-derived observations.
That this is possible at all is the substrate doing its job — every vote is an
attributed observation event, so retroactive reanalysis is a pure recompute
(I1/I11).

Question: the naive ensemble control (same four observers, no reliability
discount) beat the reliability-discounted composition. Why, and is there a
composition rule that beats knob-off?

Hypothesis: the spec's one-coin reliability (a single Beta per actor, symmetric
in vote direction) cannot represent asymmetric observers. An always-yes
optimist has sensitivity ~1.0 and specificity ~0; one-coin models it as a
0.55-accurate coin and discounts BOTH directions of its vote equally, when its
TRUE votes carry no information and its FALSE votes would be decisive. The
two-coin alternative (Dawid–Skene): per-actor confusion counts (TP, FN, FP,
TN — four integers, I11-compliant), learned only from trusted settlements
(§3.12), composed at read time by log-likelihood ratio.

IMPORTANT CAVEAT, stated up front: this is exploratory reanalysis of the run
that exposed the problem. Whatever wins here is a *candidate* mechanism to be
confirmed on a fresh, larger suite under a new pre-registration — not a result.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from candor.core.calibration import fit_isotonic                    # noqa: E402

from . import metrics                                               # noqa: E402
from .run_6_8 import split_of                                       # noqa: E402

BENCH = Path("data/bench")
ACTORS = ("tool:exact", "agent:llm_big", "agent:llm_small", "agent:optimist")

# Reproduce the run's parameters
ONE_COIN_PRIOR = (19.0, 1.0)          # reliability.REL_PRIOR_A/B
EPI_PRIOR = (1.0, 1.0)                # crisp prior after D16


# ── data: votes from the substrate's own ledger, truth from the suite ───────

def load() -> tuple[dict[str, dict[str, bool]], dict[str, bool], dict[str, str]]:
    suite = json.loads((BENCH / "suite.json").read_text(encoding="utf-8"))
    truth = {c["claim_id"]: bool(c["outcome"]) for c in suite["claims"]}
    kind = {c["claim_id"]: c["kind"] for c in suite["claims"]}

    db = sqlite3.connect(str(BENCH / "candor_calibration" / "index.sqlite3"))
    db.row_factory = sqlite3.Row
    fid2claim = {}
    for row in db.execute("SELECT id, args_json FROM facts"):
        args = json.loads(row["args_json"])
        if args and args[0] in truth:
            fid2claim[row["id"]] = args[0]
    votes: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in db.execute("SELECT fact_id, actor, outcome FROM observations"):
        cid = fid2claim.get(row["fact_id"])
        if cid is not None and row["actor"] in ACTORS:
            votes[cid][row["actor"]] = bool(row["outcome"])
    db.close()
    return dict(votes), truth, kind


# ── the three composition rules, all trained on the train split only ────────

def learn_confusion(train: list[str], votes, truth) -> dict[str, dict[str, int]]:
    out = {a: {"tp": 0, "fn": 0, "fp": 0, "tn": 0} for a in ACTORS}
    for cid in train:
        for actor, vote in votes.get(cid, {}).items():
            cell = ("tp" if vote else "fn") if truth[cid] else ("fp" if vote else "tn")
            out[actor][cell] += 1
    return out


def one_coin_p(votes_c: dict[str, bool], rel: dict[str, float]) -> float:
    """The run's rule: reliability-discounted vote counts + crisp Beta(1,1)."""
    a, b = EPI_PRIOR
    for actor, vote in votes_c.items():
        r = rel[actor]
        a += r * (1 if vote else 0)
        b += r * (0 if vote else 1)
    return a / (a + b)


def naive_p(votes_c: dict[str, bool]) -> float:
    if not votes_c:
        return 0.5
    return sum(1 for v in votes_c.values() if v) / len(votes_c)


def two_coin_p(votes_c: dict[str, bool], conf: dict[str, dict[str, int]],
               prior_logodds: float) -> float:
    """Naive-Bayes over observers with add-one smoothed confusion rates.

    Conditional independence given truth is assumed and is knowingly violated
    (the two LLMs share retrieval context); the read-time isotonic map is what
    absorbs the resulting overconfidence, exactly as it absorbs the raw
    composition's biases in the shipped design.
    """
    logodds = prior_logodds
    for actor, vote in votes_c.items():
        c = conf[actor]
        sens = (c["tp"] + 1.0) / (c["tp"] + c["fn"] + 2.0)   # P(vote T | true)
        fpr = (c["fp"] + 1.0) / (c["fp"] + c["tn"] + 2.0)    # P(vote T | false)
        if vote:
            logodds += math.log(sens / fpr)
        else:
            logodds += math.log((1.0 - sens) / (1.0 - fpr))
    return 1.0 / (1.0 + math.exp(-logodds))


# ── evaluation: identical treatment for every rule ──────────────────────────

def evaluate(name: str, raw: dict[str, float], train: list[str],
             heldout: list[str], truth) -> tuple[dict, list[float]]:
    iso = fit_isotonic([(raw[c], int(truth[c])) for c in train])
    pairs = [(iso.apply(raw[c]), int(truth[c])) for c in heldout]
    summary = metrics.summarize(pairs)
    per_item_sq = [(p - y) ** 2 for p, y in pairs]
    return {"rule": name, **summary}, per_item_sq


def main() -> None:
    votes, truth, kind = load()
    claims = sorted(truth)
    train = [c for c in claims if split_of(c) == "train"]
    heldout = [c for c in claims if split_of(c) == "heldout"]
    print(f"claims {len(claims)} (train {len(train)} / heldout {len(heldout)}), "
          f"votes for {len(votes)} claims\n")

    conf = learn_confusion(train, votes, truth)
    rel = {a: (ONE_COIN_PRIOR[0] + conf[a]["tp"] + conf[a]["tn"]) /
              (sum(ONE_COIN_PRIOR) + sum(conf[a].values())) for a in ACTORS}
    base = sum(1 for c in train if truth[c]) / len(train)
    prior_logodds = math.log(base / (1.0 - base))

    print("── per-actor parameters learned from the train split ──")
    print(f"{'actor':<18} {'one-coin rel':>12} {'sens':>7} {'fpr':>7} "
          f"{'LR(T vote)':>11} {'LR(F vote)':>11}")
    for a in ACTORS:
        c = conf[a]
        sens = (c["tp"] + 1) / (c["tp"] + c["fn"] + 2)
        fpr = (c["fp"] + 1) / (c["fp"] + c["tn"] + 2)
        print(f"{a:<18} {rel[a]:>12.3f} {sens:>7.3f} {fpr:>7.3f} "
              f"{sens/fpr:>11.2f} {(1-sens)/(1-fpr):>11.2f}")

    rules = {
        "one_coin (as run)": {c: one_coin_p(votes.get(c, {}), rel) for c in claims},
        "naive mean (control)": {c: naive_p(votes.get(c, {})) for c in claims},
        "two_coin LR": {c: two_coin_p(votes.get(c, {}), conf, prior_logodds)
                        for c in claims},
    }

    print("\n── held-out performance (train-fitted isotonic applied to all) ──")
    print(f"{'rule':<22} {'brier':>7} {'logloss':>8} {'acc':>6} {'ece':>7} {'slope':>7}")
    per_item: dict[str, list[float]] = {}
    for name, raw in rules.items():
        row, sq = evaluate(name, raw, train, heldout, truth)
        per_item[name] = sq
        print(f"{name:<22} {row['brier']:>7.4f} {row['log_loss']:>8.4f} "
              f"{row['accuracy']:>6.3f} {row['ece']:>7.4f} "
              f"{row['reliability_slope']:>7.3f}")

    boot = metrics.paired_bootstrap(per_item["naive mean (control)"],
                                    per_item["two_coin LR"])
    print(f"\ntwo_coin vs control, held-out Brier delta "
          f"{boot['delta']:+.4f}  95% CI [{boot['lo']:+.4f}, {boot['hi']:+.4f}]  "
          f"(positive = two_coin better)")
    boot2 = metrics.paired_bootstrap(per_item["one_coin (as run)"],
                                     per_item["two_coin LR"])
    print(f"two_coin vs one_coin, held-out Brier delta "
          f"{boot2['delta']:+.4f}  95% CI [{boot2['lo']:+.4f}, {boot2['hi']:+.4f}]")

    # ── where the recoverable information lives: within-count discrimination ─
    print("\n── truth rate by vote pattern (all 240 claims; the info naive "
          "mean cannot see) ──")
    patterns: dict[tuple, list[bool]] = defaultdict(list)
    for c in claims:
        v = votes.get(c, {})
        if len(v) == len(ACTORS):
            key = tuple(v[a] for a in ACTORS)
            patterns[key].append(truth[c])
    print(f"{'exact':>6} {'llm_big':>8} {'llm_small':>10} {'optimist':>9} "
          f"{'n':>4} {'truth rate':>11}")
    for key, outcomes in sorted(patterns.items(),
                                key=lambda kv: -len(kv[1]))[:10]:
        rate = sum(outcomes) / len(outcomes)
        marks = ["T" if x else "F" for x in key]
        print(f"{marks[0]:>6} {marks[1]:>8} {marks[2]:>10} {marks[3]:>9} "
              f"{len(outcomes):>4} {rate:>11.2f}")

    # ── observer accuracy by claim kind, and LLM error correlation ──────────
    print("\n── observer accuracy by claim kind (all claims) ──")
    kinds = sorted({k for k in kind.values()})
    print(f"{'actor':<18}" + "".join(f"{k:>15}" for k in kinds))
    for a in ACTORS:
        cells = []
        for kd in kinds:
            sub = [c for c in claims if kind[c] == kd and a in votes.get(c, {})]
            acc = (sum(1 for c in sub if votes[c][a] == truth[c]) / len(sub)
                   if sub else float("nan"))
            cells.append(f"{acc:>15.3f}")
        print(f"{a:<18}" + "".join(cells))

    both = [c for c in claims
            if {"agent:llm_big", "agent:llm_small"} <= votes.get(c, {}).keys()]
    e1 = [votes[c]["agent:llm_big"] != truth[c] for c in both]
    e2 = [votes[c]["agent:llm_small"] != truth[c] for c in both]
    n11 = sum(1 for a, b in zip(e1, e2) if a and b)
    n10 = sum(1 for a, b in zip(e1, e2) if a and not b)
    n01 = sum(1 for a, b in zip(e1, e2) if not a and b)
    n00 = sum(1 for a, b in zip(e1, e2) if not a and not b)
    denom = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    phi = (n11 * n00 - n10 * n01) / denom if denom else float("nan")
    print(f"\nLLM error correlation (phi): {phi:.3f}  "
          f"(errors: big {sum(e1)}, small {sum(e2)}, joint {n11} of {len(both)})")

    # ── retrieval miss decomposition (for the #2 discussion) ────────────────
    results = json.loads((BENCH / "results_6_8.json").read_text(encoding="utf-8"))
    rows = results["retrieval"]["rows"]
    c_absent = sum(1 for r in rows if r["candor"]["recall"] == 0.0)
    b_absent = sum(1 for r in rows if r["baseline"]["recall"] == 0.0)
    c_low = sum(1 for r in rows
                if r["candor"]["recall"] > 0 and r["candor"]["mrr"] < 0.5)
    print(f"\n── retrieval miss decomposition (n={len(rows)}) ──")
    print(f"gold entirely absent from top-10:  candor {c_absent}  "
          f"baseline {b_absent}")
    print(f"gold present but ranked below 2nd: candor {c_low}")

    # ── how big must suite v2 be for the frozen 0.02 margin? ────────────────
    diffs = [a - b for a, b in zip(per_item["naive mean (control)"],
                                   per_item["two_coin LR"])]
    mean_d = sum(diffs) / len(diffs)
    var_d = sum((d - mean_d) ** 2 for d in diffs) / max(1, len(diffs) - 1)
    sd = math.sqrt(var_d)
    for target in (0.01, 0.02):
        # 95% CI excluding 0 with ~80% power: need SE ≈ target / 2.8
        n_needed = (2.8 * sd / target) ** 2 if target else float("inf")
        print(f"\nper-item Brier-diff SD {sd:.4f}: to resolve a {target:.2f} "
              f"delta (95% CI, ~80% power) need ~{math.ceil(n_needed)} "
              f"held-out claims")


if __name__ == "__main__":
    main()
