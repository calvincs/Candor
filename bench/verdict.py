"""Apply the frozen §6.8 margins to a results file. Mechanical, not editorial.

The numbers below are transcribed from `bench/PREREGISTRATION.md` §4, which was
committed before any result existed (commit 35f7ebf). `check()` re-hashes that
file and refuses to render a verdict if the hash recorded in the results does
not match it — a moved goalpost voids the result rather than quietly passing.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PREREG = Path("bench/PREREGISTRATION.md")

# ── the frozen bars (PREREGISTRATION.md §4) ─────────────────────────────────
RETRIEVAL_NONINFERIORITY = 0.05      # nDCG@10 and recall@10
CALIBRATION_BRIER_MARGIN = 0.02      # baseline - candor, absolute
CALIBRATION_ECE_CEILING = 0.10       # standalone honesty bar


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)


def check(results: dict[str, Any]) -> tuple[bool, list[str]]:
    lines: list[str] = []
    recorded = results.get("prereg_sha256")
    actual = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if recorded != actual:
        return False, [
            "VOID — the pre-registration changed after the run.",
            f"  recorded {recorded}", f"  actual   {actual}"]

    r = results["retrieval"]
    c = results["calibration"]

    # ── retrieval: non-inferiority ─────────────────────────────────────────
    lines.append("RETRIEVAL (non-inferiority at "
                 f"{RETRIEVAL_NONINFERIORITY})   n={r['n']}, k={r['k']}")
    retrieval_ok = True
    for metric in ("ndcg", "recall"):
        cand = r[metric]["candor"]
        base = r[metric]["baseline"]
        floor = base - RETRIEVAL_NONINFERIORITY
        ok = cand >= floor
        retrieval_ok = retrieval_ok and ok
        lines.append(f"  {metric}@{r['k']:<3} candor {_fmt(cand)}  "
                     f"baseline {_fmt(base)}  floor {_fmt(floor)}  "
                     f"{'PASS' if ok else 'FAIL'}")
    lines.append(f"  mrr      candor {_fmt(r['mrr']['candor'])}  "
                 f"baseline {_fmt(r['mrr']['baseline'])}  (reported)")

    # ── calibration: superiority + honesty bar ─────────────────────────────
    lines.append("")
    lines.append(f"CALIBRATION   n_train={c['n_train']}, n_heldout={c['n_heldout']}")
    cb, bb = c["candor"]["brier"], c["baseline"]["brier"]
    delta = bb - cb
    boot = c["brier_bootstrap_vs_baseline"]
    margin_ok = delta >= CALIBRATION_BRIER_MARGIN
    ci_ok = boot["lo"] > 0.0
    ece = c["candor"]["ece"]
    ece_ok = ece is not None and ece <= CALIBRATION_ECE_CEILING

    lines.append(f"  brier      candor {_fmt(cb)}  baseline {_fmt(bb)}  "
                 f"delta {_fmt(delta)}  (need >= {CALIBRATION_BRIER_MARGIN})  "
                 f"{'PASS' if margin_ok else 'FAIL'}")
    lines.append(f"  bootstrap  95% CI [{_fmt(boot['lo'])}, {_fmt(boot['hi'])}]  "
                 f"lower bound > 0  {'PASS' if ci_ok else 'FAIL'}")
    lines.append(f"  ece        candor {_fmt(ece)}  "
                 f"(need <= {CALIBRATION_ECE_CEILING})  "
                 f"{'PASS' if ece_ok else 'FAIL'}")
    lines.append(f"  log_loss   candor {_fmt(c['candor']['log_loss'])}  "
                 f"baseline {_fmt(c['baseline']['log_loss'])}  (reported)")
    lines.append(f"  slope      candor {_fmt(c['candor']['reliability_slope'])}  "
                 f"baseline {_fmt(c['baseline']['reliability_slope'])}  (reported)")
    calibration_ok = margin_ok and ci_ok and ece_ok

    # ── secondary, reported not gating ─────────────────────────────────────
    ctrl = c["naive_ensemble_control"]
    cboot = c["brier_bootstrap_vs_control"]
    lines.append("")
    lines.append("SECONDARY (reported, not gating)")
    lines.append(f"  vs naive ensemble  control brier {_fmt(ctrl['brier'])}  "
                 f"delta {_fmt(ctrl['brier'] - cb)}  "
                 f"95% CI [{_fmt(cboot['lo'])}, {_fmt(cboot['hi'])}]")
    lines.append("  learned reliability (should rank tool:exact above agent:optimist):")
    for actor, rel in sorted(c["learned_reliability"].items(),
                             key=lambda kv: -kv[1]):
        lines.append(f"    {actor:<20} E[rel] {_fmt(rel)}")

    passed = retrieval_ok and calibration_ok
    lines.append("")
    lines.append(f"VERDICT: 6.8 {'PASSES' if passed else 'FAILS'}"
                 f"   (retrieval {'pass' if retrieval_ok else 'FAIL'}, "
                 f"calibration {'pass' if calibration_ok else 'FAIL'})")
    return passed, lines


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/bench/results_6_8.json")
    results = json.loads(path.read_text(encoding="utf-8"))
    passed, lines = check(results)
    print("\n".join(lines))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
