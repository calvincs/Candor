"""Apply the frozen 6.8-v2 margins. Mechanical, not editorial.

Bars transcribed from bench/PREREGISTRATION_V2.md (frozen 2026-07-26, before
any v2 result existed). Refuses to rule if that file's hash no longer matches
the one recorded in the results — a moved goalpost voids the result.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import os
PREREG = Path(f"bench/PREREGISTRATION_{os.environ.get('CANDOR_68_VARIANT', 'v2').upper()}.md")

RETRIEVAL_NONINFERIORITY = 0.05
CALIBRATION_BRIER_MARGIN = 0.02
CALIBRATION_ECE_CEILING = 0.10
CONTROL_TOLERANCE = 0.005            # Δ3, gating


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)


def check(results: dict[str, Any]) -> tuple[bool, list[str]]:
    lines: list[str] = []
    actual = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if results.get("prereg_sha256") != actual:
        return False, ["VOID — the pre-registration changed after the run.",
                       f"  recorded {results.get('prereg_sha256')}",
                       f"  actual   {actual}"]

    r, c = results["retrieval"], results["calibration"]

    lines.append(f"RETRIEVAL (non-inferiority at {RETRIEVAL_NONINFERIORITY})"
                 f"   n={r['n']}, k={r['k']}")
    retrieval_ok = True
    for metric in ("ndcg", "recall"):
        cand, base = r[metric]["candor"], r[metric]["baseline"]
        ok = cand >= base - RETRIEVAL_NONINFERIORITY
        retrieval_ok &= ok
        lines.append(f"  {metric}@{r['k']:<3} candor {_fmt(cand)}  baseline "
                     f"{_fmt(base)}  floor {_fmt(base - RETRIEVAL_NONINFERIORITY)}"
                     f"  {'PASS' if ok else 'FAIL'}")
    lines.append(f"  mrr      candor {_fmt(r['mrr']['candor'])}  baseline "
                 f"{_fmt(r['mrr']['baseline'])}  (reported)")

    cb, bb = c["candor"]["brier"], c["baseline"]["brier"]
    ctrl = c["uniform_control"]["brier"]
    boot = c["brier_bootstrap_vs_baseline"]
    delta = bb - cb
    margin_ok = delta >= CALIBRATION_BRIER_MARGIN
    ci_ok = boot["lo"] > 0.0
    ece = c["candor"]["ece"]
    ece_ok = ece is not None and ece <= CALIBRATION_ECE_CEILING
    control_ok = cb <= ctrl + CONTROL_TOLERANCE

    lines.append("")
    lines.append(f"CALIBRATION   n_train={c['n_train']}, n_heldout={c['n_heldout']}")
    lines.append(f"  brier      candor {_fmt(cb)}  baseline {_fmt(bb)}  delta "
                 f"{_fmt(delta)}  (need >= {CALIBRATION_BRIER_MARGIN})  "
                 f"{'PASS' if margin_ok else 'FAIL'}")
    lines.append(f"  bootstrap  95% CI [{_fmt(boot['lo'])}, {_fmt(boot['hi'])}]  "
                 f"lower bound > 0  {'PASS' if ci_ok else 'FAIL'}")
    lines.append(f"  ece        candor {_fmt(ece)}  (need <= "
                 f"{CALIBRATION_ECE_CEILING})  {'PASS' if ece_ok else 'FAIL'}")
    lines.append(f"  control    candor {_fmt(cb)}  vs uniform {_fmt(ctrl)}  "
                 f"(need <= control + {CONTROL_TOLERANCE})  "
                 f"{'PASS' if control_ok else 'FAIL'}   [Δ3, gating]")
    lines.append(f"  log_loss   candor {_fmt(c['candor']['log_loss'])}  baseline "
                 f"{_fmt(c['baseline']['log_loss'])}  control "
                 f"{_fmt(c['uniform_control']['log_loss'])}  (reported)")
    lines.append(f"  slope      candor {_fmt(c['candor']['reliability_slope'])}  "
                 f"baseline {_fmt(c['baseline']['reliability_slope'])}  control "
                 f"{_fmt(c['uniform_control']['reliability_slope'])}  (reported)")

    cboot = c["brier_bootstrap_vs_control"]
    lines.append("")
    lines.append("SECONDARY (reported)")
    lines.append(f"  vs control bootstrap 95% CI [{_fmt(cboot['lo'])}, "
                 f"{_fmt(cboot['hi'])}]  (positive = candor better)")

    calibration_ok = margin_ok and ci_ok and ece_ok and control_ok
    passed = retrieval_ok and calibration_ok
    lines.append("")
    lines.append(f"VERDICT: 6.8-v2 {'PASSES' if passed else 'FAILS'}"
                 f"   (retrieval {'pass' if retrieval_ok else 'FAIL'}, "
                 f"calibration {'pass' if calibration_ok else 'FAIL'})")
    return passed, lines


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else
                "data/bench/results_6_8_v2.json")
    passed, lines = check(json.loads(path.read_text(encoding="utf-8")))
    print("\n".join(lines))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
