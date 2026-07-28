# Test 6.8 v3 — pre-registration

STATUS: FROZEN — 2026-07-26, before any v3 result existed. The principal
selected the round-3 direction explicitly ("option 2 with 1 folded in") and
the margins are the same never-passed bars as v1/v2, unweakened and unmoved.

## Changes from v2 (evidence: FINDINGS F7/F8)

* **Δ6 graded observations.** LLM judges elicit a probability (the same prompt
  family the baseline's elicitor uses — no prompt advantage either way); the
  API bins it to an integer grade; composition uses the categorical response
  ledger (actor_response), learned only from train settlements, with fallback
  to the Δ1 binary LR below 10 scored responses.
* **Δ7 witness floor.** Panels are 2 seeded semantic judges + 2 seeded others
  (suite_v3.json, panel_seed 20260727). Same 1,194 items/claims and ground
  truth as suite v2 — only the panel policy changed.
* **Control fairness.** The uniform control receives the same graded values
  (mean of confidences), so beating it still isolates the reliability
  machinery, not the input upgrade.

## Systems, split, leakage: identical to v2's registration.

## Margins — unchanged, gating

    RETRIEVAL   nDCG@10 and recall@10 >= baseline − 0.05
    CALIBRATION brier(baseline) − brier(candor) >= 0.02
                bootstrap 95% CI lower bound > 0
                candor ECE <= 0.10
    CONTROL     brier(candor) <= brier(uniform_control) + 0.005

PASS = all bars hold. On failure: stop, report, diagnose; mechanisms and tests
iterate, margins never do.

## Reproduction

    CANDOR_68_VARIANT=v3 python -m bench.run_6_8_v2
    CANDOR_68_VARIANT=v3 python -m bench.verdict_v2 data/bench/results_6_8_v3.json
