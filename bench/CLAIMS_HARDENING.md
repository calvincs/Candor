# Claims hardening — staged log

A synthetic-world audit of the public claims (README.md, docs/use-cases.md,
docs/architecture.md), and the work to make the codebase meet them. Every stage
is one commit, with the measured numbers before and after, so any stage that
fails to make forward progress can be reverted on its own.

Method: worlds with **planted ground truth** and **null controls**. A detector
is only graded against both — recall on a planted signal is meaningless without
the false-positive rate on a world where nothing happened. Harness lives in
`tests/claims/`; run it with `make claims` (`CLAIMS_SCALE=4 make claims` for an
investigation-sized run).

---

## Stage 0 — baseline audit (before any change)

Existing suites at `5297a8c3`: **22 conformance + 1 xfail, 180 unit, all green.**
Nothing below is a conformance failure; the spec suite passes throughout. These
are gaps between what the code does and what the README says it does.

### Claims that held up, unchanged

| Claim | Measurement | Result |
|---|---|---|
| Asymmetric trust | 5 judges, known confusion matrices, 300 settled | sycophant LR(yes) **1.07**, specialist LR(yes) **29.8**, alarmist LR(no) **0.02**; learned rates within 0.02 of planted |
| Beats vote-averaging by 0.04 Brier | 300 held-out | **beat it by 0.099** (0.0209 vs 0.1200) |
| Near the composition ceiling | vs oracle Naive Bayes on true rates | **0.0012** Brier short of optimal |
| "0.83 is earned" | 400 held-out, weak judges | **ECE 0.0148, slope 0.998**, mean p 0.478 vs base rate 0.475 |
| Guards survive BH + MDL + holdout | 1 real + 5 noise covariates, 600 streams | recall 100% / 95%; **false-guard rate 0.8–5.8%** against α=0.05 |
| Changepoint *localization* | 2000 trials | **median error 1 observation in 120** |
| I1 replay, I8 snapshots, order-invariance | drop index / `predict_at` / shuffle | bit-exact |

### Defects found

| id | Claim broken | Measured |
|---|---|---|
| **F1** | "retract a poisoned source … as if it never spoke" (`docs/use-cases.md`) | `redact()` is keyed on a content-addressed payload hash that **excludes the actor**, so honest actors reporting the same outcome share it. Purging one bad source deleted **185 of 600 honest observations**. With the poison's payloads made artificially unique, divergence from a never-poisoned store is 0/25 — proving the blast radius is the whole defect |
| **F2** | `set_reliability()` as the soft discount lever (`docs/use-cases.md:18`) | **No-op on crisp facts** (`0.007812 → 0.007812`); works on frequency facts (`0.379 → 0.702`). The v0.3 Δ1 vote path reads `actor_confusion`, which only settlement moves; `set_reliability` writes `actor_reliability`. With F1 that leaves **no working lever** against a bad source on crisp facts |
| **F3** | "it finds the date … located, not decayed away" (README) | The sweep locates the changepoint to ~1 observation and then **discards it on write**. `apply.py:156` reads `body.get("valid_to", ev.ts)`; the curiosity body only carries `changepoint_index`. Committed `valid_to` landed **3 ms from the sweep time, 2018 ms from the true boundary** |
| **F4** | "regime changes are located" + "everything is gated" | Detection of a 0.95→0.05 break: **36%**. The recurrence veto kills 64% of true steps, because its inner CUSUM false-alarms **40% at p=0.95 and 43% at p=0.05 vs 3% at p=0.5** — worst exactly in the broken-tool regime. Meanwhile stationary p=0.9 streams yield **20% false positives**, and `gate.py:80` admits `supersede_valid_time` **unconditionally** — the only candidate kind with no validation, so periphery false positives become committed structural change |
| **F5** | "unexplained variance opens a question" (`docs/use-cases.md`) | A stream oscillating 85%/35% with no recorded context: CUSUM detects instability **492/500** and correctly rules it recurrent **480/500**, then reports **nothing** — the flag/question branch requires a recorded covariate to be independently overdispersed (`curiosity_engine.py:144`). 0/60 flags, 0/60 questions. The "log wider: the missing argument was never captured" message is unreachable from the sweep |

Minor, not tracked as a defect: replay eagerly derives sweep state that the live
store only computes inside `run_gate()`, so a store that has observed but not
gated gets a different closure hash after a rebuild. The rebuilt store is the
fresher one; `run_gate()` first and the hashes match.

---

## Stage 1 — land the claims suite

`tests/claims/` (worlds, metrics, tests) + `make claims` / `make claims-fast`.
28 tests encoding every claim above as a threshold on planted data.

Two thresholds were wrong in the first draft and were tightened rather than
loosened before being counted:

- learned-rate recovery now compares against the **prior-shrunk** expectation,
  not the raw planted rate — the sycophant's true fpr of 1.0 *should* read as
  0.91 under `FPR_PRIOR`, and asserting otherwise tests that the prior does not
  exist.
- calibration held-out set raised to 300 (ECE on 120 samples with 3 binary
  judges is bin noise, and it landed at 0.0501 against a 0.05 bar).

**Result: 17 pass, 11 fail.** The 11 are exactly F1–F5:

| failing test | defect |
|---|---|
| `test_finds_a_tool_that_broke[0.95-0.05]`, `[0.9-0.1]` | F4 power |
| `test_does_not_invent_breaks_in_stationary_streams[0.5]`, `[0.9]` | F4 FPR |
| `test_the_located_date_is_what_gets_committed` | F3 |
| `test_says_something_when_it_cannot_explain_the_variance` ×2 | F5 |
| `test_a_question_names_a_measurement_to_take` | F5 |
| `test_retracting_a_source_leaves_other_actors_untouched` | F1 |
| `test_retraction_reproduces_a_store_the_source_never_touched` | F1 |
| `test_discounting_a_source_moves_crisp_beliefs` | F2 |

Existing suites unchanged: 22 conformance + 1 xfail, 180 unit.
