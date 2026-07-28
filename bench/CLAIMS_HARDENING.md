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

---

## Stage 2 — F1: retraction is keyed on the source, not on a payload hash

**Diagnosis.** `redact()` takes a *content-addressed payload hash*, and an
observation payload is `{stmt, outcome, ctx, grade, confidence}` — no actor in
it. Two sources reporting the same outcome on the same statement in the same
context therefore share one payload. Purging a liar's hashes destroyed **185 of
600 honest observations**. That is not a corner case: agreement between sources
is the situation the whole substrate exists to reason about, so the documented
recovery path is most destructive exactly when it is most needed.

**Fix.** A new `retraction` event kind, keyed on the actor. Exclusion happens at
fold time in `_refold`: a retracted actor's event *skeletons* stay in the chain
forever (nothing is erased, I3 holds) but contribute no payload, so every
downstream number recomputes without them — including trust, because
`_apply_resolution` re-runs `score_against_settlement` on replay. Append-only
and reversible via `restore=True`; last write per actor wins.

`redact()` keeps its content-purging meaning (that is a real and separate need —
sensitive payloads) and is now honest about its scope: `redaction_scope(hash)`
reports the blast radius, and redacting a shared payload records a
`redaction_shared_payload` diagnostic.

| | before | after |
|---|---|---|
| honest observations destroyed by purging one source | 185 / 600 | **0** |
| predictions matching a store the source never touched | 0 / 12 | **12 / 12** |

`test_retracting_a_source_leaves_other_actors_untouched` and
`test_retraction_reproduces_a_store_the_source_never_touched` now pass.
Regression check: **202 passed, 1 xfailed** (conformance + unit), unchanged.

---

## Stage 3 — F2: the discount lever reaches crisp facts

**Diagnosis.** v0.3 Δ1 replaced the epistemic Beta with attributed two-coin
votes for crisp facts. Those votes read `actor_confusion`, which only a
settlement moves. `set_reliability` writes `actor_reliability` — still consulted
on the frequency path, no longer consulted on the crisp path. So the lever
`docs/use-cases.md` hands the operator for a hallucinating scraper silently did
nothing to exactly the statement type that example is about.

**Fix.** `reliability.temper(log_lr, weight)` scales a vote's evidence in
log-odds space — the same device Δ2 already uses to price correlated votes
sub-additively. Applied per actor in the crisp composition, so it covers the Δ1
binary path and the Δ6 graded path identically.

Only *explicit* overrides temper anything. Learned reliability already speaks
through the confusion ledger, and folding E[rel] in as well would double-count
the same settlements; a store with no overrides therefore returns `{}` and
composes byte-identically to before. That is what keeps this from being a
behaviour change for every existing store.

| `set_reliability(poison, external, 0.001, 100)` | before | after |
|---|---|---|
| crisp fact | 0.007812 → 0.007812 (**no-op**) | 0.007812 → **0.298828** |
| frequency fact | 0.379 → 0.702 | 0.379 → 0.702 (unchanged) |

Note the residual, which is correct behaviour rather than a defect: an
*unsettled* source's vote still composes at ~19:1, because `SENS_PRIOR`/
`FPR_PRIOR` deliberately assume newcomers are informative ("evidence must be
able to speak before it has been scored"). The operator now has a lever to
overrule that; nothing about the prior changed.

`test_discounting_a_source_moves_crisp_beliefs` now passes.
Regression check: **202 passed, 1 xfailed**, unchanged.

A first draft of this stage broke the §6.2 lexical firewall (`grep -rn weight`
outside `core/committed/`) in three doc comments. Caught by `tests/unit/
test_audit.py`, reworded, re-verified. Recording it because that invariant
exists precisely to catch careless naming, and it did.

---

## Stage 4 — F3: commit the located date, not the sweep's wall clock

**Diagnosis.** The sweep locates a changepoint to a median of 1 observation in
120 — the localization was never the problem. `apply.py` writes
`valid_to = body.get("valid_to", ev.ts)`, and the curiosity body carried only
`changepoint_index`. Every located regime change was therefore stamped with the
moment the sweep happened to run.

**Fix.** The sweep now joins `events.ts` and carries `valid_to` (the timestamp
of the observation at the located changepoint — the last moment the old regime
held), plus `changepoint_event_seq` and per-side `support` for audit. Replay-safe:
the timestamp comes from the ledger, so the sweep stays a deterministic function
of the log.

Measured with a real 2-second gap planted between regimes:

| | before | after |
|---|---|---|
| `valid_to` vs the true boundary | 2018 ms off | **0 ms** (the last observation of the old regime) |
| `valid_to` vs the sweep's clock | 3 ms off | 4022 ms off |

`test_the_located_date_is_what_gets_committed` now passes.
Regression check: **202 passed, 1 xfailed**, unchanged.

---

## Stage 5 — F4: an exact changepoint test, and a gate that actually gates

**Diagnosis, part 1 (the statistics).** `cusum_changepoint` normalises its alarm
by `sqrt(p(1-p))`, a Gaussian approximation. Bernoulli increments are badly
skewed near 0 and 1, so at p=0.95 a single failure moves the statistic by ~4.4
normalised units against an alarm level of 5 — two failures anywhere in 60
samples trip it. Measured false-alarm rate of that inner test on a **stationary**
segment: 40% at p=0.95, 43% at p=0.05, 3% at p=0.5. It was used as the
*recurrence veto*, so it destroyed 64% of genuine 0.95→0.05 breaks and 48% of
0.9→0.1 breaks — the failure was worst exactly in the broken-tool regime the
feature exists for.

**Diagnosis, part 2 (the gate).** `gate.py` admitted `supersede_valid_time`
unconditionally — the only candidate kind with no evaluation at all. So the
periphery's ~20% false-positive rate on stationary streams became committed
structural change, closing the validity window of facts that never changed.

**Fix.** Localization was never the problem (median error 1 in 120), so
`locate_changepoint` keeps the argmax-of-cumulative-deviation estimator
verbatim. What changed is the *significance* decision around it:

- `fisher_exact` — two-sided hypergeometric p-value in log space, exact at any
  base rate. Validated against an exact-integer reference to 1e-9.
- `changepoint_test` — locate, then test the two segments, Bonferroni-corrected
  by the number of split positions searched (the argmax is a maximum, so the
  naive p-value is anti-conservative by exactly that).
- `is_recurrent` — same exact machinery applied within each segment, so the
  "did it change again?" question stops depending on the base rate.
- `gate._evaluate_supersede` — steps 1/5/6: the fact must exist, the proposal
  must carry a located date, both regimes need ≥8 observations, and the level
  change must clear `SUPERSEDE_ALPHA` after correction. Tighter than the guard's
  BH α because a false supersede rewrites history while a false guard is only a
  rejected candidate.

200 replications per condition, 120 observations each, break planted at 60:

| world | detection before | detection after | |
|---|---|---|---|
| step 0.9 → 0.1 | 0.52 | **1.00** | the flagship case |
| step 0.75 → 0.35 | 0.64 | **0.77** | |
| step 0.6 → 0.4 | 0.22 | 0.06 | see note |
| **flat p=0.5** (null) | 0.08 | **0.00** | |
| **flat p=0.9** (null) | 0.20 | **0.00** | |
| **oscillating** (null) | 0.21 | **0.05** | |
| localization, median / p90 | 1 / 1 | **1 / 2** | unchanged |

Strict improvement on every world that matters: the flagship detection doubled
while both null worlds went to zero.

Two honest notes rather than wins:

- **step 0.6 → 0.4 fell to 0.06.** A 0.2 shift over 120 observations, with the
  changepoint *searched* rather than known, genuinely is not significant. The
  old 0.22 was not power, it was noise — its median localization error was 5
  observations with a p90 of 18, i.e. it was mostly "detecting" the break in the
  wrong place. Honest low power beats confident mislocation.
- **gradual drift now reads as a step 0.93 of the time** (was 0.65). A monotone
  slide has no single date, so this is a mischaracterisation of *shape* — but it
  is not a false positive: the level really did change and the old regime really
  has stopped holding. Recorded as accepted behaviour, not as a fixed defect.

New unit coverage: 5 gate fixtures for the supersede step (admit, no date, thin
regime, insignificant, unknown fact) and 5 statistics tests including a
base-rate flatness check that pins the exact defect this stage removed.

All 8 `TestRegimeChange` claims pass. Conformance + unit: **216 passed,
1 xfailed** (up from 202; 14 new unit tests, none removed).

Three of my own new unit tests failed on first run. All three were bad test
authoring, verified against independent references before changing anything: a
hand-typed Fisher constant that was simply wrong (0.0027972 vs the true
0.0027594562), a per-trial localization bound that ignored the tail (median 0,
p90 2, max 6), and an `is_recurrent` assertion on a series the primary test
never routes there. Noted because "the test failed so the code is wrong" is the
trap this whole exercise is meant to avoid.

---

## Stage 6 — F5: say something when nothing explains the variance
## and F6: Tarone's Z had the wrong denominator

**F5 diagnosis.** The flag/question branch was gated on `and tested` — at least
one *recorded covariate* had to be independently overdispersed. A stream
swinging between 85% and 35% with no useful context logged therefore produced
nothing: no flag, no question, no signal of any kind. The time axis had already
detected the instability (CUSUM alarmed 492/500 and correctly ruled it recurrent
480/500) and the routing threw that away. The `suggested_measurement([])` branch
— "log wider: the missing argument was never captured" — was unreachable code.

**F5 fix.** `temporal_dispersion`: overdispersion across contiguous time blocks
at several scales, corrected for the scales tried. It needs no covariate,
because the variance is visible in the series itself. The routing now speaks on
either ground — a covariate that clusters the variance without clearing the
guard bar, *or* dispersion over time — and the residual partition for the second
case is the time blocks themselves.

Identical instability (p=0.85/0.35, blocks of 20, 240 observations, 60 streams),
varying only what the agent happened to log:

| what was logged | flagged before | flagged after | the agent is told |
|---|---|---|---|
| nothing at all | **0 / 60** | **60 / 60** | "log wider: the disagreeing observations share no recorded covariate" |
| pure-noise context | 1 / 60 | 58 / 60 | "variance clusters beyond the recorded keys (noise)" |
| a partial proxy (`batch`) | 57 / 60 | 59 / 60 | names the proxy |
| the true driver | 60 / 60 | 60 / 60 | guard proposed; question marked *explained*, not open |

**F6, found while fixing F5.** The first version of `temporal_dispersion` used
`tarone_z` and false-alarmed on 27% of stationary p=0.1 streams while never
firing at p=0.9 — the same base-rate disease Stage 5 had just removed from the
changepoint path. Tracked to the statistic itself:

    shipped:   denom = sqrt( 2 Σ n_i(n_i-1) · p/(1-p) )
    corrected: denom = sqrt( 2 Σ n_i(n_i-1) )

The `p/(1-p)` factor does not belong to Tarone's Z. Its cost, measured on
covariate splits with **no real effect**, 6 keys tested per stream, BH at 0.05:

| true base rate | false-discovery rate, as shipped | corrected |
|---|---|---|
| 0.05 | **0.407** | 0.043 |
| 0.10 | 0.263 | 0.052 |
| 0.30 | 0.165 | 0.068 |
| 0.60 | 0.013 | 0.043 |
| 0.90 | 0.000 | 0.035 |

A scraper that succeeds 5% of the time — an ordinary broken tool — was being
handed a fabricated "works when X" condition **41% of the time**. Corrected, the
rate is flat at 3.5–6.8% against a nominal 5% at every base rate, and the
statistic is *more* powerful on real structure (0.995 vs 0.980 detection on the
oscillation world), so nothing was traded away.

Guard-path effect of the correction, 120 streams × 200 observations:

| | before | after |
|---|---|---|
| recall, strong effect | 120/120 | 120/120 |
| recall, moderate effect | 114/120 | 115/120 |
| false guards, null world at p=0.6 | 1/120 | 7/120 |
| false guards, null world at p=0.05 | ~41% (est. from the statistic) | ≤10% (tested) |

The p=0.6 rise from 0.8% to 5.8% is the test becoming *calibrated* — BH controls
the false-discovery rate at 5%, and the old 0.8% there was over-conservatism
bought with catastrophic anti-conservatism everywhere else.

**New claims tests for boundaries an agent actually meets** (6 added):
a tool that got *fixed* rather than broken (0.05 → 0.79 — the README's own
headline find is a repair, and an upward step at a low base rate was the
detector's worst regime); fabricated conditions at extreme base rates (0.06 and
0.94); and the other half of the honest-confusion claim — that a *stable* tool
at p ∈ {0.08, 0.5, 0.92} does not get pestered with questions, because a
question list that fires on everything is worthless.

**Claims suite: 34 passed, 0 failed.** Conformance + unit: **216 passed, 1
xfailed**. Structural audits (`make audit`) both clean.

---

## Stage 7 — docs match the code

The goal was that the codebase meets the claims it makes, which cuts both ways:
where the docs taught a path that does not work, the docs were wrong too.

- `docs/use-cases.md` recommended `redact(payload_hash)` to purge a bad source.
  It now teaches `retract_source` and explains why `redact` is the wrong tool
  for that job (and the right one for secrets and PII).
- `docs/api.md` gains `retract_source` and `redaction_scope`, with `redact`
  re-described as content-scoped rather than source-scoped.
- `docs/architecture.md` §curiosity now describes the exact changepoint test and
  the gate that evaluates its output, and records that instability is detected
  on the time axis without needing a covariate.
- `DEVIATIONS.md` D21-D24 record the four substantive departures from the spec
  as shipped, each citing its measurement.
- `README.md` points at `make claims` and states the count honestly.

Examples re-run: `quickstart`, `source_reliability`, and `regime_change` all
still produce their intended output, including the three-way
guard / regime-change / open-question routing.

---

## Final state

| suite | result |
|---|---|
| `tests/conformance.py` | 22 passed, 1 xfailed |
| `tests/unit/` | 194 passed (was 180; +14) |
| `tests/claims/` | **34 passed, 0 failed** (was 17 passed, 11 failed) |
| `make audit` | both structural invariants clean |

Six defects found, six fixed, none traded against another. Everything that held
up in the Stage 0 audit still holds: trust asymmetry, the Brier margin over
vote-averaging, calibration (ECE 0.0148, slope 0.998), guard recall, changepoint
localization, replay determinism, snapshot reproduction and order-invariance are
all still measured green by the same tests that measured them before.

### Where the substrate is now stronger than its README

- Regime detection on the flagship case went 0.52 → **1.00** while false
  positives on stationary streams went 0.20 → **0.00**.
- The covariate search is calibrated at every base rate, not just near p=0.5 —
  false discoveries on a mostly-failing tool dropped from 41% to ~5%.
- Instability with *nothing logged* went from silence to a question 60/60,
  which is the case the "log wide" advice was always aimed at.

### Known and accepted

- A gradual monotone drift is reported as a located step ~93% of the time. There
  is no single date to find, so this mischaracterises the *shape* — but the level
  really did change and the old regime really has stopped holding. Left as
  documented behaviour rather than papered over.
- Weak steps (0.6 → 0.4 over 120 observations) are detected ~6% of the time.
  With the changepoint searched rather than known, that shift genuinely is not
  significant; the previous 0.22 was noise, not power (median localization error
  5 observations, p90 18).
- Subtle short-period oscillation (0.80/0.40, period 10) is caught ~45% of the
  time. Large swings and outages are caught ~99-100%.
