# 6.8 post-mortem — offline reanalysis findings

Produced by `bench/backtest_composition.py`, which recomputes everything from
the suite's ground truth plus the calibration store's ledger. No model calls.
**All of this is exploratory reanalysis of the run that exposed the problem;
nothing here is "confirmed" until it survives a fresh suite under a new
pre-registration.**

## F1 — the test design muted the mechanism under test

Every claim was voted on by the **same fixed panel of four observers, once
each**. Under a fixed panel, a claim's evidence reduces to a 4-bit vote
pattern, and the observed patterns were almost perfectly separated by the two
LLM votes alone:

| exact | llm_big | llm_small | optimist | n | truth rate |
|---|---|---|---|---|---|
| T | T | T | T | 110 | 0.99 |
| T | F | F | T | 70 | 0.07 |
| F | F | F | T | 44 | 0.05 |
| T | F | T | T | 9 | 0.22 |
| F | F | T | T | 4 | 0.25 |
| T | T | F | T | 3 | 0.33 |

Mean-vote buckets align monotonically with truth rates, so **isotonic on the
unweighted mean is already close to the optimal estimator for this design**.
Actor attribution has almost no headroom: there is barely any within-count
information for a reliability model to exploit. The counts machinery (evidence
accumulating per (fact, actor) over time) was never exercised — every count was
n=1. 6.8-as-built is a crowd-labeling benchmark, not a memory benchmark.

The regime where attribution *mathematically must* matter is **sparse,
heterogeneous observation**: many actors, each fact seen by a small varying
subset. There, mean-vote isn't even comparable across facts, and per-actor
calibration learned on settled facts transfers to unsettled ones. That is also
the regime an agent memory actually lives in. Suite v1 never entered it.

## F2 — one-coin reliability is the wrong shape; two-coin learns the right
parameters but the product composition overcounts correlated votes

Held-out, train-fitted isotonic applied identically to every rule:

| rule | Brier | log loss | acc | ECE | slope |
|---|---|---|---|---|---|
| one-coin (as run) | 0.0514 | 0.2489 | 0.921 | 0.0517 | 0.824 |
| naive mean (control) | **0.0423** | 0.2381 | 0.961 | 0.0547 | 1.012 |
| two-coin LR | 0.0486 | 0.2396 | 0.921 | **0.0341** | 0.575 |

(One-coin reproduces at 0.0514 vs the run's 0.0564 — closed form vs sampled
composition and batch- vs sequential-scored reliability; same story.)

The two-coin confusion model learns **exactly the asymmetries hypothesized**:

| actor | one-coin rel | sens | fpr | LR(T vote) | LR(F vote) |
|---|---|---|---|---|---|
| tool:exact | 0.737 | 0.967 | 0.607 | 1.59 | **0.08** |
| agent:llm_big | 0.940 | 0.902 | 0.054 | 16.83 | 0.10 |
| agent:llm_small | 0.925 | 0.902 | 0.089 | 10.10 | 0.11 |
| agent:optimist | 0.586 | 0.984 | 0.982 | **1.00** | 0.92 |

The optimist's TRUE vote carries an LR of exactly 1.00 — correctly identified
as pure noise — and tool:exact's FALSE vote is decisive (0.08), both of which
the one-coin scalar cannot represent. Two-coin beats one-coin on held-out Brier
(+0.0028, 95% CI [−0.0009, +0.0065]) and has the best ECE. **But it does not
beat the naive control**, because naive-Bayes composition assumes independent
observers and the two LLMs' errors are correlated (φ = 0.475 — they share the
same retrieved context and a model family). The product of their LRs
double-counts shared evidence → slope 0.575, overconfident before isotonic.

The spec anticipates this and dodges it: §3.9 makes `shared_provenance` a
caveat that is "flagged, not modeled." This run is what that dodge costs.

## F3 — nothing at n=127 is actually settled

Per-item Brier-difference SD is 0.0876. Resolving a 0.01 delta at 95%/80%
power needs ~600 held-out claims; suite v1 had 127. Even "the reliability
machinery hurt" is directional (its CI vs control was [−0.031, +0.002]), not
established. Any retest needs ≥600 held-out claims (≥1200 total).

## F4 — retrieval loses on matching, not ranking

Gold absent from CANDOR's top-10 on 37/241 items vs 14/241 for bge-m3; another
45 ranked below position 2. Paraphrased questions vs lexical matching, as
expected. Candidate fixes in ascending spec impact: index tags/namespaces;
LLM query expansion at the untrusted boundary (ranking input only — I2
untouched); RM3 pseudo-relevance feedback; or a v0.3 amendment admitting a
dense ranker as an untrusted periphery ranking input (embeddings never in the
core, never in the committed tier, never moving a count).

## F5 — what survived

The settle → score → recalibrate pipeline behaved: reliability ordering was
learned correctly from 113 settlements, CANDOR beat the elicited baseline on
log loss / slope / ECE, and this entire reanalysis was possible offline because
every vote is an attributed ledger event. Attribution has already paid for
audit and retroactive exclusion; what is unproven is that it pays for
*accuracy* — and suite v1 could not have shown it either way.

---

# 6.8-v2 addendum (2026-07-26)

Verdict under the frozen v2 bars: **retrieval PASSES (and wins outright);
mechanism beats its knob-off control decisively; the Brier-vs-baseline margin
FAILS.** Full numbers in `data/bench/verdict_v2.txt`.

## F6 — the v1 headline finding is reversed

In the sparse regime, two-coin + context-grouped composition beats uniform
pooling by 0.040 Brier (0.0763 vs 0.1159, CI [0.0248, 0.0549]). Attribution
now demonstrably pays for accuracy exactly where Δ3 scoped the claim. CANDOR
also beats the elicited baseline on log loss (0.351 vs 0.445), slope (0.71 vs
0.60) and ECE (0.028 vs 0.039-class).

## F7 — the remaining loss is information starvation, not composition

Held-out Brier by number of semantic (LLM) judges the random panel assigned:

| LLM judges | n | candor | baseline |
|---|---|---|---|
| 0 | 19 | 0.301 | 0.132 |
| 1 | 161 | 0.094 | 0.062 |
| 2 | 253 | **0.035** | 0.044 |
| 3 | 153 | 0.105 | 0.078 |
| 4 | 20 | **0.046** | 0.064 |

CANDOR **wins the 2- and 4-judge buckets**. The overall loss is carried by the
180 claims whose panels held ≤1 semantic witness — where the substrate is
being asked to answer from votes nobody meaningfully cast, while the baseline
always performs a fresh, full top-8 read with the strongest model at query
time. (The 3-judge bucket contains intrinsically harder claims — the
panel-independent baseline degrades there too.) γ is not the cause: sweeping
γ ∈ {0, 0.25, 0.5, 0.75} moves overall Brier by <0.002, and train log-loss
selects the frozen default 0.5.

## F8 — what the Brier bar now measures

The comparison has become *stored sparse binary witnesses* vs *unbounded fresh
evidence access at read time*. No composition rule can close that gap on
claims with no informative witnesses; conversely CANDOR's read-time cost is
near zero while the baseline spends a full strong-LLM call per query. The
remaining bar is structural, not a tuning miss — the round-3 decision is which
framing to fix (witness floor via §3.11-style scheduling, graded-confidence
observations as a v0.4 mechanism, or a cost-matched comparator), and that is
the principal's call.

---

# 6.8-v3 addendum (2026-07-26)

Verdict: **FAILS the same single bar** (Brier vs fresh-reader baseline,
−0.0184); retrieval passes again; control bar passes bigger (CI [0.028,
0.056]); slope 0.90 and log loss 0.365 are the best of any system in any
round. Full ruling in `data/bench/verdict_v3.txt`.

## F9 — grading works; probability-elicitation hedging cancels it

On identical v3 panels, the graded composition scores 0.080 held-out Brier vs
0.108 for the same votes stripped to binary — Δ6 pays. But the elicit prompt
makes judges hedge: 468/2,388 semantic votes are "weak" (p ∈ [0.25, 0.75)),
where v2's boolean prompt forced decisions. Net effect ≈ zero against v2.

## F10 — the residual is structural, three rounds replicated

The baseline is the strongest judge reading the best evidence fresh at query
time, once, plus isotonic — i.e., exactly one maximal-quality graded
observation. CANDOR composes 2 stored semantic witnesses of mixed strength
plus 2 noise actors, and must beat that reader by 0.02 Brier. Across three
rounds every named defect was fixed and confirmed fixed (composition,
starvation, vote granularity); the bar has not moved and neither has the gap.
Meanwhile CANDOR beats its own knob-off control decisively and leads on log
loss, slope, and ECE in every round since v2. Conclusion: the remaining bar
measures query-time reading, which CANDOR is not; the decision on reframing
(cost-matched comparator) belongs to the principal, per the standing rule that
margins never move to fit results.

---

# Real-world round: Pernix history through the curiosity engine (2026-07-27)

1,136 outcome events (fetch/tool/search, 337 targets, 2026-03→07) extracted
from the live box's memories and replayed in historical order
(`bench/run_realworld.py`, report in `data/bench/realworld_report.json`).

## F11 — the engine found two real, corroborated regime changes

* **tool_ok(yt-dlp): 0% → 79% success, located 2026-04-30.** The memories
  independently document the cause: yt-dlp broken by a stale-venv shebang
  ("FileNotFoundError ... cai_v2 no longer existed") and then the recorded
  fix. The engine recovered the repair date from outcome data alone.
* **search_ok(*): 93% → 38%, located 2026-04-22.** Matches the corpus's
  documented collapse of `search_web` reliability and the late-April pivot to
  crawl4ai/direct-browse fallbacks.

## F12 — the gate's held-out check refused two plausible-but-wrong guards

`method==search` (117 hits / 142 misses held-out) and `method==cache`
(129/148) were proposed from discovery data and REJECTED at step 5 — the
§4.5 fishing-expedition protection doing its job on real data. fetch_ok(cnbc.com)
was dispersion-flagged with an open question (weak signal, honestly marginal),
consistent with the corpus's recurring CNBC bot-blocking theme.

The learning loop the architecture promises — observe, detect, propose, gate,
and say out loud what changed and when — ran end to end on lived history and
produced findings nobody planted.
