# Benchmarks — the honest test, honestly told

The spec builds its own execution: §6.8 requires comparators and pass margins
**pre-registered before any result exists**, with the explicit possibility
that "the architecture is elegant and unnecessary." We ran three rounds. The
margins never moved. Two of the three headline bars ended green; one failed
three times and stands as the documented trade. Everything below is
reproducible from `bench/` (pre-registrations, findings, verdict scripts);
raw corpora stay local because they derive from private data, and internal
LAN hostnames in the pre-registration documents were genericized for
publication — each carries a note marking that as the sole post-freeze edit.

## Setup

- **Corpus:** 3,715 real prose entries from a working agent's memory.
- **Suite:** LLM-authored, *mechanically verified* — every retrieval item's
  gold document is whatever an exact corpus scan says contains the answer
  span; every false claim is a programmatic, type-preserving perturbation
  re-scanned to prove it matches nothing. Nothing a model asserts is trusted
  as a label. (241 items / 240 claims in v1; 1,194 / 1,194 in v2–v3.)
- **Baseline:** dense retrieval (bge-m3) over the same corpus; for
  calibration, RAG + elicited probabilities from a 27B model + isotonic — the
  comparator the spec names, built to win.
- **Control (mandatory since v0.3):** CANDOR's own pipeline with the trust
  machinery switched off — unweighted vote pooling + the same isotonic. If
  the mechanism can't beat its own knob-off case, that's the result.

## The three rounds

| Bar (frozen throughout) | v1 | v2 | v3 |
|---|---|---|---|
| Retrieval nDCG@10 ≥ baseline − 0.05 | FAIL (−0.12) | **PASS, wins** (+0.016) | **PASS, wins** |
| Retrieval recall@10 ≥ baseline − 0.05 | FAIL | **PASS, wins** (+0.022) | **PASS, wins** |
| Brier ≤ control + 0.005 | *lost to control* | **PASS** (CI [.025,.055]) | **PASS** (CI [.028,.056]) |
| CANDOR ECE ≤ 0.10 | PASS | PASS | PASS (0.028–0.044) |
| Brier ≥ baseline + 0.02 | FAIL | FAIL (−0.015) | FAIL (−0.018) |

Log loss and reliability slope favored CANDOR over the baseline in every
round from v2 on (v3: log loss 0.365 vs 0.445; slope 0.90 vs 0.60).

## What each failure taught (and changed)

**Round 1 exposed the composition.** A single reliability scalar per source
is symmetric, and real sources aren't: the learned likelihood ratios showed an
always-yes agent's "yes" carries LR 1.00 while a lexical checker's "no"
carries LR 0.08 — information one number cannot hold. One-coin reliability
*lost to ignoring reliability entirely*. → v0.3: integer confusion ledgers,
log-likelihood-ratio composition, sub-additive grouping for correlated
evidence (their error correlation was φ = 0.475).

**Round 2 exposed the test, then vindicated the mechanism.** Fixed panels
make vote patterns a sufficient statistic, so pooling was near-optimal *by
construction*; sparse random panels (the regime an agent memory actually
lives in) flipped the result: the mechanism now beats its control decisively.
The remaining Brier loss was concentrated in claims whose random panel drew
≤1 real judge — information starvation, not composition. → v0.4: graded
observations, witness floor.

**Round 3 resolved the residual as structural.** Grading pays (0.080 vs
0.108 for the same votes stripped to binary) but probability-elicitation
makes judges hedge, roughly cancelling. After three fixed-and-confirmed
causes, the unmoved gap has a clear meaning: **the baseline is the strongest
judge reading the best evidence fresh at query time — one maximal-quality
observation per question**. Stored sparse witnesses of mixed quality don't
outscore that on Brier, while costing ~nothing at read time and winning on
honesty metrics. We chose to document the trade rather than move the bar.

## The real-world round

Replaying 1,136 outcome events extracted from a live agent's operational
memory (337 targets, four months), the curiosity engine — with no ground
truth planted —

- located a tool repair: `tool_ok(yt-dlp)` 0% → 79% at **2026-04-30**,
  matching the agent's own note about fixing a stale-venv shebang;
- located a search-reliability collapse: 93% → 38% at **2026-04-22**,
  matching the documented pivot to fallback scrapers;
- **rejected** two plausible-looking rules (`method==search`,
  `method==cache`) that failed held-out validation — the fishing-expedition
  protection working on real data;
- opened one honest question (`fetch_ok(cnbc.com)`, marginal signal) with a
  concrete suggested measurement.

## Method notes worth stealing

- Margins are frozen in a pre-registration whose SHA-256 is recorded by the
  run; the verdict script refuses to rule if the file changed.
- Every mid-run substitution (models swapped for throughput) was amended in
  writing *before results existed*, with the substitution's effect measured
  (judge agreement 97.5%) rather than asserted.
- All post-mortems were computed offline from the substrate's own ledger —
  the audit machinery analyzing its own experiment.
