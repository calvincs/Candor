# Use cases

Five patterns this substrate is actually for, each mapped to a runnable
example.

## 1. Agent memory that can be audited (and survive a bad source)

**The problem.** Your agent stores "facts" from tools, web pages, and its own
inferences. Six weeks later it confidently repeats something a hallucinating
scraper told it in March, and you can't find where the belief came from or
what else it contaminated.

**The CANDOR shape.** Every ingest is an attributed candidate through the
gate; every tool report is an attributed observation. When you discover the
March scraper was bad, you have two scalpels, both reversible:

```python
m.set_reliability("tool:bad-scraper", "external", 0.001, 100)  # discount it...
m.retract_source("tool:bad-scraper", reason="hallucinated in March")  # ...or silence it
h = m.replay()   # every downstream number recomputes as if it never spoke
```

Nothing else in the store is touched, because no number was ever stored — only
integer counts keyed by who reported them. The retracted source's events stay in
the chain forever; they simply stop contributing, and `restore=True` puts them
back. See `examples/quickstart.py`.

**Reach for `retract_source`, not `redact`, when the problem is a source.**
`redact` purges a *payload*, and payloads are content-addressed with no actor in
them — so two sources reporting the same outcome on the same statement share
one. Redacting a liar's hashes also destroys the honest reports that agreed with
it. `redaction_scope(hash)` tells you the blast radius before you fire; `redact`
is the right tool only when the *content* itself must go (secrets, PII).

## 2. Source-reliability tracking: trust that is earned, asymmetric, and cheap

**The problem.** You aggregate judgements from multiple LLMs, tools, and
heuristics of very different quality — including some that are biased rather
than merely noisy. Averaging treats a sycophant like a scientist.

**The CANDOR shape.** Let every judge vote (optionally with confidence), then
settle a training slice of claims against ground truth. The confusion ledgers
learn each source's *shape*, not just its accuracy: the always-yes agent's
"yes" ends up carrying a likelihood ratio of ~1.0 (worthless) while its rare
"no" becomes decisive; correlated judges sharing evidence get priced
sub-additively instead of double-counted. Measured on our benchmark, this
composition beat plain vote-averaging by 0.04 Brier with a confidence
interval nowhere near zero. See `examples/source_reliability.py`.

**Fits:** LLM-as-judge ensembles, moderation pipelines, multi-tool RAG
verification, human+model hybrid review queues.

## 3. Drift detection with dates: "what changed, and when?"

**The problem.** Your pipelines degrade silently. A site adds bot protection;
a dependency fixes itself after an upgrade; a model's behaviour shifts. Decay-
weighted averages just get vaguely worse; they never *say* anything.

**The CANDOR shape.** Feed outcome events with wide context. The curiosity
sweep separates three situations that look identical to an average:

- a **condition** — success depends on a recorded covariate → proposes a
  guard ("works when method=crawl4ai"), which must survive BH correction, an
  MDL check, and held-out validation;
- a **regime change** — a one-way step → proposes a valid-time supersede with
  the change *located to a date*; old counts stay with the old regime;
- **unexplained variance** — opens a question carrying the residual partition
  and a concrete suggested measurement, re-tested automatically when a new
  covariate starts being recorded.

Two v0.6 upgrades sharpen this: the sweep also searches **frames you never
logged** — hour-of-day and day-of-week from the event timestamps, the fact's
own previous outcome, interactions of recorded keys — so "backups fail at
03:00" is findable with an *empty* context dict; and admitted conditions
**keep paying rent** — one whose direction reverses or goes stale on later
observations is demoted through the same gate, and can only re-enter on fresh
post-demotion evidence. See `examples/axiom_loops.py` for both.

On its first run over a real agent's operational history this located a tool
repair (0%→79% on 2026-04-30) and a search-reliability collapse (93%→38% on
2026-04-22), both corroborated by the agent's own notes — and rejected two
plausible-looking rules that failed held-out validation. See
`examples/regime_change.py`.

**Fits:** scraper/API health, CI flakiness forensics, feed-quality
monitoring, any place "it used to work" is a bug report.

## 4. Open-vocabulary outcomes: "which one happened — could it be one we've never seen?"

**The problem.** The outcome isn't true/false and isn't a fixed enum: a login
flow resolves to a captcha, a block page, an MFA challenge — or something new
next week. Bucketing into success/failure throws away *which* failure, and a
fixed enum can't represent a value you haven't met yet, so a brand-new failure
mode reads as probability zero instead of "unknown."

**The CANDOR shape.** Declare the fact `categorical` and observe values; the
vocabulary grows as you go, and `predict()` returns a distribution with a
first-class unknown mass:

```python
m.assert_({"pred": "resolves", "args": ["login"], "stmt_type": "categorical"},
          source="runbook", actor="human:me")
m.run_gate()
for v in ["captcha"] * 8 + ["block"] * 2:
    m.observe({"pred": "resolves", "args": ["login"]}, ctx={"region": "eu"},
              actor="tool:probe", value=v)
p = m.predict({"pred": "resolves", "args": ["login"]}, budget=1000)
# p.values["captcha"].p == 8/11 ≈ 0.73, p.values["block"].p == 2/11 ≈ 0.18,
# p.unknown.p == 1/11 ≈ 0.09  — a never-seen value keeps a real probability.
```

A value you have never observed keeps a real probability (`p.unknown`);
settlement scores surprisal against it (finite even for a first-ever value); and
the curiosity sweep conditions the distribution on context (`p.by_context`) just
as it guards binary facts. See `examples/categorical.py`.

**Fits:** failure-mode classification, error-class and routing distributions,
intent/label tracking where the label set is not closed.

## 5. Goodhart watch: "did this stop being true because we started optimizing it?"

**The problem.** A metric tracks a goal — until someone targets the metric,
and the coupling silently collapses. Averaged monitoring sees "the metric got
noisy"; what actually happened is that *acting* on the system changed what the
old observations were evidence for.

**The CANDOR shape.** Log interventions as `do:` context keys — a reserved
prefix meaning "we were acting, not watching":

```python
m.observe({"pred": "metric_tracks_goal", "args": ["ctr"]}, ok,
          ctx={"do:optimize_metric": "yes"}, actor="tool:monitor")
```

Three things follow. A guard discovered on a `do:` key is labeled
**regime-dependent** — "the coupling holds only where the metric is *not* the
target" is Goodhart's law found by name, not a generic condition. Any
prediction that pools observations across the intervention boundary carries a
`regime_mixed` caveat, so the marginal announces that it averages two
different worlds (the per-regime numbers live in `distribution()`). And if
nobody logged the intervention at all, the collapse still surfaces as a
regime change located to a date. See `examples/axiom_loops.py` and the
executable battery in `tests/claims/test_axiom_battery.py`.

**The stated limit:** this is detection, not prophecy. Nothing here predicts
what an intervention *will* change before post-intervention data exists —
that needs a causal model, and the battery asserts that boundary so it stays
a measured fact about the system.

**Fits:** KPI/OKR instrumentation, reward-hacking watch for RL or agent
loops, A/B systems where shipping the winner changes the population.

## Anti-use-cases (read before adopting)

- **Not a vector database.** The optional dense ranker helps `recall`, but if
  semantic search is the whole job, use a vector store.
- **Not a query-time oracle.** Our own benchmark showed that if you can afford
  a top-tier model reading full evidence fresh on every question, it will beat
  stored sparse witnesses on raw Brier. CANDOR's trade is provenance,
  per-source learning, and ~free reads — not beating a fresh reader.
- **Not distributed.** Single writer, single sequencer, one box. By design,
  for v1.
- **No native continuous channel.** Outcomes are binary (crisp/frequency) or
  open-vocabulary categorical (use case 4); *continuous* scalars (latencies,
  scores) still need binarizing or bucketing at the boundary.
- **Not a causal-inference engine.** `do:` keys give interventions a
  vocabulary (use case 5); they do not build a causal model. The system
  labels regime dependence after the data shows it — it never anticipates an
  intervention's effect in advance.
