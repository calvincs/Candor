# CANDOR — spec v0.5 delta

Amends v0.4. Motivated by two gaps the earlier stmt_types could not express:
a flaky binary fact was reported only as "unstable" with no picture of *how* it
splits, and many real outcomes are not true/false at all but *which of an
open-ended set of values happened* — where the honest answer includes "a value
we have not seen yet." Both are additive: no committed number for a crisp or
frequency fact changes, and the only schema tightening that widens is the
`facts.stmt_type` CHECK.

## Δ8 — open-vocabulary categorical facts

A third `stmt_type`, **`categorical`**. An observation records a *value*, not a
boolean: `observe(stmt, ctx=..., actor=..., value="captcha")`. The vocabulary is
**open** — it grows as new values are observed — and the never-seen mass is a
first-class outcome, not an error bar.

**Data model (I11 — integer counts only).** One new count table, moved by the
same fold path as every other:

    fact_category_counts(fact_id, actor, value, n)      PRIMARY KEY(fact_id, actor, value)

`observations.value` becomes a nullable column carrying the realised category
for audit (the boolean `outcome` stays for crisp/frequency; which field is
authoritative is decided at fold time by the fact's `stmt_type`, so an
observation arriving before its fact is admitted still routes correctly on
replay). Both are folded into `COUNT_COLUMNS` (I11) and `_HASH_QUERIES` (so
replay and checkpoint cover per-value counts).

**Predictive (read-time, §2.4).** `predict()` on a categorical fact returns a
`CategoricalPrediction` — a distribution, not a scalar — using the
Dirichlet-process / CRP predictive with the Pitman–Yor discount pinned to
`d = 0`:

    P(v)       = n_v / (N + alpha)      for each seen value v
    P(unknown) = alpha / (N + alpha)    the never-seen mass, carried as 1 − ΣP(v)

so `Σ values[*].p + unknown.p == 1.0` exactly. `alpha` is a single
pre-registered global concentration (`CATEGORICAL_ALPHA = 1.0`), versioned into
the predictor class `categorical/v1` and ridden by the snapshot id — so
`predict_at` reproduces the full distribution bit-for-bit (I8), and the
categorical path calibrates under its own predictor class, never pooled with the
scalar path (I9). Per-value credible intervals are the Beta **marginals** of the
Dirichlet, reusing `betamath` verbatim; the unknown slice carries its own
interval. Under `d = 0`, `P(unknown)` is a function of `N` alone (not of the
distinct-value count) — that separation is the deferred `d > 0` upgrade.

**Settlement & trust (§3.8/§3.12).** A categorical `claim` freezes the C2
distribution into `claims.predicted_dist_json` (I8). `resolve(value=v*)` scores
`surprisal = −log P_frozen(v*)`, which stays **finite even for a `v*` never
seen** — it scores against the frozen unknown mass, which is the headline
payoff. Per-source trust moves by a **one-vs-rest reduction**: each (actor,
value) is projected to a binary "was it `v`?" question keyed by a virtual actor
id `catv1:<canon_json([actor, value])>`, reusing the Δ1 two-coin confusion
machinery already in `_HASH_QUERIES`. The confusion (not response) path is used,
so a single settlement scores the reported value without punishing an honest
reporter for the values it did not name.

**Curiosity (§3.10).** The sweep runs per value, one-vs-rest: each value is
projected to a binary `[value == v]` stream and put through the existing
Tarone / BH / MDL / held-out / guard pipeline (BH across K values × context
keys). An admitted guard conditions the fact on a key, and `predict()` then
fills `by_context`: `{key: {context_value: {values, unknown, n}}}`, each group
its own CRP with its own unknown slice, plus a `__residual__` group for
observations that did not record the key — the honest "cannot attribute" mass
kept distinct from the per-context unknown.

**Scope (v1).** Leaf-query only; one global versioned `alpha`. See *Deferred*.

## Δ9 — read-time distribution surfacing for binary flaky facts

`distribution(stmt)` is a **pure read-time projection** over data the store
already keeps — the fact's observations, their recorded `obs_context`, the
sweep's stored verdict, and any admitted guard. It writes nothing, moves no
count, changes no `closure_hash`, and never runs `predict()`; the scalar
prediction is byte-identical whether or not it is ever called. It returns the
per-context outcome breakdown (`modes`, with a `__residual__` bucket for
observations that did not record the key) and the unexplained share (`residual`:
`explained` = the η² correlation ratio of an admitted guard's partition,
`unexplained` = the honest remainder, plus the raw stored dispersion statistic).
It is the companion to the `unstable` caveat — not only *that* a fact is flaky,
but *how* it splits and how much no recorded variable accounts for. The `modes`
helper is shared with Δ8's `by_context`.

## Deferred (design-noted, not defects)

- **Pitman–Yor `d > 0`** — make novelty depend on the distinct-value count, not
  `N` alone.
- **Dawid–Skene confusion matrix** — a full per-source value→value confusion,
  in place of the v1 one-vs-rest reduction.
- **Joint-multinomial guard test** — a single multinomial G-test per context
  key, instead of K independent one-vs-rest sweeps.
- **Per-value changepoint** — locate a regime change in a single value's rate.
- **Non-leaf categorical queries** — categorical values inside derivation.
