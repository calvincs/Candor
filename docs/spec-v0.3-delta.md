# CANDOR — spec v0.3 delta

**Status: DRAFT for review.** Amends `candor-spec-v0.2.md` (kept verbatim as
`SPEC.md`); every clause not named here carries forward unchanged. Motivated
entirely by the pre-registered 6.8 failure and its ledger-derived post-mortem
(`bench/FINDINGS_6_8.md`). Each delta names the evidence that forced it.

The methodology this delta encodes: mechanisms and tests iterate; **margins do
not move until they are passed**. Every retest runs under a fresh
pre-registration frozen before results exist, and the uniform-pooling control
rides along forever (Δ3), so the machinery has to keep earning its existence.

---

## Δ1 — Two-coin actor model (supersedes the §3.12 single Beta)

**Evidence.** One-coin reliability is a symmetric accuracy model and observers
are asymmetric. In the 6.8 run, the learned confusion parameters show what one
scalar cannot represent: `agent:optimist`'s TRUE vote carried a likelihood
ratio of **1.00** (pure noise) while `tool:exact`'s FALSE vote carried **0.08**
(near-decisive). One-coin discounted both directions of both actors equally,
and lost to ignoring reliability entirely (FINDINGS F2).

**Change.** `actor_reliability(rel_a, rel_b)` is superseded as the *composition*
input by an integer confusion table:

```sql
actor_confusion(
  actor TEXT NOT NULL,
  frame TEXT NOT NULL CHECK(frame IN ('internal','external')),
  tp INTEGER NOT NULL, fn INTEGER NOT NULL,   -- settled true:  voted T / F
  fp INTEGER NOT NULL, tn INTEGER NOT NULL,   -- settled false: voted T / F
  PRIMARY KEY(actor, frame));
```

Four integers per (actor, frame): I11 exactly as before, the audit trail is
raw tallies, every real number is read-time. The cells move **only** through
the §3.12 trusted path — a claim settling via a `deterministic_total` oracle
scores every prior observation's *vote direction* against the settled outcome.
The old agree/disagree Beta is derivable (`agree = tp + tn`) and the
`actor_reliability` table is retained for the alea discount (below) and the
frozen v0.2 conformance hooks.

**Composition (crisp facts, epi channel).** Read-time, log-likelihood ratio:

```
sens_a ~ Beta(9.5 + tp, 0.5 + fn)        # priors are read-time constants,
fpr_a  ~ Beta(0.5 + fp, 9.5 + tn)        # E[sens]=0.95, E[fpr]=0.05, mass 10
logodds(fact) = prior + Σ_votes  logLR_a(vote)
  logLR_a(T) = log(sens_a / fpr_a);  logLR_a(F) = log((1−sens_a)/(1−fpr_a))
```

The prior encodes the same stance as v0.2's Beta(19,1): newcomers are assumed
informative; the discount is a penalty for *demonstrated* miscalibration, and
with zero settlements the composition degrades to trusting votes at LR ≈ 19,
never to ignoring them (I5-adjacent: evidence must be able to speak before it
is scored).

**Scope.** Two-coin applies to the **epi channel of crisp facts** — votes about
truth. The alea channel keeps the v0.2 `E[rel]` discount: a frequency trial is
a reported world-outcome, not a judgement, and modelling fabricated trials is
out of scope for v1 (unchanged limitation, now stated).

**Two-loop integration (I7 improvement).** Actor parameters `(sens_a, fpr_a)`
are epistemic quantities shared across every fact the actor observed, so the
prediction engine samples them **once per epistemic world** — epistemic
uncertainty now correlates across claims sharing an observer, by construction,
exactly as I7 already demands for shared facts. Sampling stays stratified and
deterministic (I8: snapshots still reproduce bit-for-bit; the confusion table
is ledger-derived, so `predict_at` rebuilds it at any historical head).

## Δ2 — `shared_provenance` becomes a mechanism, not a caveat (amends §3.9/§4.7)

**Evidence.** The two LLM observers shared retrieved context; their errors
correlated at φ = 0.475; independence-flavoured composition double-counted the
shared evidence (post-isotonic slope 0.575). v0.2 §3.9 says shared provenance
is "flagged, not modeled" — the 6.8 run is the invoice for that dodge
(FINDINGS F2).

**Change.** Observations grouped by `context_sig` (already stored per event,
§4.6) compose **sub-additively**:

```
logodds(fact) = prior + Σ_groups  ( Σ_{i∈g} logLR_i ) / m_g^γ
```

where `m_g` is the group's size and `γ ∈ [0,1]` (0 = independent, 1 = fully
redundant; default **0.5**). γ is a read-time constant in v0.3; if fitted, it
is fitted on settled (train) claims only and becomes part of the calibration
artifact whose hash rides in every snapshot (I8). Observations with no
recorded context form singleton groups. The §4.7 caveat set still carries
`shared_provenance` for the consumer's benefit; it is now also priced.

## Δ3 — §3.12's claim is re-scoped, and the control becomes permanent

**Evidence.** In a fixed-panel, dense-observation regime, isotonic over the
unweighted vote mean is near-optimal by construction, and no reliability
model — one-coin or two-coin — beat it (FINDINGS F1). What attribution
*demonstrably* delivered was audit: correct reliability ordering from 113
settlements, retroactive exclusion as pure recompute, and a full offline
post-mortem from the ledger alone.

**Change to the claim.** Actor attribution's primary, unconditional value is
**damage-bounding and auditable exclusion**. Its *accuracy* value is claimed
only for the **sparse-observation regime** — many actors, each fact observed by
a small varying subset, where per-actor calibration learned on settled facts
transfers to unsettled ones and pattern-blind pooling has no comparable
statistic. This is the regime an agent memory actually inhabits, and it is the
regime honest tests must construct (suite v2, T1).

**Change to the harness contract.** Every honest-test run (§6.8 and successors)
MUST report the **uniform-reliability control** — identical pipeline, discount
knob off — alongside the system. If the mechanism cannot beat its own
degenerate case, that result is reported, not averaged away. (The v1 harness
did this voluntarily; it is now a requirement.)

## Δ4 — crisp epistemic prior (codifies DEVIATIONS D16)

§4.2's channel semantics imply what v0.2 never stated: admission is a
structural act and observes nothing about the world, so an admitted **crisp**
fact's validity prior is uniform — `Beta(1, 1)` — while an admitted
**frequency** fact's class-validity prior remains strong (`Beta(99, 1)`),
because passing the gate genuinely is evidence the reference class is
well-formed and trials can never move that channel. Was a deviation; now spec.

## Δ5 — retrieval (DECIDED: R2, principal's selection, 2026-07-26)

6.8 v1 retrieval lost on **matching**, not ranking (gold absent from top-10:
37 vs 14 of 241). Two candidate postures:

* **R1 (no identity change):** stay lexical; add sub-token indexing (dotted
  namespaces, hyphenated identifiers) and RM3 pseudo-relevance feedback —
  stdlib, deterministic, no model calls. Retest against the same band and
  accept the result.
* **R2 (identity amendment):** admit a dense ranker as an **untrusted
  periphery ranking input** — embeddings never enter the trusted core or the
  committed tier, never touch a count, and retrieval still cannot move a
  number (I2 intact; the §0 non-goal would be narrowed from "anywhere in the
  core" honestly rather than reinterpreted silently).

**Decision: R2, with R1 kept underneath.** The §0 non-goal is narrowed from
"no text embeddings anywhere in the core" to "no embeddings in the trusted
core or the committed tier". Dense ranking enters as an injected callable —
`retrieval.py` keeps its empty import list, the I2 audit keeps meaning
something, absence or failure of the embedder degrades to lexical, and
retrieval still cannot move a number. Fusion is reciprocal-rank (k=60) over
the lexical+RM3 and dense orderings; dense vectors are a droppable disk cache
keyed by content hash.

## Conformance impact

The frozen v0.2 harness (`tests/conformance.py`) passes unchanged against a
Δ1–Δ4 build: two-coin composition touches only the crisp-epi read path, which
no numeric assertion in the v0.2 suite exercises; the alea discount, channel
routing, composition-purity and actor-discount hooks keep v0.2 semantics.
`actor_confusion` joins the integrality scan (I11) and the replay-determinism
hash. New behaviour is covered by additive tests under `tests/unit/`.
